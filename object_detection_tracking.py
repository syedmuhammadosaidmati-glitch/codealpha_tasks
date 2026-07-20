"""
================================================================================
 REAL-TIME OBJECT DETECTION AND TRACKING
 (YOLOv8 detection + built-in SORT tracker, in a single file)
================================================================================

WHAT THIS PROJECT DOES
-----------------------
1. Captures real-time video from a webcam OR reads a video file (OpenCV).
2. Runs a pre-trained YOLOv8 model (Ultralytics) on every frame to detect objects.
3. Draws bounding boxes + class labels for every detection.
4. Feeds detections into a SORT tracker (Kalman Filter + Hungarian Algorithm)
   implemented from scratch in this file -- no extra tracking library needed.
5. Displays the live video with bounding boxes, class labels, and a persistent
   tracking ID for every object, and (optionally) saves the annotated video.

--------------------------------------------------------------------------------
INSTALLATION (run once)
--------------------------------------------------------------------------------
    pip install ultralytics opencv-python numpy scipy

The very first run will auto-download the YOLOv8 weights file (yolov8n.pt,
~6 MB) from Ultralytics, so an internet connection is needed the first time.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
Webcam (default camera 0):
    python object_detection_tracking.py --source 0

A video file:
    python object_detection_tracking.py --source path/to/video.mp4

Save the annotated output to a file:
    python object_detection_tracking.py --source 0 --output result.mp4

Only track certain classes (COCO class names, e.g. person, car, dog):
    python object_detection_tracking.py --source 0 --classes person car

Use a bigger/more accurate YOLO model:
    python object_detection_tracking.py --source 0 --model yolov8s.pt

Press 'q' at any time in the video window to quit.

--------------------------------------------------------------------------------
PROJECT STRUCTURE (all inside this one file)
--------------------------------------------------------------------------------
    1. KalmanBoxTracker   -> tracks a single object's bounding box over time
    2. iou_batch / associate_detections_to_trackers -> data association
    3. Sort               -> the multi-object tracker (manages all tracks)
    4. ObjectDetector      -> wraps YOLOv8 for detection
    5. run()               -> main loop: capture -> detect -> track -> draw -> show
    6. __main__             -> CLI argument parsing
================================================================================
"""

import argparse
import time
from collections import OrderedDict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# 1. KALMAN FILTER BASED SINGLE OBJECT TRACKER
# ==============================================================================
class KalmanBoxTracker:
    """
    Tracks a single object's bounding box using a constant-velocity Kalman
    Filter. State vector: [x, y, s, r, vx, vy, vs]
        x, y  -> center of the bounding box
        s     -> scale/area of the box
        r     -> aspect ratio (kept constant)
        vx,vy,vs -> velocities of x, y, s
    """

    count = 0  # global counter used to assign unique track IDs

    def __init__(self, bbox, class_id):
        # state transition matrix (constant velocity model)
        self.ndim, dt = 4, 1.0
        self._motion_mat = np.eye(7, 7)
        for i in range(3):
            self._motion_mat[i, i + 4] = dt

        self._update_mat = np.eye(4, 7)

        # initialise Kalman filter matrices
        self.F = self._motion_mat
        self.H = self._update_mat
        self.P = np.eye(7) * 10.0          # covariance
        self.P[4:, 4:] *= 1000.0            # high uncertainty for velocities
        self.Q = np.eye(7) * 0.01           # process noise
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * 1.0            # measurement noise

        self.x = np.zeros((7, 1))
        self.x[:4, 0] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id

    @staticmethod
    def _bbox_to_z(bbox):
        """Convert [x1,y1,x2,y2] -> [cx,cy,scale,ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([cx, cy, s, r])

    @staticmethod
    def _z_to_bbox(z):
        """Convert [cx,cy,scale,ratio] -> [x1,y1,x2,y2]"""
        w = np.sqrt(max(z[2] * z[3], 0))
        h = z[2] / (w + 1e-6)
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = self._z_to_bbox(self.x[:4, 0])
        return bbox

    def update(self, bbox, class_id):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_id = class_id

        z = self._bbox_to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        return self._z_to_bbox(self.x[:4, 0])


# ==============================================================================
# 2. DATA ASSOCIATION HELPERS (IoU + Hungarian Algorithm)
# ==============================================================================
def iou_batch(boxes_a, boxes_b):
    """Vectorised IoU between two sets of boxes [x1,y1,x2,y2]."""
    boxes_a = np.expand_dims(boxes_a, 1)
    boxes_b = np.expand_dims(boxes_b, 0)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - inter

    return inter / np.maximum(union, 1e-6)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to existing trackers using IoU + the Hungarian algorithm.
    Returns: matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    iou_matrix = iou_batch(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximise IoU

    matches, unmatched_detections, unmatched_trackers = [], [], []

    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_detections.append(d)
    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trackers.append(t)

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            unmatched_detections.append(r)
            unmatched_trackers.append(c)
        else:
            matches.append([r, c])

    matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ==============================================================================
# 3. SORT -- SIMPLE ONLINE AND REALTIME TRACKING (multi-object manager)
# ==============================================================================
class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        """
        max_age      -> frames to keep a track alive without a matching detection
        min_hits     -> minimum detections before a track is shown (reduces noise)
        iou_threshold-> IoU required to associate a detection with a track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        detections: np.array of shape (N, 5) -> [x1, y1, x2, y2, class_id]
        Returns: np.array of shape (M, 6) -> [x1, y1, x2, y2, track_id, class_id]
        """
        predicted_boxes = []
        to_delete = []

        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.any(np.isnan(box)):
                to_delete.append(i)
            predicted_boxes.append(box)

        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))
        for i in reversed(to_delete):
            self.trackers.pop(i)
            predicted_boxes = np.delete(predicted_boxes, i, axis=0)

        det_boxes = detections[:, :4] if len(detections) else np.empty((0, 4))
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, predicted_boxes, self.iou_threshold
        )

        # update matched trackers with the assigned detection
        for d, t in matches:
            self.trackers[t].update(detections[d, :4], detections[d, 4])

        # create new trackers for unmatched detections
        for d in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[d, :4], detections[d, 4])
            self.trackers.append(new_trk)

        # build output and remove dead trackers
        results = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or trk.age <= self.min_hits
            ):
                box = trk.get_state()
                results.append(
                    np.concatenate((box, [trk.id + 1, trk.class_id])).reshape(1, -1)
                )
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(results) > 0:
            return np.concatenate(results)
        return np.empty((0, 6))


# ==============================================================================
# 4. YOLO OBJECT DETECTOR WRAPPER
# ==============================================================================
class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4, classes_filter=None):
        from ultralytics import YOLO  # imported here so the file still loads
                                       # even before ultralytics is installed

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # dict: id -> name

        self.class_ids_filter = None
        if classes_filter:
            wanted = set(c.lower() for c in classes_filter)
            self.class_ids_filter = {
                cid for cid, name in self.class_names.items() if name.lower() in wanted
            }

    def detect(self, frame):
        """
        Runs YOLO on a frame.
        Returns np.array of shape (N, 5) -> [x1, y1, x2, y2, class_id]
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if self.class_ids_filter is not None and cls_id not in self.class_ids_filter:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append([x1, y1, x2, y2, cls_id])

        return np.array(detections) if detections else np.empty((0, 5))


# ==============================================================================
# 5. DRAWING HELPERS
# ==============================================================================
def get_color(track_id):
    """Deterministic, visually distinct colour per track ID."""
    np.random.seed(int(track_id) * 37 + 5)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def draw_tracks(frame, tracks, class_names):
    for x1, y1, x2, y2, track_id, class_id in tracks:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        class_id = int(class_id)
        color = get_color(track_id)

        label = f"{class_names.get(class_id, 'obj')} ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame


# ==============================================================================
# 6. MAIN LOOP
# ==============================================================================
def run(source, model_path, conf_threshold, output_path, classes_filter,
        max_age, min_hits, iou_threshold):

    print("[INFO] Loading YOLO model...")
    detector = ObjectDetector(model_path, conf_threshold, classes_filter)

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    # source can be an int (webcam index) or a file path
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    print("[INFO] Starting video stream. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / cannot read frame.")
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        frame = draw_tracks(frame, tracks, detector.class_names)

        # FPS overlay
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame, f"FPS: {fps:.1f}  Objects: {len(tracks)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow("Object Detection and Tracking", frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested by user.")
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


# ==============================================================================
# 7. CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time object detection and tracking (YOLO + SORT)."
    )
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0) or path to a video file. Default: 0 (webcam).",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Path or name of a YOLO model (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Minimum detection confidence threshold (0-1). Default: 0.4",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the annotated output video (e.g. result.mp4).",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Restrict detection to specific class names, e.g. --classes person car",
    )
    parser.add_argument("--max-age", type=int, default=15,
                         help="Frames to keep a lost track alive. Default: 15")
    parser.add_argument("--min-hits", type=int, default=3,
                         help="Detections needed before showing a new track. Default: 3")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                         help="IoU threshold for matching detections to tracks. Default: 0.3")

    args = parser.parse_args()

    run(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        output_path=args.output,
        classes_filter=args.classes,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
    )
