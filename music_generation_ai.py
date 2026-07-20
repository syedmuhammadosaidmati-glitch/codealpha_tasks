"""
================================================================================
 AI MUSIC GENERATION (LSTM-based, single file project)
================================================================================

WHAT THIS PROJECT DOES
-----------------------
1. Collects/loads MIDI files (classical, jazz, or any genre folder you point it to).
2. Preprocesses them into note/chord sequences using music21.
3. Builds a deep learning model (stacked LSTM) that learns musical patterns.
4. Trains the model on the note sequences to predict "what note comes next".
5. Generates brand new music sequences from the trained model.
6. Converts the generated sequence into a MIDI file, and (optionally) renders
   that MIDI into a playable audio file (.wav) using FluidSynth.

--------------------------------------------------------------------------------
INSTALLATION (run once)
--------------------------------------------------------------------------------
    pip install music21 tensorflow numpy

Optional, only needed if you want the generated MIDI converted straight to a
.wav audio file (otherwise you'll just get a .mid file, which any MIDI player,
DAW, or even VLC can already play):
    pip install midi2audio
    # midi2audio needs the FluidSynth program + a SoundFont (.sf2) file installed
    # on your system. See: https://github.com/bzamecnik/midi2audio

--------------------------------------------------------------------------------
GETTING TRAINING DATA
--------------------------------------------------------------------------------
Put a folder of .mid / .midi files together, e.g.:
    dataset/
        classical/bach_1.mid
        classical/mozart_2.mid
        jazz/miles_1.mid
        ...

Good free sources: kern.humdrum.org, the "Classical Piano MIDI" dataset,
freemidi.org, or your own MIDI recordings. Just point --data-dir at the folder.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
Step 1 - Train the model on your MIDI dataset:
    python music_generation_ai.py train --data-dir dataset/ --epochs 100

    This will:
      - Parse every MIDI file in dataset/ (recursively) with music21
      - Build note/chord sequences
      - Train an LSTM model
      - Save the trained model to  saved_model/music_model.h5
      - Save the vocabulary mapping to  saved_model/notes_vocab.json

Step 2 - Generate new music from the trained model:
    python music_generation_ai.py generate --length 300 --output generated_music.mid

    This will:
      - Load saved_model/music_model.h5 + notes_vocab.json
      - Generate a sequence of 300 notes/chords
      - Save it as generated_music.mid
      - If midi2audio + FluidSynth + a soundfont are available, also save
        generated_music.wav

    To render audio explicitly:
    python music_generation_ai.py generate --output out.mid --audio out.wav --soundfont /path/to/soundfont.sf2

--------------------------------------------------------------------------------
PROJECT STRUCTURE (all inside this one file)
--------------------------------------------------------------------------------
    1. collect_midi_files()        -> gathers all MIDI file paths from a folder
    2. extract_notes_from_midi()   -> uses music21 to turn a MIDI file into
                                       a sequence of note/chord tokens
    3. prepare_sequences()         -> turns tokens into training X/y sequences
    4. build_model()               -> stacked LSTM network (Keras)
    5. train()                     -> full training pipeline
    6. generate_notes()            -> samples new note sequences from the model
    7. notes_to_midi()             -> converts generated tokens back into a MIDI file
    8. midi_to_audio()             -> optional MIDI -> WAV rendering
    9. __main__                    -> CLI (train / generate subcommands)
================================================================================
"""

import argparse
import glob
import json
import os
import pickle
import random

import numpy as np


# ==============================================================================
# CONFIG / PATHS
# ==============================================================================
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "music_model.h5")
VOCAB_PATH = os.path.join(MODEL_DIR, "notes_vocab.json")
SEQUENCE_LENGTH = 50  # how many previous notes the model looks at to predict the next one


# ==============================================================================
# 1. COLLECT MIDI FILES
# ==============================================================================
def collect_midi_files(data_dir):
    """Recursively find all .mid / .midi files under data_dir."""
    patterns = [
        os.path.join(data_dir, "**", "*.mid"),
        os.path.join(data_dir, "**", "*.midi"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No .mid/.midi files found under '{data_dir}'. "
            f"Add some MIDI files (classical, jazz, etc.) to that folder first."
        )
    print(f"[INFO] Found {len(files)} MIDI files in '{data_dir}'.")
    return files


# ==============================================================================
# 2. EXTRACT NOTE/CHORD TOKENS FROM A MIDI FILE (using music21)
# ==============================================================================
def extract_notes_from_midi(file_path):
    """
    Parses a MIDI file with music21 and returns a list of string tokens:
      - a single note is represented by its pitch, e.g. "C4"
      - a chord is represented by its pitch class ids joined by '.', e.g. "4.7.11"
    This is the standard encoding used for LSTM music generation.
    """
    from music21 import converter, instrument, note, chord

    tokens = []
    try:
        midi_stream = converter.parse(file_path)
        parts = instrument.partitionByInstrument(midi_stream)
        elements = parts.parts[0].recurse() if parts else midi_stream.flat.notes
    except Exception as e:
        print(f"[WARN] Skipping '{file_path}' (could not parse): {e}")
        return tokens

    for el in elements:
        if isinstance(el, note.Note):
            tokens.append(str(el.pitch))
        elif isinstance(el, chord.Chord):
            tokens.append(".".join(str(n) for n in el.normalOrder))

    return tokens


def build_corpus(midi_files):
    """Extracts note tokens from every MIDI file and concatenates them."""
    all_tokens = []
    for i, f in enumerate(midi_files, 1):
        print(f"[INFO] Parsing ({i}/{len(midi_files)}): {os.path.basename(f)}")
        all_tokens.extend(extract_notes_from_midi(f))
    if not all_tokens:
        raise RuntimeError("No note/chord data could be extracted from the MIDI files.")
    print(f"[INFO] Total note/chord tokens extracted: {len(all_tokens)}")
    return all_tokens


# ==============================================================================
# 3. PREPARE TRAINING SEQUENCES
# ==============================================================================
def prepare_sequences(tokens, sequence_length=SEQUENCE_LENGTH):
    """
    Builds (X, y) training pairs:
      X = a window of `sequence_length` consecutive note tokens (as integers)
      y = the note token that comes right after that window
    Also returns the vocabulary mappings needed to decode predictions later.
    """
    vocab = sorted(set(tokens))
    note_to_int = {note_: i for i, note_ in enumerate(vocab)}
    int_to_note = {i: note_ for note_, i in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(len(tokens) - sequence_length):
        seq_in = tokens[i:i + sequence_length]
        seq_out = tokens[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_vocab = len(vocab)
    n_patterns = len(network_input)

    # normalise input for the LSTM and one-hot encode the output
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)

    from tensorflow.keras.utils import to_categorical
    y = to_categorical(network_output, num_classes=n_vocab)

    return X, y, note_to_int, int_to_note, n_vocab


# ==============================================================================
# 4. BUILD THE LSTM MODEL
# ==============================================================================
def build_model(sequence_length, n_vocab):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation

    model = Sequential([
        LSTM(256, input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_vocab),
        Activation("softmax"),
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    return model


# ==============================================================================
# 5. TRAINING PIPELINE
# ==============================================================================
def train(data_dir, epochs, batch_size, sequence_length):
    os.makedirs(MODEL_DIR, exist_ok=True)

    midi_files = collect_midi_files(data_dir)
    tokens = build_corpus(midi_files)

    X, y, note_to_int, int_to_note, n_vocab = prepare_sequences(tokens, sequence_length)
    print(f"[INFO] Vocabulary size: {n_vocab} unique notes/chords")
    print(f"[INFO] Training patterns: {X.shape[0]}")

    model = build_model(sequence_length, n_vocab)

    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(
        MODEL_PATH, monitor="loss", save_best_only=True, verbose=1
    )

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    # save final model + vocab (in case last epoch wasn't the "best" checkpoint)
    model.save(MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump(
            {
                "note_to_int": note_to_int,
                "int_to_note": int_to_note,
                "sequence_length": sequence_length,
                "n_vocab": n_vocab,
            },
            f,
        )

    print(f"[INFO] Training complete. Model saved to '{MODEL_PATH}'")
    print(f"[INFO] Vocabulary saved to '{VOCAB_PATH}'")


# ==============================================================================
# 6. GENERATE NEW NOTE SEQUENCES
# ==============================================================================
def load_trained_model_and_vocab():
    from tensorflow.keras.models import load_model

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(
            "No trained model found. Run 'train' first, e.g.:\n"
            "  python music_generation_ai.py train --data-dir dataset/"
        )

    model = load_model(MODEL_PATH)
    with open(VOCAB_PATH) as f:
        vocab_data = json.load(f)

    # JSON turns int keys into strings, convert back
    int_to_note = {int(k): v for k, v in vocab_data["int_to_note"].items()}
    note_to_int = vocab_data["note_to_int"]
    sequence_length = vocab_data["sequence_length"]
    n_vocab = vocab_data["n_vocab"]

    return model, note_to_int, int_to_note, sequence_length, n_vocab


def generate_notes(model, note_to_int, int_to_note, sequence_length, n_vocab,
                    length=300, temperature=1.0):
    """
    Generates a new sequence of note/chord tokens by repeatedly:
      1. Feeding the last `sequence_length` tokens into the model
      2. Sampling the next token from the predicted probability distribution
      3. Appending it and sliding the window forward
    `temperature` controls randomness: <1.0 = more conservative/predictable,
    >1.0 = more experimental/random.
    """
    # start from a random seed window taken from the vocabulary itself
    start = [random.randint(0, n_vocab - 1) for _ in range(sequence_length)]
    pattern = start[:]
    generated = []

    for _ in range(length):
        input_seq = np.reshape(pattern, (1, sequence_length, 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)[0]

        # temperature sampling
        prediction = np.log(np.maximum(prediction, 1e-8)) / temperature
        exp_preds = np.exp(prediction)
        probs = exp_preds / np.sum(exp_preds)
        index = np.random.choice(len(probs), p=probs)

        generated.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return generated


# ==============================================================================
# 7. CONVERT GENERATED TOKENS BACK TO A MIDI FILE
# ==============================================================================
def notes_to_midi(generated_tokens, output_path, tempo_bpm=120):
    from music21 import stream, note, chord, tempo as m21_tempo

    output_stream = stream.Stream()
    output_stream.append(m21_tempo.MetronomeMark(number=tempo_bpm))

    offset = 0.0
    for token in generated_tokens:
        if "." in token:
            # it's a chord: token looks like "4.7.11"
            chord_notes = [note.Note(int(n)) for n in token.split(".")]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_stream.append(new_chord)
        else:
            # it's a single note, e.g. "C4"
            new_note = note.Note(token)
            new_note.offset = offset
            output_stream.append(new_note)

        offset += 0.5  # each note/chord lasts half a beat

    output_stream.write("midi", fp=output_path)
    print(f"[INFO] Generated MIDI saved to '{output_path}'")


# ==============================================================================
# 8. OPTIONAL: RENDER MIDI TO AUDIO (.wav)
# ==============================================================================
def midi_to_audio(midi_path, audio_path, soundfont=None):
    try:
        from midi2audio import FluidSynth
    except ImportError:
        print(
            "[WARN] 'midi2audio' is not installed, skipping audio rendering.\n"
            "       Install it with: pip install midi2audio\n"
            "       (also requires the FluidSynth program + a .sf2 soundfont file)"
        )
        return

    try:
        fs = FluidSynth(sound_font=soundfont) if soundfont else FluidSynth()
        fs.midi_to_audio(midi_path, audio_path)
        print(f"[INFO] Audio rendered to '{audio_path}'")
    except Exception as e:
        print(f"[WARN] Could not render audio ({e}). The .mid file is still valid "
              f"and can be played in any MIDI player, DAW, or VLC.")


# ==============================================================================
# 9. CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AI Music Generation with LSTM (train on MIDI, then generate new music)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train subcommand ----
    train_parser = subparsers.add_parser("train", help="Train the LSTM model on a MIDI dataset.")
    train_parser.add_argument("--data-dir", required=True,
                               help="Folder containing .mid/.midi files (searched recursively).")
    train_parser.add_argument("--epochs", type=int, default=100,
                               help="Number of training epochs. Default: 100")
    train_parser.add_argument("--batch-size", type=int, default=64,
                               help="Training batch size. Default: 64")
    train_parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH,
                               help=f"Length of note context window. Default: {SEQUENCE_LENGTH}")

    # ---- generate subcommand ----
    gen_parser = subparsers.add_parser("generate", help="Generate new music from a trained model.")
    gen_parser.add_argument("--length", type=int, default=300,
                             help="Number of notes/chords to generate. Default: 300")
    gen_parser.add_argument("--temperature", type=float, default=1.0,
                             help="Sampling randomness (0.5=safe, 1.5=experimental). Default: 1.0")
    gen_parser.add_argument("--output", default="generated_music.mid",
                             help="Output MIDI file path. Default: generated_music.mid")
    gen_parser.add_argument("--audio", default=None,
                             help="Optional output .wav path to also render audio.")
    gen_parser.add_argument("--soundfont", default=None,
                             help="Path to a .sf2 soundfont file (for audio rendering).")
    gen_parser.add_argument("--tempo", type=int, default=120,
                             help="Tempo of the generated piece in BPM. Default: 120")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data_dir, args.epochs, args.batch_size, args.sequence_length)

    elif args.command == "generate":
        model, note_to_int, int_to_note, sequence_length, n_vocab = load_trained_model_and_vocab()
        generated_tokens = generate_notes(
            model, note_to_int, int_to_note, sequence_length, n_vocab,
            length=args.length, temperature=args.temperature,
        )
        notes_to_midi(generated_tokens, args.output, tempo_bpm=args.tempo)

        if args.audio:
            midi_to_audio(args.output, args.audio, soundfont=args.soundfont)


if __name__ == "__main__":
    main()
