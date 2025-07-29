import os
import time
import random
import mido
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta

# --- Configuration ---
SONG_DIR = "Songs"
OUTPUT = "intelligent_output.mid"
N_GRAM = 4
MAX_NOTES = 2000
TICKS_PER_BEAT = 480
THREADS = os.cpu_count()
TOP_K = 5

# --- Utility Functions ---
def find_midi_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                yield os.path.join(root, file)

def format_eta(seconds_remaining):
    return str(timedelta(seconds=int(seconds_remaining)))

def note_to_pitch_class(note):
    return note % 12

def estimate_key(sequence):
    counts = Counter(note_to_pitch_class(n[0]) for n in sequence)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else 0

# --- MIDI Parsing ---
def parse_midi(path):
    try:
        midi = mido.MidiFile(path)
        abs_time = 0
        active_notes = {}
        sequence = []

        for msg in midi:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (abs_time, msg.velocity)
            elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                if msg.note in active_notes:
                    on_time, velocity = active_notes.pop(msg.note)
                    duration = abs_time - on_time
                    sequence.append((msg.note, int(duration * TICKS_PER_BEAT), velocity, on_time))

        # No sequence? No pass. Simple as that.
        if not sequence:
            return None

        key_est = estimate_key(sequence)
        normalized = normalize_sequence(sequence, key_est)
        return normalized
    except Exception as E:
        print(f'{E}. This line says print("{E}")')
        return None

def normalize_sequence(seq, key):
    normalized = []
    prev_off = 0
    for note, dur, vel, start_time in seq:
        rel_pitch = note - key
        ioi = start_time - prev_off
        normalized.append(((rel_pitch, dur, vel, round(start_time % 4, 2)), round(ioi, 3)))
        prev_off = start_time + (dur / TICKS_PER_BEAT)
    return normalized

def load_sequences_parallel(files):
    sequences = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(parse_midi, f): f for f in files}
        completed = 0
        with tqdm(total=len(futures), desc="Parsing MIDI", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        sequences.append(result)
                except Exception:
                    pass
                completed += 1
                elapsed = time.time() - start
                if completed:
                    rate = elapsed / completed
                    remaining = (len(futures) - completed) * rate
                    pbar.set_postfix(eta=format_eta(remaining))
                pbar.update(1)
    return sequences

# --- N-Gram Model ---
def build_ngram(sequences, n=N_GRAM):
    model = defaultdict(Counter)
    for seq in sequences:
        tokens = [entry[0] + (entry[1],) for entry in seq]
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i + n - 1])
            next_note = tokens[i + n - 1]
            model[context][next_note] += 1
    return dict(model)

# --- Fuzzy Matching Preprocessing ---
def vectorize_contexts(model):
    keys = list(model.keys())
    vectors = np.array([np.array([i for token in ctx for i in token]) for ctx in keys])
    nn = NearestNeighbors(n_neighbors=TOP_K, metric='cosine').fit(vectors)
    return keys, vectors, nn

# --- Generation ---
def generate_sequence(model, keys, vectors, nn, max_notes=MAX_NOTES):
    seed = random.choice(keys)
    result = list(seed)

    while len(result) < max_notes:
        context = tuple(result[-(N_GRAM - 1):])

        if context in model:
            choices, weights = zip(*model[context].items())
            next_note = random.choices(choices, weights=weights)[0]
        else:
            context_vec = np.array([i for token in context for i in token]).reshape(1, -1)
            distances, indices = nn.kneighbors(context_vec, n_neighbors=TOP_K)
            for idx in indices[0]:
                k = keys[idx]
                choices, weights = zip(*model[k].items())
                next_note = random.choices(choices, weights=weights)[0]
                if next_note:
                    break
            else:
                break

        result.append(next_note)

    # Build final sequence using stored IOIs
    final_sequence = []
    prev_end_time = 0.0
    for token in result:
        note, dur, vel, beat, ioi = token  # Fixed unpacking
        prev_end_time += (dur / TICKS_PER_BEAT) + ioi
        final_sequence.append(((note, dur, vel, beat), ioi))

    return final_sequence

# --- Writing ---
def write_midi(sequence, output_path, key_offset=0):
    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    for (note, dur, vel, beat), ioi in sequence:
        pitch = note + key_offset
        delta_time_on = max(0, int(ioi * TICKS_PER_BEAT))
        delta_time_off = max(0, int(dur))

        track.append(mido.Message('note_on', note=pitch, velocity=vel, time=delta_time_on))
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta_time_off))

    midi.save(output_path)

# --- Main ---
def main():
    total_start = time.time()

    print("Finding MIDI files...")
    files = list(find_midi_files(SONG_DIR))
    if not files:
        print("No MIDI files found. Exiting.")
        return
    print(f"Found {len(files)} MIDI files.\n")

    print("Loading sequences...")
    start = time.time()
    sequences = load_sequences_parallel(files)
    print(f"Loaded {len(sequences)} usable sequences in {time.time() - start:.2f}s\n")

    if not sequences:
        print("No usable sequences. Exiting.")
        return

    print("Building Markov model...")
    start = time.time()
    model = build_ngram(sequences)
    print(f"Model built in {time.time() - start:.2f}s\n")

    print("Vectorizing contexts...")
    start = time.time()
    keys, vectors, nn = vectorize_contexts(model)
    print(f"Vectors prepared in {time.time() - start:.2f}s\n")

    print("Generating sequence...")
    start = time.time()
    sequence = generate_sequence(model, keys, vectors, nn)
    print(f"Sequence generated in {time.time() - start:.2f}s\n")

    print(f"Writing to '{OUTPUT}'...")
    start = time.time()
    write_midi(sequence, OUTPUT)
    print(f"Saved in {time.time() - start:.2f}s")

    print(f"\nDone. Total runtime: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()