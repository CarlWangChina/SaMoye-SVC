"""
Build v2-mix train list by merging v1-core (full) + v2-new (subsampled).

Subsamples v2-new data to keep new data ratio <= MAX_NEW_RATIO.
Uses per-speaker cap to ensure balanced representation.

Usage:
    cd /path/to/SaMoye-SVC/SaMoye-Model
    python ../scripts/05_build_v2_mix_trainlist.py
"""
import os
import random
import sys
from collections import defaultdict

# Config
V1_TRAIN = "files/train_expand_v1_core.txt"
V2_NEW_DIR = "data_expand_v2_new"
V2_MIX_TRAIN = "files/train_expand_v2_mix.txt"
V2_MIX_VALID = "files/valid_expand_v2_mix.txt"
V1_VALID = "files/valid_expand_v1_core.txt"

MAX_NEW_RATIO = 0.35  # v2-new should be at most 35% of combined
SEED = 42
INVALID_SPEAKERS = {"本人", "自己", "自几", "本", "男", "女", "异性", "朋友", "亲友", "培训店长"}


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def get_speaker_from_line(line):
    """Extract speaker from train list line (first field is wav path like data_expand_v2_new/waves-32k/SPEAKER/file.wav)."""
    wav_path = line.split("|")[0]
    parts = wav_path.split("/")
    # Structure: data_dir/waves-32k/speaker/file.wav
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


def is_valid_speaker(spk):
    return spk and spk not in INVALID_SPEAKERS and not spk.startswith("unknown_")


def main():
    random.seed(SEED)

    # Read v1 train list
    v1_lines = read_lines(V1_TRAIN)
    v1_count = len(v1_lines)
    print(f"v1-core samples: {v1_count}")

    # Scan v2-new data to build potential train entries
    # We need to construct lines in the same format as preprocess_train.py would
    # Format: waves-32k/spk/file.wav|specs/spk/file.pt|pitch/spk/file.pit.npy|hubert/spk/file.vec.npy|whisper/spk/file.ppg.npy|speaker/spk/file.spk.npy

    v2_lines_by_speaker = defaultdict(list)
    waves32k_dir = os.path.join(V2_NEW_DIR, "waves-32k")

    if not os.path.isdir(waves32k_dir):
        print(f"ERROR: {waves32k_dir} not found. Run feature extraction first.")
        sys.exit(1)

    skipped_speaker_dirs = 0
    for spk in sorted(os.listdir(waves32k_dir)):
        if not is_valid_speaker(spk):
            skipped_speaker_dirs += 1
            continue
        spk_dir = os.path.join(waves32k_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for wav_file in sorted(os.listdir(spk_dir)):
            if not wav_file.endswith(".wav"):
                continue
            stem = wav_file[:-4]

            # Check all required features exist
            spec_path = os.path.join(V2_NEW_DIR, "specs", spk, f"{stem}.pt")
            pitch_path = os.path.join(V2_NEW_DIR, "pitch", spk, f"{stem}.pit.npy")
            hubert_path = os.path.join(V2_NEW_DIR, "hubert", spk, f"{stem}.vec.npy")
            whisper_path = os.path.join(V2_NEW_DIR, "whisper", spk, f"{stem}.ppg.npy")
            speaker_path = os.path.join(V2_NEW_DIR, "speaker", spk, f"{stem}.spk.npy")

            all_exist = all(os.path.exists(p) for p in [spec_path, pitch_path, hubert_path, whisper_path, speaker_path])
            if not all_exist:
                continue

            line = f"{V2_NEW_DIR}/waves-32k/{spk}/{wav_file}|{V2_NEW_DIR}/specs/{spk}/{stem}.pt|{V2_NEW_DIR}/pitch/{spk}/{stem}.pit.npy|{V2_NEW_DIR}/hubert/{spk}/{stem}.vec.npy|{V2_NEW_DIR}/whisper/{spk}/{stem}.ppg.npy|{V2_NEW_DIR}/speaker/{spk}/{stem}.spk.npy"
            v2_lines_by_speaker[spk].append(line)

    total_v2_available = sum(len(v) for v in v2_lines_by_speaker.values())
    print(f"v2-new available samples: {total_v2_available} ({len(v2_lines_by_speaker)} speakers)")
    print(f"skipped invalid speaker dirs: {skipped_speaker_dirs}")
    if total_v2_available == 0:
        print("ERROR: no complete v2-new samples found. Check feature extraction outputs first.")
        sys.exit(1)

    # Calculate target v2 count
    # v1 + v2 = total, v2/total <= MAX_NEW_RATIO
    # v2 <= MAX_NEW_RATIO * (v1 + v2)
    # v2 * (1 - MAX_NEW_RATIO) <= MAX_NEW_RATIO * v1
    # v2 <= MAX_NEW_RATIO / (1 - MAX_NEW_RATIO) * v1
    target_v2 = int(MAX_NEW_RATIO / (1 - MAX_NEW_RATIO) * v1_count)
    print(f"Target v2-new count: {target_v2} (ratio={MAX_NEW_RATIO:.0%})")

    if total_v2_available <= target_v2:
        # Use all v2 data
        v2_selected = []
        for lines in v2_lines_by_speaker.values():
            v2_selected.extend(lines)
        print(f"Using all {len(v2_selected)} v2-new samples (under target)")
    else:
        # Need to subsample: distribute quota proportionally per speaker,
        # with a minimum of 1 and a cap
        n_speakers = len(v2_lines_by_speaker)
        per_speaker_base = max(1, target_v2 // n_speakers)

        v2_selected = []
        # First pass: give each speaker min(available, per_speaker_base)
        remaining_quota = target_v2
        speakers_with_excess = []

        for spk, lines in v2_lines_by_speaker.items():
            take = min(len(lines), per_speaker_base)
            sampled = random.sample(lines, take)
            v2_selected.extend(sampled)
            remaining_quota -= take
            if len(lines) > per_speaker_base:
                speakers_with_excess.append((spk, lines, per_speaker_base))

        # Second pass: distribute remaining quota to speakers with excess
        if remaining_quota > 0 and speakers_with_excess:
            random.shuffle(speakers_with_excess)
            for spk, lines, already_taken in speakers_with_excess:
                if remaining_quota <= 0:
                    break
                extra = min(len(lines) - already_taken, remaining_quota)
                already_sampled = set(l for l in v2_selected if spk in l)
                available = [l for l in lines if l not in already_sampled]
                extra_sampled = random.sample(available, min(extra, len(available)))
                v2_selected.extend(extra_sampled)
                remaining_quota -= len(extra_sampled)

        print(f"Subsampled to {len(v2_selected)} v2-new samples")

    # Merge
    combined = v1_lines + v2_selected
    random.shuffle(combined)

    # Write combined train list
    os.makedirs("files", exist_ok=True)
    with open(V2_MIX_TRAIN, "w") as f:
        f.write("\n".join(combined) + "\n")

    # Copy v1 validation as v2 validation
    import shutil
    shutil.copy(V1_VALID, V2_MIX_VALID)

    actual_ratio = len(v2_selected) / len(combined)
    print(f"\n=== Result ===")
    print(f"v1-core:     {v1_count}")
    print(f"v2-new:      {len(v2_selected)}")
    print(f"Combined:    {len(combined)}")
    print(f"v2-new ratio: {actual_ratio:.1%}")
    print(f"Train list:  {V2_MIX_TRAIN}")
    print(f"Valid list:  {V2_MIX_VALID}")


if __name__ == "__main__":
    main()
