"""
Step 3: Flatten and cut additional train-data directories for v2-expand.

Adds non-denoised "自己/本人" samples from monthly recording directories,
excluding low sample-rate files (<44100 Hz) and 异性/朋友 labels.

v1-core already processed:
  - recordings_single_channel_denoised_cut80hz
  - 歌曲录音素材单声道降噪去80hz以下版

v2-expand adds the remaining directories.

Usage:
    python scripts/03_flatten_and_cut_v2.py

Output:
    SaMoye-Model/data_expand_v2_new/raw_clean/<speaker_id>/<speaker_id>_<song>_<seg>.wav
"""
import os
import re
import hashlib
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# === Config ===
ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATA = str(Path(os.environ.get("SAMOYE_TRAIN_DATA", ROOT_DIR / "train-data")))
if not os.path.isdir(TRAIN_DATA):
    fallback = ROOT_DIR / "歌曲录音素材单声道降噪去80hz以下版"
    if fallback.is_dir():
        TRAIN_DATA = str(fallback)
OUTPUT_DIR = str(ROOT_DIR / "SaMoye-Model" / "data_expand_v2_new" / "raw_clean")

# Directories already used in v1-core (skip these)
V1_DIRS = {
    "recordings_single_channel_denoised_cut80hz",
    "歌曲录音素材单声道降噪去80hz以下版",
}

# Cutting parameters (same as v1)
MIN_SEGMENT_SEC = 12.0
TARGET_SEGMENT_SEC = 30.0
MAX_SEGMENT_SEC = 40.0
OVERLAP_SEC = 3.0
SR_LOAD = 32000

# Minimum original sample rate to accept
MIN_ORIG_SR = 44100

# Labels to skip
SKIP_LABELS = {"异性", "朋友", "亲友"}

# Labels that indicate self-recordings (prefer these)
SELF_LABELS = {"自己", "本人", "自几", "本"}


def extract_speaker_name(filepath):
    """Extract speaker name from filename patterns."""
    basename = Path(filepath).stem

    # Pattern: "ZY0021-刘贺-女-自己"
    basename = basename.replace("–", "-").replace("—", "-")

    zy_match = re.match(r"^[A-Z]{2}\d+[_-](.+?)[-](男|女|本人|自己|自几)", basename)
    if zy_match:
        return zy_match.group(1)

    # Pattern: "部门-姓名-歌曲"
    parts = re.split(r"[-—–]", basename)
    if len(parts) >= 2:
        for p in parts[1:]:
            p = p.strip()
            if 2 <= len(p) <= 4 and all('\u4e00' <= c <= '\u9fff' for c in p):
                return p

    # Fallback: first Chinese name-like sequence
    cn_match = re.search(r'([\u4e00-\u9fff]{2,4})', basename)
    if cn_match:
        return cn_match.group(1)

    return "unknown_" + hashlib.md5(basename.encode()).hexdigest()[:8]


def get_label_from_filename(filepath):
    """Extract label (自己/异性/朋友/unknown) from filename."""
    basename = Path(filepath).stem
    parts = re.split(r"[-—–]", basename)
    if parts:
        last = parts[-1].strip()
        if last in SKIP_LABELS:
            return last
        if last in SELF_LABELS:
            return "self"
    # Check if any part contains skip labels
    for p in parts:
        p = p.strip()
        if p in SKIP_LABELS:
            return p
    return "unknown"


def should_include(filepath):
    """Filter: skip 异性/朋友, keep 自己/本人 and unlabeled."""
    label = get_label_from_filename(filepath)
    if label in SKIP_LABELS:
        return False
    return True


def check_sample_rate(filepath):
    """Check if original sample rate meets minimum threshold."""
    try:
        info = sf.info(filepath)
        return info.samplerate >= MIN_ORIG_SR
    except Exception:
        return False


def cut_audio(audio, sr, min_seg, target_seg, max_seg, overlap):
    """Cut audio into segments based on energy valleys."""
    total_dur = len(audio) / sr
    if total_dur <= max_seg:
        if total_dur >= min_seg:
            return [(0, len(audio))]
        else:
            return []

    frame_len = int(0.05 * sr)
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop + 1
    energy = np.array([
        np.sum(audio[i * hop:i * hop + frame_len] ** 2)
        for i in range(n_frames)
    ])

    segments = []
    start_sample = 0

    while start_sample < len(audio):
        remaining = len(audio) - start_sample
        remaining_dur = remaining / sr

        if remaining_dur <= max_seg:
            if remaining_dur >= min_seg:
                segments.append((start_sample, len(audio)))
            break

        search_start = start_sample + int(min_seg * sr)
        search_end = min(start_sample + int(max_seg * sr), len(audio))

        f_start = max(0, search_start // hop)
        f_end = min(n_frames, search_end // hop)

        if f_start >= f_end:
            segments.append((start_sample, min(start_sample + int(target_seg * sr), len(audio))))
            start_sample = start_sample + int(target_seg * sr) - int(overlap * sr)
            continue

        best_frame = f_start + np.argmin(energy[f_start:f_end])
        cut_sample = best_frame * hop

        segments.append((start_sample, cut_sample))
        start_sample = cut_sample - int(overlap * sr)

    return segments


def sanitize_filename(name):
    """Remove problematic characters from filenames."""
    name = re.sub(r'[<>:"/\\|?*\s]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_.')
    return name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all wav files from non-v1 directories
    all_files = []
    skipped_sr = 0
    skipped_label = 0

    for entry in sorted(os.listdir(TRAIN_DATA)):
        if entry in V1_DIRS:
            continue
        src_path = os.path.join(TRAIN_DATA, entry)
        if not os.path.isdir(src_path):
            continue

        for root, dirs, files in os.walk(src_path):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                fpath = os.path.join(root, f)

                if not should_include(fpath):
                    skipped_label += 1
                    continue

                if not check_sample_rate(fpath):
                    skipped_sr += 1
                    continue

                all_files.append(fpath)

    print(f"Found {len(all_files)} wav files after filtering")
    print(f"  Skipped (label=异性/朋友): {skipped_label}")
    print(f"  Skipped (SR < {MIN_ORIG_SR}): {skipped_sr}")

    # Process each file
    speaker_counts = defaultdict(int)
    total_segments = 0
    total_skipped = 0

    for fpath in tqdm(all_files, desc="Processing"):
        try:
            speaker = extract_speaker_name(fpath)
            speaker = sanitize_filename(speaker)

            audio, sr = librosa.load(fpath, sr=SR_LOAD, mono=True)

            if len(audio) / sr < MIN_SEGMENT_SEC:
                total_skipped += 1
                continue

            segments = cut_audio(audio, sr, MIN_SEGMENT_SEC, TARGET_SEGMENT_SEC, MAX_SEGMENT_SEC, OVERLAP_SEC)

            if not segments:
                total_skipped += 1
                continue

            spk_dir = os.path.join(OUTPUT_DIR, speaker)
            os.makedirs(spk_dir, exist_ok=True)

            song_name = sanitize_filename(Path(fpath).stem)
            for idx, (start, end) in enumerate(segments):
                seg_audio = audio[start:end]
                peak = np.max(np.abs(seg_audio))
                if peak > 0:
                    seg_audio = seg_audio / peak * 0.9

                seg_name = f"{speaker}_{song_name}_{idx:03d}.wav"
                seg_path = os.path.join(spk_dir, seg_name)
                sf.write(seg_path, seg_audio, sr)
                total_segments += 1

            speaker_counts[speaker] += len(segments)
        except Exception as e:
            print(f"ERROR processing {fpath}: {e}")
            total_skipped += 1

    print(f"\n=== Summary ===")
    print(f"Total input files: {len(all_files)}")
    print(f"Total segments created: {total_segments}")
    print(f"Total skipped: {total_skipped}")
    print(f"Unique speakers: {len(speaker_counts)}")
    print(f"Output directory: {OUTPUT_DIR}")

    stats_path = os.path.join(os.path.dirname(OUTPUT_DIR), "speaker_stats_v2.txt")
    with open(stats_path, "w") as f:
        for spk, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
            f.write(f"{spk}\t{count}\n")
    print(f"Speaker stats saved to {stats_path}")


if __name__ == "__main__":
    main()
