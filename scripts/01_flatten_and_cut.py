"""
Step 1: Flatten nested train-data directories into speaker-level structure,
then cut long audio into 20-40s segments using energy-based silence detection.

Usage:
    python scripts/01_flatten_and_cut.py

Output:
    data_expand_v1/raw_clean/<speaker_id>/<speaker_id>_<song>_<seg>.wav
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
TRAIN_DATA = str(ROOT_DIR / "train-data")
OUTPUT_DIR = str(ROOT_DIR / "SaMoye-Model" / "data_expand_v1" / "raw_clean")

# Only use the two high-quality denoised directories for v1-core
SOURCE_DIRS = [
    "recordings_single_channel_denoised_cut80hz",
    "歌曲录音素材单声道降噪去80hz以下版",
]

# Cutting parameters
MIN_SEGMENT_SEC = 12.0
TARGET_SEGMENT_SEC = 30.0
MAX_SEGMENT_SEC = 40.0
OVERLAP_SEC = 3.0
SR_LOAD = 32000  # load at target SR to avoid double resampling later

# Filter: only keep "自己/本人" samples from the second dir (skip 异性/朋友)
SELF_LABELS = {"自己", "本人", "自几", "本"}


def extract_speaker_name(filepath):
    """Extract speaker name from filename patterns."""
    basename = Path(filepath).stem

    # Pattern 1: "部门-姓名-歌曲" (recordings_single_channel_denoised_cut80hz)
    # e.g. "财务部-刘博妍-爱情转移" → "刘博妍"
    parts = re.split(r"[-—]", basename)

    # Pattern 2: "ZY0021-刘贺-女-异性" (歌曲录音素材单声道降噪去80hz以下版)
    # e.g. "ZY0056-刘贺-女-异性" → "刘贺"
    zy_match = re.match(r"^[A-Z]{2}\d+-(.+?)[-—](男|女)[-—]", basename)
    if zy_match:
        return zy_match.group(1)

    # Pattern 3: "职能支持-胡玮" → "胡玮"
    if len(parts) >= 2:
        # Find the part that looks like a Chinese name (2-4 chars, all Chinese)
        for p in parts[1:]:
            p = p.strip()
            if 2 <= len(p) <= 4 and all('\u4e00' <= c <= '\u9fff' for c in p):
                return p

    # Fallback: use first 2-4 Chinese character sequence found
    cn_match = re.search(r'([\u4e00-\u9fff]{2,4})', basename)
    if cn_match:
        return cn_match.group(1)

    # Last resort: hash-based ID
    return "unknown_" + hashlib.md5(basename.encode()).hexdigest()[:8]


def should_include(filepath, source_dir):
    """Filter logic: skip 异性/朋友 from second source."""
    basename = Path(filepath).stem
    if source_dir == "歌曲录音素材单声道降噪去80hz以下版":
        # Check for label in filename
        parts = re.split(r"[-—]", basename)
        label = parts[-1].strip() if parts else ""
        if label in ("异性", "朋友", "亲友"):
            return False
    return True


def cut_audio(audio, sr, min_seg, target_seg, max_seg, overlap):
    """Cut audio into segments based on energy valleys."""
    total_dur = len(audio) / sr
    if total_dur <= max_seg:
        if total_dur >= min_seg:
            return [(0, len(audio))]
        else:
            return []

    # Compute frame energy for finding cut points
    frame_len = int(0.05 * sr)  # 50ms frames
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

        # Look for best cut point around target_seg
        target_end = start_sample + int(target_seg * sr)
        search_start = start_sample + int(min_seg * sr)
        search_end = min(start_sample + int(max_seg * sr), len(audio))

        # Convert to frame indices
        f_start = max(0, search_start // hop)
        f_end = min(n_frames, search_end // hop)

        if f_start >= f_end:
            segments.append((start_sample, min(start_sample + int(target_seg * sr), len(audio))))
            start_sample = start_sample + int(target_seg * sr) - int(overlap * sr)
            continue

        # Find minimum energy frame in search window
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

    # Collect all wav files from source dirs
    all_files = []
    for src_dir in SOURCE_DIRS:
        src_path = os.path.join(TRAIN_DATA, src_dir)
        if not os.path.isdir(src_path):
            print(f"WARNING: {src_path} not found, skipping")
            continue
        for root, dirs, files in os.walk(src_path):
            for f in files:
                if f.lower().endswith(".wav"):
                    fpath = os.path.join(root, f)
                    if should_include(fpath, src_dir):
                        all_files.append((fpath, src_dir))

    print(f"Found {len(all_files)} wav files after filtering")

    # Process each file
    speaker_counts = defaultdict(int)
    total_segments = 0
    total_skipped = 0

    for fpath, src_dir in tqdm(all_files, desc="Processing"):
        try:
            speaker = extract_speaker_name(fpath)
            speaker = sanitize_filename(speaker)

            # Load audio
            audio, sr = librosa.load(fpath, sr=SR_LOAD, mono=True)

            if len(audio) / sr < MIN_SEGMENT_SEC:
                total_skipped += 1
                continue

            # Cut into segments
            segments = cut_audio(audio, sr, MIN_SEGMENT_SEC, TARGET_SEGMENT_SEC, MAX_SEGMENT_SEC, OVERLAP_SEC)

            if not segments:
                total_skipped += 1
                continue

            # Create speaker directory
            spk_dir = os.path.join(OUTPUT_DIR, speaker)
            os.makedirs(spk_dir, exist_ok=True)

            # Save segments
            song_name = sanitize_filename(Path(fpath).stem)
            for idx, (start, end) in enumerate(segments):
                seg_audio = audio[start:end]
                # Normalize
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

    # Save speaker stats
    stats_path = os.path.join(os.path.dirname(OUTPUT_DIR), "speaker_stats.txt")
    with open(stats_path, "w") as f:
        for spk, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
            f.write(f"{spk}\t{count}\n")
    print(f"Speaker stats saved to {stats_path}")


if __name__ == "__main__":
    main()
