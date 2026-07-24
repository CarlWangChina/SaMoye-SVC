#!/usr/bin/env python3
import argparse
import csv
import os
import re
import wave
from pathlib import Path


CORE_TOP_DIRS = {
    "recordings_single_channel_denoised_cut80hz",
    "歌曲录音素材单声道降噪去80hz以下版",
}

BAD_NAME_PATTERNS = [
    re.compile(r"\.\.[Ww][Aa][Vv]$"),
    re.compile(r"[—–]"),
    re.compile(r"\([0-9]+\)"),
    re.compile(r"自几"),
    re.compile(r"本人-本"),
    re.compile(r"\s{2,}"),
]


def detect_tag(filename: str) -> str:
    if any(k in filename for k in ("自己", "本人", "自几")) or "-本." in filename:
        return "self_like"
    if "异性" in filename:
        return "opposite"
    if "朋友" in filename or "亲友" in filename:
        return "friend"
    return "unknown"


def detect_name_issue(filename: str) -> str:
    hits = []
    for pattern in BAD_NAME_PATTERNS:
        if pattern.search(filename):
            hits.append(pattern.pattern)
    return ";".join(hits)


def read_wav_info(path: Path):
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / float(sr) if sr > 0 else 0.0
        return sr, duration, ""
    except Exception as exc:
        return 0, 0.0, str(exc)


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Path to train-data root")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    train_root = Path(args.train_data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = out_dir / "manifest_all.csv"
    v1_core_txt = out_dir / "v1_core_candidates.txt"
    low_quality_txt = out_dir / "low_quality_candidates.txt"
    name_issues_txt = out_dir / "name_issues.txt"
    stats_txt = out_dir / "stats.txt"

    rows = []
    source_counts = {}
    tag_counts = {}
    sr_counts = {}
    core_count = 0
    core_hours = 0.0

    for root, _, files in os.walk(train_root):
        for fname in files:
            path = Path(root) / fname
            if not is_audio(path):
                continue
            rel = path.relative_to(train_root).as_posix()
            top_dir = rel.split("/", 1)[0]
            tag = detect_tag(fname)
            name_issue = detect_name_issue(fname)

            sr, duration, read_err = (0, 0.0, "")
            if path.suffix.lower() == ".wav":
                sr, duration, read_err = read_wav_info(path)

            is_core_source = top_dir in CORE_TOP_DIRS
            is_sr_ok = sr in (44100, 48000)
            is_v1_core = is_core_source and tag == "self_like" and is_sr_ok and not name_issue and not read_err

            if is_v1_core:
                core_count += 1
                core_hours += duration / 3600.0

            source_counts[top_dir] = source_counts.get(top_dir, 0) + 1
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            sr_key = str(sr) if sr > 0 else "non_wav_or_unreadable"
            sr_counts[sr_key] = sr_counts.get(sr_key, 0) + 1

            rows.append(
                {
                    "relpath": rel,
                    "top_dir": top_dir,
                    "ext": path.suffix.lower(),
                    "sample_rate": sr,
                    "duration_sec": f"{duration:.6f}",
                    "tag": tag,
                    "name_issue": name_issue,
                    "read_error": read_err,
                    "is_core_source": int(is_core_source),
                    "is_v1_core_candidate": int(is_v1_core),
                }
            )

    rows.sort(key=lambda x: x["relpath"])

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relpath",
                "top_dir",
                "ext",
                "sample_rate",
                "duration_sec",
                "tag",
                "name_issue",
                "read_error",
                "is_core_source",
                "is_v1_core_candidate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with v1_core_txt.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["is_v1_core_candidate"] == 1:
                f.write(f"{row['relpath']}\n")

    with low_quality_txt.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["sample_rate"] in (8000, 16000, 22050, 24000):
                f.write(f"{row['relpath']}\n")

    with name_issues_txt.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["name_issue"]:
                f.write(f"{row['relpath']}\t{row['name_issue']}\n")

    with stats_txt.open("w", encoding="utf-8") as f:
        f.write(f"total_audio={len(rows)}\n")
        f.write(f"v1_core_count={core_count}\n")
        f.write(f"v1_core_hours={core_hours:.4f}\n")
        f.write("\n[tag_counts]\n")
        for k in sorted(tag_counts):
            f.write(f"{k}={tag_counts[k]}\n")
        f.write("\n[sample_rate_counts]\n")
        for k in sorted(sr_counts, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else 0)):
            f.write(f"{k}={sr_counts[k]}\n")
        f.write("\n[top_dir_counts]\n")
        for k, v in sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"{k}={v}\n")

    print(f"Wrote: {manifest_csv}")
    print(f"Wrote: {v1_core_txt}")
    print(f"Wrote: {low_quality_txt}")
    print(f"Wrote: {name_issues_txt}")
    print(f"Wrote: {stats_txt}")


if __name__ == "__main__":
    main()
