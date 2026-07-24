#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path


def sanitize_component(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120] if len(text) > 120 else text


def parse_speaker_name(filename: str) -> str:
    stem = filename
    for ext in [".wav", ".mp3", ".m4a", ".flac", ".WAV", ".MP3", ".M4A", ".FLAC"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    # Common format: CODE-NAME-gender-tag
    parts = stem.split("-")
    if len(parts) >= 4:
        candidate = parts[1].strip()
        if candidate:
            return candidate

    # Fallback: extract first Chinese name-like segment
    zh_segments = re.findall(r"[\u4e00-\u9fff]{2,8}", stem)
    if zh_segments:
        return zh_segments[0]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--candidate-list", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    train_root = Path(args.train_data).resolve()
    candidate_list = Path(args.candidate_list).resolve()
    output_root = Path(args.output_root).resolve()
    raw_clean = output_root / "raw_clean"
    reports = output_root / "reports"
    raw_clean.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    unresolved = []
    mapped = []
    dup_count = 0

    with candidate_list.open("r", encoding="utf-8") as f:
        relpaths = [line.strip() for line in f if line.strip()]

    seen_target = set()
    for rel in relpaths:
        src = train_root / rel
        if not src.exists():
            unresolved.append((rel, "missing_source"))
            continue

        speaker_name = parse_speaker_name(src.name)
        if not speaker_name:
            unresolved.append((rel, "cannot_parse_speaker"))
            continue

        speaker_id = sanitize_component(speaker_name)
        if not speaker_id:
            unresolved.append((rel, "empty_speaker_after_sanitize"))
            continue

        speaker_dir = raw_clean / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        clip_stem = sanitize_component(src.stem)
        ext = src.suffix.lower() if src.suffix else ".wav"
        target = speaker_dir / f"{clip_stem}{ext}"
        if target in seen_target:
            dup_count += 1
            target = speaker_dir / f"{clip_stem}__dup{dup_count}{ext}"

        seen_target.add(target)
        if not target.exists():
            os.symlink(src, target)

        mapped.append((rel, speaker_id, target.relative_to(output_root).as_posix()))

    mapped_tsv = reports / "v1_core_mapped.tsv"
    unresolved_tsv = reports / "v1_core_unresolved.tsv"
    stats_txt = reports / "v1_core_materialize_stats.txt"

    with mapped_tsv.open("w", encoding="utf-8") as f:
        f.write("source_relpath\tspeaker_id\ttarget_relpath\n")
        for row in mapped:
            f.write("\t".join(row) + "\n")

    with unresolved_tsv.open("w", encoding="utf-8") as f:
        f.write("source_relpath\treason\n")
        for row in unresolved:
            f.write("\t".join(row) + "\n")

    speaker_dirs = [p for p in raw_clean.iterdir() if p.is_dir()]
    clip_total = 0
    for spk in speaker_dirs:
        clip_total += len(list(spk.iterdir()))

    with stats_txt.open("w", encoding="utf-8") as f:
        f.write(f"candidate_total={len(relpaths)}\n")
        f.write(f"mapped_total={len(mapped)}\n")
        f.write(f"unresolved_total={len(unresolved)}\n")
        f.write(f"speaker_total={len(speaker_dirs)}\n")
        f.write(f"clip_total={clip_total}\n")
        f.write(f"duplicate_target_renamed={dup_count}\n")

    print(f"Wrote: {mapped_tsv}")
    print(f"Wrote: {unresolved_tsv}")
    print(f"Wrote: {stats_txt}")


if __name__ == "__main__":
    main()
