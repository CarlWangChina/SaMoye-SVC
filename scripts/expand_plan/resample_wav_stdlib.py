#!/usr/bin/env python3
import argparse
import audioop
import os
import wave
from pathlib import Path


def resample_mono_pcm16(src_path: Path, dst_path: Path, target_sr: int):
    with wave.open(str(src_path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sw != 2:
        raise RuntimeError(f"unsupported sample width {sw} in {src_path}")

    if nch == 2:
        frames = audioop.tomono(frames, sw, 0.5, 0.5)
        nch = 1
    elif nch != 1:
        raise RuntimeError(f"unsupported channel count {nch} in {src_path}")

    if sr != target_sr:
        frames, _ = audioop.ratecv(frames, sw, nch, sr, target_sr, None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(target_sr)
        out.writeframes(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-sr", type=int, choices=[16000, 32000], required=True)
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    failures = []

    for root, _, files in os.walk(in_dir):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            src = Path(root) / name
            rel = src.relative_to(in_dir)
            dst = out_dir / rel
            try:
                resample_mono_pcm16(src, dst, args.target_sr)
                ok += 1
            except Exception as exc:
                fail += 1
                failures.append((str(rel), str(exc)))

    report = out_dir.parent / f"resample_{args.target_sr}_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"ok={ok}\n")
        f.write(f"fail={fail}\n")
        for rel, err in failures:
            f.write(f"{rel}\t{err}\n")

    print(f"target_sr={args.target_sr} ok={ok} fail={fail}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
