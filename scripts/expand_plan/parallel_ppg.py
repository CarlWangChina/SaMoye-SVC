#!/usr/bin/env python3
"""Extract whisper PPG features in parallel across multiple GPUs."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'SaMoye-Model'))

import argparse
import subprocess
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--ppg-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    ppg_dir = Path(args.ppg_dir)
    ppg_dir.mkdir(parents=True, exist_ok=True)

    spk_dirs = sorted([d.name for d in wav_dir.iterdir() if d.is_dir()])
    print(f"Total speakers: {len(spk_dirs)}, splitting across {args.num_gpus} GPUs")

    chunk_size = math.ceil(len(spk_dirs) / args.num_gpus)
    chunks = [spk_dirs[i:i + chunk_size] for i in range(0, len(spk_dirs), chunk_size)]

    list_dir = Path("/tmp/ppg_gpu_splits")
    list_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        list_file = list_dir / f"gpu_{i}.txt"
        list_file.write_text("\n".join(chunk) + "\n")

    model_dir = Path(__file__).resolve().parents[2] / "SaMoye-Model"
    script = model_dir / "prepare" / "parallel_ppg_worker.py"

    procs = []
    for i, chunk in enumerate(chunks):
        gpu_id = i
        list_file = list_dir / f"gpu_{i}.txt"
        cmd = [
            sys.executable, str(script),
            "--wav-dir", str(wav_dir),
            "--ppg-dir", str(ppg_dir),
            "--spk-list", str(list_file),
            "--gpu-id", str(gpu_id),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] {len(chunk)} speakers")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    for p in procs:
        p.wait()
        if p.returncode != 0:
            print(f"WARNING: process {p.pid} exited with {p.returncode}")

    print("All PPG extraction done.")


if __name__ == "__main__":
    main()
