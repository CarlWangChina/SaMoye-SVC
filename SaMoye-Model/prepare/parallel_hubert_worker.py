#!/usr/bin/env python3
"""Single-GPU hubert worker - processes a subset of speakers."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm
from hubert import hubert_model


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--vec-dir", required=True)
    parser.add_argument("--spk-list", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES already set
    print(f"[GPU {args.gpu_id}] Loading hubert model...")

    model = hubert_model.hubert_soft(os.path.join("hubert_pretrain", "hubert-soft-0d54a1f4.pt"))
    model.eval()
    model.half()
    model.to(device)

    with open(args.spk_list) as f:
        spk_names = [line.strip() for line in f if line.strip()]

    total_done = 0
    total_skip = 0
    for spk in spk_names:
        wav_spk_dir = os.path.join(args.wav_dir, spk)
        vec_spk_dir = os.path.join(args.vec_dir, spk)
        if not os.path.isdir(wav_spk_dir):
            continue
        os.makedirs(vec_spk_dir, exist_ok=True)

        files = [f[:-4] for f in os.listdir(wav_spk_dir) if f.endswith(".wav")]
        for stem in tqdm(files, desc=f"[GPU{args.gpu_id}] hubert {spk}"):
            vec_path = os.path.join(vec_spk_dir, f"{stem}.vec.npy")
            if os.path.isfile(vec_path):
                total_skip += 1
                continue
            wav_path = os.path.join(wav_spk_dir, f"{stem}.wav")
            try:
                feats = load_audio(wav_path)
                feats = torch.from_numpy(feats).to(device)
                feats = feats[None, None, :].half()
                with torch.no_grad():
                    vec = model.units(feats).squeeze().data.cpu().float().numpy()
                    np.save(vec_path, vec, allow_pickle=False)
                total_done += 1
            except Exception as e:
                print(f"[GPU{args.gpu_id}] ERROR {wav_path}: {e}")

    print(f"[GPU {args.gpu_id}] Done: {total_done} extracted, {total_skip} skipped")


if __name__ == "__main__":
    main()
