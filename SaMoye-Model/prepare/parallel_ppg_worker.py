#!/usr/bin/env python3
"""Single-GPU whisper PPG worker - processes a subset of speakers."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from tqdm import tqdm
from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path):
    device = "cuda:0"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.half()
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--ppg-dir", required=True)
    parser.add_argument("--spk-list", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    print(f"[GPU {args.gpu_id}] Loading whisper model...")
    whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pt"))

    with open(args.spk_list) as f:
        spk_names = [line.strip() for line in f if line.strip()]

    total_done = 0
    total_skip = 0
    for spk in spk_names:
        wav_spk_dir = os.path.join(args.wav_dir, spk)
        ppg_spk_dir = os.path.join(args.ppg_dir, spk)
        if not os.path.isdir(wav_spk_dir):
            continue
        os.makedirs(ppg_spk_dir, exist_ok=True)

        files = [f[:-4] for f in os.listdir(wav_spk_dir) if f.endswith(".wav")]
        for stem in tqdm(files, desc=f"[GPU{args.gpu_id}] ppg {spk}"):
            ppg_path = os.path.join(ppg_spk_dir, f"{stem}.ppg.npy")
            if os.path.isfile(ppg_path):
                total_skip += 1
                continue
            wav_path = os.path.join(wav_spk_dir, f"{stem}.wav")
            try:
                audio = load_audio(wav_path)
                audln = audio.shape[0]
                ppgln = audln // 320
                audio = pad_or_trim(audio)
                mel = log_mel_spectrogram(audio).half().to(whisper.device)
                with torch.no_grad():
                    ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
                    ppg = ppg[:ppgln,]
                    np.save(ppg_path, ppg, allow_pickle=False)
                total_done += 1
            except Exception as e:
                print(f"[GPU{args.gpu_id}] ERROR {wav_path}: {e}")

    print(f"[GPU {args.gpu_id}] Done: {total_done} extracted, {total_skip} skipped")


if __name__ == "__main__":
    main()
