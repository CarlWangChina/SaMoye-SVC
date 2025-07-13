import numpy as np
import torch

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram

from utils import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, device) -> Whisper:
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    # print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    # torch.save({
    #     'dims': checkpoint["dims"],
    #     'model_state_dict': model.state_dict(),
    # }, "large-v2.pt")
    return model


def pred_ppg(whisper, wavPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while idx_s + 15 * 16000 < audln:
        short = audio[idx_s: idx_s + 15 * 16000]
        idx_s = idx_s + 15 * 16000
        ppgln = 15 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(
                0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if idx_s < audln:
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(
                0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    # np.save(ppgPath, ppg_a)
    return ppg_a


whisper_model = load_model(
    "pretrainModels/whisper_pretrain/large-v2.pt", device)


def whisper_infer(wav: str):
    return pred_ppg(whisper_model, wav, device)
