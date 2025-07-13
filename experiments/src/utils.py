import torch
import librosa
import numpy as np
from typing import List
from pathlib import Path

import crepe
from whisper.model import (Whisper, ModelDimensions)
from whisper.audio import (load_audio, log_mel_spectrogram)
from hubert import (hubert_model)
from vad.utils import (get_speech_timestamps)


def load_whisper_model(path, device):
    """ Load whisper model
    """

    ckptDict = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**ckptDict["dims"])
    model = Whisper(dims)

    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = cut * -1

    del model.encoder.blocks[cut:]
    model.load_state_dict(ckptDict["model_state_dict"], strict=False)

    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)

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
    return ppg_a


def load_hubert_model(path, device):
    """ Load hubert model
    """

    model = hubert_model.hubert_soft(path)
    model.eval()

    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while idx_s + 20 * 16000 < audln:
        feats = audio[idx_s: idx_s + 20 * 16000]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if idx_s < audln:
        feats = audio[idx_s:audln]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec = vec[None] if vec.ndim == 1 else vec
            vec_a.extend(vec)
    return vec_a


def compute_f0_sing(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    fmin = 50
    fmax = 1000  # by default set to 1000
    model = "full"
    batch_size = 512
    pitch = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    pitch = crepe.filter.mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


def post_process(vad_model, refWavPath: str, svcWav: str):
    """_summary_

    Args:
        ref_wave_path (str): Path of ref audio.
        svc_wave_path (str): Path of svc audio.
    Returns:
        _type_: _description_
    """
    ref_wave, _ = librosa.load(refWavPath, sr=16000)
    tmp_wave = torch.from_numpy(ref_wave).squeeze(0)
    tag_wave = get_speech_timestamps(
        tmp_wave, vad_model, threshold=0.2, sampling_rate=16000
    )

    ref_wave[:] = 0
    for tag in tag_wave:
        ref_wave[tag["start"]: tag["end"]] = 1

    ref_wave = np.repeat(ref_wave, 2, -1)

    min_len = min(len(ref_wave), len(svcWav))
    ref_wave = ref_wave[:min_len]
    svc_wave = svcWav[:min_len]
    # svc_wave[ref_wave == 0] = 0
    return svc_wave, 32000


def load_targer_speakers(readBaseDir: str) -> List:
    """ Load target speakers from ReadBaseDIr/{pitch, singer},
        Return a list of speakers which are the common speaker in pitch and singer.
    """
    readBaseDir = Path(readBaseDir)
    pitchSpeakers = [spk.stem for spk in (
        readBaseDir / "pitch").iterdir() if spk.is_file()]
    singerSpeakers = [spk.stem[:-4] for spk in (
        readBaseDir / "singer").iterdir() if spk.is_file()]  # spk.stem = "spkname.spk"

    retList = [ele for ele in pitchSpeakers if ele in singerSpeakers]
    return retList
