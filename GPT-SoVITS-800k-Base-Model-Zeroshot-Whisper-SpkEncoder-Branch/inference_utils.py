import re
import LangSegment
import torch
import numpy as np
import librosa
import ffmpeg
import utils
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

is_half = torch.cuda.is_available()


def load_audio(file, sr):
    try:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def get_svc_wav(
        origin_wav_path, ref_wav_path, ssl_model, f0_model, svc_model, hps
    ):

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )

    with torch.no_grad():
        origin_wav, sr = librosa.load(origin_wav_path, sr=32000)
        f0 = f0_model.infer_from_audio(origin_wav)
        f0 = torch.FloatTensor(f0)
        
        if is_half:
            f0 = f0.half().to(device)
        else:
            f0 = f0.to(device)

        print(f0.shape, origin_wav.shape)
        origin_wav_16k = librosa.resample(y=origin_wav, orig_sr=32000, target_sr=16000)
        origin_wav_duration = origin_wav_16k.shape[0] / sr
        origin_wav_16k = torch.from_numpy(origin_wav_16k)

        if is_half:
            origin_wav_16k = origin_wav_16k.half().to(device)
        else:
            origin_wav_16k = origin_wav_16k.to(device)
    
    with torch.no_grad():
        ref_wav_16k, sr = librosa.load(ref_wav_path, sr=16000)
        ref_wav_duration = ref_wav_16k.shape[0] / sr

        ref_wav_16k = torch.from_numpy(ref_wav_16k)

        if is_half:
            ref_wav_16k = ref_wav_16k.half().to(device)
        else:
            ref_wav_16k = ref_wav_16k.to(device)        

    audio_opt = []

    pred_semantic = ssl_model.model(origin_wav_16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
    refer_spec = get_spepc(hps, ref_wav_path)

    if is_half:
        refer_spec = refer_spec.half().to(device)
    else:
        refer_spec = refer_spec.to(device)
    
 
    generated_audio = svc_model.decode(
        ssl=pred_semantic, f0=f0.unsqueeze(0), refer_spec=refer_spec
    )
    audio = (
        generated_audio.detach().cpu().numpy()[0, 0]
    )

    audio_opt.append(audio)
    audio_opt.append(zero_wav)
    
    audio = np.concatenate(audio_opt, axis=0) * 32768
    audio = audio.astype(np.float16)
    return audio, hps.data.sampling_rate

