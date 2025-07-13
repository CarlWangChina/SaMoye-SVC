import torch
import librosa
import numpy as np
import torch.nn.functional as F
from typing import Union

import crepe
from rmvpe import RMVPE

from ipdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_average(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)


def compute_f0_mouth(path):
    # pip install praat-parselmouth
    import parselmouth

    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
    lpad = 1024 // 160
    rpad = lpad
    f0 = (
        parselmouth.Sound(x, sr)
        .to_pitch_ac(
            time_step=160 / sr,
            voicing_threshold=0.5,
            pitch_floor=30,
            pitch_ceiling=1000,
        )
        .selected_array["frequency"]
    )
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")
    return f0


def compute_f0_salience(filename, device):
    from pitch.core.salience import salience

    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    f0, t, s = salience(audio, Fs=sr, H=320, N=2048, F_min=45.0, F_max=1760.0)
    f0 = np.repeat(f0, 2, -1)  # 320 -> 160 * 2
    f0 = move_average(f0, 3)
    return f0


def compute_f0_voice(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 10 millisecond hop length
    hop_length = 160
    fmin = 50
    fmax = 1000
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
    pitch = crepe.filter.mean(pitch, 3)
    pitch = pitch.squeeze(0)
    return pitch


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
        decoder=crepe.decode.viterbi,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    pitch = crepe.filter.mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


# Temp define model load
rmvpeModel = RMVPE("rmvpe/pretrainModel/model.pt", device)


def rmvpe_compute_f0_sing(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))

    hop_length = 320
    threshold = 0.05

    p_len = audio.shape[0] // hop_length
    f0 = rmvpeModel.infer_from_audio(audio, sr, threshold)
    if torch.all(f0 == 0):
        rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
        return rtn, rtn
    # pitch, _ = post_process(audio, sr, hop_length, f0, p_len)
    pitch, _ = post_process1(audio, sr, hop_length, f0, p_len)
    pitch = rmvpe_mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


def rmvpe_mean(signals, win_length=9):
    """Averave filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """
    signals = torch.from_numpy(signals[np.newaxis, :])
    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)

    # Apply the mask by setting masked elements to zero, or make NaNs zero
    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))

    # Create a ones kernel with the same number of channels as the input tensor
    ones_kernel = torch.ones(signals.size(
        1), 1, win_length, device=signals.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=win_length // 2,
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=win_length // 2,
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    # Fill zero values with NaNs
    # avg_pooled[avg_pooled == 0] = float("nan")

    return avg_pooled.squeeze(1)

# def post_process(x, sampling_rate, hop_length, f0, pad_to):
#     if isinstance(f0, np.ndarray):
#         f0 = torch.from_numpy(f0).float().to(x.device)

#     if pad_to is None:
#         return f0

#     f0 = repeat_expand(f0, pad_to)

#     vuv_vector = torch.zeros_like(f0)
#     vuv_vector[f0 > 0.0] = 1.0
#     vuv_vector[f0 <= 0.0] = 0.0

#     # 去掉0频率, 并线性插值
#     nzindex = torch.nonzero(f0).squeeze()
#     f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
#     time_org = hop_length / sampling_rate * nzindex.cpu().numpy()
#     time_frame = np.arange(pad_to) * hop_length / sampling_rate

#     vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

#     if f0.shape[0] <= 0:
#         return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
#     if f0.shape[0] == 1:
#         return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy(), vuv_vector.cpu().numpy()

#     # 大概可以用 torch 重写?
#     f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
#     # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

#     f0 = np.repeat(f0, 2, -1)
#     return f0, vuv_vector.cpu().numpy()


def post_process(x, sampling_rate, hop_length, f0, pad_to):
    if isinstance(f0, np.ndarray):
        f0 = torch.from_numpy(f0).float().to(x.device)

    if pad_to is None:
        return f0

    f0 = repeat_expand(f0, pad_to)

    vuv_vector = torch.zeros_like(f0)
    vuv_vector[f0 > 0.0] = 1.0
    vuv_vector[f0 <= 0.0] = 0.0

    # 去掉0频率, 并线性插值
    nzindex = torch.nonzero(f0).squeeze()
    f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
    time_org = hop_length / sampling_rate * nzindex.cpu().numpy()
    time_frame = np.arange(pad_to) * hop_length / sampling_rate

    vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

    if f0.shape[0] <= 0:
        return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
    if f0.shape[0] == 1:
        return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy(), vuv_vector.cpu().numpy()

    # 大概可以用 torch 重写?
    # f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    f0 = np.interp(time_frame, time_org, f0, left=0, right=0)
    # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

    f0 = np.repeat(f0, 2, -1)
    return f0, vuv_vector.cpu().numpy()


def post_process1(x, sampling_rate, hop_len, f0, pad_to):
    if isinstance(f0, np.ndarray):
        f0 = torch.from_numpy(f0).float().to(x.device)

    if pad_to is None:
        return f0

    f0 = repeat_expand(f0, pad_to)

    vuv_vector = torch.zeros_like(f0)
    vuv_vector[f0 > 0.0] = 1.0
    vuv_vector[f0 <= 0.0] = 0.0

    nzindex = torch.nonzero(f0).squeeze()
    segments = []
    for i in range(len(nzindex) - 1):
        if nzindex[i + 1] - nzindex[i] > 1 and nzindex[i + 1] - nzindex[i] < 30:
            segments.append((nzindex[i], nzindex[i + 1]))

    for start, end in segments:
        slope = (f0[end] - f0[start]) / (end - start)
        intercept = f0[start] - slope * start
        f0[start + 1: end] = slope * \
            torch.arange(start + 1, end, device=f0.device) + intercept

    f0 = f0.cpu().numpy()
    f0 = np.repeat(f0, 2, -1)
    return f0, vuv_vector.cpu().numpy()

# 使用三次样条插值


def post_process2(x, sampling_rate, hop_length, f0, pad_to):
    if isinstance(f0, np.ndarray):
        f0 = torch.from_numpy(f0).float().to(x.device)

    if pad_to is None:
        return f0

    f0 = repeat_expand(f0, pad_to)

    vuv_vector = torch.zeros_like(f0)
    vuv_vector[f0 > 0.0] = 1.0
    vuv_vector[f0 <= 0.0] = 0.0

    nzindex = torch.nonzero(f0).squeeze()
    f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
    time_org = hop_length / sampling_rate * nzindex.cpu().numpy()
    time_frame = np.arange(pad_to) * hop_length / sampling_rate

    vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

    if f0.shape[0] <= 0:
        return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
    if f0.shape[0] == 1:
        return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[
            0]).cpu().numpy(), vuv_vector.cpu().numpy()

    from scipy.interpolate import CubicSpline
    cs = CubicSpline(time_org, f0)
    f0 = cs(time_frame)
    f0 = np.repeat(f0, 2, -1)

    return f0, vuv_vector.cpu().numpy()


def repeat_expand(
    content: Union[torch.Tensor, np.ndarray],
    target_len: int,
    mode: str = "nearest"
):
    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(
        content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]


def save_csv_pitch(pitch, path):
    with open(path, "w", encoding="utf-8") as pitch_file:
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}",
                file=pitch_file,
            )


def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding="utf-8") as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch


def pitch_infer(wav, f0_predictor="crepe"):
    if f0_predictor == "rmvpe":
        funcName = rmvpe_compute_f0_sing
    else:
        funcName = compute_f0_sing
    return funcName(wav, device)
