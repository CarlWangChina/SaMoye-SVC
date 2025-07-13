import pathlib

import numpy as np
import parselmouth


def norm_f0(f0):
    f0 = np.log2(f0)
    return f0


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0)
    if sum(uv) == len(f0):
        f0[uv] = -np.inf
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp


def get_pitch_parselmouth(wav_data, hop_size, audio_sample_rate, interp_uv=True):
    time_step = hop_size / audio_sample_rate
    f0_min = 65
    f0_max = 800

    # noinspection PyArgumentList
    f0 = (
        parselmouth.Sound(wav_data, sampling_frequency=audio_sample_rate)
        .to_pitch_ac(
            time_step=time_step, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        .selected_array["frequency"]
    )
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return time_step, f0, uv


rmvpe = None


def get_pitch_rmvpe(wav_data, hop_size, audio_sample_rate, interp_uv=True):
    global rmvpe
    if rmvpe is None:
        from rmvpe import RMVPE
        rmvpe = RMVPE(pathlib.Path(__file__).parent / 'assets' / 'rmvpe' / 'model.pt')
    f0 = rmvpe.infer_from_audio(wav_data, sample_rate=audio_sample_rate)
    uv = f0 == 0
    f0, uv = interp_f0(f0, uv)

    time_step = hop_size / audio_sample_rate
    length = (wav_data.shape[0] + hop_size - 1) // hop_size
    f0_res = resample_align_curve(f0, 0.01, time_step, length)
    uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
    if not interp_uv:
        f0_res[uv_res] = 0
    return time_step, f0_res, uv_res


def get_pitch(algorithm, wav_data, hop_size, audio_sample_rate, interp_uv=True):
    if algorithm == 'parselmouth':
        return get_pitch_parselmouth(wav_data, hop_size, audio_sample_rate, interp_uv=interp_uv)
    elif algorithm == 'rmvpe':
        return get_pitch_rmvpe(wav_data, hop_size, audio_sample_rate, interp_uv=interp_uv)
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {algorithm}")
