from pathlib import Path
import torch
import numpy as np

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer
from feature_retrieval import (
    IRetrieval,
    DummyRetrieval,
)
from .whisper_infer import whisper_infer
from .hubert_infer import hubert_infer
from .pitch_infer import pitch_infer
from utils import get_logger, get_hparams
import librosa
from vad.utils import init_jit_model, get_speech_timestamps

service_hparams = get_hparams()
device = service_hparams["device"]

hp = OmegaConf.load("configs/base.yaml")
logger = get_logger(__name__)

model = SynthesizerInfer(
    hp.data.filter_length // 2 + 1, hp.data.segment_size // hp.data.hop_length, hp
)


vad_model = init_jit_model("vad/assets/silero_vad.jit")
vad_model.eval()


def get_speaker_name_from_path(speaker_path: Path) -> str:
    suffixes = "".join(speaker_path.suffixes)
    filename = speaker_path.name
    return filename.rstrip(suffixes)


def load_svc_model(checkpoint_path, model):
    checkpoint_dict = torch.load(checkpoint_path)
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.error("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


model = load_svc_model(
    service_hparams["so_vits_model_path"]["model_path"], model)

model.eval()
model.to(service_hparams["device"])


def svc_infer(model, retrieval: IRetrieval, spk, pit, ppg, vec, hp, device):
    len_pit = pit.size()[0]
    len_vec = vec.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_vec)
    len_min = min(len_min, len_ppg)
    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500  # 25 S
        out_index = 0
        out_audio = []

        while out_index < all_frame:

            if out_index == 0:  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size

            if out_index + out_chunk + hop_frame > all_frame:  # end frame
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame * hop_size

            sub_ppg = retrieval.retriv_whisper(ppg[cut_s:cut_e, :])
            sub_vec = retrieval.retriv_hubert(vec[cut_s:cut_e, :])
            sub_ppg = sub_ppg.unsqueeze(0).to(device)
            sub_vec = sub_vec.unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size: cut_e * hop_size].to(device)
            sub_out = model.inference(
                sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        out_audio = np.asarray(out_audio)
    return out_audio


def svc_infer_pipeline(
    wave: str,
    spk_id: str,
    save_name: str = "svc_out.wav",
    shift: int = 0,
):
    """_summary_

    Args:
        wav (str): Path of raw audio.
        spk (str): Path of speaker.
        pgg (str): Path of content vector.
        vec (str): Path of hubert vector.
        pit (str): Path of pitch csv file.
        shift (int, optional): Pitch shift key.. Defaults to 0.
        enable_retrieval (bool, optional): Enable index feature retrieval. Defaults to True.
        etrieval_index_prefix (str, optional): _description_. Defaults to "".
        retrieval_ratio (float, optional): ratio of feature retrieval effect. Must be in range 0..1. Defaults to 0.5.
        n_retrieval_vectors (int, optional): get n nearest vectors from retrieval index. Works stably in range 1..3. Defaults to 3.
        hubert_index_path (str, optional): path to hubert index file. Defaults to "data_svc/indexes/speaker.../%prefix%hubert.index".

    """
    # TODO 优化音频的读取逻辑，现在的音频被反复读取
    # logger.info(f"whisper infer start ...")
    ppg = whisper_infer(wave)
    # logger.info(f"whisper infer done...")
    # logger.info(f"hubert infer start ...")

    vec = hubert_infer(wave)
    # logger.info(f"hubert infer done ...")
    # logger.info(f"pitch infer start ...")

    pit = pitch_infer(wave)
    # logger.info(f"pitch infer start ...")

    spk = np.load(f"data_svc/singer/{spk_id}.spk.npy")
    spk = torch.FloatTensor(spk)

    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)
    # ppg = torch.zeros_like(ppg)

    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)
    # vec = torch.zeros_like(vec)

    logger.info(f"pitch shift: {shift}")
    if shift != 0:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        logger.info(
            f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}"
        )
        pit = pit * (2 ** (shift / 12))
    pit = torch.FloatTensor(pit)
    retrieval = DummyRetrieval()
    out_audio = svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device)

    new_wav, _ = post_process(ref_wave_path=wave, svc_wave=out_audio)
    write(save_name, hp.data.sampling_rate, new_wav)
    return out_audio


def post_process(ref_wave_path: str, svc_wave: str):
    """_summary_

    Args:
        ref_wave_path (str): Path of ref audio.
        svc_wave_path (str): Path of svc audio.
    Returns:
        _type_: _description_
    """
    ref_wave, _ = librosa.load(ref_wave_path, sr=16000)
    tmp_wave = torch.from_numpy(ref_wave).squeeze(0)
    tag_wave = get_speech_timestamps(
        tmp_wave, vad_model, threshold=0.2, sampling_rate=16000
    )

    ref_wave[:] = 0
    for tag in tag_wave:
        ref_wave[tag["start"]: tag["end"]] = 1

    ref_wave = np.repeat(ref_wave, 2, -1)

    min_len = min(len(ref_wave), len(svc_wave))
    ref_wave = ref_wave[:min_len]
    svc_wave = svc_wave[:min_len]
    svc_wave[ref_wave == 0] = 0
    return svc_wave, 32000
