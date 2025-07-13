import torch
import librosa

from hubert import hubert_model
from utils import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def load_model(path, device):
    model = hubert_model.hubert_soft(path)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, vecPath, device):
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
            # print(vec.shape)   # [length, dim=256] hop=320
            vec_a.extend(vec)
    # np.save(vecPath, vec_a)
    return vec_a


hubert_model = load_model(
    "pretrainModels/hubert_pretrain/hubert-soft-0d54a1f4.pt", device)


def hubert_infer(wav_path: str):
    return pred_vec(hubert_model, wav_path, "vec.vgg.npy", device)
