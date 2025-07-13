import torch
import os
import soundfile as sf
from inference_utils import *
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert, rmvpe_pitch


'''
设备设置
'''
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# if use half precision for model (only consider cuda)
is_half = torch.cuda.is_available()

'''
模型ckpt路径
'''
sovits_path = "exp_dir/logs_sovits/G_110000.pth"
cnhubert_base_path = "pretrained_models/chinese-hubert-base"
rmvpe_path = 'pretrained_models/rmvpe.pt'
hps = utils.get_hparams()

'''
Load models
'''
cnhubert.cnhubert_base_path = cnhubert_base_path
rmvpe = rmvpe_pitch.get_f0_predictor(rmvpe_path)
ssl_model = cnhubert.get_cnhubert_model('pretrained_models/chinese-hubert-base')

sovits = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
)
del sovits.enc_q


if is_half:
    ssl_model = ssl_model.half().to(device)
    sovits = sovits.half().to(device)
else:
    ssl_model = ssl_model.to(device)
    sovits = sovits.to(device)
    sovits.eval()


def inference(origin_wav_path: str, ref_wav_path: str, ssl_model, f0_model, sovits_ckpt_path):
    ckpt = torch.load(sovits_ckpt_path, map_location=device)
    if 'model' in ckpt.keys():
        weight = ckpt['model']
    else:
        weight = ckpt['weight']
    
    unload_params = sovits.load_state_dict(weight, strict=False)
    print(unload_params)
    sovits.eval()

    audio, sr = get_svc_wav(
        origin_wav_path=origin_wav_path,
        ref_wav_path=ref_wav_path,
        ssl_model=ssl_model,
        f0_model=f0_model,
        svc_model=sovits,
        hps=hps
    )

    audio = audio.astype(np.int16)
    # 输出路径
    steps = sovits_ckpt_path.split('/')[-1].split('.')[0]
    output_pth = f'test_qinghuaci/{steps}.wav'
    sf.write(output_pth, audio, sr)
    return audio, sr


if __name__ == '__main__':
    ckpts = ['pretrained_models/s2G488k_modified_ml.pth']
    
    original_audio_pth = 'example_refence_audio/qinghuaci.wav'
    ref_audio_pth = 'example_refence_audio/hutao.wav'
    
    ckpt_root = 'sovits'
    ckpt_paths = [os.path.join(ckpt_root, c) for c in os.listdir(ckpt_root) if c.startswith('G')]
    
    ckpts.extend(ckpt_paths)

    for ckpt in ckpts:
        inference(original_audio_pth, ref_audio_pth, ssl_model, rmvpe, ckpt)
