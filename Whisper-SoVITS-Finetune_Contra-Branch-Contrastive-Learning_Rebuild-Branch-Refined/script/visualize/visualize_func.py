# 1. 将生成数据和降维可视化的代码一起使用，只处理前20个说话人的相应数据
""" 引入root目录 """
from pathlib import Path

# 获取当前工作目录
current_dir = Path.cwd()
data_dir = Path("/home/john/svc/data")
root_dir = current_dir.parent.parent
import sys

sys.path.append(str(root_dir))
"""""" """""" """""" """""" """"""

not_include_in_speaker_dir = [
    "nothing in it"
]  # VCTK", "Aishell, "DAMP", "popcs", "peng", "DSD"
# unseen_speakers = ["DAMP", "popcs", "peng", "DSD"]
unseen_speakers = ["unseen"]


def get_spk_wavs(dataset_path, output_path):
    wav_files = []
    embedding_files = []
    speaker_names = []
    seen_speaker_names = []
    unseen_speakers_names = []
    for speaker_dir in Path(dataset_path).iterdir():

        if any([x in speaker_dir.name for x in not_include_in_speaker_dir]):
            continue
        # seen 和 unseen 各自最多加载100个说话人
        if any([x in speaker_dir.name for x in unseen_speakers]):
            if len(set(unseen_speakers_names)) >= 200:
                continue
            unseen_speakers_names.append(speaker_dir.name)
        else:
            # if len(set(seen_speaker_names)) >= 500:
            #     continue
            seen_speaker_names.append(speaker_dir.name)
        # 每个说话人加载40个嵌入
        wavs_num = 2
        if speaker_dir.is_dir():
            # if len(list(speaker_dir.glob("*.wav"))) < wavs_num:
            #     continue
            output_speaker_dir = Path(output_path) / speaker_dir.name
            output_speaker_dir.mkdir(parents=True, exist_ok=True)
            for i, wav_file in enumerate(speaker_dir.glob("*.wav")):
                if i >= wavs_num:
                    break
                embedding_file = output_speaker_dir / f"{wav_file.stem}.spk.npy"
                if not embedding_file.exists():  # 如果嵌入文件不存在，则添加wav文件
                    wav_files.append(str(wav_file))
                embedding_files.append(str(embedding_file))
                speaker_names.append(speaker_dir.name)

    return wav_files, embedding_files, speaker_names


# 1.1. 生成数据
from models.whisper_vits_svc_spk_change.prepare import preprocess_speaker
import torch
import os


class Args:
    use_cuda = True


args = Args()


def generate_data(model_path=None):
    ckpt_name = model_path.stem if model_path else "default"
    dataset_path = data_dir / "processed_data/waves-16k"
    output_path = data_dir / f"processed_data/speaker_new/speakers_{ckpt_name}"
    thread_count = 8
    speaker_encoder_helper = preprocess_speaker.SpkEncoderHelper(
        str(root_dir / "models/whisper_vits_svc_spk_change")
    )
    if model_path:
        # 加载权重到模型
        speaker_encoder_helper.speaker_encoder.load_checkpoint_spk_change(
            model_path, eval=True, use_cuda=args.use_cuda
        )

    (wav_files, embedding_files, speaker_names) = get_spk_wavs(
        str(dataset_path), str(output_path)
    )
    if thread_count == 0:
        process_num = os.cpu_count()
    else:
        process_num = thread_count

    if wav_files:
        preprocess_speaker.extract_speaker_embeddings(
            wav_files,
            str(dataset_path),
            str(output_path),
            args,
            speaker_encoder_helper.speaker_encoder_ap,
            speaker_encoder_helper.speaker_encoder,
            process_num,
        )
    else:
        print("All embeddings are already generated. Skip this step.")
    return wav_files, embedding_files, speaker_names, ckpt_name
