import json
from pathlib import Path
import sys
import os

from pypinyin import lazy_pinyin

from .SOFA_module.infer import sofa_main as sofa_infer

# def main(
#     ckpt,
#     folder,
#     mode,
#     g2p,
#     ap_detector,
#     in_format,
#     out_formats,
#     save_confidence,
#     **kwargs,
# )
# CUDA_VISIBLE_DEVICES=0 python infer.py \
# -f /home/john/SOFA/3-10 -of trans --g2p Dictionary \
# -c ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt -d dictionary/opencpop-extension.txt


def ds_2_lab(ds_path: Path, lab_dir: Path):
    with open(ds_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    word_strings = ""
    for sequence in ds:
        word_seq = sequence["word_seq"].split()
        word_string = "".join([w for w in word_seq if w not in ("AP", "SP")])
        word_strings += word_string

    pinyin_seq = lazy_pinyin(word_strings)
    pinyin_string = " ".join(pinyin_seq)
    with open(lab_dir / (ds_path.stem + ".lab"), "w", encoding="utf-8") as f:
        f.write(pinyin_string)


def change_vocal2ds(
    folder,
    ckpt="pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt",
    dictionary="/home/john/DuiniutanqinSinger/singer/dictionaries/opencpop-extension.txt",
    cuda=0,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    # ckpt_folder = Path("/home/john/DuiniutanqinSinger/singer/postprocess/SOFA/ckpt/")
    # ckpt_folder 为本文件绝对路径加SOFA/ckpt/
    # /home/john/DuiniutanqinSinger/singer/postprocess/SOFA_module/ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt
    ckpt_folder = Path(__file__).resolve().parent / "SOFA_module/ckpt/"
    ckpt_path = ckpt_folder / ckpt

    sofa_infer(ckpt=ckpt_path, folder=folder, dictionary=dictionary)


def csv2ds(csv_path, wav_dir):
    # python convert_ds.py csv2ds /home/john/DuiniutanqinSinger/data/combined/temp/transcriptions/transcriptions.csv /home/john/DuiniutanqinSinger/data/combined/temp --pe rmvpe
    import subprocess

    # 获取本文件的目录绝对路径
    file_dir = Path(__file__).resolve().parent
    covert_ds_path = file_dir / "convert_ds.py"
    subprocess.run(
        ["python", covert_ds_path, "csv2ds", csv_path, wav_dir, "--pe", "rmvpe"]
    )


def vocal_to_ds(
    ds_path: Path,
    lab_dir: Path,
    ckpt="pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt",
    dictionary="/home/john/DuiniutanqinSinger/singer/dictionaries/opencpop-extension.txt",
    cuda=0,
):
    ds_2_lab(ds_path, lab_dir)
    change_vocal2ds(lab_dir, cuda=cuda)
    csv_path = lab_dir / "transcriptions/transcriptions.csv"
    csv2ds(csv_path, lab_dir)


if __name__ == "__main__":
    ds_path = Path("/home/john/DuiniutanqinSinger/data/ds/temp/1686672.ds")
    lab_dir = Path("/home/john/DuiniutanqinSinger/data/combined/temp/1686672/")

    # ds_2_lab(ds_path, lab_dir)

    # change_vocal2ds(folder=lab_dir)

    # csv_path = lab_dir / "transcriptions/transcriptions.csv"
    # csv2ds(csv_path, lab_dir)
    vocal_to_ds(ds_path, lab_dir)
