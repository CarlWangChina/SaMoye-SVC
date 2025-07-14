import pandas as pd
import ffmpeg
import os
import time
from mq_service.so_vits_infer_once import (
    svc_infer_pipeline, svc_infer_pipeline_panxin_choice_best, svc_infer_pipeline_panxin_choice_stage2,
    svc_infer_pipeline_panxin_choice_f0shift,
    svc_infer_pipeline_panxin_experiment)
from tqdm import tqdm
from pathlib import (Path, PosixPath)
from pydub import AudioSegment


def wav2mp3(ori_wav: str, save_mp3: str):
    try:
        stream = ffmpeg.input(ori_wav)
        stream = ffmpeg.output(stream, save_mp3)
        ffmpeg.run(stream)
    except Exception as e:
        print("transfor error ori_wav={} error msg: {}".format(ori_wav, e))


def getwavdur(fileRead):
    """ Get the duration of a wav file in seconds.
    """
    sound = AudioSegment.from_wav(fileRead)
    return sound.duration_seconds


def infer_one_Kind(readDir: PosixPath, kind: str, tmpsaveDir: str, row, f0shifts):
    for targetSpk in (readDir / f"speaker/{kind}").glob("*.wav"):
        for targetWav in (readDir / "content/h_m_l").glob("*.wav"):
            out_wav, final_save_name = svc_infer_pipeline_panxin_experiment(
                wave=targetWav,
                spk_id=targetSpk.stem,
                save_name=f"{tmpsaveDir}/{kind}/{row.name}_{row.minuts}_",
                model_path=f"export/{row.name}_{row.minuts}.pth",
                shifts=f0shifts,
                kind=kind)


def infer_time(read_csv: str, readDir: str):
    curTime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    tmpsaveDir = f"ExperimentResult_{curTime}"
    f0shifts = [0]  # -12 降八度，12 升八度
    readDir = Path(readDir)

    df = pd.read_csv(read_csv)

    os.makedirs(tmpsaveDir + "/seen", exist_ok=True)
    os.makedirs(tmpsaveDir + "/unseen", exist_ok=True)

    for row in tqdm(df.itertuples(), total=len(df)):
        infer_one_Kind(readDir, "seen", tmpsaveDir, row, f0shifts)
        infer_one_Kind(readDir, "unseen", tmpsaveDir, row, f0shifts)


if __name__ == "__main__":
    read_csv = "export_model.csv"
    readDir = "yongshengTestData"

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    infer_time(read_csv, readDir)
