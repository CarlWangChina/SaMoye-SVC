import pandas as pd
import ffmpeg
import time
from mq_service.so_vits_infer_once import (
    svc_infer_pipeline, svc_infer_pipeline_panxin_choice_best, svc_infer_pipeline_panxin_choice_stage2,
    svc_infer_pipeline_panxin_choice_f0shift)
from tqdm import tqdm
from pathlib import Path
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


def infer_time(read_csv):
    curTime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    tmpsaveDir = f"infer_test{curTime}"
    p = Path(f"{tmpsaveDir}/suno")
    sa = Path(f"{tmpsaveDir}/suno_mp3")
    pWav = Path(f"{tmpsaveDir}/sing")
    f0shifts = [0, -12]  # -12 降八度，12 升八度

    df = pd.read_csv(read_csv)

    pWav.mkdir(parents=True, exist_ok=True)
    p.mkdir(parents=True, exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df)):
        spk_id = row.name

        start_time = time.time()
        out_wav, final_save_name = svc_infer_pipeline_panxin_choice_best(
            wave="test_data/suno_test_m_30s.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/suno/{row.name}_{row.minuts}_m_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )
        duration = time.time() - start_time
        wav_dur = getwavdur(final_save_name)
        print(
            f"{final_save_name} infer time: {duration:.2f}s, wav_dur: {wav_dur:.2f}s, rtf: {duration/(len(f0shifts)*wav_dur):.2f}")

        out_wav = svc_infer_pipeline_panxin_choice_best(
            wave="test_data/suno_test_w_30s.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/suno/{row.name}_{row.minuts}_w_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )
        # test_w svc test.
        out_wav = svc_infer_pipeline_panxin_choice_best(
            wave="test_data/test_w.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/sing/{row.name}_{row.minuts}_w_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )
        # test_m svc test.
        out_wav = svc_infer_pipeline_panxin_choice_best(
            wave="test_data/test_m.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/sing/{row.name}_{row.minuts}_m_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )

    # sa.mkdir(parents=True, exist_ok=True)
    # for ori_wav in tqdm(p.iterdir(), total=len(list(p.iterdir()))):
    #     save_mp3 = sa / ori_wav.name
    #     if not save_mp3.with_suffix(".mp3").exists():
    #         wav2mp3(str(ori_wav), save_mp3=str(save_mp3.with_suffix(".mp3")))


def infer_time_auto_f0_mean(read_csv):
    curTime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    tmpsaveDir = f"infer_test{curTime}"
    p = Path(f"{tmpsaveDir}/suno")
    sa = Path(f"{tmpsaveDir}/suno_mp3")
    pWav = Path(f"{tmpsaveDir}/sing")

    df = pd.read_csv(read_csv)

    pWav.mkdir(parents=True, exist_ok=True)
    p.mkdir(parents=True, exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df)):
        spk_id = row.name

        start_time = time.time()
        out_wav, final_save_name = svc_infer_pipeline_panxin_choice_stage2(
            wave="test_data/suno_test_m_30s.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/suno/{row.name}_{row.minuts}_m_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
        )
        duration = time.time() - start_time
        wav_dur = getwavdur(final_save_name)
        print(
            f"{final_save_name} infer time: {duration:.2f}s, wav_dur: {wav_dur:.2f}s, rtf: {duration/(wav_dur):.2f}")

        out_wav, _ = svc_infer_pipeline_panxin_choice_stage2(
            wave="test_data/suno_test_w_30s.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/suno/{row.name}_{row.minuts}_w_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
        )
        # test_w svc test.
        out_wav, _ = svc_infer_pipeline_panxin_choice_stage2(
            wave="test_data/test_w.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/sing/{row.name}_{row.minuts}_w_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
        )
        # test_m svc test.
        out_wav, _ = svc_infer_pipeline_panxin_choice_stage2(
            wave="test_data/test_m.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/sing/{row.name}_{row.minuts}_m_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
        )


def infer_time_test(read_csv):
    curTime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    tmpsaveDir = f"infer_test{curTime}"
    p = Path(f"{tmpsaveDir}/suno")
    sa = Path(f"{tmpsaveDir}/suno_mp3")
    pWav = Path(f"{tmpsaveDir}/sing")
    f0shifts = [0, -12]  # -12 降八度，12 升八度

    df = pd.read_csv(read_csv)

    pWav.mkdir(parents=True, exist_ok=True)
    p.mkdir(parents=True, exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df)):
        spk_id = row.name

        start_time = time.time()
        out_wav, final_save_name = svc_infer_pipeline_panxin_choice_f0shift(
            wave="test_data/suno_test_m_30s.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/suno/{row.name}_{row.minuts}_m_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )
        duration = time.time() - start_time
        # wav_dur = getwavdur(final_save_name)
        # print(
        #     f"{final_save_name} infer time: {duration:.2f}s, wav_dur: {wav_dur:.2f}s, rtf: {duration/(len(f0shifts)*wav_dur):.2f}")

        # suno_test_w_30s
        out_wav, final_save_name = svc_infer_pipeline_panxin_choice_f0shift(
            wave="test_data/suno_test_w_30s.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/suno/{row.name}_{row.minuts}_w_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )
        # test_w svc test.
        out_wav, _ = svc_infer_pipeline_panxin_choice_f0shift(
            wave="test_data/test_w.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/sing/{row.name}_{row.minuts}_w_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )
        # test_m svc test.
        out_wav, _ = svc_infer_pipeline_panxin_choice_f0shift(
            wave="test_data/test_m.wav",
            spk_id=spk_id,
            save_name=f"{tmpsaveDir}/sing/{row.name}_{row.minuts}_m_",
            model_path=f"export/{row.name}_{row.minuts}.pth",
            shifts=f0shifts,
        )


def infer_test_one_dir(read_csv, searchDir):
    curTime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    tmpsaveDir = f"infer_test_one_dir_{curTime}_{searchDir}"
    saveDir = Path(tmpsaveDir)
    f0shifts = [0, -12]  # -12 降八度，12 升八度

    df = pd.read_csv(read_csv)

    saveDir.mkdir(parents=True, exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df)):
        spk_id = row.name

        start_time = time.time()
        for wavFile in Path(searchDir).glob("*.wav"):
            out_wav, final_save_name = svc_infer_pipeline_panxin_choice_f0shift(
                wave=wavFile,
                spk_id=spk_id,
                save_name=f"{str(saveDir)}/{row.name}_{row.minuts}_m_{wavFile.stem}",
                model_path=f"export/{row.name}_{row.minuts}.pth",
                shifts=f0shifts,
            )
            duration = time.time() - start_time
            wav_dur = getwavdur(final_save_name)
            print(
                f"{final_save_name} infer time: {duration:.2f}s, wav_dur: {wav_dur:.2f}s, rtf: {duration/(len(f0shifts)*wav_dur):.2f}")


if __name__ == "__main__":
    # read_csv = "export_model_4spks.csv"
    # read_csv = "export_model_4spksTest.csv"
    read_csv = "export_model_fewspks.csv"

    # F0 normalize and loss with tgtwav F0
    # infer_time(read_csv)

    # F0 mean less than target F0 mean
    # infer_time_auto_f0_mean(read_csv)

    # F0 shift choices
    # infer_time_test(read_csv)

    infer_test_one_dir(read_csv, "test_data/vocal_rmvpe")
