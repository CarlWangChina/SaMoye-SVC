import pandas as pd
import ffmpeg
import os
from mq_service.so_vits_infer_once import svc_infer_pipeline
from svc_export import export
from tqdm import tqdm
from pathlib import Path


def wav2mp3(ori_wav: str, save_mp3: str):
    try:
        stream = ffmpeg.input(ori_wav)
        stream = ffmpeg.output(stream, save_mp3)
        ffmpeg.run(stream)
    except Exception as e:
        print("transfor error ori_wav={} error msg: {}".format(ori_wav, e))


def export_models(read_csv):
    df = pd.read_csv(read_csv)

    for row in df.itertuples():
        ckpt = Path(row.path)
        if not ckpt.exists():
            print(f"ckpt: {row.path} not exists")

        config = Path(f"configs/{row.name}.yaml")
        if not config.exists():
            print(f"configs: {row.name} not exists")
        print(f"Exporting {row.name}...")
        if not os.path.exists(f"export/{row.name}_{row.minuts}.pth"):
            export(config=str(config), ckpt=str(ckpt),
                   save_name=f"export/{row.name}_{row.minuts}.pth",)
        else:
            print(f"{row.name}_{row.minuts}.pth already exists, skip.")
        print(f"Exporting {row.name} done.")


def infer_time(read_csv):
    df = pd.read_csv(read_csv)
    p = Path("export")

    for row in tqdm(df.itertuples()):
        # spk = Path(f"data_svc/singer/{i.stem}.spk.npy")
        # spk_id = i.stem if spk.exists() else "zihao_1_min"
        spk_id = row.name

        out_wav = svc_infer_pipeline(
            wave="test_data/suno_test_m_30s.wav",
            spk_id=spk_id,
            save_name=f"infer_test/suno/{row.name}_{row.minuts}_m.wav",
            model_path=f"export/{row.name}_{row.minuts}.pth",
        )
        out_wav = svc_infer_pipeline(
            wave="test_data/suno_test_w_30s.wav",
            spk_id=spk_id,
            save_name=f"infer_test/suno/{row.name}_{row.minuts}_w.wav",
            model_path=f"export/{row.name}_{row.minuts}.pth",
        )

    p = Path("infer_test/suno")
    for ori_wav in tqdm(p.iterdir()):
        save_mp3 = ori_wav.parent.parent / "suno_mp3" / ori_wav.name
        wav2mp3(str(ori_wav), save_mp3=str(save_mp3.with_suffix(".mp3")))


if __name__ == "__main__":
    read_csv = "export_model_fewspks.csv"

    export_models(read_csv)
    # infer_time(read_csv)
