############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger_save_log

logger = get_logger_save_log(__name__, root_path / "log" / "get_quantum_midi.log")

###########################################################################################

"""
    任务：遍历一个 csv 将 json 文件转换为 midi 文件
    
    1. 读取 csv 文件：
    路径：data/quantum/500ALL_tempo.xlsx
    内容：song_id, mp3_tempo, mp3_quantized_tempo
    
    2. 根据 song_id 获取 json 文件：
    路径：data/fixed_json/ck6w 
    文件名：{song_id}_fixed.json
    
    3. 将 json 文件转换为 midi 文件：
    3.1 先转为 ds 文件
    3.2 再转为 量化 midi 文件
    
    4. 保存到data/quantum/midi中
    加了两个小节的保存到data/quantum/midi_add_2_bars中
"""
import pandas as pd
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map

from singer.quantum_process import json_to_quantum_ds
from singer.ds_process import ds_to_quantum_midi
from singer.ds_process.midi_process import estimate_tempo, find_nearest_quantized_tempo

global song_id_file_path_stem


def get_bpm(song_id):
    mp3_dir = Path("/nfs/datasets-mp3/cb/")
    mp3_file_path = mp3_dir / f"{song_id}_src.mp3"
    if not mp3_file_path.exists():
        logger.error(f"MP3 file {mp3_file_path} does not exist, skipping.")
        raise ValueError(f"MP3 file {mp3_file_path} does not exist, skipping.")
    mp3_tempo = estimate_tempo(mp3_file_path)
    tempo = find_nearest_quantized_tempo(mp3_tempo)
    return tempo


def process_song_id(song_id):
    try:
        return song_id, get_bpm(song_id)
    except ValueError:
        return song_id, None


def process_song(row):
    global song_id_file_path_stem
    song_id = int(row["song_id"])
    # 如果mp3_quantized_tempo不存在，重新计算
    mp3_quantized_tempo = row["mp3_quantized_tempo"]

    # 读取文件路径
    json_file_path = root_path / "data/fixed_json/ck6w" / f"{song_id}_fixed.json"
    midi_file_path = root_path / "data/origin_midi/ck6w" / f"{song_id}_src.mp3_5b.mid"

    # 　生成文件路径
    quantum_ds_file_path = (
        root_path
        / "data/quantum/ds"
        / song_id_file_path_stem
        / f"{song_id}_quantized.ds"
    )
    quantum_midi_file_path = (
        root_path
        / "data/quantum/midi"
        / song_id_file_path_stem
        / f"{song_id}_quantized.mid"
    )
    quantum_midi_add_2_bars_file_path = (
        root_path
        / "data/quantum/midi_add_2_bars"
        / song_id_file_path_stem
        / f"{song_id}_quantized_add_2_bars.mid"
    )
    quantum_midi_with_chord_drum_file_path = (
        root_path
        / "data/quantum/midi_with_chord_drum"
        / song_id_file_path_stem
        / f"{song_id}_quantized_with_chord_drum.mid"
    )

    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file_path.parent.mkdir(parents=True, exist_ok=True)
    quantum_ds_file_path.parent.mkdir(parents=True, exist_ok=True)
    quantum_midi_file_path.parent.mkdir(parents=True, exist_ok=True)
    quantum_midi_add_2_bars_file_path.parent.mkdir(parents=True, exist_ok=True)
    quantum_midi_with_chord_drum_file_path.parent.mkdir(parents=True, exist_ok=True)

    json_to_quantum_ds(
        song_id,
        mp3_quantized_tempo,
        json_file_path,
        midi_file_path,
        quantum_ds_file_path,
    )

    ds_to_quantum_midi(
        song_id,
        mp3_quantized_tempo,
        quantum_ds_file_path,
        midi_file_path,
        quantum_midi_file_path,
        quantum_midi_add_2_bars_file_path,
        quantum_midi_with_chord_drum_file_path,
    )

    logger.info(f"已完成 {song_id} 的转换")


def main(tempo_file_name: str = "500ALL_tempo.xlsx"):
    global song_id_file_path_stem
    # 读取 csv 文件
    song_id_file_path = root_path / "data/quantum" / tempo_file_name
    if song_id_file_path.suffix == ".xlsx":
        df = pd.read_excel(song_id_file_path)
    elif song_id_file_path.suffix == ".csv":
        df = pd.read_csv(song_id_file_path)
    else:
        raise ValueError(f"Unsupported file type: {song_id_file_path.suffix}")
    song_id_file_path_stem = str(song_id_file_path.stem)

    # 不存在mp3_quantized_tempo，新建一个表格
    if "mp3_quantized_tempo" not in df.columns:
        song_ids = df["song_id"].tolist()
        results = process_map(process_song_id, song_ids, max_workers=cpu_count())
        # 过滤出成功的计算结果，并创建一个新的DataFrame
        successful_results = [result for result in results if result[1] is not None]
        df = pd.DataFrame(
            successful_results, columns=["song_id", "mp3_quantized_tempo"]
        )

        # 保存结果到Excel文件
        excel_path = root_path / "data/quantum" / f"{song_id_file_path_stem}_tempo.xlsx"
        df.to_excel(excel_path, index=False)

        logger.info(f"已完成 mp3_quantized_tempo 的计算")

    return 0
    song_id_tempo_list = df[["song_id", "mp3_quantized_tempo"]].to_dict(
        orient="records"
    )
    process_map(process_song, song_id_tempo_list, max_workers=cpu_count())


if __name__ == "__main__":
    # temp_file_name = "dis_2042.csv"
    # temp_file_name = "dis_2042_tempo.xlsx"
    # temp_file_name = "500ALL_tempo.xlsx"
    temp_file_name = "with_singer_68989.csv"
    main(temp_file_name)
