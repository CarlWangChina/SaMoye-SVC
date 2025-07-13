############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger

logger = get_logger(__name__)

###########################################################################################

"""
    1. 读取 midi 文件: 
    测试时：/home/john/CaichongSinger/data/origin_midi/ck6w                 
    最终：/data/john/CaichongSinger/sheetsage_data
    2. 读取歌词时间戳文件:
    测试时：/home/john/CaichongSinger/data/fixed_json/ck6w
    最终：/data/john/CaichongSinger/mfa_data
    3. 提取 bpm 其从/nfs/datasets-mp3/ 获取
    4. 根据 bpm 获取32分音符的时长，量化 midi 和 歌词时间戳
    5. 对量化后的 midi 和 歌词时间戳进行对齐，确保 多个单独音符能够 正确对应一个 歌词，不能单个音符对应多个歌词，如果有，将其拆分为多个音符
    6. 保存对齐后的ds文件
    7. （可选）生成diffsinger歌声，测试对应的ds文件
"""

import argparse
import pandas as pd
import json
from singer.quantum_process import quan_json,quan_ds,qj_midi_lyric_align
from singer.ds_process.midi_process import estimate_tempo, find_nearest_quantized_tempo
from singer.models import run_diffsinger_command_test
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
from pretty_midi import PrettyMIDI


def get_bpm(args):
    song_id, mp3_file_path = args
    try:
        mp3_tempo = estimate_tempo(mp3_file_path)
        tempo = find_nearest_quantized_tempo(mp3_tempo)
        return (song_id, tempo)
    except ValueError:
        return song_id, None


def align_midi_lyric(args):
    midi_path, lyric_path, ds_path, bpm = args
    try:
        # 读取midi文件
        midi = PrettyMIDI(str(midi_path))
        # 读取歌词时间戳文件
        ds_data = []
        with open(lyric_path, "r", encoding = "utf-8") as f:
            for line in f:
                ds_data.append(json.loads(line))
        # 量化lyric
        quan_ds_data = quan_ds(ds_data, bpm)
        qj_midi_lyric_align(midi, quan_ds_data, ds_path, bpm)
        return True
    except Exception as e:
        # logger.error(f"Error loading midi and lyric file: {e}")
        return False

def generate_vocal(args):
    ds_input_path, ds_output_path, vocal_path, cuda = args
    try:
        run_diffsinger_command_test(ds_input_path=ds_input_path, ds_output_path= ds_output_path, vocal_path=vocal_path,spk="female1", cuda=cuda)
        return True
    except Exception as e:
        # logger.error(f"Error generating vocal file: {e}")
        return False

def main(args):
    midi_dir, lyric_dir, bpm_csv, output_dir, mp3_dir, vocal_dir = (
        Path(args.midi_dir),
        Path(args.lyric_dir),
        Path(args.bpm_csv),
        Path(args.output_dir),
        Path(args.mp3_dir),
        Path(args.vocal_dir) if args.vocal_dir is not None else None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取bpm ，如果没有则需要estimate_tempo，然后保存到csv文件中
    assert bpm_csv is not None, "bpm_csv should be provided."
    if os.path.exists(bpm_csv):
        songid_bpm_pd = pd.read_csv(bpm_csv)
        assert (
            "song_id" in songid_bpm_pd.columns and "bpm" in songid_bpm_pd.columns
        ), "bpm_csv should have columns 'song_id' and 'bpm'."
        logger.info(f"Successfully loaded bpm from {bpm_csv}.")
    else:
        # 整理 song_id 和 mp3 文件
        song_id_mp3_path = []
        for midi_file in lyric_dir.glob("*.ds"):
            song_id = midi_file.stem
            # qq 开头：
            if song_id.startswith("qq"):
                mp3_file = Path(mp3_dir) / "dyqy-fix" / f"{song_id}_src.mp3"
            elif song_id.startswith("65") and len(song_id) > 10:
                mp3_file = Path(mp3_dir) / "ali/65" / f"{song_id}_src.mp3"
            else:
                mp3_file = Path(mp3_dir) / "cb" / f"{song_id}_src.mp3"
            if not mp3_file.exists():
                logger.error(f"MP3 file {mp3_file} does not exist, skipping.")
                continue
            song_id_mp3_path.append((song_id, mp3_file))
        song_id_bpm = process_map(
            get_bpm, song_id_mp3_path, max_workers=50, chunksize=2
        )
        success_song_id_bpm = [x for x in song_id_bpm if x[1] is not None]
        songid_bpm_pd = pd.DataFrame(success_song_id_bpm, columns=["song_id", "bpm"])
        songid_bpm_pd.to_csv(bpm_csv, index=False)
        logger.info(f"Successfully saved bpm to {bpm_csv}.")

    # 整理midi和lyric文件
    if False:
        midi_lyric_file_paths = []
        logger.info(f"Loading midi and lyric files, csv size: {songid_bpm_pd.shape[0]}.")
        for song_id, bpm in songid_bpm_pd.values:
            midi_path = midi_dir / f"{song_id}_src.mp3.mid"
            lyric_path = lyric_dir / f"{song_id}.ds"
            ds_path = output_dir / f"{song_id}.ds"
            if not midi_path.exists() or not lyric_path.exists():
                # logger.error(
                #     f"midi file {midi_path} or lyric file {lyric_path} does not exist, skipping."
                # )
                continue
            midi_lyric_file_paths.append((midi_path, lyric_path, ds_path, int(bpm)))
        logger.info(f"Successfully loaded {len(midi_lyric_file_paths)} midi and lyric files.")
        flags = process_map(align_midi_lyric, midi_lyric_file_paths, max_workers=50, chunksize=2)
        logger.info(f"Successfully aligned {sum([1 for x in flags if x])} midi and lyric files. error nums: {sum([1 for x in flags if not x])}")

    # 生成diffsinger歌声
    song_id_list = [1686719,102015,1686969,2001980]
    cuda = iter([0,1,2,3,4,5,6,7])
    if vocal_dir is not None:
        if song_id_list:
            logger.info(f"Generating vocal files for {len(song_id_list)} songs.")
            ds_path_vocal_dir = [
                (output_dir / f"{song_id}.ds", vocal_dir/"var_ds"/f"{song_id}.ds", vocal_dir, next(cuda))
                for song_id in song_id_list
            ]
            process_map(
                generate_vocal,
                ds_path_vocal_dir,
                max_workers=cpu_count(),
                chunksize=1,
            )
            logger.info(f"Successfully generated vocal files.")

if __name__ == "__main__":
    # python script/task5_16_midi_lyric_align/midi_lyric_align.py
    arg_parser = argparse.ArgumentParser()
    # "/data/john/CaichongSinger/data/origin_midi/ck6w"
    arg_parser.add_argument(
        "--midi_dir",
        default="/data/john/CaichongSinger/sheetsage_data",
        type=str,
        help="The directory containing the midi files.",
    )
    # 歌词时间戳文件的位置 /data/john/CaichongSinger/data/fixed_json/ck6w
    arg_parser.add_argument(
        "--lyric_dir",
        default="/data/john/CaichongSinger/mfa_data",
        type=str,
        help="The directory containing the lyric files.",
    )
    # 　bpm 使用的csv文件存放位置 /data/john/CaichongSinger/bpm_data/ck6w_bpm.csv
    arg_parser.add_argument(
        "--bpm_csv",
        default="/data/john/CaichongSinger/bpm_data/4w_bpm.csv",
        type=str,
        help="The csv file containing the bpm of the midi files.",
    )
    # 保存对齐后的ds文件的位置 /data/john/CaichongSinger/align_data
    arg_parser.add_argument(
        "--output_dir",
        default="/data/john/CaichongSinger/align_data/4w",
        type=str,
        help="The directory to save the aligned ds files.",
    )
    # origin_mp3的位置 /nfs/datasets-mp3/cb/
    # /nfs/datasets-mp3/ali/65 开头为65的mp3文件
    # /nfs/datasets-mp3/dyqy 开头为qq的数据
    # /nfs/datasets-mp3/cb 普通的长度小于10的数据
    arg_parser.add_argument(
        "--mp3_dir",
        default="/nfs/datasets-mp3/",
        type=str,
        help="The directory of the origin mp3 files.",
    )
    # 生成diffsinger歌声的位置 /data/john/CaichongSinger/ds_data/ck6w 为None时不生成
    arg_parser.add_argument(
        "--vocal_dir",
        default="/home/john/CaichongSinger/data/vocal/4w",
        type=str,
        help="The directory to save the vocal files generated by diffsinger.",
    )
    args = arg_parser.parse_args()

    main(args)
