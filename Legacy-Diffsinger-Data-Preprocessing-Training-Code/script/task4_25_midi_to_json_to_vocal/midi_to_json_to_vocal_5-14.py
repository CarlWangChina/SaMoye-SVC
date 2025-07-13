############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path
import logging

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(root_path / "log" / "ds_change_midi_to_vocal.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

###########################################################################################

"""
    将midi转为json文件，然后转为ds文件，最后转为vocal
    1. 读取midi文件
    2. 将midi转为json文件
    3. 将json文件转为ds文件
        3.1 修改json歌词
        3.2 直接将json转为ds
    4. 将ds文件转为vocal
"""

import re
from singer.human_process import midi_to_json_seperate_by_space
from singer.ds_process.json_process import change_lyric, json_to_ds
from singer.models import run_diffsinger_command_test
from singer.ds_process.midi_process import move_midi_by_max_midi
from singer.ds_process import get_spk_by_ds_for_aces_ds
from singer.ace_process import ds_to_aces, process_aces_to_audio, combine_audio


def main(midi_path, output_dir, lyric_data=None):
    # 创建文件夹
    file_name = midi_path.stem
    json_path = output_dir / "json" / f"{file_name}.json"
    ds_input_path = output_dir / "input" / f"{file_name}.ds"
    ds_output_path = output_dir / "var" / f"{file_name}.ds"
    vocal_path = output_dir / "vocal"
    ace_dir_path = output_dir / "ace" / file_name
    ogg_dir_path = output_dir / "ogg" / file_name
    mp3_path = vocal_path / f"{file_name}.mp3"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    ds_input_path.parent.mkdir(parents=True, exist_ok=True)
    ds_output_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_path.mkdir(parents=True, exist_ok=True)
    ace_dir_path.mkdir(parents=True, exist_ok=True)
    ogg_dir_path.mkdir(parents=True, exist_ok=True)

    # 读取midi文件, 并转为json文件
    json_data = midi_to_json_seperate_by_space(file_name, midi_path, json_path)
    # 将json文件转为ds文件
    # 修改json歌词
    new_lyric_path = "/home/john/CaichongSinger/data/new_lyric/daoxiang.txt"
    with open(new_lyric_path, "r", encoding="utf-8") as f:
        lyric_data = f.read()
    if lyric_data is not None:
        json_data = change_lyric(json_data, lyric_data)

    # 直接将json转为ds
    ds_data = json_to_ds(json_data, ds_input_path)

    # 将ds文件转为vocal
    spk = get_spk_by_ds_for_aces_ds(ds_data)

    if mp3_path.exists():
        logger.info(f"File {mp3_path} exists, skip")
        return
    if spk == "male2":
        run_diffsinger_command_test(ds_input_path, ds_output_path, vocal_path, spk=spk)
    else:
        ds_to_aces(ds_input_path, ace_dir_path)
        spk_id = "18" if spk == "ace_male" else "82"
        process_aces_to_audio(ace_dir_path, ogg_dir_path, spk=spk_id)
        combine_audio(ogg_dir_path, ace_dir_path, mp3_path)


if __name__ == "__main__":
    # python script/task4_25_midi_to_json_to_vocal/midi_to_json_to_vocal_5-14.py
    midi_dir = root_path / "data/human_midi/newmidi2Jhon"
    output_dir = root_path / "data/human_midi_output/newmidi2Jhon"
    output_dir.mkdir(exist_ok=True)
    move_midi_output_dir = output_dir / "moved_midi"
    move_midi_output_dir.mkdir(exist_ok=True)
    # 读取midi文件，根据要求将其进行移位
    move_midi_by_max_midi(midi_dir, move_midi_output_dir)

    for midi_file in move_midi_output_dir.glob("*.mid"):
        # 确保运行目录为根目录
        os.chdir(root_path)
        main(midi_file, output_dir)
