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
from singer.human_process import midi_to_json
from singer.ds_process.json_process import change_lyric, json_to_ds
from singer.models import run_diffsinger_command


def main(song_id: str, midi_path, output_dir, lyric_data=None):
    # 创建文件夹
    file_name = midi_path.stem
    json_path = output_dir / "json" / f"{file_name}.json"
    ds_input_path = output_dir / "input" / f"{file_name}.ds"
    ds_output_path = output_dir / "var" / f"{file_name}.ds"
    vocal_path = output_dir / "vocal"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    ds_input_path.parent.mkdir(parents=True, exist_ok=True)
    ds_output_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_path.mkdir(parents=True, exist_ok=True)

    # 读取midi文件, 并转为json文件
    json_data = midi_to_json(song_id, midi_path, json_path)
    # 将json文件转为ds文件
    # 修改json歌词
    if lyric_data is not None:
        json_data = change_lyric(json_data, lyric_data)

    # 直接将json转为ds
    json_to_ds(json_data, ds_input_path)

    # 将ds文件转为vocal
    run_diffsinger_command(ds_input_path, ds_output_path, vocal_path)


if __name__ == "__main__":
    midi_dir = root_path / "data/human_midi"
    output_dir = root_path / "data/human_midi_output"
    song_id_match = re.compile(r"_(\d+)")

    for midi_file in midi_dir.glob("*.mid"):
        # 确保运行目录为根目录
        os.chdir(root_path)

        song_id = song_id_match.search(str(midi_file)).group(1)
        main(song_id, midi_file, output_dir)
