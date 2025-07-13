############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger

logger = get_logger(__name__)

###########################################################################################
r"""
    1. 获取 修改 note_slur、word_dur 的ds_data, 注意ph_seq 要用 word_to_phoneme 函数进行生成
    详情看：singer/ds_process/json_process/json_2_ds.py
    2. 为ds_data添加ph_num
    3. 将ds_data输入var和acoustic模型
"""
import json
from singer.models import run_diffsinger_command_test
from singer.ds_process.utils import (
    add_ph_num,
    word_to_phoneme,
    ms_to_float_s,
)


def your_process(ds_data):
    # TODO: 修改 note_slur、word_dur
    return ds_data


def test_dp(output_dir, ds_path):
    ds_input_path = output_dir / "input" / f"{ds_path.stem}.ds"
    ds_output_path = output_dir / "var" / f"{ds_path.stem}.ds"
    vocal_path = output_dir / "vocal"

    ds_input_path.parent.mkdir(parents=True, exist_ok=True)
    ds_output_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_path.mkdir(parents=True, exist_ok=True)

    with open(ds_path, "r", encoding="utf-8") as f:
        ds_data = json.load(f)

    ds_data = add_ph_num(ds_data, transcription=ds_input_path)
    sound_type = "1"
    if sound_type == "1":
        spk = "male2"  # wzh
    elif sound_type == "2":
        spk = "male1"
    elif sound_type == "3":
        spk = "female1"
    run_diffsinger_command_test(ds_input_path, ds_output_path, vocal_path, spk=spk)


if __name__ == "__main__":
    # python script/task5_18_dp/test_dp.py
    output_dir = root_path / "data/new_ds/dp_test_output"
    ds_dir = root_path / "data/new_ds/dp_test"
    for ds_path in ds_dir.glob("*.ds"):
        test_dp(output_dir, ds_path)
