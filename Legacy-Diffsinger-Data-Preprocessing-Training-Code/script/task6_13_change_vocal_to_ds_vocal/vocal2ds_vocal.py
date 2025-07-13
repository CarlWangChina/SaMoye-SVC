############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
os.environ["PYTHONPATH"] = str(root_path)
sys.path.insert(0, str(root_path))

from singer.utils.logging import get_logger

logger = get_logger(__name__)

###########################################################################################

"""
    1. 读取 wav 和 其lab,获取其ph,ph_dur,f0
    2. 保存为ds文件，然后直接生成vocal
"""

import argparse
from singer.ds_process import vocal_to_ds, split_ds
from singer.models import run_diffsinger_command_test
from singer.ds_process.utils import convert_mp3_to_wav

spk_types = {
    "#1_BJ_何畅": "female1",
    "#2_BJ_崔璨": "male1",
    "#1_数字_女声全": "female1",
    "#2_数字_男声全": "male1",
    "女": "female1",
    "男": "male1",
    "#1_615_女声b": "female1",
    "#2_615_男声b": "male1",
    "#4_615_女声a": "female1",
    "#3_615_男声a": "male1",
    "617加强": "female1",
}


def main(args):
    only_vocal = args.only_vocal
    input_file = Path(args.input_file)
    output_file = args.output_file
    if output_file is None:
        output_file = input_file
    output_file = Path(output_file)
    split_output_file = output_file / "split"
    split_output_file.mkdir(exist_ok=True, parents=True)
    # 如果文件夹下没有wav文件，就转换mp3文件为wav文件
    if not list(input_file.glob("*.wav")):
        print("convert mp3 to wav")
        for mp3_file in input_file.glob("*.mp3"):
            convert_mp3_to_wav(mp3_file, input_file / (mp3_file.stem + ".wav"))
            print(f"convert {mp3_file} to wav")
    if not only_vocal:
        vocal_to_ds(input_file, cuda=0)
    for ds_file in output_file.glob("*.ds"):
        ds_path = ds_file
        new_ds_path = split_output_file / (ds_path.stem + "_split.ds")
        if not only_vocal:
            split_ds(ds_path, new_ds_path)
        spk = spk_types[str(ds_path.stem)]
        run_diffsinger_command_test(
            ds_output_path=new_ds_path,
            vocal_path=split_output_file,
            spk=spk,
            save_as_wav=True,
        )


if __name__ == "__main__":
    # python script/task6_13_change_vocal_to_ds_vocal/vocal2ds_vocal.py -i /home/john/CaichongSinger/data/xs2ds/6-17mv --only_vocal True
    parser = argparse.ArgumentParser(description="change vocal to ds vocal")
    parser.add_argument(
        "--input_file", "-i", type=str, required=True, help="input file path"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=False,
        default=None,
        help="output file path",
    )
    parser.add_argument(
        "--only_vocal",
        type=bool,
        required=False,
        default=False,
        help="only vocal",
    )
    args = parser.parse_args()
    main(args)
