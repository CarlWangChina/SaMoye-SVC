############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger

logger = get_logger(__name__)

###########################################################################################
import time
import os
import shutil
import argparse
from singer.models.DurationPredictor import infer

if __name__ == "__main__":
    # python script/task5_18_dp/infer_script.py
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/john/CaichongSinger/data/new_ds/dp_test"
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="/home/john/CaichongSinger/data/new_ds/dp_test_output",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/john/CaichongSinger/checkpoint/lyrics_to_notes/2024-05-22-14-10-02/lyrics_to_notes_model_epoch_20.pth",
    )
    args = parser.parse_args()
    infer(args)
