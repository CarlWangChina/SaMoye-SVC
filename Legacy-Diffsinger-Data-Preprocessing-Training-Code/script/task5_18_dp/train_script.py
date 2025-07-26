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

# 获取时间作为输出文件夹命名内容
time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
from singer.models.DurationPredictor import train

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python script/task5_18_dp/train_script.py
    # python script/task5_18_dp/train_script.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--data_dir", type=str, default="/app/data/samoye-exp/align_data/4w"
    )
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/app/data/samoye-exp/checkpoint/lyrics_to_notes/{time_str}",
    )
    args = parser.parse_args()
    train(args)
