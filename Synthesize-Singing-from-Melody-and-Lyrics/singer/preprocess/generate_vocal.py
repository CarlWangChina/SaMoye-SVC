""" This module is used to generate vocal from the input ds file. """

import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import pathlib


# 创建一个logger
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("my_app.log", maxBytes=1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)


def mq_run_diffsinger_command(
    ds_input_path,
    ds_output_path,
    vocal_path,
    spk="cpop_female",
    batch_size: int = 50,
    cuda: int = 7,
):
    """
    Run the command of DiffSinger.

    Args:
        ds_input_path (str): The path of the input ds file
        ds_output_path (str): The path of the output ds file
        vocal_path (str): The path of the vocal file
        spk (str, optional): The speaker. Defaults
        batch_size (int, optional): The batch size. Defaults to 50.
        cuda (int, optional): The cuda device. Defaults to 7.

    Returns:
        None
    """
    # 设置根目录
    root_dir = "/home/john/DiffSinger"
    # 更改工作目录到根目录
    os.chdir(root_dir)

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda)

    # 这是个pathlib.Path对象
    ds_output_path_base = pathlib.Path(ds_output_path).parent
    # loc = [
    #     "python",
    #     "onnxrun/infer_onnx.py",
    #     "variance",
    #     ds_input_path,
    #     "--exp",
    #     "cpop_variance",
    #     "--out",
    #     ds_output_path_base,
    #     "--predict",
    #     "pitch",
    #     "--predict",
    #     "dur",
    #     "--batch_size",
    #     str(batch_size),
    # ]
    loc = [
        "python",
        "scripts/infer.py",
        "variance",
        ds_input_path,
        "--exp",
        "baishuo_pitch2",
        "--out",
        ds_output_path_base,
    ]
    completed_process = subprocess.run(loc, env=env, check=True)
    assert (
        completed_process.returncode == 0
    ), f"Error occured when processing song {ds_input_path}"

    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2/DiffSinger"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    loc = [
        "python",
        "scripts/infer.py",
        "acoustic",
        ds_output_path,
        "--exp",
        "muer_singer",
        "--spk",
        spk,
        "--out",
        vocal_path,
        "--batch_size",
        str(batch_size),
    ]
    completed_process = subprocess.run(loc, env=env, check=True)
    assert (
        completed_process.returncode == 0
    ), f"Error occurred when processing song {ds_input_path}"
    return True
