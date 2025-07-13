""" Inference script for svc inference with auto-slice, using on self test and online service.
    Author: Xin Pan
    Data: 2024.06.19
"""

import argparse
import torch
import time
import yaml

from src.svc_wrapper import (SVC5)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False,
                        help="Path to svc service config file")
    parser.add_argument("-w", "--tgtwav", type=str, required=True,
                        help="Path of target wav file")
    parser.add_argument("-s", "--savepath", type=str, required=True,
                        help="Path to save output files.")
    return parser.parse_args()


def main_svc_inference_auto_slice(args):
    # Get input params
    savePath = args.savepath
    modelConfig = args.config
    tgtWav = args.tgtwav

    # Load config from yaml file, e.g. "configs/config.yaml"
    configs = yaml.load(
        open(modelConfig, "rt", encoding="utf8"), Loader=yaml.FullLoader)
    svcConfig = configs["svc"]

    # Define the inference class, and initialize it
    svcObj = SVC5(**svcConfig)

    for i in range(1):
        print("Inference round: ", i)
        print("memory:", torch.cuda.memory_allocated(0) / 1024 / 1024,
              "MB ", torch.cuda.memory_reserved(0) / 1024 / 1024, "MB")
        bret = svcObj.inference_with_auto_slice(tgtWav, savePath)
        print("memory:", torch.cuda.memory_allocated(0) / 1024 / 1024,
              "MB ", torch.cuda.memory_reserved(0) / 1024 / 1024, "MB")
        print(bret)


if __name__ == "__main__":
    args = get_args()
    main_svc_inference_auto_slice(args)
