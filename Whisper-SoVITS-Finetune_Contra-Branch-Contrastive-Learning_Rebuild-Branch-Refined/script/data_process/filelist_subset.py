import torch
import numpy as np
import os
from pathlib import Path

base_dir = "/home/john/svc/muer-svc-whisper-sovits-svc/models/whisper_vits_svc"


def check_nan_in_file(file_path):
    file_path = os.path.join(base_dir, file_path[2:])
    if file_path.endswith(".npy"):
        data = np.load(file_path)
    elif file_path.endswith(".pt"):
        data = torch.load(file_path)
    elif file_path.endswith(".wav"):
        return False
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return np.isnan(data).any()


def process_line(line):
    paths = line.strip().split("|")
    for path in paths:
        if check_nan_in_file(path):
            return (line, False)
    return (line, True)


from multiprocessing import Pool
from tqdm import tqdm


def process_lines(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with Pool() as p:
        results = list(
            tqdm(
                p.imap(process_line, lines),
                total=len(lines),
            )
        )
    wrong_files = [res[0] for res in results if res[1] is False]
    print(f"Wrong files are {wrong_files}")
    # Filter out None results and write to output file
    results = [res[0] for res in results if res[1] is True]
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(results)


input_file = "/home/john/svc/data/files/train_filtered_1700.txt"
output_file = "/home/john/svc/data/files/train_filtered_1700_withoutnan.txt"
process_lines(input_file, output_file)
