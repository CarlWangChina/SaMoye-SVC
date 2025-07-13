# 1. 读取/home/john/svc/muer-svc-whisper-sovits-svc/models/so-vits-svc/filelists/train.txt
# 数据为：./dataset/44k/DAMP_753992676/000004_23.wav这种内容
# 2. 检查文件是否存在，如果不存在则删除
# 3. 检查文件及其开头的 pt和npy 文件是否存在nan，如果是则删除
# 4. 需要使用多进程：from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=num_processes) as executor:
# 5. 最终保存到/home/john/svc/muer-svc-whisper-sovits-svc/models/so-vits-svc/filelists/train_without_nan.txt


import os
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

base_dir = "/home/john/svc/muer-svc-whisper-sovits-svc/models/so-vits-svc"
os.chdir(base_dir)

# 设置路径
input_file = "filelists/train.txt"
output_file = "filelists/train_without_nan.txt"


# 定义检查函数
def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return (file_path, False)

    # try:
    # 检查对应的pt文件
    pt_file = file_path.replace(".wav", ".spec.pt")
    if os.path.exists(pt_file):
        if torch.isnan(torch.load(pt_file)).any():
            print(f"{pt_file} has NaN values.")
            return (file_path, False)
    else:
        print(f"{pt_file} does not exist.")
        return (file_path, False)

    # 检查 soft.pt 文件
    soft_pt_file = file_path.replace(".wav", ".wav.soft.pt")
    if os.path.exists(soft_pt_file):
        if torch.isnan(torch.load(pt_file)).any():
            print(f"{soft_pt_file} has NaN values.")
            return (file_path, False)
    else:
        print(f"{soft_pt_file} does not exist.")
        return (file_path, False)

    # 检查对应的npy文件
    npy_file = file_path + ".f0.npy"
    if os.path.exists(npy_file):
        f0, uv = np.load(npy_file, allow_pickle=True)
        if np.isnan(f0).any() or np.isnan(uv).any():
            print(f"{npy_file} has NaN values.")
            return (file_path, False)
    else:
        print(f"{npy_file} does not exist.")
        return (file_path, False)

    return (file_path, True)

    # except Exception as e:
    #     print(f"Error processing {file_path}: {e}")
    #     return (file_path, False)


def check_file_batch(file_paths):
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    results = []
    for file_path in tqdm(file_paths, position=rank):
        results.append(check_file(file_path))
    return results


# 读取输入文件
with open(input_file, "r") as f:
    file_paths = [line.strip() for line in f]

file_paths = file_paths[:10]
# 使用多进程处理文件

num_processes = 10  # 根据你的需要设置进程数
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    tasks = []
    for i in range(num_processes):
        start = int(i * len(file_paths) / num_processes)
        end = int((i + 1) * len(file_paths) / num_processes)
        file_chunk = file_paths[start:end]
        tasks.append(executor.submit(check_file_batch, file_chunk))

    for task in tqdm(tasks, position=0):
        pass

# 需要去掉每个task的列表外层
results = [result for task in tasks for result in task.result()]
print(results)
# 过滤None值并写入输出文件
error_files = [file_path for file_path, valid in results if not valid]
print(f"Found {len(error_files)} invalid files.They are {error_files}")
valid_files = [file_path for file_path, valid in results if valid]
with open(output_file, "w") as f:
    for file_path in valid_files:
        f.write(f"{file_path}\n")

print("Processing complete.")
