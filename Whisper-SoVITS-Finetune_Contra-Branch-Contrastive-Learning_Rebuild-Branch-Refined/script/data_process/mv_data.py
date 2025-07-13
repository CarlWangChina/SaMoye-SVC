import os
import shutil
import pandas as pd
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import ffmpeg
import numpy as np
import soundfile as sf

# 定义源目录和目标目录
source_base_dir = "/home/john/svc/data/unzip_data"
target_base_dir = "/home/john/svc/data/origin_data/New_DAMP"
csv_file = Path(target_base_dir) / "file_paths.csv"

# 读取 m4a 文件的代码
def load_audio(file):
    try:
        probe = ffmpeg.probe(file)
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        sr = int(audio_info['sample_rate'])
        
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten(), sr

# 定义复制文件的函数
def copy_file(info):
    person, src_path = info
    target_dir = os.path.join(target_base_dir, person)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # 生成新的文件名
    existing_files = len(list(Path(target_dir).glob("*.wav")))
    new_file_name = f"{existing_files + 1:06}.wav"
    target_path = os.path.join(target_dir, new_file_name)

    # Check if the file path is already in copied_paths
    if src_path in copied_paths:
        return person, src_path, target_path  # Skip if already copied

    # 只在目标路径文件不存在时复制或转换
    try:
        if not os.path.exists(target_path):
            if src_path.endswith('.m4a'):
                # 转换 m4a 文件为 wav 文件
                audio_data, sr = load_audio(src_path)
                sf.write(target_path, audio_data, sr, format='wav')
            else:
                # 直接复制非 m4a 文件
                shutil.copy(src_path, target_path)

        # 添加复制完成的文件路径到copied_paths
        copied_paths.add(src_path)
        return person, src_path, target_path
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")
        return person, src_path, "error"
    

def save_to_csv(csv_file, info):
    df_new = pd.DataFrame(info, columns=["spk", "src_path", "target_path"])

    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)

        # Exclude rows that are already in df_existing
        df_new = df_new[~df_new["src_path"].isin(df_existing["src_path"])]

        # Append new data to the existing CSV
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        df_new.to_csv(csv_file, index=False)

# 全局变量，用于存储已经复制过的文件路径
copied_paths = set()

# 在程序开始时读取已经存在的文件路径
if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
    copied_paths.update(df_existing["src_path"].tolist())

def process_subfile(subfile):
    files_to_process = []
    if subfile.exists():
        for file in subfile.rglob("*.m4a"):
            # 母目录为name
            name = file.parent.name
            files_to_process.append((f"DAMP_{name}", str(file)))
    
    results = process_map(copy_file, files_to_process, max_workers=1)

    return results

# 收集所有需要处理的文件夹信息
dirs_to_process = []
damp_dir = Path(source_base_dir) / "data/DAMP"
if damp_dir.exists():
    for subfile in damp_dir.iterdir():
        dirs_to_process.append(subfile)
    
results = process_map(process_subfile, dirs_to_process, max_workers=40)

# 将结果写入或追加到CSV文件
save_to_csv(csv_file, results)