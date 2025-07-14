import os
import torch
import argparse
import subprocess

assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())
# python svc_preprocessing.py -t 36 -w datasetfinetune_raw -o datasetfinetune_svc
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, default=0, help="thread count")
parser.add_argument("-w", type=str, default="dataset_raw", help="dataset path")
parser.add_argument("-o", type=str, default="data_svc", help="output path")
args = parser.parse_args()

src_path = args.w
dst_path = args.o
commands = [
    #  f"python prepare/preprocess_a.py -w {src_path} -o {dst_path}/waves-16k -s 16000 -t {args.t}",
    #  f"python prepare/preprocess_a.py -w {src_path} -o {dst_path}/waves-32k -s 32000 -t {args.t}",
    #  f"python prepare/preprocess_crepe.py -w {dst_path}/waves-16k/ -p {dst_path}/pitch",
    #  f"python prepare/preprocess_ppg.py -w {dst_path}/waves-16k/ -p {dst_path}/whisper",
    #  f"python prepare/preprocess_hubert.py -w {dst_path}/waves-16k/ -v {dst_path}/hubert",
    #  f"python prepare/preprocess_speaker.py {dst_path}/waves-16k/ {dst_path}/speaker -t {args.t}",
    #  f"python prepare/preprocess_speaker_ave.py {dst_path}/speaker/ {dst_path}/singer",
    #  f"python prepare/preprocess_spec.py -w {dst_path}/waves-32k/ -s {dst_path}/specs -t {args.t}",
    f"python prepare/preprocess_train.py --src {dst_path}",
    #  f"python prepare/preprocess_zzz.py -w {dst_path}",
]

for command in commands:
    print(f"Command: {command}")

    process = subprocess.Popen(command, shell=True)
    outcode = process.wait()
    if outcode:
        break
# commands = [
#    "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t 0",
#    "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t 0",
#    "python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch",
#    "python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper",
#    "python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert",
#    "python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t 36",
#    "python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer",
#    "python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs -t 0",
#    "python prepare/preprocess_train.py",
#    "python prepare/preprocess_zzz.py",
# ]
