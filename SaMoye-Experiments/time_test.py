import subprocess
import time
import os
import subprocess
import concurrent.futures
from multiprocessing import Pool
from pathlib import Path
from slicer import slice_one
from pydub import AudioSegment
from omegaconf import OmegaConf


def get_wav_time_len(wav_path):
    audio = AudioSegment.from_file(wav_path)  # 替换成你的音频文件路径
    duration_in_seconds = len(audio) / 1000
    return duration_in_seconds


def main_make_target_time(timeprefix, read_dir, destbase_dir):
    chuncks = [10, 20, 30, 40, 60]

    for spk in read_dir.iterdir():
        spk_name = spk.stem
        dest_dir = destbase_dir / spk_name

        total_time_len = 0
        total = []

        write_flag = [True for i in range(len(chuncks))]
        chuncks_idx = 0
        for i in dest_dir.rglob("*.wav"):
            du = get_wav_time_len(i)
            total_time_len += du
            total.append(i)
            if chuncks_idx < len(chuncks):
                if total_time_len > chuncks[chuncks_idx] and write_flag[chuncks_idx]:
                    raw_chuncks_dir = (
                        Path(timeprefix) / f"{dest_dir.name}_{chuncks[chuncks_idx]}s")
                    raw_chuncks_dir.mkdir(parents=True, exist_ok=True)
                    for audio in total:
                        dst = raw_chuncks_dir / audio.name
                        dst.write_bytes(audio.read_bytes())

                    write_flag[chuncks_idx] = False
                    chuncks_idx += 1


def main_slice(read_dir, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for i in read_dir.iterdir():
        spk_name = i.stem
        save_dir = dest_dir / spk_name
        save_dir.mkdir(parents=True, exist_ok=True)
        slice_one(wav=str(i), out=str(save_dir),
                  min_interval=100, db_thresh=-20)  # by default db_thresh=-40
        print(i)


def main_svc_preprocessing():
    commands = [
        "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t 0",
        "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t 0",
        "python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch",
        "python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper",
        "python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert",
        "python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t 0",
        "python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer",
        "python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs -t 0",
        "python prepare/preprocess_train.py",
        # # "python prepare/preprocess_zzz.py",
    ]

    for command in commands:
        print(f"Command: {command}")
        process = subprocess.Popen(command, shell=True)
        outcode = process.wait()
        if (outcode):
            break


def generate_config_and_trainsh(search_dir, availabel_gpu):
    base = OmegaConf.load("configs/base.yaml")

    speakers = os.listdir(search_dir)
    for name in speakers:
        chkpt_dir = Path(f"chkpt/{name}")
        logs_dir = Path(f"logs/{name}")
        chkpt_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        base["train"]["batch_size"] = 2
        base["data"]["training_files"] = f"files/{name}.txt"
        base["log"]["save_interval"] = 100
        base["log"]["info_interval"] = 20
        base["log"]["pth_dir"] = str(chkpt_dir)
        base["log"]["log_dir"] = str(logs_dir)

        if not os.path.exists(f"configs/{name}.yaml"):
            OmegaConf.save(base, f=f"configs/{name}.yaml")

    cnt = 0
    train_cmds = []
    gpus = len(availabel_gpu)
    for i, name in enumerate(speakers):
        # sh_name = f"train_sh/{name}.sh"
        sh_name = f"train_shpanxin/{name}.sh"

        with open(sh_name, "w") as fw:
            # export CUDA_VISIBLE_DEVICES=4
            # python svc_trainer.py -c configs/zihao_50s.yaml -n sovits5.0
            fw.write(f"export CUDA_VISIBLE_DEVICES={cnt%gpus}\n")
            # fw.write(
            #     f"conda init && conda activate sovits5 && python svc_trainer.py -c configs/{name}.yaml -n sovits5.0")
            fw.write(
                f"python svc_trainer.py -c configs/{name}.yaml -n sovits5.0")
            train_cmd = f"bash {sh_name} > logs/{name}.log 2>&1"
            train_cmds.append(train_cmd)
            print(f"nohup bash {sh_name} > logs/{name}.log 2>&1 &")
            cnt += 1
    return train_cmds


def run_command(train_cmd):
    print(f"Starting command: {train_cmd}")
    result = subprocess.run(
        [f'bash -i train_shpanxin/{train_cmd}'], shell=True, check=True)
    print(f"Completed command: {train_cmd}, Result: {result.stdout.decode()}")
    return result.returncode


# def run_command(train_cmd):
#     print(f"Starting command: {train_cmd}")

#     # 执行训练命令
#     result = subprocess.run(
#         ["bash", "-i", f"train_shpanxin/{train_cmd}"], shell=True, check=True)

#     print(f"Completed command: {train_cmd}, Result: {result.stdout}")
#     return result.returncode


def multi_train(availabel_gpu):
    train_cmds = os.listdir(f"train_shpanxin")

    # 启动多个进程进行训练
    print(f"Total train commands: {len(train_cmds)}")
    print(f"Availabel GPUs: {availabel_gpu}")
    print(f"Train commands: {train_cmds}")

    # pool = Pool(len(availabel_gpu))
    # results = pool.map(run_command, train_cmds)
    # for result in results:
    #     print(f"Result: {result}")
    # pool.close()
    # pool.join()

    # print("All train commands completed!")


if __name__ == "__main__":
    # timeprefix=time.strftime("%Y%m%d%H%M%s", time.localtime(time.time()))
    timeprefix = "dataset_raw"
    dest_dir = Path("data/dialect_slice")  # main_slice 存储的位置
    read_dir = Path('data/dialect/tmp')  # 待切分数据
    availabel_gpu = [0, 1, 2, 3, 4, 5, 6, 7]

    main_slice(read_dir, dest_dir)  # step 1
    main_make_target_time(timeprefix, read_dir, dest_dir)  # step 2
    main_svc_preprocessing()
    train_cmds = generate_config_and_trainsh(
        "data_svc/waves-32k", availabel_gpu)

    # multi_train(availabel_gpu)  # step 3
