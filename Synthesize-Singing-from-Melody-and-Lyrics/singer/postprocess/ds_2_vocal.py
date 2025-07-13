# 将ds文件中的内容按照中间rest的位置进行计算，然后将中间进行分割
import os
import glob
import json
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

AP_TIME = 0.2
AP_MIN_TIME = 0.2
SP_MIN_TIME = 0.1
SINGING_LONGEST_TIME = 1

def load_ds(file_path):
    # 打开并读取JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 初始化结果列表
    result_list = []

    # 遍历列表中的每个字典
    for item in data:
        # 提取需要的部分
        result = {
            "offset": float(item.get("offset")),
            "text": item.get("text").split(),
            "ph_seq": item.get("ph_seq").split(),
            "ph_dur": [float(part) for part in item.get("ph_dur").split()],
            "f0_seq": [float(part) for part in item.get("f0_seq").split()],
            "f0_timestep": float(item.get("f0_timestep")),
        }
        # 将结果添加到结果列表中
        result_list.append(result)

    return result_list


def get_ph_f0_info(ph_seq, ph_dur, f0_seq, f0_timestep, sp_duration=0.1):
    # 初始化'SP'和'AP'的时间点和位置
    ph_info = []
    # 初始化当前的时间点
    current_time = 0

    # 遍历ph_seq和ph_dur
    for i, (ph, dur) in enumerate(zip(ph_seq, ph_dur)):
        # 如果音素是'SP'或'AP'，则计算其在f0_seq中的位置和值
        if i < len(ph_seq) - 1:
            # if (ph in ['SP'] and dur >= sp_duration and ph_seq[i+1] in ['AP']) or (ph in ['SP'] and i == 0):
            if (ph in ["SP"] and dur >= sp_duration) or (ph in ["SP"] and i == 0):
                # 计算'SP'或'AP'在f0_seq中的位置
                f0_index = int(current_time / f0_timestep)
                # 获取f0_seq中对应的值
                f0_value = f0_seq[f0_index] if f0_index < len(f0_seq) else None
                # 将'SP'或'AP'的位置、时间点和f0值添加到ph_info
                ph_info.append(
                    {
                        "phoneme": ph,
                        "position": i,
                        "duration": dur,
                        "time": current_time,
                        "f0_value": f0_value,
                        "f0_index": f0_index,
                    }
                )
                # print(f0_seq[f0_index], f0_seq[f0_index+1], f0_seq[f0_index+2])
            # 更新当前的时间点
        current_time += dur

    return ph_info


def split_sequences(ph_info, result, sp_duration=0.1):
    ph_seq = result["ph_seq"]
    ph_dur = result["ph_dur"]
    f0_seq, f0_timestep = result["f0_seq"], result["f0_timestep"]

    # 初始化结果列表
    sequences = []
    # 初始化当前的时间点和序列索引
    start_time = 0
    start_index = 0
    start_ph_num = 0

    last_SP = {"offset": 0, "position": 0, "duration": 0, "f0_value": 0, "f0_index": 0}
    this_SP = {
        "offset": 0,
        "position": 0,
        "duration": 0,
        "time": 0,
        "f0_value": 0,
        "f0_index": 0,
    }

    # 遍历ph_info
    for i, info in enumerate(ph_info):
        # 如果当前音素是'SP'，则将其时间长度设置为sp_duration，并将剩余的时长放在下一个的offset中
        if info["phoneme"] == "SP":
            if i == 0:  # 第一个打算是将SP去除，然后将SP的时间全部都放到offset中
                this_SP["offset"] = info["duration"]
                this_SP["position"] = info["position"] + 1
                this_SP["duration"] = 0
                this_SP["time"] = info["time"]
                this_SP["f0_value"] = info["f0_value"]
                this_SP["f0_index"] = info["f0_index"]
                start_ph_num += 1
                continue
            else:  # 后面的SP则作为前面SP的结尾，时间留的是0.1s
                # 将上一part的继承过来
                # last_SP = this_SP
                last_SP["offset"] = this_SP["offset"]
                last_SP["position"] = this_SP["position"]
                last_SP["duration"] = this_SP["duration"]
                last_SP["f0_value"] = this_SP["f0_value"]
                last_SP["f0_index"] = this_SP["f0_index"]
                this_SP["offset"] = info["duration"] - sp_duration
                this_SP["position"] = info["position"] + 1
                this_SP["duration"] = sp_duration
                this_SP["time"] = info["time"]
                this_SP["f0_value"] = info["f0_value"]
                this_SP["f0_index"] = info["f0_index"]

            # 计算'SP'开始的时间点
            start_time += last_SP["offset"]
            start_index = int(start_time / f0_timestep)
            # 计算'SP'结束的时间点
            end_time = this_SP["time"] + sp_duration
            # 计算'SP'结束时的序列索引
            end_index = int(end_time / f0_timestep)
            # 提取序列的一部分
            part_ph_seq = ph_seq[last_SP["position"] : this_SP["position"]]
            part_ph_dur = ph_dur[last_SP["position"] : this_SP["position"]]
            part_ph_dur[-1] = this_SP["duration"]
            part_f0_seq = f0_seq[start_index:end_index]

            # print(start_time, end_time)
            # 将序列的一部分添加到结果列表中
            sequences.append(
                {
                    "offset": start_time,
                    "ph_seq": part_ph_seq,
                    "ph_dur": part_ph_dur,
                    "f0_seq": part_f0_seq,
                }
            )
            # 更新当前的时间点和序列索引
            start_time = end_time
            start_index = end_index

    """最后一个"""
    # 计算'SP'开始的时间点
    start_time += this_SP["offset"]
    start_index = int(start_time / f0_timestep)
    # 提取序列的一部分
    part_ph_seq = ph_seq[this_SP["position"] :]
    part_ph_dur = ph_dur[this_SP["position"] :]
    part_ph_seq_num = part_ph_seq.__len__()
    part_f0_seq = f0_seq[start_index:]

    # 将序列的一部分添加到结果列表中
    if part_ph_seq_num > 0:
        sequences.append(
            {
                "offset": start_time,
                "ph_seq": part_ph_seq,
                "ph_dur": part_ph_dur,
                "f0_seq": part_f0_seq,
            }
        )
    return sequences


def compute_ph_f0_time(sequences, f0_timestep):
    for seq in sequences:
        ph_total_time = sum(seq["ph_dur"])
        f0_total_time = len(seq["f0_seq"]) * f0_timestep

        print(
            f"ph_total_time {ph_total_time:.6f}, f0_total_time {f0_total_time:.5f} minus : {ph_total_time - f0_total_time:.4f}"
        )


def compute_note_ph_f0_time(ph_dur, f0_seq, note_dur, f0_timestep):
    ph_total_time = sum(ph_dur)
    f0_total_time = len(f0_seq) * f0_timestep
    note_total_time = sum(note_dur)

    # print(f'ph_total_time {ph_total_time:.6f}, f0_total_time {f0_total_time:.6f} minus : {ph_total_time - f0_total_time:.6f}')
    # print(f'no_total_time {note_total_time:.6f}, f0_total_time {f0_total_time:.6f} minus : {note_total_time - f0_total_time:.6f}')
    print(
        f"no_total_time {note_total_time:.6f}, ph_total_time {ph_total_time:.6f} minus : {note_total_time - ph_total_time:.6f}"
    )


def compute_result_num(result):
    ph_seq = result["ph_seq"]
    # 计算 ph_seq 的 个数
    ph_seq_num = ph_seq.__len__()

    note_seq = result["note_seq"]
    # 计算 note_seq 的 个数
    note_num = note_seq.__len__()

    ph_num = result["ph_num"]
    # 计算 ph_num 的 所有元素的和
    ph_num_num = ph_num.__len__()
    ph_num_sum = sum(ph_num)

    note_slur = result["note_slur"]
    # 计算 note_slur 的 个数
    note_slur_num = note_slur.__len__()
    print(
        f"note_num {note_num}, note_slur_num {note_slur_num},  ph_num_num {ph_num_num}, ph_num_sum {ph_num_sum}, ph_seq_num {ph_seq_num}"
    )


def add_AP_after_SP(result):
    new_ph_seq = []
    new_ph_dur = []
    for word,dur in zip(result["ph_seq"], result["ph_dur"]):
        if word == "SP" and dur >= AP_MIN_TIME:
            new_ph_seq.append(word)
            new_ph_dur.append(dur - AP_MIN_TIME)
            new_ph_seq.append("AP")
            new_ph_dur.append(AP_MIN_TIME)
        else:
            new_ph_seq.append(word)
            new_ph_dur.append(dur)
    result["ph_seq"] = new_ph_seq
    result["ph_dur"] = new_ph_dur
    return result

def split_ds(load_path, save_path):
    result = load_ds(load_path)
    
    result[0] = add_AP_after_SP(result[0])

    ph_info = get_ph_f0_info(
        result[0]["ph_seq"],
        result[0]["ph_dur"],
        result[0]["f0_seq"],
        result[0]["f0_timestep"],
    )

    sequences = split_sequences(ph_info, result[0])

    for seq in sequences:
        seq["ph_seq"] = " ".join(seq["ph_seq"])
        seq["ph_dur"] = " ".join(str(item) for item in seq["ph_dur"])
        seq["f0_seq"] = " ".join(str(item) for item in seq["f0_seq"])
        seq["f0_timestep"] = result[0]["f0_timestep"]
    
    with open(save_path, "w") as f:
        json.dump(sequences, f, indent=4)


def get_files(folder_path, new_folder_path):
    # 获取文件夹下所有的.ds后缀文件
    ds_files = glob.glob(os.path.join(folder_path, "*.ds"))

    # 初始化结果列表
    result = []

    # 遍历所有的.ds后缀文件
    for ds_file in ds_files:
        # 获取文件名
        filename = os.path.basename(ds_file)
        # 添加'_split'到文件名
        new_filename = (
            os.path.splitext(filename)[0] + "_split" + os.path.splitext(filename)[1]
        )
        # 获取新的文件路径
        new_file_path = os.path.join(new_folder_path, new_filename)
        # 将文件路径和新的文件路径添加到结果列表
        result.append({"file_path": ds_file, "new_file_path": new_file_path})

    return result


def mq_ds2vocal(
    ds_path,
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
    root_dir = "/export/data/home/john/MuerSinger2/DiffSinger"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda)

    loc = [
        "python",
        "scripts/infer.py",
        "acoustic",
        ds_path,
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
    ), f"Error occurred when processing song {ds_path}"
    return True


def split_ds_to_vocal(ds_path, new_ds_path, vocal_path):
    split_ds(ds_path, new_ds_path)
    mq_ds2vocal(new_ds_path, vocal_path)


if __name__ == "__main__":
    ds_path = "/home/john/DuiniutanqinSinger/data/combined/temp/1686672/1686672.ds"
    new_ds_path = (
        "/home/john/DuiniutanqinSinger/data/combined/temp/1686672/1686672_split.ds"
    )
    # split_ds(ds_path, new_ds_path)

    vocal_path = "/home/john/DuiniutanqinSinger/data/vocal/1686672/"
    # mq_ds2vocal(new_ds_path, vocal_path)
    split_ds_to_vocal(ds_path, new_ds_path, vocal_path)
