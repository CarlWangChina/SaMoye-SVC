# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>
import multiprocessing
import os
import random
import time
import json
import torch
import torchaudio
import numpy as np
from torchaudio.transforms import Resample

import pika
from mqc_helper import SUBQ_NAME, get_mq_cnx
from result_pub import send_push

import sys
from urllib.parse import urljoin

import requests
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import pyloudnorm as pyln

import subprocess
import random
import string
import watermark
from omegaconf import OmegaConf
from datetime import datetime
from mutagen.easyid3 import EasyID3

config = OmegaConf.load("/configs/config.yaml")

# 配置 OSS 访问参数
END_POINT = config.mq.END_POINT
ACCESS_KEY_ID = config.mq.ACCESS_KEY_ID
ACCESS_KEY_SECRET = config.mq.ACCESS_KEY_SECRET
BUCKET_NAME = config.mq.BUCKET_NAME

current_dir = os.path.dirname(os.path.abspath(__file__))

def download_file(oss_address: str, local_path: str) -> bool:
    """
    Download file from OSS

    Args:
        oss_address (str): The address of the file in OSS
        local_path (str): The path of the file to be saved locally

    Returns:
        None
    """
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    # auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket_name = "source"
    bucket = oss2.Bucket(auth, END_POINT, bucket_name)
    # 不能包含bucket名称前面内容
    oss_sub_address = oss_address.split("source/")[-1]
    print(f" [x]-{os.getpid()} Downloading file {oss_sub_address} to {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 下载文件
        result = bucket.get_object_to_file(oss_sub_address, local_path)
        if result.status == 200:
            print(f"Download file {oss_sub_address} Success!")
            return True
    except oss2.exceptions.NoSuchKey as e:
        print(f"Download file {oss_sub_address} failed! {e}")
        return False


def download_file_from_url(url, save_path):
    """
    从给定的 URL 下载文件到指定的本地路径。

    参数：
    url: 要下载文件的 URL。
    save_path: 下载文件的本地保存路径。
    """
    # 发送 GET 请求下载文件
    response = requests.get(url, stream=True, timeout=10)
    if response.status_code == 200:
        # 打开本地文件以二进制写入模式
        with open(save_path, "wb") as file:
            # 分块下载文件并写入本地文件
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"文件已下载到：{save_path}")
    else:
        print(f"下载失败:HTTP 状态码 {response.status_code}")


def upload_file(local_path: str, oss_address: str) -> bool:
    """
    Upload file to OSS

    Args:
        local_path (str): The path of the file to be uploaded
        oss_address (str): The address of the file in OSS

    Returns:
        None

    """

    def percentage(consumed_bytes, total_bytes):
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print(f"\r{rate}% ", end="")
            sys.stdout.flush()

    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    # auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, END_POINT, BUCKET_NAME)

    # 上传文件
    try:
        result = bucket.put_object_from_file(
            oss_address, local_path, progress_callback=percentage
        )
        if result.status == 200:
            print(f"Upload file {oss_address} Success!")
            return True
    except oss2.exceptions.NoSuchKey as e:
        print(f"Upload file {oss_address} failed! {e}")
        return False

def load_audio(path: str, sr: int):
    """
    Load audio file and set sample rate.

    Args:
        path (str): The path of the audio file.
        sr (int): The target sample rate.

    Returns:
        tuple[torch.Tensor, int]: audio data, sample rate

    """
    audio, orig_sample_rate = torchaudio.load(path)  
    if audio.shape[0] > 1:  
        audio = audio.mean(dim=0)  
    
    if orig_sample_rate != sr:
        resample = Resample(orig_sample_rate, sr)  
        audio = resample(audio)  
    return audio, sr

def scale_volume_numpy(audio: np.ndarray, dB: float, sample_rate: int, lufs: float = -17.0):
    """
    Scale the volume of the audio signal.

    Args:
        audio (np.ndarray): The audio signal.
        dB (float): The target volume in dB.
        sample_rate (int): The sample rate of the audio signal.
        lufs (float): The target volume in LUFS. Defaults to -17.0.

    Returns:
        np.ndarray: The scaled audio signal.
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, lufs)
    
    volume_factor = 10 ** (dB / 20)
    audio = audio * volume_factor
    return audio

def scale_volume(audio: torch.Tensor, dB: float, sample_rate: int, lufs: float):
    """
    Scale the volume of the audio signal.

    Args:
        audio (torch.Tensor): The audio signal.
        dB (float): The target volume in dB.
        sample_rate (int): The sample rate of the audio signal.
        lufs (float): The target volume in LUFS. Defaults to -17.0.

    Returns:
        torch.Tensor: The scaled audio signal.
    """
    audio = audio.numpy()
    audio = scale_volume_numpy(audio, dB, sample_rate, lufs)
    return torch.from_numpy(audio)

def merge_data(vocal:str, acc:str, outpath:str, uid:str="", mid:str=""):
    """
    Merge vocal and acc data.

    Args:
        vocal (str): The path of the vocal data.
        acc (str): The path of the acc data.
        outpath (str): The path of the output file.
        uid (str): The user ID.
        mid (str): The music ID.

    Returns:
        None
    """
    audio_vocal, sample_rate = load_audio(vocal,sr=48000)
    audio_acc, sample_rate = load_audio(acc,sr=48000)

    if audio_vocal.dim() > 1:  
        audio_vocal = audio_vocal.mean(dim=0)  
    if audio_acc.dim() > 1:  
        audio_acc = audio_acc.mean(dim=0)

    #merge
    audio_len = min([audio_vocal.shape[0], audio_acc.shape[0]])

    scaled_vocal = scale_volume(audio_vocal[:audio_len], dB=config.render.volume_db.vocal, sample_rate=sample_rate, lufs=config.render.volume_norm.vocal)
    scaled_acc = scale_volume(audio_acc[:audio_len], dB=config.render.volume_db.acc, sample_rate=sample_rate, lufs=config.render.volume_norm.acc)

    audio =  scaled_vocal + scaled_acc
    audio = scale_volume(audio, dB=0, sample_rate=sample_rate, lufs=config.render.volume_norm.output).to(torch.float32)

    # print(audio, audio.dtype)

    torchaudio.save(
        outpath,
        audio.view(1,-1),
        sample_rate,
    )
    # 将时间格式化为字符串，精确到毫秒
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")

    #添加盲水印，由于影响音质，暂时关闭
    # wm_str = f" DuiNiuTanQin {time_string} {mid} {uid}"  # 水印
    # if len(wm_str)<128:
    #     wm_str = " "*(128-len(wm_str)) + wm_str

    # wm_bits = watermark.bytes2bin(wm_str.encode('utf-8'))
    
    # # 嵌入水印
    # echo_wm = watermark.EchoWatermark(pwd=111002)
    # echo_wm.verbose = True
    # echo_wm.embed(origin_filename=outpath+"_wm.wav", wm_bits=wm_bits, embed_filename=outpath+"_wm_out.wav")
    
    # subprocess.run([
    #         "ffmpeg", "-i",
    #         os.path.abspath(outpath+"_wm_out.wav"),
    #         "-ac", "1", 
    #         outpath
    #     ], check=True)
    
    # os.remove(outpath+"_wm.wav")
    # os.remove(outpath+"_wm_out.wav")

    audio = EasyID3(outpath)

    # 添加元数据

    copyright = json.dumps({"ServiceProvider":"DuiNiuTanQin", "Time":time_string, "ContentID":mid, "UID":uid})
    audio['copyright'] = "aigc: "+copyright

    # 保存修改
    audio.save()

def add_effect(config_path:str, infile:str, outfile:str):
    """
    Add effect to the input audio file.

    Args:
        config_path (str): The path to the effector configuration file.
        infile (str): The path to the input audio file.
        outfile (str): The path to the output audio file.

    Returns:
        None
    """
    try:
        rand_str = generate_random_string(8)
        tmp_wav = os.path.abspath(infile)+f"_{rand_str}.wav"
        subprocess.run([
            "ffmpeg", "-i",
            os.path.abspath(infile),
            "-ac", "1", 
            tmp_wav
        ], check=True)
        command = [
            os.path.join(current_dir, "effector/build/eff-cli"),  # eff-cli可执行文件路径
            os.path.abspath(config_path), 
            tmp_wav, 
            os.path.abspath(outfile)
        ]

        subprocess.run(command, check=True)

    except subprocess.CalledProcessError as e:
        print(f"add_effect failed:{e}")

    finally:
        os.remove(tmp_wav)

def generate_random_string(length:int):
    """
    Generate a random string of specified length.

    Args:
        length (int): The length of the random string.

    Returns:
        str: The generated random string.
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def merge_data_remote(vocal_url:str, acc_url:str, mid:str, uid:str):
    """
    Merge vocal and acc data from remote url.

    Args:
        vocal_url (str): The URL of the vocal data.
        acc_url (str): The URL of the acc data.
        mid (str): The MID of the song.
        uid (str): The UID of the user.

    Returns:
        str: The URL of the merged audio file.
    """
    rand_id = generate_random_string(16)
    mp3_oss_address = f"music/{rand_id}/{mid}.mp3"
    try:
        download_file_from_url(vocal_url, f"/tmp/{mid}_vocal.mp3")
        download_file_from_url(acc_url, f"/tmp/{mid}_acc.mp3")
        add_effect(
            config_path=config.render.effector,
            infile=f"/tmp/{mid}_vocal.mp3",
            outfile=f"/tmp/{mid}_vocal_eq.wav")
        merge_data(f"/tmp/{mid}_vocal_eq.wav", f"/tmp/{mid}_acc.mp3", f"/tmp/{mid}.mp3", uid=uid, mid=mid)
        upload_file(f"/tmp/{mid}.mp3", mp3_oss_address)
        return f"https://caichong-source.oss-cn-zhangjiakou.aliyuncs.com/{mp3_oss_address}"
    finally:
        if not config.render.keepfile:
            os.remove(f"/tmp/{mid}_vocal.mp3")
            os.remove(f"/tmp/{mid}_vocal_eq.wav")
            os.remove(f"/tmp/{mid}_acc.mp3")
            os.remove(f"/tmp/{mid}.mp3")

def callback(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes
):
    pid = os.getpid()
    data = json.loads(body)
    print(f" [x]-{pid} ATTN: Sending the ok msg immediately!")

    result = []
    for it in data:
        result.append({"mid":it["mid"],"oss":"","server":"end"})
    send_push({"type":"ok","msg":result})
    
    print(f" [x]-{pid} Start merge data...")
    # This is just a simulation of time-consuming work
    
    result = []
    for it in data:
        try:
            result.append({"mid":it["mid"], "oss":merge_data_remote(it["soundoss"],it["tmposs"],it["mid"],uid=it["uid"]),"server":"end"})
        except Exception as e:
            print(e)
    
    # time.sleep(random.randint(3, 9))
    print(f" [x]-{pid} Done! Sending succ type msg...")

    send_push({"type":"succ","msg":result})

def subscribe():
    # TODO read from config file
    cnx_rec_max_t = 4
    cnx_rec_times = 0

    while cnx_rec_times <= cnx_rec_max_t:
        try:
            # don't forget to add param `--network="host"` when running docker container
            connection = get_mq_cnx()

            # channel for input, sub for the task messages
            chan_ip = connection.channel()

            chan_ip.queue_declare(queue=SUBQ_NAME, durable=True)

            chan_ip.basic_qos(prefetch_count=1)
            chan_ip.basic_consume(
                queue=SUBQ_NAME,
                auto_ack=True,
                on_message_callback=callback)

            chan_ip.start_consuming()
        # Don't recover if connection was closed by broker
        except pika.exceptions.ConnectionClosedByBroker:
            break
        # Don't recover on channel errors
        except pika.exceptions.AMQPChannelError:
            break
        # Recover on all other connection errors
        except pika.exceptions.AMQPConnectionError:
            cnx_rec_times += 1
            time.sleep(2)
            continue

def start():
    workers_num = 8
    mpp = multiprocessing.Pool(processes=workers_num)
    for i in range(workers_num):
        mpp.apply_async(subscribe)
    mpp.close()
    mpp.join()

# if __name__ == '__main__':
#         add_effect(
#             config_path=config.render.effector,
#             infile="/ori/65447922d463fb8072a64517b15239c4395c9f9e_vocal.mp3",
#             outfile="/ori/65447922d463fb8072a64517b15239c4395c9f9e_vocal_eq.wav")
#         merge_data(vocal="/ori/65447922d463fb8072a64517b15239c4395c9f9e_vocal_eq.wav", acc="/ori/65447922d463fb8072a64517b15239c4395c9f9e_acc.mp3", outpath="/ori/65447922d463fb8072a64517b15239c4395c9f9e.mp3")

#     import librosa
#     import soundfile
#     audio_file = '/export/data/datasets-mp3/ali/16/1691673_src.mp3'
#     audio_data, sample_rate = load_audio(audio_file, sr=48000)

#     # 应用滤波器
#     filtered_audio = apply_filter(audio_data, sample_rate)
#     # print(filtered_audio)
#     soundfile.write("1691673_eq.mp3", data=filtered_audio, samplerate=sample_rate)
