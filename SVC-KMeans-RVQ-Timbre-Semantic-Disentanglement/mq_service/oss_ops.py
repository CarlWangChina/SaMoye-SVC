"""
    Transfer data between MQ/OSS and Local Server

"""

import os
import sys
from urllib.parse import urljoin

import requests
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from utils import get_logger, get_hparams

log = get_logger(__name__)
oss_hparams = get_hparams()["oss"]

# 配置 OSS 访问参数
END_POINT = oss_hparams["END_POINT"]  # OSS 外网节点
# 不能包含bucket名称：source
# https://caichong-avatar.oss-cn-zhangjiakou.aliyuncs.com/source/data4oss/1116445/1116445_origin.mid
ACCESS_KEY_ID = oss_hparams["OSS_ACCESS_KEY_ID"]
ACCESS_KEY_SECRET = oss_hparams["OSS_ACCESS_KEY_SECRET"]
BUCKET_NAME = oss_hparams["BUCKET_NAME"]  # 你的 bucket 名称


def merge_oss_address(
    oss_address: str,
    end_point: str = END_POINT,
    bucket_name: str = BUCKET_NAME,
) -> str:
    """
    Merge the OSS address

    Args:
        end_point (str): The end point of the OSS
        bucket_name (str): The name of the bucket
        oss_address (str): The address of the file in OSS

    Returns:
        str: The total address of the file in OSS

    """
    # 合并 URL：https://bucket_name.end_point/oss_address
    end_point = end_point.split("https://")[-1]
    # https://caichong-source.oss-cn-zhangjiakou.aliyuncs.com/test/f511af8750d3270aac16cd8fdcb04ff9.mp3
    total_oss_address = urljoin(
        "https://" + bucket_name + "." + end_point, oss_address)
    return total_oss_address


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
    log.info(
        f" [x]-{os.getpid()} Downloading file {oss_sub_address} to {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 下载文件
        result = bucket.get_object_to_file(oss_sub_address, local_path)
        if result.status == 200:
            log.info(f"Download file {oss_sub_address} Success!")
            return True
    except oss2.exceptions.NoSuchKey as e:
        log.error(f"Download file {oss_sub_address} failed! {e}")
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
        log.info(f"文件已下载到：{save_path}")
    else:
        log.info(f"下载失败:HTTP 状态码 {response.status_code}")


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
            log.info(f"\r{rate}% ", end="")
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
            log.info(f"Upload file {oss_address} Success!")
            return True
    except oss2.exceptions.NoSuchKey as e:
        log.error(f"Upload file {oss_address} failed! {e}")
        return False
