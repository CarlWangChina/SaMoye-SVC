# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>

import requests
import oss2
import os
import sys
import auenhan.config_loader

import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

config = auenhan.config_loader.config
END_POINT = auenhan.config_loader.END_POINT
ACCESS_KEY_ID = auenhan.config_loader.ACCESS_KEY_ID
ACCESS_KEY_SECRET = auenhan.config_loader.ACCESS_KEY_SECRET
BUCKET_NAME = auenhan.config_loader.BUCKET_NAME

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
    logger.info(" [x]-%d Downloading file %s to %s", os.getpid(), oss_sub_address, local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 下载文件
        result = bucket.get_object_to_file(oss_sub_address, local_path)
        if result.status == 200:
            logger.info("Download file %s Success!", oss_sub_address)
            return True
    except oss2.exceptions.NoSuchKey as e:
        logger.error("Download file %s failed! %s", oss_sub_address, e)
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
        logger.info("文件已下载到：%s", save_path)
    else:
        logger.error("下载失败:HTTP 状态码 %s", response.status_code)


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
            logger.info("Upload file %s Success!", oss_address)
            return True
    except oss2.exceptions.NoSuchKey as e:
        logger.error("Upload file %s failed! %s", oss_address, e)
        return False

