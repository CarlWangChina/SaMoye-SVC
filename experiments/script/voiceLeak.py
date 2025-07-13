""" 针对音色泄露做一些实验
统计平均情况下  同一个人的不同内容之间的距离   以及不同人的不同内容之间的距离   第一个距离是不是第二个更近

"""

import numpy as np
import logging
import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path

from cluster import (get_cluster_model, get_cluster_center_result)

from ipdb import set_trace

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity

# def cosine_similarity(v1: list, v2: list):
#     num = float(np.dot(v1, v2))  # 向量点乘
#     denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
#     return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def main():
    # Hubert-soft
    # Target spkeakers
    spkDirs = ["data_svc/hubert/beijing_female_slice_60s",
               "data_svc/hubert/beijingmale_sing_60s",
               "data_svc/hubert/sichuan_rap_60s",
               "data_svc/hubert/using3_60s"]
    spkDirs = [Path(dir) for dir in spkDirs]

    spkAvgVec = {}  # 每个发音人所有素材vec的平均值。
    spkDataAvgVec = {}  # 每个发音人，每个素材vec的平均值。

    for spkDir in spkDirs:
        # 定义一个空np.array
        spkName = spkDir.name

        if not spkName in spkDataAvgVec:
            spkDataAvgVec[spkName] = np.empty((0, 256))

        vecDatas = np.empty((0, 256))
        for file in spkDir.glob("*.vec.npy"):
            vecData = np.load(file)
            vecDatas = np.concatenate((vecDatas, vecData), axis=0)
            spkDataAvgVec[spkName] = np.concatenate(
                (spkDataAvgVec[spkName], np.mean(vecData, axis=0)[np.newaxis, :]), axis=0)
            # logging.info(f"vecDatas shape: {vecDatas.shape}")

        # Save one to spkAvgVec
        spkAvgVec[spkName] = np.mean(vecDatas, axis=0)

    # Calculate distance between one's vecs, different vec of A and avg vec of A
    for spk in spkDataAvgVec:
        if not spk in spkAvgVec:
            raise KeyError(f"{spk} not in spkAvgVec")

        for idx, vec in enumerate(spkDataAvgVec[spk]):
            similarity = cosine_similarity(vec, spkAvgVec[spk])
            logging.info(f"Smilarity between {spk} and {idx} is {similarity}")

    logging.info("-" * 100)
    # Calculate distance between different vecs, avg vec of A and other spkeakers vec
    for spk in spkAvgVec:
        if not spk in spkDataAvgVec:
            raise KeyError(f"{spk} not in spkDataAvgVec")

        differentSpks = list(spkDataAvgVec.keys())
        differentSpks.remove(spk)  # not calc with itself
        for spk2 in differentSpks:
            for idx, vec in enumerate(spkDataAvgVec[spk2]):
                similarity = cosine_similarity(vec, spkAvgVec[spk])
                # if similarity < 0.0:
                #     set_trace()

                #     dot_product = np.dot(vec, spkAvgVec[spk])
                #     norm_vector1 = np.linalg.norm(vec)
                #     norm_vector2 = np.linalg.norm(spkAvgVec[spk])
                #     similarity = dot_product / (norm_vector1 * norm_vector2)
                logging.info(
                    f"Smilarity between {spk} and {spk2}_{idx} is {similarity}")


def main2():
    # whisper
    # Target spkeakers
    spkDirs = ["data_svc/whisper/beijing_female_slice_60s",
               "data_svc/whisper/beijingmale_sing_60s",
               "data_svc/whisper/sichuan_rap_60s",
               "data_svc/whisper/using3_60s"]
    spkDirs = [Path(dir) for dir in spkDirs]

    spkAvgVec = {}  # 每个发音人所有素材vec的平均值。
    spkDataAvgVec = {}  # 每个发音人，每个素材vec的平均值。

    for spkDir in spkDirs:
        # 定义一个空np.array
        spkName = spkDir.name

        if not spkName in spkDataAvgVec:
            spkDataAvgVec[spkName] = np.empty((0, 1280))

        vecDatas = np.empty((0, 1280))
        for file in spkDir.glob("*.ppg.npy"):
            vecData = np.load(file)
            vecDatas = np.concatenate((vecDatas, vecData), axis=0)
            spkDataAvgVec[spkName] = np.concatenate(
                (spkDataAvgVec[spkName], np.mean(vecData, axis=0)[np.newaxis, :]), axis=0)
            # logging.info(f"vecDatas shape: {vecDatas.shape}")

        # Save one to spkAvgVec
        spkAvgVec[spkName] = np.mean(vecDatas, axis=0)

    # Calculate distance between one's vecs, different vec of A and avg vec of A
    for spk in spkDataAvgVec:
        if not spk in spkAvgVec:
            raise KeyError(f"{spk} not in spkAvgVec")

        for idx, vec in enumerate(spkDataAvgVec[spk]):
            similarity = cosine_similarity(vec, spkAvgVec[spk])
            logging.info(f"Smilarity between {spk} and {idx} is {similarity}")

    logging.info("-" * 100)
    # Calculate distance between different vecs, avg vec of A and other spkeakers vec
    for spk in spkAvgVec:
        if not spk in spkDataAvgVec:
            raise KeyError(f"{spk} not in spkDataAvgVec")

        differentSpks = list(spkDataAvgVec.keys())
        differentSpks.remove(spk)  # not calc with itself
        for spk2 in differentSpks:
            for idx, vec in enumerate(spkDataAvgVec[spk2]):
                similarity = cosine_similarity(vec, spkAvgVec[spk])
                # if similarity < 0.0:
                #     set_trace()

                #     dot_product = np.dot(vec, spkAvgVec[spk])
                #     norm_vector1 = np.linalg.norm(vec)
                #     norm_vector2 = np.linalg.norm(spkAvgVec[spk])
                #     similarity = dot_product / (norm_vector1 * norm_vector2)
                logging.info(
                    f"Smilarity between {spk} and {spk2}_{idx} is {similarity}")


def main_hubertsoft_kmeans(kmModelPath: str):
    # Hubert-soft+kMeans
    # Target spkeakers
    kmModelPath = Path(kmModelPath)
    kmModel = get_cluster_model(kmModelPath)

    spkDirs = ["data_svc/hubert/beijing_female_slice_60s",
               "data_svc/hubert/beijingmale_sing_60s",
               "data_svc/hubert/sichuan_rap_60s",
               "data_svc/hubert/using3_60s"]
    spkDirs = [Path(dir) for dir in spkDirs]

    fw = open(
        f"hubertsoft_kmeans_{kmModelPath.stem}.txt", "wt", encoding="utf8")

    spkAvgVec = {}  # 每个发音人所有素材vec的平均值。
    spkDataAvgVec = {}  # 每个发音人，每个素材vec的平均值。

    for spkDir in spkDirs:
        # 定义一个空np.array
        spkName = spkDir.name

        if not spkName in spkDataAvgVec:
            spkDataAvgVec[spkName] = np.empty((0, 256))

        vecDatas = np.empty((0, 256))
        for file in spkDir.glob("*.vec.npy"):
            vecData = np.load(file)
            vecData = get_cluster_center_result(kmModel, vecData)
            vecDatas = np.concatenate((vecDatas, vecData), axis=0)
            spkDataAvgVec[spkName] = np.concatenate(
                (spkDataAvgVec[spkName], np.mean(vecData, axis=0)[np.newaxis, :]), axis=0)
            # logging.info(f"vecDatas shape: {vecDatas.shape}")

        # Save one to spkAvgVec
        spkAvgVec[spkName] = np.mean(vecDatas, axis=0)

    # Calculate distance between one's vecs, different vec of A and avg vec of A
    for spk in spkDataAvgVec:
        if not spk in spkAvgVec:
            raise KeyError(f"{spk} not in spkAvgVec")

        for idx, vec in enumerate(spkDataAvgVec[spk]):
            similarity = cosine_similarity(vec, spkAvgVec[spk])
            logging.info(f"Smilarity between {spk} and {idx} is {similarity}")
            fw.write(f"Smilarity between {spk} and {idx} is {similarity}\n")

    logging.info("-" * 100)
    # Calculate distance between different vecs, avg vec of A and other spkeakers vec
    for spk in spkAvgVec:
        if not spk in spkDataAvgVec:
            raise KeyError(f"{spk} not in spkDataAvgVec")

        differentSpks = list(spkDataAvgVec.keys())
        differentSpks.remove(spk)  # not calc with itself
        for spk2 in differentSpks:
            for idx, vec in enumerate(spkDataAvgVec[spk2]):
                similarity = cosine_similarity(vec, spkAvgVec[spk])
                # if similarity < 0.0:
                #     set_trace()

                #     dot_product = np.dot(vec, spkAvgVec[spk])
                #     norm_vector1 = np.linalg.norm(vec)
                #     norm_vector2 = np.linalg.norm(spkAvgVec[spk])
                #     similarity = dot_product / (norm_vector1 * norm_vector2)
                logging.info(
                    f"Smilarity between {spk} and {spk2}_{idx} is {similarity}")
                fw.write(
                    f"Smilarity between {spk} and {spk2}_{idx} is {similarity}\n")


if __name__ == "__main__":
    main_hubertsoft_kmeans("chkpt/kmeans/kmeans_10000.pt")
