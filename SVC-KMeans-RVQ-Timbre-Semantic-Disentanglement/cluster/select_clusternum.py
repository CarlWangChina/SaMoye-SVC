""" Select best K-means cluster through elbow method(SSE curve)
"""

import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ipdb import set_trace

from kmeans import KMeansGPU

# Define logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    # Set default device to while GPU
    torch.cuda.set_device(0)

    # Define start end, steps
    start, end, step = 500, 5600, 100

    # Load hubert-soft features
    ReadDir = args.dataset
    saveDir = args.output
    saveDir.mkdir(exist_ok=True, parents=True)
    features = []

    for npyFile in tqdm(ReadDir.rglob("*.vec.npy"), desc="Loading features", unit="file", total=len(list(ReadDir.rglob("*.vec.npy")))):
        features.extend(np.load(npyFile))

    # list to np.array
    features = np.array(features).astype(np.float32)
    logger.info(
        f"feature memory useage={features.nbytes / 1024**2}MB, shape={features.shape}, dtype={features.dtype}")
    logger.info(f"Clustering features of shape: {features.shape}")

    if args.gpu:
        features = torch.from_numpy(features).cuda()

    # 手肘图法2——基于SSE
    distortions = []  # 用来存放设置不同簇数时的SSE值
    for i in range(start, end, step):
        if args.gpu:
            kmeans = KMeansGPU(n_clusters=i,
                               mode="euclidean",
                               verbose=2,
                               max_iter=500,
                               tol=1e-2)
            _, inertia = kmeans.fit_predict(features)
            logger.info(f"K-means with {i} clusters, inertia={inertia}")
        else:
            kmeans = KMeans(n_clusters=i,
                            init="k-means++",
                            verbose=True,
                            max_iter=500,
                            tol=1e-2)
            kmeans.fit(features)
        # 获取K-means算法的SSE
        distortions.append(inertia.item() if args.gpu else kmeans.inertia_)

        # Save K-means model
        x = {
            "n_features_in_": kmeans.n_features_in_ if args.gpu is False else features.shape[1],
            "_n_threads": kmeans._n_threads if args.gpu is False else 4,
            "cluster_centers_": kmeans.cluster_centers_ if args.gpu is False else kmeans.centroids.cpu().numpy(),
            "n_clusters": i,
        }
        torch.save(x, saveDir / f"kmeans_select_{i}.pt")
        logger.info(
            f"Save K-means model with {i} clusters to {saveDir / f'kmeans_select_{i}.pt'}")

    # 绘制曲线
    plt.plot(range(start, end, step), distortions, marker="o")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("cluster number")
    plt.ylabel("SSE")
    # 保存图片
    plt.savefig("./SSE_curve.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default="./data_svc/hubert",
                        help='path of training data directory')
    parser.add_argument('--output', type=Path, default="chkpt/kmeans_select",
                        help='path of model output directory')
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
