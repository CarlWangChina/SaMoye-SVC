import argparse
import logging
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import (KMeans, MiniBatchKMeans)

from kmeans import KMeansGPU

from ipdb import set_trace
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# gpu_minibatch真拉，虽然库支持但是也不考虑
def train_cluster(in_dir, n_clusters, use_minibatch=True, verbose=False, use_gpu=False):
    features = []

    Files = in_dir.rglob("*.vec.npy")
    for npyFile in tqdm(in_dir.rglob("*.vec.npy"), desc="Loading features", unit="file", total=len(list(Files))):
        features.extend(np.load(npyFile))

    features = np.array(features).astype(np.float32)
    print(features.nbytes / 1024**2,
          "MB , shape:", features.shape, features.dtype)
    logger.info(f"Clustering features of shape: {features.shape}")

    t = time.time()
    if (use_gpu is False):
        if use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, verbose=verbose, batch_size=4096, max_iter=80).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters,
                            verbose=verbose).fit(features)
    else:
        kmeans = KMeansGPU(n_clusters=n_clusters, mode='euclidean',
                           verbose=2 if verbose else 0, max_iter=500, tol=1e-2)
        features = torch.from_numpy(features)  # .to(device)
        kmeans.fit_predict(features)

    print(time.time() - t, "s")

    x = {
        "n_features_in_": kmeans.n_features_in_ if use_gpu is False else features.shape[1],
        "_n_threads": kmeans._n_threads if use_gpu is False else 4,
        "cluster_centers_": kmeans.cluster_centers_ if use_gpu is False else kmeans.centroids.cpu().numpy(),
        "n_clusters": n_clusters,
    }
    print("end")

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default="./dataset/44k",
                        help='path of training data directory')
    parser.add_argument('--output', type=Path, default="logs/44k",
                        help='path of model output directory')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='to use GPU')
    args = parser.parse_args()

    checkpoint_dir = Path(args.output)
    dataset = Path(args.dataset)
    use_gpu = args.gpu
    n_clusters = 10000

    x = train_cluster(dataset, n_clusters, use_minibatch=True,
                      verbose=False, use_gpu=use_gpu)

    checkpoint_path = checkpoint_dir / f"kmeans_{n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(x, checkpoint_path)
