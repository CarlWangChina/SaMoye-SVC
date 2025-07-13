import numpy as np
import torch
import sys
sys.path.append(".")
from pathlib import Path

import cluster

from ipdb import set_trace


if __name__ == "__main__":
    ReadPath = Path("chkpt/kmeans/kmeans_900.pt")
    clusterModel = cluster.get_cluster_model(ReadPath)

    testHubertFeat = np.load("data_svc/hubert/baorong/000001_0.vec.npy")

    cluster_c = cluster.get_cluster_center_result(
        clusterModel, testHubertFeat)
    clusterdata = cluster.get_cluster_result(clusterModel, testHubertFeat)
    cluster_c = torch.FloatTensor(cluster_c)
    set_trace()
    print(cluster_c)
