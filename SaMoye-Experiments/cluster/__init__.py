import torch
import numpy as np
from sklearn.cluster import KMeans


def get_cluster_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    km = KMeans(checkpoint["n_clusters"])
    km.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
    km.__dict__["_n_threads"] = checkpoint["_n_threads"]
    km.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"]
    return km


def get_cluster_result(model, x):
    """
        x: np.array [t, 256]
        return cluster class result
    """
    return model.predict(x)


def get_cluster_center_result(model, x):
    """x: np.array [t, 256]"""
    if isinstance(x, list):
        x = np.array(x)
    predict = model.predict(x)
    return model.cluster_centers_[predict]


# def get_center(model, x, speaker):
#     return model[speaker].cluster_centers_[x]
