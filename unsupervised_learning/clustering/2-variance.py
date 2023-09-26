#!/usr/bin/env python3
'''task 3'''
import numpy as np


def variance(X, C):
    '''total intra cluster variance of datas set'''
    if X.shape[1] != C.shape[1]:
        return None

    k = C.shape[0]
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)**2
    clss = np.argmin(distances, axis=1)
    var = 0.0
    for i in range(k):
        cluster_points = X[clss == i]
        cluster_center = C[i]
        var += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1)**2)
    return var
