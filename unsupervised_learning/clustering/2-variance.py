#!/usr/bin/env python3
'''task 3'''
import numpy as np


def variance(X, C):
    '''total intra cluster variance of datas set'''
    if X.shape[1] != C.shape[1]:
        return None

    n, d = X.shape
    k = C.shape[0]

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)**2

    clss = np.argmin(distances, axis=1)

    var = np.sum(distances[np.arange(n), clss])

    return var