#!/usr/bin/env python3
'''task 3'''
import numpy as np


def variance(X, C):
    '''total intra cluster variance of datas set'''
    if type(X) is not np.ndarray or type(C) is not np.array:
        return None

    if len(X.shape) != 2 or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C ** 2, axis=2)

    clss = np.argmin(distances, axis=1)

    var = np.sum(distances[np.arange(len(X)), clss])

    return var
