#!/usr/bin/env python3
'''task 1'''
import numpy as np


def initialize(X, k):
    '''initializes centroid or  centr of cluster'''
    if type(X) is not np.ndarray or if type(k) is not int:
        return None
    if k <= 0 or k >= X.shape[0]:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    centroids = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))
    return centroids
