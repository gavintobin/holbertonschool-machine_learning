#!/usr/bin/env python3
'''task 1'''
import numpy as np


def initialize(X, k):
    '''initializes centroid or  centr of cluster'''
    if k <= 0 or k > X.shape[0]:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    centroids = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))
    return centroids

def kmeans(X, k, iterations=1000):
    '''performs k mean'''
    if k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    centroids = initialize(X, k)
    #calc distances between data ppoints and cluster centroid
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis, :] - centoids, axis=2)

        clss = np.argmin(X, axis=1)
    #updated centroids based off mean
        for clss in range(k):
            updated = np.array(X.mean(axis=0))
        empty = np.isnan(updated).any(axis=1)
        if empty.any():
            updated[empty] = initialize(X, empty.sum())
        if np.all(centroids = updated):
            return centroids, clss

        centroids = C
        C = updated
        return C, clss
