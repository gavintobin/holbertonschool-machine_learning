#!/usr/bin/env python3
'''task 1'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def initialize(X, k):
    '''initializes centroid or  centr of cluster'''
    n, d = X.shape
    min = np.min(d)
    max = np.max(d)

    centroids = np.random.uniform(low=min, high=max, size=k)
    return centroids
