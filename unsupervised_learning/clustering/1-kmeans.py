#!/usr/bin/env python3
'''task 1'''
import numpy as np


def initialize(X, k):
    '''initializes centroid or  centr of cluster'''
    if type(X) is not np.ndarray or type(k) is not int:
        return None
    if k <= 0 or k >= X.shape[0]:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    centroids = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))
    return centroids

def kmeans(X, k, iterations=1000):
    '''perfos k mean'''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    if type(k) is not int or X.shape[0] <= k or k <= 0:
        return None, None
    centroids = initialize(X, k)

    #  figure out  distance between data points and centroids
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    # Assign each data point to the cluster  the closest centroid
    clss = np.argmin(distances, axis=1)

    for _ in range(iterations):
        # Update cluster centroids based on the mean of assigned data points
        new_C = np.array([X[clss == i].mean(axis=0) if len(X[clss == i])>0 else initialize(X,1).reshape(1,1) for i in range(k)])

        #checks  convergence
        if np.all(centroids == new_C):
            return new_C, clss
        centroids = new_C

        #  figure out  distance between data points and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
        # Assign each data point to the cluster  the closest centroid
        clss = np.argmin(distances, axis=1)

    return centroids, clss
