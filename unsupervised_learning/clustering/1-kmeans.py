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
    centroids = initialize(X, k)

    for _ in range(iterations):
        #  figure out  distance between data points and centroids
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)

        # Assign each data point to the cluster with the closest centroid
        clss = np.argmin(distances, axis=1)

        # Update cluster centroids based on the mean of assigned data points
        new_C = np.array([X[clss == i].mean(axis=0) for i in range(k)])

        # Handle clusters with no data points by reinitializing their centroids
        empty_clusters = np.isnan(new_C).any(axis=1)
        if empty_clusters.any():
            new_C[empty_clusters] = initialize(X, empty_clusters.sum())
            #checks for convergence
        if np.all(centroids == new_C):
            return C, clss
        C = new_C

    return C, clss
