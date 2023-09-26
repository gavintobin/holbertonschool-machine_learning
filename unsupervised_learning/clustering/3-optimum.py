#!/usr/bin/env python3
'''task 3'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variancee


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''test optimum number of clusters'''
    if kmax is None:
        kmax = X.shape[0] // 2

    if kmin < 1 or kmax < kmin:
        return None, None

    results = []
    dvars = []

    for k in range(kmin, kmax + 1):
        C, distances = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, distances))
        if k == kmin:
            base = var
            dvars.append(base - var)

    return results, dvars


def kmeans(X, k, iterations=1000):
    '''perfos k mean'''
    centroids = initialize(X, k)

    for _ in range(iterations):
        #  figure out  distance between data points and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign each data point to the cluster  the closest centroid
        clss = np.argmin(distances, axis=1)

        # Update cluster centroids based on the mean of assigned data points
        new_C = np.array([X[clss == i].mean(axis=0) for i in range(k)])

        # Handle clusters  no data points by reinitializing their centroids
        empty_clusters = np.isnan(new_C).any(axis=1)
        if np.any(empty_clusters):
            new_C[empty_clusters] = initialize(X, empty_clusters.sum())
            #checks  convergence
        if np.all(centroids == new_C):
            return new_C, clss
        centroids = new_C

    return centroids, clss



def initialize(X, k):
    '''initializes centroid or  centr of cluster'''
    if k <= 0 or k > X.shape[0]:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    centroids = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))
    return centroids
