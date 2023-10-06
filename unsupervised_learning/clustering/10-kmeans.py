#!/usr/bin/env python3
'''task1'''
import sklearn.cluster


def kmeans(X, k):
    '''kmean easy way'''
    centroids, clss, _ = sklearn.cluster.k_means(X, k)

    return centroids, clss
