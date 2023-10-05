#!/usr/bin/env python3
'''task12'''
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    '''does aglo clust'''
    lm = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(lm, t=dist, criterion='distance')

    scipy.cluster.hierarchy.dendrogram(lm, color_threshold=dist)

    plt.show()

    return clss
