#!/usr/bin/env python3
'''task 1'''
import numpy as np

def pca(X, var=0.95):
    ''' performs PCA on dataset'''
    #perform svd
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    explained_variance_ratio = s / np.sum(s)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # which ones to keep
    num_components_to_keep = np.argmax(cumulative_variance_ratio >= var) + 1

    # Reduce dimensions and get w matrix
    W = u[:, :num_components_to_keep]

    return W
