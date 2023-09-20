#!/usr/bin/env python3
'''task 1'''
import numpy as np

def pca(X, var=0.95):
    ''' performs PCA on dataset'''
    #perform svd
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    # cvr and evr
    evr = np.cumsum(s**2) / np.sum(s**2)

    # which ones to keep
    num_components_to_keep = np.argmax(evr >= var) + 1

    # Reduce dimensions and get w matrix
    W = vt[:, :num_components_to_keep + 1].T

    return W
