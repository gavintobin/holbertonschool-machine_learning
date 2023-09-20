#!/usr/bin/env python3
'''task 2'''
import numpy as np


def pca(X, ndim):
    '''pca pt 2'''
    mean_centered_data = X - np.mean(X, axis=0)
    cov_mat = np.cov(mean_centered_data, rowvar=False)

    # do SVD
    U, S, _ = np.linalg.svd(cov_mat)

    # keep ndim principal components
    keep = U[:, :ndim]

    # Step 4: Transform the data to the new reduced-dimensional space
    T = np.dot(mean_centered_data, keep)

    return T
