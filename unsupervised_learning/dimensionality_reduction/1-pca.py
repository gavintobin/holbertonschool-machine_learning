#!/usr/bin/env python3
'''task 2'''
import numpy as np


def pca(X, ndim):
    '''pca pt 2'''
    mean_centered_data = np.mean(X, axis=0)
    
    # do SVD
    U, S, Vt = np.linalg.svd(X - mean_centered_data)

    # keep ndim principal components
    keep = Vt[:ndim].T

    # Step 4: Transform the data to the new reduced-dimensional space
    T = np.dot(X - mean_centered_data, keep)

    return T
