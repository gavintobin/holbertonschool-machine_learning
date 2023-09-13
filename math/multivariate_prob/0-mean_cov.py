#!/usr/bin/env python3
'''task 1'''
import numpy as np

def mean_cov(X):
    '''calc mean and covar '''
    # Check if X is a 2D numpy.ndarray
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Get the shape of the array
    n, d = X.shape

    # Check if there are multiple data points
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate the covariance matrix
    cov = np.zeros((d, d))
    for i in range(n):
        deviation = X[i:i+1, :] - mean
        cov += np.dot(deviation.T, deviation)

    cov /= (n - 1)

    return mean, cov
