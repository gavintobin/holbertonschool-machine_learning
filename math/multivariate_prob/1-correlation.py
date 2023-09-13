#!/usr/bin/env python3
'''task 2'''
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


def correlation(C):
    '''calcs coreelation msatrix'''
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Get the shape of the array
    d, d = C.shape

    # Check if there are multiple data points
    if d < 2:
        raise ValueError("X must contain multiple data points")

    std_dev = np.sqrt(np.diagonal(C))

    # Calculate the correlation matrix
    d = C.shape[0]  # Number of dimensions
    correlation_matrix = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            correlation_matrix[i, j] = C[i, j] / (std_dev[i] * std_dev[j])

    return correlation_matrix
