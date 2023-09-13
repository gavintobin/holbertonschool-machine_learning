#!/usr/bin/env python3
'''task 3'''
import numpy as np


class MultiNormal:
    def __init__(self, data):
        # Check if data is a 2D numpy.ndarray
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Get the shape of the array
        d, n = data.shape

        # Check if there are multiple data points
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate the covariance matrix
        self.cov = self.cov(data)


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
