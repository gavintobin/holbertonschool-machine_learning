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
        self.cov = self.calculate_covariance(data)

    def calculate_covariance(self, data):
        '''calc koe'''
        d, n = data.shape
        cov = np.zeros((d, d))

        for i in range(d):
            for j in range(d):
                # Calculate the covariance between dimensions i and j
                xi = data[i, :]
                xj = data[j, :]
                nn = (n - 1)
                cov[i, j] = np.dot(xi - self.mean[i], xj - self.mean[j]) / nn

        return cov
