#!/usr/bin/env python3
'''task 3'''
import numpy as np
import scipy.linalg


class MultiNormal:
    '''multi class'''
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

        try:
            self.L = scipy.linalg.cholesky(self.cov, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite.")

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
    def pdf(self, x):
        # Check if x is a numpy.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Check if x has the correct shape (d, 1)
        d, _ = self.mean.shape
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculate the Mahalanobis distance
        diff = x - self.mean
        manlob_dist = np.linalg.solve(self.L, diff)
        manlob_dist = np.dot(manlob_dist.T, manlob_dist)

        # Calculate the PDF
        pdf = np.exp(-0.5 * manlob_dist) / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov)))

        return pdf
