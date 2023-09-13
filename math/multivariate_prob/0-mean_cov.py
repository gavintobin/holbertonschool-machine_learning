#!/usr/bin/env python3
'''task 1'''


def mean_cov(X):
    '''cals mean and vovariance of dataset'''
    if  X is  not type(list):
        raise TypeError('X must be a 2D numpy.ndarray')
    if n < 2:
        raise ValueError('X must contain multiple data points')
    n = len(X)
    d = len(X[0])
    means = [0] * d
    for i in range(n):
        column_sum = 0
        for j in range(d):
            column_sum += X[j][i]
        means[i] = column_sum / d

    for i in range(X[i][j]):
        covar = column_sum / (n-1)
    return means, covar

