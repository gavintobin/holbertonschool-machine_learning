#!/usr/bin/env python3
'''task 3'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''test optimum number of clusters'''
    if kmax is None:
        kmax = X.shape[0] // 2

    if kmin < 1 or kmax < kmin:
        return None, None

    results = []
    dvars = []
    vars = []

    for k in range(kmin, kmax + 1):
        C, distances = kmeans(X, k, iterations)
        vars.append(variance(X, C))
        results.append((C, distances))

    for var in vars:
        d_vars.append(vars[0] - var)
    
    return results, dvars
