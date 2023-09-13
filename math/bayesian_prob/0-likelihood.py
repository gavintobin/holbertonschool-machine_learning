#!/usr/bin/env python3
'''task 1'''
import numpy as np


def likelihood(x, n, P):
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is an integer greater than or equal to 0 and not greater than n
    if not isinstance(x, int) or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        raise ValueError('x cannot be greater than n')
    # Check if P is a list
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in the range [0, 1]
    if any(p < 0 or p > 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the likelihood for each probability in P
     # Calculate the likelihood for each probability in P
    likelihoods = np.array([(np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x))) * (p ** x) * ((1 - p) ** (n - x)) for p in P])

    return likelihoods
