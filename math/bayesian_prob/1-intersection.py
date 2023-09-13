#!/usr/bin/env python3
'''task 1'''
import numpy as np


def intersection(x, n, P, Pr):
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

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    intersection_values = P * Pr

    return intersection_values
