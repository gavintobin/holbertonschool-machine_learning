#!/usr/bin/env python3
'''task4'''
import numpy as np


def posterior(x, n, P, Pr):
    """
    Posterior Calculation
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')

    err = 'x must be an integer that is greater than or equal to 0'

    if not isinstance(x, int) or x < 0:
        raise ValueError(err)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')


    fact = np.math.factorial
    #calulates binomial coeefficient
    bico = fact(n) / (fact(x) * fact(n - x))
    succprob = (P ** x)
    failprob = ((1 - P) ** (n - x))
    intersection_values = Pr * bico * succprob * failprob
    post = intersection_values / np.sum(intersection_values)
    return post
