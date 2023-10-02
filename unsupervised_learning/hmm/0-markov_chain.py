#!/usr/bin/env python3
'''task 1'''
import numpy as np


def markov_chain(P, s, t=1):
    '''determines probability  of a markof chain being in particular state'''
    n = P.shape[0]


    if P.shape != (n, n) or s.shape != (1, n) or not isinstance(t, int) or t < 0:
        return None

    ptrans = np.linalg.matrix_power(P, t)

    sprobs = s.dot(ptrans)

    return sprobs
