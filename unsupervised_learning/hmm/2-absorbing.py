#!/usr/bin/env python3
'''task 1'''
import numpy as np


def absorbing(P):
    '''determines if teh mc is absorbing'''
    n = P.shape[0]

    if P.shape != (n, n):
        return False

    for i in range(n):
        if np.all(P[i] == 0) and P[i, i] == 1:
            return True

    return False
