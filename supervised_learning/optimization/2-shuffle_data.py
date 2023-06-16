#!/usr/bin/env python3
'''task 3'''

import numpy as np


def shuffle_data(X, Y):
    '''shuffles dp in 2 matrixes in same way'''
    shuffledx = np.random.permutation(X, axis=1)
    shuffledy = np.random.permutation(Y, axis=1)
    return shuffledx, shuffledy
