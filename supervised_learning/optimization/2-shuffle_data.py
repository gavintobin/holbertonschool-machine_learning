#!/usr/bin/env python3
'''task 3'''

import numpy as np


def shuffle_data(X, Y):
    '''shuffles dp in 2 matrixes in same way'''
    shuffledx = np.random.permutation(X.shape[0])
    shuffledy = np.random.permutation(Y.shape[0])
    return X[shuffledx], Y[shuffledy]
