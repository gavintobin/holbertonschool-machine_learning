#!/usr/bin/env python3
'''task 3'''

import numpy as np


def shuffle_data(X, Y):
    '''shuffles dp in 2 matrixes in same way'''
    i = np.random.permutation(X.shape[0])
    return X[i], Y[i]
