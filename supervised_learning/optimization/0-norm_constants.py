#!/usr/bin/env python3
'''task 1'''

import numpy as np


def normalization_constants(X):
    '''calcs normalized constants in matrix'''
    mean = np.mean(X, axis=1)
    stddev = np.std(X, axis=1)
    return mean, stddev
