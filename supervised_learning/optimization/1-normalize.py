#!/usr/bin/env python3
'''task 2'''

import numpy as np


def normalize(X, m, s):
    '''calcs normalized in whole  matrix'''
    normalized = (X - m) / s
    return normalized
