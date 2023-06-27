#!/usr/bin/env python3
'''task 2'''
import numpy as np


def sensitivity(confusion):
    '''calcs sens of each class'''
    true = np.diag(confusion)
    actual = np.sum(confusion, axis=1)
    sens = true / actual

    return sens
