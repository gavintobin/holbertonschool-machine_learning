#!/usr/bin/env python3

'''task 2'''
import numpy as np


def precision(confusion):
    '''calcs sens of each class'''
    true = np.diag(confusion)
    false = np.sum(confusion, axis=0)
    ppv = true / false
    return ppv
