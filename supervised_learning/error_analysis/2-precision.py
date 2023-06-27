#!/usr/bin/env python3

'''task 2'''
import numpy as np


def precision(confusion):
    '''calcs sens of each class'''
    true = np.diag(confusion)
    false = []
    for i in range(confusion):
        newfalse = false.append(sum(true[:, i]) - true[i, i])
    ppv = true / (true + newfalse)
    return ppv
