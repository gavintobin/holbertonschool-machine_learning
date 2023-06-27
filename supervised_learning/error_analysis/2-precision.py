#!/usr/bin/env python3
'''task 2'''
import numpy as np


def precision(confusion):
    '''calcs sens of each class'''
    truepos = np.sum(confusion, axis=1)
    falsepos = np.sum(confusion, axis=0)
    score = truepos / (falsepos + truepos)
    return score
