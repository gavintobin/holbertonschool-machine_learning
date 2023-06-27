#!/usr/bin/env python3
'''task 4'''
import numpy as np


def specificity(confusion):
    '''calcs spec for each class'''
    truepos = np.diag(confusion)
    falseneg = confusion.sum(axis=1) - truepos
    spec = truepos / (truepos + falseneg)
    return spec
