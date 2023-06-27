#!/usr/bin/env python3
'''task 4'''
import numpy as np


def specificity(confusion):
    '''calcs spec for each class'''
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TNV = np.sum(confusion) - (FP + FN + TP)
    spec = TNV / (TNV + FP)

    return spec
