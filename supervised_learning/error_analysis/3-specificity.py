#!/usr/bin/env python3
'''task 4'''
import numpy as np


def specificity(confusion):
    '''calcs spec for each class'''
    TP = np.diag (confusion)
    FP = confusion.sum (axis=0) - TP
    FN = confusion.sum (axis=1) - TP
    TNV = confusion.values() - (FP + FN + TP)
    TN = sum(TNV)
    spec = TN / (TN + FP)

    return spec
