#!/usr/bin/env python3
'''task 4'''
import numpy as np


def specificity(confusion):
    '''calcs spec for each class'''
    FP = confusion.sum (axis=0) - TP
    FN = confusion.sum (axis=1) - TP
    TP = np.diag (confusion)
    TN = confusion.values.sum () - (FP + FN + TP)
    spec = TN / (TN + FP)

    return spec
