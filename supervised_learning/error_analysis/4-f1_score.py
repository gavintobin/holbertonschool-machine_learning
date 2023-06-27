#!/usr/bin/env python3
'''task 5'''
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision



def f1_score(confusion):
    '''calcs f1 score'''
    numer = sensitivity(confusion) * precision(confusion)
    denom = sensitivity(confusion) + precision(confusion)
    f1 = 2 * (numer / denom)
    return f1

