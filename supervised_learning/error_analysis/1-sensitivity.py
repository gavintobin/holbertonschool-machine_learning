#!/usr/bin/env python3
'''task 2'''
import numpy as np


def sensitivity(confusion):
    '''calcs sens of each class'''
    tpr = np.diag(confusion)
    total  =np.sum(confusion)
    proba = tpr / total
