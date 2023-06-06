#!/usr/bin/env python3
'''hot encode'''
import numpy as np

def one_hot_encode(Y, classes):
    """hes on fiyaaaa"""
    if not isinstance(Y, np.ndarray) or len(Y) == 0 or not isinstance(classes, int) or classes <= 0:
        return None
    m = Y.shape[0]
    for i in range(m):
        one_hot = np.zeros((classes, len(Y)))
        one_hot[Y, np.arange(len(Y))] = 1

    return one_hot
