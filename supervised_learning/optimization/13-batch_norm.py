#!/usr/bin/env python3
'''task 13'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''normalizes unactivated output of nn'''
    var = np.var(Z, axis=0)
    mean = np.mean(Z, axis=0)
    normz = (Z - mean) / np.sqrt(var + epsilon)
    ZZ = gamma * normz + beta
    return ZZ
