#!/usr/bin/env python3 
'''task 10'''
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update Variables For Adam"""
    v = ((beta1 * v) + ((1 - beta1) * grad))
    s = ((beta2 * s) + ((1 - beta2) * grad ** 2))
    newV = (v / (1 - beta1 ** t))
    newS = (s / (1 - beta2 ** t))
    var = var - alpha * newV / (np.sqrt(newS) + epsilon)

    return var, v, s
