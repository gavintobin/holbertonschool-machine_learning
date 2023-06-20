#!/usr/bin/env python3
'''task 8'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''update vars using alg'''
    um = (beta2 * s + ((grad ** 2) * (1 - beta2)))
    uv = (var - (alpha * (grad / ((um ** (1/2)) + epsilon))))

    return uv, um
