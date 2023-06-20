#!/usr/bin/env python3
'''task 6'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''updates vaiables w gd'''
    um = beta1 + v + (grad * (1 - beta1))
    uv = var- (alpha * um)
    return uv, um
