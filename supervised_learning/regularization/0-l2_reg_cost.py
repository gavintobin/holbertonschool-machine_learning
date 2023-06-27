#!/usr/bin/env python3
'''task 1'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''cost using l2 reg'''
    regcost = 0

    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        regcost += np.sum(np.square(W))

    regcost *= (lambtha / (2 * m))
    total = cost + regcost
    return total
