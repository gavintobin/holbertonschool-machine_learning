#!/usr/bin/env python3
'''task 6'''
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''drop out reg using gd'''
    m = Y.shape[1]

    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        if i == L:
            dz = A - Y
        else:
            dz = da * (1 - (A ** 2))
            dz = (dz * cache['D' + str(i)] / keep_prob)
    
        W = weights['W' + str(i)]
        da = np.matmul(W.T, dz)
        dw = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
    return weights
