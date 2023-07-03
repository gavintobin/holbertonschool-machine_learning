#!/usr/bin/env python3
'''task 2'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''updates w and b using gd w l2 regularization'''
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        if i == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - np.power(A, 2))

        W = weights['W' + str(i)]

        dA = np.matmul(W.T, dZ)

        j = ((lambtha / m) * W)
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + j
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

    return weights
