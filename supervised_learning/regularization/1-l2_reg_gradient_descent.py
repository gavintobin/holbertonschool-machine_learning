#!/usr/bin/env python3
'''task 2'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''updates w and b using gd w l2 regularization'''
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        W = weights["W" + str(i)]
        aprev = cache['A' + str(i - 1)]
        dw = (1 / m) + np.dot(dz, aprev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if i > 1:
            da = np.dot(W.T, dz)
            dz = da + (1 - cache['A' + str(i - 1)]**2)
        W -= alpha * dw
        weights['b' + str(i)] -= alpha * db
