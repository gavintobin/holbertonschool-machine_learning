#!/usr/bin/env python3
'''task 5'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''does fp using dropout'''
    cache = {}
    cache['A0'] = X
    A_prev = cache['A' + str(i - 1)]
    
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            dropout = np.random.binomial(n=1, p=keep_prob, size=A.shape)
            A = A * dropout / keep_prob
            cache['D' + str(i)] = dropout

        cache['A' + str(i)] = A

    return cache
