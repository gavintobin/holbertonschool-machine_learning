#!/usr/bin/env python3
'''task 5'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''does fp using dropout'''
    cache = {}
    cache['A0'] = X
    
    for l in range(1, L + 1):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        
        if l < L:
            D = np.random.rand(Z.shape[0], Z.shape[1])
            D = D < keep_prob
            A = np.multiply(np.tanh(Z), D) / keep_prob
            cache['D' + str(l)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
    
    return cache
