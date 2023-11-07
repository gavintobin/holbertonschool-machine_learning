#!/usr/bin/env python3
'''task 2'''
import numpy as np


def rnn(rnn_cell, X, h_0):
    '''rnn forward prop'''
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0

    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return H[1:], Y
