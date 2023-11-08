#!/usr/bin/env python3
'''task 1'''

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    ''' forward prop for deep rnn'''
    t, m, i = X.shape
    l = len(rnn_cells)
    h = h_0.shape[2]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            c = rnn_cells[layer]
            h_prev = H[step, layer]
            h_next, y = c.forward(h_prev, x_t)
            H[step + 1][layer] = h_next
            x_t = h_next
    return H, Y
