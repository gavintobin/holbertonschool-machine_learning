#!/usr/bin/env python3
'''task 8''''

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    '''forward prop for bi directional rnn'''
    t, m, _ = X.shape
    h = h_0.shape[1]
    H = np.zeros((t, m, 2 * h))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    h_fwd = h_0
    h_bwd = h_t

    for step in range(t):
        x_t = X[step]

        # Calc the forward hidden state
        h_fwd = bi_cell.forward(h_fwd, x_t)

        # Calcthe backward hidden state
        x_bwd = X[t - step - 1]
        h_bwd = bi_cell.backward(h_bwd, x_bwd)

        # Concatenate the forward and backward hidden states
        H[step] = np.concatenate((h_fwd, h_bwd), axis=1)

        # Calc the output for the current time step
        Y[step] = bi_cell.output(H[step])

    return H, Y
