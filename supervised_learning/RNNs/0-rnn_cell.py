#!/usr/bin/env python3
'''taslk 1'''
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        # Concatenate previous hidden state and input data
        hx_concat = np.concatenate((h_prev, x_t), axis=1)

        # calc the next hidden state using tanh
        h_next = np.tanh(np.dot(hx_concat, self.Wh) + self.bh)

        # calc the output using softmax
        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, y
