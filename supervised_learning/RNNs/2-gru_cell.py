#!/usr/bin/env python3
'''task 2'''
import numpy as np


class GRUCell:
    '''grucell class'''
    def __init__(self, i, h, o):
        # update gate
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        # reset gate
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        # intermediate hidden state
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        #  weights and biases for output
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        '''forward  prop'''
        zx_concat = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(np.dot(zx_concat, self.Wz) + self.bz)

        rx_concat = np.concatenate((h_prev, x_t), axis=1)
        r = sigmoid(np.dot(rx_concat, self.Wr) + self.br)

        hx_concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.dot(hx_concat, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_hat

        # calc the output using softmax activation
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y


def sigmoid(x):
    '''sigmoid helper'''
    return 1 / (1 + np.exp(-x))


def softmax(x):
    '''softmax helper'''
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
