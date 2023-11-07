#!/usr/bin/env python3
'''task 4'''
import numpy as np


class LSTMCell:
    '''lstm cell class'''
    def __init__(self, i, h, o):
        # Initialize weights and biases for the forget gate
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        # Initialize weights and biases for the update gate
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        # Initialize weights and biases for the intermediate cell state
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        # Initialize weights and biases for the output gate
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        # Initialize weights and biases for the output
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        '''forward func'''
        # concatenate previous hidden state and cell state with input data
        fc_concat = np.concatenate((h_prev, x_t), axis=1)

        # calc forget gate
        ft = sigmoid(np.dot(fc_concat, self.Wf) + self.bf)

        # Calc update gate
        ut = sigmoid(np.dot(fc_concat, self.Wu) + self.bu)

        # cvalc intermediate cell state
        c_hat = np.tanh(np.dot(fc_concat, self.Wc) + self.bc)

        # update cell state
        c_next = ft * c_prev + ut * c_hat

        # calc output gate
        ot = sigmoid(np.dot(fc_concat, self.Wo) + self.bo)

        # next hidden state
        h_next = ot * np.tanh(c_next)

        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y


def sigmoid(x):
    '''siggyg'''
    return 1 / (1 + np.exp(-x))


def softmax(x):
    '''softmaxy'''
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
