#!/usr/bin/env python3
'''task 5'''

import numpy as np


class BidirectionalCell:
    '''bi dir cell class of RNN '''
    def __init__(self, i, h, o):
        #  weights and biases for  forward
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))

        #  weights and biases for backward
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))

        #  weights and biases for the outputs
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        '''forward prop'''
        # Calc the hidden state in the forward direction
        hf_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(hf_concat, self.Whf) + self.bhf)
        return h_next
