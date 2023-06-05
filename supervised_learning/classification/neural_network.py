#!/usr/bin/env python3
'''network with single hidden layer'''

import numpy as np

class NeuralNetwork:
    '''sigle layer network class'''
    def __init__(self, nx, nodes):
        self.W1 = np.random.randn(1, nx)
        self.b1 = 0
        self.A1 = 0
        self.W2 = np.random.randn(1, nx)
        self.b2 = 0
        self.A2 = 0
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
