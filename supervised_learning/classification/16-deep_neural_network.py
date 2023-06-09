#!/usr/bin/env python3
"""deep neural network w binary classif."""
import numpy as np

class DeepNeuralNetwork:
    '''dnn class'''
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

