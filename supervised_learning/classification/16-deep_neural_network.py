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
        if type(layers) is not list or len(layers) < 1:
            raise TypeError('layers must be a list of postive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev = nx
        for i in range(self.L):
            self.weights['W' + str(i + 1)] = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

