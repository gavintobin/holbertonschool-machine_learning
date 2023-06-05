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
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of postive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i, layer_size in enumerate(layers):
            if i == 0:
                input_size = nx
            else:
                input_size = layers[i-1]
                w = np.random.randn(layer_size, input_size) * np.sqrt(2 / input_size)
                self.weights['W{}'.format(i + 1)] = w
                self.weights['b'.format(i + 1)] = np.zeros((layer_size, 1))
