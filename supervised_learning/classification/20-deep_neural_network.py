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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(i + 1)] = w
            else:
                prev = layers[i-1]
                w = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
                self.__weights['W' + str(i + 1)] = w
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    def evaluate(self, X, Y):
        '''evaluates the predictions'''
        A, _ = self.forward_prop(X)
        pred = np.where(A > 0.5, 1, 0)
        cst = self.cost(Y, A)
        return pred, cst

    def cost(self, Y, A):
        '''calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def sig(self, x):
        '''sigmoid helper func'''
        return 1/(1 + np.exp(-x))

    def forward_prop(self, X):
        '''f prop func'''
        A = X
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W_curr = self.__weights['W' + str(i)]
            b_curr = self.__weights['b' + str(i)]
            Z = np.matmul(W_curr, A) + b_curr
            A = self.sig(Z)
            self.__cache['A' + str(i)] = A
        return (A, self.__cache)

    @property
    def L(self):
        """layer getter"""
        return self.__L

    @property
    def cache(self):
        '''itermed val getter'''
        return self.__cache

    @property
    def weights(self):
        '''weight getter'''
        return self.__weights
