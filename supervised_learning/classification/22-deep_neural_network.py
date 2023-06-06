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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        return self.evaluate(X, Y)

    def evaluate(self, X, Y):
        '''evaluates the predictions'''
        A, _ = self.forward_prop(X)
        pred = np.where(A > 0.5, 1, 0)
        cst = self.cost(Y, A)
        return pred, cst

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calcs gd"""
        m = Y.shape[1]
        L = self.__L

        A = cache["A" + str(L)]
        dZ = A - Y

        for l in range(L, 0, -1):
            A_prev = cache["A" + str(l - 1)]
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db

            if l > 1:
                dZ = dA * (A_prev * (1 - A_prev))

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

