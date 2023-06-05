#!/usr/bin/env python3
'''single layer network'''

import numpy as np


class NeuralNetwork:
    '''class'''
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calcs gd"""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2)
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        m = Y.shape[1]

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1)
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def evaluate(self, X, Y):
        '''evaluates the predictions'''
        m = X.shape[1]
        A = self.forward_prop(X)[1]
        pred = np.where(A > 0.5, 1, 0)
        cst = self.cost(Y, A)
        return pred, cst

    def cost(self, Y, A):
        '''calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def forward_prop(self, X):
        '''forward prop function'''
        self.__A1 = self.sig(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = self.sig(np.matmul(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def sig(self, x):
        '''sigmoid helper func'''
        return 1/(1 + np.exp(-x))

    @property
    def W1(self):
        '''gets weight'''
        return self.__W1

    @property
    def b1(self):
        '''gets bias'''
        return self.__b1

    @property
    def A1(self):
        '''gets a out'''
        return self.__A1

    @property
    def W2(self):
        '''gets weight'''
        return self.__W2

    @property
    def b2(self):
        '''gets bias'''
        return self.__b2

    @property
    def A2(self):
        '''gets a out'''
        return self.__A2
