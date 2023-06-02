#!/usr/bin/env python3
'''neuron class that defines singleneuron performing binary classification'''
import numpy as np


class Neuron:
    '''neauron class'''
    def __init__(self, nx):
        '''innit'''
        self.__A = 0
        self.__b = 0
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.randn(1, nx)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calcs gd"""
        m = Y.shape[1]
        self.forward_prop(X)
        dZ = A - Y
        dW = (1/m) * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def cost(self, Y, A):
        '''calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def forward_prop(self, X):
        '''forward prop function'''
        self.__A = self.sig(np.matmul(self.__W, X) + self.__b)
        return (self.__A)

    def sig(self, x):
        '''sigmoid helper func'''
        return 1/(1 + np.exp(-x))

    def evaluate(self, X, Y):
        '''evaluates the predictions'''
        m = X.shape[1]
        self.forward_prop(X)
        pred = np.where(self.__A > 0.5, 1, 0)
        cst = self.cost(Y, self.__A)
        return pred, cst

    @property
    def W(self):
        '''gets weight'''
        return self.__W

    @property
    def b(self):
        '''gets bias'''
        return self.__b

    @property
    def A(self):
        '''gets a out'''
        return self.__A
