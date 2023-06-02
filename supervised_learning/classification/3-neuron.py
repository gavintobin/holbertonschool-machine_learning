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

    def cost(self, Y, A):
        '''calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1.0000001 - A) * np.log(1.0000001 - A))
        return cost


    def forward_prop(self, X):
        '''forward prop function'''
        self.__A = self.sig(np.matmul(self.__W, X) + self.__b)
        return (self.__A)

    def sig(self, x):
        '''sigmoid helper func'''
        return 1/(1 + np.exp(-x))

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
