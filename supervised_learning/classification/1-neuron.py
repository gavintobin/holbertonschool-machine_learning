#!/usr/bin/env python3
'''neauron class that defines singleneuron performing binary classification'''
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
