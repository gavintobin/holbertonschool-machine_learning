#!/usr/bin/env python3
'''neauron class that defines singleneuron performing binary classification'''
import numpy as np


class Neuron:
    '''neauron class'''
    def __init__(self, nx):
        '''innit'''
        self.__A = 0
        self.__b = 0
        self.__W = np.random.randn(1, nx)
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
  
    def get_weight(self):
        '''gets weight'''
        return self.__W

    def get_bias(self):
        return self.__b

    def get_actout(self):
        return self.__A
