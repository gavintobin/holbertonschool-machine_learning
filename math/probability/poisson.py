#!/usr/bin/env python3
"""class poisson"""
pi = 3.1415926536
e = 2.7182818285

class Poisson:
    '''poisson class'''
    def __init__(self, data=None, lambtha=1.):
        """innit"""
        self.lambtha = float(lambtha)
        if data  is None:
            if lambtha <= 0:
                raise ValueError('data must be a positive value')
        if data is not None:
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            if type(data) != list:
                raise TypeError('data must be a list')
            else:
                self.lambtha = sum(data) / len(data)


'''def cdf(self, k):
    """calculatew value of the cdf for given number of successess"""
    if type(k) != int:'''
     
        
