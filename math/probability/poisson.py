#!/usr/bin/env python3
"""class poisson"""
π = 3.1415926536
e = 2.7182818285

def __init__(self, data=None, lambtha=1.):
    """innit"""
    self.lambtha=float(lambtha)
    if data == None:
        if lambtha <= 0:
            raise ValueError('data must be a positive value')
        if data:
            if len(data) < 2:
            raise ValueError('data must contain multiple values')
            if type(data) != list:
                raise TypeError('data must be a list')
            sum(data) / sum(lambtha)


def cdf(self, k):
    """calculatew value of the cdf for given number of successess"""
    if type(k) != int:
     
        
