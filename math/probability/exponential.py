#!/usr/bin/env python3
'''exponential'''
pi = 3.1415926536
e = 2.7182818285


class Exponential:
    """expo class"""
    def __init__(self, data=None, lambtha=1.):
        """initialization"""
        self. lambtha = float(lambtha)
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            self.lambtha = 1. / (sum(data) / len(data))

    def pdf(self, x):
        """expo pdf caluclation"""
        if x < 0:
            return 0
        else:
            return (self.lambtha * e ** ((self.lambtha * -1) * x))

    def cdf(self, x):
        """cdf formula"""
        if x < 0:
            return 0
        else:
            return 1 - (e ** (-self.lambtha * x))
