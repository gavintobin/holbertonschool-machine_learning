#!/usr/bin/env python3
"""class poisson"""
pi = 3.1415926536
e = 2.7182818285


class Poisson:
    '''poisson class'''
    def __init__(self, data=None, lambtha=1.):
        """innit"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = sum(data) / len(data)


    def pmf(self, k):
        '''calculate value of the cdf for given number of successess'''

        def factorial(self, n):
            """helper func since cant yoompooort modules"""
            if n == 0:
                return 1
            fact = 1
            for i in range(1, n + 1):
                fact += i
            return fact

        if type(k) != int:
        self.k = int(k)

        if k < 0 or k > len(self.data):
            return 0

        pmf_value = (self.lambtha ** k) * (e ** -self.lambtha) / self.factorial(k)
        return pmf_value
