#!/usr/bin/env python3
"""class poisson"""
pi = 3.1415926536
e = 2.7182818285


class Poisson:
    '''poisson class'''
    def __init__(self, data=None, lambtha=1.):
        """innit"""
        self.lambtha = float(lambtha)
        self.data = data
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

        def factorial(n):
            """helper func since cant yoompooort modules"""
            if n == 0:
                return 1
            fact = 1
            for i in range(1, n + 1):
                fact *= i
            return fact

        if type(k) != int:
            self.k = int(k)

        else:
            self.k = int(k)

        if k < 0:
            return 0

        pmf_numer = (e ** (self.lambtha * -1)) * (self.lambtha ** self.k)
        pmf_denom = factorial(self.k)
        return pmf_numer / pmf_denom

    def cdf(self, k):
        '''calculates value using cdf formula'''

        def factorial(n):
            """helper func since cant yoompooort modules"""
            if n == 0:
                return 1
            fact = 1
            for i in range(1, n + 1):
                fact *= i
            return fact

        k = int(k)

        if k < 0:
            return 0

        cdf_val = sum(self.pmf(i) for i in range(k + 1))
        return cdf_val

