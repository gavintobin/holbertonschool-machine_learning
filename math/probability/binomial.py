#!/usr/bin/env python3
"""binomial class"""

pi = 3.1415926536
e = 2.7182818285


class Binomial:
    """binom class"""
    def __init__(self, data=None, n=1, p=0.5):
        """init"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if self.n <= 0:
                raise ValueError('n must be a positive value')
            if self.p <= 0 or self.p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = 0
            for number in data:
                variance = variance + (number - mean) ** 2
            variance = variance / len(data)
            q = variance / mean
            p1 = 1 - q
            n1 = (sum(data) / p1) / len(data)
            self.n = int(round(n1))
            self.p = float(mean/self.n)

    def pmf(self, k):
        """pmf formula"""
        def factorial(n):
            """helper func since cant yoompooort modules"""

        k = int(k)

        if k < 0:
            return 0

        c = (self.factorial(k) * self.factorial(self.n - k))
        a = (self.factorial(self.n) / c)

        return (a * (self.p ** k) * ((1 - self.p) ** (self.n - k)))

    def factorial(self, k):
        """factorial helper function"""
        result = 1
        for i in range(1, k+1):
            result *= i
        return result

    def cdf(self, k):
        """cdf formula"""
        k = int(k)
        if k < 0:
            return 0
        x = 0
        for i in range(0, k + 1):
            x += (self.pmf(i))
        return x
