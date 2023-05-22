#!/usr/bin/env python
"""normal class"""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """normal class"""
    def __init__(self, data=None, mean=0, stddev=1):
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive integer')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            diff = 0
            for i in data:
                diff += (i - self.mean) ** 2
            self.stddev = ((diff / len(data) ** .5))
