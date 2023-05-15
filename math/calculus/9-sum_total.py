#!/usr/bin/python3
"""docstuff"""


def summation_i_squared(n):
    """sigma sum"""
    i = 1
    x = 1
    while x <= n:
        i += x ** 2
        x += 1
        print(x)
    return i


