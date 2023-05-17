#!/usr/bin/env python3
"""docstuff"""


def summation_i_squared(n):
    """sigma sum"""
    if (n < 1):
        return
    return sum(map(lambda n: n * n, range(1, n + 1)))
