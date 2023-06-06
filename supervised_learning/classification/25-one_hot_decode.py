#!/usr/bin/env python3
"""One Cold Decode?"""
import numpy as np


def one_hot_decode(one_hot):
    """Decode Fun"""
    if type(one_hot) is not np.ndarray:
        return None
    if one_hot.ndim != 2:
        return None
    return np.argmax(one_hot, axis=0)
