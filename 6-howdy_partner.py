#!/usr/bin/env python3
""" documention """
import numpy as np


def cat_arrays(arr1, arr2):
    """kitty cat"""
    cat = np.concatenate((arr1 + arr2))
    return cat