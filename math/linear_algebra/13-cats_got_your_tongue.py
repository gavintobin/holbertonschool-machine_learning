#!/usr/bin/env python3
''' cat the cat plz'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''funky boy'''
    new = np.concatenate((mat1, mat2), axis=0)
    return new
