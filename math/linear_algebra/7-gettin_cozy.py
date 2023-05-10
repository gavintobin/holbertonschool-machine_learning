#!/usr/bin/env python3
''' doc for cozy boy'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''doggy boy'''
    def cat_matrices2D(mat1, mat2, axis=0):
        if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            arr3 = mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            arr3 = [x + x2 for x, x2 in zip(mat1, mat2)]
    return arr3

