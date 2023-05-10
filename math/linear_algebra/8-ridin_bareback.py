#!/usr/bin/env python3
'''' bareback is how i ride'''


def mat_mul(mat1, mat2):
    '''mulmatmulnatmulmat'''
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)] for row in mat1]

            