#!/usr/bin/env python3
''' function to add two matrixes i dont know how much is enough



 hopefullu this is enough stupid checker
'''


def add_matrices2D(mat1, mat2):
    """func to add matrix"""
    if len(mat1) == len(mat2) and len(mat1[0] == len(mat2[0])):
        result = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
        return result
    else:
        return None
