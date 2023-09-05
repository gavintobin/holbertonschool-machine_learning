#!/usr/bin/env python3
'''task 1'''
import numpy as np


def determinant(matrix):
    '''det matrix'''
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    num_rows = len(matrix)
    if num_rows == 0:
        return 1

    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a square matrix")

    # Calculate the det
    det = np.linalg.det(matrix)

    return round(det)

