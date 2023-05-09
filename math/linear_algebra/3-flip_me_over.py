#!/usr/bin/env python3
''' taask  3 with hopefully eniough documentation necessary to pass checker'''


def matrix_transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0 for j in range(rows)] for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            transposed[i][j] = matrix[j][i]
    return transposed
