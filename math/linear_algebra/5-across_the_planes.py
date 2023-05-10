#!/usr/bin/env python3
''' function to add two matrixes'''


def add_matrixes2d(mat1, mat2):
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            sum[i][j] = mat1[i][j] + mat2[i][j]
    return sum