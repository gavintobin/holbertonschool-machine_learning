#!/usr/bin/env python3
''' function to add two matrixes'''


def add_matrices2D(mat1, mat2):
   result = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
   return result
