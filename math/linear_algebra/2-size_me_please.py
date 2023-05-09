#!/usr/bin/env python3
'''task2'''

def matrix_shape(matrix):
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    shape.append(len(matrix))
    return shape
