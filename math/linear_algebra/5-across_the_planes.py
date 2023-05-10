#!/usr/bin/env python3
''' function to add two matrixes'''


def add_matrices2D(mat1, mat2):
    summ = [map(sum, zip(*t)) for t in zip(mat1, mat2)]

    return summ