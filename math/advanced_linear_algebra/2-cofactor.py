#!/usr/bin/env python3
'''task 1'''
def determinant(matrix):
    '''det of matrix'''
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square or empty
    num_rows = len(matrix)
    if num_rows == 0:
        return 1  #case for 0x0 matrix, determinant is 1

    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a square matrix")

    # 1x1 matrix case
    if num_rows == 1:
        return matrix[0][0]

    # Recursive caalc of det for larger matrix
    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(num_cols):
        submatrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
        cofactor = matrix[0][i] * determinant(submatrix)
        det += cofactor * (-1) ** i

    return det

def cofactor(matrix):
    '''cofactor'''
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square or empty
    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a non-empty square matrix")

    # Initialize an empty cofactor matrix
    cofactor_matrix = []

    # Iterate through rows and columns to calc cofactors
    for i in range(num_rows):
        cofactor_row = []
        for j in range(num_cols):
            submatrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            determinant_submatrix = determinant(submatrix)
            cofactor_value = (-1) ** (i + j) * determinant_submatrix
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
