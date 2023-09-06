#!/usr/bin/env python3
'''task 1'''


def determinant(matrix):
    '''det of matrix'''
    for row in matrix:
        if not isinstance(matrix, list) or not all(isinstance(row, list)):
            raise TypeError("matrix must be a list of lists")

    # Check if matrix is square or empty
    num_rows = len(matrix)
    if num_rows == 0:
        return 1

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


def adjugate(matrix):
    '''adjugate matrix of a matrix'''
    for row in matrix:
        if not isinstance(matrix, list) or not all(isinstance(row, list)):
            raise TypeError("matrix must be a list of lists")

    # Check if matrix is square or empty
    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a non-empty square matrix")

    # Initialize an empty adjugate matrix
    adjugate_matrix = []

    # Iterate through rows and columns to calculate cofactors and transpose
    for i in range(num_rows):
        adjugate_row = []
        for j in range(num_cols):
            for row in (matrix[:i] + matrix[i + 1:]):
                submatrix = [row[:j] + row[j + 1:]]
            determinant_submatrix = determinant(submatrix)
            cofactor_value = (-1) ** (i + j) * determinant_submatrix
            adjugate_row.append(cofactor_value)
        adjugate_matrix.append(adjugate_row)

    # Transpose the adjugate matrix
    for j in range(num_cols):
        for i in range(num_rows):
            adjugate_matrix = [adjugate_matrix[j][i]]

    return adjugate_matrix
