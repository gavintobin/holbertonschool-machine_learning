#!/usr/bin/env python3
'''task 6 '''
import numpy as np


def adjugate(matrix):
    '''adj matrix'''
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
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
            submatrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            determinant_submatrix = determinant(submatrix)
            cofactor_value = (-1) ** (i + j) * determinant_submatrix
            adjugate_row.append(cofactor_value)
        adjugate_matrix.append(adjugate_row)

    # Transpose the adjugate matrix
    adjugate_matrix = [[adjugate_matrix[j][i] for j in range(num_cols)] for i in range(num_rows)]

    return adjugate_matrix


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



def inverse(matrix):
    '''inverse'''
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square or empty
    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate the determinant of the input matrix
    det = determinant(matrix)

    # If the determinant is zero, the matrix is singular and has no inverse
    if det == 0:
        return None

    # Calculate the adjugate matrix using the previously defined adjugate function
    adjugate_matrix = adjugate(matrix)

    # Calculate the inverse matrix by dividing each element of the adjugate matrix by the determinant
    inverse_matrix = [[adjugate_matrix[i][j] / det for j in range(num_cols)] for i in range(num_rows)]

    return inverse_matrix


    import numpy as np

def definiteness(matrix):
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues of the matrix
    eigenvalues, _ = np.linalg.eig(matrix)

    # Count positive, negative, and zero eigenvalues
    num_positive = np.sum(eigenvalues > 0)
    num_negative = np.sum(eigenvalues < 0)
    num_zero = np.sum(eigenvalues == 0)

    # Determine definiteness based on eigenvalues
    if num_positive == matrix.shape[0]:
        return "Positive definite"
    elif num_positive > 0 and num_zero > 0:
        return "Positive semi-definite"
    elif num_negative == matrix.shape[0]:
        return "Negative definite"
    elif num_negative > 0 and num_zero > 0:
        return "Negative semi-definite"
    else:
        return "Indefinite"
