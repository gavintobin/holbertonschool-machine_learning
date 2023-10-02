#!/usr/bin/env python3
'''task 2'''
import numpy as np


def regular(P):
    '''deter,ines state probs of reg mc'''
    n = P.shape[0]
    if P.shape != (n, n):
        return None

    if np.any(P <= 0):
        return None

    # checks for reg mc
    is_regular = np.all(P > 0) and np.all(np.isclose(np.sum(P, axis=1), 1.0))

    if not is_regular:
        return None

    # find steady state (left eigenvector of P corresponding to eigenvalue 1)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # find the index of the eigenvalue equal to 1 (within some tolerance)
    eigenvalue_index = np.where(np.isclose(eigenvalues, 1.0))[0]

    # edge case steady state doesnt exist
    if len(eigenvalue_index) != 1:
        return None

    # find corresponding eigenvector
    steady_state_vector = eigenvectors[:, eigenvalue_index[0]].real

    # normalize ss vect to make sure is 1
    steady_state_vector /= np.sum(steady_state_vector)

    return steady_state_vector.reshape(1, -1)
