
#!/usr/bin/env python3
import numpy as np


def marginal(x, n, P, Pr):
    '''marginal'''
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    err = 'x must be an integer that is greater than or equal to 0'
    if not isinstance(x, int) or x < 0:
        raise ValueError(err)

    if x > n:
        raise ValueError('x cannot be greater than n')
    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if Pr is a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if all values in P and Pr are in the range [0, 1]
    if (P < 0).any() or (P > 1).any():
        raise ValueError(f"All values in P must be in the range [0, 1]")

    if (Pr < 0).any() or (Pr > 1).any():
        raise ValueError(f"All values in Pr must be in the range [0, 1]")

    # Check if Pr sums to 1 (within a small tolerance)
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    # Calculate the marginal probability using the factorial-based method
    marginal_probability = np.sum(Pr * (np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x)) * (P ** x) * ((1 - P) ** (n - x))))

    return marginal_probability
