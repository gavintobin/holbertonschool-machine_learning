#!/usr/bin/env python3
'''task 5'''
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''perform the forward algorithm for hmm'''
    T = len(Observation)
    N, _ = Emission.shape

    if T == 0 or N == 0 or M == 0:
        return None, None
  
    # initialize the forward probabilities matrix B
    B = np.zeros((N, T))

    # initialize the first column of B using Initial and Emission probabilities
    B[:, -1] = 1

    # forward pass to compute B and the scaling factors
    for t in range(T - 2, -1 , -1):
        for j in range(N):
            for i in range(N):
                E = Emission[i, Observation[t + 1]] * B[i, t + 1]
                B[j, t] += Transition[j, i] * E

    # Compute the likelihoods
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
