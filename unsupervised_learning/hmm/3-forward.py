#!/usr/bin/env python3
'''task 4'''
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''perform the forward algorithm for hmm'''
    T = len(Observation)
    N, _ = Emission.shape

    # initialize the forward probabilities matrix F
    F = np.zeros((N, T))

    # initialize the scaling factor list to avoid underflow
    scale = np.zeros(T)

    # initialize the first column of F using Initial and Emission probabilities
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # icale the first column and store the scaling factor
    scale[0] = 1.0 / np.sum(F[:, 0])
    F[:, 0] *= scale[0]

    # forward pass to compute F and the scaling factors
    for t in range(1, T):
        for j in range(N):
            for i in range(N):
                E =  Emission[j, Observation[t]]
                F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * Emission[j, Observation[t]] * E
        # fcale the column and store the scaling factor
        scale[t] = 1.0 / np.sum(F[:, t])
        F[:, t] *= scale[t]

    # Compute the likelihoods
    P = np.sum(F[:, -1])

    return P, F
