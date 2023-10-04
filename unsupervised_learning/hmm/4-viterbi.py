#!/usr/bin/env python3
'''task 4'''
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    ''' that calculates the most likely sequence of h
    dden states for a hidden markov model:'''
    T = len(Observation)
    N = len(Initial)

    # Initialize the Viterbi path and probability tables
    V = np.zeros((N, T))
    path = np.zeros((N, T), dtype=int)

    # Initialize the first column of the Viterbi table
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    path[:, 0] = 0

    # Fill in the rest of the Viterbi table
    for t in range(1, T):
        for n in range(N):
            probs = V[:, t-1] * Transition[:, n] * Emission[n, Observation[t]]
            path[n, t] = np.argmax(probs)
            V[n, t] = np.max(probs)

    # Find the most likely final state
    final_state = np.argmax(V[:, -1])

    # redo the most likely path
    path_list = [final_state]
    for t in range(T - 1, 0, -1):
        final_state = path[final_state, t]
        path_list.insert(0, final_state)

    # calc the prob of the most likely path
    P = np.max(V[:, -1])

    return path_list, P
