#!/usr/bin/env python3
'''task 11'''
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''reverse time decay'''
    decayed_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return decayed_alpha
