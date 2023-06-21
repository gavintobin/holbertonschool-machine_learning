#!/usr/bin/env python3
'''task 09'''

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''rms optimo'''
    return tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta2,
                                     epsilon=epsilon).minimize(loss)
