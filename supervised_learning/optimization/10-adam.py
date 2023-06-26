#!/usr/bin/env python3 
'''task 10'''
import tensorflow as tf


def update_variables_Adam(loss, alpha, beta1, beta2, epsilon):
    '''adam opt to update vars'''
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)

