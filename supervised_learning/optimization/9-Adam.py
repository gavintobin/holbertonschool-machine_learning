#!/usr/bin/env python3 
'''task 10'''
import tensorflow as tf


def update_variables_Adam(alpha, beta1,
                          beta2, epsilon, var, grad, v, s, t):
    '''adam opt to update vars'''
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon,
                                   var, grad, )
