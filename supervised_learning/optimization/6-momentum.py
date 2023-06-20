#!/usr/bin/env python3
'''task 7'''
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    ''' create train op for nn'''
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
