#!/usr/bin/env python3
'''clac gd in two line... love it'''

import tensorflow as tf

def create_trian_op(loss, alpha):
    '''love it'''
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
