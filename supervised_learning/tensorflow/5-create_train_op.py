#!/usr/bin/env python3
'''clac gd in two line... love it'''

import tensorflow as tf


def create_train_op(loss, alpha):
    '''love it'''
    opty = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return opty
