#!/usr/bin/env python3
'''clac gd in two line... love it'''

import tensorflow as tf


def create_trian_op(loss, alpha):
    '''love it'''
    opty = tf.train.GradientDescentOptimizer(alpha).minimise(loss)
    return opty
