#!/usr/bin/env python3
'''creates layer'''

import tensorflow as tf


def create_layer(prev, n, activation):
    '''function 2'''
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=initializer, name="layer")
    return layer(prev)

