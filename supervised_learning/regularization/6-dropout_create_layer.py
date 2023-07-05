#!/usr/bin/env python3
'''task7'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''creates layer w drop out reg'''
    layerreg = tf.layers.Dropout(rate=keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_regularizer=layerreg,
        kernel_initializer=init
    )
    return dropout
