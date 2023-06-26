#!/usr/bin/env python3
'''task 14'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''same but in tf'''
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.dense(inputs=prev, units=n, kenerl_initializer=init)

    mean, var = tf.nn.moments(base, 0)

    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)

    eps = 1 * (10 ** -8)
    normz = tf.nn.batch_normalization(base, mean, var, beta, gamma, eps)
    return activation(normz)
