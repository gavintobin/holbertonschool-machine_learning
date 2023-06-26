#!/usr/bin/env python3
'''task 14'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''same but in tf'''
    kerninit = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l = tf.layers.dense(prev, n, kerninit)

    mean, var = tf.nn.moments(l, 0)

    gamma = tf.Variable(tf.ones[n])
    beta = tf.Variable(tf.zeros[n])

    eps = 1 * (10 ** -8)
    normz = tf.nn.batch_normalization(l, mean, var, beta, gamma, eps)
    return activation(normz)
