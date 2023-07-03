#!/usr/bin/env python3
'''task 4'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    ''' creates tf layer w l2 reg'''
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    
    layer = tf.layers.dense(inputs=prev, units=n, activation=activation, kernel_regularizer=regularizer)
    
    return layer(prev)
