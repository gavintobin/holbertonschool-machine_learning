#!/usr/bin/env python3
'''hopefully our loss is small'''

import tensorflow as tf

def calculate_loss(y, y_pred):
    '''this will help us to see how small'''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=y_pred))
    return loss
