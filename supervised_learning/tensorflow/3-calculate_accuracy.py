#!/usr/bin/env python3
'''accuracy testing'''

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    '''how close are we'''
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = tf.argmax(y, axis=1)
    correct = tf.equal(y_pred_labels, y_true_labels)
    acurracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acurracy
