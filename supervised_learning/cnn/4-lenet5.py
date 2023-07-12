#!/usr/bin/env python3
'''task 4'''
import tensorflow as tf


def lenet5(x, y):
    '''builds modded fersion of lenet5 architechure using tf'''
    he_init = tf.contrib.layers.variance_scaling_initializer()

    first_conv = tf.layers.conv2d(x, filters=6,
                                  kernal_size=(5, 5), padding='same',
                                  activation=tf.nn.relu,
                                  kernel_initializer=he_init)
    first_pool = tf.layers.max_pooling2d(first_conv, pool_size=(2, 2),
                                         strides=(2, 2))
    sec_conv = tf.layers.cov2d(first_pool, filters=16,
                               kernel_size=(5,5), padding='valid', 
                               activation=tf.nn.relu, 
                               kernel_initializer=he_init)
    sec_pool = tf.layers.max_pooling2d(sec_conv, pool_size=(2, 2),
                                       strides=(2, 2))
    flat = tf.layers.flatten(sec_pool)

    firstfull = tf.layers.dense(flat, units=120, activation=tf.nn.relu,
                                kernel_initializer=he_init)
    secfull = tf.layers.dense(firstfull, units=84, activation=tf.nn.relu,
                              kernel_initalizer=he_init)
    logits = tf.layers.dense(secfull, units=10, kernel_initializer=he_init)
    output = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=logits)
    optimizer = tf.train.AdamOptimizer
    trainop = optimizer.moinimize(loss)

    corpred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corpred, tf.float32))

    return output, trainop, loss, accuracy
