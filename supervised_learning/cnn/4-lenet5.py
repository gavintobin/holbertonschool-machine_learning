#!/usr/bin/env python3
'''task 4'''
import tensorflow as tf


def lenet5(x, y):
    '''builds modded fersion of lenet5 architechure using tf'''
    he_init = tf.contrib.layers.variance_scaling_initializer()

    firstconv = tf.layers.conv2d(x, filters=6, kernel_size=(
        5, 5), padding='same', activation=tf.nn.relu,
        kernel_initializer=he_init)

    firstpool = tf.layers.max_pooling2d(firstconv, pool_size=(2, 2),
                                    strides=(2, 2))

    seconv = tf.layers.conv2d(firstpool, filters=16, kernel_size=(
        5, 5), padding='valid', activation=tf.nn.relu,
        kernel_initializer=he_init)

    secpool = tf.layers.max_pooling2d(seconv, pool_size=(2, 2),
                                    strides=(2, 2))

    flat = tf.layers.flatten(secpool)

    firstfull = tf.layers.dense(
        flat, units=120, activation=tf.nn.relu,
        kernel_initializer=he_init)

    secfull = tf.layers.dense(firstfull, units=84, activation=tf.nn.relu,
                          kernel_initializer=he_init)

    logits = tf.layers.dense(secfull, units=10,
                             kernel_initializer=he_init)
    output = tf.nn.softmax(logits)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=logits)

    optimizer = tf.train.AdamOptimizer()
    trainop = optimizer.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, trainop, loss, accuracy
