#!/usr/bin/env python3
'''task 7'''
import tensorflow as tf

def create_momentum_op(loss, alpha, beta1):
    ''' create train op for nn'''
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = tf.train.optimizer.minimum(loss)
    return train_op

