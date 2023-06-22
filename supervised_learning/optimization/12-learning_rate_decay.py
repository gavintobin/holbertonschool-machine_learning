#!/usr/bin/env python3
'''task 12'''

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''inverse time decary to calc lrd'''
    tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, stair_case=True)
