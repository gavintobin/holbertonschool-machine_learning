#!/usr/bin/env python3
'''task 3'''
import tensorflow as tf


def l2_reg_cost(cost):
    '''calcs l2 cost'''
    return cost + tf.losses.get_regularization_losses(scope=None)
