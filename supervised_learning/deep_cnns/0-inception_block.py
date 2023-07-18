#!/usr/bin/env python 3
'''task 1'''
import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''builds inception block'''
    F1, F3R, F3, F5R, F5, FPP = filters
    conv_1by = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                               activation='relu')(A_prev)
    conv_1byb4 = K.layers.Conv2d(filters=F3R, kernel_size=(1, 1),
                                 activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2d(filters=F3, kernel_size=(3, 3),
                               activation='relu')(conv_1byb4)
    conv_5x5b4 = K.layers.Conv2d(filters=F5R, kernel_size=(1, 1),
                                 activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2d(filters=F5, kernel_size=(5, 5),
                               activation='relu')(conv_5x5b4)
    pool = K.layers.MaxPool2d(pool_size=(3, 3),
                              strides=(1, 1), padding='same')(A_prev)
    pool_1x1 = K.layers.Conv2d(filters=FPP, activation='relu')(pool)

