#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K


def identity_block(A_prev, filters):
    '''builds id block '''
    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    first_1by =  K.layers.Conv2D(filters=F11,
                                kernel_size=(1, 1),
                                strides=1,
                                padding='same',
                                kernel_initializer=init
                                )(A_prev)
    first_batchnorm = K.layers.BatchNormalization(axis=3)(first_1by)
    first_relu = K.layers.ReLU()(first_batchnorm)

    conv3by = K.layers.Conv2D(filters=F3,
                                kernel_size=(3, 3),
                                strides=1,
                                padding='same',
                                kernel_initializer=init)(first_relu)
    sec_batchnorm = K.layers.BatchNormalization(axis=3)(conv3by)
    sec_relu = K.layers.ReLU()(sec_batchnorm)

    sec_1by =  K.layers.Conv2D(filters=F12,
                                kernel_size=(1, 1),
                                strides=1,
                                padding='same',
                                kernel_initializer=init)(sec_relu)

    third_batchnorm = K.layers.BatchNormalization(axis=3)(sec_1by)

    layer = K.layers.Add()([sec_batchnorm, A_prev])
    activated = K.layers.ReLU()(layer)
    return activated
