#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    '''buils transition layer'''
    init = K.initializers.he_normal()
    filters = (int)(nb_filters * compression)

    firstbatch = K.layers.BatchNormalization(axis=3)(X)
    firstrelu = K.layers.ReLU()(firstbatch)
    firstconv = K.layers.Conv2D(filters=filters,
                                kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=init)(firstrelu)

    pool =  K.layers.AveragePooling2D(pool_size=(2, 2), strides=2,
                                      padding='same')(firstconv)
    return (pool, filters)
