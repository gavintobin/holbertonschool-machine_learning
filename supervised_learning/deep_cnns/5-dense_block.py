#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    '''builds dense block'''

    init = K.initializers.he_normal()

    for i in range(layers):
        firstbatch = K.layers.BatchNormalization(axis=3)(X)
        firstrelu = K.layers.ReLU()(firstbatch)
        firstconv = K.layers.Conv2D(filters=(4 * growth_rate),
                                    kernel_size=(1, 1),
                                    padding='same',
                                    kernel_initializer=init)(firstrelu)

        secbatch = K.layers.BatchNormalization(axis=3)(firstconv)
        secrelu = K.layers.ReLU()(secbatch)
        seconv = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 kernel_initializer=init)(secrelu)
        concatenation = K.layers.Concatenate()([X, seconv])
        nb_filters = nb_filters + growth_rate

    return (concatenation, nb_filters)
