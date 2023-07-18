#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    '''builds densenet 121'''
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    firstbatch = K.layers.BatchNormalization(axis=3)(input)
    firstrelu = K.layers.Activation('relu')(firstbatch)
    firstconv = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=2,
                             padding='same',
                             kernel_initializer=init
                             )(firstrelu)
    pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=2,
                                       padding='same'
                                       )(firstconv)

    firstdb, nb_filters = dense_block(X=pool,nb_filters=64,
                                      growth_rate=growth_rate,
                                      layers=6)
    firsttb, nb_filters = transition_layer(X=firstdb,
                                           nb_filters=nb_filters,
                                           compression=compression)
    secdb, nb_filters = dense_block(X=firsttb, nb_filters=nb_filters,
                                    growth_rate=growth_rate,layers=12)
    sectb, nb_filters = transition_layer(X=secdb, nb_filters=nb_filters,
                                         compression=compression)
    thirddb, nb_filters = dense_block(X=sectb, nb_filters=nb_filters,
                                      growth_rate=growth_rate,layers=24)
    thirdtb, nb_filters = transition_layer(X=thirddb,
                                           nb_filters=nb_filters,
                                           compression=compression)
    fouthdb, nb_filters = dense_block(X=thirdtb, nb_filters=nb_filters,
                                      growth_rate=growth_rate,
                                      layers=16)

    global_average = K.layers.AveragePooling2D(pool_size=(7, 7),
                                               strides=1,
                                               )(fouthdb)

    softmax = K.layers.Dense(units=1000,activation='softmax',
                             kernel_initializer=init)(global_average)

    return K.Model(inputs=input, outputs=softmax)
