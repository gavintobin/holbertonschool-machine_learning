#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''builds inception network'''
    inputs = K.Input(shape=(224, 224, 3))

    conv_7x7_2s = K.layers.Conv2D(filters=64,
                                  kernel_size=(7, 7),
                                  strides=2,
                                  padding='same',
                                  activation='relu'
                                  )(inputs)

    MaxPool3x3_2s = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=2,
                                          padding='same'
                                          )(conv_7x7_2s)

    conv_1x1_1v = K.layers.Conv2D(filters=64,
                                  kernel_size=(1, 1),
                                  padding='valid',
                                  activation='relu'
                                  )(MaxPool3x3_2s)

    conv_3x3_1s = K.layers.Conv2D(filters=192,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding='same',
                                  activation='relu'
                                  )(conv_1x1_1v)

    MaxPool3x3_2s_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=2,
                                            padding='same'
                                            )(conv_3x3_1s)

    inception_layer_0 = inception_block(MaxPool3x3_2s_1,
                                        [64, 96, 128, 16, 32, 32])

    inception_layer_1 = inception_block(inception_layer_0,
                                        [128, 128, 192, 32, 96, 64])

    MaxPool3x3_2s_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=2,
                                            padding='same'
                                            )(inception_layer_1)

    inception_layer_2 = inception_block(MaxPool3x3_2s_2,
                                        [192, 96, 208, 16, 48, 64])

    inception_layer_3 = inception_block(inception_layer_2,
                                        [160, 112, 224, 24, 64, 64])

    inception_layer_4 = inception_block(inception_layer_3,
                                        [128, 128, 256, 24, 64, 64])

    inception_layer_5 = inception_block(inception_layer_4,
                                        [112, 144, 288, 32, 64, 64])

    inception_layer_6 = inception_block(inception_layer_5,
                                        [256, 160, 320, 32, 128, 128])

    MaxPool3x3_2s_3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=2,
                                            padding='same'
                                            )(inception_layer_6)

    inception_layer_7 = inception_block(MaxPool3x3_2s_3,
                                        [256, 160, 320, 32, 128, 128])

    inception_layer_8 = inception_block(inception_layer_7,
                                        [384, 192, 384, 48, 128, 128])

    AvgPool7x7_1v = K.layers.AveragePooling2D(pool_size=(7, 7),
                                              strides=1,
                                              padding='valid'
                                              )(inception_layer_8)

    dropout = K.layers.Dropout(.4)(AvgPool7x7_1v)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax'
                             )(dropout)

    return K.Model(inputs=inputs, outputs=softmax)
