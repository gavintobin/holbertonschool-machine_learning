#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''builds inception network'''
    input = K.Input(shape=(224, 224, 3))

    conv7by_2 = K.layers.Conv2D(filters=64,
                                  kernel_size=(7, 7),
                                  strides=2,
                                  padding='same',
                                  activation='relu'
                                  )(input)

    MaxPool3by_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=2,
                                          padding='same'
                                          )(conv7by_2)

    conv1by_1 = K.layers.Conv2D(filters=64,
                                  kernel_size=(1, 1),
                                  padding='valid',
                                  activation='relu'
                                  )(MaxPool3by_2)

    conv3by_1 = K.layers.Conv2D(filters=192,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding='same',
                                  activation='relu'
                                  )(conv1by_1)

    MaxPool3by_2_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=2,
                                            padding='same'
                                            )(conv3by_1)
    inception0 = inception_block(MaxPool3by_2_1,
                                        [64, 96, 128, 16, 32, 32])

    inception1 = inception_block(inception0,
                                        [128, 128, 192, 32, 96, 64])
    MaxPool3by_2_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=2,
                                            padding='same'
                                            )(inception1)

    inception2 = inception_block(MaxPool3by_2_2,
                                 [192, 96, 208, 16, 48, 64])

    inception3 = inception_block(inception2,
                                 [160, 112, 224, 24, 64, 64])

    inception4 = inception_block(inception3,
                                 [128, 128, 256, 24, 64, 64])

    inception5 = inception_block(inception4,
                                 [112, 144, 288, 32, 64, 64])

    inception6 = inception_block(inception5,
                                 [256, 160, 320, 32, 128, 128])

    MaxPool3by_2_3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=2,
                                            padding='same'
                                            )(inception6)

    inception7 = inception_block(MaxPool3by_2_3,
                                 [256, 160, 320, 32, 128, 128])

    inception8 = inception_block(inception7,
                                        [384, 192, 384, 48, 128, 128])

    AvgPool7x7_1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                              strides=1,
                                              padding='valid'
                                              )(inception8)
    dropout = K.layers.Dropout(.4)(AvgPool7x7_1)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax')(dropout)
    return K.Model(inputs=input, outputs=softmax)
