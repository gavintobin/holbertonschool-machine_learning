#!/usr/bin/env python3
'''task2'''
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    '''buikds resnet 50 '''
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    first1by = K.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               stride=2,
                               padding='same',
                               kernel_initializer=init)(input)
    first_batchnorm = K.layers.BatchNormalization(axis=3)(first1by)
    first_relu = K.layers.ReLU()(first_batchnorm)

    first_Maxpool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=2,
                                          padding='same')(first_relu)

    firstconv2 = projection_block(first_Maxpool, [64, 64, 256], s=1)
    seconv2 = identity_block(firstconv2, [64, 64, 256])
    thirdconv2 = identity_block(seconv2, [64, 64, 256])

    firstconv3 = projection_block(thirdconv2, [128, 128, 512])
    secondv3 = identity_block(firstconv3, [128, 128, 512])
    thirdconv3 = identity_block(secondv3, [128, 128, 512])
    fourthconv3 = identity_block(thirdconv3, [128, 128, 512])

    firstconv4 = projection_block(fourthconv3, [256, 256, 1024])
    seconv4 = identity_block(firstconv4, [256, 256, 1024])
    thirdconv4 = identity_block(seconv4, [256, 256, 1024])
    fourthconv4 = identity_block(thirdconv4, [256, 256, 1024])
    fifthconv4 = identity_block(fourthconv4, [256, 256, 1024])
    sixthconv4 = identity_block(fifthconv4, [256, 256, 1024])

    firstconv5 = projection_block(sixthconv4, [512, 512, 2048])
    seconv5 = identity_block(firstconv5, [512, 512, 2048])
    thirdconv5 = identity_block(seconv5, [512, 512, 2048])

    pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1,
                                     padding='valid')(thirdconv5)

    softmax = K.layers.Dense(units=1000, activation=softmax)(pool)

    return  K.Model(inputs=input, outputs=softmax)
    
