#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''adam op w cross entro loss '''
    model = network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model
