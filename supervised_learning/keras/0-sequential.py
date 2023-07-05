#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''builds keras NN'''
    model = K.Sequential()

    model.add(K.InputLayer(input_shape=(nx, )))

    for i in range(len(layers)):
        model.add(K.layers.Dense(units=layers[i], activation=activations,
                                 kernal_regulizer=K.regularizers.l2(lambtha)))

        if i < len(layers):
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
