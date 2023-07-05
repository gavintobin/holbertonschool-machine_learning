#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''builds keras NN'''
    model = K.Sequential()

    model.add(K.layers.InputLayer(input_shape=(nx, )))

    for i in range(len(layers)):
        model.add(K.layers.Dense(units=layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha)))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
