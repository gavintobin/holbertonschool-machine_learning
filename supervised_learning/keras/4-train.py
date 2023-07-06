#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    '''train da model'''
    model = network.fit(x=data, y=labels, batch_size=batch_size,
                        epochs=epochs, verbose=verbose,
                        shuffle=shuffle)
    return model
