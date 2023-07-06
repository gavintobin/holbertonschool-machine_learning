#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''self explanatory'''
    return network.predict(x=data, verbose=verbose)
