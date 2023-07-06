#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''self explanatory'''
    return network.evaluate(x=data, y=labels, verbose=verbose)
