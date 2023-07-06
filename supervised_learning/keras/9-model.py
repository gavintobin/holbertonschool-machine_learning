#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def save_model(network, filename):
    '''self explanatory'''
    network.save(filename)


def load_model(filename):
    '''self explanatory'''
    return K.models.load_model(filename)
