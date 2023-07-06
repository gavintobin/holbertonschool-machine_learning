#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
   '''self explainatory'''
   network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    '''self explanatory'''
    network.load_weights(filepath=filename)
