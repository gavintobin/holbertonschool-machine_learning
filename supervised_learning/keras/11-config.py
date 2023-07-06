#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def save_config(network, filename):
    '''save'''
    with open(filename, 'w') as file:
        file.write(network.to_json())
    return None


def load_config(filename):
    '''load'''
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())
