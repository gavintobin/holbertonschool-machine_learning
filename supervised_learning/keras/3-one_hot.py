#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''converts labels into one hot matrix'''

    return K.utils.to_categorical(labels, classes)
