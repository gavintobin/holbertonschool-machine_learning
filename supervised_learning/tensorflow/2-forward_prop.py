#!/usr/bin/env python3
'''forward prop'''
create_layer = __import__('1-create_layer').create_layer
import tensorflow as tf

def forward_prop(x, layer_sizes=[], activations=[]):
    '''forward prop using tf'''
    assert len(layer_sizes) == len(activations), "layer_sizes and activations must have the same length"

    prev = x
    for size, activation in zip(layer_sizes, activations):
        prev = create_layer(prev, size, activation)

    return prev
