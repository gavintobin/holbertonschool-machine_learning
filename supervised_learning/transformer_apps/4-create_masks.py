#!/usr/bin/env python3
'''task4'''
import tensorflow as tf

def create_masks(inputs, target):
    '''create mask'''
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Lookahead mask for the 1st attention block in the decoder
    combined_mask = create_lookahead_mask(tf.shape(target)[1])
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    combined_mask = tf.maximum(combined_mask, target_padding_mask[:, tf.newaxis, tf.newaxis, :])

    return encoder_mask, combined_mask, decoder_mask

def create_lookahead_mask(size):
    '''lookahead'''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
