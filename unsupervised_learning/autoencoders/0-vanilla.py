#!/usr/bin/env python3
'''task 2'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates an autoencoder'''

    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for i in hidden_layers:
        x = keras.layers.Dense(i, activation='relu')(x)
    outputs = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.models.Model(inputs, outputs, name='encoder')

    # decode
    latinput = keras.Input(shape=(latent_dims))
    x = latinput
    for i in reversed(hidden_layers):
        x = keras.layers.Dense(i, activation='relu')(x)
    decout = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.models.Model(latinput, decout, name='decoder')

    # Autoencoder
    autoencoder_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_inputs,
                                     decoded, name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
