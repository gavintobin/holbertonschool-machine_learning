#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    '''creats sparse auto encoder'''
    x = keras.Input(shape=(input_dims,))
    xhat = x

    for i in hidden_layers:
        xhat = keras.layers.Dense(i, activation='relu')(xhat)

    la = keras.regularizers.l1(lambtha)

    lat = keras.layers.Dense(latent_dims, activation='relu',
                             activity_regularizer=la)(xhat)

    encoder = keras.models.Model(x, lat)

    decin = keras.Input(shape=(lat,))
    decout = decin

    for i in reversed(hidden_layers):
        decout = keras.layers.Dense(i,
                                    activation='relu')(decout)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(decout)
    decoder = keras.models.Model(decin, output)

    # encoder
    autoencoder_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_inputs,
                                     autoencoder_output)

    # Compile
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
