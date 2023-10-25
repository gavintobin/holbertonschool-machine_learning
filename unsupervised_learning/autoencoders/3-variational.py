#!/usr/bin/env python3
'''tadsk 3'''
import tensorflow as tf
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates variational autoencoder'''
    def sampling(args):
        '''sampling helper func'''
        zmean, zlogvar = args
        shp = keras.backend.shape(zmean)
        epsilon = keras.backend.random_normal(shape=shp)
        return z_mean + keras.backend.exp(0.5 * zlogvar) * epsilon

    x = keras.Input(shape=(input_dims,))
    xhat = x

    for i in hidden_layers:
        x = keras.layers.Dense(i, activation='relu')(x)

    zmean = keras.layers.Dense(latent_dims, activation=None,
                               name="zmean")(x)
    zlogvar = keras.layers.Dense(latent_dims, activation=None,
                                 name="zlogvar")(x)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,),
                            name="z")([zmean, zlogvar])

    encoder = keras.models.Model(x, [z, zmean, zlogvar])

    # Decoder
    latin = keras.layers.Input(shape=(latent_dims,))
    y = latin

    for i in reversed(hidden_layers):
        y = keras.layers.Dense(i, activation='relu')(y)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(y)

    decoder = keras.models.Model(latin, output)

    # Variational Autoencoder
    autoencoder_inputs = keras.layers.Input(shape=(input_dims,))
    encoded, z_mean, z_log_var = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    auto = keras.models.Model(autoencoder_inputs, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
