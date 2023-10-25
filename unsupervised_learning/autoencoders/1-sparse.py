#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    '''creates sparse autoencoder'''
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    lam = keras.regularizers.l1(lambtha)
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu',
                               activity_regularizer=lam)(x)
    e = keras.layers.Dense(latent_dims,
                           activation='relu',
                           activity_regularizer=lam)(x)

    encoder = keras.models.Model(encoder_inputs, e, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims,
                                         activation='sigmoid')(x)

    decoder = keras.models.Model(latent_inputs,
                                 decoder_outputs, name='decoder')

    # Autoencoder
    autoencoder_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_inputs,
                                     decoded, name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
