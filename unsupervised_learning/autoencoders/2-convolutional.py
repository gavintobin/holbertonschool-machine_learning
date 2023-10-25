#!/usr/bin/env python3
'''task 2'''
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    '''creates convo autoencoder'''
    # Encoder
    x = keras.Input(shape=input_dims)
    x = xhat

    # Conv layers
    for i in filters:
        xhat = keras.layers.Conv2D(i,
                                   (3, 3),
                                   activation='relu',
                                   padding='same')(xhat)

        xhat = keras.layers.MaxPooling2D((2, 2),
                                         padding='same')(xhat)

    encoder = keras.models.Model(input_img, xhat)

    # Decoder
    latin = keras.Input(shape=latent_dims)
    y = latin

    # Add Convolutional Layers In Reverse Order
    for i in reversed(filters[:-1]):
        y = keras.layers.Conv2D(i,
                                (3, 3),
                                activation='relu',
                                padding='same')(y)
        y = keras.layers.UpSampling2D((2, 2))(y)

    y = keras.layers.Conv2D(filters[0],
                            (3, 3),
                            activation='sigmoid',
                            padding='valid')(y)

    y = keras.layers.UpSampling2D((2, 2))(y)

    y = keras.layers.Conv2D(input_dims[-1],
                            (3, 3),
                            activation='sigmoid',
                            padding='same')(y)

    decoder = keras.models.Model(latent_input, y)

    # Autoencoder
    autoencoder_inputs = keras.Input(shape=input_dims)
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_inputs, decoded)

    # Compile
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
