#!/usr/bin/env python3
'''tadsk 3'''
import tensorflow as tf
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates variational autoencoder'''
    # Encoder
    encoder_inputs = K.layers.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    # Latent space parameters
    z_mean = K.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dims), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = K.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = K.models.Model(encoder_inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = K.layers.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = K.layers.Dense(units, activation='relu')(x)
    decoder_outputs = K.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = K.models.Model(latent_inputs, decoder_outputs, name='decoder')

    # Autoencoder
    autoencoder_inputs = K.layers.Input(shape=(input_dims,))
    encoded, z_mean, z_log_var = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = K.models.Model(autoencoder_inputs, decoded, name='autoencoder')

    # Loss function for VAE
    def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
        reconstruction_loss = K.losses.binary_crossentropy(x, x_decoded_mean)
        reconstruction_loss *= input_dims
        kl_loss = -0.5 * K.backend.sum(1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    autoencoder.compile(optimizer='adam', loss=lambda x, x_decoded_mean: vae_loss(x, x_decoded_mean, z_mean, z_log_var))

    return encoder, decoder, autoencoder
