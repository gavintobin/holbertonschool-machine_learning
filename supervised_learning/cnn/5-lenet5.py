#!/usr/bin/env python3
'''task 5'''
import tensorflow.keras as K


def lenet5(X):
    '''same as before =but now keras'''
    firstconv = K.layers.Conv2D(filters=6, kernel_size=(
        5, 5), padding='same', activation='relu',
        kernel_initializer='he_normal')(X)

    firstpool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(firstconv)

    seconv = K.layers.Conv2D(filters=16, kernel_size=(
        5, 5), padding='valid', activation='relu',
        kernel_initializer='he_normal')(firstpool)

    secpool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(seconv)

    flat = K.layers.Flatten()(secpool)

    firstfull = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer='he_normal')(flat)
    
    secfull = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer='he_normal')(firstfull)

    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer='he_normal')(secfull)

    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
