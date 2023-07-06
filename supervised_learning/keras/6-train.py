#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    '''inttroduces early stopping now'''
    if validation_data and early_stopping:
        EARLY = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                          patience=patience)
        return network.fit(x=data,
                           y=labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle,
                           validation_data=validation_data,
                           callbacks=[EARLY])
    else:
        return network.fit(x=data,
                           y=labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle,
                           validation_data=validation_data)
