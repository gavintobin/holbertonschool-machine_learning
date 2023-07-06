#!/usr/bin/env python3
'''task 1'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False,
                filepath=None, verbose=True,
                shuffle=False):
    '''same but savingbest it'''

    list = []

    if validation_data and early_stopping:
        list.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              atience=patience))

    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Scheduler"""
            return (alpha / (1 + decay_rate * epoch))
        list.append(K.callbacks.LearningRateScheduler(scheduler,
                                                      verbose=1))

    if save_best:
        list.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                save_best_only=True))

    model = network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=list)
    return model
