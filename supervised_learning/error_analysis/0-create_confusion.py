#!/usr/bin/env python3
'''task 1'''
import numpy as np


def  create_confusion_matrix(labels, logits):
    '''creates a confusion matrix'''
    m, classes = labels.shape


    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)


    confusion = np.zeros((classes, classes), dtype=np.int32)


    for i in range(m):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        confusion[true_label, predicted_label] += 1

    return confusion
