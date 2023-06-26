#!/usr/bin/env python3
'''task 2'''
import numpy as np

def sensitivity(confusion):
    '''calcs sens of each class'''
    classes = confusion.shape[0]
    sensitivity_values = np.zeros(classes, dtype=np.float32)

    for i in range(classes):
        true_positives = confusion[i, i]
        actual_positives = np.sum(confusion[i, :])
        sensitivity_values[i] = true_positives / actual_positives

    return sensitivity_values
