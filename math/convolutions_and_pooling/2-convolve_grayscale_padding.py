#!/usr/bin/env python3
"""task 1"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''greyscale convo but w custom padding'''

    kh, kw = kernel.shape
    ph, pw = padding

    m = images.shape[0]
    h = images.shape[1] + (2 * padding[0]) - kh + 1
    w = images.shape[2] + (2 * padding[1]) - kw + 1

    PADDED = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    CONVO = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            img = PADDED[:, i:i + kh, j:j + kw]
            CONVO[:, i, j] = np.sum(img * kernel, axis=(1, 2))

    return CONVO
