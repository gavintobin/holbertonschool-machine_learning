#!/usr/bin/env python3
"""task 1"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''performs same convo'''
    m, h, w = images.shape
    kh, kw = kernel.shape

    H_PAD = kh // 2
    W_PAD = kw // 2

    PADDED = np.pad(images, ((0, 0), (H_PAD, H_PAD),
                             (W_PAD, W_PAD)), mode='constant')

    CONVO = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            img = PADDED[:, i:i+kh, j:j+kw]
            CONVO[:, i, j] = np.sum(img * kernel, axis=(1, 2))

    return CONVO
