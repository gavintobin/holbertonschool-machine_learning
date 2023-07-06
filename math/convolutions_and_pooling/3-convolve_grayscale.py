#!/usr/bin/env python3
"""task 1"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    '''performs greyscale convo'''

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh) + kh - h) // 2 + 1
        pw = (((w - 1) * sw) + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    images = np.pad(images, ((0, 0), (ph, ph),
                             (pw, pw)), mode='constant')

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    CONVO = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            CONVO[:, i, j] = np.sum(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
                                    * kernel, axis=(1, 2))

    return CONVO
