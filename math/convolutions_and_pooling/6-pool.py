#!/usr/bin/env python3
"""task 1"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''performs pooling'''

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    POOL = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            img = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            if mode == 'max':
                POOL[:, i, j, :] = np.max(img, axis=(1, 2))
            elif mode == 'avg':
                POOL[:, i, j, :] = np.mean(img, axis=(1, 2))

    return POOL
