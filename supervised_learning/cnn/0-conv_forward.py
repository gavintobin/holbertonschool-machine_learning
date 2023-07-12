#!/usr/bin/env python3
'''task 1'''
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''forward prop of conv. layer of nn'''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        pw = 0, 0

    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / kw) + 1

    paddeda_prev = np.pad(A_prev, ((0, 0), (ph, ph),
                                   (pw, pw), (0, 0)), mode='constant')
    z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                i_beg, i_end = i * sh, i * sh + kw
                j_beg, j_end = j * sw, j * sw + kw
                aslice = paddeda_prev[:, i_beg:i_end, j_beg:j_end, :]
                z[:, i, j, k] = np.sum(aslice * W[:, :, :, k],
                                       axis=(1, 2, 3)) + b[:, :, :, k]
    A = activation(z)
    return A
