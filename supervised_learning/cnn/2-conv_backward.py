#!/usr/bin/env python3
'''task 2'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''backprops over hidden layers'''
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev -1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    
    paddeda_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          mode='constant')
    dw = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for j in range (h_new):
            for k in range(w_new):
                for ll in range(c_new):
                    h_beg = j * sh
                    h_end = h_beg + kh
                    w_beg = k * sw
                    w_end = w_beg + kw

                    asliced = paddeda_prev[i, h_beg:h_beg, w_beg:w_end, :]
                    val = dZ[i, j, k, ll]
                    paddeda_prev[i, h_beg:h_end, w_beg:w_end, :] += W[:, :, :, ll] * val[np.newaxis, np.newaxis]
                    dw[:, :, :, ll] += asliced * val
                    db[:, :, :, ll] += val
    if padding == 'same':
        daprev = paddeda_prev[:, ph:-ph, pw:-pw, :]
    elif padding == 'valid':
        daprev = paddeda_prev
    
    return daprev, dw, db




