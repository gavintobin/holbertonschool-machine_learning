#!/usr/bin/env python3
'''task 3'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''backprops over pooling layer'''
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    daprev = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for ll in range(c):
                    h_beg = j * sh
                    h_end = h_beg + kh
                    w_beg = k * sw
                    w_end = w_beg + kw

                    if mode == 'max':
                        mask = (A_prev[i, h_beg:h_end,
                                       w_beg:w_end, ll] == np.max(
                            A_prev[i, h_beg:h_end,
                                   w_beg:w_end, ll]))
                        daprev[i, h_beg:h_end, w_beg:w_end,
                                ll] += mask * dA[i, j, k, ll]
                    elif mode == 'avg':
                        val = dA[i, j, k, ll] / (kh * kw)
                        daprev[i, h_beg:h_end, w_beg:w_end,
                                ll] += np.ones((kh, kw)) * val
    return daprev
