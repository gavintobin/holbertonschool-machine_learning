#!/usr/bin/env python3
'''task 2'''
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''forward prop over pooling layers'''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kw) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    newa = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for ll in range(c_prev):
                    h_beg = j * sh
                    h_end = h_beg + kh
                    w_beg = k * sw
                    w_end = w_beg + kw

                    aslice = A_prev[i, h_beg:h_end, w_beg:w_end, ll]

                    if mode == 'max':
                        newa[i, j, k, ll] = np.max(aslice)
                    elif mode == 'avg':
                        newa[i, j, k, ll] = np.mean(aslice)
    return newa
