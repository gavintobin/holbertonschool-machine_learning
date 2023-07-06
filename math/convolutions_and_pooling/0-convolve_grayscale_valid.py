#!/usr/bin/env python3
"""task 1"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
   ''' performs valid convo of greyscale pics'''
   m, h, w = images.shape
   kh, kw = kernel.shape
   H = h - kh + 1
   W = w - kw + 1
   
   img = np.zeros((m, H, W))
   for i in range(H):
       for j in range(W):
            p = images[:, i:i+kh, j:j+kw]
            img[:, i, j] = np.sum(p * kernel, axis=(1, 2))
   return img
