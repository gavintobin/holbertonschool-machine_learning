#!/usr/bin/env python3
'''task 1'''
import numpy as np


def pca(X, var=0.95):
    ''' performs PCA on dataset'''
    n, d = X.shape

    #conv. mat to covar mat
    covarmat = cov(X)
    #compute evalsk and evectss

    evall, evect = np.linalg.eigh(covarmat)
    #sort them in desc order
    
    indicessorted = np.argsort(evall)[ : :-1]
    evalls = evalls[indicessorted]
    evect = evect[:, indicessorted]

    #calc explained var
    vari = np.sum(evalls)
    varratio = evalls / vari

    #determine which to keep
    cumivar = np.cumsum(varratio)
    keep = np.argmax(cumivar >= var) + 1

    #most important components
    mostimp = evect[:, :keep]
    #weights mat

    W = mostimp
    return W


def cov(x):
    '''covariance helper func'''
    n, d = x.shape
    mean = np.mean(x, axis=0)  # Calculate the mean along each dimension

    # Subtract the mean from each data point
    centered_data = x - mean

    # Compute the covariance matrix
    cov_matrix = np.dot(centered_data.T, centered_data) / (n - 1)

    return cov_matrix

