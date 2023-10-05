#!/usr/bin/env python3
'''task 3'''
import sklearn.mixture


def gmm(X, k):
   '''does gmm easy way'''
    gmm = sklearn.mixture.GaussianMixture(n_components=k)

    gmm.fit(X)

    pie = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_

    clss = gmm.predict(X)

    bic = gmm.bic(X)

    return pie, m, S, clss, bic
