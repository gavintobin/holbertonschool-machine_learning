#!/usr/bin/env python3
'''task 1'''
import numpy as np


class GaussianProcess():
    '''gaussian class'''

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        '''noiseless 1d gaussian process'''
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        '''calcs covariance matrix'''
        m, n = X1.shape[0], X2.shape[0]
        K = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                d = np.linalg.norm(X1[i] - X2[j])
                K[i, j] = self.sigma_f**2 * np.exp(-0.5 * (d / self.l)**2)

        return K

    def predict(self, X_s):
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.flatten(), np.diag(cov_s)
