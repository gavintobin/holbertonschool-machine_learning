#!/usr/bin/env python3
'''task 3'''
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''bayes opt class'''
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        '''innit func'''
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        '''acusitiom func'''
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        Z = imp / (sigma + 1e-8)
        EI = imp * (norm.cdf(Z) + 1e-8) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        '''optomidse func'''
        X_opt, Y_opt = None, None

        for i in range(iterations):
            X_next, _ = self.acquisition()
            # Convert X_next from a 2D array to a 1D array

            if X_next in self.gp.X:
                # If the next point has already been sampled, stop early
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

            if Y_opt is None or Y_next < Y_opt:
                X_opt, Y_opt = X_next, Y_next

        return X_opt, Y_opt
