#!/usr/bin/env python3
'''neuron class that defines singleneuron performing binary classification'''
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    '''neauron class'''
    def __init__(self, nx):
        '''innit'''
        self.__A = 0
        self.__b = 0
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.randn(1, nx)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calcs gd"""
        m = Y.shape[1]
        dZ = A - Y
        dW = (1/m) * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ)
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def cost(self, Y, A):
        '''calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def forward_prop(self, X):
        '''forward prop function'''
        self.__A = self.sig(np.matmul(self.__W, X) + self.__b)
        return (self.__A)

    def sig(self, x):
        '''sigmoid helper func'''
        return 1/(1 + np.exp(-x))

    def evaluate(self, X, Y):
        '''evaluates the predictions'''
        m = X.shape[1]
        self.forward_prop(X)
        pred = np.where(self.__A > 0.5, 1, 0)
        cst = self.cost(Y, self.__A)
        return pred, cst

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''trains neuron'''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if type(step) is not int:
            raise TypeError('step must be an integer')
        if step <= 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')

        cstval = []

        for i in range(iterations + 1):
            propped = self.forward_prop(X)
            self.gradient_descent(X, Y, propped, alpha)

            if verbose is True and i % step == 0:
                cst = self.cost(Y, self.__A)
                cstval.append(cst)
                print('cost after {} interations: {}'.format(i, cst))

        if graph:
            it_ax = range(0, iterations + 1, step)
            plt.plot(it_ax, cstval, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        pred, cst = self.evaluate(X, Y)
        return pred, cst

    @property
    def W(self):
        '''gets weight'''
        return self.__W

    @property
    def b(self):
        '''gets bias'''
        return self.__b

    @property
    def A(self):
        '''gets a out'''
        return self.__A
