#!/usr/bin/env python3
"""deep neural network w binary classif."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    '''dnn class'''
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(i + 1)] = w
            else:
                prev = layers[i-1]
                w = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
                self.__weights['W' + str(i + 1)] = w
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

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
        def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Lets train our neural network.
        X is the input data
        Y is the correctly labeled data for the input data
        iterations is how many times we're going to train
        alpha is our learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        graphx = []
        graphy = []
        for i in range(0, iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if verbose:
                if i == 0 or i % step == 0:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))
            if graph:
                if i == 0 or i % step == 0:
                    current_cost = self.cost(Y, A)
                    graphy.append(current_cost)
                    graphx.append(i)
                plt.plot(graphx, graphy)
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
            if verbose or graph:
                if type(step) is not int:
                    raise TypeError("step must be in integer")
                if step <= 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
        if graph:
            plt.show()
        return (self.evaluate(X, Y))

    def evaluate(self, X, Y):
        '''evaluates the predictions'''
        A, _ = self.forward_prop(X)
        pred = np.where(A > 0.5, 1, 0)
        cst = self.cost(Y, A)
        return pred, cst

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calcs gd"""
        m = Y.shape[1]
        L = self.__L

        A = cache["A" + str(L)]
        dZ = A - Y

        for i in range(L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

            if i > 1:
                dZ = dA * (A_prev * (1 - A_prev))

    def cost(self, Y, A):
        '''calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def sig(self, x):
        '''sigmoid helper func'''
        return 1/(1 + np.exp(-x))

    def forward_prop(self, X):
        '''f prop func'''
        A = X
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W_curr = self.__weights['W' + str(i)]
            b_curr = self.__weights['b' + str(i)]
            Z = np.matmul(W_curr, A) + b_curr
            A = self.sig(Z)
            self.__cache['A' + str(i)] = A
        return (A, self.__cache)

    def save(self, filename):
        '''saves object to file'''
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        '''loads'''
        try:
            with open(filename, "rb") as file:
                obj = pickle.load(file)
                if isinstance(obj, DeepNeuralNetwork):
                    return obj
                else:
                    return None
        except FileNotFoundError:
            return None

    @property
    def L(self):
        """layer getter"""
        return self.__L

    @property
    def cache(self):
        '''itermed val getter'''
        return self.__cache

    @property
    def weights(self):
        '''weight getter'''
        return self.__weights
