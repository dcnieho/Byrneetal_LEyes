# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as alg

def biqubic_calibration_with_cross_term(x, y, Y):
    X = np.zeros((len(x), 6))
    X[:, 0] = 1
    X[:, 1] = x
    X[:, 2] = y
    X[:, 3] = x ** 2
    X[:, 4] = y ** 2
    X[:, 5] = x * y

    x_1 = np.dot(X.T, X)
    x_2 = np.dot(alg.inv(x_1), X.T)
    coeff = np.dot(x_2, Y)

    return coeff

def biqubic_estimation_with_cross_term(x, y, coeff):
    X = np.zeros((len(x), 6))
    X[:, 0] = 1
    X[:, 1] = x
    X[:, 2] = y
    X[:, 3] = x ** 2
    X[:, 4] = y ** 2
    X[:, 5] = x * y

    return np.dot(X, coeff)