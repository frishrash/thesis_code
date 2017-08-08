# -*- coding: utf-8 -*-
"""
Created on Sun Aug 06 21:05:58 2017

@author: gal
"""

import numpy as np
import scipy as sp
import pandas as pd
import dataset as ds
from dataset import NSL

def bcls_alm(X, Y, gamma, lam, mu):
    """ BCLS optimization

    X is data given as d by n (Pandas default is n by d)
    Y is indicator label matrix given as n by c
    """
    X = np.matrix(X.T)  # X is d by n and Pandas default is n by d
    Y = np.matrix(Y)
    iterations = 600
    dim, n = X.shape
    H = sp.sparse.identity(n) - 1.0/n*np.ones([n, n])
    X = X.dot(H)

    c = Y.shape[1]  # Number of clusters
    Lambda = np.zeros((n, c))
    rho = 1.005
    P = np.identity(dim).dot(np.linalg.inv(X.dot(X.T)+gamma*np.identity(dim)))
    # P = np.nan_to_num(P)

    objs = []

    for i in xrange(iterations):
        # print('Iteration %d' % i)

        # Solve W and b
        W = P.dot(X.dot(Y))
        b = np.matrix(y.mean(axis=0)).T
        E = X.T.dot(W) + np.ones((n, 1)).dot(b.T) - Y

        # Solve Z
        Z1 = (-2*lam*np.ones((n, n)) + (mu+2*n*lam)*sp.sparse.identity(n)) / \
             (np.power(mu, 2) + 2*n*lam*mu)
        Z2 = (mu*Y + Lambda)
        Z = Z1.dot(Z2)

        # Solve Y
        V = 1/(2+mu)*(2*X.T.dot(W) + 2*np.ones((n, 1))*b.T + mu*Z - Lambda)
        ind = np.argmax(V, axis=1)
        Y = np.zeros((n, c))
        for i, i2 in enumerate(ind):
            Y[i][i2] = 1

        # Update Lambda and mu according to ALM
        Lambda = Lambda + mu*(Y-Z)
        mu = np.min([mu*rho, 100000])

        # Objective value
        val = E.T.dot(E).trace() + gamma*(W.T.dot(W).trace()) + \
            lam*Y.T.dot(np.ones((n, n))).dot(Y).trace()
        objs.append(val.tolist()[0][0])
    return(Y)

samples = 100
clusters = 5
x = NSL(ds.NSL_TRAIN20, encoding=ds.ENC_NUMERIC, scaling=ds.SCL_NONE)
x = x.ds.iloc[range(0, samples)]
x = x - x.mean()
y = np.zeros((samples, clusters))
for i in xrange(samples):
    y[i][np.random.randint(clusters)] = 1
    #y[i][1] = 1
res = bcls_alm(x, y, 0.01, 0.3, 0.1)
print(res)