# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 16:07:39 2017

@author: gal
"""

import time
import warnings

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import cycle, islice
from random import randint


def exclusive_lasso(X, n_clusters, gamma=0.5):
    """ Balanced K-means based on exclusive lasso regulator

    X - data matrix, dimensions: d x n
    """
    MAX_ITERS = 100
    n_dims, n_samples = X.shape

    # Initialize indicaotr matrix F (dimensions n x k)
    F = np.zeros((n_samples, n_clusters), dtype=np.int8)
    for i in xrange(n_samples):
        F[i][randint(0, n_clusters-1)] = 1

    conv = False
    iteration = 0
    while iteration < MAX_ITERS and not conv:
        conv = True
        # Calculate H
        H = X.dot(F.dot(np.linalg.inv(F.T.dot(F))))

        # Caculate F
        # For each row
        for i in xrange(n_samples):
            # curr_ind = F[i].tolist().index(1)
            curr_ind = F[i].nonzero()[0][0]
            # Find the indicator column that minimizes X-HF^T + gTr(F^T11^TF)
            F[i] = [0]*n_clusters
            results = []
            for j in xrange(n_clusters):
                F[i][j] = 1
                res = np.linalg.norm(X-H.dot(F.T)) + \
                    gamma*F.T.dot(np.ones([n_samples, n_samples])).dot(F).trace()
                results.append(res)
                F[i][j] = 0
            new_ind = np.argmin(results)
            F[i][new_ind] = 1
            if (curr_ind != new_ind):
                conv = False
        iteration = iteration + 1
        print iteration
    return map(lambda x: x.index(1), F.tolist())

np.random.seed(0)
n_samples = 300
n_clusters = 4

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=175)

X, y = varied

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=n_clusters)
y_pred = km.fit_predict(X)
y_pred = exclusive_lasso(X.T, n_clusters, gamma=0.009)
print(np.unique(y_pred, return_counts=True))

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))
plt.figure()

plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xticks(())
plt.yticks(())

