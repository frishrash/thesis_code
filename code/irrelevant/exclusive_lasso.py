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
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import cycle, islice
from random import randint
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition.nmf import _update_coordinate_descent, check_array, check_random_state, _initialize_nmf

def nmf_solve(X, n_clusters, gamma=0.5):
    """ Balanced K-means based on exclusive lasso regulator using NMF
    """

    random_state = None
    
    W, H = _initialize_nmf(X, n_components=n_clusters, init='random',
                       random_state=random_state)
    
    Ht = check_array(H.T, order='C')
    X = check_array(X, accept_sparse='csr')

    # L1 and L2 regularization
    l1_H, l2_H, l1_W, l2_W = 0, 0, 0, 0
    update_H = True
    shuffle = False
    verbose = True
    tol = 1e-4
    
    max_iter = 200

#==============================================================================
#     if regularization in ('both', 'components'):
#         alpha = float(alpha)
#         l1_H = l1_ratio * alpha
#         l2_H = (1. - l1_ratio) * alpha
#     if regularization in ('both', 'transformation'):
#         alpha = float(alpha)
#         l1_W = l1_ratio * alpha
#         l2_W = (1. - l1_ratio) * alpha
#==============================================================================

    rng = check_random_state(random_state)

    for n_iter in range(max_iter):
        violation = 0.

        # Update W
        violation += _update_coordinate_descent(X, W, Ht, l1_W, l2_W,
                                                shuffle, rng)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, W, l1_H, l2_H,
                                                    shuffle, rng)

        if n_iter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def exclusive_lasso(X, n_clusters, gamma=0.5):
    """ Balanced K-means based on exclusive lasso regulator

    X - data matrix, dimensions: d x n
    """
    MAX_ITERS = 100
    n_dims, n_samples = X.shape

    # Initialize indicaotr matrix F (dimensions n x k)
    F = np.zeros((n_samples, n_clusters), dtype=np.int8)
    print F.shape
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
                res = np.power(np.linalg.norm(X-H.dot(F.T)), 2) + \
                    gamma * F.T.dot(np.ones([n_samples, n_samples])
                                    ).dot(F).trace()
                results.append(res)
                F[i][j] = 0
            new_ind = np.argmin(results)
            F[i][new_ind] = 1
            if (curr_ind != new_ind):
                conv = False
        iteration = iteration + 1
        print iteration
    return map(lambda x: x.index(1), F.tolist())


def exclusive_lasso3(X, n_clusters, gamma=0.5):
    """ Balanced K-means based on exclusive lasso regulator

    X - data matrix, dimensions: features x samples
    """
    MAX_ITERS = 100
    n_dims, n_samples = X.shape

    # Initialize indicaotr matrix F
    F = np.zeros((n_samples, n_clusters), dtype=np.int8)
    for i in xrange(n_samples):
        F[i, randint(0, n_clusters-1)] = 1

    conv = False
    iteration = 0
    while iteration < MAX_ITERS and not conv:
        conv = True
        # Calculate H = XF(F^TF)^-1
        H = X.dot(sp.linalg.pinv(F.T))

        # Caculate F
        # For each row (sample)
        for i in xrange(n_samples):
            # Get current indicator
            curr_ind = F[i].nonzero()[0][0]

            # We want to find the indicator column that minimizes
            # X-HF^T + gamma * Tr(F^T11^TF)
            #
            # The trace equals sum of squared samples in each cluster
            F[i, curr_ind] = 0

            results = []
            tmpH = H.dot(F.T)
            tmpH = np.delete(tmpH, i, axis=1)  # Delete i-th column from HF^T
            tmpX = np.delete(X, i, axis=1)  # Delete i-th sample from X
            base_sum = ((tmpX-tmpH)**2).sum()

            for j in xrange(n_clusters):
                F[i, j] = 1
                tr = (F.sum(axis=0)**2).sum()
                res = base_sum + ((X[:, i]-H[:, j])**2).sum() + gamma * tr
                results.append(res)
                F[i, j] = 0

            new_ind = np.argmin(results)
            F[i, new_ind] = 1
            if (curr_ind != new_ind):
                conv = False
        iteration = iteration + 1
        print iteration
    return map(lambda x: x.index(1), F.tolist())


def exclusive_lasso4(X, n_clusters, gamma=0.5):
    """ Balanced K-means based on exclusive lasso regulator

    X - data matrix, dimensions: features x samples
    """
    MAX_ITERS = 100
    n_dims, n_samples = X.shape

    # Initialize indicaotr matrix F
    F = np.zeros((n_samples, n_clusters), dtype=np.int8)
    for i in xrange(n_samples):
        F[i, randint(0, n_clusters-1)] = 1

    conv = False
    iteration = 0
    while iteration < MAX_ITERS and not conv:
        conv = True
        # Calculate H = XF(F^TF)^-1
        H = X.dot(sp.linalg.pinv(F.T))
        
        base_sum = ((X-H.dot(F.T))**2).sum(axis=0)
        F_counts = F.sum(axis=0)

        # Caculate F
        # For each row (sample)
        for i in xrange(n_samples):
            # Get current indicator
            curr_ind = F[i].nonzero()[0][0]
            F_counts[curr_ind] = F_counts[curr_ind] - 1

            results = []
            for j in xrange(n_clusters):
                F_counts[j] = F_counts[j] + 1
                tr = (F_counts**2).sum()
                res = ((X[:, i]-H[:, j])**2).sum() - base_sum[i] + gamma * tr
                results.append(res)
                F_counts[j] = F_counts[j] - 1
            new_ind = np.argmin(results)
            F[i, curr_ind] = 0
            F[i, new_ind] = 1
            F_counts[new_ind] = F_counts[new_ind] + 1
            # We want to find the indicator column that minimizes
            # X-HF^T + gamma * Tr(F^T11^TF)
            #
            # The trace equals sum of squared samples in each cluster
            
            if (curr_ind != new_ind):
                conv = False
        iteration = iteration + 1
        print iteration
    return map(lambda x: x.index(1), F.tolist())


def exclusive_lasso5(X, n_clusters, gamma=0.5):
    """ Balanced K-means based on exclusive lasso regulator

    X - data matrix, dimensions: features x samples
    """
    MAX_ITERS = 100
    n_dims, n_samples = X.shape

    # Initialize indicaotr matrix F
    F = np.zeros((n_samples, n_clusters), dtype=np.int8)
    for i in xrange(n_samples):
        F[i, randint(0, n_clusters-1)] = 1

    conv = False
    iteration = 0
    results = np.zeros(n_clusters, dtype=np.float64)

    while iteration < MAX_ITERS and not conv:
        conv = True
        # Calculate H = XF(F^TF)^-1
        H = X.dot(sp.linalg.pinv(F.T))

        base_sum = ((X-H.dot(F.T))**2).sum(axis=0)
        F_counts = F.sum(axis=0)

        # Caculate F
        # For each row (sample)
        for i in xrange(n_samples):
            # Get current indicator
            curr_ind = F[i].nonzero()[0][0]
            F_counts[curr_ind] = F_counts[curr_ind] - 1

            # We want to find the indicator column that minimizes
            # X-HF^T + gamma * Tr(F^T11^TF)
            for j in xrange(n_clusters):
                F_counts[j] = F_counts[j] + 1
                tr = (F_counts**2).sum()
                res = ((X[:, i]-H[:, j])**2).sum() - base_sum[i] + gamma * tr
                results[j] = res
                F_counts[j] = F_counts[j] - 1
            new_ind = results.argmin()

            F_counts[new_ind] = F_counts[new_ind] + 1

            if (curr_ind != new_ind):
                conv = False
                F[i, curr_ind] = 0
                F[i, new_ind] = 1
        iteration = iteration + 1
        print iteration
    return map(lambda x: x.index(1), F.tolist())


def exclusive_lasso6(X, n_clusters, gamma=0.1):
    """Balanced K-means using an exclusive LASSO regulator [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Input data.
    n_clusters : int
        Number of clusters.
    gamma: float, optional
        Controls the regulator, the bigger the more balanced clustering but
        less seperation between clusters. Default: 0.1.

    Returns
    -------
    out : ndarray, shape (n_samples,)
        Clusters assignments

    Notes
    -----
    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    References
    ----------
    .. [1] Chang, Xiaojun, et al., "Balanced k-means and min-cut clustering."
        arXiv preprint arXiv:1411.6235, 2014.
    """

    MAX_ITERS = 100
    n_dims, n_samples = X.shape

    # F is indicaotr matrix with dimensions <samples> x <clusters>
    # Each row indicates data point, the column with 1 indicates the cluster
    # it is assigned to. Rest of columns must be 0.

    # Initialize F with random clusters assignment
    F = np.zeros((n_samples, n_clusters), dtype=np.int8)
    for i in xrange(n_samples):
        F[i, randint(0, n_clusters-1)] = 1

    conv = False
    iteration = 0
    while iteration < MAX_ITERS and not conv:
        conv = True  # Assume convergence, unless F changes during iteration
        H = X.dot(sp.linalg.pinv(F.T))  # H = XF(F^TF)^-1

        # Now we should fixate H and update F row by row, on each row we should
        # assign the column that minimizes ||X-HF^T||^2_F + gamma*Tr(F^T11^TF).
        # Notice that:
        # ||X-HF^T||^2_F equals sum of squared elements of the matrix X-HF^T.
        # Tr(F^T11^TF) equals sum of squared clusters' sizes.

        # Now X and H don't change, only F. If F's i-th row and j-th column is
        # one, then (HF^T)'s i-th column is a copy of H's j-th column.

        # We first calculate sum of squared elements of matrix X-HF^T of
        # every column, before F changed.
        base_sum = ((X-H.dot(F.T))**2).sum(axis=0)  # Sum per column
        total_base_sum = base_sum.sum()  # Total sum

        # For trace calculation, we need sum of squared clusters' sizes, which
        # is the sum of F's columns, squared, and summarized once again.
        # We save the repeated summary over F's columns with a dedicated array.
        F_counts = F.sum(axis=0)

        # Calculate F
        # For each row in F
        for i in xrange(n_samples):
            # Get the indicator (non-zero column) of current (i-th) row
            curr_ind = F[i].nonzero()[0][0]

            # Update F's columns sums, as if current row is all zeros
            F_counts[curr_ind] = F_counts[curr_ind] - 1

            # To save looping over F's columns, we calculate the trace value
            # for all possible indicator columns at once using a matrix.

            # Initialize F_mat, all rows are duplicates of F_counts which is
            # the sum of F's columns when current row in F is all zeros.
            F_mat = np.repeat(F_counts, n_clusters).reshape(n_clusters,
                                                            n_clusters).T
            # On every row of F_mat, add one to a different column.
            F_mat = F_mat + np.eye(n_clusters)

            # Now the j-th row of F_mat represent the sum of F's columns when
            # current row in F has the j-th column set as indicator.
            # We square every element and sum the rows. In the end, traces
            # contains <n_clusters> values, each is the trace value for a
            # different indicator column.
            traces = (F_mat**2).sum(axis=1)

            res = np.repeat(X[:, i], n_clusters).reshape(n_dims, n_clusters)
            res = np.power(res - H, 2).sum(axis=0) - base_sum[i] + total_base_sum + gamma * traces
            new_ind = res.argmin()
            F_counts[new_ind] = F_counts[new_ind] + 1

            if (curr_ind != new_ind):
                conv = False
                F[i, curr_ind] = 0
                F[i, new_ind] = 1
        iteration = iteration + 1
        print iteration
    return F.nonzero()[1]

np.random.seed(0)
n_samples = 500
n_clusters = 3

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=175, n_features=3)

X, y = varied

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=n_clusters)
y_pred = km.fit_predict(X)
y_pred = exclusive_lasso6(np.array(X.T), n_clusters, gamma=0.05)
#y_pred = nmf_solve(np.array(X), n_clusters, gamma=0.1)
print(np.unique(y_pred, return_counts=True))

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color=colors[y_pred])

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xticks(())
plt.yticks(())

