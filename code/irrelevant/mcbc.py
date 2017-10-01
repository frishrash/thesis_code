# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 06:08:43 2017

@author: gal
"""

import time
import warnings

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import cycle, islice
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min
from sklearn.neighbors import KDTree


def mcbc(data, n_clusters, m, max_iter=100, tol=1e-4):
    n_samples, n_dims = data.shape
    km = KMeans(n_clusters=n_clusters, max_iter=1)
    centers = km.fit(data).cluster_centers_

    for iter in xrange(max_iter):
        # Assign points to clusters
        nearest_center, dists = pairwise_distances_argmin_min(data, centers)
        is_unfree = np.zeros([n_samples])

        mh = np.zeros([n_clusters])
        iter_cluster_assignments = np.zeros([n_samples], dtype=np.uint)

        unsatisified = 0
        for cluster in xrange(n_clusters):
            members = np.where(nearest_center == cluster)[0]

            if len(members) > m:
                tree = KDTree(np.insert(data[[members]], 0,
                                        centers[cluster].reshape(1, -1), axis=0))
                # Potential bug that needs fixing tree.query may return distances
                # to other centers rather than datapoints
                dist, ind = tree.query(centers[cluster].reshape(1, -1), k=m+1)
                is_unfree[members[ind[0][1:]-1]] = 1
                mh[cluster] = 0
            else:
                is_unfree[[members]] = 1
                mh[cluster] = m - len(members)

            unsatisified = unsatisified + mh[cluster]
            iter_cluster_assignments[[members]] = cluster

        hneari = np.zeros([n_samples], dtype = np.uint)
        hsupporti = np.zeros([n_samples], dtype = np.uint)
        dists = euclidean_distances(data, centers)        

        print("Iteration %d, unsatisfied: %d" % (iter, unsatisified))
        while unsatisified > 0:
            
            #for cluster in xrange(n_clusters):
            #    members = np.where(iter_cluster_assignments == cluster)[0]
            #    print("Cluster %d:" % cluster)
            #    print("Size: %d, unfree: %d, mh: %d" %
            #          (len(members),
            #           np.sum(is_unfree[[members]]),
            #           mh[cluster]))            
            
            
            xi0 = -1
            min_xi0 = np.inf
            for sample in xrange(n_samples):
                if not is_unfree[sample]:
                    dneari = np.min(dists[sample])
                    hneari[sample] = np.argmin(dists[sample])
                    
                    mh_candidates_ind = mh.nonzero()[0]
                    dsupporti = np.min(dists[sample][[mh_candidates_ind]])
                    mh_min_ind = np.argmin(dists[sample][[mh_candidates_ind]])
                    hsupporti[sample] = mh_candidates_ind[mh_min_ind]

                    diff = dsupporti**2 - dneari**2
                    if diff < min_xi0:
                        min_xi0 = diff
                        xi0 = sample
           
            is_unfree[xi0] = 1
            iter_cluster_assignments[xi0] = hsupporti[xi0]
            if mh[hsupporti[xi0]] > 0:
                mh[hsupporti[xi0]] = mh[hsupporti[xi0]] - 1
            unsatisified = unsatisified - 1
            #print("Moving sample %d from cluster %d to cluster %d" %
            #      (xi0, hneari[xi0], hsupporti[xi0]))

        # Update centers
        new_centers = []
        for cluster in xrange(n_clusters):
            members = np.where(iter_cluster_assignments == cluster)[0]
            #print(len(members))
            #print("Old centers for cluster %d:" % cluster)
            #print(centers[cluster])
            new_centers.append(np.mean(data[[members]], axis=0))
        new_centers = np.array(new_centers)
        center_shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=0))
        center_shift_total = np.sum(center_shift)
        print(center_shift_total)
        if center_shift_total ** 2 < tol:
            print("center shift %e within tolerance %e" % (center_shift_total, tol))
            break
        centers = new_centers
            #print("New centers for cluster %d:" % cluster)
            #print(centers[cluster])
        #print(mh)
        #print(np.unique(iter_cluster_assignments, return_counts=True))
        #print
    return iter_cluster_assignments

np.random.seed(0)
n_samples = 1000
n_clusters = 4

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=175, n_features=3)

X, y = varied

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)


y_pred = mcbc(X, n_clusters, m=n_samples / n_clusters)
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