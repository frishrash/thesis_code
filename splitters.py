# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:16:15 2017

@author: gal
"""

from random import randint
import numpy as np
from sklearn.cluster import KMeans
from dataset import split_ds


class ClusterWrapper:
    def __init__(self, class_name, *args, **kwargs):
        self.algo = class_name(*args, **kwargs)
        self.name = self.algo.__class__.__name__
        self.k = self.algo.n_clusters
        self.info = ''
        self.features = None
        self.features_desc = 'All'
        if ("fit_predict" not in dir(self.algo)):
            raise Exception('No fit_predict method for %s' % self.name)

        if hasattr(self.algo, 'n_clusters'):
            self.info = 'k=%d' % self.algo.n_clusters

    def limit_features(self, features, features_desc):
        self.features = features
        self.features_desc = features_desc
        return self

    def split(self, ds):
        if self.features is None:
            clustering_result = self.algo.fit_predict(ds)
        else:
            # Remove features not in dataset columns, e.g., if dataset
            # is numeric and some requested features are not
            features = set(self.features)
            non_existing_features = features - set(ds.columns)
            features = features - non_existing_features

            # Try to add features that their names are substrings of existing
            # dataset columns, e.g., if features were hot-encoded
            for x in non_existing_features:
                features = features | set(ds.columns[
                                            ds.columns.str.contains(x)])
            clustering_result = self.algo.fit_predict(ds[list(features)])

        # Save clustering result - list size of ds with values as cluster ids
        self.clustering_result = clustering_result

        # Return actual clusters as list of Pandas DataFrames
        return split_ds(clustering_result, ds)


class KMeansBal:
    def __init__(self, n_clusters=8, clusters_factor=3, init='k-means++',
                 n_init=10, max_iter=300, tol=1e-4,
                 precompute_distances='auto', verbose=0, random_state=None,
                 copy_x=True, n_jobs=1, algorithm='auto'):
        self.n_clusters = n_clusters
        self.algo = KMeans(n_clusters=n_clusters*clusters_factor, init=init,
                           n_init=n_init, max_iter=max_iter, tol=tol,
                           precompute_distances=precompute_distances,
                           verbose=verbose, random_state=random_state,
                           copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        result = self.algo.fit_predict(X, y)
        clusters_map, cluster_sizes = self._balance_clusters(result,
                                                             self.n_clusters)

        # Build reverse map for translating old clusters to new ones
        dict = {}
        for i, d in enumerate(clusters_map):
            for x in d:
                dict[x] = i

        return [dict[x] for x in result]

    def _balance_clusters(self, clusters, num_clusters):
        labels, counts = np.unique(clusters, return_counts=True)
        total_samples = counts.sum()
        ideal_cluster_size = total_samples / num_clusters
        new_clusters_map = [[] for _ in xrange(num_clusters)]
        new_clusters_sizes = [0] * num_clusters

        for cls_ind in np.argsort(counts)[::-1][:len(labels)]:
            cls_size = counts[cls_ind]
            cls_label = labels[cls_ind]

            new_cluster = -1
            for i in np.argsort(new_clusters_sizes)[::-1][:num_clusters]:
                if (new_clusters_sizes[i] + cls_size <= ideal_cluster_size):
                    new_cluster = i
                    break
            if (new_cluster == -1):
                new_cluster = np.argmin(new_clusters_sizes)
            new_clusters_sizes[new_cluster] += cls_size
            new_clusters_map[new_cluster].append(cls_label)

        return (new_clusters_map, new_clusters_sizes)


class RoundRobin:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, ds):
        return [i % self.n_clusters for i in xrange(ds.shape[0])]


class RandomRoundRobin:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, ds):
        return [randint(0, self.n_clusters-1) for i in xrange(ds.shape[0])]


class NoSplit:
    def __init__(self, n_clusters):
        self.n_clusters = 1

    def fit_predict(self, ds):
        return [0 for _ in xrange(ds.shape[0])]
