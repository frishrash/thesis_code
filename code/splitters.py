# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:16:15 2017

@author: gal
"""

from random import randint
import metis
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
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


class MultiPart:
    """Multilevel k-way partitioning scheme for irregular graphs.

    This class produces a weighted graph. Each vertex represents a data point.
    Its edges represent distances to its N nearest neighbors. The graph is then
    partitioned using multilevel k-way partition scheme, minimizing edges' sum.

    Prerequisite packages:
        metis
        networkx

    More info on METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview

    Bibtex: G. Karypis and V. Kumar. Multilevel k-way partitioning scheme for
    irregular graphs. Journal of Parallel and Distributed Computing,
    48(1):96–129, 1998.
    """

    dataset_id_cache = None
    graph_cache = None

    def __init__(self, n_clusters=8, nearest_neighbors=100, leaf_size=50,
                 metric='euclidean', seed=None):
        self.n_clusters = n_clusters
        self.nearest_neighbors = nearest_neighbors
        self.leaf_size = leaf_size
        self.metric = metric
        self.seed = seed

    def fit_predict(self, X, y=None):
        """ Partitions a dataset """

        if (MultiPart.dataset_id_cache == id(X)):
            G = MultiPart.graph_cache
        else:
            # Build KD-Tree for efficient nearest-neighbors query
            kdt = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
            distances, indices = kdt.query(X, k=self.nearest_neighbors,
                                           return_distance=True)
            distances = MinMaxScaler(feature_range=(0, 1000)).fit_transform(
                distances)

            # Build graph for k-way paprtitioning
            G = nx.Graph()
            G.add_nodes_from(xrange(len(X)))

            # Add distances of nearest neighbors as weighted edges
            # The add_edge gets the tuple (index_from, index_to, distance)
            for i, x in enumerate(zip(distances, indices)):
                for j, dist in enumerate(x[0]):
                    # We insert opposite of distance since metis can only
                    # *minimze* edge-cut and we want the cut to be between
                    # least similar records (which have biggest distance)
                    G.add_edge(i, x[1][j], weight=int(1000-dist))
            G.graph['edge_weight_attr'] = 'weight'

            # Cache the graph
            MultiPart.dataset_id_cache = id(X)
            MultiPart.graph_cache = G

        # Parition the graph
        if (self.seed is None):
            (edgecuts, parts) = metis.part_graph(G, self.n_clusters,
                                                 objtype='cut')
        else:
            (edgecuts, parts) = metis.part_graph(G, self.n_clusters,
                                                 objtype='cut', seed=self.seed)
        return parts


class KMeansBal:
    def __init__(self, n_clusters=8, clusters_factor=3, init='k-means++',
                 n_init=10, max_iter=300, tol=1e-4,
                 precompute_distances='auto', verbose=0, random_state=None,
                 copy_x=True, n_jobs=1, algorithm='auto'):
        self.n_clusters = n_clusters
        self.km = KMeans(n_clusters=n_clusters*clusters_factor, init=init,
                         n_init=n_init, max_iter=max_iter, tol=tol,
                         precompute_distances=precompute_distances,
                         verbose=verbose, random_state=random_state,
                         copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        result = self.km.fit_predict(X, y)
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
        new_clusters_map = [[] for _ in xrange(num_clusters)]
        new_clusters_sizes = [0 for _ in xrange(num_clusters)]

        # Go over every original KMeans cluster, ordered from biggest to
        # smallest
        for cls_ind in np.argsort(counts)[::-1][:len(labels)]:
            cls_size = counts[cls_ind]
            cls_label = labels[cls_ind]

            # Add to smallest output cluster
            new_cluster = np.argmin(new_clusters_sizes)

            # Update output clusters map
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
