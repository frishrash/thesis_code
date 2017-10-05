# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:16:15 2017

@author: gal
"""

from random import randint
import time
import pandas as pd
import metis
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


class EXLasso:
    """Balanced K-means using an exclusive LASSO regulator [1]_.
    In addition to the convergence criteria described in the paper, convergence
    can be specified in terms of tolerance with regards to inertia (clusters
    centers shift).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    n_clusters : int
        Number of clusters.
    gamma: float, deafult: 0.1
        Controls the regulator, the greater the more balanced clustering but
        less seperation between clusters.
    init: {'kmeans', 'random'}, default: 'random'
        How to initialize the matrix F. Default is K-means.
    max_iter: int, default: 100
        Maximum number of iterations of this algorithm for a single run.
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is given,
        it fixes the seed. Defaults to the global numpy random number
        generator.
    tol: float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    Returns
    -------
    out : ndarray, shape (n_samples,)
        Cluster's assignments

    Notes
    -----
    Objective function:

    .. math:: \min_{F \in Ind} ||X-HF^T||^2_F + \gamma Tr(F^T11^TF)

    References
    ----------
    .. [1] Chang, Xiaojun, et al., "Balanced k-means and min-cut clustering."
        arXiv preprint arXiv:1411.6235, 2014.
    """

    def __init__(self, n_clusters=8, gamma=0.1, init='random', max_iter=100,
                 random_state=None, tol=1e-4, verbose=False):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

    def fit_predict(self, X, y=None):
        """ Partitions a dataset """
        if X.__class__ == pd.core.frame.DataFrame:
            X = X.as_matrix()
        X = X.T
        n_dims, n_samples = X.shape

        # F is indicaotr matrix with dimensions (n_samples, n_clusters)
        # Each row represents a sample. The column with 1 indicates the cluster
        # it is assigned to. Rest of columns must be 0.

        # Initialize F
        F = np.zeros((n_samples, self.n_clusters), dtype=np.int8)
        random_state = check_random_state(self.random_state)
        if (self.init == 'random'):
            for i in xrange(n_samples):
                F[i, random_state.randint(0, self.n_clusters - 1)] = 1
        else:
            km = KMeans(n_clusters=self.n_clusters, max_iter=1,
                        init='k-means++', random_state=self.random_state)
            pred = km.fit_predict(X.T)
            for i in xrange(n_samples):
                F[i][pred[i]] = 1

        I = np.eye(self.n_clusters)  # Simple identity matrix to be used later
        samples = np.ones(n_samples, dtype=np.bool)  # Sample ids for iteration
        X_dup = np.tile(X.T, self.n_clusters)  # X clone to be used later

        conv = False
        iteration = 0
        H = X.dot(np.linalg.pinv(F.T))  # H = XF(F^TF)^-1

        while iteration < self.max_iter and not conv:
            start = time.time()
            conv = True  # Assume convergence, unless F changes
            # We should fixate H and update F row by row, on each row we should
            # assign the column that minimizes the objective function.
            # Notice that:
            # ||X-HF^T||^2_F equals sum of squared elements of matrix X-HF^T.
            # Tr(F^T11^TF) equals sum of squared clusters' sizes.

            # We first calculate sum of squared elements of matrix X-HF^T of
            # every column, before F changes. We will use this later.
            base_sum = ((X - H.dot(F.T))**2).sum(axis=0)  # Sum per column
            total_base_sum = base_sum.sum()  # Total sum

            # For trace calculation, we need sum of squared clusters' sizes,
            # which is the sum of F's columns, squared, and summarized.
            # We save the repeated summary over F's columns with a dedicated
            # array F_counts that will maintain this informaiton.
            F_counts = F.sum(axis=0)

            # Get all indicator columns
            indicators = F.nonzero()[1]

            # Calculate F
            # For each row in F
            for i in samples.nonzero()[0]:
                # Get the indicator (non-zero column) of current (i-th) row
                curr_ind = indicators[i]

                # Update F's columns sums, as if current row is all zeros
                F_counts[curr_ind] = F_counts[curr_ind] - 1

                # To save looping over F's columns, we calculate the trace
                # for all possible indicator columns at once using a matrix.

                # Initialize F_mat, all rows are duplicates of F_counts which
                # is the sum of F's columns when current row in F is all zeros.
                # Performance of repeat is better than tile and broadcast_to
                F_mat = np.repeat(F_counts, self.n_clusters).reshape(
                    self.n_clusters, self.n_clusters).T

                # On every row of F_mat, add one to a different column.
                F_mat = F_mat + I

                # Now the j-th row of F_mat represent the sum of F's columns
                # when current row in F has the j-th column set as indicator.
                # We square every element and sum the rows. In the end, traces
                # contains <n_clusters> values, each is the trace value for a
                # different indicator column.
                traces = (F_mat**2).sum(axis=1)

                # We want to calculate sum of squared elements of X-HF^T for
                # every possible column indicator. Since the sum was already
                # calculated per column in the beginning of the iteration
                # (base_sum), we can only calculate the diffs.

                # If F's i-th row and j-th column is one, then (HF^T)'s i-th
                # column is a copy of H's j-th column.
                # Hence, the value of sum of squared elements of X-HF^T equals
                # total_base_sum (result with F from itearation start) minus
                # the value from iteration start originating from current row
                # (base_sum[i]) plus sum of squared elements of (i-th column of
                # X minus j-th column of H), j being the indicator column.

                # To save looping over F's columns, we create new matrix with
                # all columns duplicates of the i-th column of X and substract
                # H. The j-th column of the result is how the i-th column of
                # X-HF^T would be as if j is the indicator column.
                # Then we square all elements in result and summarize columns.

                # At this point we have two arrays of size <n_clusters>: traces
                # and the last calculated sum. We multiply traces with the
                # scalar gamma and add. Final result is all possible values of
                # ||X-HF^T||^2_F + gamma*Tr(F^T11^TF), given current row of F.
                res = X_dup[i].reshape(self.n_clusters, n_dims)
                res = total_base_sum - base_sum[i] + ((res - H.T)**2).sum(
                    axis=1)
                res2 = res + self.gamma * traces

                # Index of minimal value is set as the indicator column
                new_ind = res2.argmin()
                F_counts[new_ind] = F_counts[new_ind] + 1  # Update column sums

                # If indicator column was changed, set convergence to False and
                # update F.
                if (curr_ind != new_ind):
                    conv = False
                    F[i, curr_ind] = 0
                    F[i, new_ind] = 1

            iteration = iteration + 1
            # print iteration
            H_new = X.dot(np.linalg.pinv(F.T))  # H = XF(F^TF)^-1

            # If centers movement is small, consider convergence
            centers_shift = np.sqrt(np.sum((H - H_new) ** 2, axis=0))

            centers_shift_total = np.sum(centers_shift)
            # print(centers_shift_total)
            if centers_shift_total ** 2 < self.tol:
                conv = True

            H = H_new
            if self.verbose:
                print('Iteration %d, %f secs' % (iteration,
                                                 (time.time() - start)))
        return F.nonzero()[1]


class WKMeans:
    """Weighted K-means.

    Features are first standardized to zero mean and unit variance. Then every
    feature gets a weight proportional to how evenly its values are spread
    across the range.

    The weight of every feature is :math:`(\mu(D) / \sigma(D))^{-4}` where
    :math:`D` is the differences between consecutive values.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1,
                 algorithm='auto'):
        self.n_clusters = n_clusters
        self.km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                         max_iter=max_iter, tol=tol,
                         precompute_distances=precompute_distances,
                         verbose=verbose, random_state=random_state,
                         copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)

    @staticmethod
    def _features_imbalance(ds):
        scores = []
        for col in ds.columns:
            try:
                diffs = np.ediff1d(ds[col].sort_values())
                result = np.std(diffs) / np.mean(diffs)
                scores.append(result**2)
            except Exception:
                scores.append(0)
        return np.nan_to_num(scores)

    def fit_predict(self, X, y=None):
        # Scale data to unit variance
        data = pd.DataFrame(StandardScaler().fit_transform(X),
                            columns=X.columns)

        # Calculate features weights according to their imbalance
        features_weights = WKMeans._features_imbalance(X)

        # Apply features weights
        for i, col in enumerate(X.columns):
            if features_weights[i] != 0:
                data[col] = data[col] / features_weights[i]
        else:
            data[col] = 0

        # Run K-means
        return self.km.fit_predict(data)


class MultiPart:
    """Multilevel k-way partitioning scheme for irregular graphs.

    This class produces a weighted graph. Each vertex represents a data point.
    Its edges represent distances to its N nearest neighbors. The graph is then
    partitioned using multilevel k-way partition scheme [1]_ based on `METIS`_,
    minimizing edges' sum.

    Prerequisite packages:
        metis
        networkx

    .. _METIS:
        http://glaros.dtc.umn.edu/gkhome/metis/metis/overview

    References
    ----------
    .. [1] G. Karypis and V. Kumar, "Multilevel k-way partitioning scheme for
        irregular graphs". Journal of Parallel and Distributed Computing,
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
