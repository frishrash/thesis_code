# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:46:06 2017

@author: gal
"""

import time
import pickle
import numpy as np
from splitters import ClusterWrapper
from settings import MAX_CLUSTERS_STD


def is_feasible(model_data, max_sigma=MAX_CLUSTERS_STD):
    max_std = np.max(map(lambda x: x['CLUSTERING_STD'], model_data))

    # Check maximal clustering sizes std. dev
    if max_std > max_sigma:
        return False

    # Check that all clustering returned in requested size
    if not all(map(lambda x: x['k'] == len(x['SPLIT_SIZES']), model_data)):
        return False

    return True


class ClusteringModel:
    """Represents a clustering model.

    In the paper the clustering model consists of clustering algorithm,
    feature set, and preprocessing options: scaling and encoding.

    In practice, this class gets the sclaing and encoding options encapsulated
    within dataset instance and minimal and maximal k as well.
    """

    def __init__(self, dataset, features=None, features_desc='All',
                 min_k=3, max_k=9):
        self.splitters = []
        self.splits = []
        self.min_k = min_k
        self.max_k = max_k
        self.features = features
        self.features_desc = features_desc
        self.dataset = dataset

    def gen_model(self, class_, *args, **kwargs):
        self.algorithm = class_.__name__
        self.splitters = []
        for i in xrange(self.min_k, self.max_k+1):
            splitter = ClusterWrapper(class_, n_clusters=i, *args, **kwargs)
            if (self.features is not None):
                splitter.limit_features(self.features, self.features_desc)
            self.splitters.append(splitter)
        return self

    def _is_feasible(self, splits):
        split_sizes = map(lambda x: len(x), splits)
        min_split_size = np.min(split_sizes)
        max_split_size = np.max(split_sizes)
        clustering_std = np.std(split_sizes)
        if min_split_size < MIN_CLUSTER_SIZE:
            return False
        if max_split_size > MAX_CLUSTER_SIZE:
            return False
        if clustering_std > MAX_CLUSTERS_STD:
            return False

        return True

    def run(self):
        split_times = []
        all_splits = []
        clustering_results = []
        for splitter in self.splitters:
            split_start = time.clock()
            splits = splitter.split(self.dataset.ds)
            split_times.append(time.clock() - split_start)
            clustering_results.append(splitter.clustering_result)
            all_splits.append(splits)

        # Add more splits information to self.splits_info
        self._splits_info(all_splits)
        for i, t in enumerate(split_times):
            self.splits_info[i]['SPLIT_TIME'] = t
        for i, clustering_result in enumerate(clustering_results):
            self.splits_info[i]['CLUSTERING_RESULT'] = clustering_result
        for i, k in enumerate(xrange(self.min_k, self.max_k+1)):
            self.splits_info[i]['k'] = k

    def save(self, file_name):
        pickle.dump(self.splits_info, open(file_name, 'wb'))

    def _splits_info(self, all_splits):
        self.splits_info = []
        for splits in all_splits:
            split_sizes = map(lambda x: len(x), splits)
            clustering_info = {
                'SPLIT_SIZES': split_sizes,
                'MIN_SPLIT_SIZE': np.min(split_sizes),
                'MAX_SPLIT_SIZE': np.max(split_sizes),
                'CLUSTERING_STD': np.std(split_sizes),
                'MINMAX_RATIO': float(np.max(split_sizes))/np.min(split_sizes),
                'PURITY': [],
                'LABEL_COUNTS': []
                }
            for split in splits:
                labels = self.dataset.get_labels(split)
                vc = labels.value_counts()
                max_count = vc.max()
                purity = max_count.astype(float) / split.shape[0]
                clustering_info['PURITY'].append(purity)
                clustering_info['LABEL_COUNTS'].append(vc.to_dict())
            self.splits_info.append(clustering_info)
