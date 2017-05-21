# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:40:33 2017

@author: gal
"""

import os
import numpy as np
from sklearn.cluster import KMeans, Birch
from os import listdir
from os.path import join, splitext
import pickle
import csv

import dataset as ds
from dataset import NSL
from clustering_model import ClusteringModel as CM
from splitters import RoundRobin, RandomRoundRobin, NoSplit
from settings import MODELS_DIR, MAX_CLUSTERS_STD, FEASIBILITY_REPORT


def create_clustering_models():
    models = []
    nsl_features = [NSL.FEATURES_2SECS_HOST,
                    NSL.FEATURES_2SECS_SERVICE,
                    NSL.FEATURES_100CONNS_HOST,
                    NSL.FEATURES_EXPERT,
                    NSL.FEATURES_TCP,
                    np.append(NSL.FEATURES_2SECS_HOST,
                              NSL.FEATURES_2SECS_SERVICE),
                    NSL.FEATURES_2SECS
                    ]
    nsl_descs = ['2 secs same dest host',
                 '2 secs same service',
                 '100 connections same host',
                 'expert features',
                 'single TCP features',
                 'all 2 secs',
                 'all history based features'
                 ]
    for dataset in (ds.NSL_TRAIN20, ds.NSL_TEST):
        for encoding in (ds.ENC_NUMERIC, ds.ENC_HOT):
            for scaling in (ds.SCL_NONE, ds.SCL_MINMAX):
                nsl = NSL(dataset, encoding, scaling)

                # Add round robin, random robin and baseline only once
                if (encoding == ds.ENC_NUMERIC and scaling == ds.SCL_NONE):
                    models.append(CM(nsl).gen_model(RoundRobin))
                    models.append(CM(nsl).gen_model(RandomRoundRobin))
                    models.append(CM(nsl, min_k=1, max_k=1).gen_model(NoSplit))

                # Add all clustering models
                for f, d in zip(nsl_features, nsl_descs):
                    models.append(CM(nsl, f, d).gen_model(KMeans))
                    models.append(CM(nsl, f, d).gen_model(Birch))

    for model in models:
        print ("Running model %s %s %s %s %s" % (model.dataset.ds_name,
                                                 model.dataset.encoding,
                                                 model.dataset.scaling,
                                                 model.algorithm,
                                                 model.features_desc))
        model.run()
        file_name = "%s_%s_%s_%s_%s.dmp" % (model.algorithm,
                                            model.features_desc,
                                            model.dataset.ds_name,
                                            model.dataset.encoding,
                                            model.dataset.scaling,
                                            )
        model.save(os.path.join(MODELS_DIR, file_name))


def is_feasible(model_data):
    max_std = np.max(map(lambda x: x['CLUSTERING_STD'], model_data))
    # Check maximal clustering sizes std. dev
    if max_std > MAX_CLUSTERS_STD:
        return False
    # Check that all clustering returned in requested size
    if not all(map(lambda x: x['k'] == len(x['SPLIT_SIZES']), model_data)):
        return False
    return True


def clustering_feasibility_report():
    csv_file = open(FEASIBILITY_REPORT, 'wb')
    csvwriter = csv.writer(csv_file)

    csvwriter.writerow(['Algorithm', 'Features', 'Dataset', 'Encoding',
                        'Scaling', 'Valid?', 'Max Std.', 'Max Ratio',
                        'Max Split Size', 'Min Split Size'])

    for f in listdir(MODELS_DIR):
        algo, features, dataset, encoding, scaling = f.split('_')
        data = pickle.load(open(join(MODELS_DIR, f), 'rb'))

        max_std = np.max(map(lambda x: x['CLUSTERING_STD'], data))
        max_ratio = np.max(map(lambda x: x['MINMAX_RATIO'], data))
        max_split = np.max(map(lambda x: x['MAX_SPLIT_SIZE'], data))
        min_split = np.min(map(lambda x: x['MIN_SPLIT_SIZE'], data))
        valid = is_feasible(data)
        name, _ = splitext(f)
        line = name.split('_') + [valid, max_std, max_ratio, max_split,
                                  min_split]
        csvwriter.writerow(line)
    csv_file.close()


# create_clustering_models()
clustering_feasibility_report()
