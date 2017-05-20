# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:40:33 2017

@author: gal
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans, Birch
import dataset as ds
from dataset import NSL
from clustering_model import ClusteringModel as CM
from splitters import RoundRobin, RandomRoundRobin, NoSplit


def eval_models():
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

                models.append(CM(nsl).gen_model(RoundRobin))
                models.append(CM(nsl).gen_model(RandomRoundRobin))
                models.append(CM(nsl, min_k=1, max_k=1).gen_model(NoSplit))
                for f, d in zip(nsl_features, nsl_descs):
                    models.append(CM(nsl, f, d).gen_model(KMeans))
                    models.append(CM(nsl, f, d).gen_model(Birch))

    for model in models:
        print ("Checking model %s %s %s %s %s" % (model.dataset.ds_name,
                                                  model.dataset.encoding,
                                                  model.dataset.scaling,
                                                  model.algorithm,
                                                  model.features_desc))
        if (model.eval_model()):
            file_name = "%s_%s_%s_%s_%s.dmp" % (model.dataset.ds_name,
                                                model.dataset.encoding,
                                                model.dataset.scaling,
                                                model.algorithm,
                                                model.features_desc)
            print("Passed!")
            pickle.dump(model, open(file_name, 'wb'))

eval_models()
