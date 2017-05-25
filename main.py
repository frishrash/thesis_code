# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:40:33 2017

@author: gal
"""

import os
import numpy as np
from sklearn.cluster import KMeans, Birch
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from os import listdir
from os.path import join, splitext
import pickle
import csv

import dataset as ds
from dataset import NSL
from clustering_model import ClusteringModel as CM
from eval import EvalClassifier, EV_TIME_TRN, EV_TIME_TST
from eval import EV_FSCORE, EV_PRE, EV_REC, EV_AUC, EV_CM
from classifier import ClfFactory
from clustering_model import is_feasible
from splitters import RoundRobin, RandomRoundRobin, NoSplit
from reports import feasible_models
from settings import MODELS_DIR, CLUSTERS_REPORT, CLASSIFIERS_REPORT


dt3 = ClfFactory(DT, random_state=0, max_depth=3)
dt3e = ClfFactory(DT, random_state=0, criterion='entropy', max_depth=3)
dt3eb = ClfFactory(DT, random_state=0, max_depth=3, criterion='entropy',
                   class_weight='balanced')
rf3 = ClfFactory(RF, random_state=0, min_samples_leaf=20, max_depth=3)
mlp = ClfFactory(MLPClassifier, random_state=0)
mnb = ClfFactory(MultinomialNB)
gnb = ClfFactory(GaussianNB)
perceptron = ClfFactory(Perceptron)
nn = ClfFactory(KNeighborsClassifier, n_neighbors=1, n_jobs=-1)
svmlin = ClfFactory(svm.SVC, kernel='linear', probability=True)
svmlinovr = ClfFactory(svm.SVC, kernel='linear', probability=True,
                       decision_function_shape='ovr')
svmbaglinovr = ClfFactory(BaggingClassifier,
                          svm.SVC(kernel='linear',
                                  probability=True,
                                  decision_function_shape='ovr'))
svmrbf = ClfFactory(svm.SVC, kernel='rbf')
ee = ClfFactory(EllipticEnvelope)


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

                # The following splitters are features agnostic. However, they
                # are added for every encoding and scaling since it matters for
                # the ML classifiers later on in the process
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


def clustering_feasibility_report():
    csv_file = open(CLUSTERS_REPORT, 'wb')
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


def eval_classifiers():
    csv_file = open(CLASSIFIERS_REPORT, 'wb')
    csvwriter = csv.writer(csv_file)
    header = ['Dataset', 'Encoding', 'Scaling', 'Algo', 'Features', 'K',
              'Split Sizes', 'Classifier', 'Classifier Info', 'K-fold',
              'Training Time', 'Testing Time', 'F-Score', 'Precision',
              'Recall']
    labels = NSL.standard_labels()
    for a in labels:
        for b in labels:
            header.append("True %s, Predicted %s" % (a, b))
    for a in labels:
        header.append("AUC %s" % a)
    csvwriter.writerow(header)
    classifiers = [nn, dt3, rf3, mlp, svmlin]

    for f in listdir(MODELS_DIR):
        algo, features, dataset, encoding, scaling = splitext(f)[0].split('_')
        data = pickle.load(open(join(MODELS_DIR, f), 'rb'))
        if is_feasible(data):
            ds_ = NSL(dataset, scaling=scaling, encoding=encoding)
            for classifier in classifiers:
                # Evaluate SVM only when min-max scaled (time constraint)
                if classifier.name == 'SVM' and scaling != 'Min-max':
                    continue
                print('Working on %s, classifier %s' % (f, classifier.name))
                ev = EvalClassifier(ds_, data, classifier, calc_prob=True)

                # If feasible (no errors during cross-validation)
                if ev.eval():
                    for i, res in enumerate(ev.results):
                        line = [dataset, encoding, scaling, algo, features,
                                data[i]['k'], ' - '.join(map(lambda x: str(x),
                                                         data[i]['SPLIT_SIZES']
                                                             )),
                                classifier.name, classifier.info, ev.kfold,
                                '%.2f' % res[EV_TIME_TRN],
                                '%.2f' % res[EV_TIME_TST],
                                '%.2f' % res[EV_FSCORE],
                                '%.2f' % res[EV_PRE],
                                '%.2f' % res[EV_REC]]

                        line = np.append(line, res[EV_CM].flatten())
                        if EV_AUC in res:
                            for lbl in NSL.standard_labels():
                                line = np.append(line,
                                                 '%.2f' % res[EV_AUC][lbl])

                        csvwriter.writerow(line)
                        csv_file.close()
                        csv_file = open(CLASSIFIERS_REPORT, 'ab')
                        csvwriter = csv.writer(csv_file)

    csv_file.close()


def feasible_models_output():
    ds = feasible_models()
    ds.to_csv('test2.csv', index=False)

# create_clustering_models()
# clustering_feasibility_report()
# eval_classifiers()
feasible_models_output()
