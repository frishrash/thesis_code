# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:40:33 2017

@author: gal
"""

import os
import numpy as np
import pandas as pd
import argparse
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
from os.path import join, splitext, basename, isfile
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
from splitters import KMeansBal, MultiPart, WKMeans, EXLasso
from reports import feasible_classifiers, plot_classifier, plot_scalability
from reports import plot_max_ratio, plot_roc, plot_class_distribution
from reports import plot_classifier_info
from settings import MODELS_DIR, GRAPHS_DIR, CLFS_DIR, REPORTS_DIR
from settings import CLUSTERS_REPORT, CLASSIFIERS_REPORT, FEASIBLES_REPORT


dt3 = ClfFactory(DT, random_state=0, max_depth=3)
dt4 = ClfFactory(DT, random_state=0, max_depth=4)
dt5 = ClfFactory(DT, random_state=0, max_depth=5)
dt6 = ClfFactory(DT, random_state=0, max_depth=6)
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
                    NSL.FEATURES_2SECS,
                    NSL.FEATURES
                    ]
    nsl_descs = ['2 secs same dest host',
                 '2 secs same service',
                 '100 connections same host',
                 'expert features',
                 'single TCP features',
                 'all 2 secs',
                 'all history based features',
                 'all features'
                 ]
    for dataset in (ds.NSL_TRAIN20, ds.NSL_TEST):
        for encoding in (ds.ENC_NUMERIC, ds.ENC_HOT):
            for scaling in (ds.SCL_NONE, ds.SCL_MINMAX, ds.SCL_STD):
                nsl = NSL(dataset, encoding, scaling)
                models.append(CM(nsl).gen_model(MultiPart, seed=0))
                models.append(CM(nsl).gen_model(WKMeans, random_state=0))
                models.append(CM(nsl).gen_model(EXLasso, random_state=0,
                              gamma=0.1, tol=1e-2, verbose=True))
                models.append(CM(nsl).gen_model(KMeansBal, random_state=0,
                              clusters_factor=5))

        for encoding in (ds.ENC_NUMERIC, ds.ENC_HOT):
            for scaling in (ds.SCL_NONE, ds.SCL_MINMAX, ds.SCL_STD):
                nsl = NSL(dataset, encoding, scaling)

                # The following splitters are features agnostic. However, they
                # are added for every encoding and scaling since it matters for
                # the ML classifiers later on in the process
                models.append(CM(nsl).gen_model(RoundRobin))
                models.append(CM(nsl).gen_model(RandomRoundRobin))
                models.append(CM(nsl, min_k=1, max_k=1).gen_model(NoSplit))

                # Add all clustering models
                for f, d in zip(nsl_features, nsl_descs):
                    models.append(CM(nsl, f, d).gen_model(KMeans,
                                                          random_state=0))

    for model in models:
        file_name = "%s_%s_%s_%s_%s.dmp" % (model.algorithm,
                                            model.features_desc,
                                            model.dataset.ds_name,
                                            model.dataset.encoding,
                                            model.dataset.scaling,
                                            )
        if (isfile(os.path.join(MODELS_DIR, file_name))):
            print("Skipping %s" % file_name)
            continue
        print ("Running model %s %s %s %s %s" % (model.dataset.ds_name,
                                                 model.dataset.encoding,
                                                 model.dataset.scaling,
                                                 model.algorithm,
                                                 model.features_desc))
        model.run()
        model.save(os.path.join(MODELS_DIR, file_name))


def clustering_feasibility_report():
    csv_file = open(CLUSTERS_REPORT, 'wb')
    csvwriter = csv.writer(csv_file)

    csvwriter.writerow(['Algorithm', 'Features', 'Dataset', 'Encoding',
                        'Scaling', 'Valid?', 'Max Std.', 'Max Ratio',
                        'Max Split Size', 'Min Split Size'])

    for f in listdir(MODELS_DIR):
        algo, features, dataset, encoding, scaling = f.split('_')
        try:
            data = pickle.load(open(join(MODELS_DIR, f), 'rb'))
        except Exception:
            continue
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


def feasibile_clustering_models(report_file=CLUSTERS_REPORT):
    csv_file = open(report_file, 'wb')
    csvwriter = csv.writer(csv_file)

    csvwriter.writerow(['Algorithm', 'Features', 'Dataset', 'Encoding',
                        'Scaling', 'Max Std.', 'Max Ratio',
                        'Avg. Split Time', 'Valid?', 'Other DS valid?'])
    known_files = set()

    for f in listdir(MODELS_DIR):
        try:
            algo, features, dataset, encoding, scaling = splitext(f)[0].\
                split('_')
        except Exception:
            print('Invalid file model file name %s, skipping' % f)
            continue
        print('Working on %s' % f)
        file_name = join(MODELS_DIR, f)
        if file_name in known_files:
            continue
        data = pickle.load(open(file_name, 'rb'))
        max_std = np.max(map(lambda x: x['CLUSTERING_STD'], data))
        max_ratio = np.max(map(lambda x: x['MINMAX_RATIO'], data))
        avg_split_time = np.mean(map(lambda x: x['SPLIT_TIME'], data))
        other_dataset = 'NSL Test+'
        if dataset == 'NSL Test+':
            other_dataset = 'NSL Train+ 20%'
        other_f = '_'.join([algo, features, other_dataset, encoding,
                            scaling])
        other_f = join(MODELS_DIR, other_f + '.dmp')

        is_model_valid = False
        if is_feasible(data):
            is_model_valid = True

        is_other_valid = False
        if not isfile(other_f):
            print("Error, %s not found!" % other_f)
            line = [algo, features, dataset, encoding, scaling, max_std,
                    max_ratio, avg_split_time, is_model_valid, None]
            csvwriter.writerow(line)
        else:
            known_files.add(other_f)
            other_data = pickle.load(open(other_f, 'rb'))
            if is_feasible(other_data):
                is_other_valid = True

            line = [algo, features, dataset, encoding, scaling, max_std,
                    max_ratio, avg_split_time, is_model_valid, is_other_valid]
            csvwriter.writerow(line)
            max_std = np.max(map(lambda x: x['CLUSTERING_STD'], other_data))
            max_ratio = np.max(map(lambda x: x['MINMAX_RATIO'], other_data))
            avg_split_time = np.mean(map(lambda x: x['SPLIT_TIME'],
                                         other_data))
            line = [algo, features, other_dataset, encoding, scaling, max_std,
                    max_ratio, avg_split_time, is_other_valid, is_model_valid]
            csvwriter.writerow(line)
    csv_file.close()


def eval_classifiers(report_file=CLASSIFIERS_REPORT, classifiers=None,
                     clusters_file=CLUSTERS_REPORT):
    clfmap = {'NN': nn, 'DT': dt3, 'RF': rf3, 'MLP': mlp,
              'SVM': svmlin}
    classifiers_ = []
    for clf in classifiers:
        classifiers_.append(clfmap[clf])

    clusters = pd.read_csv(clusters_file)
    csv_file = open(report_file, 'wb')
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

    for i, row in clusters.iterrows():
        if row['Valid?'] is not True or row['Other DS valid?'] is not True:
            continue
        algo = row['Algorithm']
        features = row['Features']
        dataset = row['Dataset']
        encoding = row['Encoding']
        scaling = row['Scaling']

        model_file = '%s_%s_%s_%s_%s.dmp' % (algo, features, dataset, encoding,
                                             scaling)
        model_file = join(MODELS_DIR, model_file)
        if not isfile(model_file):
            print('Model %s was not found!' % model_file)
            continue

        data = pickle.load(open(model_file, 'rb'))
        ds_ = NSL(dataset, scaling=scaling, encoding=encoding)
        for classifier in classifiers_:
            # Evaluate SVM only when min-max scaled (time constraint)
            if classifier.name == 'SVM' and scaling != 'Min-max':
                continue

            dump_file = '%s_%s_%s_%s_%s_%s.dmp' % (algo, features,
                                                   dataset, encoding, scaling,
                                                   classifier.name)

            results = []
            if (isfile(join(CLFS_DIR, dump_file))):
                print('Classifier %s already exists, loading results from it' %
                      dump_file)
                results = pickle.load(open(join(CLFS_DIR, dump_file), 'rb'))
            else:
                print('Working on %s' % dump_file)
                ev = EvalClassifier(ds_, data, classifier, calc_prob=True)
                if ev.eval():
                    pickle.dump(ev.results, open(join(CLFS_DIR, dump_file),
                                'wb'))
                    results = ev.results
                else:
                    print("Error evaluating %s" % dump_file)
                    continue

            # Create report
            for i, res in enumerate(results):
                line = [dataset, encoding, scaling, algo, features,
                        data[i]['k'], ' - '.join(map(lambda x: str(x),
                                                 data[i]['SPLIT_SIZES']
                                                     )),
                        classifier.name, classifier.info, 5,
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
                csv_file = open(report_file, 'ab')
                csvwriter = csv.writer(csv_file)
    csv_file.close()


def feasible_models_output():
    data = feasible_classifiers()
    data.to_csv(FEASIBLES_REPORT, index=False)

    plot_classifier(data, 'NSL Test+', 'DT', 'AUC U2R',
                    #order=[3, 4, 5, 0, 2, 1],
                    file_name=os.path.join(GRAPHS_DIR, 'nsltst-dt-aucu2r.png'))
    # plot_classifier(data, 'NSL Test+', 'RF', 'F-Score',
    #               order=[3, 4, 5, 0, 2, 1],
    #               file_name=os.path.join(GRAPHS_DIR, 'nsltst-rf-fscore.png'))
    plot_classifier(data, 'NSL Test+', 'DT', 'F-Score',
                    #order=[3, 4, 5, 0, 2, 1],
                    file_name=os.path.join(GRAPHS_DIR, 'nsltst-dt-fscore.png'))
    plot_classifier(data, 'NSL Test+', 'SVM', 'F-Score',
                    order=[3, 4, 5, 0, 2, 1],
                    file_name=os.path.join(GRAPHS_DIR, 'nsltst-svm-fscore.png')
                    )

    plot_classifier(data, 'NSL Test+', 'kNN', 'F-Score',
                    #order=[3, 4, 5, 0, 2, 1],
                    file_name=os.path.join(GRAPHS_DIR, 'nsltst-knn-fscore.png')
                    )

    plot_classifier(data, 'NSL Test+', 'MLPClassifier', 'F-Score',
                    #order=[3, 4, 5, 0, 2, 1],
                    file_name=os.path.join(GRAPHS_DIR, 'nsltst-mlp-fscore.png')
                    )

    plot_scalability(data, 'NSL Test+', 'KMeans', ["Min-max"], 'F-Score',
                     file_name=os.path.join(GRAPHS_DIR,
                                            'nsltst-kmeans-sca-fscore.png'))
    plot_max_ratio(data, ['KMeans', 'NoSplit', 'RoundRobin',
                          'RandomRoundRobin'],
                   # order=[3, 4, 5, 0, 2, 1],
                   file_name=os.path.join(GRAPHS_DIR, 'max-ratio-all.png'))
    plot_roc(data, 'NSL Test+', 'Min-max', 'Hot Encode', 'DT', 9, 'U2R',
             order=[3, 5, 4, 0, 1, 2],
             file_name=os.path.join(GRAPHS_DIR,
                                    'nsltst-roc-u2r-dt-9-minmax-onehot.png'))
    plot_class_distribution('KMeans', '100 connections same host', 'NSL Test+',
                            'Hot Encode', 'Min-max', 9,
                            file_name=os.path.join(GRAPHS_DIR,
                                                   'class-dist-kmeans.png'))
    plot_class_distribution('RoundRobin', 'All',
                            'NSL Test+', 'Hot Encode', 'Min-max', 9,
                            file_name=os.path.join(GRAPHS_DIR,
                                                   'class-dist-rr.png'))

    # Important: report 'classifiers_dt456.csv' was produced manually !!!
    # Todo: refactor models creation such that DT4, DT5, DT6 dumps will be on
    # separate files so report can be produced programatically
    data2 = pd.read_csv(join(REPORTS_DIR, 'classifiers_dt456.csv'))
    plot_classifier_info(pd.concat([data, data2]), 'DT',
                         # order=[3, 4, 5, 0, 2, 1],
                         file_name=os.path.join(GRAPHS_DIR,
                                                'baseline-dt-comparison.png'))

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", help='sub-command help')
subparsers.add_parser('CM', help='Create clustering models')
parser_fr = subparsers.add_parser('FR', help='Create feasibility report')
parser_fr.add_argument('-O', dest='output', default=CLUSTERS_REPORT,
                       help='Output file')
parser_ec = subparsers.add_parser('EC', help='Evaluate classifiers')
parser_ec.add_argument('-n', '--NN', dest='classifiers', action='append_const',
                       const='NN', help='Evaluate 1-Nearest-Neighbor')
parser_ec.add_argument('-d', '--DT', dest='classifiers', action='append_const',
                       const='DT', help='Evaluate Decision Tree')
parser_ec.add_argument('-r', '--RF', dest='classifiers', action='append_const',
                       const='RF', help='Evaluate Random Forest')
parser_ec.add_argument('-m', '--MLP', dest='classifiers',
                       action='append_const', const='MLP',
                       help='Evaluate Multilayer Perceptron')
parser_ec.add_argument('-s', '--SVM', dest='classifiers',
                       action='append_const', const='SVM', help='Evaluate SVM')
parser_ec.add_argument('-O', dest='output', default=CLASSIFIERS_REPORT,
                       help='Output file')
parser_ec.add_argument('-C', dest='clusters_report', default=CLUSTERS_REPORT,
                       help='Output file')
subparsers.add_parser('PO', help='Plot output')


args = parser.parse_args()
if args.command == 'EC' and not args.classifiers:
    parser.error("At least one classifier must be specified")

if args.command == 'CM':
    create_clustering_models()
elif args.command == 'FR':
    feasibile_clustering_models(report_file=args.output)
elif args.command == 'PO':
    feasible_models_output()
elif args.command == 'EC':
    eval_classifiers(report_file=args.output, classifiers=args.classifiers,
                     clusters_file=args.clusters_report)
    
#if not args.mode:
#    parser.error("One of --process or --upload must be given")

# create_clustering_models()
#feasibile_clustering_models()
#clustering_feasibility_report()
# eval_classifiers()
# feasible_models_output()
