# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:56:43 2017

@author: gal
"""

import os

NSL_BASE_DIR = r'C:\Users\gal\Desktop\thesis\Datasets\NSL-KDD'
NSL_TRAIN_FILE = os.path.join(NSL_BASE_DIR, 'KDDTrain+.txt')
NSL_TRAIN20_FILE = os.path.join(NSL_BASE_DIR, 'KDDTrain+_20Percent.txt')
NSL_TEST_FILE = os.path.join(NSL_BASE_DIR, 'KDDTest+.txt')

THESIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(THESIS_DIR, 'reports')
MODELS_DIR = os.path.join(THESIS_DIR, 'models')
GRAPHS_DIR = os.path.join(THESIS_DIR, 'graphs')
CLFS_DIR = os.path.join(THESIS_DIR, 'classifiers')

CLUSTERS_REPORT = os.path.join(REPORTS_DIR, 'clusters.csv')
CLASSIFIERS_REPORT = os.path.join(REPORTS_DIR, 'tst-classifiers.csv')

MAX_CLUSTERS_STD = 6000
MAX_CLUSTERS_RATIO = 25

for directory in [THESIS_DIR, REPORTS_DIR, MODELS_DIR, GRAPHS_DIR, CLFS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
