# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:56:43 2017

@author: gal
"""

import os

NSL_BASE_DIR = r"C:\Users\gal\Desktop\thesis\Datasets\NSL-KDD"
NSL_TRAIN_FILE = os.path.join(NSL_BASE_DIR, "KDDTrain+.txt")
NSL_TRAIN20_FILE = os.path.join(NSL_BASE_DIR, "KDDTrain+_20Percent.txt")
NSL_TEST_FILE = os.path.join(NSL_BASE_DIR, "KDDTest+.txt")

MIN_CLUSTER_SIZE = 450
MAX_CLUSTER_SIZE = 15550
MAX_CLUSTERS_STD = 4500
