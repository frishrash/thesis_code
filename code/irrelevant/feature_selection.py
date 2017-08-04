# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:20:09 2017

@author: gal
"""

from random import randint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import csv
from scipy.stats import chisquare
import dataset as ds
import pandas as pd
from dataset import NSL

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


def check_stability(ds, kfold=5):
    skf = StratifiedKFold(n_splits=kfold)
    for train_index, test_index in skf.split(ds, ds.index):
        X_train, X_test = ds.iloc[train_index], ds.iloc[test_index]
        train_imbalance = np.array(features_imbalance(X_train))
        test_imbalance = np.array(features_imbalance(X_test))
        dist = np.sqrt(((train_imbalance - test_imbalance)**2).sum())
        print(zip(train_imbalance, test_imbalance))


def check_stability2(ds, frac=0.05, n_iter=5):
    for _ in xrange(n_iter):
        x1 = features_imbalance3(ds.sample(frac=frac))
        x2 = features_imbalance3(ds.sample(frac=frac))
        dist = np.sqrt(((np.array(x1) - np.array(x2))**2).sum())
        print(dist)


def features_imbalance2(ds):
    scores = []
    for col in ds.columns:
        scores.append(np.std(MinMaxScaler().fit_transform(ds[col])))
    return scores


def features_imbalance3(ds):
    df_scaled = pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(ds), columns=ds.columns)
    scores = []
    for col in df_scaled.columns:
        try:
            col = np.array(df_scaled[col])
            col.sort()
            #result = np.var(np.ediff1d(col)) / np.mean(col)
            result = np.std(col) / np.mean(col)
            scores.append(result)
        except Exception:
            scores.append(0)
    return np.nan_to_num(scores)

    
def random_score(n_iter=20, frac=0.2, dataset=ds.NSL_TRAIN):
    data = NSL(dataset, ds.ENC_NUMERIC, ds.SCL_NONE).ds
    scores = np.array([0 for _ in xrange(data.shape[1])]).astype(float)
    for _ in xrange(n_iter):
        score = features_imbalance3(data)
        scores = scores + score
    return scores / n_iter


def random_tests(n_iter=20):
    csv_file = open('random.csv', 'wb')
    csvwriter = csv.writer(csv_file)
    for _ in xrange(n_iter):
        dataset = [ds.NSL_TRAIN, ds.NSL_TEST][randint(0,1)]
        frac = float(randint(1,100)) / 100
        clusters = randint(3,9)
        score = test(dataset, n_clusters=clusters, frac=frac)
        result_str = ' - '.join([str(i) for i in score])
        minmax_ratio = float(np.max(score)) / np.min(score)
        csvwriter.writerow([dataset, clusters, frac, result_str, minmax_ratio])
        print score
    csv_file.close()


def balance_prescale(ds, n_clusters=5):
    for col in ds.columns:
        col_std = ds[col].std()
        # if col_std != 0:
        #    ds[col] = ds[col].astype(float) / (ds[col].std())
        try:
            res = pd.cut(ds[col], n_clusters)
            labels, counts = np.unique(res, return_counts=True)
            counts = np.concatenate((counts, [0] * (n_clusters - len(counts))))
            statistic, pvalue = chisquare(counts)
            print('%f, %f, %f' % (pvalue, statistic, col_std / ds[col].mean()))
            if pvalue < 0.05:  # We reject the hypothesis that we are uniform
                ds[col] = ds[col] / statistic**2  # Scale down
        except Exception, e:
            print(str(e))
            pass

def features_score(ds, n_clusters):
    ds_master = pd.DataFrame(StandardScaler().fit_transform(ds),
                          columns=ds.columns)
    km = KMeans(n_clusters=n_clusters)
    weights = {x:float(1) for x in ds.columns}

    is_done = False
    i = 1
    while (not is_done):
        km_result = km.fit_predict(ds_master)
        labels, counts = np.unique(km_result, return_counts=True)
        current_ratio = float(counts.max()) / counts.min()
        print('Starting iteration %d, current ratio: %f' % (i, current_ratio))
        print(str(weights))
        is_done = True
        for col in ds.columns:
            tmp_ds = ds_master.copy()
            tmp_ds[col] = 0
            km_result = km.fit_predict(tmp_ds)
            labels, counts = np.unique(km_result, return_counts=True)
            minmax_ratio = float(counts.max()) / counts.min()
            if minmax_ratio < current_ratio:
                weights[col] = weights[col] / 2
                ds_master[col] = ds_master[col] / 2
                is_done = False
            elif minmax_ratio > current_ratio:
                weights[col] = weights[col] * 2
                ds_master[col] = ds_master[col] * 2
                is_done = False
        i = i + 1

def features_imbalance(ds, r=range(3, 10), reverse=False):
    scores = []
    for col in ds.columns:
        try:
            minmax_ratios = []
            minmax_diffs = []
            for i in r:
                res = pd.cut(ds[col], i)
                labels, counts = np.unique(res, return_counts=True)
                normalized_dist = float(np.max(counts) - np.min(counts)) /\
                    len(ds)
                #minmax_ratio = float(np.max(counts)) / np.min(counts)
                #minmax_ratios.append(minmax_ratio)
                #diff = float(np.max(counts) - np.min(counts)) / (len(ds) / i)
                minmax_diffs.append(normalized_dist)
            if (reverse):
                r.reverse()
            scores.append(np.average(minmax_diffs, weights=r)**4)
            # scores.append(np.max(minmax_ratios))
            #scores.append(np.average(minmax_ratios, weights=r) / len(ds))
        except Exception:
            scores.append(0)
    return scores

def test(data=ds.NSL_TRAIN20, frac=0.1, n_clusters=9):
    nsl = NSL(data, ds.ENC_NUMERIC, ds.SCL_NONE)
    data = pd.DataFrame(StandardScaler().fit_transform(
                        nsl.ds.sample(frac=frac)),
                        columns=nsl.ds.columns)
    scores = features_imbalance3(data)
    #scores = [4.9687348601488857e-06, 0.048005449773971885, 0.30090289606345372, 0.039999999997479371, 0.000583154682335534, 0.071427437376211239, 0.0006369753925856707, 0.0064933003025774748, 2.0059375750962523e-05, 0.12727121443046172, 0.0059171597629407305, 0.0071941874935393642, 0.11714027214813312, 0.0010052146319188752, 0.019230616569268669, 0.0019378614210404626, 0.0, 0.99999999993698407, 0.00084245998309771147, 9.3999531402734481e-08, 2.9174852834226813e-07, 4.1833931281974199e-07, 3.8195776838186883e-07, 1.1509224936799674e-06, 1.6633409340012933e-06, 1.2001268628255685e-07, 1.4342506054173522e-06, 2.8705243184823012e-06, 4.3492993562524208e-08, 6.8499758177045603e-08, 1.5217451831180736e-07, 9.5622234800440755e-07, 5.3457550733435897e-07, 4.43769710268333e-06, 2.7885047378256104e-07, 2.9052731267311483e-07, 6.6749552744942154e-07, 6.5967876419635182e-07]
    # scores2 = np.sqrt(np.array(scores).astype(float))
    for i, col in enumerate(nsl.ds.columns):
        if scores[i] is not None and not np.isnan(scores[i]) and scores[i] != 0:
            data[col] = data[col] / scores[i]
    else:
        nsl.ds[col] = 0    
    km = KMeans(n_clusters=n_clusters, random_state=0)
    res = km.fit_predict(data)
    labels, counts = np.unique(res, return_counts=True)
    return counts


def balance_score(ds, r=range(5, 6)):
    minmax_ratios = []
    for i in r:
        km = KMeans(random_state=0, n_clusters=i)
        result = km.fit_predict(ds)
        labels, counts = np.unique(result, return_counts=True)
        minmax_ratios.append(float(np.max(counts)) / np.min(counts))

    #score = np.average(minmax_ratios, weights=r)
    score = np.max(minmax_ratios)
    return score

nsl = NSL(ds.NSL_TRAIN20, ds.ENC_NUMERIC, ds.SCL_NONE)

best_known_features = ['num_access_files', 'num_compromised', 'rerror_rate', 'urgent', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate', 'srv_serror_rate', 'is_host_login', 'wrong_fragment', 'serror_rate', 'num_shells', 'num_outbound_cmds', 'is_guest_login', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 'hot', 'dst_host_srv_count', 'logged_in', 'srv_rerror_rate', 'dst_host_srv_diff_host_rate', 'num_root', 'dst_host_same_src_port_rate', 'root_shell', 'su_attempted', 'dst_host_count', 'num_file_creations', 'count', 'land', 'same_srv_rate', 'dst_host_diff_srv_rate', 'srv_diff_host_rate', 'diff_srv_rate', 'num_failed_logins', 'dst_host_serror_rate']

i = 0
is_done = False
current_features = nsl.ds.columns.tolist()

worst_balance = float('inf')

while (i < 37 and not is_done):
    worst_feature = None
    for feature in current_features:
        features_to_inspect = list(set(current_features) - set([feature]))
        s = balance_score(nsl.ds[features_to_inspect])
        if (s < worst_balance):
            print('The most unbalancing feature so far: %s (%f)' % (feature, s))
            worst_feature = feature
            worst_balance = s
    print('Worst feature for iteration %d: %s (%f)' % (i, worst_feature,
                                                       worst_balance))

    if (worst_feature is None):
        is_done = True
    else:
        current_features = list(set(current_features) - set([worst_feature]))

    i = i + 1

print(current_features)
