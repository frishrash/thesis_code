# -*- coding: utf-8 -*-
"""
Created on Mon May 22 00:08:58 2017

@author: gal
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

from dataset import split_ds, NSL

EV_CM = 'Confusion Matrix'
EV_TIME_TRN = 'Training Time'
EV_TIME_TST = 'Testing Time'
EV_PRE = 'Precision'
EV_REC = 'Recall'
EV_FSCORE = 'F-Score'
EV_AUC = 'AUC'
EV_FPR = 'FPR'
EV_TPR = 'TPR'
EV_CLASSES = 'Classes'

EV_PRED = 'Predicted'
EV_PROB = 'Probabilities'
EV_TRUE = 'True Labels'

EV_CLU_PUR = 'Cluster Purity'
EV_CLU_AMI = 'Cluster AMI'
EV_CLU_ARI = 'Cluster ARI'


class EvalClassifier():
    def __init__(self, dataset, clustering_model_data, clf, kfold=5,
                 calc_prob=False):
        self.dataset = dataset
        self.clustering_data = clustering_model_data
        self.clf = clf
        self.kfold = kfold
        self.calc_prob = calc_prob

    @staticmethod
    def calc_scores(y_true, y_pred, y_prob, classes, calc_roc=True):
        result = {
            EV_CM: confusion_matrix(y_true, y_pred, classes),
            EV_PRE: precision_score(y_true, y_pred, classes, None, 'weighted'),
            EV_REC: recall_score(y_true, y_pred, classes, None, 'weighted'),
            EV_FSCORE: f1_score(y_true, y_pred, classes, None, 'weighted'),
            EV_CLASSES: classes
            }

        if (calc_roc):
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i, lbl in enumerate(classes):
                fpr[lbl], tpr[lbl], _ = roc_curve(y_true, y_prob[:, i],
                                                  pos_label=lbl)
                roc_auc[lbl] = auc(fpr[lbl], tpr[lbl])
            result.update({
                           EV_FPR: fpr,
                           EV_TPR: tpr,
                           EV_AUC: roc_auc
                          })
        return result

    @staticmethod
    def cross_val(ds, clf, kfold, classes=NSL.standard_labels(),
                  calc_prob=True):
        labels = NSL.get_labels(ds)

        # skf = StratifiedKFold(labels, n_splits=kfold) # SciKit 0.17
        skf = StratifiedKFold(n_splits=kfold, random_state=0)
        trn_timer = 0
        tst_timer = 0

        global_true = pd.core.series.Series()
        global_pred = np.array([])
        global_prob = None

        # for train_index, test_index in skf: # SciKit 0.17
        for train_index, test_index in skf.split(ds, labels):
            X_train, X_test = ds.iloc[train_index], ds.iloc[test_index]
            Y_train, Y_test = labels.iloc[train_index], labels.iloc[test_index]
            # local_clf = clone(clf)
            local_clf = clf.clone()

            start = time.clock()
            local_clf.fit(X_train, Y_train)
            trn_timer += time.clock() - start

            start = time.clock()
            y_pred = local_clf.predict(X_test)
            if (calc_prob):
                y_prob = local_clf.predict_proba(X_test)
            tst_timer += time.clock() - start

            global_true = global_true.append(Y_test)
            global_pred = np.append(global_pred, y_pred)

            if (calc_prob):
                # If local classifier didn't learn all classes we need to "pad"
                # probabilities matrix with zeros
                if (len(local_clf.classes_) != len(classes)):
                    for i, cls in enumerate(classes):
                        if cls in local_clf.classes_:
                            column = y_prob[:, np.where(local_clf.classes_ ==
                                                        cls)[0][0]]
                        else:
                            column = np.zeros((y_prob.shape[0], 1))

                        if i == 0:
                            tmp_prob = column
                        else:
                            tmp_prob = np.column_stack((tmp_prob, column))
                    y_prob = tmp_prob
                if global_prob is None:
                    global_prob = y_prob
                else:
                    global_prob = np.row_stack((global_prob, y_prob))

        result = {
                  EV_PRED: global_pred,
                  EV_PROB: global_prob,
                  EV_TRUE: global_true,
                  EV_TIME_TRN: trn_timer,
                  EV_TIME_TST: tst_timer
                  }

        result.update(EvalClassifier.calc_scores(global_true, global_pred,
                                                 global_prob, classes,
                                                 calc_roc=False))
        return result

    def eval(self):
        self.results = []
        for clustering in self.clustering_data:
            clusters = split_ds(clustering['CLUSTERING_RESULT'],
                                self.dataset.ds)

            # Global confusion matrix across splits
            global_true = pd.core.series.Series()
            global_pred = np.array([])
            global_prob = None
            total_trn_time = 0
            total_tst_time = 0

            for cluster in clusters:
                try:
                    # TODO: add multithread support for SVM, kNN and MLPClassifier
                    res = EvalClassifier.cross_val(cluster, self.clf,
                                                   self.kfold,
                                                   calc_prob=self.calc_prob)
                except (ValueError, AttributeError) as e:
                    print(u'\x1b[0;31m Error: ' + str(e) + u'\x1b[0m')
                    return False

                global_true = global_true.append(res[EV_TRUE])
                global_pred = np.append(global_pred, res[EV_PRED])
                if global_prob is None:
                    global_prob = res[EV_PROB]
                else:
                    global_prob = np.row_stack((global_prob, res[EV_PROB]))
                total_trn_time += res[EV_TIME_TRN]
                total_tst_time += res[EV_TIME_TST]

            classes = self.dataset.standard_labels()
            if (not classes):
                classes = list(set(global_pred.tolist() + global_true.unique().
                                   tolist()))
                classes.sort()
            result = EvalClassifier.calc_scores(global_true, global_pred,
                                                global_prob, classes,
                                                calc_roc=self.calc_prob)
            result.update({EV_TIME_TRN: total_trn_time,
                           EV_TIME_TST: total_tst_time})
            self.results.append(result)
        return True
