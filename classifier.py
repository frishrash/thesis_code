# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 08:19:09 2016

@author: gal
"""
from sklearn import tree
from sklearn.externals import joblib
from sklearn.base import clone


class ClfFactory(object):
    def __init__(self, class_name, *args, **kwargs):
        self.algo = class_name(*args, **kwargs)
        self.name = self.algo.__class__.__name__
        self.info = ''
        self.calc_prob = True
        if ("fit" not in dir(self.algo)):
            raise Exception('No fit method for %s' % self.name)
        if ("predict" not in dir(self.algo)):
            raise Exception('No predict method for %s' % self.name)

        if (self.name == 'SVC'):
            self.name = 'SVM'
            self.info = self.algo.kernel.capitalize()
            if (self.algo.decision_function_shape is not None):
                self.info = self.info + ', ' + \
                            self.algo.decision_function_shape
        elif (self.name == 'DecisionTreeClassifier'):
            self.name = 'DT'
            if (self.algo.max_depth is not None):
                self.info = 'Depth %d, %s' % (self.algo.max_depth,
                                              self.algo.criterion.capitalize())
        elif (self.name == 'RandomForestClassifier'):
            self.name = 'RF'
            if (self.algo.max_depth is not None):
                self.info = 'Depth %d, %s' % (self.algo.max_depth,
                                              self.algo.criterion.capitalize())
        elif (self.name == 'KNeighborsClassifier'):
            self.name = 'kNN'
            self.info = 'k=%d' % self.algo.n_neighbors
        elif (self.name == 'OneClassSVM'):
            self.name = 'OC-SVM'
            self.info = self.algo.kernel.capitalize()
        elif (self.name == 'IsolationForest'):
            self.name = 'iForest'

    def clone(self):
        new_clf = self.__class__(self.algo.__class__)
        new_clf.algo = clone(self.algo)
        new_clf.info = self.info
        new_clf.name = self.name
        return new_clf

    def fit(self, *args, **kwargs):
        self.algo.fit(*args, **kwargs)
        self.classes_ = self.algo.classes_
        return self

    def predict(self, ds):
        return self.algo.predict(ds)

    def predict_proba(self, ds):
        return self.algo.predict_proba(ds)

    def dump_model(self, file_name):
        joblib.dump(self.algo, file_name)

    def test(self):
        print(self.__class__.__name__)

    def visualize(self, file_name):
        if (self.algo.__class__.__name__ == 'DecisionTreeClassifier'):
            with open(file_name, 'w') as f:
                print(self.algo.__class__)
                f = tree.export_graphviz(self.algo, out_file=f, filled=True,
                                         class_names=self.algo.classes_,
                                         rounded=True, special_characters=True)
