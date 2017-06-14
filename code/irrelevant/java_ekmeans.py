# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:00:58 2017

@author: gal
"""


import jnius_config
jnius_config.add_options('-Xss8192m', '-Xmx4g')
jnius_config.set_classpath('.', 'ekmeans-2.0.0.jar')
from jnius import autoclass
from random import randrange

import dataset as ds
from dataset import NSL

points = []
nsl = NSL(ds.NSL_TRAIN20, ds.ENC_NUMERIC, ds.SCL_MINMAX)
for row in nsl.ds.itertuples(index=False):
    points.append(list(row))

centroids = []
for i in xrange(3):
    centroids.append([randrange(0, 1, _int=float)
                     for _ in xrange(nsl.ds.shape[1])])
EKmeans = autoclass('ca.pjer.ekmeans.EKmeans')

#points = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
#centroids = [[0,0,0], [9,9,9]]

ekmeans = EKmeans(points, centroids)
ekmeans.setIteration(1)
ekmeans.setEqual(True)

ekmeans.run()