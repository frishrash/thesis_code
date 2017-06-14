# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:58:00 2017

@author: gal
"""
import metis
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree

import dataset as ds
from dataset import NSL

nsl = NSL(ds.NSL_TRAIN20, ds.ENC_NUMERIC, ds.SCL_MINMAX)

kdt = KDTree(nsl.ds, leaf_size=50, metric='euclidean')
distances, indices = kdt.query(nsl.ds, k=100, return_distance=True)

G = nx.Graph()
G.add_nodes_from(xrange(len(nsl.ds)))

for i, x in enumerate(zip(distances, indices)):
    for ii, dist in enumerate(x[0]):
        G.add_edge(i, x[1][ii], weight=int(dist))
        print(int(dist))

G.graph['edge_weight_attr'] = 'weight'

(edgecuts, parts) = metis.part_graph(G, 5, objtype='cut')
labels, counts = np.unique(parts, return_counts=True)
