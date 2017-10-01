import numpy as np
import dataset as ds
from dataset import NSL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils.validation import check_array
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from exlasso import _exlasso
from splitters import EXLasso

data = NSL(ds.NSL_TRAIN20, ds.ENC_NUMERIC, ds.SCL_MINMAX)

X = check_array(data.ds, order="C")

n_clusters = 2


ex = EXLasso(n_clusters, init='random', gamma=0.3)
#res = ex.fit_predict(X)
#print(np.unique(res, return_counts=True))

np.random.seed(0)
n_samples = 10000

varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=175, n_features=3)

X, y = varied
X = StandardScaler().fit_transform(X)
# y_pred = ex.fit_predict(np.array(X))
_, y_pred, _ = _exlasso.exlasso(np.ascontiguousarray(X), n_clusters, verbose=True)

print(np.unique(y_pred, return_counts=True))

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color=colors[y_pred])

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xticks(())
plt.yticks(())