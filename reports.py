# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:37:49 2017

@author: gal
"""

from os import listdir
from os.path import join, splitext
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns # At import it changes matplotlib settings, even unused
from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.lines import Line2D
# import matplotlib.lines as mlines
import pickle
import numpy as np
import pandas as pd


from settings import CLASSIFIERS_REPORT, MODELS_DIR
from settings import MAX_CLUSTERS_STD, MAX_CLUSTERS_RATIO
from clustering_model import is_feasible


def select(df, h):
    out = df
    for k, v in h.iteritems():
        if (v.__class__ == str):
            out = out[out[k] == v]
        elif (v.__class__ == list):
            out = out[out[k].isin(v)]
        elif (v.__class__ == bool):
            out = out[out[k] == v]
        else:
            exit
    return out


def data_add_column(ds, data, att):
    """Augments classifiers dataset with attribute from clustering model data
    """
    new_ds = ds['K'].apply(lambda x:
                           [y for y in data if y['k'] == x][0][att])
    new_ds.rename(att, inplace=True)
    return pd.concat([ds, new_ds], axis=1)


def feasible_models(sigma=MAX_CLUSTERS_STD):
    """Produce dataset of feasible models

    Feasibility criteria:
        Std. deviation of cluster sizes < predefined threshold (6000)
        Requested number of clusters returned (not always the case with Birch)
        All classifiers returned valid F-score for both training and testing
    """
    clf_data = pd.read_csv(CLASSIFIERS_REPORT)
    feasible_records = []

    for f in listdir(MODELS_DIR):
        algo, features, dataset, encoding, scaling = splitext(f)[0].split('_')
        data = pickle.load(open(join(MODELS_DIR, f), 'rb'))

        # Step 1 - filter feasible splitters that have classification result(s)
        if is_feasible(data, sigma):
            clf_rec = select(clf_data, {'Dataset': dataset,
                                        'Algo': algo,
                                        'Features': features,
                                        'Encoding': encoding,
                                        'Scaling': scaling})

            # Only clusterings that have classification results
            if len(clf_rec) > 0:
                # Augment information
                clf_rec = data_add_column(clf_rec, data, 'MINMAX_RATIO')
                clf_rec = data_add_column(clf_rec, data, 'MAX_SPLIT_SIZE')
                clf_rec = data_add_column(clf_rec, data, 'MIN_SPLIT_SIZE')
                clf_rec = data_add_column(clf_rec, data, 'CLUSTERING_STD')
                feasible_records.append(clf_rec)

    # Merge everything gathered until now to a single DataFrame
    final_records = pd.concat(feasible_records)

    # Step 2 - select feasible clustering models

    # Group data by Algo, Features, Encoding, Scaling and count records
    piv = pd.pivot_table(final_records, values=[],
                         index=['Algo', 'Features', 'Encoding', 'Scaling'],
                         aggfunc=lambda x: len(x))
    piv = piv.reset_index().rename(columns={0: 'Count'})

    # Join with data
    result = pd.merge(final_records, piv, how='inner',
                      on=['Algo', 'Features', 'Encoding', 'Scaling'])

    # Group data by Algo, Features, Encoding, Scaling and find maximal minmax
    # ratio
    piv = pd.pivot_table(result, values=['MINMAX_RATIO'],
                         index=['Algo', 'Features', 'Encoding', 'Scaling'],
                         aggfunc=np.max)
    piv = piv.reset_index().rename(columns={'MINMAX_RATIO': 'MODEL_MAX_RATIO'})

    # Join with data
    result = pd.merge(result, piv, how='inner',
                      on=['Algo', 'Features', 'Encoding', 'Scaling'])

    # Select clustering models for which all classifiers returned scores on
    # both training and testing. There are 5 classifiers for scaling None and
    # 4 classifiers for scaling min-max. Therefore we should select records
    # with:
    #   2 datasets x 7 k's x 5 classifiers = 70 records for min-max scaling.
    #   2 datasets x 7 k's x 4 classifiers = 56 records for no scaling.
    # For the baseline (NoSplit) similar calculation but with single k
    filt = (((result['Count'] == 70) & (result['Scaling'] == 'Min-max')) |
            ((result['Count'] == 56) & (result['Scaling'] == 'None')) |
            ((result['Count'] == 10) & (result['Scaling'] == 'Min-max') &
             (result['Algo'] == 'NoSplit')) |
            ((result['Count'] == 8) & (result['Scaling'] == 'None') &
             (result['Algo'] == 'NoSplit'))) & \
           (result['MODEL_MAX_RATIO'] < MAX_CLUSTERS_RATIO)

    return result[filt]

    # Aggregate data according to algorithm, features and count records
    piv = pd.pivot_table(final_records, values=[], index=['Algo', 'Features'],
                         aggfunc=lambda x: len(x))
    piv = piv.reset_index().rename(columns={0: 'Count1'})

    # Join with data
    result = pd.merge(final_records, piv, how='inner', on=['Algo', 'Features'])

    # Aggregate according to algorithm, features, scaling and count records
    piv = pd.pivot_table(final_records, values=[], index=['Algo', 'Features',
                                                          'Scaling'],
                         aggfunc=lambda x: len(x))

    # Join with data
    piv = piv.reset_index().rename(columns={0: 'Count2'})
    result = pd.merge(result, piv, how='inner', on=['Algo', 'Features',
                                                    'Scaling'])

    # Select models with all clustering options evaluated:
    # 2 dataset x 2 encoding x (5 classifiers min-max + 4 classifiers no scale)
    # x 7 different k's = 252
    # -- OR --
    # Models with all clustering options using min-max scaling:
    # 2 dataset x  2 encoding x 5 classifiers x 7 k's = 140
    # --- OR ---
    # Models with all clustering options without scaling:
    # 2 dataset x 2 encoding x 4 classifiers x 7 k's = 112 and make sure that's
    # indeed record without scale (as opposed to min-max with 4 classifiers)
    # --- OR ---
    # The baseline: 2 dataset x 2 encoding x 2 scaling x 4 classifiers = 36
    result = result[(result['Count1'] == 252) | (result['Count2'] == 140) |
                    ((result['Count2'] == 112) & (result['Scaling'] is None)) |
                    ((result['Count1'] == 36) & (result['Algo'] == 'NoSplit'))]

    return result


def pretty_xaxis(l):
    out = []
    for t in l:
        t = str(t)
        entry = ""
        if t.find('Birch') != -1:
            entry = 'BIRCH'
        elif t.find('KMeans') != -1:
            entry = 'K-means'
        elif t.find('NoSplit') != -1:
            entry = 'Baseline'
        elif t.find('Random') != -1:
            entry = 'Unified Random'
        elif t.find('Robin') != -1:
            entry = 'Round Robin'

        if t.find('100 connections') != -1:
            entry += ' [3]'
        elif t.find('all history') != -1:
            entry += ' [7]'
        elif t.find('same dest') != -1:
            entry += ' [1]'
        elif t.find('same service') != -1:
            entry += ' [2]'
        elif t.find('all 2 secs') != -1:
            entry += ' [6]'
        elif t.find('TCP') != -1:
            entry += ' [5]'

        out.append(entry)
    return out


def plot_bars(df, columns, score, file_name=None, order=None, ylim_upper=1,
              ylim_lower=0.7, rot=30, bar_width=0.8, naming_fn=None,
              horizontalalignment='center', plot_colors=[(0.7, 0.7, 0.7, 1)],
              xlabel=''):

    piv = pd.pivot_table(df, values=[score], index=columns,
                         aggfunc=[np.average])

    err_std = pd.pivot_table(df, values=[score], index=columns,
                             aggfunc=[np.std])

    # Reorder columns
    if order is not None:
        piv = piv.iloc[order]
        err_std = err_std.iloc[order]

    # Matplotlib settings
    matplotlib.rcParams.update({'font.family': 'Times New Roman'})

    if file_name is not None:
        plt.ioff()
    else:
        plt.ion()

    # Create Figure
    p = plt.figure(figsize=(10, 6))
    ax = p.add_subplot(111)
    ax.set_axis_bgcolor('white')

    ax.grid(True, linestyle='--', linewidth='0.5', color='black', axis='y')
    ax.set_axisbelow(True)

    piv.plot(ax=ax, kind='bar', width=bar_width, legend=False,
             color=plot_colors, rot=rot, yerr=err_std.values.T,
             error_kw=dict(lw=1, capsize=5, capthick=1))

    if naming_fn is not None:
        ax.set_xticklabels(naming_fn(ax.get_xticklabels()),
                           horizontalalignment=horizontalalignment)

    plt.ylim(ylim_lower, ylim_upper)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    if (score == 'F-Score'):
        ax.set_ylabel('Average F-score', fontsize=26)
    elif (score.find('AUC') != -1):
        ax.set_ylabel('Area Under Curve', fontsize=26)
    ax.set_xlabel(xlabel, fontsize=26)
    plt.tight_layout()
    if file_name is not None:
        p.savefig(file_name)
        plt.close(p)
    else:
        plt.show()


def plot_classifier(data, dataset, classifier, score, file_name=None,
                    order=None):
    x = select(data, {'Dataset': dataset,
                      'Classifier': [classifier]})
    cmap_greys = LinearSegmentedColormap.from_list("my_colormap",
                                                   ((0.5, 0.5, 0.5),
                                                    (0.9, 0.9, 0.9)),
                                                   N=3, gamma=1.0)
    colors = [cmap_greys(float(i)/3) for i in range(0, 4)]
    plot_colors = [colors[0], colors[1], colors[1], colors[2], colors[2],
                   colors[2], colors[2], colors[2], colors[2]]
    plot_bars(x, ['Algo', 'Features'], score, file_name, order,
              naming_fn=pretty_xaxis, horizontalalignment='right',
              plot_colors=plot_colors)


def plot_scalability(data, dataset, algo, scaling, score, file_name=None,
                     order=None):
    filters = {'Dataset': dataset, 'Scaling': scaling, 'Algo': [algo]}
    x = select(data, filters)
    plot_bars(x, ['K'], score, file_name, order, ylim_lower=0.8,
              xlabel='Number of Clusters (k)', rot=0,
              plot_colors=[(0.9, 0.9, 0.9, 1.0)])
