# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:37:49 2017

@author: gal
"""

from os.path import join
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns # At import it changes matplotlib settings, even unused
from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.lines import Line2D
# import matplotlib.lines as mlines
import pickle
import numpy as np
import pandas as pd


from settings import CLASSIFIERS_REPORT, MODELS_DIR, CLFS_DIR
from settings import CLUSTERS_REPORT


def select(df, h):
    out = df
    for k, v in h.iteritems():
        if (v.__class__ == str or v.__class__ == int):
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


def feasible_classifiers():
    """Produce dataset of feasible classifiers

    Feasibility criteria:
        Clustering feasibilty criteria + each classifier has results for all
        K's for all datasets.
    """

    clf_data = pd.read_csv(CLASSIFIERS_REPORT)

    # Group data by Algo, Features, Encoding, Scaling and count records
    piv = pd.pivot_table(clf_data, values=[], index=['Algo', 'Features',
                         'Encoding', 'Scaling', 'Classifier'],
                         aggfunc=lambda x: len(x))
    piv = piv.reset_index().rename(columns={0: 'Count'})

    # Add the count column to original clf_data
    result = pd.merge(clf_data, piv, how='inner', on=['Algo', 'Features',
                      'Encoding', 'Scaling', 'Classifier'])

    # Read clusters report
    clusters_data = pd.read_csv(CLUSTERS_REPORT).rename(columns={
        'Algorithm': 'Algo', 'Max Ratio': 'MODEL_MAX_RATIO'})

    # Add Max Std., Max Ratio, and Avg. Split Time to our data
    result = pd.merge(result, clusters_data, how='inner', on=['Algo',
                      'Features', 'Encoding', 'Scaling', 'Dataset'])

    # Return only records that had all K's evaluated for both train and test
    filt = ((result['Count'] == 14) |
            ((result['Count'] == 2) & (result['Algo'] == 'NoSplit')))
    return result[filt]


def short_legend_entry(algo, features):
    pretty_algo = {'KMeans': 'K-means',
                   'KMeansBal3': 'K-means B3',
                   'KMeansBal4': 'K-means B4',
                   'KMeansBal5': 'K-means B5',
                   'NoSplit': 'Baseline',
                   'Birch': 'BIRCH',
                   'RoundRobin': 'Round Robin',
                   'RandomRoundRobin': 'Unified Random',
                   'WKMeans': 'K-means W',
                   'Multipart': 'GraphPart',
                   'EXLasso': 'EXLasso'}
    features_tbl = {'100 connections same host': 3,
                    'all 2 secs': 6,
                    'all history based features': 7,
                    'single TCP features': 5}
    if features not in features_tbl:
        return pretty_algo[algo]
    else:
        return '%s [%d]' % (pretty_algo[algo], features_tbl[features])


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
              xlabel='', print_table=False):

    piv = pd.pivot_table(df, values=[score], index=columns,
                         aggfunc=[np.average])

    err_std = pd.pivot_table(df, values=[score], index=columns,
                             aggfunc=[np.std])
    if print_table:
        print(piv)
        print(err_std)

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
    elif (score.find('MODEL_MAX_RATIO') != -1):
        ax.set_ylabel('Maximal Ratio', fontsize=26)
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
              plot_colors=[(0.9, 0.9, 0.9, 1.0)], print_table=True)


def plot_max_ratio(data, algo, file_name=None, order=None):
    x = select(data, {'Algo': algo})
    plot_bars(x, ['Algo', 'Features'], 'MODEL_MAX_RATIO', file_name, order,
              naming_fn=pretty_xaxis, horizontalalignment='right',
              ylim_upper=18, plot_colors=[(0.9, 0.9, 0.9, 1.0)])


def legend_clf_info(legend):
    out = []
    for t in legend:
        s = str(t.get_text())
        parts = s.translate(None, '()').split(',')
        parts = map(lambda x: x.strip(), parts)
        algo = short_legend_entry(parts[0], parts[1])
        clf = parts[3].replace('Depth ', 'DT')
        out.append('%s, %s' % (algo, clf))
    return out


def plot_classifier_info(data, classifier, file_name=None, order=None):
    x = select(data, {'Algo': ['KMeans', 'NoSplit'],
                      'Classifier': classifier,
                      'Classifier Info': ['Depth 3, Gini',
                                          'Depth 4, Gini', 'Depth 5, Gini']})
    plot_bars(x, ['Algo', 'Features', 'Classifier', 'Classifier Info'],
              'F-Score', file_name, order, naming_fn=legend_clf_info,
              horizontalalignment='right', plot_colors=[(0.9, 0.9, 0.9, 1.0)])


def plot_roc(data, dataset, scaling, encoding, classifier, k, label,
             file_name=None, order=None):
    markers = {'No Split': '',  # * 8
               'Round Robin': '',
               'Random Round Robin': '',
               'KMeans': '*',
               'Birch': ''}

    line_styles = {'No Split': '-',  # * 8
                   'Round Robin': '--',
                   'Random Round Robin': '--',
                   'KMeans': '-',
                   'Birch': '-'}
    matplotlib.rcParams.update({'font.family': 'Times New Roman'})
    if file_name:
        plt.ioff()
    else:
        plt.ion()
    plt.ioff()
    p = plt.figure(figsize=(10, 6.5))
    ax = p.add_subplot(111)
    ax.set_axis_bgcolor('white')
    ax.grid(True, linestyle='--', linewidth='0.5', color=(0.2, 0.2, 0.2, 1),
            axis='both')
    ax.set_axisbelow(True)
    # ax.xaxis.tick_top()
    # ax.tick_params(direction='out', pad=50)
    # ax.xaxis.set_label_position('top')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    scope = 0.65  # 0.65
    plt.xlim([0.0, scope])
    plt.ylim([1.0-scope, 1.0])
    legend = []
    # if order is None:
    #     order = xrange(0, num_results)

    line_widths = [12, 8, 8, 6, 6, 6, 6, 6]
    line_styles = ['-', '--', '--', '-', '-', '-', '-', '-']
    markers = ['', '', '', 'o', '^', 'v', 's', 'p']
    mark_every = [10, 10, 10, (20, 1000), (25, 100), (20, 100), (30, 100)]

    num_results = len(pd.pivot_table(data, values=[],
                                     index=['Algo', 'Features'],
                                     aggfunc=lambda x: len(x)))
    cmap_greys_full = LinearSegmentedColormap.from_list('my_colormap',
                                                        ((0.7, 0.7, 0.7, 1),
                                                         (0.2, 0.2, 0.2, 1)),
                                                        N=num_results,
                                                        gamma=1.0)
    colors_full = [cmap_greys_full(float(i)/num_results) for i in
                   range(0, num_results+1)]

    x = select(data, {'Dataset': dataset,
                      'Scaling': scaling,
                      'Encoding': encoding,
                      'Classifier': classifier})
    reverse_cols = {}
    for i, col in enumerate(x.columns):
        reverse_cols[col] = i

    i = 0
    results = []
    for r in x.iterrows():
        algo = r[1][reverse_cols['Algo']]
        k_ = r[1][reverse_cols['K']]
        scaling = r[1][reverse_cols['Scaling']]
        encoding = r[1][reverse_cols['Encoding']]
        # We want to include the baseline so we don't originally select with
        # specific K, to include it in selection (since it is always k=1), then
        # we filter out all irrelevant K's expect for baseline
        if k != k_ and algo != 'NoSplit':
            continue
        features = r[1][reverse_cols['Features']]
        clf_file = '%s_%s_%s_%s_%s_%s.dmp' % (algo, features, dataset,
                                              encoding, scaling, classifier)

        model = pickle.load(open(join(CLFS_DIR, clf_file), 'rb'))
        index = 0 if algo == 'NoSplit' else k-3
        result = {'Algo': algo,
                  'Features': features,
                  'Scaling': scaling,
                  'Encoding': encoding,
                  'TPR': model[index]['TPR'][label],
                  'FPR': model[index]['FPR'][label],
                  'AUC': model[index]['AUC'][label]}
        results.append(result)

    if order is not None:
        results = [z for (y, z) in sorted(zip(order, results))]

    for i, result in enumerate(results):
        plt.plot(result['FPR'], result['TPR'],
                 linewidth=line_widths[i],
                 color=colors_full[i],
                 linestyle=line_styles[i],
                 marker=markers[i],
                 markersize=29,
                 markevery=mark_every[i])
        line = '%s: %s%.2f' % (short_legend_entry(result['Algo'],
                                                  result['Features']),
                               'AUC=' if i == 0 else '',
                               result['AUC'])
        legend.append(line)
        i = i+1

    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    l = ax.legend(legend, loc='lower right', bbox_to_anchor=[1.03, -0.04],
                  prop={'size': 26}, frameon=True, ncol=1)
    frame = l.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.tight_layout()
    if file_name is not None:
        p.savefig(file_name)
        plt.close(p)
    else:
        plt.show()


def plot_class_distribution(algo, features, dataset, encoding, scaling, k,
                            percentage=True, file_name=None):
    model_file = "%s_%s_%s_%s_%s.dmp" % (algo, features, dataset, encoding,
                                         scaling)
    model = pickle.load(open(join(MODELS_DIR, model_file), 'rb'))
    data = pd.DataFrame(model[k-3]['LABEL_COUNTS']).T
    if percentage:
        for col in data.columns:
            data[col] = data[col]*100/data[col].sum()
    data = data.T

    # Create Figure
    p = plt.figure(figsize=(10, 4.7))
    ax = p.add_subplot(111)
    ax.set_axis_bgcolor('white')

    ax.grid(True, linestyle='--', linewidth='0.5', color='black', axis='y')
    ax.set_axisbelow(True)

    cmap_greys = LinearSegmentedColormap.from_list("my_colormap",
                                                   ((0.2, 0.2, 0.2),
                                                    (0.95, 0.95, 0.95)),
                                                   N=5, gamma=1.0)
    colors = [cmap_greys(float(i)/5) for i in range(0, 5)]

    data.plot(ax=ax, kind='bar', stacked=True, legend=True, color=colors,
              rot=0, width=0.8)
    ax.set_xticklabels(range(1, len(data)+1))

    if percentage:
        plt.ylim(0, 115)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    ax.set_ylabel('Flow Label Distribution', fontsize=26)
    ax.set_xlabel('Cluster Number', fontsize=26)

    l = ax.legend(data.columns, loc=3, bbox_to_anchor=[0., 0.9, 1., .102],
                  prop={'size': 21}, frameon=True, ncol=5, mode="expand",
                  borderaxespad=0.)
    frame = l.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    plt.tight_layout()
    if file_name is not None:
        p.savefig(file_name)
        plt.close(p)
    else:
        plt.show()
