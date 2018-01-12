# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:37:49 2017

@author: gal
"""

from os.path import join, isfile
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns # At import it changes matplotlib settings, even unused
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cmx
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

    # Add rest of raw data from model files, such as split time per every K
    clusters_df = []
    for i, row in clusters_data.iterrows():
        algo = row['Algo']
        features = row['Features']
        dataset = row['Dataset']
        encoding = row['Encoding']
        scaling = row['Scaling']
        model_file = '%s_%s_%s_%s_%s.dmp' % (algo, features, dataset, encoding,
                                             scaling)
        model_file = join(MODELS_DIR, model_file)
        if not isfile(model_file):
            print('Model %s was not found!' % model_file)
            continue
        model_data = pd.DataFrame.from_dict(pickle.load(
                                            open(model_file, 'rb')))
        model_data = model_data[['CLUSTERING_STD', 'MINMAX_RATIO',
                                 'MAX_SPLIT_SIZE', 'MIN_SPLIT_SIZE',
                                 'SPLIT_TIME', 'k']]
        model_data = model_data.rename(columns={'k': 'K'})
        model_data['Algo'] = row['Algo']
        model_data['Dataset'] = row['Dataset']
        model_data['Encoding'] = row['Encoding']
        model_data['Scaling'] = row['Scaling']
        model_data['Features'] = row['Features']
        clusters_df.append(model_data)
    result = pd.merge(result, pd.concat(clusters_df), how='inner',
                      on=['Algo', 'Encoding', 'Scaling', 'Features',
                      'Dataset', 'K'])
    #return result
    # Return only records that had all K's evaluated for both train and test
    #filt = (((result['Count'] == 70) & (result['Scaling'] == 'Min-max')) |
    #        ((result['Count'] == 56) & (result['Scaling'] != 'Min-max')) |
    #        ((result['Count'] == 10) & (result['Scaling'] == 'Min-max') &
    #         (result['Algo'] == 'NoSplit')) |
    #        ((result['Count'] == 8) & (result['Scaling'] != 'Min-max') &
    #         (result['Algo'] == 'NoSplit')))
    filt = ((result['Count'] == 14) |
            ((result['Count'] == 2) & (result['Algo'] == 'NoSplit')))
    return result[filt]


def short_legend_entry(algo, features):
    pretty_algo = {'KMeans': 'K-means',
                   'KMeansBal3': 'K-means B3',
                   'KMeansBal4': 'K-means B4',
                   'KMeansBal5': 'KC-means',
                   'NoSplit': 'Baseline',
                   'Birch': 'BIRCH',
                   'RoundRobin': 'Round Robin',
                   'RandomRoundRobin': 'Uniform Random',
                   'WKMeans': 'WK-means',
                   'MultiPart': 'Multipart',
                   'EXLasso': 'XLK-means'}
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
        elif t.find('WKMeans') != -1:
            entry = 'WK-means'
        elif t.find('KMeansBal') != -1:
            entry = 'KC-means'
        elif t.find('KMeans') != -1:
            entry = 'K-means'
        elif t.find('NoSplit') != -1:
            entry = 'Baseline'
        elif t.find('Random') != -1:
            entry = 'Uniform Random'
        elif t.find('Robin') != -1:
            entry = 'Round Robin'
        elif t.find('MultiPart') != -1:
            entry = 'Multipart'
        elif t.find('EXLasso') != -1:
            entry = 'XLK-means'

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


def plot_lines(df, columns, index, values, file_name=None, order=None, rot=0,
               naming_fn=pretty_xaxis, cmap='PuBu_r', xlabel='', ylabel='',
               legend_loc='top right', ylim_lower=None):

    piv = pd.pivot_table(df, columns=columns, index=index, values=values,
                         aggfunc=np.mean)

    # Reorder columns
    if order is not None:
        for level, keys in enumerate(order):
            piv = piv.reindex_axis(keys, level=level, axis=1)

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

    ax.grid(True, linestyle='--', linewidth='0.5', color='black', axis='both')
    ax.set_axisbelow(True)

    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=5)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = [1, 2, 3, 4, 4, 4, 4, 0]
    line_styles = [':', '-', '-'] + ['-'] * 4 + ['-']
    markers = ['', 'p', '*', 'o', '^', 'v', 's', '']

    for i, col in enumerate(piv.columns):
        plt.plot(piv[col], color=scalarMap.to_rgba(colors[i]),
                 linestyle=line_styles[i], marker=markers[i], linewidth=4,
                 markersize=15)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = pretty_xaxis(labels)

    l = ax.legend(new_labels, loc=legend_loc, prop={'size': 20}, frameon=True,
                  ncol=3)
    frame = l.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    ylim_lower_, ylim_upper_ = ax.get_ylim()
    if ylim_lower_ is not None:
        ylim_lower_ = ylim_lower
    plt.ylim(ylim_lower_, ylim_upper_)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.set_xlabel(xlabel, fontsize=26)
    plt.tight_layout()
    if file_name is not None:
        p.savefig(file_name)
        plt.close(p)
    else:
        plt.show()


def plot_bars(df, columns, score, file_name=None, order=None, ylim_upper=1,
              ylim_lower=0.7, rot=30, bar_width=0.8, naming_fn=None,
              horizontalalignment='center', plot_colors=[(0.7, 0.7, 0.7, 1)],
              plot_hatches=[], plot_hatch_colors=[],
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
        for level, keys in enumerate(order):
            piv = piv.reindex(keys, level=level)
    #
    #    piv = piv.iloc[order]
    #    err_std = err_std.iloc[order]

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

    # Hack for printing hatches with color different than bars' edges
    # We first plot the filled bars without edges
    piv.plot(ax=ax, kind='bar', width=bar_width, legend=False,
             color=plot_colors, rot=rot, yerr=err_std.values.T,
             zorder=1, lw=0,
             error_kw=dict(lw=1, capsize=5, capthick=1))

    # Plot the hatches
    bars = ax.patches
    for bar, hatch, hatch_color in zip(bars, plot_hatches, plot_hatch_colors):
        bar.set_hatch(hatch)
        bar.set_edgecolor(hatch_color)

    # Re-plot without fill color but with default edge color and width
    piv.plot(ax=ax, kind='bar', width=bar_width, legend=False,
             rot=rot, yerr=err_std.values.T,
             zorder=2, color='none',
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
        ax.set_ylabel('MMR', fontsize=26)
    ax.set_xlabel(xlabel, fontsize=26)
    plt.tight_layout()
    if file_name is not None:
        p.savefig(file_name)
        plt.close(p)
    else:
        plt.show()


def plot_classifier(data, dataset, classifier, score, file_name=None,
                    order=None, ylim_lower=None, print_table=False,
                    cmap='PuBu_r'):
    x = select(data, {'Dataset': dataset,
                      'Classifier': [classifier]})

    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=4)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    plot_colors = [scalarMap.to_rgba(1)] * 1 + \
                  [scalarMap.to_rgba(2)] * 2 + \
                  [scalarMap.to_rgba(3)] * 20

    plot_hatches = ['/', '.', '.']
    plot_hatch_colors = ['white'] * 3

    plot_bars(x, ['Algo', 'Features'], score, file_name, order,
              naming_fn=pretty_xaxis, horizontalalignment='right',
              plot_colors=plot_colors, ylim_lower=ylim_lower,
              plot_hatches=plot_hatches, plot_hatch_colors=plot_hatch_colors,
              print_table=print_table)


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
              ylim_upper=20, plot_colors=[(0.9, 0.9, 0.9, 1.0)])


def legend_clf_info(legend):
    out = []
    for t in legend:
        s = str(t.get_text())
        parts = s.translate(None, '()').split(',')
        parts = map(lambda x: x.strip(), parts)
        algo = short_legend_entry(parts[0], parts[1])
        clf = parts[3].replace('Depth ', 'd=')
        if parts[0] == 'NoSplit':
            out.append('%s, %s' % (algo, clf))
        else:
            out.append(algo)
    return out


def plot_classifier_info(data, classifier, file_name=None, order=None,
                         cmap='PuBu_r'):
    x = select(data, {'Algo': ['NoSplit', 'MultiPart', 'EXLasso',
                               'KMeansBal5', 'KMeans', 'WKMeans'],
                      'Dataset': 'NSL Test+',
                      'Classifier': classifier,
                      'Classifier Info': ['Depth 3, Gini', 'Depth 4, Gini',
                                          'Depth 5, Gini', 'Depth 6, Gini']})
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=4)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    plot_hatches = ['/', '/', '/', '/']
    plot_hatch_colors = ['white'] * 4
    plot_bars(x, ['Algo', 'Features', 'Classifier', 'Classifier Info'],
              'F-Score', file_name, order, naming_fn=legend_clf_info,
              horizontalalignment='right',
              plot_colors=[scalarMap.to_rgba(1)] * 4 +
                          [scalarMap.to_rgba(3)] * 8,
              plot_hatches=plot_hatches, plot_hatch_colors=plot_hatch_colors)


def plot_roc_lb(data, dataset, classifier, k, label,
                file_name=None, order=None, cmap='PuBu_r'):
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

    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=3)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    algo_to_lb = {'EXLasso': 'CLB',
                  'KMeansBal5': 'CLB',
                  'MultiPart': 'CLB',
                  'WKMeans': 'CLB',
                  'KMeans': 'CLB',
                  'NoSplit': 'Baseline',
                  'RandomRoundRobin': 'Traditional',
                  'RoundRobin': 'Traditional'}

    color1 = scalarMap.to_rgba(2)
    color2 = scalarMap.to_rgba(1)
    color3 = scalarMap.to_rgba(0)
    colors = {'EXLasso': color1,
              'KMeansBal5': color1,
              'MultiPart': color1,
              'WKMeans': color1,
              'KMeans': color1,
              'NoSplit': color3,
              'RandomRoundRobin': color2,
              'RoundRobin': color2}
    linestyle1 = '-'
    linestyle2 = '--'
    linestyle3 = ':'
    line_styles = {'EXLasso': linestyle1,
                   'KMeansBal5': linestyle1,
                   'MultiPart': linestyle1,
                   'WKMeans': linestyle1,
                   'KMeans': linestyle1,
                   'NoSplit': linestyle3,
                   'RandomRoundRobin': linestyle2,
                   'RoundRobin': linestyle2}

    linewidth1 = 5
    linewidth2 = 2
    linewidth3 = 8
    line_widths = {'EXLasso': linewidth1,
                   'KMeansBal5': linewidth1,
                   'MultiPart': linewidth1,
                   'WKMeans': linewidth1,
                   'KMeans': linewidth1,
                   'NoSplit': linewidth3,
                   'RandomRoundRobin': linewidth2,
                   'RoundRobin': linewidth2}                   

    x = select(data, {'Dataset': dataset,
                      'Classifier': classifier})
    reverse_cols = {}
    for i, col in enumerate(x.columns):
        reverse_cols[col] = i

    i = 0
    results = []
    max_auc = {}
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

        lb_fam = algo_to_lb[algo]
        if lb_fam not in max_auc:
            max_auc[lb_fam] = model[index]['AUC'][label]
        elif model[index]['AUC'][label] > max_auc[lb_fam]:
            max_auc[lb_fam] = model[index]['AUC'][label]
        results.append(result)

    if order is not None:
        results = [z for (y, z) in sorted(zip(order, results))]

    for i, result in enumerate(results):
        plt.plot(result['FPR'], result['TPR'],
                 linewidth=line_widths[result['Algo']],
                 linestyle=line_styles[result['Algo']],
                 color=colors[result['Algo']])
        line = '%s: %s%.2f' % (short_legend_entry(result['Algo'],
                                                  result['Features']),
                               'AUC=' if i == 0 else '',
                               result['AUC'])
        # legend.append(line)

    plt.xlabel('False Positive Rate', fontsize=32)
    plt.ylabel('True Positive Rate', fontsize=32)
    legend = ['Clustering LB (Max. AUC=%.2f)' % max_auc['CLB'],
              'Traditional LB (Max. AUC=%.2f)' % max_auc['Traditional'],
              'Baseline (Max. AUC=%.2f)' % max_auc['Baseline']]
    l = ax.legend(legend, loc='lower right',
                  prop={'size': 31}, frameon=True, ncol=1)
    l.legendHandles[0].set_color(color1)
    l.legendHandles[0].set_linestyle(linestyle1)
    l.legendHandles[0].set_linewidth(linewidth1)
    l.legendHandles[1].set_color(color2)
    l.legendHandles[1].set_linestyle(linestyle2)
    l.legendHandles[1].set_linewidth(linewidth2)
    l.legendHandles[2].set_color(color3)
    l.legendHandles[2].set_linestyle(linestyle3)
    l.legendHandles[2].set_linewidth(linewidth3)
    frame = l.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.tight_layout()
    if file_name is not None:
        p.savefig(file_name)
        plt.close(p)
    else:
        plt.show()


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
                            percentage=True, file_name=None, cmap='viridis'):
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

    data.plot(ax=ax, kind='bar', stacked=True, legend=True,
              cmap=cmap, rot=0, width=0.8)
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
