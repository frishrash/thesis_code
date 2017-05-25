# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:37:49 2017

@author: gal
"""

from os import listdir
from os.path import join, splitext
import pickle
import pandas as pd

from settings import CLASSIFIERS_REPORT, MODELS_DIR, MAX_CLUSTERS_STD
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
        All clustering options (datasets x encodings x scalings x classifiers)
            return valid F-score, with the exception of SVM evaluated only on
            min-max scaling
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
                feasible_records.append(clf_rec)

    # Merge everything gathered until now to a single DataFrame
    final_records = pd.concat(feasible_records)

    # Step 2 - select models that are fully feasible

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
