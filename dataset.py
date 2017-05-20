# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:11:20 2017

@author: gal
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import settings

# Public consts
NSL_TRAIN = 'NSL Train+'
NSL_TRAIN20 = 'NSL Train+ 20%'
NSL_TEST = 'NSL Test+'

SCL_NONE = 'None'
SCL_MINMAX = 'Min-max'
SCL_STD = 'Standard'

ENC_NUMERIC = 'Numeric'
ENC_HOT = 'Hot Encode'


class NSL:
    """ Loads the NSL-KDD dataset """

    _DS_TO_FILE = {
        NSL_TRAIN: settings.NSL_TRAIN_FILE,
        NSL_TRAIN20: settings.NSL_TRAIN20_FILE,
        NSL_TEST: settings.NSL_TEST_FILE
        }
    FEATURES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
        ]
    FEATURES_NUMERIC = [
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
        ]
    FEATURES_2SECS = [
        'count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_count', 'srv_serror_rate', 'srv_rerror_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
    FEATURES_2SECS_HOST = [
        'count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_count'
        ]
    FEATURES_2SECS_SERVICE = [
        'srv_serror_rate', 'srv_rerror_rate', 'srv_diff_host_rate'
        ]
    FEATURES_100CONNS_HOST = [
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_ate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
        ]
    FEATURES_TCP = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent'
        ]
    FEATURES_EXPERT = [
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login'
        ]
    _STANDARD_LABELS = ("DOS", "NORMAL", "PROBE", "R2L", "U2R")
    _CLASS_LABELS = {
        'apache2': 'DOS', 'back': 'DOS', 'buffer_overflow': 'U2R',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'U2R',
        'imap': 'R2L', 'ipsweep': 'PROBE', 'land': 'DOS',
        'loadmodule': 'U2R', 'mailbomb': 'DOS', 'mscan': 'PROBE',
        'multihop': 'R2L', 'named': 'R2L', 'neptune': 'DOS', 'nmap': 'PROBE',
        'perl': 'U2R', 'phf': 'R2L', 'pod': 'DOS', 'portsweep': 'PROBE',
        'processtable': 'DOS', 'ps': 'U2R', 'rootkit': 'U2R', 'saint': 'PROBE',
        'satan': 'PROBE', 'sendmail': 'R2L', 'smurf': 'DOS',
        'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'spy': 'R2L',
        'sqlattack': 'U2R', 'teardrop': 'DOS', 'udpstorm': 'DOS',
        'warezclient': 'R2L', 'warezmaster': 'R2L', 'worm': 'R2L',
        'xlock': 'R2L', 'xsnoop': 'R2L', 'xterm': 'U2R', 'normal': 'NORMAL',
        }

    def __init__(self, dataset, encoding=ENC_NUMERIC, scaling=SCL_NONE):
        self.ds_file = self._DS_TO_FILE[dataset]
        self.ds_name = dataset
        self.encoding = encoding
        self.scaling = scaling

        if (encoding == ENC_NUMERIC):
            self.ds = pd.read_csv(
                self.ds_file, names=self.FEATURES + ['class', 'dummy'],
                index_col=-1, usecols=self.FEATURES_NUMERIC + ['class'])
        elif (encoding == ENC_HOT):
            self.ds = pd.read_csv(
                self.ds_file, names=self.FEATURES + ['class'], index_col=-1,
                usecols=self.FEATURES + ['class'])
            self.ds = pd.get_dummies(
                self.ds, columns=['flag', 'protocol_type', 'service'])

        if (scaling == SCL_MINMAX):
            self.ds = pd.DataFrame(
                MinMaxScaler().fit_transform(self.ds.values),
                index=self.ds.index,
                columns=self.ds.columns
                )
        elif (scaling == SCL_STD):
            self.ds = pd.DataFrame(
                StandardScaler().fit_transform(self.ds.values),
                index=self.ds.index,
                columns=self.ds.columns
                )

    @classmethod
    def get_labels(cls, ds):
        return ds.index.to_series().apply(lambda x: cls._CLASS_LABELS[x])

    @classmethod
    def standard_labels(cls):
        return cls._STANDARD_LABELS

    def labels(self):
        return self.ds.index.to_series().apply(lambda x: self._CLASS_LABELS[x])
