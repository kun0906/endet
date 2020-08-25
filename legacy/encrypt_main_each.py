"""Detect abnormal flows progressively using LSTM

run in the command under the examples directory
   PYTHONPATH=../:./ python3.7 lstm/rnn_main_single.py > rnn_main_single.txt 2>&1 &
"""
# Author: kun.bj@outlook.com
# license: xxx
import os
import random
import subprocess

import numpy as np
import sklearn
import torch

from model.rnn import RNN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.scorer import metric
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from pcap.parser import PCAP_PKTS, _pcap2flows
from util.tool import dump_data, load_data

import pandas as pd

RANDOM_STATE = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def set_random_state(random_state=100):
    """To make reproducible results

    Returns
    -------

    """

    random.seed(random_state)
    np.random.seed(random_state)

    torch.manual_seed(random_state)
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


set_random_state(random_state=RANDOM_STATE)


def raw2features(raw_features, header=True, MTU=1500, normalize=True):
    """Extract features for the detection model

    Parameters
    ----------
    raw_features:
        each flow := fid, [feat_0, feat_1, feat_2, ..., ]

    Returns
    -------
    features:
        each flow := [feat, ... ]
    """

    def normalize_bytes(flow):
        return [[v / 255 for v in pkt] for pkt in flow]

    def quantile(vs, qs=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]):
        data = list(np.quantile(vs, q=qs))

        data += [np.std(vs).item()]

        return data

    X = []
    for i, (fid, v_lst) in enumerate(raw_features):
        feat_0 = v_lst[0]
        feat_i_lst = v_lst[1:]

        # if header:
        #     tmp_v = [v['header'] + v['payload'] for v in feat_i_lst]
        #     tmp_v = [v + [0] * (MTU - len(v)) if len(v) < MTU else v[:MTU] for v in tmp_v]
        # else:
        #     payload_len = MTU - 40
        #     tmp_v = [v['payload'] for v in feat_i_lst]
        #     tmp_v = [v + [0] * (payload_len - len(v)) if len(v) < payload_len else v[:payload_len] for v in tmp_v]

        # {'duration': 0.0, 'n_pkts': 0, 'n_bytes': 0,
        tmp_v = [fid] + [feat_0['sport'], feat_0['duration'], feat_0['n_pkts'], feat_0['n_bytes']] + quantile(
            feat_0['ttl'])
        tmp_v2 = quantile([v['IAT'] for v in feat_i_lst]) + quantile([v['pkt_len'] for v in feat_i_lst]) + \
                 list(np.sum(np.asarray([np.asarray(v['flags']) for v in feat_i_lst]), axis=0))

        tmp_v += tmp_v2
        # if normalize:
        #     tmp_v = normalize_bytes(tmp_v)

        X.append(tmp_v)

    return X


def keep_ip(input_file, out_file='', kept_ips=['']):
    if out_file == '':
        ips_str = '-'.join(kept_ips)
        out_file = os.path.splitext(input_file)[0] + f'-src_{ips_str}.pcap'  # Split a path in root and extension.

    print(out_file)
    # only keep srcIPs' traffic
    srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
    cmd = f"tshark -r {input_file} -w {out_file} {srcIP_str}"

    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def merge_pcaps(input_files, out_file):
    cmd = f"mergecap -w {out_file} " + ' '.join(input_files)

    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def get_X_y(train_csvs, out_file=os.path.join('data/datacon/train_feats', f'train.csv')):
    vs_arr = []
    for i, csv_file, label in enumerate(train_csvs):
        vs = pd.read_csv(csv_file)
        # print(len(vs.values))
        tmp_arr = np.asarray(vs.values[:, 1:], dtype=float)
        n_vs = len(tmp_arr)
        fid = vs.values[0, 0]
        vs = [fid] + list(np.sum(tmp_arr, axis=0) / n_vs) + list(np.std(tmp_arr, axis=0)) + [label]
        vs_arr.append(vs)

    train_file = out_file
    pd.DataFrame(vs_arr).to_csv(train_file, header=None, index=None)

    train_data = pd.read_csv(train_file).values
    # fid = train_data[:, 0]
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    return X_train, y_train


def load_flow_data(overwrite=False, random_state=100, full_flow=True, filter_ip=1,
                   feature_flg=1, merge_csv=1, merge_pcap=0):
    """Get raw features from PCAP and store them into disk

    Parameters
    ----------
    overwrite: boolean
    full_flow: boolean
        use full flows, not subflows
    Returns
    -------
        X_norm, y_norm, X_abnorm, y_abnorm
    """

    train_dir = 'data/datacon/train'
    out_dir = 'data/datacon/train_feats'

    in_norm_file = os.path.join('data/datacon/', 'white.pcap')
    in_abnorm_file = os.path.join('data/datacon/', 'black.pcap')
    test_csvs = []
    train_csvs = []
    csv_files = []
    if not os.path.exists(in_norm_file) or not os.path.exists(in_abnorm_file):
        for sdir in ['white', 'black']:
            out_files = []
            tmp_sdir = os.path.join(train_dir, sdir)
            if not os.path.exists(tmp_sdir):
                os.makedirs(tmp_sdir)
            for i, file_name in enumerate(os.listdir(tmp_sdir)):
                pcap_file = os.path.join(tmp_sdir, file_name)
                print(f'{i}:{pcap_file}')
                tmp_odir = os.path.join(out_dir, sdir)
                if not os.path.exists(tmp_odir):
                    os.makedirs(tmp_odir)

                #### filter srcIP
                if filter_ip:
                    ips = [file_name.split('.pcap')[0]]
                    out_file = os.path.join(tmp_odir, file_name)
                    out_file = keep_ip(pcap_file, out_file, kept_ips=ips)
                    out_files.append(out_file)

                if feature_flg:
                    # get features for each pcap
                    # note: this work uses full flows, not subflows
                    norm_pp = PCAP_PKTS(pcap_file=pcap_file, flow_ptks_thres=2, verbose=10,
                                        random_state=RANDOM_STATE)
                    if full_flow:
                        flows = _pcap2flows(norm_pp.pcap_file, norm_pp.flow_ptks_thres,
                                            verbose=norm_pp.verbose)
                        norm_pp.flows = flows
                    else:
                        norm_pp.pcap2flows()

                    norm_pp.flows2bytes()
                    out_norm_file = os.path.join(tmp_odir, file_name + '.dat')
                    dump_data(norm_pp.features, out_norm_file)

                    X = raw2features(load_data(out_norm_file), header=False)
                    csv_file = out_norm_file + '.csv'
                    pd.DataFrame(X).to_csv(csv_file, header=None, index=None)
                    if 'white' in sdir:
                        csv_files.append([csv_file, 0])
                    else:
                        csv_files.append([csv_file, 1])

    else:
        # raise ValueError('file exists!')
        pass

    train_csvs, test_csvs = train_test_split(csv_files, test_size=0.1)
    #

    return train_csvs, test_csvs


def eval_pcap(clf, test_csvs):
    y_pred = []
    y_true = []
    fid = []
    for i, csv_file, label in enumerate(test_csvs):
        if i == 0:
            y_true.append(label)
        vs = pd.read_csv(csv_file)
        # print(len(vs.values))
        X_arr = np.asarray(vs.values[:, 1:], dtype=float)
        # n_vs = len(tmp_arr)
        tmp_fid = vs.values[0, 0]
        # vs = [fid] + list(np.sum(tmp_arr, axis=0) / n_vs) + list(np.std(tmp_arr, axis=0)) + [label]
        # vs_arr.append(vs)
        tmp_y_pred = clf.predict(X_arr)
        if sum([v for v in y_pred if v == 0]) > len(tmp_y_pred) / 2:
            tmp_y_pred = 0
        else:
            tmp_y_pred = 1

        if i == 0:
            y_true.append(label)
            fid.append(tmp_fid)

        y_pred.append(tmp_y_pred)

    report = sklearn.metrics.classification_report(y_true, y_pred)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    for i, (_y_t, _y_p) in enumerate(zip(y_true, y_pred)):
        if _y_p == 1 and _y_t == 0:  # FP (normal is 0, abnormal is 1)
            print('FP:', fid[i, 0])
        elif _y_p == 0 and _y_t == 1:  # FN
            print('FN:', fid[i, 0])


def main(random_state=100):
    train_csvs, test_csvs = load_flow_data(random_state=random_state)
    X_train, y_train = get_X_y(train_csvs)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    cv = True
    if cv:
        kf = KFold(n_splits=5)
        for _train_idx, _test_idx in kf.split(X_train, y=y_train):
            # print("%s %s" % (train, test))
            _X_train = X_train[_train_idx]
            _y_train = np.asarray(y_train)[_train_idx]
            _X_test = X_train[_test_idx]
            _y_test = np.asarray(y_train)[_test_idx]

            clf = RandomForestClassifier(random_state=0)
            clf = GradientBoostingClassifier(random_state=0)
            # clf = DecisionTreeClassifier() # min_samples_leaf=2
            # clf = LogisticRegression(C=1e5)
            # clf = sklearn.svm.SVC()
            clf.fit(_X_train[:, 1:], _y_train)
            # scores = cross_val_score(clf, X_train, y_train, cv=5)
            # print(scores)

            # eval on train set
            eval_pcap(clf, train_csvs)
            # eval on test set
            eval_pcap(clf, test_csvs)
    else:
        # # clf = RandomForestClassifier(random_state=0)
        # dt = DecisionTreeClassifier(random_state=10)
        # dt.fit(X_train, y_train)
        # print(f'dt.tree_.max_depth: {dt.tree_.max_depth}')
        # print(dt.tree_.max_depth)

        parameters = {'min_samples_leaf': [1, 3, 5, 7, 9, 10, 20], 'max_depth': [1, 5, 10, 20, 30, None]}
        parameters = {'min_samples_leaf': [1], 'max_depth': [None]}
        dt = DecisionTreeClassifier(random_state=10)
        clf = GridSearchCV(dt, parameters, verbose=1, n_jobs=-1)
        print(clf)
        clf.fit(X_train[:, 1:], y_train)

        for _param, value in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
            print(_param, value)
        # print(clf.cv_results_)
        # summarize the results of the random parameter search
        print(clf.best_score_)
        print(clf.best_estimator_)
        clf = clf.best_estimator_

    # eval on train set
    eval_pcap(clf, train_csvs)
    # eval on test set
    eval_pcap(clf, test_csvs)


if __name__ == '__main__':
    main(random_state=RANDOM_STATE)
