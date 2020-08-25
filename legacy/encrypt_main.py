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
    header = False
    payload = False

    for i, (fid, v_lst) in enumerate(raw_features):
        feat_0 = v_lst[0]
        feat_i_lst = v_lst[1:]

        if header:
            tmp_v = [v['header'] + v['payload'] for v in feat_i_lst]
            tmp_v = [v + [0] * (MTU - len(v)) if len(v) < MTU else v[:MTU] for v in tmp_v]
        elif payload:
            payload_len = MTU - 40
            tmp_v = [v['payload'] for v in feat_i_lst]
            tmp_v = [v + [0] * (payload_len - len(v)) if len(v) < payload_len else v[:payload_len] for v in tmp_v]
        else:
            # {'duration': 0.0, 'n_pkts': 0, 'n_bytes': 0,
            tmp_v = [fid] + [feat_0['sport'], feat_0['duration'], feat_0['n_pkts'], feat_0['n_bytes']]
            # tmp_v2=[]
            tmp_v2 = list(np.sum(np.asarray([np.asarray(v['alert']) for v in feat_i_lst]), axis=0)) + \
                     quantile(feat_0['ttl']) + quantile([v['IAT'] for v in feat_i_lst]) + quantile(
                [v['pkt_len'] for v in feat_i_lst]) + \
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


def load_flow_data(overwrite=False, random_state=100, full_flow=True, filter_ip=0,
                   feature_flg=1, merge_csv=0, merge_pcap=0):
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
    norm_ips = []
    abnorm_ips = []
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
                    # pcap_file ='data/datacon/train/black/192.168.84.149.pcap' # for debug
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
                        norm_ips.extend([x[0][0] for x in X if not x[0][0].startswith('192.168')])  # srcIP
                        norm_ips.extend([x[0][1] for x in X if not x[0][1].startswith('192.168')])  # dstIP
                    else:
                        abnorm_ips.extend([x[0][0] for x in X if not x[0][0].startswith('192.168')])
                        abnorm_ips.extend([x[0][1] for x in X if not x[0][1].startswith('192.168')])
                    # break
            ### merge csv:

            if merge_csv:
                vs_arr = []
                for i, file_name in enumerate(os.listdir(tmp_odir)):
                    if file_name.endswith('.dat.csv'):
                        csv_file = os.path.join(tmp_odir, file_name)
                        vs = pd.read_csv(csv_file)
                        # print(len(vs.values))
                        tmp_arr = np.asarray(vs.values[:, 1:], dtype=float)
                        n_vs = len(tmp_arr)
                        fid = vs.values[0, 0]

                        vs = [fid] + [n_vs] + list(np.sum(tmp_arr, axis=0)) + list(np.std(tmp_arr, axis=0))
                        vs_arr.append(vs)

                csv_file = os.path.join('data/datacon/train_feats', f'{sdir}-sum.csv')
                pd.DataFrame(vs_arr).to_csv(csv_file, header=None, index=None)

            ### merge pcaps
            if merge_pcap:
                if 'white' in sdir:
                    merge_pcaps(out_files, out_file=in_norm_file)
                else:
                    merge_pcaps(out_files, out_file=in_abnorm_file)

    else:
        # raise ValueError('file exists!')
        pass

    print(f'set(norm_ips): {set(norm_ips)}')
    print(f'set(abnorm_ips): {set(abnorm_ips)}')

    norm_file = os.path.join('data/datacon/train_feats', f'white-sum.csv')
    X_norm = pd.read_csv(norm_file).values
    fid = X_norm[:, 0]
    y_norm = [0] * len(X_norm)

    abnorm_file = os.path.join('data/datacon/train_feats', f'black-sum.csv')
    X_abnorm = pd.read_csv(abnorm_file).values
    y_abnorm = [1] * len(X_abnorm)

    print(f'y_norm: {len(X_norm)}, y_abnorm: {len(y_abnorm)}')

    X_train = np.concatenate([X_norm, X_abnorm], axis=0)
    y_train = y_norm + y_abnorm
    X_test = X_train
    y_test = y_train

    csv_file = os.path.join('data/datacon/train_feats', f'feat-label.csv')
    vs_arr = np.concatenate([X_train, np.asarray(y_train).reshape(-1, 1)], axis=1)
    pd.DataFrame(vs_arr).to_csv(csv_file, header=None, index=None)

    return X_train, y_train, X_test, y_test

    # out_norm_file = in_norm_file + '-raw_normal_features.dat'
    # out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'
    #
    # if overwrite or not os.path.exists(out_norm_file) or not os.path.exists(out_abnorm_file):
    #     # note: this work uses full flows, not subflows
    #     norm_pp = PCAP_PKTS(pcap_file=in_norm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
    #     if full_flow:
    #         flows = _pcap2flows(norm_pp.pcap_file, norm_pp.flow_ptks_thres,
    #                             verbose=norm_pp.verbose)
    #         norm_pp.flows = flows
    #     else:
    #         norm_pp.pcap2flows()
    #
    #     norm_pp.flows2bytes()
    #     out_norm_file = in_norm_file + '-raw_normal_features.dat'
    #     dump_data(norm_pp.features, out_norm_file)
    #
    #     abnorm_pp = PCAP_PKTS(pcap_file=in_abnorm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
    #     if full_flow:
    #         flows = _pcap2flows(abnorm_pp.pcap_file, abnorm_pp.flow_ptks_thres,
    #                                     verbose=abnorm_pp.verbose)
    #         abnorm_pp.flows = flows
    #     else:
    #         abnorm_pp.pcap2flows(interval=norm_pp.interval)
    #     abnorm_pp.flows2bytes()
    #
    #     out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'
    #     dump_data(abnorm_pp.features, out_abnorm_file)
    #
    # X_norm = raw2features(load_data(out_norm_file), header=False)
    # y_norm = [0] * len(X_norm)
    # X_abnorm = raw2features(load_data(out_abnorm_file), header=False)
    # y_abnorm = [1] * len(X_abnorm)
    #
    # print(f'y_norm: {len(y_norm)}, y_abnorm: {len(y_abnorm)}')
    # # return split_train_test(X_norm, y_norm, X_abnorm, y_abnorm, random_state)
    # return X_norm+X_abnorm, y_norm+y_abnorm, X_norm+X_abnorm, y_norm+y_abnorm


def split_train_test(X_norm, y_norm, X_abnorm, y_abnorm, random_state=100):
    """Split train and test set

    Parameters
    ----------
    X_norm
    y_norm
    X_abnorm
    y_abnorm
    random_state

    Returns
    -------

    """

    # X_norm = sklearn.utils.shuffle(X_norm, random_state)
    random.Random(random_state).shuffle(X_norm)
    size = int(len(y_norm) // 2) if len(y_norm) <= len(y_abnorm) else min(400, len(y_abnorm))
    X_test = X_norm[-size:] + X_abnorm[:size]
    y_test = y_norm[-size:] + y_abnorm[:size]
    X_train = X_norm[:-size]
    y_train = y_norm[:-size]
    print(f'X_train: {len(X_train)}, X_test: {len(X_test)}')

    return X_train, y_train, X_test, y_test


def main(random_state=100):
    X_train, y_train, X_test, y_test = load_flow_data(random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

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

            y_pred = clf.predict(_X_train[:, 1:])
            report = sklearn.metrics.classification_report(_y_train, y_pred)
            print(report)

            cm = confusion_matrix(_y_train, y_pred)
            print(cm)

            # eval on test set
            y_pred = clf.predict(_X_test[:, 1:])
            # y_pred = vote_pred(y_pred)
            report = sklearn.metrics.classification_report(_y_test, y_pred)
            print(report)

            cm = confusion_matrix(_y_test, y_pred)
            print(cm)
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
    y_pred = clf.predict(X_train[:, 1:])
    report = sklearn.metrics.classification_report(y_train, y_pred)
    print(report)
    cm = confusion_matrix(y_train, y_pred)
    print(cm)

    # eval on test set
    y_pred = clf.predict(X_test[:, 1:])

    for i, (_y_t, _y_p) in enumerate(zip(y_test, y_pred)):
        if _y_p == 1 and _y_t == 0:  # FP (normal is 0, abnormal is 1)
            print('FP:', X_test[i, 0])
        elif _y_p == 0 and _y_t == 1:  # FN
            print('FN:', X_test[i, 0])

    # y_pred = vote_pred(y_pred)
    report = sklearn.metrics.classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # rnn = RNN(n_epochs=100, in_dim=1460, out_dim=10, n_layers=1, lr=1e-3, bias=False, random_state=random_state)
    #
    # rnn.train(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, split=False)
    #
    # rnn.test(X_test=X_test, y_test=y_test, split=True)


if __name__ == '__main__':
    main(random_state=RANDOM_STATE)
