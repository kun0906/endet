"""Detect abnormal flows progressively using LSTM

run in the command under the examples directory
   PYTHONPATH=../:./ python3.7 examples/rnn_main_single.py > rnn_main_single.txt 2>&1 &
"""
# Author: kun.bj@outlook.com
# license: xxx
import os
import random

import numpy as np
import sklearn
import torch

from model.rnn import RNN
from pcap.parser import PCAP_PKTS, _pcap2flows
from util.tool import dump_data, load_data

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

    X = []
    for i, (fid, v_lst) in enumerate(raw_features):
        feat_0 = v_lst[0]
        feat_i_lst = v_lst[1:]

        if header:
            tmp_v = [v['header'] + v['payload'] for v in feat_i_lst]
            tmp_v = [v + [0] * (MTU - len(v)) if len(v) < MTU else v[:MTU] for v in tmp_v]
        else:
            payload_len = MTU - 40
            tmp_v = [v['payload'] for v in feat_i_lst]
            tmp_v = [v + [0] * (payload_len - len(v)) if len(v) < payload_len else v[:payload_len] for v in tmp_v]

        if normalize:
            tmp_v = normalize_bytes(tmp_v)

        X.append(tmp_v)

    return X


def load_flow_data(overwrite=False, random_state=100, full_flow=True):
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
    datasets = [
        # # # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',
        # 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
        # # # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
        # # # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
        # # # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
        # # # # # # #
        # # # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
        # # # # # # # #
        #'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
         'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
        # # # # #
        # # # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
        # # # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
        # # # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
        # # # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
        # # # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
        #
        # # # 'WRCCDC/2020-03-20',
        # # # 'DEFCON/ctf26',
        # 'ISTS/2015',
        # 'MACCDC/2012',
        #
        # # # # #
        # # # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
        # # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        # # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
        #
        'demo'
    ]

    dataset_name = datasets[0]
    print(f'dataset: {dataset_name}')
    in_dir = 'data/data_reprst/pcaps'
    if dataset_name == 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196':
        in_norm_file = f'{in_dir}/{dataset_name}/2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap'
    elif dataset_name == 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15':
        in_norm_file = f'{in_dir}/{dataset_name}/2017-05-02_CTU_Normal_32-src_10.0.2.15_normal.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap'
    elif dataset_name == 'ISTS/2015':
        in_norm_file = f'{in_dir}/{dataset_name}/snort.log-merged-3pcaps.pcap'
        # in_norm_file = f'{in_dir}/{dataset_name}/snort.log-merged-srcIP_10.2.4.30.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/snort.log-merged-srcIP_10.2.4.30.pcap'
    elif dataset_name == 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42':
        in_norm_file = f'{in_dir}/{dataset_name}/samsung_camera-2daysactiv-src_192.168.143.42-normal.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/samsung_camera-2daysactiv-src_192.168.143.42-anomaly.pcap'
    else:
        in_norm_file = 'data/lstm/demo_normal.pcap'
        in_abnorm_file = 'data/lstm/demo_abnormal.pcap'

    out_norm_file = in_norm_file + '-raw_normal_features.dat'
    out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'

    if overwrite or not os.path.exists(out_norm_file) or not os.path.exists(out_abnorm_file):
        # note: this work uses full flows, not subflows
        norm_pp = PCAP_PKTS(pcap_file=in_norm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
        if full_flow:
            flows = _pcap2flows(norm_pp.pcap_file, norm_pp.flow_ptks_thres,
                                verbose=norm_pp.verbose)
            norm_pp.flows = flows
        else:
            norm_pp.pcap2flows()

        norm_pp.flows2bytes()
        out_norm_file = in_norm_file + '-raw_normal_features.dat'
        dump_data(norm_pp.features, out_norm_file)

        abnorm_pp = PCAP_PKTS(pcap_file=in_abnorm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
        if full_flow:
            flows = _pcap2flows(abnorm_pp.pcap_file, abnorm_pp.flow_ptks_thres,
                                verbose=abnorm_pp.verbose)
            abnorm_pp.flows = flows
        else:
            abnorm_pp.pcap2flows(interval=norm_pp.interval)
        abnorm_pp.flows2bytes()

        out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'
        dump_data(abnorm_pp.features, out_abnorm_file)

    X_norm = raw2features(load_data(out_norm_file), header=False)
    y_norm = [0] * len(X_norm)
    X_abnorm = raw2features(load_data(out_abnorm_file), header=False)
    y_abnorm = [1] * len(X_abnorm)

    return split_train_test(X_norm, y_norm, X_abnorm, y_abnorm, random_state)


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
    random.Random(random_state).shuffle(X_norm)  #注意此处打乱数据的作用
    size = int(len(y_norm) // 2) if len(y_norm) <= len(y_abnorm) else min(400, len(y_abnorm))
    X_test = X_norm[-size:] + X_abnorm[:size]
    y_test = y_norm[-size:] + y_abnorm[:size]
    X_train = X_norm[:-size]
    y_train = y_norm[:-size]
    print(f'X_train: {len(X_train)}, X_test: {len(X_test)}')

    return X_train, y_train, X_test, y_test


def main(random_state=100):
    X_train, y_train, X_test, y_test = load_flow_data(random_state=random_state)

    rnn = RNN(n_epochs=100, in_dim=1460, out_dim=10, n_layers=1, lr=1e-3, bias=False, random_state=random_state)

    rnn.train(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, split=True)

    rnn.test(X_test=X_test, y_test=y_test, split=True)


if __name__ == '__main__':
    main(random_state=RANDOM_STATE)
