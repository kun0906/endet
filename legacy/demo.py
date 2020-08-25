from collections import Counter
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def plot_data(x, y, xlabel='range', ylabel='auc', title=''):
    # with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.plot(x, y, '*', alpha=0.9)
    # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0., 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks(x)
    # plt.yticks(y)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def IP_info(input_file):
    csvs = pd.read_csv(input_file)
    # print(csvs.info())
    # print(csvs.values)
    # print(Counter(list([v.item() for v in csvs.values])))
    ips = list([v.item() for v in csvs.values])

    return ips


def IP_stat():
    white_ips = 'gp/data/white_ip.csv'
    black_ips = 'gp/data/black_ip.csv'
    w_ips = IP_info(white_ips)
    b_ips = IP_info(black_ips)

    x = [int(v.split('.')[-2]) for v in w_ips]
    y = [int(v.split('.')[-1]) for v in w_ips]
    plot_data(x, y, title='white')

    x = [int(v.split('.')[-2]) for v in b_ips]
    y = [int(v.split('.')[-1]) for v in b_ips]
    plot_data(x, y, title='black')

    test_ips = 'gp/data/test_ip.csv'
    t_ips = IP_info(test_ips)

    x = [int(v.split('.')[-2]) for v in t_ips]
    y = [int(v.split('.')[-1]) for v in t_ips]
    plot_data(x, y, title='test')
    print(Counter(w_ips + b_ips), len(w_ips + b_ips))


def feature_info(input_file):
    csvs = pd.read_csv(input_file)
    # features = list([v.item() for v in csvs.values])
    features = csvs.values
    # print(csvs.info())
    print(csvs.describe())

    return features


def feature_stat():
    white = 'gp/data/white-1.csv'
    black = 'gp/data/black-1.csv'
    w_features = feature_info(white)
    b_features = feature_info(black)

    # pd.w

    # print(Counter(w_features + b_features))


if __name__ == '__main__':
    IP_stat()
    # feature_stat()
