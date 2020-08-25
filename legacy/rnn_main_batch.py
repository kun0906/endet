"""

"""
# Author: kun.bj@outlook.com
# license: xxx
import os
import random

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.metrics import roc_curve
from torch.autograd import Variable
from torch.utils.data import TensorDataset

from lstm.model.rnn import GRUModel, LSTMModel, PaddedTensorDataset
from pcap.parser import PCAP, PCAP_PKTS
from util.tool import dump_data, load_data

RANDOM_STATE = 100
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def set_random_state():
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)


set_random_state()


#
# def generate_batch_instances(X, y):
#
#
#     for :
#         yield
#

class RNN:

    def __init__(self, n_epochs=10, batch_size=64, in_dim=28, hid_dim=128,
                 n_layers=1, out_dim=10, lr=0.1, random_state=100):

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
        self.out_dim = out_dim

        self.lr = lr

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.random_state = random_state

    def train(self, X_train, y_train=None, X_val=None, y_val=None):

        # train_loader = torch.utils.data.DataLoader(dataset=(torch.Tensor(X_train), torch.Tensor(y_train)),
        #                                            batch_size=self.batch_size,
        #                                            shuffle=True)
        # train_loader = zip(X_train, y_train)
        # torch.nn.utils.rnn.pad_sequence(l, batch_first=True, padding_value=0)

        # In order to use batch instances to train the model, we must make the batch instances has the same length.
        # However, it doesn't require for testing.
        X_train = torch.nn.utils.rnn.pad_sequence([Tensor(v) for v in X_train], batch_first=True, padding_value=0)
        y_train = Tensor(y_train).view(-1, 1)
        train_set = TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        self.model = LSTMModel(self.in_dim, self.hid_dim, self.n_layers, self.out_dim)
        # self.model = GRUModel(self.in_dim, self.hid_dim, self.n_layers, self.out_dim)

        #######################
        #  USE GPU FOR MODEL  #
        #######################

        if torch.cuda.is_available():
            self.model.cuda()

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        # Optimizers
        b1 = 0.5
        b2 = 0.999
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(b1, b2))
        """
         STEP 7: TRAIN THE MODEL
        """
        loss_list = []

        for epoch in range(self.n_epochs):

            epoch_loss = []
            for i, (images, labels) in enumerate(train_loader):
                # print(f'i: {i}')
                # Load images as Variable
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                # labels = [labels]
                # images = Tensor(images)
                # labels = Tensor(labels)

                # if torch.cuda.is_available():
                #     images = Variable(images.view(1, images.shape[0], self.in_dim).cuda())
                #     labels = Variable(labels.cuda())
                # else:
                #     images = Variable(images.view(1, images.shape[0], self.in_dim))
                #     labels = Variable(labels)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                # outputs.size() --> 100, 10
                outputs = self.model(images)

                # Calculate Loss: softmax --> cross entropy loss
                # loss = self.criterion(outputs, labels)
                # loss = self.criterion(outputs, torch.LongTensor(labels.data.numpy()))
                # loss =self.criterion(outputs, labels.type(torch.LongTensor).view(-1,))
                loss = self.criterion(outputs, labels)
                if torch.cuda.is_available():
                    loss.cuda()

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                epoch_loss.append(loss.item())

            loss_list.append(sum(epoch_loss) / (i + 1))

            if epoch % 2 == 0:
                self.val_test(X_val, y_val)
                print('epoch: {}. epoch_loss: {}. val_acc: {}, val_auc: {}'.format(epoch, sum(epoch_loss), self.val_acc,
                                                                                   self.val_auc))
                self.test(X_val, y_val)
                print('epoch: {}. epoch_loss: {}. test_acc: {}, test_auc: {}'.format(epoch, sum(epoch_loss),
                                                                                     self.test_acc,
                                                                                     self.test_auc))

                self.model.train()  # updating model

        self.train_losses = loss_list

    def val_test(self, X_val, y_val=None):

        # set model in eval mode.
        self.model.eval()

        X_val = torch.nn.utils.rnn.pad_sequence([Tensor(v) for v in X_val], batch_first=True, padding_value=0)
        y_val = Tensor(y_val).view(-1, 1)
        val_set = TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)

        # Calculate Accuracy
        correct = 0
        total = 0
        y_score = []
        # Iterate through test dataset
        for i, (images, labels) in enumerate(val_loader):
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            # if torch.cuda.is_available():
            #     images = Variable(images.view(images.shape[0], -1, self.in_dim).cuda())
            # else:
            #     images = Variable(images.view(images.shape[0], -1, self.in_dim))

            # Forward pass only to get logits/output
            outputs = self.model(images)

            # Get predictions from the maximum value
            values, indics = torch.max(outputs.data, 1)
            predicted = Tensor([1 if v > 0.5 else 0 for v in values.data]).flatten()
            y_score.extend(values.data.numpy().flatten().tolist())

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels.flatten()).sum()

        self.val_acc = correct / total

        fpr, tpr, _ = roc_curve(y_val.data.numpy(), y_score, pos_label=1)
        self.val_auc = metrics.auc(fpr, tpr)
        # print(f'val_auc: {self.val_auc}')

    def test(self, X_test=None, y_test=None):
        """ test one by one

        Parameters
        ----------
        X_test
        y_test

        Returns
        -------

        """
        # set model in eval mode.
        self.model.eval()

        test_loader = zip(X_test, y_test)

        # Calculate Accuracy
        correct = 0
        total = 0
        y_score = []
        # Iterate through test dataset
        for i, (images, labels) in enumerate(test_loader):
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            labels = [labels]
            # print(i, len(images))
            images = Tensor(images)
            labels = Tensor(labels)
            if torch.cuda.is_available():
                images = Variable(images.view(1, images.shape[0], self.in_dim).cuda())
            else:
                images = Variable(images.view(1, images.shape[0], self.in_dim))

            # Forward pass only to get logits/output
            outputs = self.model(images)

            # Get predictions from the maximum value
            values, indexs = torch.max(outputs.data, 1)
            predicted = 1 if values.data > 0.5 else 0
            y_score.extend(values.data.numpy())

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        self.test_acc = correct / total

        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        self.test_auc = metrics.auc(fpr, tpr)
        # print(f'final test_auc: {self.test_auc}')


def load_img_data():
    """
        STEP 1: LOADING DATASET
    """
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    return train_dataset, test_dataset


def raw2features(raw_features):
    X = []
    for i, (fid, v_lst) in enumerate(raw_features.items()):
        feat_0 = v_lst[0]
        feat_i_lst = v_lst[1:]

        tmp_v = [v['header'] + v['payload'] for v in feat_i_lst]
        tmp_v = [v + [0] * (1500 - len(v)) if len(v) < 1500 else v[:1500] for v in tmp_v]

        X.append(tmp_v)

    return X


def load_flow_data(overwrite=False):
    in_norm_file = 'data/lstm/demo_normal.pcap'
    in_abnorm_file = 'data/lstm/demo_abnormal.pcap'
    out_norm_file = in_norm_file + '-raw_normal_features.dat'
    out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'

    if overwrite or not os.path.exists(out_norm_file) or not os.path.exists(out_abnorm_file):
        norm_pp = PCAP_PKTS(pcap_file=in_norm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
        norm_pp.pcap2flows()
        norm_pp.flows2bytes()

        out_norm_file = in_norm_file + '-raw_normal_features.dat'
        dump_data(norm_pp.features, out_norm_file)

        abnorm_pp = PCAP_PKTS(pcap_file=in_abnorm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
        abnorm_pp.pcap2flows(interval=norm_pp.interval)
        abnorm_pp.flows2bytes()

        out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'
        dump_data(abnorm_pp.features, out_abnorm_file)

    X_norm = raw2features(load_data(out_norm_file))
    y_norm = [0] * len(X_norm)
    X_abnorm = raw2features(load_data(out_abnorm_file))
    y_abnorm = [1] * len(X_abnorm)

    return X_norm, y_norm, X_abnorm, y_abnorm


def main(random_state=100):
    X_norm, y_norm, X_abnorm, y_abnorm = load_flow_data()
    random.Random(random_state).shuffle(X_norm)
    size = min(100, len(y_abnorm))
    X_test = X_norm[-size:] + X_abnorm[:size]
    y_test = y_norm[-size:] + y_abnorm[:size]
    X_train = X_norm[:-size]
    y_train = y_norm[:-size]
    print(f'X_train: {len(X_train)}, X_test: {len(X_test)}')

    rnn = RNN(n_epochs=500, in_dim=1500, out_dim=1, n_layers=1, random_state=random_state)
    rnn.train(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test)
    rnn.test(X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    main(random_state=RANDOM_STATE)
