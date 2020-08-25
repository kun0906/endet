"""RNN includes LSTM and GRU

"""
# Author: kun.bj@outlook.com
# license: xxx

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchsummary
from sklearn import metrics
from sklearn.metrics import roc_curve
from torch.autograd import Variable

from util.plot import plot_data

RANDOM_STATE = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Flatten(nn.Module):
    # def __init__(self):
    #     self._out_dim = 1
    def forward(self, input):
        # self._out_dim = input.view(input.size(0), -1)[1]
        return input.view(input.size(0), -1)


class ODCNN(nn.Module):  # one dimension CNN
    def __init__(self, in_dim, out_dim, n_layers, kernel_size=5, stride=3, bias=False):
        """Network architecture

        Parameters
        ----------
        img_shape
        sl
        """

        super(ODCNN, self).__init__()

        # in_dim = int(np.prod(self.img_shape))
        # create input layer
        self.encoder = nn.Sequential()
        # self.encoder.add_module('linear_in', nn_pdf.Linear(in_dim, h_dim))
        out_channels = 128  # cnn output chanenls after input layer
        input_layer = torch.nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      bias=bias)
        _out_dim = (in_dim - (kernel_size - stride)) // stride
        self.encoder.add_module(f'input_layer', input_layer)
        self.encoder.add_module('leakyrelu_in', nn.LeakyReLU())
        # self.encoder.add_module('elu_in', nn.ELU())
        # self.encoder.add_module('dropout_in', nn.Dropout())

        # create hidden layers
        in_channels = out_channels
        # out_channels = int(in_channels //2)
        out_channels = 32
        for i in range(n_layers):
            # self.encoder.add_module(f'linear_{i + 1}', nn_pdf.Linear(h_dim, h_dim))
            # kernel_size = i + 1
            layer_i = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
            _out_dim = (_out_dim - (kernel_size - stride)) // stride
            self.encoder.add_module(f'conv_{i + 1}', layer_i)
            self.encoder.add_module(f'leakyrelu_{i + 1}', nn.LeakyReLU())
            # self.encoder.add_module(f'elu_{i+1}', nn.ELU())
            # self.encoder.add_module(nn_pdf.Tanh()),
            # self.encoder.add_module(nn_pdf.Sigmoid()),
            # self.encoder.add_module(nn_pdf.Dropout())
            in_channels = int(out_channels)
            # out_channels = int(in_channels // 2)
            # print(out_channels)

        # create ouput layer
        # out = out.view(x.shape[0], out.size(1) * out.size(2))
        self.encoder.add_module('flatten', Flatten())
        # _out_dim = list(self.encoder)[-1]._out_dim
        _out_dim = _out_dim * out_channels
        self.encoder.add_module('linear_full_1', nn.Linear(_out_dim, out_dim, bias=bias))

    def forward(self, xs):
        """ each flow (here is 'xs') will have multi-packets, in which each packet is a 'x'

        Parameters
        ----------
        xs: list of packets ('x')

            each flow (here is 'xs') will have multi-packets, in which each packet is a 'x'
        Returns
        -------
        out_feats:
            list of extracted features
        """

        out_feats = []
        for x in xs:
            # input x : 23 x 59049 x 1
            # expected conv1d input : minibatch_size x num_channel x width
            # img_flat = img.view(img.size(0), -1)
            x = x.view(1, 1, -1)

            _out = self.encoder(x)  # _out.shape: (hid, 100)
            # bound = 50  # torch.exp(torch.tensor(5e+1))= tensor(5.1847e+21)
            # encoded_feats = torch.tensor(1.0 / np.sqrt(2 * np.pi)) * torch.exp(
            #     -(encoded_feats ** 2 / 2).clamp(-bound, bound))  # normal distribution pdf with mu=0, std=1
            out_feats.append(_out)

        return out_feats


class LSTMModel(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, out_dim, cnn_flg=True, bias=False):
        """LSTM model, not a lstm cell

        Parameters
        ----------
        in_dim: int
            The number of features of the input data.

        hid_dim: int
            The output size of a lstmcell.

        n_layers: int
            The number of layers of lstmcell stacked together.

        out_dim: int
            The LSTM output dimension.

        """
        super(LSTMModel, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.cnn_flg = cnn_flg

        if self.cnn_flg:
            # # CNN
            # convonluation layers is 1 + n_layers (2)
            out_dim_cnn = 100
            self.odcnn = ODCNN(in_dim=in_dim, out_dim=out_dim_cnn, n_layers=2, kernel_size=5, stride=3, bias=bias)
            print(f'{torchsummary.summary(self.odcnn, input_size=(in_dim,))}')
            # LSTM

            self.lstm = nn.LSTM(input_size=out_dim_cnn, hidden_size=hid_dim,
                                num_layers=n_layers, batch_first=True, bias=bias)
        else:
            # LSTM
            self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid_dim,
                                num_layers=n_layers, batch_first=True, bias=bias)

        # two fully connection layers
        hid_dim2 = 10  # the output size of the first fully connection layer
        self.fc1 = nn.Linear(hid_dim, hid_dim2, bias=bias)
        self.fc2 = nn.Linear(hid_dim2, out_dim, bias=bias)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """One forward calculation

        Parameters
        ----------
        x: flows with variable lengths

        Returns
        -------
            the output of LSTMModel
        """
        outs = []
        # Propagate input through LSTM
        for i, _x in enumerate(X):
            # Due to the variable length of each flow, so each time we just input one flow (here is '_x').

            if self.cnn_flg:
                # 1DCNN
                # print(f'i: {i}, _x.shape: {_x.shape}')
                _x = self.odcnn(_x)  # 1D CNN extracts features from packets, (list of packets)
                _x = torch.cat(_x)

            # LSTM
            # Initialize hidden state with zeros
            if torch.cuda.is_available():
                # (n_layers, batch_size, hid_dim)
                h0 = Variable(torch.zeros(self.n_layers, 1, self.hid_dim).cuda())
            else:
                h0 = Variable(torch.zeros(self.n_layers, 1, self.hid_dim))

            # Initialize cell state
            if torch.cuda.is_available():
                # (n_layers, batch_size, hid_dim)
                c0 = Variable(torch.zeros(self.n_layers, 1, self.hid_dim).cuda())
            else:
                c0 = Variable(torch.zeros(self.n_layers, 1, self.hid_dim))

            # (batch_size, seq_length, in_dim)
            _x = _x.view(1, _x.shape[0], _x.shape[1])
            outputs, (h_out, c_out) = self.lstm(_x, (h0, c0))

            # only use the top h_out of the last hidden layer, i.e., h_out[-1]
            out = h_out[-1].view(-1, self.hid_dim)

            # the fully connection layers
            # out.size() --> 100, 10
            out = self.fc1(out)
            out = self.leakyrelu(out)

            # out.size() --> 10, 10
            out = self.fc2(out)
            out = self.leakyrelu(out)

            outs.append(out)

        # print(torch.cat(outs).shape, len(outs))
        return torch.cat(outs)


def split_instance(X, y):
    """Split each flow into subflow

    Parameters
    ----------
    X: list
        One flow
    y: int
        label

    Returns
    -------
    new_Xs: list
        subflows
    new_ys: list
        labels
    """
    new_Xs = []
    new_ys = []
    # one flow will split into "[pkt0], [pkt0, pkt1], [pkt0, pkt1, pkt2] , ..."
    v = []
    for _x in X:
        v.append(_x)
        new_Xs.append(v)
        new_ys.append(y)

    return new_Xs, new_ys


class RNN:
    def __init__(self, n_epochs=10, batch_size=64, in_dim=28, hid_dim=128,
                 n_layers=1, out_dim=10, lr=1e-3, bias=False, random_state=100, verbose=10):

        """Interface of all RNNs

        Parameters
        ----------
        n_epochs: int

        batch_size: int

        in_dim: int

        hid_dim: int

        n_layers: int

        out_dim: int

        lr: float
            learning rate

        random_state: int

        verbose: int
            print level

        """

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
        self.out_dim = out_dim
        self.bias = bias
        self.lr = lr

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.random_state = random_state
        self.verbose = verbose

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def init_center_c(self, train_loader, net, eps=1e-3):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0

        c = torch.zeros(self.out_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(get_batch_data(train_loader, self.batch_size)):
                # print(f'i: {i}, len(labels): {len(labels)}')
                images = [Tensor(v) for v in images]

                # # Clear gradients w.r.t. parameters
                # optimizer.zero_grad()
                outputs = self.model(images)
                # print(i, outputs)
                # inputs, _, _ = data
                # inputs = inputs.to(self.device)
                # outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        print(f'centers: {c}')
        return c

    def train(self, X_train, y_train=None, X_val=None, y_val=None, split=True):
        """Fit the lstm model

        Parameters
        ----------
        X_train
        y_train
        X_val
        y_val

        split: boolean
            Split each flow into subflows if True.

        Returns
        -------
            self

        """
        if split:
            # split each train instance
            new_X_train = []
            new_y_train = []
            for x, y in zip(X_train, y_train):
                new_xs, new_ys = split_instance(x, y)
                new_X_train.extend(new_xs)
                new_y_train.extend(new_ys)

            X_train = new_X_train
            y_train = new_y_train
            print(f'X_train: {len(X_train)}')

        self.model = LSTMModel(self.in_dim, self.hid_dim, self.n_layers, self.out_dim, bias=self.bias)
        # self.model = GRUModel(self.in_dim, self.hid_dim, self.n_layers, self.out_dim)

        # use GPU
        if torch.cuda.is_available():
            self.model.cuda()

        optimizer_name = 'Adam'
        if optimizer_name == 'Adam':
            b1 = 0.5
            b2 = 0.999
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(b1, b2))
        else:  # default is SGD
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # fit the model
        train_losses = []
        train_set = list(zip(X_train, y_train))

        # get centers
        self.centers = self.init_center_c(train_set, self.model)

        for epoch in range(self.n_epochs):
            # shuffle the train set in each epoch
            train_loader = sklearn.utils.shuffle(train_set, random_state=epoch)
            epoch_loss = []  # each epoch loss
            for i, (images, labels) in enumerate(get_batch_data(train_loader, self.batch_size)):
                # print(f'i: {i}, len(labels): {len(labels)}')

                images = [Tensor(v) for v in images]
                # labels = [labels]
                # labels = Tensor(labels).view(-1, 1)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                outputs = self.model(images)

                # # Calculate Loss
                # if i == 0 and epoch == 0:
                #     # Use the average of the first forwarded outputs as centers
                #     self.centers = torch.mean(outputs, dim=0).detach()
                loss = self.criterion(outputs, self.centers.repeat(repeats=(len(images), 1)))
                loss.to(device=self.device)

                # soft-boundary # not implemented yet
                # loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                epoch_loss.append(loss.item())

            train_losses.append(sum(epoch_loss) / (i + 1))

            # Evaluate the model on val set
            if epoch % 2 == 0:
                self.R, dists = self.get_normal_thres(X_train, q=0.9)
                self.val_test(X_val, y_val)
                print('epoch: {}, epoch_loss: {}, val_acc: {}, val_auc: {}, R: {}'.format(epoch,
                                                                                          sum(epoch_loss),
                                                                                          self.val_acc,
                                                                                          self.val_auc,
                                                                                          self.R))

                # turn on the model training mode again.
                self.model.train()

        # get the normal threshold from train set
        self.R, self.normal_dists = self.get_normal_thres(X_train, q=0.9)
        print(f'normal_thres: {np.quantile(self.normal_dists, q=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95])} '
              f'when q=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95]')
        print(f'R (normal_thres): {self.R} when q=0.9')
        self.norm_qs_lst = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.norm_thre_lst = np.quantile(self.normal_dists, q=self.norm_qs_lst)

    def get_normal_thres(self, X_train, q=0.9):
        """Get the normal threshold from train set

        Parameters
        ----------
        X_train:
            The train set
        q: float
            quantile on a range (0, 1)

        Returns
        -------
        R: float
            The normal threshold
        dists: arr
            All distances between instances and centers
        """
        # switch the model mode to "eval" mode
        self.model.eval()
        outputs = self.model([Tensor(v) for v in X_train])
        if torch.cuda.is_available():
            dists = torch.sum((outputs.cpu() - self.centers.cpu()) ** 2, dim=1).data.numpy()
        else:
            dists = torch.sum((outputs - self.centers) ** 2, dim=1).data.numpy()
        R = np.quantile(dists, q)

        return R, dists

    def val_test(self, X_val=None, y_val=None, split=True):
        """Valilate instance one by one

        Parameters
        ----------
        X_val:

        y_val:

        Returns
        -------

        """
        # set model in the "eval" mode.
        self.model.eval()

        correct = 0
        total = 0
        y_score = []
        # Iterate through test dataset one by one
        val_loader = zip(X_val, y_val)
        for i, (images, labels) in enumerate(get_batch_data(val_loader, self.batch_size, last_batch=True)):
            # print(i, len(images))
            images = [Tensor(v) for v in images]
            labels = Tensor(labels)

            # Forward pass only to get output
            outputs = self.model(images)

            # the distance between outputs and centers
            if torch.cuda.is_available():
                dists = torch.sum((outputs.cpu() - self.centers.cpu()) ** 2, dim=1).data.numpy()
            else:
                dists = torch.sum((outputs - self.centers) ** 2,
                                  dim=1).data.numpy()  # self.centers obtained from training phase
            y_score.extend(dists)

            # # Total number of labels
            # total += labels.size(0)

            # Total correct predictions
            # predicted = R[R > self.normal_thres ]
            predicted = Tensor([1 if v > self.R else 0 for v in dists])
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
        # get accuracy
        self.val_acc = correct.item() / len(y_val)

        # get AUC
        fpr, tpr, _ = roc_curve(y_val, y_score, pos_label=1)
        self.val_auc = metrics.auc(fpr, tpr)

    def score_function(self, dist_x):
        """ abnormal score confidence

        Parameters
        ----------
        dist_x

        Returns
        -------

        """
        for i in range(len(self.norm_thre_lst)):
            if i == 0 and dist_x < self.norm_thre_lst[0]:
                score = dist_x
                # normal confidence (1-self.norm_qs_list[i] - self.norm_qs_lst[0]) * 100
                confidence = 0  # # abnormal confidence
                break
            elif self.norm_thre_lst[i - 1] <= dist_x < self.norm_thre_lst[i]:
                find = True
                score = dist_x
                # normal confidence (1-self.norm_qs_list[i] - self.norm_qs_lst[0]) * 100
                confidence = (self.norm_qs_list[i] - self.norm_qs_lst[0]) * 100  # abnormal confidence
                break

        if not find:
            upper = 2 * self.norm_thre_lst[i]
            if self.norm_thre_lst[i] <= dist_x < upper:
                score = dist_x
                confidence = 90
            else:
                score = dist_x
                confidence = 100

        return score, confidence

    def test(self, X_test=None, y_test=None, split=True):
        """Online testing instance one by one

        Parameters
        ----------
        X_test:

        y_test:

        split: True
            split each instance into subflows

        Returns
        -------

        """
        # set model in the "eval: mode.
        self.model.eval()

        test_loader = zip(X_test, y_test)

        # Calculate Accuracy
        correct = 0
        total = 0
        y_score = []
        outputs = []
        # Iterate through test dataset one by one
        for i, (X, y) in enumerate(test_loader):

            if split:
                new_Xs, new_ys = split_instance(X, y)
                res = []
                # testing subflow with adding more packets
                for i, (_x, _y) in enumerate(zip(new_Xs, new_ys)):
                    # print(i, len(X))
                    _x = Tensor(_x)
                    _y = Tensor([_y])
                    if torch.cuda.is_available():
                        _x = Variable(_x.view(1, _x.shape[0], self.in_dim).cuda())
                    else:
                        _x = Variable(_x.view(1, _x.shape[0], self.in_dim))

                    # Forward pass only to get logits/output
                    if torch.cuda.is_available():
                        v = torch.sum((self.model(_x).cpu() - self.centers.cpu()) ** 2, dim=1).data.numpy()
                    else:
                        v = torch.sum((self.model(_x) - self.centers) ** 2, dim=1).data.numpy()
                    res.append(v.item())  # [subflow1_res, subflow2_res,...] (for each flow)
                outputs.append(res)

        results = []
        R = np.quantile(self.normal_dists, q=0.9)
        succeeded_flows = []
        failed_flows = []

        for vs in outputs:
            find_flg = False
            _scores = []
            for j, v in enumerate(vs):  # res := [subflow1_res, subflow2_res,...] (for each flow)
                _score, _confidence = self.score_function(v)
                _scores.append([_score, _confidence])
                if v > R:
                    find_flg = True
                    break

            res = {'needed_ptks': j + 1, 'R': R, 'prob': v, 'find_flg': find_flg, 'time': 0, '_scores': _score}
            results.append(res)

            if res['find_flg']:
                succeeded_flows.append(res['needed_ptks'])
            else:
                failed_flows.append(res['needed_ptks'])

        if len(succeeded_flows) > 0:
            print(f'average_needed_pkts: {np.mean(succeeded_flows)} +/- {np.std(succeeded_flows)} '
                  f'when R: {R}, succeeded_flows: {len(succeeded_flows)}, all_flows: {len(y_test)}')
        if len(failed_flows) > 0:
            print(f'failed_flows: {np.mean(failed_flows)} +/- {np.std(failed_flows)} when R: {R}, '
                  f'failed_flows: {len(failed_flows)}, all_flows: {len(y_test)}')

        # plot the figure of num of packets and aucs
        min_flow_len = min([len(vs) for vs in outputs])
        print(f'***min_flow_len: {min_flow_len}')
        aucs = []
        accs = []
        part_results = np.asarray([v[:min_flow_len] for v in outputs])
        for i in range(min_flow_len):
            fpr, tpr, _ = roc_curve(y_test, part_results[:, i], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)

            y_pred = [1 if v > self.R else 0 for v in part_results[:, i]]
            acc = metrics.accuracy_score(y_test, y_pred)
            accs.append(acc)

        plot_data(range(1, min_flow_len + 1), aucs, xlabel='num of packets in each flow', ylabel='auc', title='')
        plot_data(range(1, min_flow_len + 1), accs, xlabel='num of packets in each flow', ylabel='accuracy', title='')

        print(f'aucs: {aucs}')
        print(f'accs: {accs}')


def get_batch_data(data_tuple, size=32, last_batch=True):
    """Get each batch instances (each instance has variable length)

    Parameters
    ----------
    data_tuple: tuple
        (X, y)

    size: int
        The number of instances in each batch

    last_batch: boolean
        It return the last batch instances if True, otherwise, just drop the last one.

    Returns
    -------
        A batch of instances

    """
    X = []
    Y = []
    cnt = 0
    for i, (x, y) in enumerate(data_tuple):
        X.append(x)
        Y.append(y)
        cnt += 1
        if cnt >= size:
            yield X, Y
            X = []
            Y = []
            cnt = 0

    if last_batch and len(X) > 0:
        yield X, Y
