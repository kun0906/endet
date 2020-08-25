"""

"""
# Author: kun.bj@outlook.com
# license:
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from lstm.model.rnn import GRUModel, LSTMModel

RANDOM_STATE = 100
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def set_random_state():
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)


set_random_state()


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

        self.criterion = nn.CrossEntropyLoss()
        self.random_state = random_state

    def train(self, X_train, y_train=None, X_val=None, y_val=None):
        train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        self.model = LSTMModel(self.in_dim, self.hid_dim, self.n_layers, self.out_dim)
        # self.model = GRUModel(self.in_dim, self.hid_dim, self.n_layers, self.out_dim)

        #######################
        #  USE GPU FOR MODEL  #
        #######################

        if torch.cuda.is_available():
            self.model.cuda()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        """
         STEP 7: TRAIN THE MODEL
        """
        loss_list = []

        for epoch in range(self.n_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Load images as Variable
                #######################
                #  USE GPU FOR MODEL  #
                #######################

                if torch.cuda.is_available():
                    images = Variable(images.view(images.shape[0], -1, self.in_dim).cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images.view(images.shape[0], -1, self.in_dim))
                    labels = Variable(labels)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                # outputs.size() --> 100, 10
                outputs = self.model(images)

                # Calculate Loss: softmax --> cross entropy loss
                loss = self.criterion(outputs, labels)

                if torch.cuda.is_available():
                    loss.cuda()

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                loss_list.append(loss.item())

            if epoch % 2 == 0:
                self.test(X_test=X_val)
                print('opoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), self.test_accuracy))

                self.model.train()  # updating model

        self.train_losses = loss_list

    def test(self, X_test, y_test=None):

        # set model in eval mode.
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(dataset=X_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        # Calculate Accuracy
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            if torch.cuda.is_available():
                images = Variable(images.view(images.shape[0], -1, self.in_dim).cuda())
            else:
                images = Variable(images.view(images.shape[0], -1, self.in_dim))

            # Forward pass only to get logits/output
            outputs = self.model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

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

        self.test_accuracy = 100 * correct / total


def load_data():
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


def main(random_state=100):
    train_dataset, test_dataset = load_data()

    rnn = RNN(n_epochs=5, out_dim=10, n_layers=1, random_state=random_state)
    rnn.train(X_train=train_dataset, X_val=test_dataset)
    rnn.test(X_test=test_dataset, y_test=None)


if __name__ == '__main__':
    main(random_state=RANDOM_STATE)
