import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(110, )
        self.rnn2 = nn.RNN()
        self.fc1 = nn.Linear(None, 50)
        self.fc2 = nn.Linear(50, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


