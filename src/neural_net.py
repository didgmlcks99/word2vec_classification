import random

import torch
import torch.nn as nn

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.linear1 = torch.nn.Linear(300, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 2)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x