import torch.nn as nn
import torch.nn.functional as F

# Building model
class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden , hidden)
        self.l3 = nn.Linear(hidden, output)

        # self.linear1 = torch.nn.Linear(input, hidden)
        # self.activation = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(hidden, output)
        # self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)

        return out

        # x = self.linear1(x)
        # x = self.activation(x)
        # x = self.linear2(x)
        # x = self.softmax(x)

        # return x