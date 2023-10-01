import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model, self).__init__()

        self.name = 'MNIST_model'

        self.fc1 = nn.Linear(28*28*1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x