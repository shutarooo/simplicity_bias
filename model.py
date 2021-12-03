import os
import torch
from torch import nn
#from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import math
import itertools

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x):
        #x = self.flatten(x)
        #logits = torch.heaviside(self.linear_relu_stack(x), torch.tensor([0.0], device=device))
        return self.linear_relu_stack(x)


class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        #x = self.flatten(x)
        #logits = torch.heaviside(self.linear_relu_stack(x), torch.tensor([0.0], device=device))
        return self.linear_relu_stack(x)