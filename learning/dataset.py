import os

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import itertools
import random

import sys
sys.path.append('../')
from lz_compression import customized_lz

class CustomImageDataset(Dataset):
    def __init__(self, device, transform=None, target_transform=None):
        self.inputs = []
        self.device = device
        for e in itertools.product([0, 1], repeat=7):
            self.inputs.append(e)

            #print(type(e[0]))
            #break
        self.labels = list('0'*21 + '1'*13 + '0'*18 + '1'*24 + '0'*39 + '1'*13)
        #self.labels = [str(random.randint(0, 1)) for i in range(2**7)]

        print(customized_lz(self.labels, False))
<<<<<<< HEAD
        self.labels = torch.Tensor([int(i) for i in self.labels]).to(self.device)
=======
        self.labels = torch.Tensor([int(i) for i in self.labels]).float()
>>>>>>> c6ca09e19dc522282240c92a6b83b1a9ba2bac56
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input = self.inputs[idx]
        if self.transform:
            image = self.transform(input)
        if self.target_transform:
            label = self.target_transform(label)
        #print(input)
        #sample = {"input": torch.Tensor(input), "label": label}
<<<<<<< HEAD
        sample = [torch.Tensor(input).to(self.device), label]
=======
        sample = [torch.Tensor(input).float(), label]
>>>>>>> c6ca09e19dc522282240c92a6b83b1a9ba2bac56
        return sample