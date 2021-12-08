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
        for e in itertools.product([0, 1], repeat=10):
            self.inputs.append(e)

            #print(type(e[0]))
            #break
        self.labels = list('0'*20 + '1'*10 + '0'*60 + '1'*40 + '0'*10 + '1'*5 + '0'*20 + '1'*10\
                            + '0'*60 + '1'*30 + '0'*10 + '1'*10 + '0'*30 + '1'*20 + '0'*80 + '1'*50\
                            + '0'*30 + '1'*30 + '0'*60 + '1'*70 + '0'*10 + '1'*20 + '0'*80 + '1'*20\
                            + '0'*120 + '1'*30 + '0'*20 + '1'*45 + '0'*24)
        print(len(self.labels))
        #self.labels = list('0'*210 + '1'*130 + '0'*60 + '1'*74 + '0'*160 + '1'*220 + '0'*80 + '1'*90)  1024桁
        #self.labels = list('0'*13 + '1'*3 + '0'*36 + '1'*41 + '0'*8 + '1'*2 + '0'*25)  126桁
        #self.labels = [str(random.randint(0, 1)) for i in range(2**7)]

        print(customized_lz(self.labels, False))
#<<<<<<< HEAD
#        self.labels = torch.Tensor([int(i) for i in self.labels]).to(self.device)
#=======
        self.labels = torch.Tensor([int(i) for i in self.labels]).float()
#>>>>>>> c6ca09e19dc522282240c92a6b83b1a9ba2bac56
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
#<<<<<<< HEAD
#        sample = [torch.Tensor(input).to(self.device), label]
#=======
        sample = [torch.Tensor(input).float(), label]
#>>>>>>> c6ca09e19dc522282240c92a6b83b1a9ba2bac56
        return sample