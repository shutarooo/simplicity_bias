from model import NeuralNetwork
import torch
from torch import nn
import math
from functions import calc_freq, get_params, save_dic
import initializer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NeuralNetwork().to(device)

dic = calc_freq(6, device, model, initializer.weight_init_wide_uniform)
dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])
save_dic(dic, 'wide_uniform')