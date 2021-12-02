from model import DeepNeuralNetwork
from functions import calc_freq, save_dic
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = DeepNeuralNetwork().to(device)
dic = calc_freq(6, device, model)

dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])
save_dic(dic, '/data/func_freq_norm/norm_13')

