from model import NeuralNetwork, DeepNeuralNetwork
import torch
from torch import nn
import math
from functions import calc_freq, get_params, save_dic
from initializer import weight_init_unit_uniform, weight_init_wide_uniform
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help='the architecture of the model: NN or DNN')
parser.add_argument("digit", type=int, help='the number of sample: 10**digit')
parser.add_argument("initializer", type=str, help='the type of initializer: unit or wide')
parser.add_argument("path", type=str, help='relative path for json file: ex. data/.../unit_uniform')
args = parser.parse_args() 

print(args.test)





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NeuralNetwork().to(device) if args.model == 'NN' else DeepNeuralNetwork().to(device) if args.model == 'DNN' else ''
digit = args.digit
initializer = weight_init_unit_uniform if args.initializer == 'unit' else weight_init_wide_uniform if args.initializer == 'wide' else ''
path = args.path

<<<<<<< HEAD
dic = calc_freq(digit, device, model, initializer)
=======
dic = calc_freq(6, device, model, initializer.weight_init_unit_uniform)
>>>>>>> 5eca0853d5af97d5e57ad46b20f3854ea97c7362
dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])
save_dic(dic, path)
