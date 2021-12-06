from model import DeepNeuralNetwork
from functions import calc_freq, save_dic, print_ouput, print_input
from initializer import weight_init_unit_uniform
import torch
import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = DeepNeuralNetwork().to(device)
#print(model.linear_relu_stack)
#sys.exit()
#model.linear_relu_stack[4].register_forward_hook(print_ouput)
#model.linear_relu_stack[5].register_forward_hook(print_input)

dic = calc_freq(6, device, model, initializer='self')

dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])
save_dic(dic, '/data/func_freq_norm/norm_135')

