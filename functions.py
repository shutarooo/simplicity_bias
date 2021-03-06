import torch
from torch import nn
import math
import itertools
import json
import os
import sys
import matplotlib.pyplot as plt

from initializer import weight_init_gauss, weight_normalize


def calc_norm(model):
    cnt = 0
    norm = 0
    #norm_log = 0
    param_layer = 0

    for tens in model.linear_relu_stack:
        if isinstance(tens, torch.nn.Linear):
            # print(tens)
            x = torch.norm(tens.weight, p='fro')
            # print(x)
            norm += x.cpu().detach().numpy()
            #norm_log += torch.log(x).detach().numpy()
    '''
    for param_tensor in model.state_dict():
        cnt += 1

        #print(param_tensor)
        
        if cnt % 2 == 1:
            param_layer = model.state_dict()[param_tensor]
            #print(param_layer.shape)
            continue
        else:
            #print(model.state_dict()[param_tensor].shape)
            x = torch.cat((param_layer, model.state_dict()[param_tensor].unsqueeze(1)), 1)
         
            norm += torch.norm(x, p='fro')
    '''
    return norm


def norm_initialize(model):

    norm = calc_norm(model)
    # print(norm)
    # print(norm)
    # print(model.linear_relu_stack[0].bias)

    # print(model)
    for i in range(len(model.linear_relu_stack)):
        if isinstance(model.linear_relu_stack[i], nn.Linear):
            model.linear_relu_stack[i].weight = nn.Parameter(
                model.linear_relu_stack[i].weight * (13.5/norm))
            if i == 4:
                break
            model.linear_relu_stack[i].bias = nn.Parameter(
                model.linear_relu_stack[i].bias * (13.5/norm))

    '''
    for param_tensor in model.state_dict():
        
        model.linear_relu_stack = nn.Parameter(model.state_dict()[param_tensor] * (9.0/norm))
    
    cnt = 0
    norm = 0
    #norm_log = 0
    param_layer = 0

    for param_tensor in model.state_dict():
        cnt += 1
        print(param_tensor)
        
        if cnt % 2 == 1:
            param_layer = model.state_dict()[param_tensor]

            print(param_layer.shape)
            continue
        else:
            print(model.state_dict()[param_tensor].shape)
            #x = torch.cat((param_layer, model.state_dict()[param_tensor].unsqueeze(1)), 1)
         
            #norm += torch.norm(x, p='fro')
    #return norm
    '''

    # print(calc_norm(model))
    # print()


# ?????????????????????????????????????????????????????????????????????????????????????????????
def calc_freq(digit, device, model, initializer='norm'):
    function_freq = {}
    binary_input = []
    for e in itertools.product([0, 1], repeat=10):
        binary_input.append(e)
    tensor_input = torch.tensor(binary_input, device=device)
    tensor_input = tensor_input.float()

    # ????????????????????????NN??????????????????function_freq????????????????????????
    for k in range(10**digit):
        # ????????????????????????NN????????????
        if initializer == 'norm':
            model.apply(weight_init_gauss)
            norm_initialize(model)
            # print(model.linear_relu_stack[4].bias)
            # print(torch.mean(torch.abs(model.linear_relu_stack[4].weight)))

        elif initializer == 'self':
            nn.init.normal_(model.linear_relu_stack[0].weight, 0, 1)
            nn.init.normal_(model.linear_relu_stack[0].bias, 0, 1)
            nn.init.normal_(model.linear_relu_stack[2].weight, 0, 1)
            nn.init.normal_(model.linear_relu_stack[2].bias, 0, 1)
            nn.init.normal_(model.linear_relu_stack[4].weight, 0, 1)
            norm_initialize(model)

        else:
            model.apply(initializer)
            #print('norm: {}'.format(calc_norm(model)))

        # ????????????????????????????????????????????????NN??????????????????????????????out_seq???????????????
        out_seq = []
        for input in tensor_input:
            output = torch.heaviside(
                model(input)-0.5, torch.tensor([0.0], device=device))
            out_seq.append(output.tolist())

        # ??????????????????????????????function_freq???????????????????????????
        split_array = []
        x = ''
        last_index = len(out_seq) - 1
        for i, out in enumerate(out_seq):
            x += str(int(out[0]))
            if (i+1) % 60 == 0:
                split_array.append(x)
                x = ''
            elif i == last_index:
                split_array.append(x)

        keys = []
        for s in split_array:
            keys.append(int('0b'+s, 2))

        '''
        print('out: {} {} {}'.format(split_array[0], split_array[1], split_array[2]))
        print('key: {} {} {}'.format(keys[0], keys[1], keys[2]))
        print('')
        '''

        # print(keys)
        if tuple(keys) in function_freq:
            function_freq[tuple(keys)] += 1
        else:
            function_freq[tuple(keys)] = 1

        if k % 10**(digit-1) == 0:
            print('{}% finished.'.format(k/10**(digit-1)*10))

    return function_freq

# parameter????????????????????????????????????parameters???list???????????????


def get_params(model):
    params = []
    for param in model.parameters():
        dim = param.dim()
        param_list = torch.Tensor.tolist(param)

        for row in param_list:
            if dim == 2:
                for p in row:
                    params.append(p)
            if dim == 1:
                params.append(row)

    print(len(params))
    return params

# json???????????????


def save_dic(dic, path):
    path_file = os.getcwd() + path + '.json'
    with open(path_file, 'w') as f:
        json.dump(dic, f)


bias = 0


def print_ouput(module, input, output):
    '''
    plt.hist(input.detach().numpy())
    plt.title('input')
    plt.show()
    '''

    global bias
    bias = module.bias.detach().numpy()
    '''
    plt.hist(module.bias.detach().numpy())
    plt.title('bias')
    plt.show()
    #plt.savefig('./images/no_bias/output.pdf')
    '''


def print_input(module, input, output):
    # print(input)
    '''
    plt.hist(input[0].detach().numpy())
    plt.title('input')
    plt.show()
    '''

    print('input: {}'.format(input[0].item() - bias))

    print('bias: {}'.format(bias))

    '''
    plt.hist(input[0].detach().numpy() - bias)
    plt.title('input')
    plt.show()
    '''
