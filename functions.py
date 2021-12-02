import torch
from torch import nn
import math
import itertools
import json
import os
import sys

from initializer import weight_init_gauss, weight_normalize

def calc_norm(model):
    cnt = 0
    norm = 0
    #norm_log = 0
    param_layer = 0

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
    return norm

def norm_initialize(model):
    model.apply(weight_init_gauss)
    norm = calc_norm(model)
    #print(norm)
    #print(model.linear_relu_stack[0].bias)

    #print(model)
    for i in range(len(model.linear_relu_stack)):
        if isinstance(model.linear_relu_stack[i], nn.Linear):
            model.linear_relu_stack[i].weight = nn.Parameter(model.linear_relu_stack[i].weight * (13.0/norm))
            model.linear_relu_stack[i].bias = nn.Parameter(model.linear_relu_stack[i].bias * (13.0/norm))

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


    #print(calc_norm(model))
    #print()


# サンプルサイズの桁数を受け取って関数が現れた頻度を辞書型で返す
def calc_freq(digit, device, model, initializer='norm'):
    function_freq = {}
    binary_input = []
    for e in itertools.product([0, 1], repeat=7):
        binary_input.append(e)
    tensor_input = torch.tensor(binary_input, device=device)
    tensor_input = tensor_input.float()

    # ランダム初期化→NNの関数推定→function_freqに記録を繰り返す
    for k in range(10**digit):
        # ランダムな重みでNNを初期化
        if initializer == 'norm':
            norm_initialize(model)
        
        else:
            model.apply(initializer)

        # 全ての入力に対して出力を計算し、NNが表現している関数をout_seqで取得する
        out_seq = []
        for input in tensor_input:
            output = torch.heaviside(model(input), torch.tensor([0.0], device=device))
            out_seq.append(output.tolist())
        
        # 省メモリであるようにfunction_freqを定義して記録する
        split_array = []
        x = ''
        for i, out in enumerate(out_seq):
            x += str(int(out[0]))
            if (i+1)%60==0:
                split_array.append(x)
                x = ''
            elif i==127:
                split_array.append(x)
        
        keys = []
        for s in split_array:
            keys.append(int('0b'+s, 2))

        '''
        print('out: {} {} {}'.format(split_array[0], split_array[1], split_array[2]))
        print('key: {} {} {}'.format(keys[0], keys[1], keys[2]))
        print('')
        '''

        
        
        #print(keys)
        if tuple(keys) in function_freq:
            function_freq[tuple(keys)] += 1
        else:
            function_freq[tuple(keys)] = 1
        
        if k % 10**(digit-1) == 0:
            print('{}% finished.'.format(k/10**(digit-1)*10))
    
    return function_freq

# parameterの分布を可視化するためにparametersをlistにして返す
def get_params(model):
    params = []
    for param in model.parameters():
        dim=param.dim()
        param_list = torch.Tensor.tolist(param)
        
        for row in param_list:
            if dim==2:
                for p in row:
                    params.append(p)
            if dim==1:
                params.append(row)

    print(len(params))
    return params

# json形式で保存
def save_dic(dic, path):
    path_file = os.getcwd() + path +'.json'
    with open(path_file, 'w') as f:
        json.dump(dic , f)