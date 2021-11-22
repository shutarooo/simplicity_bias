import torch
from torch import nn
import math
import itertools
import json
import os


# サンプルサイズの桁数を受け取って関数が現れた頻度を辞書型で返す
def calc_freq(digit, device, model, initializer):
    function_freq = {}
    binary_input = []
    for e in itertools.product([0, 1], repeat=7):
        binary_input.append(e)
    tensor_input = torch.tensor(binary_input, device=device)
    tensor_input = tensor_input.float()

    # ランダム初期化→NNの関数推定→function_freqに記録を繰り返す
    for k in range(10**digit):
        # ランダムな重みでNNを初期化
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