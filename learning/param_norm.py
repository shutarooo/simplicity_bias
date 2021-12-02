import torch
import sys

sys.path.append('../')
from model import DeepNeuralNetwork

import json
import matplotlib.pyplot as plt

import glob
import re

norm_list = [0]*10
norm_log_list = [0]*10
epoch_list = [0]*10

files = glob.glob("./model/dnn/*.json")
for file in files:
    print(re.findall('_.*j', file)[0][1:-2])
    index = re.findall('_.*j', file)[0][1:-2]
    with open(file) as f:
        data = json.load(f)  # ファイルオブジェクトfinをデコードする
        epoch_list[int(index) - 1] = data['epochs']

files = glob.glob("./model/dnn/*.pth")
for file in files:
    print(file)
    index = re.findall('_.*p', file)[0][1:-2]
    print(index)
    model = torch.load(file)

    cnt = 0
    norm = 0
    norm_log = 0
    param_layer = 0

    for param_tensor in model.state_dict():
        print(param_tensor)
        cnt += 1
        
        if cnt % 2 == 1:
            param_layer = model.state_dict()[param_tensor]
            continue
        else:
            x = torch.cat((param_layer, model.state_dict()[param_tensor].unsqueeze(1)), 1)
            #print(x.size())
            norm_log += torch.log(torch.norm(x, p='fro'))
            norm += torch.norm(x, p='fro')
    sys.exit()
    norm_log_list[int(index) - 1] = norm_log
    norm_list[int(index) - 1] = norm

#print(norm_list)



fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(1, 2, 2)

ax1.scatter(epoch_list, norm_list)
plt.show()

