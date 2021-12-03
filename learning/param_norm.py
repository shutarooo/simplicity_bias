import torch
import sys

sys.path.append('../')
from model import DeepNeuralNetwork

import json
import matplotlib.pyplot as plt

import glob
import re

def print_ouput(module, input, output):
    print(module)
    '''
    plt.hist(output.detach().numpy())
    plt.title('output')
    plt.show()
    '''
    plt.hist(module.weight.detach().numpy())
    plt.title('bias')
    plt.show()

norm_list = [0]*10
norm_log_list = [0]*10
epoch_list = [0]*10

bias_0 = []
bias_2 = []
bias_4 = []

weight_0 = []
weight_2 = []
weight_4 = []

files = glob.glob("./model3/dnn/*.json")
for file in files:
    print(re.findall('_.*j', file)[0][1:-2])
    index = re.findall('_.*j', file)[0][1:-2]
    #print(index)
    with open(file) as f:
        data = json.load(f)  # ファイルオブジェクトfinをデコードする
        epoch_list[int(index) - 1] = data['epochs']

files = glob.glob("./model3/dnn/*.pth")
for file in files:
    print(file)
    index = re.findall('_.*p', file)[0][1:-2]
    print(index)
    model = torch.load(file)
    #model.linear_relu_stack[2].register_forward_hook(print_ouput)
    y = model(torch.tensor([1,0,0,1,1,1,0]).float())

    cnt = 0
    norm = 0
    norm_log = 0
    param_layer = 0

    bias_0.append(torch.mean(torch.abs(model.linear_relu_stack[0].bias)).item())
    bias_2.append(torch.mean(torch.abs(model.linear_relu_stack[2].bias)).item())
    #bias_4.append(torch.mean(torch.abs(model.linear_relu_stack[4].bias)).item())

    weight_0.append(torch.mean(torch.abs(model.linear_relu_stack[0].weight)).item())
    weight_2.append(torch.mean(torch.abs(model.linear_relu_stack[2].weight)).item())
    weight_4.append(torch.mean(torch.abs(model.linear_relu_stack[4].weight)).item())

    for tens in model.linear_relu_stack:
        if isinstance(tens, torch.nn.Linear):
            print(tens)
            x = torch.norm(tens.weight, p='fro')
            #print(x)
            norm += x.detach().numpy()
            norm_log += torch.log(x).detach().numpy()
    
    norm_log_list[int(index) - 1] = norm_log
    norm_list[int(index) - 1] = norm

    '''
    for param_tensor in model.state_dict():
        print(param_tensor)
        cnt += 1
        
        if cnt % 2 == 1:
            param_layer = model.state_dict()[param_tensor]
            continue
        else:
            x = torch.cat((param_layer, model.state_dict()[param_tensor].unsqueeze(1)), 1)
            norm_log += torch.log(torch.norm(x, p='fro'))
            norm += torch.norm(x, p='fro')

    norm_log_list[int(index) - 1] = norm_log
    norm_list[int(index) - 1] = norm
    '''

#print(norm_list)
print(f"bias_0: {bias_0}\n")
print(f"bias_2: {bias_2}\n")
#print(f"bias_4: {bias_4}\n")

print(f"weight_0: {weight_0}\n")
print(f"weight_2: {weight_2}\n")
print(f"weight_4: {weight_4}\n")


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(1, 2, 2)

ax1.scatter(epoch_list, norm_list)
plt.show()


