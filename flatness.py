from torch import nn
import torch
from model import DeepNeuralNetwork_2
import sys
sys.path.append('../')


def calc_flatness(model, dataset, loss_fn, rho, p, device):
    # 勾配を計算する
    loss = 0
    for x, y in dataset:
        pred = model(x)
        loss += loss_fn(pred, y.unsqueeze(0))
    loss.backward()

    # (2)の分母のノルムを計算する
    param = torch.zeros(0).to(device)
    for m in model.linear_relu_stack:
        if type(m) == nn.Linear:
            # print(param.size())
            # print(nn.utils.parameters_to_vector(m.weight).size())
            param = torch.cat(
                (param, nn.utils.parameters_to_vector(m.weight)), 0)
            # print(param.size())
            if m.bias is not None:
                param = torch.cat(
                    (param, nn.utils.parameters_to_vector(m.bias)), 0)
                # print(param.size())
                # print()
    norm = torch.norm(param, p=p)
    #print(f"norm: {norm.item()}")

    # epsを計算する
    q = p/(p-1)
    eps_list = []
    for m in model.linear_relu_stack:
        if type(m) == nn.Linear:
            eps_list.append(rho *
                            torch.sign(m.weight.grad) *
                            torch.abs(m.weight.grad) ** (q-1) / norm ** (q/p))
            if m.bias is not None:
                eps_list.append(rho *
                                torch.sign(m.bias.grad) *
                                torch.abs(m.bias.grad) ** (q-1) / norm ** (q/p))
    '''
    for m in model.linear_relu_stack:
        if m == nn.Linear:
            eps_list.append(rho *
                       torch.sign(m.weight.grad) *
                       torch.abs(m.weight.grad) ** (q-1) / norm ** (q/p))
            if m.bias is not None:
                eps_list.append(rho *
                        torch.sign(m.bias.grad) *
                        torch.abs(m.bias.grad) ** (q-1) / norm ** (q/p))
    '''

    # epsを用いて、元のモデルのparameterからepsだけずれたモデルを構築する
    model_for_flat = DeepNeuralNetwork_2().to(device)

    eps_idx = 0
    for idx in range(len(model.linear_relu_stack)):
        m = model.linear_relu_stack[idx]
        m_flat = model_for_flat.linear_relu_stack[idx]
        if type(m) == nn.Linear and type(m_flat) == nn.Linear:
            m_flat.weight = torch.nn.Parameter((m.weight + eps_list[eps_idx]))
            eps_idx += 1
            if m.bias is not None:
                m_flat.bias = torch.nn.Parameter(m.bias + eps_list[eps_idx])
                eps_idx += 1

    loss_eps = 0
    for x, y in dataset:
        pred = model_for_flat(x)
        loss_eps += loss_fn(pred, y.unsqueeze(0))

    sharpness = (loss_eps/824 - loss/824)/(1+loss/824) * 100
    #print(f"sharpness: {sharpness}")
    return (1/sharpness).item()
