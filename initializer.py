from torch import nn
import torch
import math

def weight_init_gauss(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight, 0, 1)
        if m.weight is None:
            print('none')
            return
        nn.init.normal_(m.bias, 0, 1)

def weight_normalize(m, norm, const):
    if type(m)==nn.Linear:
        m.weight = m.weight * (const/norm)

def weight_init_uniform(m):
    if type(m)==nn.Linear:
        nn.init.uniform_(m.weight, -1/math.sqrt(7), 1/math.sqrt(7))

def weight_init_unit_uniform(m):
    if type(m)==nn.Linear:
        nn.init.uniform_(m.weight, -1, 1)

def weight_init_wide_uniform(m):
    if type(m)==nn.Linear:
        while True:
            nn.init.uniform_(m.weight, -10, 10)
            if torch.norm(m.weight, p='fro') > 1:
                break

def weight_init_unit_uniform(m):
    if type(m)==nn.Linear:
        nn.init.uniform_(m.weight, -1, 1)
        nn.init.uniform_(m.bias, -1, 1)
