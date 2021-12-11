from model import DeepNeuralNetwork_2
import itertools
from learning.dataset import CustomImageDataset
from lz_compression import customized_lz
from functions import calc_norm
from initializer import init_10_input
from flatness import calc_flatness
import torch
from torch.utils.data import DataLoader
from torch import nn
import json
import sys


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 予測と損失の計算
        # print(X)
        pred = model(X)

        '''
        print(pred)
        print(pred.shape)
        print(y.shape)
        '''
        loss = loss_fn(pred, y.unsqueeze(1))

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        '''


def test_loop(dataloader, model, loss_fn, is_print):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # print(pred)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            '''
            print(pred)
            print(torch.round(pred))
            print(y)
            print((torch.round(pred) == y).type(torch.float).sum().item())
            #print(f"{t.sum().item()}")
            '''
            '''
            print(torch.round(pred))
            print(y)
            print((torch.round(pred) == y.unsqueeze(1)).type(torch.float).sum().item())
            '''
            correct += (torch.round(pred) == y.unsqueeze(1)
                        ).type(torch.float).sum().item()

    #print(f"correct num: {correct}")
    # sys.exit()
    test_loss /= size
    correct /= size
    #print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if is_print:
        print(
            f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # sys.exit()
    return test_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = DeepNeuralNetwork_2().to(device)
model.apply(init_10_input)

train_size = 824
val_size = 200
test_size = 0

dataset = CustomImageDataset(device)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=8)

learning_rate = 1e-2

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

binary_input = []
for e in itertools.product([0, 1], repeat=10):
    binary_input.append(e)
tensor_input = torch.tensor(binary_input, device=device)
tensor_input = tensor_input.float()

epochs = 1
norm_list = []
loss_list = []
diff_list = []
flatness_list = []
epoch_comp = []
last_loss = 0
is_overfitting = 0
overfit_max = 10
while True:

    is_print = False
    if epochs % 100 == 0:
        print(f"Epoch {epochs+1}--------------")
        is_print = True
        # print(epoch_comp)
        # sys.exit()

    train_loop(train_dataloader, model, loss_fn, optimizer)
    loss = test_loop(val_dataloader, model, loss_fn, is_print)
    loss_list.append(loss)
    norm_list.append(calc_norm(model))

    if epochs % 10 == 0:
        flatness_list.append(calc_flatness(
            model, train_dataset, loss_fn, 1e-3, 2, device))

    if last_loss < loss:
        if is_overfitting == 0:
            torch.save(model, 'learning/model_flat/model_1.pth')
        is_overfitting += 1
    else:
        is_overfitting = 0
    last_loss = loss

    if is_overfitting >= overfit_max:
        print(epochs)
        print(len(epoch_comp))
        epochs -= overfit_max
        epoch_comp = epoch_comp[:-(overfit_max-1)]
        norm_list = norm_list[:-overfit_max]
        diff_list = diff_list[:-(overfit_max-1)]
        loss_list = norm_list[:-overfit_max]
        break

    if epochs >= 3000:
        break

    # if epochs % 5 ==0:
    out_seq = []
    diff = 0
    for i in range(len(tensor_input)):
        output = torch.heaviside(
            model(tensor_input[i])-0.5, torch.tensor([0.0], device=device))
        output = str(int(output.tolist()[0]))
        if output != str(int(dataset.labels[i].item())):
            diff += 1
        out_seq.append(output)
    diff_list.append(diff)
    # print("".join(out_seq))
    #print(not ('0' in out_seq))
    # sys.exit()
    epoch_comp.append([epochs, customized_lz(out_seq, False)])
    epochs += 1


print("Done!")
print("epoch: {}".format(epochs))
print(len(epoch_comp))

path_file = 'learning/model_flat/info_1.json'
with open(path_file, 'w') as f:
    json.dump({'flatness': flatness_list,
               'diff': diff_list,
               'norm': norm_list,
               'comp_process': epoch_comp,
               'loss_process': loss_list,
               'target_function': dataset.labels.tolist(),
               'epochs': epochs,
               'train_size': 112,
               'val_size': 16,
               'threshold': 0.9}, f)
