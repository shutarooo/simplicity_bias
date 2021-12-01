
from dataset import CustomImageDataset
import torch
from torch.utils.data import DataLoader
from torch import nn
import json
import sys
sys.path.append('../')
import model

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        
        # 予測と損失の計算
        #print(X)
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
            #print(pred)
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
            correct += (torch.round(pred) == y.unsqueeze(1)).type(torch.float).sum().item()
    


            
    #print(f"correct num: {correct}")
    #sys.exit()
    test_loss /= size
    correct /= size
    if is_print:
        print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #sys.exit()
    return correct

train_size = 112
val_size = 16
test_size = 0

dataset = CustomImageDataset()

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size = 8)
val_dataloader = DataLoader(val_dataset, batch_size = 8)
test_dataloader = DataLoader(test_dataset, batch_size = 8)


learning_rate = 1e-3

model_dnn = model.DeepNeuralNetwork()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model_dnn.parameters(), lr=learning_rate)

epochs = 0
correct = 0
while correct < 0.9:

    is_print = False
    if epochs % 1000 == 0:
        print(f"Epoch {epochs+1}--------------")
        is_print = True
    
    train_loop(train_dataloader, model_dnn, loss_fn, optimizer)
    correct = test_loop(val_dataloader, model_dnn, loss_fn, is_print)
    epochs += 1
    
        
print("Done!")

torch.save(model_dnn, 'model/dnn/model_10.pth')
path_file = 'model/dnn/info_10.json'
with open(path_file, 'w') as f:
    json.dump({'target_function': dataset.labels.tolist(), 'epochs': epochs, 'train_size': 112, 'val_size': 16, 'threshold': 0.9} , f)
