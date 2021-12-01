
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


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    #print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model_dnn = model.DeepNeuralNetwork().to(device)

train_size = 112
val_size = 16
test_size = 0

dataset = CustomImageDataset(device)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size = 8)
val_dataloader = DataLoader(val_dataset, batch_size = 1)
test_dataloader = DataLoader(test_dataset, batch_size = 8)

learning_rate = 1e-2


loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model_dnn.parameters(), lr=learning_rate)

epochs = 0
correct = 0
while correct < 0.9:
    #print(f"Epoch {epochs+1}-------------------------------")
    train_loop(train_dataloader, model_dnn, loss_fn, optimizer)
    correct = test_loop(test_dataloader, model_dnn, loss_fn)
    epochs += 1
print("Done!")
print("epoch: {}".format(epochs))

torch.save(model_dnn, 'model/dnn/model.pth')
path_file = 'model/dnn/info.json'
with open(path_file, 'w') as f:
    json.dump({'target_function': dataset.labels.tolist(), 'train_size': 112, 'val_size': 16, 'threshold': 0.9} , f)
