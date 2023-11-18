from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
# Téléchargement des données

from datamaestro import prepare_dataset


#  TODO: 

class MonDataset(Dataset):
    def __init__(self,datax,lab):
        self.datax = datax/datax.max()
        self.labels = lab
    def __getitem__(self,index):
        return self.datax[index], self.labels[index]
    def __len__(self):
        return len(self.datax)



# for x,y in data:
#     #print(x,y)
#     print(x[0].shape)



def test(dataloader, model, loss_fn,flatfn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            X = flatfn(X).float()
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # BCELoss
            #pred = (pred > 0.5).long()
            #correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Auto-Encodeur

class autoencodeur(torch.nn.Module):
    def __init__(self,input_len,output_len, batch_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.DoubleTensor(input_len, output_len))
        self.bias1 = torch.nn.Parameter(torch.DoubleTensor(output_len))
        self.bias2 = torch.nn.Parameter(torch.DoubleTensor(input_len))
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.lin1 = nn.Linear(input_len,output_len,dtype=torch.double)
        # On partage les poids pour que ce soit symétrique
        
        # self.encode = torch.nn.Sequential(
        #     lin1,
        #     torch.nn.ReLU())
        # self.decode = torch.nn.Sequential(
        #     lin2,
        #     torch.nn.Sigmoid())
    
    def forward(self,x):
        #x = torch.nn.functional.linear(x.float(),self.W.t(),self.bias1)
        x = self.lin1(x)
        #x = torch.nn.functional.relu(x)
        x = self.relu(x)

        x = torch.nn.functional.linear(x, self.lin1.weight.t(), self.bias2)
        #x = torch.nn.functional.linear(x.float(),self.W,self.bias2)
        #x = torch.sigmoid(x)
        x = self.sig(x)
        return x

# Highway Network
class HighLayer(torch.nn.Module):
    def __init__(self,input_len):
        super().__init__()
        self.t = torch.nn.Linear(input_len,input_len)
        self.relu = torch.nn.ReLU()
        self.h = torch.nn.Linear(input_len,input_len)
        self.sig = torch.nn.Sigmoid()
    
    def forward(self,x):
        T = self.sig(self.t(x))
        H = self.sig(self.h(x))
        return H*T + x*(1-T)


class HighNetwork(torch.nn.Module):
    def __init__(self,input_len,midlen,labels_size):
        super().__init__()
        self.lin = torch.nn.Linear(input_len,midlen)
        self.relu = torch.nn.ReLU()
        self.high = HighLayer(midlen)
        self.lin2 = torch.nn.Linear(midlen,labels_size)
    
    def forward(self,x):

        return self.lin2(self.high(self.relu(self.lin(x))))



class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0



ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()


# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


BATCH_SIZE = 100
train_loader = DataLoader(MonDataset(train_images,train_labels), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(MonDataset(test_images,test_labels))
nb_lab = len(np.unique(test_labels))

# for x,y in data:
#     #print(x,y)
#     print(x[0].shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# savepath = Path("model.pch")
# in_size = 28*28
# out_size = 512

# autoenc = autoencodeur(in_size,out_size,BATCH_SIZE)
# autoenc = autoenc.to(device)

# if savepath.is_file():
#     with savepath.open("rb") as fp:
#         state = torch.load(fp)  # Resume from the saved model
# else:
#     model = autoenc
#     model = model.to(device)
#     optim = torch.optim.SGD(model.parameters(), lr=1e-2)
#     state = State(model, optim)

# loss = torch.nn.MSELoss()
# flat = torch.nn.Flatten()
# for epoch in range(state.epoch, 100):
#     print("Epoch: ", epoch)
#     tmpLoss= []
#     for x, y in train_loader:
#         x = x.double()
#         state.optim.zero_grad()
#         x = flat(x)
#         x = x.to(device)
#         xhat = state.model(x)
#         l = loss(xhat, x)
#         l.backward()
#         state.optim.step()
#         state.iteration += 1
#         tmpLoss.append(l.detach().item())
#     print("Loss : ",np.mean(tmpLoss))

#     with savepath.open("wb") as fp:
#         state.epoch = epoch + 1
#         torch.save(statmodele, fp)


savepath = Path("modelHigh.pch")
in_size = 28*28
midsize = 512
# Highway Test
network = HighNetwork(in_size,midsize,nb_lab)


if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)  # Resume from the saved model
else:
    model = network
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    state = State(model, optim)

loss = torch.nn.CrossEntropyLoss()
flat = torch.nn.Flatten()
for epoch in range(state.epoch, 100):
    print("Epoch: ", epoch)
    tmpLoss= []
    for x, y in train_loader:
        x = x.float()
        y = y.to(device)
        state.optim.zero_grad()
        x = flat(x)
        x = x.to(device)
        xhat = state.model(x)
        l = loss(xhat, y)
        l.backward()
        state.optim.step()
        state.iteration += 1
        tmpLoss.append(l.detach().item())
    print("Loss : ",np.mean(tmpLoss))
    #test(test_loader,model,loss,flat,device)

    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)

test(test_loader,model,loss,flat,device)