from utils import RNN, device,  ForecastMetroDataset
from pathlib import Path
from torch.utils.data import  DataLoader
import torch

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

length = 10

#  TODO:  Question 3 : Prédiction de séries temporelles
hidden = 100
rnn = RNN(DIM_INPUT, hidden, DIM_INPUT)
rnn.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
for epoch in range(20):
    for i, (x, y) in enumerate(data_train):
        x = x.to(device)
        # Batch, Time, Station, {in, out}
        x = torch.swapaxes(x, 0, 1)
        y = torch.swapaxes(y, 0, 1)
        size = x.shape
        # print(x[0][0][0])
        # print(x[0][0][1])
        x = x.reshape(size[0],size[1]*size[2], size[3])
        # print(x[0][0])
        # print(x[0][1])
        y = y.reshape(x.shape)
        z = torch.zeros(x.shape)
        optimizer.zero_grad()
        t = 0
        losses = []
        for res in rnn(x):
            y_pred = rnn.decode(res)
            losses.append(loss_function(y_pred, y[t]))
            t+=1
        loss = sum(losses)/len(losses)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch : ", epoch, " Iteration : ", i, " Loss : ", loss.item())
    # Test Performance
    with torch.no_grad():
        correct = 0
        total = 0
        l = 0
        for x, y in data_test:
            x = x.to(device)
            y = y.to(device)
            x = torch.swapaxes(x, 0, 1)
            y = torch.swapaxes(y, 0, 1)
            size = x.shape
            x = x.reshape(size[0],size[1]*size[2], size[3])
            y = y.reshape(x.shape)
            z = torch.zeros(x.shape)
            t = 0
            losses = []
            for res in rnn(x):
                y_pred = rnn.decode(res)
                losses.append(loss_function(y_pred, y[t]))
                t+=1
            l = sum(losses)/len(losses)
        
        print("Loss (test): ", round(l.item()/len(data_test), 3))

with Path("model.pch").open("wb") as fp:
    torch.save(rnn, fp)