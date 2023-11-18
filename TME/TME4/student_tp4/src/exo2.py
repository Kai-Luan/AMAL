from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
hidden = 100
rnn = RNN(DIM_INPUT, hidden, CLASSES)
rnn.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

for epoch in range(50):
    for i, (x, y) in enumerate(data_train):
        x = x.to(device)
        x = torch.permute(x, (1, 0, 2))
        y = y.to(device)
        optimizer.zero_grad()
        h = torch.zeros(x.size(1), hidden)
        y_pred = rnn(x,h)
        y_pred = rnn.decode(y_pred[-1])
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch : ", epoch, " Iteration : ", i, " Loss : ", loss.item())
    with torch.no_grad():
        correct = 0
        total = 0
        l = 0
        for x, y in data_test:
            x = x.to(device)
            y = y.to(device)
            x = torch.permute(x, (1, 0, 2))
            h = torch.zeros(x.size(1), hidden)
            y_pred = rnn(x,h)
            y_pred = rnn.decode(y_pred[-1])
            l += criterion(y_pred, y)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print("Accuracy : ", round(correct/total, 3), round(l.item()/len(data_test), 3))


