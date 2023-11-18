from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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
savepath = Path("model.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        rnn = torch.load(fp)  # Resume from the saved model


with torch.no_grad():
    for x, y in data_test:
        x = x.to(device)
        y = y.to(device)
        
        x = torch.swapaxes(x, 0, 1)
        y = torch.swapaxes(y, 0, 1)
        size = x.shape
        x = x.reshape(size[0],size[1]*size[2], size[3])
        y = y.reshape(x.shape)

        index = torch.randint(len(x[0]), (1,))
        x = x[:,index,:]
        y = y[:,index,:]
        t = 0
        y_pred = [rnn.decode(res) for res in rnn(x)]    
        
        # PLOT
        fig, axs = plt.subplots(2)
        y_pred = torch.tensor(np.array(y_pred))

        print("x: ", x.shape)
        print("y: ", y.shape)
        print("pred: ", y_pred.shape)
        for i in [0, 1]:
            a = len(x)
            b = len(y)
            
            axs[i].plot(torch.arange(1,a+1), y[:,0,i], label='y')
            axs[i].plot(torch.arange(1,a+1), y_pred[:,0,i], label='pred')
            axs[i].legend()
        plt.show()
        break