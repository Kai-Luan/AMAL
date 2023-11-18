from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
from pathlib import Path
import string
import unicodedata
import torch
import torch.nn as nn
from pathlib import Path



## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

#  TODO:  Question 3 : Prédiction de séries temporelles
hidden = 100
DIM_INPUT = len(id2lettre.keys())
rnn = RNN(DIM_INPUT, hidden, DIM_INPUT)
rnn.to(device)
savepath = Path("modelTrump.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        rnn = torch.load(fp)  # Resume from the saved model

length = 1000
with torch.no_grad():
    s = []
    # lettre vide
    x = torch.zeros((1,1,DIM_INPUT))
    x[0,0,10] = 1
    h = None    
    for _ in range(length):
        h = rnn(x, h)[-1]
        index = rnn.decode(h).argmax(1)[0]
        x = torch.zeros((1,1,DIM_INPUT))
        x[0,0,index] = 1
        s.append(index.item())
    s = code2string(s)
    print(s)

