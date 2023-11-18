import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from pathlib import Path

from utils import RNN, device
import numpy as np

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

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize 
        full_text = normalize(text) # clean text
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        # 0, ..., 0, t1, ..., tn (pour avoir la même taille sur tous les items)
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]
    
#  TODO: 
PATH = "data/"
batch_size = 100

data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

hidden = 100
DIM_INPUT = len(id2lettre.keys())
rnn = RNN(DIM_INPUT, hidden, DIM_INPUT, activation=torch.nn.ReLU)

loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

savepath = Path("modelTrump.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        rnn = torch.load(fp)  # Resume from the saved model

rnn.to(device)

print("NUMBER OF BATCH: ", len(data_trump))
for epoch in range(20):
    for i, (x,y) in enumerate(data_trump):
        optimizer.zero_grad()
        x = x.to(device) # batch, lettre
        y = y.to(device) # batch, lettre
        j = 0
        while torch.all(y[:,j]==0):
            j+=1
        x = x[:,j:]
        y = y[:,j:]
        x = nn.functional.one_hot(x, num_classes=DIM_INPUT).float()
        x = torch.swapaxes(x, 0, 1)
        lm = [loss(rnn.decode(res), y[:,j]) for j,res in enumerate(rnn(x))]
        l = sum(lm)/len(lm)
        l.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch : ", epoch, " Iteration : ", i, " Loss : ", round(l.item(),4))
    with torch.no_grad():
        c = 0
        for i, (x,y) in enumerate(data_trump):
            j = 0
            while torch.all(x[:,j] == 0):
                j+=1
            x = x[:,j:].to(device) # batch, lettre
            y = y[:,j:].to(device) # batch, lettre
            x = nn.functional.one_hot(x, num_classes=DIM_INPUT).float()
            x = torch.swapaxes(x, 0, 1)
            for j in range(x.size(1)):
                rnn(x[:,1,:][y])
            s = [rnn.decode(res).argmax(1)[0].item() for res in rnn(x)]
            s = code2string(s)
            y = code2string(y[0])
            print("Reel: ==================\n", y)
            print("Pred: ==================\n", s)
            print()
            if c >= 5: break
            c+=1
    with Path("modelTrump.pch").open("wb") as fp:
        torch.save(rnn, fp)
