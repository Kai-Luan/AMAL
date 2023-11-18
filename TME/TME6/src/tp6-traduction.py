import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List

import time
import re
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm




logging.basicConfig(level=logging.INFO)
FILE = "data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=86

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
class translateModel(nn.Module):
    def __init__(self, 
    encoder_args,
    decoder_args,
    hidden_size,
    stop_word,
    device=None):
        super().__init__()
        vocal_size, embedding_size = encoder_args
        self.emb_from = nn.Embedding(vocal_size, embedding_size)
        self.encoder = nn.GRU(embedding_size, hidden_size)

        vocal_size, embedding_size = decoder_args
        self.emb_to = nn.Embedding(vocal_size, embedding_size)
        self.decoder = nn.GRU(embedding_size, hidden_size)
        
        self.w_out = nn.Linear(hidden_size, vocal_size)
        self.stop_word = stop_word
        self.device = device
    
    def forward(self, x, h=None):
        return self.encoder(self.emb_from(x), h)[1]
    
    def forward_decode(self, x, h = None):
        return self.decoder(self.emb_to(x), h)[0]
    
    def generate(self, hidden, lenseq=None):
        x = SOS[:,:hidden.shape[1]]
        out = []
        for _ in range(lenseq):
            x, hidden = self.decoder(self.emb_to(x), hidden)
            x = self.w_out(x)
            out.append(x)
            x = x.argmax(2)
        return torch.vstack(out)

    def decode(self, x):
        return self.w_out(x.to(device))

SOS = torch.full((1,MAX_LEN), Vocabulary.SOS)
SOS = SOS.to(device)        

def evaluate(loader, model, loss, num_classes, title='Test Set', p = 1):
    with torch.no_grad():
        A = tm.classification.MulticlassAccuracy(num_classes=num_classes, ignore_index=0, topk = 1)
        A.to(device)
        test_loss = 0
        for x, _, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            h = model(x)
            yhat = None
            if torch.rand(1) < p:
                x = torch.vstack((SOS[:,:y.shape[1]], y[:-1]))
                #print("Decoder: ", x.shape)
                yhat = model.forward_decode(x, h)
                yhat = model.decode(yhat)
            else: yhat = model.generate(h,y.shape[0])
            yhat = yhat.swapaxes(1,2)
            test_loss += loss(yhat, y).item()
            A(yhat, y)

        test_loss = round(test_loss / len(test_loader), 3)
        acc = round(A.compute().item(), 3)
        print(f'{title}: Accuracy: {acc} | Loss: {test_loss}')

def run_train(loader, model, loss, optimizer, p=0.9):
    for x, _, y, _ in tqdm(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        h = model(x)
        yhat = None
        if torch.rand(1) < p:
            x = torch.vstack((SOS[:,:y.shape[1]], y[:-1]))
            #print("Decoder: ", x.shape)
            yhat = model.forward_decode(x, h)
            yhat = model.decode(yhat)
        else: yhat = model.generate(h, y.shape[0])
        yhat = yhat.swapaxes(1,2)
        #print(yhat.shape, y.shape)
        #exit()
        l = loss(yhat, y)
        l.backward()
        optimizer.step()
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("NUMBER OF BATCH: ", len(train_loader))
    max_epoch = 50
    # Model params
    num_classes = len(vocFra)
    rnn = translateModel(
        (len(vocEng),100),
        (len(vocFra),100),
        hidden_size= 200,
        stop_word= Vocabulary.SOS
    )
    rnn.to(device)
    loss = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

    print(f'FR: {len(vocFra)}, ENG: {len(vocEng)}')
    for epoch in range(max_epoch):
        print("Epoch: ", epoch)
        p = 1
        run_train(train_loader, rnn, loss, optimizer, p)
        evaluate(train_loader, rnn, loss, num_classes, title= 'Train Set',p=p)
        evaluate(test_loader, rnn, loss, num_classes, title='Test Set', p=p)
        print()
        