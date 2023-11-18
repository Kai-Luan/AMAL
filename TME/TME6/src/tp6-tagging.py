import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
import torchmetrics as tm

from pathlib import Path

logging.basicConfig(level=logging.INFO)
ds = prepare_dataset('org.universaldependencies.french.gsd')


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)
class State:
    def __init__(self, model, encoder, optim):Vocabulary

class seq2seq(nn.Module):
    def __init__(self, vocal_size, embedding_size, hidden_size, output_size):
        super().__init__()
        self.emb = nn.Embedding(vocal_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size)
        self.w_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.encoder( self.emb(x))
        return self.w_out(out)


def evaluate(loader, model, loss, title='Test Set'):
    with torch.no_grad():
        A = tm.classification.MulticlassAccuracy(num_classes=19, ignore_index=0)
        A.to(device)
        test_loss = 0
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            yhat = rnn(x).swapaxes(1,2)
            test_loss += loss(yhat, y).item()
            A(yhat, y)

        test_loss = round(test_loss / len(test_loader), 3)
        acc = round(A.compute().item(), 3)
        print(f'{title}: Accuracy: {acc} | Loss: {test_loss}')

def train(loader, model, loss, optimizer):
    p = 0.2
    for (x,y) in tqdm(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        # Randomly add OOV to x
        index = torch.bernoulli(torch.ones(x.shape)*p)>0.5
        x[index] = 1

        yhat = model(x).swapaxes(1,2)
        l = loss(yhat, y)
        l.backward()
        optimizer.step()

if __name__ == "__main__":
    print("NUMBER OF BATCH: ", len(train_loader))
    max_epoch = 100
    # Model params
    vocal_len = len(words)
    emb_dim = 32
    hidden_size = 64
    num_classes = len(tags)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss(ignore_index=0)
    rnn = seq2seq(vocal_len, emb_dim, hidden_size, num_classes)
    rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)


    for epoch in range(max_epoch):
        print("Epoch: ", epoch)
        train(train_loader, rnn, loss, optimizer)
        evaluate(train_loader, rnn, loss, title= 'Train Set')
        evaluate(test_loader, rnn, loss, title='Test Set')
        print()
        
            
