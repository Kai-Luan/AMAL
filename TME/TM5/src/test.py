
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
from pathlib import Path

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    return torch.nn.functional.cross_entropy(output, target, reduction='none', ignore_index=padcar)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=torch.nn.ReLU, ):
        super(RNN,self).__init__()
        # input
        self.wi = nn.Linear(input_size, hidden_size, bias = False , dtype= torch.float32)
        # encode
        self.wh = nn.Linear(hidden_size, hidden_size ,dtype = torch.float32)
        self.wd = nn.Linear(hidden_size, output_size ,dtype = torch.float32)
        self.hidden_size = hidden_size
    
    def one_step(self, x, h):
        return torch.tanh(self.wi(x) + self.wh(h))
    
    def decode(self, h):
        return  torch.relu(self.wd(h))
    
    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(1), self.hidden_size)
        out = [h]
        for i in range(x.size(0)):
            h = self.one_step(x[i], out[-1])
            out.append(h)
        out.pop(0)
        out = torch.stack(out)
        return out # Time, Batch, Hidden

class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, input_size, hidden_size, output_size):
        RNN.__init__(input_size, hidden_size, output_size)
        hx_dim = hidden_size+input_size
        self.wc = nn.Linear(hx_dim, hidden_size)
        self.wf = nn.Linear(hx_dim, hidden_size)
        self.wi = nn.Linear(hx_dim, hidden_size)
        self.wo = nn.Linear(hx_dim, output_size)
        self.hidden_size = hidden_size
        self.C = None
    
    def one_step(self, x, h):
        if self.C is None:
            self.C = torch.zeros(x.shape[1], self.hidden_size)
        hx = torch.cat([h,x], dim=1)
        ft = torch.sigmoid(self.wf(hx)) # ?
        it = torch.sigmoid(self.wi(hx))
        self.C = ft*self.C + it*torch.tanh(self.wc(hx))
        o = torch.sigmoid(self.wo(hx))
        return o*torch.tanh(self.C)


class GRU(RNN):
    def __init__(self, input_size, hidden_size, output_size):
        RNN.__init__(self, input_size, hidden_size, output_size)
        hx_dim = hidden_size+input_size
        self.wz = nn.Linear(hx_dim, hidden_size)
        self.wr = nn.Linear(hx_dim, hidden_size)
        self.wh = nn.Linear(hx_dim, hidden_size)
        self.hidden_size = hidden_size
    
    def one_step(self, x, h):
        hx = torch.cat([h,x], dim=1)
        zt = torch.sigmoid(self.wz(hx)) 
        rt = torch.sigmoid(self.wr(hx))
        return (1-zt)*h + zt*torch.tanh(self.wh(torch.cat([rt*h,x], dim=1)))

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
PATH = "data/"
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_trump = DataLoader(TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), collate_fn=pad_collate_fn, batch_size= batch_size, shuffle=True)

hidden = 100
DIM_INPUT = len(id2lettre.keys())
feature = 100
encoder = nn.Embedding(DIM_INPUT, feature)
rnn = GRU(feature, hidden, DIM_INPUT)

if False:
    if Path("modelTrump.pch").is_file():
        with Path("modelTrump.pch").open("rb") as fp:
            rnn = torch.load(fp)  # Resume from the saved model
    if Path("encoder.pch").is_file():
        with Path("encoder.pch").open("rb") as fp:
            encoder = torch.load(fp)  # Resume from the saved model



loss = nn.CrossEntropyLoss(ignore_index=0)
loss_test = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
optimizer = torch.optim.Adam(rnn.parameters(), lr=5e-4)
rnn.to(device)

print("NUMBER OF BATCH: ", len(data_trump))
for epoch in range(10):
    for i, x in enumerate(data_trump):
        optimizer.zero_grad()
        x = x.to(device) # lettre, Batch
        x,y = encoder(x[:-1]),x[1:]
        # Time, Batch, Hidden
        yhat = rnn.decode(rnn(x))
        yhat = yhat.swapaxes(2,1)
        l = loss(yhat, y)
        print(loss_test(yhat, y))
        exit()
        l.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch : ", epoch, " Iteration : ", i, " Loss : ", round(l.item(),4))