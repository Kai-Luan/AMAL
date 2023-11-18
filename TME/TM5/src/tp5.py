
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
from pathlib import Path

#  TODO: 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    return torch.nn.functional.cross_entropy(output, target, reduction='none', ignore_index=padcar)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=None):
        super(RNN,self).__init__()
        # input
        self.wi = nn.Linear(input_size, hidden_size, bias = False , dtype= torch.float32)
        # encode
        self.wh = nn.Linear(hidden_size, hidden_size ,dtype = torch.float32)
        self.wd = nn.Linear(hidden_size, output_size ,dtype = torch.float32)
        self.hidden_size = hidden_size
        self.out_activation = activation
    
    def one_step(self, x, h):
        return torch.tanh(self.wi(x) + self.wh(h))
    
    def decode(self, h):
        out = self.wd(h)
        if self.out_activation is not None: out = self.out_activation(out)
        return out
    
    def forward(self, x, h=None):
        if h is None: 
            h = torch.zeros(x.size(1), self.hidden_size)
        out = [h]
        for i in range(x.size(0)):
            out.append(self.one_step(x[i], out[-1]))
        return torch.stack(out[1:])

class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, input_size, hidden_size, output_size):
        RNN.__init__(input_size, hidden_size, output_size)
        hx_dim = hidden_size+input_size
        self.wh = None
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
        self.wi = None
        self.wz = nn.Linear(hx_dim, hidden_size)
        self.wr = nn.Linear(hx_dim, hidden_size)
        self.wh = nn.Linear(hx_dim, hidden_size)
        self.hidden_size = hidden_size
    
    def one_step(self, x, h):
        hx = torch.cat([h,x], dim=1)
        zt = torch.sigmoid(self.wz(hx))
        rt = torch.sigmoid(self.wr(hx))
        return (1-zt)*h + zt*torch.tanh(self.wh(torch.cat([rt*h,x], dim=1)))


class State:
    def __init__(self, model, encoder, optim):
        self.rnn = model
        self.encoder = encoder
        self.optimizer = optim
        self.epoch, self.Iteration = 0, 0

# #TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
if __name__ == "__main__":
    PATH = "data/"
    batch_size = 100
    data_trump = DataLoader(TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), collate_fn=pad_collate_fn, batch_size= batch_size, shuffle=True)

    hidden = 100
    DIM_INPUT = len(id2lettre.keys())
    feature = 100
    max_epoch = 20


    print(f'{device = }')
    savepath = Path("state.pch")
    loss = nn.CrossEntropyLoss(ignore_index=0)
    if False and savepath.is_file():
        with savepath.open("rb")as fp:
            state=torch.load(fp)
    else:
        encoder = nn.Embedding(DIM_INPUT, feature)
        rnn = GRU1(feature, hidden, DIM_INPUT, activation=None)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
        state = State(rnn, encoder, optimizer)

    state.rnn.to(device)
    state.encoder.to(device)
    print("NUMBER OF BATCH: ", len(data_trump))
    for epoch in range(state.epoch, max_epoch):
        for i, x in enumerate(data_trump):
            state.optimizer.zero_grad()
            x = x.to(device) # lettre, Batch
            x,y = state.encoder(x[:-1]),x[1:]
            # Time, Batch, Hidden
            yhat = state.rnn.decode(state.rnn(x)).swapaxes(2,1)
            l = loss(yhat, y)
            l.backward()
            state.optimizer.step()
            if i % 10 == 0:
                print("Epoch : ", epoch, " Iteration : ", i, " Loss : ", round(l.item(),4))
        with torch.no_grad():
            c = 0
            for i, x, in enumerate(data_trump):
                x = x.to(device) # lettre, batch
                x,y = state.encoder(x[:-1]),x[1:]
                xi = x[:,1,:].unsqueeze(1)
                # Time, Batch, P
                s = state.rnn.decode(state.rnn(xi).to(device)).squeeze().argmax(1).tolist()
                y =  y[:,1].tolist()
                for j in range(len(s)):
                    if s[j] == EOS_IX:
                        s = s[:j]
                        break
                for j in range(len(y)):
                    if y[j] == EOS_IX:
                        y= y[:j]
                        break

                s = code2string(s)
                y = code2string(y)
                print("Reel: ==================\n", y)
                print("Pred: ==================\n", s)
                print()
                if c >= 3: break
                c+=1
        start = torch.randint(2,DIM_INPUT, (1,)).item()
        S = generate_beam(state.rnn, state.encoder, code2string, eos=1, k=2, maxlen=100, start=start)
        print("======== Generate ===================")
        for i,s in enumerate(S):
            print(f'{s}\n')
        with savepath.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state,fp)


