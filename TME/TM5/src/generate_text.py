from pathlib import Path
from textloader import  string2code, id2lettre, code2string
from generate import generate_beam
from tp5 import GRU, State
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    savepath = Path("state.pch")
    if savepath.is_file():
        with savepath.open("rb")as fp:
            state=torch.load(fp, map_location=torch.device('cpu'))
    S = generate_beam(state.rnn, state.encoder, code2string, eos=1, k=10, maxlen=100, start=3)
    for i,s in enumerate(S):
        print(f'{s}\n')
