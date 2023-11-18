from pathlib import Path
from textloader import  string2code, id2lettre, code2string
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn).
         Initialise le réseau avec start (ou à 0 si start est vide) 
         et génère une séquence de longueur maximale 200 
         ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    h = rnn(emb(start))[-1]
    s = [rnn.decode(h).argmax(1)]
    for _ in range(maxlen):
        if s[-1] == eos: break
        h = rnn(s[-1].unsqueeze(1), h)
        s.append(rnn.decode(h).argmax(1))
    return decoder(s)



def generate_beam(rnn, emb, decoder, eos, k, start=0, maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    def topk(P):
        P = torch.log(P)
        for c in P.argsort()[-k:]:
            T.append((t+[c.item()], p+P[c].item(), h))
    
    compute = p_nucleus(decoder, 0.95)
    def sample(P):
        log_P = torch.log(P)
        for c in torch.multinomial(compute(P), k, replacement=False):
            T.append((t+[c.item()], p+log_P[c].item(), h))

    #  TODO:  Implémentez le beam Search
    S = [([start],0, None)]
    with torch.no_grad():
        for _ in range(maxlen):
            T = []
            for s in S: 
                t, p, h = s
                if t[-1] == eos: 
                    T.append(s)
                    continue
                if h is not None: h = h[0]
                x = torch.tensor(t[-1]).reshape(1,1)
                h = rnn(emb(x), h)
                P = rnn.decode(h).squeeze()
                #sample(P)
                topk(P)

            T.sort(key=lambda y: y[1], reverse=True)
            T = T[:k]
            S = T
        return [decoder(s) for (s,p,_) in S]


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        p = torch.softmax(h, dim=-1)
        p_sorted, indices = p.sort(descending=True)
        p_cum = torch.cumsum(p_sorted, dim=-1)
        nucleus = p_cum < alpha
        p[indices[nucleus]] = 0
        return p
    return compute

