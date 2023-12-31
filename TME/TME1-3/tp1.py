# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        #  TODO:  Renvoyer la valeur de la fonction
        return ((yhat-y)**2).sum()/len(yhat)

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)

        return (2/len(yhat))*(yhat-y)*grad_output,(-2/len(yhat))*(yhat-y)*grad_output

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    """Début d'implementation de la fonction Linear"""
    @staticmethod
    def forward(ctx, x,w,b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x,w,b)
        #  TODO:  Renvoyer la valeur de la fonction
        return x@w+b

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        x,w,b = ctx.saved_tensors

        #print(w.shape,x.shape,grad_output.shape)
        #print((w.T).shape)

        return grad_output@w.T, x.T@grad_output, grad_output.sum(0)

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

