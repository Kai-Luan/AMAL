import torch
from tp1 import mse, linear
#from tp1 import mse, MSE
# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))


#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)

x = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
w = torch.randn(5,10, requires_grad=True, dtype=torch.float64)
b = torch.randn(10, requires_grad=True, dtype=torch.float64)

torch.autograd.gradcheck(linear, (x, w,b))