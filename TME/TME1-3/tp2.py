import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm



writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

datax = torch.randn_like(datax)
datay = torch.randn_like(datay)

mse = torch.nn.MSELoss()
linear1 = torch.nn.Linear(datax.shape[1],5)
tanH = torch.nn.Tanh()
linear2 = torch.nn.Linear(5,1)


# TODO: 
# epochmax = 100
# eps = 1e-2
# optim = torch.optim.SGD(params=[linear2.weight,linear2.bias, linear1.weight,linear1.bias],lr=eps)
# optim.zero_grad()

# Question 1

# Batch ver
# for epoch in range(epochmax):
#     x = tanH(linear1(datax))
#     x = linear2(x)
#     loss = mse.forward(x,datay)

#     writer.add_scalar('Loss/train', loss, epoch)

#     # Sortie directe
#     print(f"Itérations {epoch}: loss {loss}")

#     ##  TODO:  Calcul du backward (grad_w, grad_b)
#     loss.backward()
#     ##  TODO:  Mise à jour des paramètres du modèle
#     # with torch.no_grad():
#     #     linear.weight -= eps* linear.weight.grad
#     #     linear.bias -= eps* linear.bias.grad
#     #     # linear.weight = torch.nn.Parameter(linear.weight.data - eps* linear.weight.grad.data)
#     #     # linear.bias = torch.nn.Parameter(linear.bias.data - eps* linear.bias.grad.data)
    
#     # linear.zero_grad()
#     optim.step()
#     optim.zero_grad()

# Mini-batch
batch_size = 100
for epoch in range(epochmax):
    permutation = torch.randperm(datax.size()[0])
    for i in range(0,datax.size()[0], batch_size):

        index = permutation[i:i+batch_size]
        batch_x, batch_y = datax[index],datay[index]
        x = tanH(linear1(batch_x))
        x = linear2(x)
        loss = mse.forward(x,batch_y)
        loss.backward()

        optim.step()
        optim.zero_grad()
    with torch.no_grad():
        x = tanH(linear1(datax))
        x = linear2(x)
        loss = mse.forward(x,datay)
        writer.add_scalar('Loss/train', loss, epoch)
        # Sortie directe
        print(f"Itérations {epoch}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    #loss.backward()
    ##  TODO:  Mise à jour des paramètres du modèle
    # with torch.no_grad():
    #     linear.weight -= eps* linear.weight.grad
    #     linear.bias -= eps* linear.bias.grad
    #     # linear.weight = torch.nn.Parameter(linear.weight.data - eps* linear.weight.grad.data)
    #     # linear.bias = torch.nn.Parameter(linear.bias.data - eps* linear.bias.grad.data)
    
    # linear.zero_grad()

# Stochastique

# for epoch in range(epochmax):
#     permutation = torch.randperm(datax.size()[0])
#     for i in range(0,datax.size()[0]):

#         index = permutation[i]
#         elemx, elemy = datax[index],datay[index]
#         x = tanH(linear1(elemx))
#         x = linear2(x)
#         loss = mse.forward(x,elemy)
#         loss.backward()

#         optim.step()
#         optim.zero_grad()
#     with torch.no_grad():
#         x = tanH(linear1(datax))
#         x = linear2(x)
#         loss = mse.forward(x,datay)
#         writer.add_scalar('Loss/train', loss, epoch)
#         # Sortie directe
#         print(f"Itérations {epoch}: loss {loss}")

#     ##  TODO:  Calcul du backward (grad_w, grad_b)
#     #loss.backward()
#     ##  TODO:  Mise à jour des paramètres du modèle
#     # with torch.no_grad():
#     #     linear.weight -= eps* linear.weight.grad
#     #     linear.bias -= eps* linear.bias.grad
#     #     # linear.weight = torch.nn.Parameter(linear.weight.data - eps* linear.weight.grad.data)
#     #     # linear.bias = torch.nn.Parameter(linear.bias.data - eps* linear.bias.grad.data)
    
#     # linear.zero_grad()

epochmax = 100
eps = 1e-2
model = torch.nn.Sequential(linear1,tanH,linear2)
optim = torch.optim.SGD(model.parameters(), lr=eps)
optim.zero_grad()

# Question 2 
for epoch in range(epochmax):
    x = model.forward(datax)
    loss = mse.forward(x,datay)

    print(f"Itérations {epoch}: loss {loss}")

    loss.backward()
    
    optim.step()
    optim.zero_grad()


