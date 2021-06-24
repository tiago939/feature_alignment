import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os, random, sys, math
import numpy as np
import matplotlib.pyplot as plt
import networks

device = 'cuda'
batch_size = 25
Z = 128
T = 1

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False, num_workers=1)


model = networks.Encoder(Z).to(device)
generator = networks.Generator().to(device)
checkpoint = torch.load('./checkpoint_fa/ckpt.pth')
model.load_state_dict(checkpoint['encoder_state_dict'])
generator.load_state_dict(checkpoint['generator_state_dict'])

model.eval()
generator.eval()

z = torch.empty((batch_size,Z), device=device).normal_(mean=0.0,std=1.0)
labels = torch.zeros((batch_size,10),device=device)
for i in range(batch_size):
    l = random.randint(0,9)
    labels[i][l] = 1.0

r =torch.zeros((batch_size,3,32,32), device=device)
for _ in range(T):
    r.requires_grad_(True)
    out = model(r,1)
    meanf = model(out,3)
    log_variancef = model(out,4)
    stdf = torch.exp(0.5*log_variancef)
    zf = meanf + stdf
    Cf = model(out,2)

    cost = 0.5*torch.sum( (z-zf)**2.0) + 0.5*torch.sum( (labels-Cf)**2.0)
    r = r - torch.autograd.grad(cost, r, retain_graph=True, create_graph=True)[0]
    r = torch.sigmoid(r)

g = generator(r,labels)
plt.clf()
grid_img = torchvision.utils.make_grid(g.cpu(), nrow=5)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.show()
