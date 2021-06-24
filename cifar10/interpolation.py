import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os, random, sys, math
import numpy as np
import matplotlib.pyplot as plt
import networks

device = 'cuda'
batch_size = 1
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

z=[]
labels = []
for batch_idx, (inputs, targets) in enumerate(trainloader):

    if batch_idx >= 4:
        break

    inputs = inputs.to(device)
    inputs = torch.clamp(inputs, min=0.01,max=0.99)
    targets = targets.to(device)

    outputs = model(inputs, 1)
    C = model(outputs, 2)

    label = torch.zeros(C.shape, device=device)
    for i in range(batch_size):
        l = targets[i].item()
        label[i][l] = 1.0

    output = model(inputs,1)
    mean = model(output,3)
    log_variance = model(output,4)
    std = torch.exp(0.5*log_variance)
    eps = torch.empty(std.shape, device=device).normal_(mean=0.0,std=1.0)
    latent = mean + std*eps

    z.append(latent)
    labels.append(label)

imgs=[]
alpha = 0.0
while alpha <= 1.0:
    beta = 0.0
    while beta <= 1.0:
        p = (1.0-alpha)*((1.0-beta)*z[0] + beta*z[1]) + alpha*((1.0-beta)*z[2] + beta*z[3])
        q = (1.0-alpha)*((1.0-beta)*labels[0] + beta*labels[1]) + alpha*((1.0-beta)*labels[2] + beta*labels[3])

        u = 0.0*torch.rand((1,3,32,32), device=device)
        u.requires_grad_(True)
        for _ in range(T):
            out = model(u,1)
            Cf = model(out,2)
            meanf = model(out,3)
            log_variancef = model(out,4)
            stdf = torch.exp(0.5*log_variancef)
            zf = meanf + stdf
            cost =  0.5*torch.sum( (p - zf)**2.0) + 0.5*torch.sum( (q-Cf)**2.0)
            u = u- torch.autograd.grad(cost, u, retain_graph=True, create_graph=True)[0]

        u = torch.sigmoid(u)
        g = generator(u ,q)
        imgs.append(g[0].cpu())
        beta += 0.1
    alpha += 0.1

plt.clf()
grid_img = torchvision.utils.make_grid(imgs, nrow=11)
plt.axis('off')
plt.imshow(grid_img.permute(1,2,0))
plt.show()
