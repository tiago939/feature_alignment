import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os, random, sys
import numpy as np
import matplotlib.pyplot as plt
import networks

#random seed
manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True

device = 'cuda'
batch_size = 128
Z = 128
epochs = 1
T = 1

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# Model
print('==> Building model..')
model = networks.Memory(Z).to(device)

for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

'''
# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load('./checkpoint_memory/ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
'''

model.train()
# Training
for epoch in range(1):
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        print('epoch: ', epoch, ' batch: ', batch_idx)

        inputs = inputs.to(device)
        x = torch.clamp(inputs, min=0.01, max=0.99)
        for layer in range(12):
            r = torch.zeros(x.shape, device=device)
            r.requires_grad_(True)

            fx = model.encoder[layer](x)

            for _ in range(T):
                fr = model.encoder[layer](r)
                cost = 0.5*torch.sum( (fx - fr)**2.0)
                r = r - torch.autograd.grad(cost, r, retain_graph=True, create_graph=True)[0]

            loss = 0.5*torch.sum( (x - r)**2.0)
            if layer == 0:
                r = torch.tanh(r)

            for p in model.encoder[layer].parameters():
                p.grad = torch.autograd.grad(loss, p, retain_graph=True, create_graph=True)[0]

            x = fx.detach()

        optimizer.step()

print('Saving..')
if not os.path.isdir('checkpoint_memory'):
    os.mkdir('checkpoint_memory')
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, './checkpoint_memory/ckpt.pth')
print('done!')
