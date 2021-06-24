import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os, random, sys, math
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
Z = 128 #latent vector size
epochs = 1 #number of epochs
T = 1 #number of feature iterations
P = 0.0 #weight of the perceptual loss

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# Model
print('==> Building model..')
encoder = networks.Encoder(Z).to(device)
discriminator = networks.Discriminator().to(device)
generator = networks.Generator().to(device)

for m in encoder.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)

for m in generator.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)

for m in discriminator.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)

optimizer = optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5,0.999))
optimizer_gen = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5,0.999))
optimizer_discr = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5,0.999))


#loads checkpoint
checkpoint = torch.load('./checkpoint_fa/ckpt.pth',map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
generator.load_state_dict(checkpoint['generator_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
optimizer_discr.load_state_dict(checkpoint['optimizer_discr_state_dict'])
optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])


# Training
encoder.train()
discriminator.train()
generator.train()

for epoch in range(epochs):

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('epoch: ', epoch, ' batch: ', batch_idx)
        for m in encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.data = torch.clamp(m.weight.data, min=-(2**0.5), max=2**0.5)

        inputs = inputs.to(device)
        inputs = torch.clamp(inputs, min=0.01, max=0.99)
        outputs = encoder(inputs, 1)
        C = encoder(outputs, 2)

        labels = torch.zeros(C.shape, device=device)
        for i in range(len(labels)):
            l = targets[i].item()
            labels[i][l] = 1.0

        output = encoder(inputs,1)
        mean = encoder(output,3)
        log_variance = encoder(output,4)
        std = torch.exp(0.5*log_variance)
        eps = torch.empty(std.shape, device=device).normal_(mean=0.0,std=1.0)
        z = mean + std*eps

        #extract feature
        r = torch.zeros(inputs.shape, device=device)
        r.requires_grad_(True)
        for _ in range(T):
            out = encoder(r,1)
            meanf = encoder(out,3)
            log_variancef = encoder(out,4)
            stdf = torch.exp(0.5*log_variancef)
            Cf = encoder(out,2)
            zf = meanf + stdf
            cost = 0.5*torch.sum( (z - zf)**2.0) + 0.5*torch.sum( (labels-Cf)**2.0)
            r= r - torch.autograd.grad(cost, r, retain_graph=True, create_graph=True)[0]
        r = torch.sigmoid(r)

        RL = 0.5*torch.sum( (inputs-r)**2.0) #align feaute and inputs
        KLD = -0.5*torch.sum( 1.0 + log_variance - mean**2.0 - torch.exp(log_variance), axis=1)
        beta = torch.rand(KLD.shape, device=device)

        loss = torch.sum(beta*KLD) + RL
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        g = generator(r.detach(), labels)
        output_g = encoder(g, 1)
        mean_g = encoder(output_g, 3)
        log_variance_g = encoder(output_g, 4)
        std_g = torch.exp(0.5*log_variance_g)

        output = encoder(inputs,1)
        mean = encoder(output,3)
        log_variance = encoder(output,4)
        std = torch.exp(0.5*log_variance)

        D = discriminator(g)
        loss = 0.5*torch.sum( (1.0-D)**2.0) + P*(0.5*torch.sum( (mean-mean_g)**2.0) + 0.5*torch.sum( (std - std_g)**2.0))
        optimizer_gen.zero_grad()
        loss.backward()
        optimizer_gen.step()

        D_real = discriminator(inputs)
        loss = 0.5*torch.sum( (1.0-D_real)**2.0)
        optimizer_discr.zero_grad()
        loss.backward()
        optimizer_discr.step()

        z = torch.empty(z.shape, device=device).normal_(mean=0.0,std=1.0)
        r = torch.zeros(inputs.shape, device=device)
        r.requires_grad_(True)
        out = encoder(r,1)
        meanf = encoder(out,3)
        log_variancef = encoder(out,4)
        stdf = torch.exp(0.5*log_variancef)
        Cf = encoder(out,2)
        zf = meanf + stdf
        cost = 0.5*torch.sum( (z - zf)**2.0) + 0.5*torch.sum( (labels-Cf)**2.0)
        r = r - torch.autograd.grad(cost, r, retain_graph=True, create_graph=True)[0]
        r = torch.sigmoid(r)
        g = generator(r,labels)

        D_fake = discriminator(g.detach())
        loss = 0.5*torch.sum( (-1.0-D_fake)**2.0)
        optimizer_discr.zero_grad()
        loss.backward()
        optimizer_discr.step()

print('Saving..')
if not os.path.isdir('checkpoint_fa'):
    os.mkdir('checkpoint_fa')
torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'optimizer_gen_state_dict': optimizer_gen.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_discr_state_dict': optimizer_discr.state_dict(),
        }, './checkpoint_fa/ckpt.pth')
print('done!')
