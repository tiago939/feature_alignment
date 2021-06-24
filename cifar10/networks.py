import torch 
import torch.nn as nn
import numpy as np

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class unFlatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, 1, 32, 32)

class asinh(torch.nn.Module):
    def forward(self, x):
        return torch.asinh(x)

class Encoder(nn.Module):

    def __init__(self,Z):
        super(Encoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            Flatten(),
            nn.Linear(2048,2048),
            nn.LeakyReLU(),
            )

        self.mean = nn.Linear(2048, Z)
        self.log_variance = nn.Linear(2048, Z)
        self.classifier = nn.Linear(2048,10)

    def forward(self, x, mode):
        if mode == 1:
            x = self.block1(x)
            return x

        if mode == 2:
            x = self.classifier(x)
            return x

        if mode == 3:
            x = self.mean(x)
            return x

        if mode == 4:
            x = self.log_variance(x)
            return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.generator0 = nn.Sequential(
            nn.Linear(10,  32*32),
            nn.LeakyReLU(),
            unFlatten(),
            )

        self.generator = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 3, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid()
            )

    def forward(self, x, y):
            y = self.generator0(y)
            x = torch.cat((x,y),1)
            x = self.generator(x)
            return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            Flatten(),
            nn.Linear(2048,2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1)
            )

    def forward(self, x):
            x = self.encoder(x)
            return x

class Memory(nn.Module):

    def __init__(self,Z):
        super(Memory, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            asinh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            asinh(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            asinh(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            asinh(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            asinh(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            asinh(),

            Flatten(),
            nn.Linear(2048, 2048),
            asinh(),
            nn.Linear(2048, Z)
            )


    def forward(self, x):
        x = self.encoder(x)
        return x
