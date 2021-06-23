import torch 
import torch.nn as nn
import numpy as np

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class unFlatten2(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, 1, 28, 28)

class Encoder(nn.Module):

    def __init__(self,Z):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            Flatten(),
            nn.Linear(3136,4096),
            nn.LeakyReLU(),
            )

        self.mean = nn.Linear(4096, Z)
        self.log_variance = nn.Linear(4096, Z)
        self.classifier = nn.Linear(4096,10)

    def forward(self, x, mode):
        if mode == 1:
            x = self.encoder(x)
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

        self.label = nn.Sequential(
            nn.Linear(10, 28*28),
            unFlatten2()
            )

        self.generator = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            )

    def forward(self, x, y):
            y = self.label(y)
            x = torch.cat((x,y),dim=1)
            x = self.generator(x)
            return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            Flatten(),
            nn.Linear(3136,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,1)
            )

    def forward(self, x):
            x = self.discriminator(x)
            return x

class Memory(nn.Module):

    def __init__(self,Z):
        super(Memory, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),

            Flatten(),
            nn.Linear(3136,4096),
            nn.Tanh(),
            nn.Linear(4096,Z)
            )

    def forward(self, x,):
        x = self.encoder(x)
        return x
