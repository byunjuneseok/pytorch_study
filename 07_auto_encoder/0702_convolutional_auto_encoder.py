import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64*3*3
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.fc(z.view(-1, 64 * 3 * 3))

        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 64 * 3 * 3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1),
            #             nn.MaxUnpool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            #             nn.MaxUnpool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        out = self.decoder(z.view(-1, 64, 3, 3))

        return out


if __name__ == "__main__":
    mnist_train = dsets.MNIST(root='data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    mnist_test = dsets.MNIST(root='data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    batch_size = 100

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                              batch_size=500,
                                              shuffle=False)

    E = Encoder().cuda()
    D = Decoder().cuda()

    loss = nn.MSELoss()
    optimizer = optim.Adam(list(E.parameters()) + list(D.parameters()), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):

        total_batch = len(mnist_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.cuda()

            pre = D(E(X))
            cost = loss(pre, X)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

    print("Learning Finished!")

