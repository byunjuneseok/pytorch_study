import torch.nn as nn
import torch.nn.init
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Linear(28 * 28, 28)
        self.decoder = nn.Linear(28, 28 * 28)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z = self.encoder(x)

        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28)

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
                                              batch_size=1,
                                              shuffle=False)

    model = AE().cuda()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    for epoch in range(num_epochs):

        total_batch = len(mnist_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.cuda()

            recon_X = model(X)
            cost = loss(recon_X, X)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

    print("Learning Finished!")