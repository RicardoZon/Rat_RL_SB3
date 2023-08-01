import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import scipy.io as scio
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


def cart2pol(x: np.ndarray, y: np.ndarray):
    theta = np.arctan2(y, x)
    rho = np.sqrt(y ** 2 + x ** 2)
    return theta, rho


def Sdraw(x1, x2):
    plt.figure()
    if x1 is None:
        plt.plot(x2)
    else:
        plt.plot(x1, x2)
    plt.show()


class MyDataset(Dataset):
    def __init__(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("Error")
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class FitNet(nn.Module):
    def __init__(self):
        super(FitNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plt.figure()
    plt.semilogy(losses)
    plt.show()

if __name__ == '__main__':
    CF = [0.0, -0.045]
    CH = [0.0, -0.05]
    data_io = scio.loadmat("boundary_hl.mat")
    y = data_io['y']
    z = data_io['z']
    y, z = np.float64(y), np.float64(z)

    C = CH

    theta, rho = cart2pol(y-C[0], z-C[1])
    Sdraw(theta, rho)


    data = MyDataset(theta, rho)
    theta_T = torch.tensor(theta, dtype=torch.float64, device=device)

    # Training
    batch_size = 32
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    #
    for X1, Y1 in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X1.shape}")
        print(f"Shape of y: {Y1.shape} {Y1.dtype}")
        break

    model = FitNet().to(device)
    model = model.double()
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    losses = []


    epochs = 3000
    for t in range(epochs):
        if t%20 == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
    plt.figure()
    plt.plot(losses)
    plt.show()
        # test(test_dataloader, model, loss_fn)
    print("Done!")


    # test
    rho_Train = model(theta_T).detach().cpu().numpy()
    # yt = CF[0] + rhot * np.cos(theta)
    # zt = CF[1] + rhot * np.sin(theta)

    plt.figure()
    plt.plot(theta, rho)
    plt.plot(theta, rho_Train)
    plt.show()

    # scio.savemat('fit_Torch_fl_kineYH.mat', {'theta': theta, 'rho_f': rho_Train})


    # torch.save(model.state_dict(), "fl_boundary.pth")

