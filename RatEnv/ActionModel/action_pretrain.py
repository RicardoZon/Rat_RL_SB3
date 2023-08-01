import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = "cpu"
print(f"Using {device} device")


def plot_simple2(datas: list, leg=None, xlabel="X", ylabel="Y", figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    for data in datas:
        ax.plot(data[0], data[1])
    ax.set_xlabel(xlabel)  # 为x轴命名为“x”
    ax.set_ylabel(ylabel)  # 为y轴命名为“y”
    if leg is not None:
        ax.legend(leg)
    # plt.axis('equal')
    ax.grid()
    fig.show()
    return fig, ax


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


if __name__ == '__main__':
    data_io = scio.loadmat("../test/ActionPre_front_kineYH.mat")
    thetaH = data_io['theta']
    ratioH = data_io['ratio']
    plot_simple2([[thetaH, ratioH]])
