import torch
import torch.nn as nn
import torchvision


class Features(nn.Module):
    def __init__(self):
        self.names = torch.hub.list("moabitcoin/ig65m-pytorch")
        # ['r2plus1d_34_32_ig65m', 'r2plus1d_34_32_kinetics', 'r2plus1d_34_8_ig65m', 'r2plus1d_34_8_kinetics']
        self.model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=2, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

