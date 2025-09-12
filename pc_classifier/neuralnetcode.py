"""
Neuralnetcode module for PointNet
"""

from itertools import pairwise

import torch
import torch.nn as nn
from torch.nn import Sequential


def make_conv_seq(layersizes: list[int]) -> nn.Sequential:
    layers = []
    for i, o in pairwise(layersizes):
        layers.extend([nn.Conv1d(i, o, 1), nn.BatchNorm1d(o), nn.ReLU()])
    return nn.Sequential(*layers)


def make_lin_seq(layersizes: list[int]) -> Sequential:
    """Make a sequential model

    Args:
        layersizes (list[int]): The sizes of the layers
        n (int): The number of points

    Returns:
        Sequential: The sequential model
    """
    layer = []
    for i, o in pairwise(layersizes):
        layer.extend([nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()])
    return Sequential(*layer)


class TNet(torch.nn.Module):
    """TNet module"""

    def __init__(
        self,
        k: int,
        conv_channels: list[int] = [64, 128, 1024],
        lin_channels: list[int] = [1024, 512, 256],
    ):
        """Initialize the TNet

        Args:
            k (int): The number of points
            conv_channels (list[int]): The number of channels for the convolutional layers
            lin_channels (list[int]): The number of channels for the linear layers
        """
        super().__init__()
        self.k = k
        self.layer_conv = make_conv_seq([k, *conv_channels])
        self.layer_lin = make_lin_seq([conv_channels[-1], *lin_channels])

        self.fc_rot = torch.nn.Linear(lin_channels[-1], k * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        x = self.layer_conv(x)
        x = torch.max(x, 2)[0]
        x = self.layer_lin(x)
        x = self.fc_rot(x)

        x = x.view(-1, self.k, self.k)
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0)
        x = x + identity

        return x


class PointNet(torch.nn.Module):
    """PointNet module"""

    def __init__(self, num_classes: int):
        """Initialize the PointNet"""
        super().__init__()

        # Input TNet
        self.input_tnet = TNet(k=3)

        # Input conv layer
        self.convlayer = torch.nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(64)

        # Feature TNet
        self.feature_tnet = TNet(k=64)

        # Shared MLP 2
        self.mlp_2 = make_conv_seq([64, 128, 1024])

        # Fully connected layer
        self.fully_connected = make_lin_seq([1024, 512, 256])
        self.dropout = torch.nn.Dropout(0.3)
        self.final = torch.nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        x = x.transpose(2, 1)
        # Input TNet
        t1 = self.input_tnet(x)  # (B, 3, 3)
        x = torch.bmm(t1, x)  # (B, 3, N)

        # Shared MLP 1
        x = torch.relu(self.bn1(self.convlayer(x)))

        # Feature TNet
        t2 = self.feature_tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t2)
        x = x.transpose(2, 1)

        # Shared MLP 2
        x = self.mlp_2(x)

        # Gloabal Max Pooling
        x = torch.max(x, 2)[0]

        # Fully connected layer
        x = self.fully_connected(x)

        # Dropout
        x = self.dropout(x)

        # Final layer
        x = self.final(x)

        return x
