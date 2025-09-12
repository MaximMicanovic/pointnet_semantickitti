import torch

from pc_classifier.neuralnetcode import PointNet, TNet


def test_tnet():
    tnet = TNet(k=3)
    tnet.eval()
    tnet.forward(torch.randn(1, 3, 1024))


def test_pointnet():
    pointnet = PointNet(num_classes=25)
    pointnet.eval()
    pointnet.forward(torch.randn(1, 1024, 3))
