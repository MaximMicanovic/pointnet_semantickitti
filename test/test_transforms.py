import numpy as np

from pc_classifier.transforms import Drop


def test_drop():
    drop = Drop(0.5)
    points = np.random.randn(100, 3)
    points = drop(points)
    assert points.shape == (100, 3)
