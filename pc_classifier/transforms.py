"""
Transforms module for SemanticKITTI dataset
"""

from abc import ABC, abstractmethod

import numpy as np


class Transforms(ABC):
    """
    Transforms class for SemanticKITTI dataset
    """

    @abstractmethod
    def __call__(self):
        pass


class RandomRotate(Transforms):
    """
    Rotate transform for SemanticKITTI dataset
    """

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the points

        Args:
            points (np.ndarray): The points to transform

        Returns:
            np.ndarray: The transformed points
        """

        a_matrix = np.random.randn(3, 3)  # random matrix
        q_matrix, _ = np.linalg.qr(a_matrix)  # QR decomposition
        # Make sure determin ant is +1
        if np.linalg.det(q_matrix) < 0:
            q_matrix[:, 0] = -q_matrix[:, 0]
        q_matrix.astype(points.dtype)
        return points.dot(q_matrix)


class RandomMirror(Transforms):
    """
    Mirror transform for SemanticKITTI dataset
    """

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the points

        Args:
            points (np.ndarray): The points to transform

        Returns:
            np.ndarray: The transformed points
        """
        mirrored_points = points * np.random.choice([1, -1], size=3).astype(
            points.dtype
        )
        return mirrored_points


class Drop(Transforms):
    """
    Drop random points from the dataset
    """

    def __init__(self, drop_rate: float):
        """
        Initialize the Drop transform

        Args:
            drop_rate (float): The rate of points to drop
        """
        self.chance = drop_rate

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the points

        Args:
            points (np.ndarray): The points to transform

        Returns:
            np.ndarray: The transformed points
        """

        mask = np.random.rand(points.shape[0]) > self.chance
        mask = np.repeat(mask, 3).reshape(points.shape)
        return points * mask.astype(points.dtype)


class Jitter(Transforms):
    """
    Jitter transform for SemanticKITTI dataset
    """

    def __init__(self, jitter_range: float):
        """
        Initialize the Jitter transform
        """
        self.jitter_range = jitter_range

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the points
        """

        return (
            points
            + np.random.uniform(
                -self.jitter_range, self.jitter_range, size=points.shape
            )
        ).astype(points.dtype)
