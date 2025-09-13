"""This is for segmenting data with the model"""

import pandas as pd
import numpy as np

# My own little wheel
import rustlib

from pc_classifier.neuralnetcode import PointNet
from pc_classifier.dataset import PointsOneSegment

import torch
from torch import autocast, cpu
from torch.utils.data import DataLoader


def making_npndarray(points: list[float]):
    """Making npndarray

    Args:
        points (list[list[float]]): The points

    Returns:
        np.ndarray: The points
    """
    np_points = np.zeros((len(points), 3))
    for i, point in enumerate(points):
        np_points[i] = point
    return np_points.astype(np.float32)


def load_points(path: str, amount: int):
    """Load points from the rustlib

    Args:
        path (str): The path to the points

    Returns:
        np.ndarray: The points
    """
    points = rustlib.read_lidar_data(path, amount)
    points = rustlib.flip_points(points)
    points = rustlib.flip_points(points)
    # points = rustlib.normalize_points(points, 30)
    return making_npndarray(points)


def load_model(device: torch.device, num_classes: int):
    """
    Load the model
    """
    model = PointNet(num_classes)
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    model.eval()
    model.to(device)
    return model


def corresponding_colors(outputs: list[list[int]]):
    """Corresponding colors

    Args:
        outputs (torch.Tensor): The outputs

    Returns:
        list[list[int]]: The colors
    """
    color_map: dict[int, list[int]] = {
        0: [255, 255, 255],
        1: [0, 0, 255],
        2: [245, 150, 100],
        3: [245, 230, 100],
        4: [250, 80, 100],
        5: [150, 60, 30],
        6: [255, 0, 0],
        7: [180, 30, 80],
        8: [255, 0, 0],
        9: [30, 30, 255],
        10: [200, 40, 255],
        11: [90, 30, 150],
        12: [255, 0, 255],
        13: [255, 150, 255],
        14: [75, 0, 75],
        15: [75, 0, 175],
        16: [0, 200, 255],
        17: [50, 120, 255],
        18: [0, 150, 255],
        19: [170, 255, 150],
        20: [0, 175, 0],
        21: [0, 60, 135],
        22: [80, 240, 150],
        23: [150, 240, 255],
        24: [0, 0, 255],
        25: [255, 255, 50],
    }
    colors = []
    for output in outputs:
        rgb_colors = color_map[torch.argmax(output).item()]
        colors.append([rgb / 255.0 for rgb in rgb_colors])
    return colors


def reforming_points(points: list[np.ndarray]):
    """Reforming points"""
    return_points = []
    for point in points:
        return_points.extend(point.tolist())
    return return_points


def model_segment(
    device: torch.device, points: np.ndarray, num_classes: int
) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    """Model segment

    Args:
        points (np.ndarray): The points
        num_classes (int): The number of classes

    Returns:
        tuple[np.ndarray, list[list[int]]]: The points and colors
    """
    model = load_model(device, num_classes)
    dataset = PointsOneSegment(points)

    loader = DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=6, prefetch_factor=1
    )

    return_outputs = []

    for step, point in enumerate(loader):
        point = point.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                model_output = model(point)
                return_outputs.extend(model_output.cpu())
        if step % 10 == 0:
            print(f"Step {step} / {len(loader)} completed")

    return points, corresponding_colors(return_outputs)


def save_classified_points(points: list[np.ndarray], colors: list[torch.Tensor]):
    """Save classified points"""
    save_points = []
    for point_color in zip(points, colors):
        save_points.append(point_color)
    pd.DataFrame(save_points).to_feather("classified_points.feather")
    print("Points saved")


def load_classified_points() -> (list[np.ndarray], list[np.ndarray]):
    """Load classified points"""
    loaded_points = pd.read_feather("classified_points.feather")
    print("Classified points loaded")
    return loaded_points[0], loaded_points[1]


def main():
    """
    Main function
    """
    amount_points_to_load = 5_000_000
    num_classes = 25

    """
    loaded_points = pd.read_feather("classified_points.feather")
    loaded_points[0] = rustlib.normalize_points(loaded_points[0], 30)
    rustlib.render_points(loaded_points[0], loaded_points[1])
    """

    # Initialize device and points
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = load_points("pcd_data/asciiout.pcd", amount_points_to_load)

    # points = np.random.choice(len(points), size=100_000, replace=False)
    print("Amount of points after random choice", len(points))

    # Model segments data
    points, colors = model_segment(device, points, num_classes)
    print("Model segmented data")

    save_classified_points(points, colors)
    # rustlib.render_points(points, colors)


if __name__ == "__main__":
    main()
