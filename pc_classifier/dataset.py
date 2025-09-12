"""
Dataset module for SemanticKITTI dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from pc_classifier.transforms import Transforms


def making_label_to_list(label: int, num_classes: int) -> np.ndarray:
    """Making label to list

    Args:
        label (int): Label to make to list
        num_classes (int): Number of classes

    Returns:
        np.ndarray: One-hot encoded label
    """

    label_map = {
        0: 0,
        1: 0,
        10: 1,
        11: 2,
        13: 3,
        15: 4,
        16: 5,
        18: 6,
        20: 7,
        30: 8,
        31: 9,
        32: 10,
        40: 11,
        44: 12,
        48: 13,
        49: 14,
        50: 15,
        51: 16,
        52: 17,
        60: 18,
        70: 19,
        71: 20,
        72: 21,
        80: 22,
        81: 23,
        99: 24,
        252: 1,
        253: 2,
        254: 8,
        255: 4,
        256: 5,
        257: 3,
        258: 6,
        259: 7,
    }
    label_list = np.zeros(num_classes, dtype=np.float32)
    label_list[label_map[label]] = 1.0
    return label_list


def read_kitti_velodyne_bin(path: str) -> np.ndarray:
    """Read a KITTI velodyne binary file and return a tensor of shape (N, 4)

    Args:
        path (str): Path to the KITTI velodyne binary file

    Returns:
        np.ndarray: Array of shape (N, 4)
    """

    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts = pts[:, :3]
    return pts


def read_kitti_labels(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read SemanticKITTI point labels (uint32) and split semantic and instance ids.

    Lower 16 bits: semantic label; Upper 16 bits: instance id.
    Returns (semantic_labels, instance_ids), both shape (N,).
    """
    labels_uint32 = np.fromfile(path, dtype=np.uint32)
    semantic = (labels_uint32 & 0xFFFF).astype(np.int32)
    instance = (labels_uint32 >> 16).astype(np.int32)
    return semantic, instance


def load_bin_paths(bin_paths: list[Path], label_root: Path) -> list[Path]:
    """Load bin paths from csv

    Args:
        bin_paths (list[Path]): List of bin paths
        label_root (Path): Root path to the label files

    Raises:
        RuntimeError: No matching bin/label files found.

    Returns:
        list[Path]: List of frames
    """
    frames = []

    # If file saved_paths.csv found
    if Path("saved_paths.csv").exists():
        print("Saved paths found, loading from csv")
        frames = pd.read_csv(
            "saved_paths.csv",
            names=["bin_path", "label_path", "num_points"],
        ).to_dict(orient="records")

    else:
        print("No saved paths found, loading from bin/label files")
        for p in bin_paths:
            # Expect structure: .../sequences/<seq_id>/velodyne/<name>.bin
            # Map to labels:   .../sequences/<seq_id>/labels/<name>.label
            seq_id = p.parent.parent.name
            lab = label_root / seq_id / "labels" / f"{p.stem}.label"
            num_points = len(read_kitti_velodyne_bin(str(p)))

            if not lab.exists():
                continue

            frames.append(
                {
                    "bin_path": str(p),
                    "label_path": str(lab),
                    "num_points": num_points,
                }
            )
        if not frames:
            raise RuntimeError("No matching bin/label files found.")

        # Save frames to csv
        print("Saving frames to csv")
        pd.DataFrame(frames).to_csv(
            "saved_paths.csv",
            index=False,
            header=None,
        )

    return frames


class PointsDataset(Dataset):
    """PointsDataset class for SemanticKITTI dataset

    Args:
        Dataset (Dataset): Dataset class
        bin_dir (str): Path to the bin directory
        label_dir (str): Path to the label directory
        num_classes (int | None): Number of classes
    """

    def __init__(
        self,
        bin_dir: str,
        label_dir: str,
        num_classes: int | None = None,
        transforms: list[Transforms] = None,
    ):
        """Initialize the PointsDataset

        Args:
            bin_dir (str): Path to the bin directory
            label_dir (str): Path to the label directory
            num_classes (int | None, optional): Number of classes. Defaults to None.

        Raises:
            RuntimeError: No matching bin/label files found.
        """
        # Collect all .bin files recursively across all sequence folders
        self.bin_paths: list[Path] = sorted(Path(bin_dir).rglob("*.bin"))
        # Root directory that contains all per-sequence label folders
        self.label_root: Path = Path(label_dir)

        # Sum of points
        self.sum_points: int = 0

        # Number of classes
        self.num_classes: int = num_classes

        # Current tree for current scan && current scan id
        self.current_tree: cKDTree = None
        self.current_scan_id: int = None

        # Counter for swapping frames
        self.counter, self.counter_swap = 0, 0

        # Current points, semantic raw, and instance
        self.current_pts, self.current_sem_raw, self.current_inst = None, None, None

        # Transforms to apply to the points
        self.transforms = transforms

        # Keep only frames that have both bin and corresponding label
        self.frames = load_bin_paths(self.bin_paths, self.label_root)

        # Stable dataset length across scans for samplers/dataloaders
        self.most_points_in_a_scan: int = max(f["num_points"] for f in self.frames)

        new_frame_idx = np.random.randint(0, len(self.frames))
        self.load_scan(new_frame_idx)

    def load_scan(self, frame_id):
        """Load a scan from a frame

        Args:
            frame_id (int): The frame id

        """
        if self.current_scan_id == frame_id:
            pass

        self.current_pts = read_kitti_velodyne_bin(self.frames[frame_id]["bin_path"])
        self.current_sem_raw, self.current_inst = read_kitti_labels(
            self.frames[frame_id]["label_path"]
        )
        self.current_tree = cKDTree(self.current_pts)

        self.current_scan_id = frame_id

    def __getitem__(self, idx: int):
        if self.counter >= self.counter_swap:
            new_frame_idx = np.random.randint(0, len(self.frames))
            self.load_scan(new_frame_idx)
            self.counter = 0

            self.counter_swap = np.random.randint(50, 200)
        else:
            self.counter += 1

        # Map incoming global index to a valid local index for the active scan
        # Det här måste jag komma ihåg as smart den ökar chansen för det första punkterna i scanen
        local_idx = int(idx) % self.current_pts.shape[0]

        # Here i will find the 1024 closest points to the point im working in
        _, neighbor_indices = self.current_tree.query(
            self.current_pts[local_idx], k=1024
        )
        closest_pts = self.current_pts[neighbor_indices]
        closest_pts = closest_pts - self.current_pts[local_idx]

        if self.transforms:
            for transform in self.transforms:
                closest_pts = transform(closest_pts)

        # remap moving labels to static IDs and build one-hot for this point
        label = self.current_sem_raw[local_idx].astype(np.int64)
        current_label_list = making_label_to_list(int(label), self.num_classes)
        return (closest_pts, current_label_list)

    def __len__(self) -> int:
        return self.most_points_in_a_scan


class PointsOneSegment(Dataset):
    """PointsOneSegment class for SemanticKITTI dataset

    Args:
        Dataset (Dataset): Dataset class
        points (np.ndarray): Points float32[N, 3]
    """

    def __init__(self, points: np.ndarray):
        if len(points) == 0:
            raise ValueError("PointsOneSegment has no points to query.")
        self.points: np.ndarray = points
        self.tree: cKDTree = cKDTree(points)

    def __getitem__(self, idx: int):
        # Ensure k does not exceed available points
        _, neighbor_indices = self.tree.query(self.points[idx], k=1024)
        closest_pts = self.points[neighbor_indices]
        closest_pts = closest_pts - self.points[idx]
        return closest_pts

    def __len__(self) -> int:
        return len(self.points)
