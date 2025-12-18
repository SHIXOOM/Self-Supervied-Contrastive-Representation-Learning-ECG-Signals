"""
ECG Dataset classes for contrastive learning and downstream tasks.

Provides memory-efficient datasets that load data from disk on-demand
using memory-mapped numpy arrays.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from src.augmentor import DualAugmenter


class ECGContrastiveTrainDataset(torch.utils.data.Dataset):
    """
    ECG dataset for contrastive learning that loads paired segments from disk.

    Loads segment pairs on-demand from memory-mapped .npy files for memory efficiency.
    Each pair (segment_a[i], segment_b[i]) represents two 5s segments from the same
    10s ECG recording, forming a natural positive pair for contrastive learning.

    Augmentations are applied to both segments using DualAugmenter.
    Data is already normalized during preprocessing (per-sample per-channel z-score).

    Args:
        segment_a_path: Path to .npy file containing first segments (N, 2500, 12)
        segment_b_path: Path to .npy file containing second segments (N, 2500, 12)
        labels_path: Path to .npy file containing integer labels (N,)
        dual_augmenter: DualAugmenter instance for generating augmented views
        device: Device to place tensors on
    """

    def __init__(
        self,
        segment_a_path: Union[str, Path],
        segment_b_path: Union[str, Path],
        labels_path: Union[str, Path],
        dual_augmenter: DualAugmenter,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.segment_a = np.load(segment_a_path, mmap_mode="r") # mmap_mode = r should make the loading lazy,
        # to avoid memory cluttering
        self.segment_b = np.load(segment_b_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.dual_augmenter = dual_augmenter
        self.device = device

        assert len(self.segment_a) == len(self.segment_b) == len(self.labels), (
            f"Mismatched lengths: segment_a={len(self.segment_a)}, "
            f"segment_b={len(self.segment_b)}, labels={len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert mmap data to tensors (this creates a copy, avoiding read-only issues)
        seg_a = torch.from_numpy(np.array(self.segment_a[idx])).float()
        seg_b = torch.from_numpy(np.array(self.segment_b[idx])).float()

        # Apply augmentations to both segments
        # seg_a and seg_b are natural positive pairs from the same ECG
        aug_a, aug_b = self.dual_augmenter(ecg1=seg_a, ecg2=seg_b)

        return aug_a, aug_b


class ECGDataset(torch.utils.data.Dataset):
    """
    ECG dataset for downstream tasks that loads segments from disk.

    Combines both segment files into a single dataset where each segment
    is treated as an independent sample. This doubles the effective dataset size.

    For N original ECG recordings:
    - Indices 0 to N-1 correspond to segment_a samples
    - Indices N to 2N-1 correspond to segment_b samples

    Data is already normalized during preprocessing (per-sample per-channel z-score).

    Args:
        segment_a_path: Path to .npy file containing first segments (N, 2500, 12)
        segment_b_path: Path to .npy file containing second segments (N, 2500, 12)
        labels_path: Path to .npy file containing integer labels (N,)
    """

    def __init__(
        self,
        segment_a_path: Union[str, Path],
        segment_b_path: Union[str, Path],
        labels_path: Union[str, Path],
    ) -> None:
        self.segment_a = np.load(segment_a_path, mmap_mode="r")
        self.segment_b = np.load(segment_b_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        assert len(self.segment_a) == len(self.segment_b) == len(self.labels), (
            f"Mismatched lengths: segment_a={len(self.segment_a)}, "
            f"segment_b={len(self.segment_b)}, labels={len(self.labels)}"
        )

        self.n_original = len(self.labels)
        self.num_classes = int(np.max(self.labels)) + 1

    def __len__(self) -> int:
        # Each segment treated as independent sample
        # NOTE: Changed to return original length only,
        # since we are not using both segments anymore (to use it uncomment the else block in __getitem__) and double the length
        return self.n_original

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < self.n_original:
            # First half: segment_a samples
            # copy=True to create writable array from read-only memory-mapped source
            signal = np.array(self.segment_a[idx], dtype=np.float32, copy=True)
            label = int(self.labels[idx])
        # else:
        #     # Second half: segment_b samples
        #     original_idx = idx - self.n_original
        #     signal = np.array(self.segment_b[original_idx], dtype=np.float32, copy=True)
        #     label = int(self.labels[original_idx])

        signal_tensor = torch.from_numpy(signal).float()
        return signal_tensor, label
