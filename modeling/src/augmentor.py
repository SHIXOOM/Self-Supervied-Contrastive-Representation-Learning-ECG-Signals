import random
from typing import Callable, List, Tuple

import torch


class DualAugmenter:
    """
    Augmenter that applies augmentations to two input signals.

    Takes two ECG signals and applies a randomly sampled augmentation to each,
    producing two augmented views.

    Args:
        augmentation_pool: List of augmentation functions to sample from.
                          Each augmentation should accept a torch.Tensor and return a torch.Tensor.
    """

    def __init__(self, augmentation_pool: List[Callable[[torch.Tensor], torch.Tensor]]):
        self.augmentation_pool = augmentation_pool

    def _sample_augmentation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Randomly sample an augmentation from the pool."""
        # NOTE: Currently set to one augmentation (k=1), which is Random Lead Masking.
        return random.sample(self.augmentation_pool, k=1)[0]

    def __call__(
        self, ecg1: torch.Tensor, ecg2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to two input signals.

        Args:
            ecg1: First input ECG tensor of shape (time_steps, num_channels)
            ecg2: Second input ECG tensor of shape (time_steps, num_channels)

        Returns:
            Tuple of (aug1, aug2) - two augmented views as tensors
        """
        augmenter1 = self._sample_augmentation()
        augmenter2 = self._sample_augmentation()

        aug1 = augmenter1(ecg1)
        aug2 = augmenter2(ecg2)

        return aug1, aug2
