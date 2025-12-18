"""
Reprocessing script to convert the current half-split ECG segments into 
interleaved segments for contrastive learning.

Original approach: signal split in half (a = signal[:2500], b = signal[2500:])
New approach: interleaved sampling where:
  - Segment A: even time indices (0, 2, 4, ...) -> length 2500
  - Segment B: odd time indices (1, 3, 5, ...) -> length 2500
Both segments have length 2500 (half of original signal).
"""

import numpy as np
import shutil
from pathlib import Path


def reconstruct_original(segment_a: np.ndarray, segment_b: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original signal from the two halves.
    Assumes original was split as: a = original[:, :2500, :], b = original[:, 2500:, :]
    Data shape: (samples, time, channels)
    """
    return np.concatenate([segment_a, segment_b], axis=1)


def create_interleaved_segments(original: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create interleaved segments from the original signal.
    
    For a signal of shape (samples, 5000, channels):
      - Segment A: values at even time indices (0, 2, 4, ...) -> shape (samples, 2500, channels)
      - Segment B: values at odd time indices (1, 3, 5, ...) -> shape (samples, 2500, channels)
    
    Returns two arrays of shape (samples, 2500, channels).
    """
    segment_a = original[:, 0::2, :]  # Even time indices
    segment_b = original[:, 1::2, :]  # Odd time indices
    
    return segment_a, segment_b


def process_split(split_name: str, input_dir: Path, output_dir: Path) -> None:
    """
    Process a single data split (train/val/test).
    
    Loads the existing half-split segments, reconstructs the original,
    then creates the new zero-interleaved segments.
    """
    print(f"Processing {split_name} split...")
    
    segment_a_path = input_dir / f"{split_name}_segment_a.npy"
    segment_b_path = input_dir / f"{split_name}_segment_b.npy"
    labels_path = input_dir / f"{split_name}_labels.npy"
    meta_path = input_dir / f"{split_name}_meta.csv"
    
    segment_a_old = np.load(segment_a_path)
    segment_b_old = np.load(segment_b_path)
    
    print(f"  Loaded segment_a: {segment_a_old.shape}, segment_b: {segment_b_old.shape}")
    
    original = reconstruct_original(segment_a_old, segment_b_old)
    print(f"  Reconstructed original: {original.shape}")
    
    segment_a_new, segment_b_new = create_interleaved_segments(original)
    print(f"  Created interleaved segment_a: {segment_a_new.shape}, segment_b: {segment_b_new.shape}")
    
    np.save(output_dir / f"{split_name}_segment_a.npy", segment_a_new)
    np.save(output_dir / f"{split_name}_segment_b.npy", segment_b_new)
    
    shutil.copy(labels_path, output_dir / f"{split_name}_labels.npy")
    shutil.copy(meta_path, output_dir / f"{split_name}_meta.csv")
    
    print(f"  Saved to {output_dir}")


def main():
    base_dir = Path(__file__).parent.parent / "data"
    input_dir = base_dir / "processed"
    output_dir = base_dir / "processed_interleaved_split"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    for split in ["train", "val", "test"]:
        process_split(split, input_dir, output_dir)
        print()
    
    print("Reprocessing complete!")
    
    print("\nVerification:")
    for split in ["train"]:
        seg_a = np.load(output_dir / f"{split}_segment_a.npy")
        seg_b = np.load(output_dir / f"{split}_segment_b.npy")
        
        print(f"  {split} segment_a shape: {seg_a.shape}")
        print(f"  {split} segment_b shape: {seg_b.shape}")
        
        # Verify interleaving pattern on first sample
        print(f"  First sample, first 10 values of segment_a: {seg_a[0, 0, :10]}")
        print(f"  First sample, first 10 values of segment_b: {seg_b[0, 0, :10]}")
        
        # Verify interleaving by checking reconstruction
        original_reconstructed = np.zeros((seg_a.shape[0], 5000, seg_a.shape[2]), dtype=seg_a.dtype)
        original_reconstructed[:, 0::2, :] = seg_a
        original_reconstructed[:, 1::2, :] = seg_b
        print(f"  Reconstructed original shape: {original_reconstructed.shape}")


if __name__ == "__main__":
    main()
