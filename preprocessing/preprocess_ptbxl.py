"""PTB-XL Preprocessing Pipeline for Contrastive Learning

This script preprocesses the PTB-XL dataset according to the following specifications:
1. Loads raw ECG waveforms at 500 Hz using the official loading approach (wfdb)
2. Segments 10s signals into two non-overlapping 5s segments (2500 samples each)
3. Applies per-sample per-channel z-score normalization
4. Removes multi-label samples (only keeps samples with exactly one diagnostic superclass)
5. Outputs paired .npy arrays where index i in segment_a pairs with index i in segment_b

Output files (in data/processed/):
    - {split}_segment_a.npy: Shape (N, 2500, 12) - First 5s segment
    - {split}_segment_b.npy: Shape (N, 2500, 12) - Second 5s segment  
    - {split}_labels.npy: Shape (N,) - Integer labels
    - {split}_meta.csv: Metadata including ecg_id, patient_id, age, sex, label, scp_codes, strat_fold

Label mapping: {'NORM': 0, 'MI': 1, 'STTC': 2, 'HYP': 3, 'CD': 4}
"""

import ast
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


LABEL_TO_INT = {"NORM": 0, "MI": 1, "STTC": 2, "HYP": 3, "CD": 4}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
SAMPLING_RATE = 500
SEGMENT_LENGTH = 2500  # 5 seconds at 500 Hz


def load_raw_data(df: pd.DataFrame, path: str) -> np.ndarray:
    """
    Load raw ECG signals at 500 Hz using wfdb.
    
    Uses filename_hr column for high-resolution (500 Hz) recordings.
    Each recording has 5000 samples (10 seconds) across 12 leads.
    """
    signals = []
    for filename in tqdm(df.filename_hr, desc="Loading signals"):
        signal, _ = wfdb.rdsamp(path + filename)
        signals.append(signal)
    return np.array(signals)


def load_scp_mapping(path: str) -> pd.DataFrame:
    """
    Load SCP statements and filter to diagnostic codes only.
    
    Returns DataFrame indexed by SCP code with diagnostic_class column
    mapping each code to one of: NORM, MI, STTC, HYP, CD.
    """
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1.0]
    return agg_df


def aggregate_diagnostic(scp_codes: dict, agg_df: pd.DataFrame) -> list:
    """
    Map SCP codes to diagnostic superclasses.
    
    For each SCP code in the input dictionary, looks up its diagnostic_class
    in agg_df and returns the unique set of superclasses.
    """
    superclasses = []
    for code in scp_codes.keys():
        if code in agg_df.index:
            superclasses.append(agg_df.loc[code].diagnostic_class)
    return list(set(superclasses))





def normalize_sample(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply per-sample per-channel z-score normalization.
    
    For each channel independently: (signal - mean) / std
    where mean and std are computed over the time dimension.
    """
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (signal - mean) / std


def segment_signal(signal: np.ndarray) -> tuple:
    """
    Split 10s signal into two 5s segments.
    
    Input: (5000, 12) array
    Output: Two (2500, 12) arrays
    """
    segment_a = signal[:SEGMENT_LENGTH]
    segment_b = signal[SEGMENT_LENGTH:]
    return segment_a, segment_b


def process_split(
    df: pd.DataFrame,
    signals: np.ndarray,
    agg_df: pd.DataFrame,
    split_name: str,
) -> tuple:
    """
    Process a single data split (train/val/test).
    
    Steps:
    1. Aggregate diagnostic superclasses for each sample
    2. Remove samples with no labels or multiple labels
    3. Segment signals into 5s pairs
    4. Normalize each segment independently
    
    Returns:
        segment_a: (N, 2500, 12) array
        segment_b: (N, 2500, 12) array
        labels: (N,) integer array
        meta_df: DataFrame with metadata
    """
    print(f"\nProcessing {split_name} split ({len(df)} samples)...")
    
    # Aggregate diagnostic superclasses
    df = df.copy()
    df["diagnostic_superclass"] = df.scp_codes.apply(
        lambda x: aggregate_diagnostic(x, agg_df)
    )
    
    # Filter out samples with no diagnostic labels
    has_labels_mask = df.diagnostic_superclass.apply(lambda x: len(x) > 0)
    no_label_count = (~has_labels_mask).sum()
    print(f"  Samples with no diagnostic labels (removed): {no_label_count}")
    
    # Filter out multi-label samples (keep only single-label samples)
    single_label_mask = df.diagnostic_superclass.apply(lambda x: len(x) == 1)
    multi_label_count = (has_labels_mask & ~single_label_mask).sum()
    print(f"  Multi-label samples (removed): {multi_label_count}")
    
    df = df[single_label_mask]
    valid_indices = np.where(single_label_mask.values)[0]
    signals = signals[valid_indices]
    
    print(f"  Single-label samples retained: {len(df)}")
    
    # Extract the single label from the list
    df["resolved_label"] = df.diagnostic_superclass.apply(lambda x: x[0])
    
    # Final label distribution
    label_dist = Counter(df.resolved_label.tolist())
    print(f"  Label distribution: {dict(label_dist)}")
    
    # Segment and normalize signals
    segments_a = []
    segments_b = []
    
    for i, signal in enumerate(tqdm(signals, desc=f"Segmenting {split_name}")):
        seg_a, seg_b = segment_signal(signal)
        seg_a = normalize_sample(seg_a)
        seg_b = normalize_sample(seg_b)
        segments_a.append(seg_a)
        segments_b.append(seg_b)
    
    segment_a = np.array(segments_a, dtype=np.float32)
    segment_b = np.array(segments_b, dtype=np.float32)
    
    # Convert labels to integers
    labels = np.array(
        [LABEL_TO_INT[label] for label in df.resolved_label],
        dtype=np.int64
    )
    
    # Prepare metadata
    meta_df = pd.DataFrame({
        "ecg_id": df.index,
        "patient_id": df.patient_id.values,
        "age": df.age.values,
        "sex": df.sex.values,
        "label_int": labels,
        "label_str": df.resolved_label.values,
        "scp_codes": df.scp_codes.apply(str).values,
        "strat_fold": df.strat_fold.values,
    })
    
    print(f"  Output shapes: segment_a={segment_a.shape}, segment_b={segment_b.shape}, labels={labels.shape}")
    
    return segment_a, segment_b, labels, meta_df


def clear_old_data(data_dir: Path):
    """Remove existing preprocessed data files."""
    # Clear data/raw CSV files
    raw_dir = data_dir / "raw"
    old_files = [
        "train_signal.csv", "train_meta.csv",
        "valid_signal.csv", "valid_meta.csv",
        "test_signal.csv", "test_meta.csv",
    ]
    for f in old_files:
        filepath = raw_dir / f
        if filepath.exists():
            print(f"Removing {filepath}")
            filepath.unlink()
    
    # Clear data/mmap npy files
    mmap_dir = data_dir / "mmap"
    if mmap_dir.exists():
        for f in mmap_dir.glob("*.npy"):
            print(f"Removing {f}")
            f.unlink()
    
    # Clear data/processed if exists
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {processed_dir}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    ptbxl_dir = data_dir / "raw" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    output_dir = data_dir / "processed"
    
    ptbxl_path = str(ptbxl_dir) + "/"
    
    print("=" * 60)
    print("PTB-XL Preprocessing Pipeline for Contrastive Learning")
    print("=" * 60)
    print(f"PTB-XL path: {ptbxl_path}")
    print(f"Output directory: {output_dir}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Segment length: {SEGMENT_LENGTH} samples (5 seconds)")
    print(f"Label mapping: {LABEL_TO_INT}")
    
    # Clear old data
    print("\n" + "-" * 40)
    print("Clearing old preprocessed data...")
    clear_old_data(data_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n" + "-" * 40)
    print("Loading PTB-XL metadata...")
    Y = pd.read_csv(ptbxl_path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    print(f"Total records: {len(Y)}")
    
    # Load SCP mapping
    agg_df = load_scp_mapping(ptbxl_path)
    print(f"Diagnostic SCP codes: {len(agg_df)}")
    
    # Load raw signals at 500 Hz
    print("\n" + "-" * 40)
    print("Loading raw ECG signals at 500 Hz...")
    X = load_raw_data(Y, ptbxl_path)
    print(f"Signal array shape: {X.shape}")
    
    # Split by strat_fold: 1-8 train, 9 val, 10 test
    print("\n" + "-" * 40)
    print("Splitting data by strat_fold...")
    
    train_mask = Y.strat_fold <= 8
    val_mask = Y.strat_fold == 9
    test_mask = Y.strat_fold == 10
    
    splits = {
        "train": (Y[train_mask], X[train_mask.values]),
        "val": (Y[val_mask], X[val_mask.values]),
        "test": (Y[test_mask], X[test_mask.values]),
    }
    
    # Process each split
    for split_name, (df, signals) in splits.items():
        segment_a, segment_b, labels, meta_df = process_split(
            df, signals, agg_df, split_name
        )
        
        # Save outputs
        print(f"  Saving {split_name} data...")
        np.save(output_dir / f"{split_name}_segment_a.npy", segment_a)
        np.save(output_dir / f"{split_name}_segment_b.npy", segment_b)
        np.save(output_dir / f"{split_name}_labels.npy", labels)
        meta_df.to_csv(output_dir / f"{split_name}_meta.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 60)
    
    # Summary
    print("\nOutput file summary:")
    for f in sorted(output_dir.glob("*")):
        if f.suffix == ".npy":
            arr = np.load(f)
            print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            df = pd.read_csv(f)
            print(f"  {f.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
