#!/usr/bin/env python3
"""
Single GPU Training Script for ECG Contrastive Learning

This script trains the ECGEncoder model using contrastive learning on a single GPU.
It supports TensorBoard logging, and periodic t-SNE visualization
with clustering metrics.

Usage (with session persistence for SSH):
    # Using nohup (simplest)
    nohup python -m src.single_gpu_training --epochs 100 --batch_size 96 > train.log 2>&1 &

    # Using screen
    screen -S ecg_training
    python -m src.single_gpu_training --epochs 100 --batch_size 96
    # Detach with Ctrl+A, D. Reattach with: screen -r ecg_training

    # Using tmux
    tmux new -s ecg_training
    python -m src.single_gpu_training --epochs 100 --batch_size 96
    # Detach with Ctrl+B, D. Reattach with: tmux attach -t ecg_training

To monitor TensorBoard:
    tensorboard --logdir=runs/ --bind_all
"""

import argparse
import math
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import ecgmentations as E
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.augmentor import DualAugmenter
from src.checkpoint_manager import CheckpointManager
from src.datasets import ECGContrastiveTrainDataset, ECGDataset
from src.dual_view_transformer import ECGEncoder, ECGModelConfig


# Augmentation Classes
class RandomLeadMasking:
    """Randomly masks (zeros out) each ECG lead independently with probability p."""

    def __init__(self, p: float = 0.5):
        self.mask_prob = p

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        ecg_augmented = ecg
        num_channels = ecg_augmented.shape[1]
        mask = torch.rand(num_channels) < self.mask_prob
        ecg_augmented[:, mask] = 0.0
        return ecg_augmented


class GaussianNoise:
    """Adds Gaussian noise independently to each ECG lead with probability prob."""

    def __init__(self, std: float = 0.05, prob: float = 1.0) -> None:
        self.std = std
        self.prob = prob

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        num_channels = ecg.shape[1]
        mask = (torch.rand(num_channels, device=ecg.device) < self.prob).view(
            1, num_channels
        )
        noise = torch.randn_like(ecg) * self.std
        return ecg + noise * mask


class PerChannelGaussianNoise:
    """
    Adds Gaussian noise with channel-specific random standard deviations.
    Each channel receives noise independently with probability prob.
    """

    def __init__(
        self, std_min: float = 0.01, std_max: float = 0.1, prob: float = 0.5
    ) -> None:
        self.std_min = std_min
        self.std_max = std_max
        self.prob = prob

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        num_channels = ecg.shape[1]
        channel_stds = (
            torch.rand(num_channels, device=ecg.device) * (self.std_max - self.std_min)
            + self.std_min
        )
        mask = (torch.rand(num_channels, device=ecg.device) < self.prob).view(
            1, num_channels
        )
        channel_stds = channel_stds.view(1, num_channels)
        noise = torch.randn_like(ecg) * channel_stds
        return ecg + noise * mask


class ComposeAugmentations:
    """Applies a sequence of augmentations to ECG signals in order."""

    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            ecg = aug(ecg)
        return ecg






class GracefulKiller:
    """Handle SIGINT and SIGTERM for graceful shutdown with checkpoint saving."""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print(
            f"\n[!] Received signal {signum}. Will save checkpoint and exit after current epoch..."
        )
        self.kill_now = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single GPU training for ECG Contrastive Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=350, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=80, help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=12,
        help="Micro-batches to accumulate before optimizer step",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for NTXentLoss"
    )
    parser.add_argument(
        "--num_workers", type=int, default=20, help="Number of data loading workers"
    )

    # Model configuration
    parser.add_argument(
        "--encoder_embed_dim", type=int, default=512, help="Encoder embedding dimension"
    )
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument(
        "--time_token_dim", type=int, default=1024, help="Time token dimension"
    )
    parser.add_argument(
        "--channel_token_dim", type=int, default=1024, help="Channel token dimension"
    )
    parser.add_argument(
        "--time_heads", type=int, default=4, help="Number of time attention heads"
    )
    parser.add_argument(
        "--channel_heads", type=int, default=4, help="Number of channel attention heads"
    )
    parser.add_argument(
        "--time_layers", type=int, default=4, help="Number of time transformer layers"
    )
    parser.add_argument(
        "--channel_layers",
        type=int,
        default=4,
        help="Number of channel transformer layers",
    )
    parser.add_argument(
        "--ff_multiplier", type=int, default=6, help="Feedforward dimension multiplier"
    )
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument(
        "--channel_token_dropout",
        type=float,
        default=0.25,
        help="Probability of dropping the entire channel token stream during training",
    )
    parser.add_argument(
        "--fusion_residual_dropout",
        type=float,
        default=0.25,
        help="Dropout applied to fusion residual path",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=512,
        help="Projection head output dimension",
    )
    parser.add_argument(
        "--fusion_hidden_dim",
        type=int,
        default=4096,
        help="Fusion head hidden dimension",
    )
    parser.add_argument(
        "--fusion_dropout", type=float, default=0.3, help="Fusion head dropout"
    )

    # Data type
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model data type (bf16 requires Ampere+ GPU)",
    )

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../models/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="../runs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Visualization and metrics
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=1,
        help="Interval (epochs) for t-SNE visualization and clustering metrics",
    )
    parser.add_argument(
        "--tsne_samples",
        type=int,
        default=1000,
        help="Number of samples to use for t-SNE visualization",
    )

    # Checkpointing
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep",
    )

    # Scheduler (linear warmup + OneCycleLR)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=20,
        help="Linear warmup epochs before OneCycleLR",
    )
    parser.add_argument(
        "--max_lr", type=float, default=6e-3, help="OneCycleLR maximum learning rate"
    )
    parser.add_argument(
        "--onecycle_pct_start",
        type=float,
        default=0.1,
        help="OneCycleLR warmup percentage",
    )
    parser.add_argument(
        "--onecycle_div_factor",
        type=float,
        default=30.0,
        help="OneCycleLR div_factor (initial lr = max_lr/div_factor)",
    )
    parser.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=100.0,
        help="OneCycleLR final_div_factor",
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]


def create_model(args: argparse.Namespace, device: torch.device) -> ECGEncoder:
    """Create and return the ECGEncoder model."""
    
    # NOTE: I hardcoded a more parameter-efficient model configuration here instead of using args.
    config = ECGModelConfig(
        sequence_length=2500,  
        num_channels=12,  
        encoder_embed_dim=384, 
        d_model=512,  
        time_token_dim=256, 
        channel_token_dim=256,  
        time_heads=6,   
        channel_heads=6,  
        time_layers=4,  
        channel_layers=4,  
        ff_multiplier=4,   
        dropout=0.2,  
        projection_dim=256,  
        time_conv_kernel_size=3,  
        channel_conv_kernel_size=5,  
        channel_conv_stride=2,  
        channel_token_dropout=args.channel_token_dropout,
        fusion_residual_dropout=args.fusion_residual_dropout,
        dtype=torch.bfloat16, 
        fusion_hidden_dim=1024,  
        fusion_dropout=0.2,  
        use_flash_attention=True,  
    )
    model = ECGEncoder(config)
    model = model.to(device)
    return model


def create_datasets(
    args: argparse.Namespace, device: torch.device
) -> Tuple[ECGContrastiveTrainDataset, ECGContrastiveTrainDataset, ECGDataset]:
    """Create training, validation, and evaluation datasets."""
    data_dir = Path(args.data_dir)

    augmentation_pool = [
        ComposeAugmentations(
            [PerChannelGaussianNoise(prob=0.01), RandomLeadMasking(p=0.5)]
        ),
    ]
    dual_augmenter = DualAugmenter(augmentation_pool=augmentation_pool)

    train_dataset = ECGContrastiveTrainDataset(
        segment_a_path=data_dir / "train_segment_a.npy",
        segment_b_path=data_dir / "train_segment_b.npy",
        labels_path=data_dir / "train_labels.npy",
        dual_augmenter=dual_augmenter,
        device=device,
    )

    val_dataset = ECGContrastiveTrainDataset(
        segment_a_path=data_dir / "val_segment_a.npy",
        segment_b_path=data_dir / "val_segment_b.npy",
        labels_path=data_dir / "val_labels.npy",
        dual_augmenter=dual_augmenter,
        device=device,
    )

    # ECGDataset for embedding extraction (no augmentation)
    train_eval_dataset = ECGDataset(
        segment_a_path=data_dir / "train_segment_a.npy",
        segment_b_path=data_dir / "train_segment_b.npy",
        labels_path=data_dir / "train_labels.npy",
    )

    return train_dataset, val_dataset, train_eval_dataset


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from the model."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            representations, _, _, _ = model(signals)
            all_embeddings.append(representations.to(torch.float32).cpu().numpy())
            all_labels.extend(
                labels.cpu().numpy() if torch.is_tensor(labels) else labels
            )

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels_array = np.array(all_labels, dtype=int)
    return embeddings, labels_array


def compute_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    return {
        "silhouette_score": silhouette_score(embeddings, labels),
        "davies_bouldin_score": davies_bouldin_score(embeddings, labels),
        "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
    }


def create_tsne_figure(
    embeddings: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    metrics: Dict[str, float],
) -> plt.Figure:
    """Create a t-SNE visualization figure."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=20,
    )

    unique_labels = np.unique(labels)
    label_names = {0: "NORM", 1: "MI", 2: "STTC", 3: "HYP", 4: "CD"}
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scatter.cmap(scatter.norm(label)),
            markersize=8,
            label=label_names.get(label, f"Class {label}"),
        )
        for label in unique_labels
    ]
    ax.legend(
        handles=handles,
        title="Diagnostic Class",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title(
        f"Epoch {epoch} | Silhouette: {metrics['silhouette_score']:.3f} | "
        f"Davies-Bouldin: {metrics['davies_bouldin_score']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter = None,
    global_step: int = 0,
    grad_accum_steps: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[float, int]:
    """Train for one epoch and return average loss and updated global step."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_count = 0

    for batch_idx, (aug1, aug2) in enumerate(train_loader):
        aug1, aug2 = aug1.to(device), aug2.to(device)

        _, proj1, time1, chan1 = model(aug1)
        _, proj2, time2, chan2 = model(aug2)

        main_loss = loss_fn(proj1, proj2)
        aux_time_loss = loss_fn(time1, time2)
        aux_channel_loss = loss_fn(chan1, chan2)
        raw_loss = main_loss + aux_time_loss + aux_channel_loss
        loss = raw_loss / grad_accum_steps
        loss.backward()

        accum_loss += raw_loss.item()
        accum_count += 1
        total_loss += raw_loss.item()

        is_update_step = (accum_count == grad_accum_steps) or (
            batch_idx == num_batches - 1
        )
        if not is_update_step:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        branch_time_norm = None
        branch_channel_norm = None
        if (
            getattr(model, "_last_fused_time", None) is not None
            and model._last_fused_time.grad is not None
        ):
            branch_time_norm = model._last_fused_time.grad.norm().item()
        if (
            getattr(model, "_last_fused_channel", None) is not None
            and model._last_fused_channel.grad is not None
        ):
            branch_channel_norm = model._last_fused_channel.grad.norm().item()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        global_step += 1

        if writer is not None:
            step_loss = accum_loss / accum_count
            writer.add_scalar("Loss_Batch/train", step_loss, global_step)
            writer.add_scalar("Loss_Batch/main", main_loss.item(), global_step)
            writer.add_scalar("Loss_Batch/aux_time", aux_time_loss.item(), global_step)
            writer.add_scalar(
                "Loss_Batch/aux_channel", aux_channel_loss.item(), global_step
            )
            writer.add_scalar("GradNorm_Batch/train", grad_norm.item(), global_step)
            if branch_time_norm is not None:
                writer.add_scalar("GradNorm_Batch/time", branch_time_norm, global_step)
            if branch_channel_norm is not None:
                writer.add_scalar(
                    "GradNorm_Batch/channel", branch_channel_norm, global_step
                )

        accum_loss = 0.0
        accum_count = 0

    return total_loss / num_batches, global_step


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter = None,
    val_step: int = 0,
) -> Tuple[float, int]:
    """Validate for one epoch and return average loss and updated val step."""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for aug1, aug2 in val_loader:
            aug1, aug2 = aug1.to(device), aug2.to(device)

            _, proj1, time1, chan1 = model(aug1)
            _, proj2, time2, chan2 = model(aug2)

            main_loss = loss_fn(proj1, proj2)
            aux_time_loss = loss_fn(time1, time2)
            aux_channel_loss = loss_fn(chan1, chan2)
            loss = main_loss + aux_time_loss + aux_channel_loss

            batch_loss = loss.item()
            total_loss += batch_loss
            val_step += 1

            # Log per-batch validation loss to TensorBoard
            if writer is not None:
                writer.add_scalar("Loss_Batch/val", batch_loss, val_step)
                writer.add_scalar("Loss_Batch/val_main", main_loss.item(), val_step)
                writer.add_scalar(
                    "Loss_Batch/val_aux_time", aux_time_loss.item(), val_step
                )
                writer.add_scalar(
                    "Loss_Batch/val_aux_channel", aux_channel_loss.item(), val_step
                )

    return total_loss / num_batches, val_step


def main():
    args = parse_args()
    killer = GracefulKiller()

    # Setup device
    if not torch.cuda.is_available():
        print("[!] CUDA not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"[*] Using device: {device}")
    print(f"[*] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[*] BF16 supported: {torch.cuda.is_bf16_supported()}")

    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        print("[!] BF16 not supported on this GPU. Falling back to FP16.")
        args.dtype = "fp16"

    # Create model
    print(f"\n[*] Creating model with dtype={args.dtype}...")
    model = create_model(args, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total parameters: {total_params:,}")
    print(f"[*] Trainable parameters: {trainable_params:,}")

    # Create datasets
    print("\n[*] Creating datasets...")
    train_dataset, val_dataset, train_eval_dataset = create_datasets(args, device)
    print(f"[*] Train samples: {len(train_dataset)}")
    print(f"[*] Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create subset of training data for t-SNE visualization
    # Using num_workers=0 to avoid multiprocessing issues with memory-mapped arrays
    tsne_indices = np.random.choice(
        len(train_eval_dataset),
        size=min(args.tsne_samples, len(train_eval_dataset)),
        replace=False,
    )
    tsne_subset = Subset(train_eval_dataset, tsne_indices)
    tsne_loader = DataLoader(
        tsne_subset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Optimizer and loss
    base_lr = args.max_lr / args.onecycle_div_factor
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    loss_fn = SelfSupervisedLoss(NTXentLoss(temperature=args.temperature))

    # LR scheduler: linear warmup (to base_lr) then OneCycleLR for the remaining epochs
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    if warmup_steps >= total_steps:
        warmup_steps = max(total_steps - 1, 1)

    warmup_scheduler = (
        LambdaLR(optimizer, lr_lambda=lambda step: step / max(1, warmup_steps))
        if warmup_steps > 0
        else None
    )

    onecycle_epochs = max(args.epochs - args.warmup_epochs, 1)
    onecycle_scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=onecycle_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.onecycle_pct_start,
        div_factor=args.onecycle_div_factor,
        final_div_factor=args.onecycle_final_div_factor,
        anneal_strategy="cos",
    )

    if warmup_scheduler is not None:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, onecycle_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = onecycle_scheduler

    # Checkpoint manager (using silhouette score - higher is better)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=args.keep_last_n,
        save_best=True,
        mode="max",
    )

    # TensorBoard writer
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = Path(args.tensorboard_dir) / f"ecg_contrastive_{run_name}"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"\n[*] TensorBoard logs: {tensorboard_dir}")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_silhouette = float("-inf")
    loss_history = []
    val_loss_history = []
    silhouette_history = []

    if args.resume:
        print(f"\n[*] Resuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_manager.load_checkpoint(args.resume, model, optimizer)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_silhouette = checkpoint.get("best_silhouette", float("-inf"))
        loss_history = checkpoint.get("loss_history", [])
        val_loss_history = checkpoint.get("val_loss_history", [])
        silhouette_history = checkpoint.get("silhouette_history", [])
        print(
            f"[*] Resumed from epoch {checkpoint['epoch']}, best_silhouette={best_silhouette:.4f}"
        )

    # Log hyperparameters
    writer.add_text(
        "hyperparameters",
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"temperature={args.temperature}, dtype={args.dtype}, "
        f"encoder_embed_dim={args.encoder_embed_dim}, d_model={args.d_model}, "
        f"dropout={args.dropout}, projection_dim={args.projection_dim}",
    )

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Temperature: {args.temperature}, Dtype: {args.dtype}")
    print(
        f"t-SNE visualization every {args.vis_interval} epochs with {args.tsne_samples} samples"
    )
    print("=" * 80 + "\n")

    # Global step counters for per-batch logging
    global_step = 0
    val_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if killer.kill_now:
            print("\n[!] Graceful shutdown requested. Saving checkpoint...")
            checkpoint_manager.save_checkpoint(
                epoch=epoch - 1,
                model=model,
                optimizer=optimizer,
                metric=silhouette_history[-1] if silhouette_history else float("-inf"),
                config=None,
                additional_info={
                    "args": vars(args),
                    "loss_history": loss_history,
                    "val_loss_history": val_loss_history,
                    "silhouette_history": silhouette_history,
                    "best_silhouette": best_silhouette,
                    "scheduler_state_dict": scheduler.state_dict(),
                },
            )
            print("[*] Checkpoint saved. Exiting.")
            break

        # Training
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            args.epochs,
            writer,
            global_step,
            args.grad_accum_steps,
            scheduler,
        )
        loss_history.append(train_loss)

        # Validation
        val_loss, val_step = validate_epoch(
            model, val_loader, loss_fn, device, epoch, args.epochs, writer, val_step
        )
        val_loss_history.append(val_loss)

        # Log epoch-level loss to TensorBoard (separate graphs from batch-level)
        writer.add_scalar("Loss_Epoch/train", train_loss, epoch)
        writer.add_scalar("Loss_Epoch/val", val_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Compute silhouette score for checkpointing criterion
        embeddings, labels = extract_embeddings(model, tsne_loader, device)
        metrics = compute_clustering_metrics(embeddings, labels)
        current_silhouette = metrics["silhouette_score"]
        silhouette_history.append(current_silhouette)

        # Log clustering metrics
        writer.add_scalar(
            "Clustering/silhouette_score", metrics["silhouette_score"], epoch
        )
        writer.add_scalar(
            "Clustering/davies_bouldin_score", metrics["davies_bouldin_score"], epoch
        )
        writer.add_scalar(
            "Clustering/calinski_harabasz_score",
            metrics["calinski_harabasz_score"],
            epoch,
        )

        # Print epoch summary
        print(
            f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Silhouette: {current_silhouette:.4f}"
        )

        # Check if this is the best silhouette score
        is_best = current_silhouette > best_silhouette
        if is_best:
            best_silhouette = current_silhouette
            print("  ✓ New best silhouette score!")

        # Save checkpoint (using silhouette score as the metric)
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metric=current_silhouette,
            config=None,
            additional_info={
                "args": vars(args),
                "loss_history": loss_history,
                "val_loss_history": val_loss_history,
                "silhouette_history": silhouette_history,
                "best_silhouette": best_silhouette,
                "scheduler_state_dict": scheduler.state_dict(),
            },
        )

        # t-SNE visualization every vis_interval epochs
        if epoch % args.vis_interval == 0:
            print(f"\n[*] Creating t-SNE visualization (epoch {epoch})...")

            # Create and log t-SNE figure
            fig = create_tsne_figure(embeddings, labels, epoch, metrics)
            writer.add_figure("t-SNE/embeddings", fig, epoch)
            plt.close(fig)

            print(f"  Silhouette Score:        {metrics['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Score:    {metrics['davies_bouldin_score']:.4f}")
            print(
                f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}"
            )

        print()

    # Final summary
    writer.close()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final Training Loss: {loss_history[-1]:.4f}")
    print(f"Final Validation Loss: {val_loss_history[-1]:.4f}")
    print(f"Final Silhouette Score: {silhouette_history[-1]:.4f}")
    print(f"Best Silhouette Score: {best_silhouette:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
