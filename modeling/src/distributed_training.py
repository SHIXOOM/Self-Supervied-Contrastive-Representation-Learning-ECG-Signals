"""
Distributed training script for ECG contrastive learning using PyTorch DDP.

Feature parity with single GPU pipeline: same model config, augmentations,
optimizer/scheduler (warmup + OneCycleLR), logging, clustering metrics, and
T-SNE. Uses global all-gather NT-Xent for stronger negatives across GPUs.

Launch via SLURM:
    sbatch slurm_train.sbatch

Environment variables (set by SLURM or torchrun export):
    SLURM_PROCID: Global rank of the process
    SLURM_LOCALID: Local rank within the node
    SLURM_NTASKS: Total number of processes (world_size)
    SLURM_NODEID: Node ID
    MASTER_ADDR: Address of the master node
    MASTER_PORT: Port for distributed communication
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import math
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure package imports mirror single GPU script
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.augmentor import DualAugmenter
from src.datasets import ECGContrastiveTrainDataset, ECGDataset
from src.dual_view_transformer import ECGEncoder, ECGModelConfig


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training run."""

    # Paths
    data_dir: str = "../data/processed"
    checkpoint_dir: str = "../models/checkpoints"
    tensorboard_dir: str = "../runs"

    # Training
    epochs: int = 500
    batch_size: int = 92  # per-GPU batch
    temperature: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    num_workers: int = 4
    val_frequency: int = 1
    early_stop_patience: int = 4

    # Model
    channel_token_dropout: float = 0.35
    fusion_residual_dropout: float = 0.25
    dropout: float = 0.3
    encoder_embed_dim: int = 384
    d_model: int = 512
    time_token_dim: int = 128
    channel_token_dim: int = 256
    time_heads: int = 6
    channel_heads: int = 6
    time_layers: int = 4
    channel_layers: int = 4
    ff_multiplier: int = 5
    projection_dim: int = 256
    fusion_hidden_dim: int = 2048
    fusion_dropout: float = 0.3
    dtype: str = "bf16"

    # Scheduler (warmup + OneCycle)
    warmup_epochs: int = 15
    max_lr: float = 5e-4
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 25
    onecycle_final_div_factor: float = 100

    # Visualization
    vis_interval: int = 1
    tsne_samples: int = 1000

    # Checkpoints
    keep_last_n: int = 2
    resume: Optional[str] = None


def parse_args() -> DistributedTrainingConfig:
    parser = argparse.ArgumentParser(
        description="Distributed training for ECG contrastive learning (DDP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=92, help="Per-GPU batch size")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_frequency", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=4)

    # Model configuration
    parser.add_argument("--encoder_embed_dim", type=int, default=384)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--time_token_dim", type=int, default=128)
    parser.add_argument("--channel_token_dim", type=int, default=256)
    parser.add_argument("--time_heads", type=int, default=6)
    parser.add_argument("--channel_heads", type=int, default=6)
    parser.add_argument("--time_layers", type=int, default=4)
    parser.add_argument("--channel_layers", type=int, default=4)
    parser.add_argument("--ff_multiplier", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--channel_token_dropout", type=float, default=0.35)
    parser.add_argument("--fusion_residual_dropout", type=float, default=0.25)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--fusion_hidden_dim", type=int, default=2048)
    parser.add_argument("--fusion_dropout", type=float, default=0.3)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # Paths
    parser.add_argument("--data_dir", type=str, default="../data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="../models/checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, default="../runs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--keep_last_n", type=int, default=2)

    # Visualization and metrics
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--tsne_samples", type=int, default=1000)

    # Scheduler
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--max_lr", type=float, default=5e-4)
    parser.add_argument("--onecycle_pct_start", type=float, default=0.1)
    parser.add_argument("--onecycle_div_factor", type=float, default=25.0)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=100.0)

    args = parser.parse_args()
    return DistributedTrainingConfig(**vars(args))


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return dtype_map[dtype_str]


class GlobalBatchNTXentLoss(nn.Module):
    """
    NT-Xent loss that gathers embeddings from all distributed ranks.

    Collects embeddings across all GPUs via all_gather, then computes
    the contrastive loss using the full global batch as negative samples.
    This ensures that with 2 GPUs × 64 samples = 128 total samples,
    each sample has 127 negatives instead of just 63.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute global NT-Xent loss across all ranks.

        Args:
            z1: Embeddings from view 1, shape (local_batch, embed_dim), L2-normalized
            z2: Embeddings from view 2, shape (local_batch, embed_dim), L2-normalized

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics (accuracy, etc.)
        """
        local_batch_size = z1.shape[0]

        if dist.is_initialized():
            world_size = dist.get_world_size()
            z1_gathered = self._all_gather_with_grad(z1)
            z2_gathered = self._all_gather_with_grad(z2)
        else:
            world_size = 1
            z1_gathered = z1
            z2_gathered = z2

        global_batch_size = local_batch_size * world_size

        # Concatenate all embeddings: [z1_all, z2_all] -> (2 * global_batch, embed_dim)
        embeddings = torch.cat([z1_gathered, z2_gathered], dim=0)

        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create labels: positive pairs are at positions (i, i + global_batch_size)
        labels = torch.arange(global_batch_size, device=z1.device)
        labels = torch.cat([labels + global_batch_size, labels], dim=0)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * global_batch_size, device=z1.device, dtype=torch.bool)
        similarity.masked_fill_(mask, float("-inf"))

        loss = F.cross_entropy(similarity, labels)

        # Compute accuracy for monitoring
        with torch.no_grad():
            predictions = similarity.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

        metrics = {"contrastive_accuracy": accuracy, "global_batch_size": global_batch_size}

        return loss, metrics

    def _all_gather_with_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather tensors while preserving gradients for the local shard."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        tensors_gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gathered, tensor)

        # Replace the gathered tensor at current rank with the original
        # to preserve gradients (all_gather doesn't propagate gradients)
        tensors_gathered[rank] = tensor

        return torch.cat(tensors_gathered, dim=0)


class DistributedCheckpointManager:
    """
    Checkpoint manager for distributed training.

    Only rank 0 saves checkpoints to avoid redundant writes.
    All ranks can load checkpoints for resuming training.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        rank: int = 0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.rank = rank
        self.best_loss = float("inf")

        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        loss: float,
        config: DistributedTrainingConfig,
        additional_info: Optional[Dict] = None,
    ) -> None:
        """Save checkpoint (rank 0 only)."""
        if self.rank != 0:
            return

        # Unwrap DDP module
        model_state = (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "config": asdict(config),
            "best_loss": self.best_loss,
        }

        if additional_info:
            checkpoint.update(additional_info)

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"[Rank 0] Saved checkpoint: {checkpoint_path}")

        # Save best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"[Rank 0] New best model with loss: {loss:.4f}")

        # Save latest for easy resume
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond keep_last_n."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        for old_ckpt in checkpoints[: -self.keep_last_n]:
            old_ckpt.unlink()
            print(f"[Rank 0] Removed old checkpoint: {old_ckpt.name}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        map_location: Optional[str] = None,
    ) -> Dict:
        """Load checkpoint on all ranks."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.pt"

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Handle DDP-wrapped model
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(
            f"[Rank {self.rank}] Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"with loss {checkpoint['loss']:.4f}"
        )
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest = self.checkpoint_dir / "latest.pt"
        return latest if latest.exists() else None


class DistributedMetricTracker:
    """
    Aggregates metrics across distributed ranks and logs to TensorBoard.

    Only rank 0 writes to TensorBoard. Metrics are reduced via all_reduce
    before logging to show global training statistics.
    """

    def __init__(
        self,
        tensorboard_dir: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.writer = None

        if self.rank == 0:
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
        reduce: bool = True,
    ) -> float:
        """
        Log a scalar metric, optionally reducing across ranks.

        Args:
            tag: Metric name for TensorBoard
            value: Local value
            step: Global step (epoch or iteration)
            reduce: If True, compute mean across all ranks

        Returns:
            The (possibly reduced) value
        """
        if reduce and dist.is_initialized() and self.world_size > 1:
            tensor = torch.tensor(value, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            value = tensor.item() / self.world_size

        if self.rank == 0 and self.writer:
            self.writer.add_scalar(tag, value, step)

        return value

    def log_learning_rate(self, lr: float, step: int) -> None:
        """Log current learning rate (no reduction needed)."""
        if self.rank == 0 and self.writer:
            self.writer.add_scalar("train/learning_rate", lr, step)

    def log_throughput(self, samples_per_sec: float, step: int) -> None:
        """Log training throughput across all GPUs."""
        # Sum throughput across ranks (total samples/sec)
        if dist.is_initialized() and self.world_size > 1:
            tensor = torch.tensor(samples_per_sec, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            samples_per_sec = tensor.item()

        if self.rank == 0 and self.writer:
            self.writer.add_scalar("perf/throughput_samples_per_sec", samples_per_sec, step)

    def log_memory(self, step: int) -> None:
        """Log GPU memory usage."""
        if self.rank == 0 and self.writer:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.writer.add_scalar("perf/gpu_memory_allocated_gb", allocated, step)
            self.writer.add_scalar("perf/gpu_memory_reserved_gb", reserved, step)

    def log_tsne_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray],
        step: int,
    ) -> None:
        """Log T-SNE visualization of embeddings to TensorBoard."""
        if self.rank != 0 or self.writer is None:
            return

        print(f"[Rank 0] Computing T-SNE for {len(embeddings)} samples...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Log as matplotlib figure
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        if labels is not None:
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap="tab10",
                alpha=0.6,
                s=10,
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10)

        ax.set_title(f"T-SNE Embedding Visualization (Epoch {step})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        self.writer.add_figure("embeddings/tsne", fig, step)
        plt.close(fig)
        print(f"[Rank 0] T-SNE visualization logged for epoch {step}")

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed process group from SLURM environment variables.

    Returns:
        rank: Global rank of this process
        local_rank: Local rank within the node (for GPU assignment)
        world_size: Total number of processes
    """
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Initialize NCCL backend
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    # Set device for this process
    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] Initialized: local_rank={local_rank}, "
        f"world_size={world_size}, master={master_addr}:{master_port}"
    )

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class RandomLeadMasking:
    """Randomly masks (zeros out) each ECG lead independently with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.mask_prob = p

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand(ecg.shape[1], device=ecg.device) < self.mask_prob).view(1, ecg.shape[1])
        return ecg * (~mask)


class PerChannelGaussianNoise:
    """Adds Gaussian noise with channel-specific std; applied per-channel with probability."""

    def __init__(self, std_min: float = 0.01, std_max: float = 0.1, prob: float = 0.5) -> None:
        self.std_min = std_min
        self.std_max = std_max
        self.prob = prob

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        num_channels = ecg.shape[1]
        channel_stds = torch.rand(num_channels, device=ecg.device) * (self.std_max - self.std_min) + self.std_min
        mask = (torch.rand(num_channels, device=ecg.device) < self.prob).view(1, num_channels)
        noise = torch.randn_like(ecg) * channel_stds.view(1, num_channels)
        return ecg + noise * mask


class ComposeAugmentations:
    """Compose augmentations in order."""

    def __init__(self, augmentations: List) -> None:
        self.augmentations = augmentations

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            ecg = aug(ecg)
        return ecg


def create_augmentation_pool() -> List:
    """Create augmentation pool for contrastive learning."""
    return [ComposeAugmentations([PerChannelGaussianNoise(prob=0.5), RandomLeadMasking(p=0.5)])]


class GracefulKiller:
    """Handle SIGINT/SIGTERM and request graceful exit after current epoch."""

    kill_now = False

    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame) -> None:  # type: ignore[override]
        print(f"\n[!] Received signal {signum}. Will save checkpoint and exit after current epoch...")
        self.kill_now = True


def gather_embeddings_for_tsne(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    device: torch.device,
    rank: int,
    world_size: int,
    autocast_dtype: Optional[torch.dtype],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Gather embeddings (and labels) from all ranks for T-SNE/clustering."""
    model.eval()
    local_embeddings: List[torch.Tensor] = []
    local_labels: List[torch.Tensor] = []
    samples_collected = 0
    samples_per_rank = max(1, num_samples // world_size)

    with torch.no_grad():
        for signals, labels in dataloader:
            if samples_collected >= samples_per_rank:
                break

            signals = signals.to(device, non_blocking=True)
            with autocast(dtype=autocast_dtype) if autocast_dtype else nullcontext():
                representations, _ = model(signals)

            local_embeddings.append(representations.float())
            labels_tensor = labels if torch.is_tensor(labels) else torch.tensor(labels)
            local_labels.append(labels_tensor.to(device=device, dtype=torch.long))
            samples_collected += signals.shape[0]

    if not local_embeddings:
        model.train()
        return None, None

    local_embeddings = torch.cat(local_embeddings, dim=0)[:samples_per_rank]
    local_labels_tensor = torch.cat(local_labels, dim=0)[:samples_per_rank]

    # Gather embeddings on rank 0
    if world_size > 1:
        gathered_embeddings = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(local_labels_tensor) for _ in range(world_size)]
        dist.gather(local_embeddings, gathered_embeddings if rank == 0 else None, dst=0)
        dist.gather(local_labels_tensor, gathered_labels if rank == 0 else None, dst=0)

        if rank == 0:
            all_embeddings = torch.cat(gathered_embeddings, dim=0).cpu().numpy()
            all_labels = torch.cat(gathered_labels, dim=0).cpu().numpy()
            model.train()
            return all_embeddings, all_labels
        model.train()
        return None, None

    model.train()
    return local_embeddings.cpu().numpy(), local_labels_tensor.cpu().numpy()


class DistributedTrainer:
    """Main distributed training orchestrator."""

    def __init__(self, config: DistributedTrainingConfig) -> None:
        self.config = config

        # Setup distributed environment
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.autocast_dtype = get_dtype(self.config.dtype)

        if self.config.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            if self.rank == 0:
                print("[!] BF16 not supported on this GPU. Falling back to FP16.")
            self.config.dtype = "fp16"
            self.autocast_dtype = torch.float16
        if self.config.dtype == "fp32":
            self.autocast_dtype = None

        # Initialize components
        self.checkpoint_manager = DistributedCheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_last_n=config.keep_last_n,
            rank=self.rank,
        )

        self.metric_tracker = DistributedMetricTracker(
            tensorboard_dir=config.tensorboard_dir,
            rank=self.rank,
            world_size=self.world_size,
        )

        self.loss_fn = GlobalBatchNTXentLoss(temperature=config.temperature)

        # Training state
        self.epochs_without_improvement = 0

    def _create_model(self) -> nn.Module:
        """Create and wrap model with DDP."""
        model_config = ECGModelConfig(
            sequence_length=2500,
            num_channels=12,
            encoder_embed_dim=self.config.encoder_embed_dim,
            d_model=self.config.d_model,
            time_token_dim=self.config.time_token_dim,
            channel_token_dim=self.config.channel_token_dim,
            time_heads=self.config.time_heads,
            channel_heads=self.config.channel_heads,
            time_layers=self.config.time_layers,
            channel_layers=self.config.channel_layers,
            ff_multiplier=self.config.ff_multiplier,
            dropout=self.config.dropout,
            projection_dim=self.config.projection_dim,
            time_conv_kernel_size=3,
            channel_conv_kernel_size=5,
            channel_conv_stride=2,
            channel_token_dropout=self.config.channel_token_dropout,
            fusion_residual_dropout=self.config.fusion_residual_dropout,
            dtype=self.autocast_dtype if self.autocast_dtype else torch.float32,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
            fusion_dropout=self.config.fusion_dropout,
            use_flash_attention=True,
        )

        model = ECGEncoder(model_config).to(self.device)

        # Wrap with DDP
        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

        if self.rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def _create_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
        """Create distributed data loaders and t-SNE subset loader."""
        augmentation_pool = create_augmentation_pool()
        dual_augmenter = DualAugmenter(augmentation_pool)

        data_dir = Path(self.config.data_dir)

        train_dataset = ECGContrastiveTrainDataset(
            segment_a_path=data_dir / "train_segment_a.npy",
            segment_b_path=data_dir / "train_segment_b.npy",
            labels_path=data_dir / "train_labels.npy",
            dual_augmenter=dual_augmenter,
            device=self.device,
        )

        val_dataset = ECGContrastiveTrainDataset(
            segment_a_path=data_dir / "val_segment_a.npy",
            segment_b_path=data_dir / "val_segment_b.npy",
            labels_path=data_dir / "val_labels.npy",
            dual_augmenter=dual_augmenter,
            device=self.device,
        )

        train_eval_dataset = ECGDataset(
            segment_a_path=data_dir / "train_segment_a.npy",
            segment_b_path=data_dir / "train_segment_b.npy",
            labels_path=data_dir / "train_labels.npy",
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Deterministic subset for t-SNE and clustering (unaugmented)
        rng = np.random.default_rng(42)
        tsne_indices = rng.choice(
            len(train_eval_dataset),
            size=min(self.config.tsne_samples, len(train_eval_dataset)),
            replace=False,
        )
        tsne_subset = Subset(train_eval_dataset, tsne_indices)
        tsne_loader = DataLoader(
            tsne_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        if self.rank == 0:
            print(f"Train dataset: {len(train_dataset)} samples")
            print(f"Val dataset: {len(val_dataset)} samples")
            print(
                f"Effective batch size: {self.config.batch_size * self.world_size}"
            )

        return train_loader, val_loader, tsne_loader, train_sampler, val_sampler

    def _create_optimizer_and_scheduler(
        self, model: nn.Module
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Create optimizer with warmup + OneCycle schedule."""
        base_lr = self.config.max_lr / self.config.onecycle_div_factor
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.config.grad_accum_steps))
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = min(self.config.warmup_epochs * steps_per_epoch, max(total_steps - 1, 1))

        warmup_scheduler = (
            LambdaLR(optimizer, lr_lambda=lambda step: step / max(1, warmup_steps))
            if warmup_steps > 0
            else None
        )

        onecycle_epochs = max(self.config.epochs - self.config.warmup_epochs, 1)
        onecycle_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.max_lr,
            epochs=onecycle_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config.onecycle_pct_start,
            div_factor=self.config.onecycle_div_factor,
            final_div_factor=self.config.onecycle_final_div_factor,
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

        return optimizer, scheduler

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
    ) -> Tuple[float, float, int, int]:
        """Run one training epoch; returns avg loss, throughput, global_step, tb_step."""
        model.train()

        total_loss = 0.0
        total_samples = 0
        epoch_start = time.time()
        global_step = 0
        tb_step = 0
        last_loss_value = None

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(self.rank != 0))

        for batch_idx, (aug_a, aug_b) in enumerate(progress):
            aug_a = aug_a.to(self.device, non_blocking=True)
            aug_b = aug_b.to(self.device, non_blocking=True)

            with autocast(dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext():
                _, proj_a = model(aug_a)
                _, proj_b = model(aug_b)
                loss, metrics = self.loss_fn(proj_a, proj_b)

            loss_to_backprop = loss / self.config.grad_accum_steps
            loss_to_backprop.backward()

            last_loss_value = loss.item()

            total_loss += loss.item() * aug_a.shape[0]
            total_samples += aug_a.shape[0]
            global_step += 1

            take_step = (batch_idx + 1) % self.config.grad_accum_steps == 0

            branch_time_norm = None
            branch_channel_norm = None
            if take_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.grad_clip
                )

                encoder = model.module if hasattr(model, "module") else model
                if getattr(encoder, "_last_fused_time", None) is not None and encoder._last_fused_time.grad is not None:
                    branch_time_norm = encoder._last_fused_time.grad.norm().item()
                if getattr(encoder, "_last_fused_channel", None) is not None and encoder._last_fused_channel.grad is not None:
                    branch_channel_norm = encoder._last_fused_channel.grad.norm().item()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                if self.rank == 0 and self.metric_tracker.writer:
                    self.metric_tracker.writer.add_scalar("Loss_Batch/train", loss.item(), tb_step)
                    self.metric_tracker.writer.add_scalar("GradNorm_Batch/train", grad_norm.item(), tb_step)
                    if branch_time_norm is not None:
                        self.metric_tracker.writer.add_scalar("GradNorm_Batch/time", branch_time_norm, tb_step)
                    if branch_channel_norm is not None:
                        self.metric_tracker.writer.add_scalar("GradNorm_Batch/channel", branch_channel_norm, tb_step)
                tb_step += 1

            if self.rank == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{metrics['contrastive_accuracy']:.2%}",
                )

        epoch_time = time.time() - epoch_start

        # Flush leftover gradients if batches did not align with grad_accum_steps
        remainder = global_step % self.config.grad_accum_steps
        if remainder != 0 and total_samples > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if self.rank == 0 and self.metric_tracker.writer:
                if last_loss_value is not None:
                    self.metric_tracker.writer.add_scalar("Loss_Batch/train", last_loss_value, tb_step)
                self.metric_tracker.writer.add_scalar("GradNorm_Batch/train", grad_norm.item(), tb_step)
            tb_step += 1

        avg_loss = total_loss / max(1, total_samples)
        throughput = total_samples / max(epoch_time, 1e-6)

        return avg_loss, throughput, global_step, tb_step

    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        tb_val_step: int,
    ) -> Tuple[float, int]:
        """Run validation and return average loss and updated tb step."""
        model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for aug_a, aug_b in val_loader:
                aug_a = aug_a.to(self.device, non_blocking=True)
                aug_b = aug_b.to(self.device, non_blocking=True)

                with autocast(dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext():
                    _, proj_a = model(aug_a)
                    _, proj_b = model(aug_b)
                    loss, _ = self.loss_fn(proj_a, proj_b)

                batch_size = aug_a.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if self.rank == 0 and self.metric_tracker.writer:
                    self.metric_tracker.writer.add_scalar("Loss_Batch/val", loss.item(), tb_val_step)
                tb_val_step += 1

        if dist.is_initialized() and self.world_size > 1:
            loss_tensor = torch.tensor(total_loss, device=self.device)
            sample_tensor = torch.tensor(total_samples, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            total_samples = sample_tensor.item()

        avg_loss = total_loss / max(1, total_samples)
        return avg_loss, tb_val_step

    def train(self, resume_from: Optional[str] = None) -> Dict:
        """Main training loop."""
        killer = GracefulKiller()

        model = self._create_model()
        train_loader, val_loader, tsne_loader, train_sampler, val_sampler = self._create_dataloaders()
        # Store for scheduler step calculations
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tsne_loader = tsne_loader

        if self.rank == 0 and self.metric_tracker.writer:
            self.metric_tracker.writer.add_text(
                "hyperparameters",
                (
                    f"epochs={self.config.epochs}, batch_size={self.config.batch_size}, "
                    f"temperature={self.config.temperature}, dtype={self.config.dtype}, "
                    f"encoder_embed_dim={self.config.encoder_embed_dim}, d_model={self.config.d_model}, "
                    f"dropout={self.config.dropout}, projection_dim={self.config.projection_dim}, "
                    f"max_lr={self.config.max_lr}, warmup_epochs={self.config.warmup_epochs}, "
                    f"onecycle_div_factor={self.config.onecycle_div_factor}"
                ),
            )

        optimizer, scheduler = self._create_optimizer_and_scheduler(model)

        start_epoch = 1
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        tb_step = 0
        tb_val_step = 0

        # Resume from checkpoint if specified
        if resume_from:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                model, optimizer, scheduler, resume_from, map_location=self.device
            )
            start_epoch = checkpoint["epoch"] + 1
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            tb_step = checkpoint.get("tb_step", 0)
            tb_val_step = checkpoint.get("tb_val_step", 0)

        if self.rank == 0:
            print(f"\nStarting training from epoch {start_epoch}")
            print(f"Training for {self.config.epochs} epochs")
            print(
                f"Global batch size: {self.config.batch_size * self.world_size} | "
                f"Grad accum: {self.config.grad_accum_steps} | Max LR: {self.config.max_lr}"
            )
            print("-" * 60)

        for epoch in range(start_epoch, self.config.epochs + 1):
            epoch_start = time.time()

            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            # Train
            train_loss, throughput, _, tb_step = self._train_epoch(
                model, train_loader, optimizer, scheduler, epoch
            )
            train_loss = self.metric_tracker.log_scalar("train/loss", train_loss, epoch)
            train_losses.append(train_loss)

            # Log metrics
            self.metric_tracker.log_learning_rate(optimizer.param_groups[0]["lr"], epoch)
            self.metric_tracker.log_throughput(throughput, epoch)
            self.metric_tracker.log_memory(epoch)

            # Validation
            val_loss = None
            if epoch % self.config.val_frequency == 0:
                val_loss, tb_val_step = self._validate(model, val_loader, tb_val_step)
                val_loss = self.metric_tracker.log_scalar("val/loss", val_loss, epoch)
                val_losses.append(val_loss)

                if self.rank == 0 and self.metric_tracker.writer:
                    self.metric_tracker.writer.add_scalar("Loss_Epoch/train", train_loss, epoch)
                    self.metric_tracker.writer.add_scalar("Loss_Epoch/val", val_loss, epoch)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.rank == 0:
                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            # T-SNE visualization
            if epoch % self.config.vis_interval == 0:
                embeddings, labels = gather_embeddings_for_tsne(
                    model.module if hasattr(model, "module") else model,
                    self.tsne_loader,
                    self.config.tsne_samples,
                    self.device,
                    self.rank,
                    self.world_size,
                    self.autocast_dtype,
                )
                if embeddings is not None and labels is not None and self.rank == 0:
                    metrics = {
                        "silhouette_score": silhouette_score(embeddings, labels),
                        "davies_bouldin_score": davies_bouldin_score(embeddings, labels),
                        "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
                    }

                    if self.metric_tracker.writer:
                        self.metric_tracker.writer.add_scalar(
                            "Clustering/silhouette_score", metrics["silhouette_score"], epoch
                        )
                        self.metric_tracker.writer.add_scalar(
                            "Clustering/davies_bouldin_score", metrics["davies_bouldin_score"], epoch
                        )
                        self.metric_tracker.writer.add_scalar(
                            "Clustering/calinski_harabasz_score", metrics["calinski_harabasz_score"], epoch
                        )

                    self.metric_tracker.log_tsne_embeddings(embeddings, labels, epoch)

            # Step scheduler
            # Scheduler already stepped per optimizer step

            # Save checkpoint
            epoch_time = time.time() - epoch_start
            self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss if val_loss is not None else train_loss,
                config=self.config,
                additional_info={
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_val_loss": best_val_loss,
                    "epoch_time": epoch_time,
                    "tb_step": tb_step,
                    "tb_val_step": tb_val_step,
                },
            )

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stop_patience:
                if self.rank == 0:
                    print(
                        f"\nEarly stopping triggered after {self.config.early_stop_patience} "
                        f"epochs without improvement."
                    )
                break

            if killer.kill_now:
                if self.rank == 0:
                    print("\n[!] Graceful shutdown requested. Saving checkpoint...")
                self.checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=val_loss if val_loss is not None else train_loss,
                    config=self.config,
                    additional_info={
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "best_val_loss": best_val_loss,
                        "epoch_time": epoch_time,
                        "tb_step": tb_step,
                        "tb_val_step": tb_val_step,
                    },
                )
                break

        self.metric_tracker.close()

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch,
        }


def main():
    """Entry point for distributed training."""
    config = parse_args()
    trainer = DistributedTrainer(config)

    # Check for resume
    resume_path = config.resume or trainer.checkpoint_manager.get_latest_checkpoint()
    if resume_path:
        print(f"Found checkpoint, resuming from: {resume_path}")

    results = trainer.train(resume_from=resume_path)

    cleanup_distributed()

    if trainer.rank == 0:
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Final epoch: {results['final_epoch']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
