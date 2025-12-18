import os
from dataclasses import asdict
from pathlib import Path

import torch

from src import DualAugmenter


class CheckpointManager:
    """
    Manages model checkpointing during training.

    Features:
    - Saves model, optimizer, and training state after each epoch
    - Keeps track of best model based on a metric (supports min or max mode)
    - Maintains only the last N checkpoints to save disk space
    - Supports resuming training from saved checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str = "../models/checkpoints",
        keep_last_n: int = 3,
        save_best: bool = True,
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        metric: float,
        config,
        additional_info: dict = None,
    ):
        """Save checkpoint with model state, optimizer state, and metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
            "config": asdict(config)
            if config is not None and hasattr(config, "__dataclass_fields__")
            else config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        # Save regular epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model based on mode (min: lower is better, max: higher is better)
        is_best = (self.mode == "min" and metric < self.best_metric) or (
            self.mode == "max" and metric > self.best_metric
        )
        if self.save_best and is_best:
            self.best_metric = metric
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with metric: {metric:.4f}")

        # Save latest checkpoint (for easy resume)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints to save disk space."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Remove old checkpoints beyond keep_last_n
        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[: -self.keep_last_n]:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint.name}")

    def load_checkpoint(
        self, checkpoint_path: str, model, optimizer=None, map_location=None
    ):
        """
        Load checkpoint and restore model and optimizer states.

        Returns:
            Dictionary with epoch, loss, and config information
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        metric_value = checkpoint.get("metric", checkpoint.get("loss", 0.0))
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with metric {metric_value:.4f}"
        )
        return checkpoint

    def get_latest_checkpoint(self):
        """Get path to the latest checkpoint if it exists."""
        latest_path = self.checkpoint_dir / "latest.pt"
        return str(latest_path) if latest_path.exists() else None
