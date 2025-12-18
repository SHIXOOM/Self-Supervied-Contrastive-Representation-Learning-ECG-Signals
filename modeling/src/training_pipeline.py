
import time
from typing import Optional
from src import CheckpointManager
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model,
    dataset,
    optimizer,
    loss_fn,
    config,
    batch_size=64,
    epochs=1,
    checkpoint_manager: CheckpointManager = None,
    resume_from: str = None,
    device: Optional[torch.device] = None,
    val_dataset=None,
    scheduler=None,
    lr_decay_start_epoch=15,
    lr_decay_factor=0.5,
):
    """
    Train the ECG encoder with contrastive learning.
    
    Args:
        model: ECGEncoder model
        dataset: ECGContrastiveTrainDataset
        optimizer: Optimizer (e.g., Adam)
        loss_fn: Loss function (e.g., NTXentLoss)
        config: Model configuration
        batch_size: Batch size for training
        epochs: Number of epochs to train
        checkpoint_manager: CheckpointManager instance for saving checkpoints
        resume_from: Path to checkpoint to resume training from
        device: Torch device to run training on (auto-selected if None)
        val_dataset: Optional validation dataset for evaluation after each epoch
        scheduler: Optional LR scheduler (defaults to decay after `lr_decay_start_epoch`)
        lr_decay_start_epoch: Epoch index at which the default scheduler lowers the LR
        lr_decay_factor: Multiplicative factor applied after the decay epoch
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    model.to(device)
    start_epoch = 0
    loss_history = []
    val_loss_history = []
    if scheduler is None:
        scheduler = LambdaLR(
            optimizer,
            lambda epoch: lr_decay_factor ** (epoch // lr_decay_start_epoch),
        )
    
    # Resume from checkpoint if specified
    if resume_from:
        if checkpoint_manager is None:
            raise ValueError("resume_from requires a checkpoint_manager instance")
        checkpoint = checkpoint_manager.load_checkpoint(
            resume_from, model, optimizer, map_location=device
        )
        start_epoch = checkpoint['epoch']
        loss_history = checkpoint.get('loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        print(f"Resuming training from epoch {start_epoch + 1}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True) if val_dataset else None
    
    training_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0.0
        total_grad_norm = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + epochs} [Train]")
        
        for batch_idx, (aug1, aug2) in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)
            aug1 = aug1.to(device, non_blocking=True)
            aug2 = aug2.to(device, non_blocking=True)

            _, proj1 = model(aug1)
            _, proj2 = model(aug2)


            loss = loss_fn(proj1, proj2)
            loss.backward()
            
            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with real-time metrics
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad_norm': f'{grad_norm.item():.2f}'
            })

        # Calculate training epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_grad_norm = total_grad_norm / len(dataloader) if len(dataloader) else 0.0
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        
        loss_history.append(avg_loss)
        
        # Validation phase
        val_loss = None
        if val_dataloader:
            model.eval()
            val_total_loss = 0.0
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + epochs} [Val]")
            
            with torch.no_grad():
                for aug1, aug2 in val_progress_bar:
                    aug1 = aug1.to(device, non_blocking=True)
                    aug2 = aug2.to(device, non_blocking=True)

                    _, proj1 = model(aug1)
                    _, proj2 = model(aug2)

                    loss = loss_fn(proj1, proj2)
                    val_total_loss += loss.item()
                    
                    val_progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
            val_loss = val_total_loss / len(val_dataloader)
            val_loss_history.append(val_loss)
        
        # Print epoch summary
        summary = (f"Epoch [{epoch + 1}/{start_epoch + epochs}] "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Grad Norm: {avg_grad_norm:.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | ")
        if val_loss is not None:
            summary += f"Val Loss: {val_loss:.4f} | "
        summary += f"Time: {epoch_time:.1f}s | Total: {total_time:.1f}s"
        print(summary)
        
        # Save checkpoint after each epoch
        if checkpoint_manager:
            additional_info = {
                'loss_history': loss_history,
                'avg_grad_norm': avg_grad_norm,
                'epoch_time': epoch_time,
                'total_training_time': total_time
            }
            if val_loss is not None:
                additional_info['val_loss_history'] = val_loss_history
                additional_info['val_loss'] = val_loss
            if scheduler is not None:
                additional_info['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                config=config,
                additional_info=additional_info
            )
        if scheduler is not None:
            scheduler.step()

    total_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    result = {'loss_history': loss_history}
    if val_loss_history:
        result['val_loss_history'] = val_loss_history
    return result