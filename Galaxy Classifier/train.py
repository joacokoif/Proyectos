"""
train.py — Training Pipeline
=============================
Complete training loop with early stopping, learning rate scheduling,
class-weighted loss, and comprehensive metric tracking.

Training Strategy:
    1. Compute class weights using sklearn (handles imbalanced classes)
    2. Use CrossEntropyLoss with class weights
    3. Adam optimizer with weight decay (L2 regularization)
    4. ReduceLROnPlateau scheduler (adapts LR based on val loss)
    5. Early stopping (prevents overfitting, saves best model)
    6. Track: loss, accuracy, precision, recall per epoch

Reproducibility:
    All random seeds are set via set_seed() before any computation.
    This ensures identical results across runs on the same hardware.

Usage:
    python train.py                           # Train ResNet18
    python train.py --model efficientnet_b0   # Train EfficientNet
    python train.py --model resnet18 --epochs 30 --lr 5e-5

Author: Galaxy Classifier Project
"""

import os
import json
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from pathlib import Path

import config
from dataset import create_dataloaders
from model import create_model


# ═══════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════

def set_seed(seed: int = config.SEED):
    """Set all random seeds for full reproducibility.
    
    Ensures identical results across runs by controlling randomness in:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU operations
    - PyTorch CUDA operations (all GPUs)
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For full determinism (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"  🎲 Random seed set: {seed}")


# ═══════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is seen for 'patience' consecutive epochs. Saves the best model
    checkpoint automatically.
    
    Why it matters:
        Neural networks can memorize training data if trained too long.
        Early stopping acts as implicit regularization — it stops at
        the epoch where the model generalizes best to unseen data.
    
    Args:
        patience: Epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as improvement
        save_path: Path to save best model checkpoint
    """
    
    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = 1e-4,
        save_path: str = None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path or str(config.MODELS_DIR / "best_model.pth")
        
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, model: nn.Module, epoch: int):
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Current model (saved if best)
            epoch: Current epoch number
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            
            # Save best model
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'model_name': model.backbone_name,
            }, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  ⏹ Early stopping triggered (no improvement for {self.patience} epochs)")
                print(f"    Best epoch: {self.best_epoch} (val_loss: {self.best_loss:.4f})")


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Target device (CPU/GPU)
        
    Returns:
        Dict with epoch metrics (loss, accuracy, precision, recall)
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="    Training", leave=False):
        # Handle both (image, label) and (image, label, soft_label) formats
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute epoch metrics
    n_samples = len(all_labels)
    epoch_loss = running_loss / n_samples
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1
    }


def validate(model, dataloader, criterion, device):
    """Evaluate the model on validation/test data.
    
    Args:
        model: Neural network model
        dataloader: Validation/Test DataLoader
        criterion: Loss function
        device: Target device
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="    Validating", leave=False):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    n_samples = len(all_labels)
    
    return {
        'loss': running_loss / n_samples,
        'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }


# ═══════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def train_model(
    model_name: str = None,
    num_epochs: int = None,
    learning_rate: float = None,
    batch_size: int = None,
) -> dict:
    """Complete training pipeline with all bells and whistles.
    
    Pipeline:
        1. Set random seeds for reproducibility
        2. Create data loaders with weighted sampling
        3. Compute class weights for loss function
        4. Initialize model, optimizer, scheduler
        5. Train with early stopping
        6. Save best model + training history
    
    Args:
        model_name: "resnet18" or "efficientnet_b0"
        num_epochs: Max training epochs
        learning_rate: Initial learning rate
        batch_size: Training batch size
        
    Returns:
        Dict with training history and best metrics
    """
    # ─── Configuration ───
    model_name = model_name or config.DEFAULT_MODEL
    num_epochs = num_epochs or config.NUM_EPOCHS
    lr = learning_rate or config.LEARNING_RATE
    bs = batch_size or config.BATCH_SIZE
    
    print("\n" + "═"*60)
    print(f"  TRAINING — {model_name.upper()}")
    print("═"*60)
    
    # ─── Seed ───
    set_seed(config.SEED)
    
    # ─── Data ───
    print("\n  Loading data...")
    data = create_dataloaders(batch_size=bs, use_soft_labels=config.USE_SOFT_LABELS)
    dataloaders = data["dataloaders"]
    datasets = data["datasets"]
    
    # ─── Class Weights (sklearn) ───
    # Computed from training set to handle class imbalance
    train_labels = datasets["train"].targets
    unique_classes = np.unique(train_labels)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.DEVICE)
    print(f"\n  Class weights (balanced): {class_weights.tolist()}")
    
    # ─── Model ───
    model = create_model(name=model_name)
    
    # ─── Loss + Optimizer + Scheduler ───
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR
    )
    
    # ─── Early Stopping ───
    save_path = str(config.MODELS_DIR / f"best_{model_name}.pth")
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        save_path=save_path
    )
    
    # ─── Training History ───
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [],
        'lr': []
    }
    
    # ─── Training Loop ───
    print(f"\n  Starting training ({num_epochs} epochs, lr={lr})")
    print(f"  {'─'*56}")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch {epoch}/{num_epochs} (lr={current_lr:.2e})")
        
        # Train
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_metrics = validate(
            model, dataloaders["val"], criterion, config.DEVICE
        )
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        for key in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"    Train — loss: {train_metrics['loss']:.4f} | "
              f"acc: {train_metrics['accuracy']:.4f} | "
              f"prec: {train_metrics['precision']:.4f} | "
              f"rec: {train_metrics['recall']:.4f}")
        print(f"    Val   — loss: {val_metrics['loss']:.4f} | "
              f"acc: {val_metrics['accuracy']:.4f} | "
              f"prec: {val_metrics['precision']:.4f} | "
              f"rec: {val_metrics['recall']:.4f}")
        
        # Early stopping check
        early_stopping(val_metrics['loss'], model, epoch)
        if early_stopping.should_stop:
            break
    
    # ─── Training Summary ───
    elapsed = time.time() - start_time
    
    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE — {model_name.upper()}")
    print(f"{'═'*60}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best epoch: {early_stopping.best_epoch}")
    print(f"  Best val loss: {early_stopping.best_loss:.4f}")
    print(f"  Model saved: {save_path}")
    
    # Save training history
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.RESULTS_DIR / f"history_{model_name}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")
    
    return {
        'history': history,
        'best_epoch': early_stopping.best_epoch,
        'best_val_loss': early_stopping.best_loss,
        'model_path': save_path,
        'elapsed_time': elapsed
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train galaxy classifier")
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=config.AVAILABLE_MODELS,
        help=f"Model architecture (default: {config.DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help=f"Max training epochs (default: {config.NUM_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Learning rate (default: {config.LEARNING_RATE})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Train both ResNet18 and EfficientNet-B0 for comparison"
    )
    args = parser.parse_args()
    
    if args.compare:
        # Train both models for comparison
        results = {}
        for model_name in config.AVAILABLE_MODELS:
            result = train_model(
                model_name=model_name,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size
            )
            results[model_name] = result
        
        # Print comparison
        print(f"\n{'═'*60}")
        print(f"  MODEL COMPARISON")
        print(f"{'═'*60}")
        for name, result in results.items():
            print(f"  {name:>20}: val_loss={result['best_val_loss']:.4f} "
                  f"(best_epoch={result['best_epoch']}, "
                  f"time={result['elapsed_time']:.1f}s)")
    else:
        train_model(
            model_name=args.model,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size
        )
    
    print(f"\n  Next step: python evaluate.py --model {args.model}")


if __name__ == "__main__":
    main()
