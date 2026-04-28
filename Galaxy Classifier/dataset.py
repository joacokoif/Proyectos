"""
dataset.py — PyTorch Dataset & Data Loaders
============================================
Custom dataset class for Galaxy Zoo 2 images with configurable
transforms for training, validation, and testing.

Data Augmentation Strategy:
    - Galaxies have NO preferred orientation → heavy rotation augmentation
    - RandomRotation(180°) captures all possible orientations
    - Horizontal/Vertical flips double the effective dataset
    - ColorJitter handles exposure/contrast variations across SDSS images
    - RandomResizedCrop simulates different angular sizes

Why Augmentation Matters:
    Galaxy morphology is rotation-invariant — a spiral galaxy looks like
    a spiral regardless of orientation. Strong augmentation teaches the 
    model this invariance and prevents overfitting on orientation artifacts.

Author: Galaxy Classifier Project
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import Counter

import config


# ═══════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════

def get_transforms(split: str = "train") -> transforms.Compose:
    """Get image transforms for a given split.
    
    Training transforms include aggressive augmentation because:
    - Galaxy morphology is rotation-invariant
    - SDSS images vary in exposure/contrast
    - We need to maximize effective training data
    
    Validation/Test transforms only resize and normalize.
    
    Args:
        split: One of "train", "val", "test"
        
    Returns:
        torchvision.transforms.Compose pipeline
    """
    if split == "train":
        return transforms.Compose([
            # Grayscale prevents overfitting to SDSS fake-color mappings
            transforms.Grayscale(num_output_channels=3),
            # Resize with random crop to simulate varying angular sizes
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE, 
                scale=(0.8, 1.0),     # Crop between 80-100% of image
                ratio=(0.9, 1.1)      # Nearly square crops
            ),
            # Galaxies have no preferred orientation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),  # Full rotation invariance
            # Handle SDSS exposure/contrast variations
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            # Random erasing to improve generalization
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.06)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])
    else:
        # Val/Test: deterministic transforms with CenterCrop
        return transforms.Compose([
            # Grayscale to match training and ignore color bias
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(config.IMAGE_SIZE + 32),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class GalaxyDataset(Dataset):
    """PyTorch Dataset for galaxy morphology classification.
    
    Loads images from the standard folder structure:
        data/{split}/{class_name}/*.jpg
    
    Supports both hard labels (class indices) and soft labels
    (vote probability distributions from Galaxy Zoo 2).
    
    Args:
        root_dir: Path to split directory (e.g., data/train)
        transform: Image transform pipeline
        soft_labels_path: Optional path to soft labels CSV
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose = None,
        soft_labels_path: str = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_transforms("val")
        self.classes = config.CLASSES
        self.class_to_idx = config.CLASS_TO_IDX
        
        # Collect all image paths and labels
        self.samples = []
        self.targets = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((str(img_path), class_idx))
                self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
        
        # Load soft labels if available
        self.soft_labels = None
        if soft_labels_path and Path(soft_labels_path).exists():
            soft_df = pd.read_csv(soft_labels_path)
            # Build a mapping from filepath to soft label array
            self.soft_labels = {}
            for _, row in soft_df.iterrows():
                fp = str(Path(row['filepath']).name)  # Match by filename
                self.soft_labels[fp] = np.array([
                    row['soft_elliptical'],
                    row['soft_spiral'],
                    row['soft_irregular']
                ], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Return soft labels if available
        if self.soft_labels is not None:
            filename = Path(img_path).name
            if filename in self.soft_labels:
                soft_label = torch.tensor(self.soft_labels[filename])
                return image, label, soft_label
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """Get number of samples per class."""
        counter = Counter(self.targets.tolist())
        return {config.IDX_TO_CLASS[k]: v for k, v in counter.items()}
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training.
        
        Classes with fewer samples get higher weights, compensating
        for the natural imbalance in galaxy morphology datasets
        (irregulars are much rarer than spirals/ellipticals).
        
        Returns:
            Tensor of class weights, shape (NUM_CLASSES,)
        """
        counts = np.bincount(self.targets, minlength=config.NUM_CLASSES)
        total = len(self.targets)
        weights = total / (config.NUM_CLASSES * counts.astype(np.float64))
        return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def create_dataloaders(
    batch_size: int = None,
    num_workers: int = None,
    use_weighted_sampler: bool = True,
    use_soft_labels: bool = False
) -> dict:
    """Create train/val/test DataLoaders with optional weighted sampling.
    
    Weighted sampling addresses class imbalance by oversampling minority 
    classes during training. Each batch will have approximately equal
    representation of all classes.
    
    Args:
        batch_size: Batch size (default: config value)
        num_workers: DataLoader workers (default: config value)
        use_weighted_sampler: Use WeightedRandomSampler for training
        use_soft_labels: Load soft labels
        
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders and datasets
    """
    bs = batch_size or config.BATCH_SIZE
    # On Windows/CPU, use 0 workers to avoid memory issues
    if num_workers is not None:
        nw = num_workers
    elif config.DEVICE.type == 'cpu' or os.name == 'nt':
        nw = 0
    else:
        nw = config.NUM_WORKERS
    use_pin = config.DEVICE.type == 'cuda'
    
    soft_path = str(config.SOFT_LABELS_PATH) if use_soft_labels else None
    
    # Create datasets
    datasets = {
        "train": GalaxyDataset(
            config.TRAIN_DIR, 
            transform=get_transforms("train"),
            soft_labels_path=soft_path
        ),
        "val": GalaxyDataset(
            config.VAL_DIR, 
            transform=get_transforms("val"),
            soft_labels_path=soft_path
        ),
        "test": GalaxyDataset(
            config.TEST_DIR, 
            transform=get_transforms("test"),
            soft_labels_path=soft_path
        ),
    }
    
    # Print dataset sizes
    print(f"\n  Dataset sizes:")
    for split_name, ds in datasets.items():
        counts = ds.get_class_counts()
        total = len(ds)
        count_str = " | ".join(f"{k}: {v}" for k, v in counts.items())
        print(f"    {split_name:>5}: {total:>4} images ({count_str})")
    
    # Build weighted sampler for training (handles class imbalance)
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampler and len(datasets["train"]) > 0:
        class_weights = datasets["train"].get_class_weights()
        sample_weights = class_weights[datasets["train"].targets]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(datasets["train"]),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling
        print(f"\n  Using WeightedRandomSampler (class weights: {class_weights.tolist()})")
    
    # Create DataLoaders
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=bs,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=use_pin,
            drop_last=True
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=use_pin
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=use_pin
        ),
    }
    
    return {"dataloaders": dataloaders, "datasets": datasets}
