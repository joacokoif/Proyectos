"""
prepare_dataset.py — Dataset Preparation & Splitting
=====================================================
Processes the raw Galaxy Zoo 2 catalog, assigns labels, creates
stratified train/val/test splits, and organizes the folder structure.

Label Assignment Logic (per user specification):
    if smooth > 0.7:
        label = "elliptical"
    elif features > 0.7:
        label = "spiral"
    elif odd_yes > 0.5:
        label = "irregular"
    else:
        continue  # discard ambiguous cases

Supports both hard labels and soft labels (GZ2 vote probabilities).

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --soft-labels

Author: Galaxy Classifier Project
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

import config


# ═══════════════════════════════════════════════════════════════
# SEED FOR REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════
np.random.seed(config.SEED)


# ═══════════════════════════════════════════════════════════════
# LABEL ASSIGNMENT
# ═══════════════════════════════════════════════════════════════

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign morphological labels based on GZ2 debiased vote fractions.
    
    Uses the hierarchical decision logic:
        1. smooth > 0.7  → elliptical
        2. features > 0.7 → spiral  
        3. odd_yes > 0.5  → irregular
        4. else → discard (ambiguous, too noisy for training)
    
    This approach avoids label noise by only keeping high-confidence
    classifications where citizen scientists largely agreed.
    
    Args:
        df: DataFrame with 'smooth', 'features', 'odd' columns
        
    Returns:
        DataFrame with only labeled rows (ambiguous cases removed)
    """
    labels = []
    
    for _, row in df.iterrows():
        smooth = row.get('smooth', 0)
        features = row.get('features', 0)
        odd = row.get('odd', 0)
        
        if smooth > config.ELLIPTICAL_THRESHOLD:
            labels.append("elliptical")
        elif features > config.SPIRAL_THRESHOLD:
            labels.append("spiral")
        elif odd > config.IRREGULAR_THRESHOLD:
            labels.append("irregular")
        else:
            labels.append(None)  # Ambiguous — will be discarded
    
    df = df.copy()
    df['label'] = labels
    
    # Remove ambiguous cases
    n_before = len(df)
    df = df.dropna(subset=['label'])
    n_removed = n_before - len(df)
    
    if n_removed > 0:
        print(f"  ⚠ Discarded {n_removed} ambiguous galaxies (low agreement)")
    
    return df


def generate_soft_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate soft labels from GZ2 vote probabilities.
    
    Instead of hard labels (one-hot), soft labels represent the actual
    distribution of human votes. This captures uncertainty — a galaxy
    that 60% of people classified as spiral and 30% as elliptical
    gets a softer label than one with 95% spiral agreement.
    
    Soft Labels Format:
        label_probs = [smooth, features, odd_yes]
        (normalized to sum to 1)
    
    This can be used with KLDivLoss or soft cross-entropy during training
    for potentially better calibration.
    
    Args:
        df: DataFrame with 'smooth', 'features', 'odd' columns
        
    Returns:
        DataFrame with soft label columns
    """
    df = df.copy()
    
    # Extract the three key vote fractions
    smooth = df['smooth'].fillna(0).values
    features = df['features'].fillna(0).values
    odd = df['odd'].fillna(0).values
    
    # Stack and normalize to form probability distributions
    probs = np.column_stack([smooth, features, odd])
    
    # Normalize each row to sum to 1 (handle zero-sum edge case)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    probs = probs / row_sums
    
    df['soft_elliptical'] = probs[:, 0]
    df['soft_spiral'] = probs[:, 1]
    df['soft_irregular'] = probs[:, 2]
    
    return df


# ═══════════════════════════════════════════════════════════════
# DATASET SPLITTING
# ═══════════════════════════════════════════════════════════════

def create_splits(df: pd.DataFrame) -> dict:
    """Create stratified train/val/test splits.
    
    Uses stratified splitting to maintain class proportions across
    all splits. This is critical for imbalanced datasets (irregulars
    are typically much rarer than spirals/ellipticals).
    
    Args:
        df: Labeled DataFrame
        
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    train_ratio, val_ratio, test_ratio = config.SPLIT_RATIOS
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=config.SEED
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df['label'],
        random_state=config.SEED
    )
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    print(f"\n  Split distribution:")
    for split_name, split_df in splits.items():
        counts = split_df['label'].value_counts()
        total = len(split_df)
        print(f"    {split_name:>5} ({total:>4} total): " + 
              " | ".join(f"{cls}: {cnt}" for cls, cnt in counts.items()))
    
    return splits


def organize_folders(splits: dict):
    """Copy images into the standard folder structure for PyTorch.
    
    Creates:
        data/train/elliptical/  data/train/spiral/  data/train/irregular/
        data/val/elliptical/    data/val/spiral/    data/val/irregular/
        data/test/elliptical/   data/test/spiral/   data/test/irregular/
    
    Args:
        splits: Dict with 'train', 'val', 'test' DataFrames
    """
    split_dirs = {
        'train': config.TRAIN_DIR,
        'val': config.VAL_DIR,
        'test': config.TEST_DIR
    }
    
    # Clean existing folders
    for split_dir in split_dirs.values():
        if split_dir.exists():
            shutil.rmtree(split_dir)
    
    total_copied = 0
    
    for split_name, split_df in splits.items():
        split_dir = split_dirs[split_name]
        
        for _, row in split_df.iterrows():
            src = Path(row['filepath'])
            if not src.exists():
                continue
            
            dst_dir = split_dir / row['label']
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            
            shutil.copy2(src, dst)
            total_copied += 1
    
    print(f"\n  ✓ Organized {total_copied} images into folder structure")
    
    # Print final structure
    print(f"\n  Final structure:")
    for split_name, split_dir in split_dirs.items():
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                n_files = len(list(class_dir.glob("*.jpg")))
                print(f"    {split_name}/{class_dir.name}: {n_files} images")


# ═══════════════════════════════════════════════════════════════
# CLASS BALANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_balance(df: pd.DataFrame):
    """Analyze and report class balance statistics.
    
    Imbalanced classes are common in galaxy morphology:
    - Spirals and ellipticals dominate Galaxy Zoo
    - Irregulars are rare (~5-10% of all galaxies)
    
    This function reports the imbalance ratio and suggests
    mitigation strategies (oversampling, class weights, etc.)
    """
    counts = df['label'].value_counts()
    total = len(df)
    
    print(f"\n  Class Balance Analysis:")
    print(f"  {'Class':>12} | {'Count':>6} | {'Ratio':>8} | {'Bar'}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*30}")
    
    max_count = counts.max()
    for cls, count in counts.items():
        ratio = count / total
        bar = "█" * int(30 * count / max_count)
        print(f"  {cls:>12} | {count:>6} | {ratio:>7.1%} | {bar}")
    
    imbalance_ratio = max_count / counts.min()
    print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 3:
        print("  ⚠ Significant class imbalance detected.")
        print("  → Using class weights in training (computed automatically)")
        print("  → Consider oversampling minority class if needed")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Galaxy Zoo 2 dataset for training"
    )
    parser.add_argument(
        "--soft-labels", action="store_true",
        help="Generate soft labels from vote probabilities"
    )
    parser.add_argument(
        "--catalog", type=str, default=str(config.CATALOG_PATH),
        help="Path to catalog CSV (from download_data.py)"
    )
    args = parser.parse_args()
    
    print("\n" + "═"*60)
    print("  DATASET PREPARATION")
    print("═"*60)
    
    # Load catalog
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"  ✗ Catalog not found: {catalog_path}")
        print(f"    Run 'python download_data.py' first.")
        return
    
    df = pd.read_csv(catalog_path)
    print(f"  ✓ Loaded {len(df)} galaxies from catalog")
    
    # Assign labels (with ambiguity filtering)
    df = assign_labels(df)
    print(f"  ✓ Labeled {len(df)} galaxies")
    
    # Analyze class balance
    analyze_balance(df)
    
    # Generate soft labels if requested
    if args.soft_labels:
        df = generate_soft_labels(df)
        soft_path = config.SOFT_LABELS_PATH
        df[['filepath', 'label', 'soft_elliptical', 'soft_spiral', 'soft_irregular']].to_csv(
            soft_path, index=False
        )
        print(f"\n  ✓ Soft labels saved: {soft_path}")
    
    # Create stratified splits
    splits = create_splits(df)
    
    # Organize into folder structure
    organize_folders(splits)
    
    print("\n" + "═"*60)
    print("  ✓ Dataset preparation complete!")
    print("  Next step: python train.py")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
