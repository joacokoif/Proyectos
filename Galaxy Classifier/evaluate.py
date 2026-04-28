"""
evaluate.py — Model Evaluation & Error Analysis
================================================
Comprehensive evaluation of trained galaxy classifiers with:
- Confusion matrix (seaborn heatmap)
- Per-class precision, recall, F1-score
- Balanced accuracy (handles class imbalance)
- ROC curves + AUC per class (one-vs-rest)
- Sample predictions (correct + incorrect)
- Systematic error analysis

Usage:
    python evaluate.py                         # Evaluate ResNet18
    python evaluate.py --model efficientnet_b0 # Evaluate EfficientNet
    python evaluate.py --compare               # Compare both models

Author: Galaxy Classifier Project
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
from tqdm import tqdm

import config
from dataset import create_dataloaders, get_transforms, GalaxyDataset
from model import create_model
from train import set_seed
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# LOAD TRAINED MODEL
# ═══════════════════════════════════════════════════════════════

def load_trained_model(model_name: str = None) -> nn.Module:
    """Load a trained model from checkpoint.
    
    Args:
        model_name: Model architecture name
        
    Returns:
        Trained model in eval mode
    """
    model_name = model_name or config.DEFAULT_MODEL
    checkpoint_path = config.MODELS_DIR / f"best_{model_name}.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found: {checkpoint_path}\n"
            f"Run 'python train.py --model {model_name}' first."
        )
    
    model = create_model(name=model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['val_loss']:.4f})")
    
    return model


# ═══════════════════════════════════════════════════════════════
# COLLECT PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def get_predictions(model, dataloader, device=config.DEVICE):
    """Run inference on entire dataset, collecting predictions + probabilities.
    
    Args:
        model: Trained model in eval mode
        dataloader: DataLoader to evaluate
        device: Target device
        
    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (softmax)
        image_paths: Corresponding image file paths
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating"):
            images = batch[0].to(device)
            labels = batch[1]
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ═══════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name: str = "model"):
    """Plot a beautiful confusion matrix heatmap.
    
    The confusion matrix reveals:
    - Diagonal: correct predictions per class
    - Off-diagonal: misclassification patterns
    - Which galaxy types the model confuses most
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: For plot title and save filename
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title(f'Confusion Matrix — {model_name.upper()}\n(Raw Counts)', fontsize=13)
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('True', fontsize=11)
    
    # Normalized (%)
    sns.heatmap(
        cm_normalized, annot=True, fmt='.1%', cmap='Oranges',
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        ax=axes[1],
        vmin=0, vmax=1,
        cbar_kws={'label': 'Rate'}
    )
    axes[1].set_title(f'Confusion Matrix — {model_name.upper()}\n(Normalized)', fontsize=13)
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('True', fontsize=11)
    
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# ROC CURVES + AUC
# ═══════════════════════════════════════════════════════════════

def plot_roc_curves(y_true, y_probs, model_name: str = "model"):
    """Plot ROC curves with AUC for each class (one-vs-rest).
    
    ROC Analysis reveals:
    - How well the model discriminates each class
    - AUC near 1.0 = excellent, near 0.5 = random
    - Class-specific performance differences
    
    Args:
        y_true: Ground truth labels
        y_probs: Prediction probabilities [N, num_classes]
        model_name: For plot title and save filename
    """
    # Binarize labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=range(config.NUM_CLASSES))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (class_name, color) in enumerate(zip(config.CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr, color=color, lw=2.5,
            label=f'{class_name.capitalize()} (AUC = {roc_auc:.3f})'
        )
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (One-vs-Rest) — {model_name.upper()}', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / f"roc_curves_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curves saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# SAMPLE PREDICTIONS VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def show_predictions(model, dataset, model_name: str = "model", n_samples: int = 12):
    """Show sample predictions with images, true labels, and confidence.
    
    Displays a grid of galaxy images with:
    - Green border: correct prediction
    - Red border: incorrect prediction
    - Confidence percentage from softmax
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        model_name: For save filename
        n_samples: Number of samples to show
    """
    model.eval()
    
    # Get deterministic transform for display
    display_transform = get_transforms("test")
    
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    ncols = 4
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img_path = dataset.samples[idx][0]
        true_label = dataset.samples[idx][1]
        
        # Load and classify
        img = Image.open(img_path).convert("RGB")
        input_tensor = display_transform(img).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_label = output.argmax(dim=1).item()
            confidence = probs[pred_label].item()
        
        # Display
        ax = axes[i]
        ax.imshow(img)
        
        true_name = config.IDX_TO_CLASS[true_label]
        pred_name = config.IDX_TO_CLASS[pred_label]
        correct = true_label == pred_label
        
        color = '#2ecc71' if correct else '#e74c3c'
        icon = '✓' if correct else '✗'
        
        ax.set_title(
            f"{icon} True: {true_name}\nPred: {pred_name} ({confidence:.1%})",
            fontsize=10, color=color, fontweight='bold'
        )
        
        # Border color
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Sample Predictions — {model_name.upper()}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / f"predictions_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Sample predictions saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_errors(y_true, y_pred, y_probs, model_name: str = "model"):
    """Systematic error analysis: what the model gets wrong and why.
    
    Analyzes:
    - Which classes are most confused
    - Error rate per class
    - Average confidence on correct vs incorrect predictions
    - Systematic misclassification patterns
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        model_name: Model name for report
    """
    print(f"\n  {'═'*56}")
    print(f"  ERROR ANALYSIS — {model_name.upper()}")
    print(f"  {'═'*56}")
    
    # Per-class error rates
    print(f"\n  Per-class error analysis:")
    print(f"  {'Class':>12} | {'Errors':>6} | {'Total':>6} | {'Error%':>7} | {'Avg Conf (✓)':>12} | {'Avg Conf (✗)':>12}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*12}-+-{'-'*12}")
    
    for class_idx, class_name in enumerate(config.CLASSES):
        mask = y_true == class_idx
        class_preds = y_pred[mask]
        class_probs = y_probs[mask]
        
        errors = (class_preds != class_idx).sum()
        total = mask.sum()
        error_rate = errors / total if total > 0 else 0
        
        # Confidence on correct vs incorrect
        correct_mask = class_preds == class_idx
        correct_conf = class_probs[correct_mask, class_idx].mean() if correct_mask.any() else 0
        incorrect_conf = class_probs[~correct_mask, class_idx].mean() if (~correct_mask).any() else 0
        
        print(f"  {class_name:>12} | {errors:>6} | {total:>6} | {error_rate:>6.1%} | "
              f"{correct_conf:>11.3f} | {incorrect_conf:>11.3f}")
    
    # Most common misclassification pairs
    print(f"\n  Top misclassification patterns:")
    cm = confusion_matrix(y_true, y_pred)
    
    errors_list = []
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                errors_list.append((config.CLASSES[i], config.CLASSES[j], cm[i, j]))
    
    errors_list.sort(key=lambda x: x[2], reverse=True)
    
    for true_cls, pred_cls, count in errors_list[:5]:
        print(f"    {true_cls:>12} → {pred_cls:<12} ({count} errors)")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if errors_list:
        top_error = errors_list[0]
        print(f"    • Most confused: {top_error[0]} ↔ {top_error[1]}")
        
        if 'spiral' in top_error[0] and 'elliptical' in top_error[1]:
            print(f"    • This is expected: face-on spirals with large bulges")
            print(f"      can appear smooth/elliptical, especially at low resolution.")
        
        if 'irregular' in top_error[0] or 'irregular' in top_error[1]:
            print(f"    • Irregular galaxies are inherently ambiguous — even human")
            print(f"      classifiers disagree on many irregular morphologies.")
    
    print(f"\n  Suggested improvements:")
    print(f"    • Add more irregular galaxy samples (data augmentation)")
    print(f"    • Try ensemble of ResNet + EfficientNet")
    print(f"    • Use higher resolution images (SDSS scale=0.2)")
    print(f"    • Fine-tune with lower learning rate on hard examples")


# ═══════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def evaluate_model(model_name: str = None):
    """Run complete evaluation pipeline on test set.
    
    Args:
        model_name: Model architecture name
    """
    model_name = model_name or config.DEFAULT_MODEL
    
    print("\n" + "═"*60)
    print(f"  EVALUATION — {model_name.upper()}")
    print("═"*60)
    
    set_seed(config.SEED)
    
    # Load model
    model = load_trained_model(model_name)
    
    # Load test data
    data = create_dataloaders(use_weighted_sampler=False)
    test_loader = data["dataloaders"]["test"]
    test_dataset = data["datasets"]["test"]
    
    # Get predictions
    y_true, y_pred, y_probs = get_predictions(model, test_loader)
    
    # ─── Classification Report ───
    print(f"\n  Classification Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=config.CLASSES,
        digits=4
    )
    print(report)
    
    # ─── Balanced Accuracy ───
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    # ─── Standard Accuracy ───
    accuracy = (y_true == y_pred).mean()
    print(f"  Standard Accuracy: {accuracy:.4f}")
    
    # ─── Confusion Matrix ───
    plot_confusion_matrix(y_true, y_pred, model_name)
    
    # ─── ROC Curves ───
    plot_roc_curves(y_true, y_probs, model_name)
    
    # ─── Sample Predictions ───
    show_predictions(model, test_dataset, model_name)
    
    # ─── Error Analysis ───
    analyze_errors(y_true, y_pred, y_probs, model_name)
    
    # ─── Save metrics ───
    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'classification_report': classification_report(
            y_true, y_pred, target_names=config.CLASSES, output_dict=True
        )
    }
    
    import json
    metrics_path = config.RESULTS_DIR / f"metrics_{model_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved: {metrics_path}")
    
    return metrics


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate galaxy classifier")
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=config.AVAILABLE_MODELS,
        help=f"Model to evaluate (default: {config.DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Evaluate and compare both models"
    )
    args = parser.parse_args()
    
    if args.compare:
        results = {}
        for model_name in config.AVAILABLE_MODELS:
            try:
                metrics = evaluate_model(model_name)
                results[model_name] = metrics
            except FileNotFoundError as e:
                print(f"  ⚠ {e}")
        
        # Comparison summary
        if len(results) > 1:
            print(f"\n{'═'*60}")
            print(f"  MODEL COMPARISON SUMMARY")
            print(f"{'═'*60}")
            print(f"  {'Model':>20} | {'Accuracy':>10} | {'Balanced Acc':>12}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*12}")
            for name, m in results.items():
                print(f"  {name:>20} | {m['accuracy']:>9.4f} | {m['balanced_accuracy']:>11.4f}")
    else:
        evaluate_model(args.model)


if __name__ == "__main__":
    main()
