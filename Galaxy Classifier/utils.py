"""
utils.py — Utilities, Grad-CAM & Visualization
================================================
Helper functions for inference, interpretability, and visualization.

Key Features:
    - predict_image(path): Classify any galaxy image
    - Grad-CAM: Visualize what the model "sees" in each galaxy
    - Training curves: Loss/accuracy plots
    - Model comparison charts
    - NASA image inference demo

Grad-CAM (Gradient-weighted Class Activation Mapping):
    Shows which spatial regions of the galaxy image most influenced
    the model's classification decision. For galaxy morphology:
    - Spirals: the model should focus on spiral arms
    - Ellipticals: the model should focus on the smooth central bulge
    - Irregulars: the model may focus on asymmetric/clumpy regions

Author: Galaxy Classifier Project
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from dataset import get_transforms
from model import create_model, get_target_layer


# ═══════════════════════════════════════════════════════════════
# IMAGE DENORMALIZATION
# ═══════════════════════════════════════════════════════════════

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for display.
    
    Converts a normalized tensor back to [0, 1] range for matplotlib.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        numpy array [H, W, C] in [0, 1]
    """
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    
    img = tensor.cpu().clone()
    img = img * std + mean
    img = img.clamp(0, 1)
    
    return img.permute(1, 2, 0).numpy()


# ═══════════════════════════════════════════════════════════════
# PREDICT IMAGE
# ═══════════════════════════════════════════════════════════════

def predict_image(
    image_path: str,
    model=None,
    model_name: str = None,
    show_gradcam: bool = True,
    save_path: str = None
) -> dict:
    """Classify a single galaxy image with optional Grad-CAM overlay.
    
    This is the main inference function — load any galaxy image and
    get a classification with confidence scores and visual explanation.
    
    Args:
        image_path: Path to input image
        model: Pre-loaded model (optional — loads from checkpoint if None)
        model_name: Model architecture name (used if model is None)
        show_gradcam: Whether to generate Grad-CAM visualization
        save_path: Where to save the visualization (None = auto)
        
    Returns:
        Dict with prediction, confidence, and class probabilities
    
    Example:
        >>> result = predict_image("nasa_galaxy.jpg")
        >>> print(f"This is a {result['prediction']} galaxy ({result['confidence']:.1%})")
    """
    model_name = model_name or config.DEFAULT_MODEL
    
    # Load model if not provided
    if model is None:
        checkpoint_path = config.MODELS_DIR / f"best_{model_name}.pth"
        model = create_model(name=model_name, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load and transform image
    img = Image.open(image_path).convert("RGB")
    original_img = img.copy()
    
    transform = get_transforms("test")
    input_tensor = transform(img).unsqueeze(0).to(config.DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
    
    prediction = config.IDX_TO_CLASS[pred_idx]
    confidence = probs[pred_idx].item()
    class_probs = {config.IDX_TO_CLASS[i]: probs[i].item() for i in range(config.NUM_CLASSES)}
    
    result = {
        'prediction': prediction,
        'confidence': confidence,
        'class_probabilities': class_probs,
        'image_path': str(image_path)
    }
    
    # ─── Visualization ───
    if show_gradcam:
        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8])
        
        # Original image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(original_img)
        ax1.set_title("Original Galaxy", fontsize=12)
        ax1.axis('off')
        
        # Grad-CAM overlay
        ax2 = fig.add_subplot(gs[1])
        try:
            target_layer = [get_target_layer(model)]
            cam = GradCAM(model=model, target_layers=target_layer)
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # Prepare original image for overlay (resize to match)
            img_resized = original_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            cam_image = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
            ax2.imshow(cam_image)
            ax2.set_title("Grad-CAM\n(Model Attention)", fontsize=12)
        except Exception as e:
            ax2.text(0.5, 0.5, f"Grad-CAM\nunavailable", 
                    ha='center', va='center', fontsize=11)
            ax2.set_title("Grad-CAM", fontsize=12)
        ax2.axis('off')
        
        # Heatmap only
        ax3 = fig.add_subplot(gs[2])
        try:
            ax3.imshow(grayscale_cam, cmap='jet', alpha=0.8)
            ax3.set_title("Attention Heatmap", fontsize=12)
        except:
            ax3.text(0.5, 0.5, "N/A", ha='center', va='center')
        ax3.axis('off')
        
        # Prediction bar chart
        ax4 = fig.add_subplot(gs[3])
        classes = list(class_probs.keys())
        values = list(class_probs.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax4.barh(classes, values, color=colors, edgecolor='white', linewidth=0.5)
        ax4.set_xlim(0, 1)
        ax4.set_xlabel('Probability', fontsize=10)
        ax4.set_title('Prediction', fontsize=12)
        
        # Annotate bars
        for bar, val in zip(bars, values):
            ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(
            f"Galaxy Classification: {prediction.upper()} ({confidence:.1%})",
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        
        # Save
        if save_path is None:
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = str(config.RESULTS_DIR / f"prediction_{img_name}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Prediction visualization saved: {save_path}")
        result['visualization_path'] = save_path
    
    print(f"\n  Result: {prediction.upper()} ({confidence:.1%})")
    for cls, prob in class_probs.items():
        bar = "█" * int(20 * prob)
        print(f"    {cls:>12}: {prob:.3f} {bar}")
    
    return result


# ═══════════════════════════════════════════════════════════════
# GRAD-CAM BATCH VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def gradcam_gallery(
    model,
    dataset,
    model_name: str = "model",
    n_samples: int = 9
):
    """Generate a gallery of Grad-CAM visualizations.
    
    Shows where the model focuses for different galaxy types,
    helping verify that it learns meaningful morphological features:
    - Spiral: attention on arms
    - Elliptical: attention on central bulge/smooth profile
    - Irregular: attention on asymmetric/clumpy regions
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        model_name: For save filename
        n_samples: Number of samples
    """
    model.eval()
    
    target_layers = [get_target_layer(model)]
    cam = GradCAM(model=model, target_layers=target_layers)
    transform = get_transforms("test")
    
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    ncols = 3
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(4 * ncols * 2, 4 * nrows))
    
    for i, idx in enumerate(indices):
        img_path = dataset.samples[idx][0]
        true_label = dataset.samples[idx][1]
        true_name = config.IDX_TO_CLASS[true_label]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        input_tensor = transform(img).unsqueeze(0).to(config.DEVICE)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0][pred_idx].item()
        pred_name = config.IDX_TO_CLASS[pred_idx]
        
        # Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        cam_image = show_cam_on_image(img_array, grayscale_cam[0], use_rgb=True)
        
        # Plot
        row = i // ncols
        col = (i % ncols) * 2
        
        # Original
        axes[row, col].imshow(img_resized)
        correct = true_label == pred_idx
        color = '#2ecc71' if correct else '#e74c3c'
        axes[row, col].set_title(f"True: {true_name}", fontsize=10)
        axes[row, col].axis('off')
        
        # Grad-CAM
        axes[row, col + 1].imshow(cam_image)
        axes[row, col + 1].set_title(
            f"Pred: {pred_name} ({confidence:.0%})", 
            fontsize=10, color=color, fontweight='bold'
        )
        axes[row, col + 1].axis('off')
    
    # Hide unused
    for j in range(i + 1, nrows * ncols):
        row = j // ncols
        col = (j % ncols) * 2
        if row < len(axes) and col + 1 < len(axes[row]):
            axes[row, col].set_visible(False)
            axes[row, col + 1].set_visible(False)
    
    plt.suptitle(f'Grad-CAM Gallery — {model_name.upper()}', fontsize=14, y=1.01)
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / f"gradcam_gallery_{model_name}.png"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grad-CAM gallery saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# TRAINING CURVES
# ═══════════════════════════════════════════════════════════════

def plot_training_history(model_name: str = None):
    """Plot training curves from saved history.
    
    Creates a 2x2 grid showing:
    - Loss (train vs val)
    - Accuracy (train vs val)
    - Precision (train vs val)
    - Learning rate schedule
    
    Args:
        model_name: Model name (loads history_{model_name}.json)
    """
    model_name = model_name or config.DEFAULT_MODEL
    history_path = config.RESULTS_DIR / f"history_{model_name}.json"
    
    if not history_path.exists():
        print(f"  ✗ History not found: {history_path}")
        return
    
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ─── Loss ───
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ─── Accuracy ───
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # ─── Precision & Recall ───
    axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='Train Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Val Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['train_recall'], 'b--', label='Train Recall', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, history['val_recall'], 'r--', label='Val Recall', linewidth=2, alpha=0.7)
    axes[1, 0].set_title('Precision & Recall', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # ─── Learning Rate ───
    axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2, marker='o', markersize=3)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History — {model_name.upper()}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / f"training_curves_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved: {save_path}")


def compare_models():
    """Generate comparison plots for ResNet vs EfficientNet.
    
    Creates side-by-side training curves and accuracy comparisons
    to help decide which model performs better for galaxy classification.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'resnet18': '#3498db', 'efficientnet_b0': '#e74c3c'}
    
    for model_name in config.AVAILABLE_MODELS:
        history_path = config.RESULTS_DIR / f"history_{model_name}.json"
        if not history_path.exists():
            continue
        
        with open(history_path) as f:
            history = json.load(f)
        
        epochs = range(1, len(history['val_loss']) + 1)
        color = colors.get(model_name, '#333')
        
        axes[0].plot(epochs, history['val_loss'], color=color, linewidth=2, label=model_name)
        axes[1].plot(epochs, history['val_accuracy'], color=color, linewidth=2, label=model_name)
    
    axes[0].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.suptitle('ResNet18 vs EfficientNet-B0', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Model comparison saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# NASA INFERENCE DEMO
# ═══════════════════════════════════════════════════════════════

def demo_nasa_inference(model_name: str = None):
    """Run the trained model on NASA images for demonstration.
    
    This showcases the model's ability to classify real-world galaxy
    images from NASA that it has never seen during training.
    
    Args:
        model_name: Model to use for inference
    """
    model_name = model_name or config.DEFAULT_MODEL
    
    print("\n" + "═"*60)
    print("  NASA IMAGE INFERENCE DEMO")
    print("═"*60)
    
    if not config.NASA_DIR.exists():
        print("  ✗ No NASA images found. Run 'python download_data.py' first.")
        return
    
    nasa_images = list(config.NASA_DIR.glob("*.jpg"))
    if not nasa_images:
        print("  ✗ No NASA images found.")
        return
    
    print(f"  Found {len(nasa_images)} NASA images")
    
    # Load model once
    checkpoint_path = config.MODELS_DIR / f"best_{model_name}.pth"
    model = create_model(name=model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Classify each NASA image
    results = []
    for img_path in nasa_images[:12]:  # Max 12 for visualization
        result = predict_image(
            str(img_path), 
            model=model, 
            show_gradcam=True
        )
        results.append(result)
    
    print(f"\n  ✓ NASA inference demo complete ({len(results)} images classified)")
    return results
