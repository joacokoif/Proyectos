"""
model.py — Galaxy Classification Models
========================================
Defines model architectures for galaxy morphology classification
using transfer learning from ImageNet-pretrained backbones.

Architecture Overview:
    ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
    │  Pretrained      │     │  Adaptive    │     │  Custom FC  │
    │  Backbone        │ ──→ │  AvgPool     │ ──→ │  Head       │
    │  (frozen/unfrozen)│    │              │     │  → 3 classes│
    └──────────────────┘     └──────────────┘     └─────────────┘

Why Transfer Learning:
    - ImageNet features (edges, textures, shapes) transfer well to galaxies
    - Spiral arms ≈ curved edges, bulges ≈ circular patterns
    - Dramatically reduces training time and data requirements
    - Fine-tuning adapts high-level features to astronomical morphology

Supported Models:
    - ResNet18:       Small, fast, good baseline (11.7M params)
    - EfficientNet-B0: Better accuracy/efficiency ratio (5.3M params)

Author: Galaxy Classifier Project
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# ═══════════════════════════════════════════════════════════════
# GALAXY CLASSIFIER
# ═══════════════════════════════════════════════════════════════

class GalaxyClassifier(nn.Module):
    """Galaxy morphology classifier using transfer learning.
    
    Wraps a pretrained backbone (ResNet18 or EfficientNet-B0) and
    replaces the final classification head with a custom head for
    3-class galaxy classification.
    
    The custom head uses dropout for regularization (galaxies can be
    ambiguous even for human classifiers, so the model should also
    Express uncertainty).
    
    Args:
        backbone_name: "resnet18" or "efficientnet_b0"
        num_classes: Number of output classes (default: 3)
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone weights for initial training
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Load pretrained backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            
            # ResNet18 final FC: Linear(512, 1000) → replace
            in_features = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()  # Remove original head
            
            # Custom classification head
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            
            # For Grad-CAM: target layer is the last conv block
            self.target_layer = self.backbone.layer4[-1]
        
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            
            # EfficientNet-B0 classifier: Linear(1280, 1000) → replace
            in_features = self.backbone.classifier[1].in_features  # 1280
            self.backbone.classifier = nn.Identity()  # Remove original head
            
            # Custom classification head
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            
            # For Grad-CAM: target layer is the last conv block
            self.target_layer = self.backbone.features[-1]
        
        else:
            raise ValueError(
                f"Unknown backbone: '{backbone_name}'. "
                f"Available: {config.AVAILABLE_MODELS}"
            )
        
        # Optionally freeze backbone (only train the classifier head)
        if freeze_backbone:
            self.freeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone + classifier.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def freeze_backbone(self):
        """Freeze all backbone parameters (train only classifier head).
        
        Useful for initial fine-tuning: train the head first, then
        unfreeze and fine-tune the entire network with a lower LR.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"  ❄ Backbone frozen ({self.backbone_name})")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"  🔥 Backbone unfrozen ({self.backbone_name})")
    
    def get_num_params(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }


# ═══════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════

def create_model(
    name: str = None,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
    freeze_backbone: bool = config.FREEZE_BACKBONE
) -> GalaxyClassifier:
    """Factory function to create a galaxy classifier model.
    
    This is the recommended way to instantiate models. It handles
    all configuration and moves the model to the correct device.
    
    Args:
        name: Model name ("resnet18" or "efficientnet_b0")
        num_classes: Number of output classes
        pretrained: Use ImageNet weights
        freeze_backbone: Freeze backbone initially
        
    Returns:
        GalaxyClassifier model on the configured device
    
    Example:
        >>> model = create_model("resnet18")
        >>> model = create_model("efficientnet_b0", freeze_backbone=True)
    """
    model_name = name or config.DEFAULT_MODEL
    
    model = GalaxyClassifier(
        backbone_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(config.DEVICE)
    
    # Report model stats
    params = model.get_num_params()
    print(f"\n  Model: {model_name}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Total params:     {params['total']:>10,}")
    print(f"  Trainable params: {params['trainable']:>10,}")
    print(f"  Frozen params:    {params['frozen']:>10,}")
    
    return model


def get_target_layer(model: GalaxyClassifier):
    """Get the target layer for Grad-CAM visualization.
    
    Returns the last convolutional layer of the backbone,
    which captures the highest-level spatial features.
    
    Args:
        model: GalaxyClassifier instance
        
    Returns:
        Target layer module
    """
    return model.target_layer
