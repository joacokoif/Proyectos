"""
run_pipeline.py — End-to-End Pipeline
======================================
Runs the complete galaxy classification pipeline:
    1. Download data (Galaxy Zoo 2 + NASA)
    2. Prepare dataset (labels + splits)
    3. Train model(s)
    4. Evaluate
    5. Generate visualizations
    6. Run NASA inference demo

Usage:
    python run_pipeline.py                    # Full pipeline, ResNet18
    python run_pipeline.py --model efficientnet_b0
    python run_pipeline.py --compare          # Train & compare both models
    python run_pipeline.py --skip-download    # Skip if data exists

Author: Galaxy Classifier Project
"""

import argparse
import sys
from pathlib import Path

import config
from train import set_seed


def run_pipeline(
    model_name: str = None,
    compare: bool = False,
    skip_download: bool = False,
    per_class: int = None
):
    """Execute the full ML pipeline."""
    
    model_name = model_name or config.DEFAULT_MODEL
    per_class = per_class or config.IMAGES_PER_CLASS
    
    set_seed(config.SEED)
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  🌌 GALAXY CLASSIFIER — FULL PIPELINE".center(58) + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # ─── Step 1: Download Data ───
    if not skip_download:
        print("\n" + "▶ STEP 1/5: Downloading Data")
        print("─" * 60)
        
        from download_data import download_galaxy_zoo_data, download_nasa_images
        
        if not config.CATALOG_PATH.exists():
            download_galaxy_zoo_data(images_per_class=per_class)
        else:
            print("  ✓ Galaxy Zoo data already exists, skipping download")
        
        if not config.NASA_DIR.exists() or len(list(config.NASA_DIR.glob("*.jpg"))) == 0:
            download_nasa_images()
        else:
            print("  ✓ NASA images already exist, skipping download")
    else:
        print("  ⏭ Skipping download (--skip-download)")
    
    # ─── Step 2: Prepare Dataset ───
    print("\n" + "▶ STEP 2/5: Preparing Dataset")
    print("─" * 60)
    
    from prepare_dataset import main as prepare_main
    
    if not config.TRAIN_DIR.exists() or len(list(config.TRAIN_DIR.rglob("*.jpg"))) == 0:
        # Use sys.argv trick to call prepare_main with no extra args
        old_argv = sys.argv
        sys.argv = ['prepare_dataset.py', '--soft-labels']
        prepare_main()
        sys.argv = old_argv
    else:
        print("  ✓ Dataset already prepared, skipping")
    
    # ─── Step 3: Train ───
    print("\n" + "▶ STEP 3/5: Training Model(s)")
    print("─" * 60)
    
    from train import train_model
    
    if compare:
        models_to_train = config.AVAILABLE_MODELS
    else:
        models_to_train = [model_name]
    
    for m_name in models_to_train:
        checkpoint = config.MODELS_DIR / f"best_{m_name}.pth"
        if checkpoint.exists():
            print(f"  ✓ {m_name} already trained, skipping")
            continue
        train_model(model_name=m_name)
    
    # ─── Step 4: Evaluate ───
    print("\n" + "▶ STEP 4/5: Evaluating Model(s)")
    print("─" * 60)
    
    from evaluate import evaluate_model
    
    for m_name in models_to_train:
        evaluate_model(m_name)
    
    # ─── Step 5: Visualizations ───
    print("\n" + "▶ STEP 5/5: Generating Visualizations")
    print("─" * 60)
    
    from utils import (
        plot_training_history, 
        gradcam_gallery, 
        compare_models as compare_models_fn,
        demo_nasa_inference
    )
    from dataset import GalaxyDataset
    from evaluate import load_trained_model
    from dataset import get_transforms
    
    for m_name in models_to_train:
        plot_training_history(m_name)
        
        # Grad-CAM gallery
        model = load_trained_model(m_name)
        test_dataset = GalaxyDataset(config.TEST_DIR, transform=get_transforms("test"))
        gradcam_gallery(model, test_dataset, m_name)
    
    if compare:
        compare_models_fn()
    
    # NASA inference demo
    demo_nasa_inference(models_to_train[0])
    
    # ─── Done ───
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  ✅ PIPELINE COMPLETE!".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Results saved in: {config.RESULTS_DIR}")
    print(f"  Models saved in:  {config.MODELS_DIR}")
    print(f"\n  To deploy: streamlit run app.py")
    print(f"  To predict: python -c \"from utils import predict_image; predict_image('image.jpg')\"")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete galaxy classification pipeline"
    )
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=config.AVAILABLE_MODELS
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Train and compare both models"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download step"
    )
    parser.add_argument(
        "--per-class", type=int, default=config.IMAGES_PER_CLASS,
        help=f"Images per class (default: {config.IMAGES_PER_CLASS})"
    )
    args = parser.parse_args()
    
    run_pipeline(
        model_name=args.model,
        compare=args.compare,
        skip_download=args.skip_download,
        per_class=args.per_class
    )


if __name__ == "__main__":
    main()
