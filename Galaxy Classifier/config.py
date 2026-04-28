"""
config.py — Galaxy Classifier Configuration
============================================
Central configuration file for all hyperparameters, paths, and settings.
Modify this file to customize the entire pipeline without touching other modules.

Architecture:
    - All magic numbers live here
    - Other modules import from config
    - Override via CLI args in individual scripts

Author: Galaxy Classifier Project
"""

import sys
import torch
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════
SEED = 42

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "galaxy_zoo" / "raw"
NASA_DIR = DATA_DIR / "nasa_images"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CATALOG_PATH = DATA_DIR / "galaxy_zoo" / "catalog.csv"
SOFT_LABELS_PATH = DATA_DIR / "soft_labels.csv"

# ═══════════════════════════════════════════════════════════════
# CLASSES
# ═══════════════════════════════════════════════════════════════
CLASSES = ["elliptical", "spiral", "irregular"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ═══════════════════════════════════════════════════════════════
# DATASET PARAMETERS
# ═══════════════════════════════════════════════════════════════
IMAGES_PER_CLASS = 500          # Max images to download per class
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

# Label thresholds (Galaxy Zoo 2 debiased vote fractions)
# Logic:
#   if smooth > ELLIPTICAL_THRESHOLD  → elliptical
#   elif features > SPIRAL_THRESHOLD  → spiral
#   elif odd_yes > IRREGULAR_THRESHOLD → irregular
#   else → discard (ambiguous)
ELLIPTICAL_THRESHOLD = 0.7
SPIRAL_THRESHOLD = 0.7
IRREGULAR_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ═══════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════
AVAILABLE_MODELS = ["resnet18", "efficientnet_b0"]
DEFAULT_MODEL = "resnet18"
PRETRAINED = True
FREEZE_BACKBONE = False  # Set True to freeze backbone initially

# ═══════════════════════════════════════════════════════════════
# SOFT LABELS (BONUS)
# ═══════════════════════════════════════════════════════════════
USE_SOFT_LABELS = False  # Use GZ2 vote probabilities instead of hard labels

# ═══════════════════════════════════════════════════════════════
# SDSS DATA ACCESS
# ═══════════════════════════════════════════════════════════════
SDSS_SQL_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
SDSS_CUTOUT_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
SDSS_SCALE = 0.4        # arcsec/pixel
SDSS_IMG_SIZE = 424      # pixels (downloaded, then resized to IMAGE_SIZE)

# ═══════════════════════════════════════════════════════════════
# NASA API (inference/demo ONLY — NOT used for training)
# ═══════════════════════════════════════════════════════════════
NASA_API_URL = "https://images-api.nasa.gov/search"
NASA_SEARCH_QUERIES = [
    "spiral galaxy", "elliptical galaxy", "irregular galaxy",
    "galaxy hubble", "galaxy NGC"
]
NASA_MAX_IMAGES = 50

# ═══════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
