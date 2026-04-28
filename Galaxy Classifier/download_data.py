"""
download_data.py — Dataset Acquisition
=======================================
Downloads Galaxy Zoo 2 data (vote table + SDSS images) for training,
and NASA images for inference/demo purposes.

┌─────────────────────────────────────────────────────────────────┐
│  NOTE: NASA images are used ONLY for inference/demo purposes.  │
│  They are NOT used for training due to lack of reliable labels.│
└─────────────────────────────────────────────────────────────────┘

Data Flow:
    1. Query SDSS for Galaxy Zoo 2 vote fractions + coordinates
    2. Download SDSS cutout images for labeled galaxies
    3. Save catalog CSV with vote fractions + image paths
    4. Download NASA images separately for inference demos

Usage:
    python download_data.py                    # Download default subset
    python download_data.py --per-class 1000   # Download 1000 per class
    python download_data.py --nasa-only        # Only NASA demo images

Author: Galaxy Classifier Project
"""

import os
import io
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config


# ═══════════════════════════════════════════════════════════════
# SDSS SQL QUERIES — Galaxy Zoo 2 Data
# ═══════════════════════════════════════════════════════════════
# These queries fetch galaxies from the Galaxy Zoo 2 table in SDSS DR17.
# We use debiased vote fractions for reliable morphological classification.

def _build_query(where_clause: str, n: int) -> str:
    """Build an SDSS SQL query for Galaxy Zoo 2 data.
    
    The zoo2MainSpecz table contains debiased vote fractions from
    Galaxy Zoo 2 (Willett et al. 2013, Hart et al. 2016).
    """
    return f"""
    SELECT TOP {n}
        gz.dr7objid as objid,
        gz.ra, gz.dec,
        gz.t01_smooth_or_features_a01_smooth_debiased as smooth,
        gz.t01_smooth_or_features_a02_features_or_disk_debiased as features,
        gz.t06_odd_a14_yes_debiased as odd,
        gz.t08_odd_feature_a22_irregular_debiased as irregular_feat,
        gz.t01_smooth_or_features_a01_smooth_count as smooth_count,
        gz.t01_smooth_or_features_a02_features_or_disk_count as features_count,
        gz.t06_odd_a14_yes_count as odd_count
    FROM zoo2MainSpecz AS gz
    WHERE {where_clause}
    ORDER BY gz.ra
    """


# Elliptical: smooth > threshold, high vote count
ELLIPTICAL_QUERY = lambda n: _build_query(
    f"gz.t01_smooth_or_features_a01_smooth_debiased > {config.ELLIPTICAL_THRESHOLD} "
    f"AND gz.t01_smooth_or_features_a01_smooth_count > 20",
    n
)

# Spiral: features/disk > threshold, high vote count
SPIRAL_QUERY = lambda n: _build_query(
    f"gz.t01_smooth_or_features_a02_features_or_disk_debiased > {config.SPIRAL_THRESHOLD} "
    f"AND gz.t01_smooth_or_features_a02_features_or_disk_count > 20",
    n
)

# Irregular: odd features > threshold (per user's feedback — more faithful to GZ)
# Uses t06_odd_a14_yes ("is anything odd?") + t08_odd_feature_a22_irregular
IRREGULAR_QUERY = lambda n: _build_query(
    f"gz.t06_odd_a14_yes_debiased > {config.IRREGULAR_THRESHOLD} "
    f"AND gz.t06_odd_a14_yes_count > 10",
    n
)


# ═══════════════════════════════════════════════════════════════
# SDSS DATA FETCHING
# ═══════════════════════════════════════════════════════════════

def query_sdss(sql_query: str) -> pd.DataFrame:
    """Execute a SQL query against SDSS DR17 SkyServer.
    
    Args:
        sql_query: Valid SDSS SQL query string
        
    Returns:
        DataFrame with query results
        
    Raises:
        ConnectionError: If SDSS server is unreachable
    """
    params = {
        "cmd": sql_query,
        "format": "csv"
    }
    
    print(f"  -> Querying SDSS SkyServer...")
    response = requests.get(config.SDSS_SQL_URL, params=params, timeout=120)
    
    # Handle JSON error responses from SDSS
    if response.status_code != 200:
        try:
            err = response.json()
            raise ValueError(f"SDSS error: {err.get('ErrorMessage', response.text[:500])}")
        except (ValueError, KeyError):
            response.raise_for_status()
    
    # SDSS CSV has a '#Table1' header line — skip it
    text = response.text
    lines = text.strip().split('\n')
    csv_lines = [l for l in lines if not l.startswith('#')]
    csv_text = '\n'.join(csv_lines)
    
    df = pd.read_csv(io.StringIO(csv_text))
    
    # Check for error messages from SDSS
    if df.empty or (len(df.columns) == 1 and "ERROR" in str(df.columns[0])):
        raise ValueError(f"SDSS query failed: {response.text[:500]}")
    
    return df


def download_sdss_image(objid: int, ra: float, dec: float, save_path: Path) -> bool:
    """Download a galaxy cutout image from SDSS SkyServer.
    
    Uses the SDSS Image Cutout Service to fetch a JPEG image centered
    on the given RA/Dec coordinates.
    
    Args:
        objid: SDSS DR7 Object ID
        ra: Right Ascension (degrees)
        dec: Declination (degrees)  
        save_path: Where to save the image
        
    Returns:
        True if download succeeded, False otherwise
    """
    if save_path.exists():
        return True
    
    params = {
        "ra": ra,
        "dec": dec,
        "scale": config.SDSS_SCALE,
        "width": config.SDSS_IMG_SIZE,
        "height": config.SDSS_IMG_SIZE,
        "opt": ""  # No overlay annotations
    }
    
    try:
        response = requests.get(config.SDSS_CUTOUT_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Verify it's a valid image
        img = Image.open(io.BytesIO(response.content))
        if img.size[0] < 50 or img.size[1] < 50:
            return False
        
        # Save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, "JPEG", quality=95)
        return True
        
    except Exception:
        return False


def download_gz2_class(class_name: str, query_fn, n_images: int) -> pd.DataFrame:
    """Download Galaxy Zoo 2 images for a specific morphological class.
    
    Args:
        class_name: "elliptical", "spiral", or "irregular"
        query_fn: Function that returns SQL query given count
        n_images: Number of images to download
        
    Returns:
        DataFrame with successfully downloaded galaxies
    """
    print(f"\n{'='*60}")
    print(f"  Downloading {class_name.upper()} galaxies ({n_images} images)")
    print(f"{'='*60}")
    
    # Request extra to account for download failures
    query_count = int(n_images * 1.3)
    
    # Query SDSS for galaxy data
    try:
        df = query_sdss(query_fn(query_count))
        print(f"  ✓ Found {len(df)} candidates in Galaxy Zoo 2")
    except Exception as e:
        print(f"  ✗ SDSS query failed: {e}")
        return pd.DataFrame()
    
    # Download images
    save_dir = config.RAW_DIR / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    
    with tqdm(total=min(n_images, len(df)), desc=f"  Downloading {class_name}") as pbar:
        for _, row in df.iterrows():
            if len(successful) >= n_images:
                break
            
            filename = f"{int(row['objid'])}.jpg"
            save_path = save_dir / filename
            
            if download_sdss_image(
                objid=int(row['objid']),
                ra=float(row['ra']),
                dec=float(row['dec']),
                save_path=save_path
            ):
                row_data = row.to_dict()
                row_data['filename'] = filename
                row_data['class'] = class_name
                row_data['filepath'] = str(save_path)
                successful.append(row_data)
                pbar.update(1)
            
            # Rate limiting — be polite to SDSS servers
            time.sleep(0.1)
    
    print(f"  ✓ Downloaded {len(successful)}/{n_images} {class_name} images")
    return pd.DataFrame(successful)


def download_galaxy_zoo_data(images_per_class: int = None) -> pd.DataFrame:
    """Download complete Galaxy Zoo 2 dataset from SDSS.
    
    Queries the GZ2 debiased vote table and downloads SDSS cutout images
    for the three target classes: elliptical, spiral, irregular.
    
    Args:
        images_per_class: Number of images per class (default: config value)
        
    Returns:
        Combined catalog DataFrame
    """
    n = images_per_class or config.IMAGES_PER_CLASS
    
    print("\n" + "═"*60)
    print("  GALAXY ZOO 2 — DATASET DOWNLOAD")
    print("  Training dataset with citizen-science labels")
    print("═"*60)
    
    # Download each class
    queries = {
        "elliptical": ELLIPTICAL_QUERY,
        "spiral": SPIRAL_QUERY,
        "irregular": IRREGULAR_QUERY
    }
    
    all_data = []
    for class_name, query_fn in queries.items():
        df = download_gz2_class(class_name, query_fn, n)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("\n  ✗ No data downloaded. Check your internet connection.")
        print("    Alternatively, see README.md for manual download instructions.")
        return pd.DataFrame()
    
    # Combine and save catalog
    catalog = pd.concat(all_data, ignore_index=True)
    config.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(config.CATALOG_PATH, index=False)
    
    print(f"\n  ✓ Catalog saved: {config.CATALOG_PATH}")
    print(f"  ✓ Total images: {len(catalog)}")
    print(f"\n  Class distribution:")
    for cls, count in catalog['class'].value_counts().items():
        print(f"    {cls:>12}: {count} images")
    
    return catalog


# ═══════════════════════════════════════════════════════════════
# NASA IMAGE LIBRARY — Demo/Inference Images
# ═══════════════════════════════════════════════════════════════
# NOTE: These images are used ONLY for inference/demo purposes.
# They are NOT used for training due to lack of reliable labels.

def download_nasa_images(max_images: int = None) -> list:
    """Download galaxy images from NASA Image and Video Library.
    
    These are real NASA/Hubble/James Webb galaxy images used to
    demonstrate the model's inference capability on new, unseen data.
    
    ┌────────────────────────────────────────────────────────────┐
    │  IMPORTANT: NASA images are for INFERENCE/DEMO ONLY.      │
    │  They have no reliable morphological labels for training.  │
    └────────────────────────────────────────────────────────────┘
    
    Args:
        max_images: Maximum number of images to download
        
    Returns:
        List of downloaded image paths
    """
    n = max_images or config.NASA_MAX_IMAGES
    save_dir = config.NASA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "═"*60)
    print("  NASA IMAGE LIBRARY — Demo Images")
    print("  For inference/demo ONLY (no training labels)")
    print("═"*60)
    
    downloaded = []
    
    for query in config.NASA_SEARCH_QUERIES:
        if len(downloaded) >= n:
            break
        
        print(f"\n  Searching: '{query}'...")
        
        try:
            params = {
                "q": query,
                "media_type": "image",
                "page_size": min(20, n - len(downloaded))
            }
            
            response = requests.get(config.NASA_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("collection", {}).get("items", [])
            
            for item in items:
                if len(downloaded) >= n:
                    break
                
                links = item.get("links", [])
                metadata = item.get("data", [{}])[0]
                title = metadata.get("title", "unknown").replace(" ", "_")[:50]
                nasa_id = metadata.get("nasa_id", "unknown")
                
                if not links:
                    continue
                
                img_url = links[0].get("href", "")
                if not img_url:
                    continue
                
                # Download the image
                try:
                    img_response = requests.get(img_url, timeout=30)
                    img_response.raise_for_status()
                    
                    filename = f"nasa_{nasa_id}.jpg"
                    save_path = save_dir / filename
                    
                    img = Image.open(io.BytesIO(img_response.content))
                    img = img.convert("RGB")
                    img.save(save_path, "JPEG", quality=95)
                    
                    downloaded.append({
                        "filepath": str(save_path),
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", ""),
                        "nasa_id": nasa_id
                    })
                    
                except Exception:
                    continue
                
                time.sleep(0.2)  # Rate limiting
        
        except Exception as e:
            print(f"  ✗ Search failed for '{query}': {e}")
            continue
    
    # Save NASA metadata
    if downloaded:
        nasa_df = pd.DataFrame(downloaded)
        nasa_df.to_csv(config.NASA_DIR / "nasa_metadata.csv", index=False)
        print(f"\n  ✓ Downloaded {len(downloaded)} NASA images → {save_dir}")
    
    return downloaded


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download Galaxy Zoo 2 + NASA galaxy images"
    )
    parser.add_argument(
        "--per-class", type=int, default=config.IMAGES_PER_CLASS,
        help=f"Images per class (default: {config.IMAGES_PER_CLASS})"
    )
    parser.add_argument(
        "--nasa-only", action="store_true",
        help="Only download NASA demo images"
    )
    parser.add_argument(
        "--skip-nasa", action="store_true",
        help="Skip NASA images download"
    )
    args = parser.parse_args()
    
    if not args.nasa_only:
        download_galaxy_zoo_data(args.per_class)
    
    if not args.skip_nasa:
        download_nasa_images()
    
    print("\n" + "═"*60)
    print("  ✓ Download complete!")
    print("  Next step: python prepare_dataset.py")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
