# Unified V3 Pipeline: End-to-End Guide

This document details the **Unified V3** pipeline, which integrates multiple fundus image datasets (ODIR, EyePACS, Cataract, PALM, ADAM, etc.) into a single, robust training framework. This pipeline is optimized for Apple Silicon (MPS) and ensures strict patient-level isolation to prevent data leakage.

---

## 1. Dataset Overview (Unified V3)

The Unified V3 dataset combines the following sources to address class imbalance and improve rare disease detection:

| Dataset | Primary Condition | Images | Role |
|---------|-------------------|--------|------|
| **ODIR-5K** | Multi-label (8 classes) | ~6,400 | Base dataset (real-world distribution) |
| **EyePACS** | Diabetic Retinopathy | ~35,000 | Massive DR pre-training |
| **Cataract (Kaggle)** | Cataract | ~1,000 | Boosts Cataract class |
| **PALM** | Pathologic Myopia | ~1,200 | Boosts Myopia class |
| **ADAM** | AMD | ~800 | Boosts AMD class |
| **HRF** | Hypertensive Retinopathy | ~45 | High-res reference for HTN |

**Total Size:** ~47,000+ images from ~32,000+ patients.

---

## 2. Pipeline Steps

### Step 1: Data Preprocessing & Merging
**Script:** `scripts/data_prep/preprocess_dataset.py`

This script loads each raw dataset, standardizes the labels to the ODIR-5K format (D, G, C, A, H, M, O), and merges them into a master CSV.

```bash
python scripts/data_prep/preprocess_dataset.py
```
**Output:** `data/processed/unified_v3/unified_dataset_v3.csv`

### Step 2: Stratified Splitting
**Script:** `scripts/data_prep/create_unified_splits_v3.py`

This script creates patient-level Train/Val/Test splits.
*   **Critical Feature:** It uses a "priority stratification" logic (HTN > Cataract > AMD...) to ensure rare diseases are evenly distributed across splits.
*   **Leakage Prevention:** Ensures no patient ID appears in both Train and Test.

```bash
python scripts/data_prep/create_unified_splits_v3.py
```
**Outputs:**
*   `data/processed/unified_v3/unified_train_v3.csv`
*   `data/processed/unified_v3/unified_val_v3.csv`
*   `data/processed/unified_v3/unified_test_v3.csv`

### Step 3: Training (Optimized for Apple Silicon)
**Script:** `scripts/training/train.py`

This is a custom "Pure PyTorch" training loop designed to bypass PyTorch Lightning issues on macOS (MPS).

**Key Features:**
*   **Global Patient IDs:** Automatically generates IDs like `EyePACS_13701` to prevent ID collisions between datasets.
*   **Dynamic Stratification:** Groups rare label combinations into a `rare_combination` bucket during K-Fold cross-validation to prevent errors.
*   **Focal Loss:** Uses `gamma=2.0` (and `gamma_rare=3.0`) to focus learning on hard/rare examples.
*   **Class Weights:** `[1.0, 5.0, 8.0, 60.0, 150.0, 60.0, 1.0]` to heavily penalize missing rare diseases (HTN, AMD).

```bash
# Run training
python scripts/training/train.py
```

### Step 4: Audit & Validation
**Script:** `scripts/analysis/audit_dataset.py`

Run this script at any time to verify data integrity. It checks:
1.  **Image Existence:** Are all files present on disk?
2.  **Patient Grouping:** Are left/right eyes of the same patient (e.g., `13701_left`, `13701_right`) correctly grouped?
3.  **Data Leakage:** Is there any overlap between Train/Val folds?

```bash
python scripts/analysis/audit_dataset.py
```

---

## 3. Directory Structure

```
ODRV2/
├── scripts/
│   ├── training/
│   │   └── train.py            # Main training loop
│   ├── data_prep/
│   │   ├── preprocess_dataset.py       # Merges datasets
│   │   └── create_unified_splits_v3.py # Creates Train/Val/Test
│   ├── analysis/
│   │   └── audit_dataset.py    # Integrity checks
│   ├── evaluation/             # Evaluation scripts
│   └── download/               # Dataset downloaders
├── data/
│   └── processed/
│       └── unified_v3/         # CSV files reside here
└── models/
    └── unified_v3/             # Checkpoints saved here
```

## 4. Common Issues & Fixes

*   **MPS Freezing:** If the training freezes on Mac, ensure you are using `scripts/training/train.py` (Pure PyTorch) and NOT the old Lightning scripts.
*   **Patient ID Collisions:** The pipeline automatically handles this by creating `global_patient_id` (e.g., `ODIR_10` vs `EyePACS_10`).
*   **Missing Images:** Run `scripts/analysis/audit_dataset.py` to identify missing files.
