# ODIR Fundus Classification Pipeline

## Overview

This project trains a multi-label classifier for ocular disease identification using the ODIR fundus dataset. It implements a full end-to-end workflow covering label generation, patient-level fold assignment, deterministic image preprocessing, quality-control artifacts, model training with metadata fusion, and MLflow experiment tracking.

The default configuration targets seven diagnostic classes ("D", "G", "C", "A", "H", "M", "O") using paired left/right eye images and patient metadata.

## Repository Highlights

- `RAW DATA FULL/full_df.csv` – source metadata and diagnostic keywords.
- `scripts/build_labels.py` – converts free-text diagnostics into eye-level multi-hot labels.
- `scripts/prepare_data.py` – assigns deterministic five-fold patient splits.
- `scripts/pretrain_checks.py` – regenerates labels/folds and exports QC plots plus preprocessed image samples.
- `src/preprocessing/pipeline.py` – deterministic fundus loading, crop, and vignette preprocessing.
- `src/augmentation/fundus.py` – train/validation Albumentations transforms.
- `src/datamodules` – PyTorch Lightning data module wrapping the ODIR dataset.
- `src/models` – Dual-eye ConvNeXt backbone and metadata-conditioned classification head.
- `src/training` – CLI entry point, Lightning module, and training loop with MLflow logging.
- `configs/` – Hydra-style configuration with hardware, datamodule, and training defaults.

## Data Preparation Workflow

1. **Label generation** (`scripts/build_labels.py`):
   - Parses left/right diagnostic keyword fields.
   - Applies regex heuristics per class via `src/utils/label_parser.py`.
   - Produces eye-specific and union labels, saved to `data/processed/odir_eye_labels.csv`.
2. **Patient split** (`scripts/prepare_data.py`):
   - Shuffles patients with a fixed seed and assigns five folds.
   - Saves to `data/processed/odir_folds.csv`, ensuring train/validation separation by patient.
3. **QC artifacts** (`scripts/pretrain_checks.py`):
   - Rebuilds labels and folds to guarantee freshness.
   - Exports class, sex, and fold distributions plus age histograms to `data/processed/qc/`.
   - Samples a validation batch, runs the preprocessing stack, and writes `preprocessed_examples.png`.

## Image Preprocessing & Augmentation

- **Loading**: `cv2.imread` followed by RGB conversion.
- **Scaling & cropping**: Images are scaled so the retina fills the 512×512 frame, then center-cropped.
- **Circular vignette**: A soft mask attenuates corner pixels while retaining the retinal disc.
- **Normalization**: Inputs are scaled to `[0, 1]` and later normalized with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Train-time augmentations** (`build_training_augmentations`): Random resized crop, flips, affine jitter, brightness/contrast, CLAHE, hue/saturation, Gaussian noise, and coarse dropout.
- **Validation transforms**: Deterministic resize and normalization only.

The pipeline is deterministic across runs except for the stochastic augmentations applied during training.

## Data Module & Batching

`src/datamodules/lightning.py` wraps the dataset into a Lightning `DataModule`:

- Loads patient metadata and fold assignments.
- Applies the preprocessing pipeline to both eyes per sample.
- Converts age (float) and sex (binary) metadata into tensors.
- Builds train/validation dataloaders with configurable workers, pin-memory, and persistent-worker flags.
- Handles fold assignment automatically if the CSV lacks a `fold` column, ensuring reproducibility.

## Model Architecture

- **Backbone**: `DualFundusBackbone` leverages a shared ConvNeXt (via `timm`) for left and right images. Features are concatenated and projected into a 1024-D embedding.
- **Metadata conditioner**: `MetadataConditioner` learns per-feature gating from age/sex metadata via a small MLP, enabling metadata-aware scaling of visual embeddings.
- **Classifier head**: A dropout-regularized linear layer outputs seven logits for multi-label classification.

## Training Loop

- **Loss**: `nn.BCEWithLogitsLoss` with optional `pos_weight` tensor (class weights read from `configs/training.yaml`) to mitigate class imbalance.
- **Optimization**: AdamW + cosine annealing scheduler.
- **Framework**: PyTorch Lightning handles training/validation loops and integration with the data module.
- **Logging**: MLflow experiment `fundus-odir` captures losses and hyperparameters. Checkpoints are stored under `models/` (`model.ckpt`).
- **Hardware**: Default config targets Apple Silicon (`accelerator: mps`), but CPU/GPU can be selected by editing `configs/training.yaml`.

## Running the Pipeline

1. **Regenerate labels & QC** (recommended before training):

   ```bash
   PYTHONPATH=. .venv/bin/python scripts/pretrain_checks.py
   ```

2. **Launch training** (uses current config):

   ```bash
   PYTHONPATH=. .venv/bin/python src/training/cli.py
   ```

3. **Inspect outputs**:

   - QC artifacts: `data/processed/qc/`
   - Processed CSVs: `data/processed/`
   - MLflow logs: default tracking directory or configured server.
   - Model checkpoint: `models/model.ckpt`

## Advantages

- **Deterministic preprocessing** ensures consistent crops and reproducible QC snapshots.
- **Dual-eye feature sharing** extracts symmetric information while remaining parameter-efficient.
- **Metadata fusion** incorporates age/sex, capturing clinically relevant priors.
- **Built-in QC pipeline** surfaces label distributions and visual samples before training begins.
- **Class imbalance controls** via configurable `pos_weight` reduce dominance of common labels.
- **Hydra-style configs** simplify experimenting with different hardware, folds, or backbone settings.

## Limitations

- **Label heuristics** rely on regex patterns; mis-typed or rare diagnoses may be missed or mislabeled.
- **Fold assignment** is a simple shuffled split without enforcing class stratification or patient demographics.
- **Evaluation metrics** currently limited to loss; additional AUROC/AP tracking would give richer insight.
- **Augmentation symmetry**: left/right eyes are augmented independently, which may disrupt paired anatomical correspondence.
- **Metadata scope**: only age and binary sex are used; other clinical fields remain untapped.
- **Hardware-specific tuning**: defaults optimize for macOS/MPS; CUDA users may want to revisit precision and worker settings.

## Suggested Extensions

- Introduce stratified patient splits or cross-validation scheduling.
- Add richer diagnostics (AUROC, sensitivity/specificity per class) and confusion matrices.
- Experiment with co-teaching or attention mechanisms to better fuse bilateral context.
- Incorporate additional metadata (e.g., medical history) if available.
- Integrate advanced illumination correction or green-channel emphasis for retina-specific contrast.

## License

Ensure you have permission to use the ODIR dataset and follow its associated license terms. This repository assumes data access has been granted.
