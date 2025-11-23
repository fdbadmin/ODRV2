# ODIR Fundus Classification Pipeline

## Overview

This project trains a multi-label classifier for ocular disease identification using the ODIR fundus dataset. It implements a full end-to-end workflow covering label generation, patient-level fold assignment, deterministic image preprocessing, quality-control artifacts, model training with metadata fusion, and MLflow experiment tracking.

The default configuration targets seven diagnostic classes ("D", "G", "C", "A", "H", "M", "O") using single-eye training samples (left and right eyes expanded separately) alongside patient metadata.

## Repository Highlights

- `RAW DATA FULL/full_df.csv` – source metadata and diagnostic keywords.
- `scripts/build_labels.py` – converts free-text diagnostics into eye-level multi-hot labels.
- `scripts/prepare_data.py` – assigns deterministic five-fold patient splits.
- `scripts/pretrain_checks.py` – regenerates labels/folds and exports QC plots plus preprocessed image samples.
- `src/preprocessing/pipeline.py` – deterministic fundus loading, crop, and vignette preprocessing.
- `src/augmentation/fundus.py` – train/validation Albumentations transforms.
- `src/datamodules` – PyTorch Lightning data module wrapping the ODIR dataset.
- `src/models` – Single-eye ConvNeXt backbone and metadata-conditioned classification head.
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
- **Scaling & cropping**: Images are scaled so the retina fills the 448×448 frame, then center-cropped.
- **Circular vignette**: A soft mask attenuates corner pixels while retaining the retinal disc.
- **Normalization**: Inputs are scaled to `[0, 1]` and later normalized with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Train-time augmentations** (`build_training_augmentations`): Lightweight random resized crop, flips, moderate affine jitter, brightness/contrast, hue/saturation tweaks, Gaussian blur, and low-variance Gaussian noise.
- **Validation transforms**: Deterministic resize and normalization only.

The pipeline is deterministic across runs except for the stochastic augmentations applied during training.

## Data Module & Batching

`src/datamodules/lightning.py` wraps the dataset into a Lightning `DataModule`:

- Loads patient metadata and fold assignments.
- Expands each patient row into two single-eye samples (left/right) while reusing metadata.
- Applies the preprocessing pipeline to each eye independently.
- Converts age (float) and sex (binary) metadata into tensors for conditioning.
- Builds train/validation dataloaders with configurable workers, pin-memory, and persistent-worker flags.
- Handles fold assignment automatically if the CSV lacks a `fold` column, ensuring reproducibility.

## Model Architecture

- **Backbone**: `FundusBackbone` applies a ConvNeXt encoder (via `timm`) to a single fundus image and projects features into a 1024-D embedding.
- **Metadata conditioner**: `MetadataConditioner` learns per-feature gating from age/sex metadata via a small MLP, enabling metadata-aware scaling of visual embeddings.
- **Classifier head**: A dropout-regularized linear layer outputs seven logits for multi-label classification.

## Training Loop

- **Loss**: `nn.BCEWithLogitsLoss` with optional `pos_weight` tensor (class weights read from `configs/training.yaml`) to mitigate class imbalance.
- **Optimization**: AdamW + cosine annealing scheduler.
- **Framework**: PyTorch Lightning handles training/validation loops and integration with the data module.
- **Metrics & logging**: Each validation epoch logs macro F1 and per-class F1 to MLflow alongside losses, with human-readable summaries printed to the console.
- **Early stopping**: Training halts when `val_macro_f1` fails to improve for 10 consecutive epochs.
- **Checkpoints**: Model weights are saved under `models/` (`model.ckpt`).
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
   - Training console: per-epoch `val_macro_f1`, per-class F1 breakdown, and early-stopping status.

## Inference Service

- `src/webapp/main.py` boots a FastAPI app backed by the trained Lightning module.
- `/predict` expects a single fundus image upload alongside age and sex metadata.
- Inputs are preprocessed with the same 448×448 crop/vignette and ImageNet normalization used during training, ensuring consistent logits for deployment.

## Advantages

- **Deterministic preprocessing** ensures consistent crops and reproducible QC snapshots.
- **Single-eye deployment** aligns the training loop with workflows where only one fundus image is available.
- **Metadata fusion** incorporates age/sex, capturing clinically relevant priors.
- **Built-in QC pipeline** surfaces label distributions and visual samples before training begins.
- **Class imbalance controls** via configurable `pos_weight` reduce dominance of common labels.
- **Hydra-style configs** simplify experimenting with different hardware, folds, or backbone settings.
- **Actionable metrics**: Macro and per-class F1 are logged each epoch, and early stopping prevents over-training once improvements stall.

## Limitations

- **Label heuristics** rely on regex patterns; mis-typed or rare diagnoses may be missed or mislabeled.
- **Fold assignment** is a simple shuffled split without enforcing class stratification or patient demographics.
- **Evaluation metrics** focus on macro/per-class F1; additional AUROC/AP tracking would give richer insight.
- **Shared metadata**: duplicate age/sex metadata is assumed adequate for each eye; richer eye-specific context (e.g., laterality flags) is not used.
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
