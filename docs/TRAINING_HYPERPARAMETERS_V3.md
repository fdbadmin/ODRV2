# Unified V3 Training Hyperparameters

This document details the hyperparameters and configuration used for training the **Unified V3** Ocular Disease Recognition model.

## 🏗️ Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Backbone** | `convnext_base` | Pretrained on ImageNet-1K |
| **Feature Dimension** | `1024` | Output dimension of the backbone |
| **Dropout** | `0.3` | Applied before the classifier head |
| **Input Resolution** | `448 x 448` | High-resolution fundus images |
| **Pretrained** | `True` | Transfer learning from ImageNet |

## ⚙️ Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| **Batch Size** | `8` | Optimized for Apple Silicon (MPS) memory |
| **Epochs** | `30` | Maximum training epochs per fold |
| **Learning Rate** | `2e-4` | Initial learning rate |
| **Weight Decay** | `1e-2` | L2 regularization |
| **Optimizer** | `AdamW` | Adaptive moment estimation with weight decay |
| **Scheduler** | `CosineAnnealingLR` | Decays LR to 0 over 30 epochs |
| **Early Stopping** | `5` epochs | Patience for validation F1 score |
| **Cross-Validation** | `5-Fold` | Stratified by patient ID |

## 📉 Loss Function

We use a custom **Adaptive Focal Loss** to handle the extreme class imbalance in the unified dataset.

- **Base Gamma ($\gamma$)**: `2.0` (Standard Focal Loss)
- **Rare Disease Gamma ($\gamma_{rare}$)**: `3.0` (Aggressive focusing for hard examples)
- **Rare Classes**: Cataract, AMD, Hypertension, Myopia

### Class Weights ($\alpha$)

Weights are applied to the loss function to penalize errors on rare classes more heavily.

| Class | Weight | Rationale |
|-------|--------|-----------|
| **Diabetes (D)** | `1.0` | Abundant samples (~30k) |
| **Glaucoma (G)** | `5.0` | Moderate imbalance |
| **Cataract (C)** | `8.0` | Moderate imbalance |
| **AMD (A)** | `60.0` | Rare condition |
| **Hypertension (H)** | `150.0` | Extremely rare (~200 samples) |
| **Myopia (M)** | `60.0` | Rare condition |
| **Other (O)** | `1.0` | Common class |

## 🖥️ Hardware Optimization (Apple Silicon)

The training loop is specifically optimized for Apple Silicon chips (M1/M2/M3/M4/M5+) using the Metal Performance Shaders (MPS) backend.

- **Device**: `mps` (Metal Performance Shaders)
- **Num Workers**: `2` (Balanced for I/O vs Compute)
- **Pin Memory**: `False` (Disabled for MPS stability)
- **Thread Count**: `4` (Optimized for performance cores)

## 🔄 Data Augmentation

### Training (Heavy Augmentation)

- **Resize**: 448x448
- **Horizontal Flip**: p=0.5
- **Vertical Flip**: p=0.5
- **Random Rotate**: ±30 degrees
- **Color Jitter**: Brightness, Contrast (mild)
- **Normalization**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Validation/Test

- **Resize**: 448x448
- **Normalization**: ImageNet stats
