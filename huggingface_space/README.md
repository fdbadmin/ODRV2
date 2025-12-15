---
title: ODRV2 - Ocular Disease Recognition
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# ODRV2 - Ocular Disease Recognition

Multi-label fundus image classification for screening 7 ocular conditions:

- **Diabetic Retinopathy**
- **Glaucoma** 
- **Cataract**
- **Age-related Macular Degeneration (AMD)**
- **Hypertensive Retinopathy**
- **Pathological Myopia**
- **Other Pathology**

## Model Details

- **Architecture**: ConvNeXt-Base (88.6M parameters)
- **Training**: 5-fold cross-validation, pruned to 3-fold ensemble
- **Performance**: 0.82 Macro F1 on held-out test set
- **Datasets**: ODIR, APTOS, RFMiD, and other public fundus datasets

## Usage

1. Upload a fundus photograph
2. Click "Analyze Image"
3. View disease probabilities and attention heatmaps

## Disclaimer

⚠️ This is a research/screening tool only. Results must be verified by a qualified ophthalmologist.
