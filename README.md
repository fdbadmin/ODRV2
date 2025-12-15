# ODRV2 — Unified Multi-Disease Fundus Screening (Research)

ODRV2 is a research pipeline for **multi-label ocular disease detection** from fundus photographs.

- **Diseases (7):** Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertensive Retinopathy, Myopia, Other
- **Core model:** ConvNeXt-Base backbone + multi-label head
- **Training:** 5-fold CV on Unified Dataset V3; best-performing *pruned ensemble* uses folds **0/1/4**
- **Explainability:** GradCAM-style attention maps

Important: this repository is a **research/screening tool**. It is **not** a medical device. Any outputs must be verified by qualified clinicians and validated externally before clinical use.

## Headline results (Unified Dataset V3 holdout)

From [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) on a **fully held-out** test set (**6,282 images / 4,058 patients**):

- **Macro F1:** **0.8189**
- **Macro AUC-ROC:** **0.9742**

What that means (quickly):

- **F1** balances *precision* (few false positives) and *recall* (few false negatives).
- **Macro** averages the metric across the 7 diseases so each disease has equal weight.

Per-class F1 on the holdout test:

- Diabetic Retinopathy: **0.76**
- Glaucoma: **0.86**
- Cataract: **0.79**
- AMD: **0.85**
- Hypertensive Retinopathy: **0.63**
- Myopia: **0.93**
- Other: **0.91**

Strengths / limitations (at a glance):

- **Strengths:** strong overall discrimination (high macro AUC), particularly strong F1 for Myopia/Other/Glaucoma/AMD.
- **Limitations:** performance varies by disease and operating threshold; rare diseases have small supports; Hypertensive Retinopathy recall is the weakest in the final holdout evaluation; image quality and domain shift can materially affect results.

## Contents

- [At a glance](#at-a-glance)
- [Results](#results)
- [Key metrics explained](#key-metrics-explained)
- [Pipeline overview](#pipeline-overview)
- [Quickstart (inference)](#quickstart-inference)
- [Quickstart (training)](#quickstart-training)
- [Evaluation & audits](#evaluation--audits)
- [Deployments](#deployments)
- [Comparison to published work (optional)](#comparison-to-published-work-optional)
- [Repository map](#repository-map)
- [Documentation index](#documentation-index)
- [Citation](#citation)

## At a glance

**Unified Dataset V3 (current research dataset)**

- **Images:** 32,157
- **Patients:** 20,287
- **Holdout test:** 6,282 images from 4,058 patients (never seen during training)

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the full methodology and final evaluation.

## Results

### Unified V3 holdout (final pruned ensemble)

From [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md):

- **Macro F1:** 0.8189
- **Macro AUC-ROC:** 0.9742
- **Macro precision:** 0.9075
- **Macro recall:** 0.7632

Per-class metrics (precision/recall/F1/AUC) and supports are reported in the technical report.

### External validation (glaucoma-specific)

The final report includes an external evaluation on HYGD (215 images) with strong glaucoma performance (see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)).

## Key metrics explained

This project is **multi-label**: each image can have zero, one, or multiple conditions present. The model outputs a probability per condition; each condition becomes “positive” if its probability exceeds a chosen threshold.

**Confusion-matrix terms (per class)**

- **True positive (TP):** condition present and predicted present
- **False positive (FP):** condition absent but predicted present
- **False negative (FN):** condition present but predicted absent

**Precision / Recall / F1**

- **Precision:** of the images predicted positive, how many truly are positive?  
  $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall (Sensitivity):** of the truly positive images, how many did we detect?  
  $\text{Recall} = \frac{TP}{TP + FN}$
- **F1 score:** harmonic mean of precision and recall (balances FP and FN).  
  $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

**AUC-ROC**

- **AUC-ROC:** threshold-independent measure of ranking quality (how well positives are scored above negatives across all possible thresholds). Higher is better, but it does not by itself choose an operating point.

**Macro averaging (why “Macro F1” matters here)**

- **Macro** metrics compute the metric **per class**, then average across classes (each disease has equal weight). This is useful when the label distribution is imbalanced and you care about performance across all diseases, not just the most common ones.

**Thresholding (operating point)**

- Reported **F1/precision/recall** depends on thresholds. Threshold selection strategy and the final per-class thresholds used for evaluation are documented in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Pipeline overview

The repo contains **two related pipelines**:

1) **Unified V3 “pure PyTorch” training** (recommended on macOS/MPS)
   - Entry: `scripts/training/train.py`
   - Uses patient-level CV folds created from globally unique patient identifiers

2) **Lightning/Hydra-based training + FastAPI inference service** (more production-like)
   - Inference service: `src/inference/service.py`
   - Configs: `configs/*.yaml`

High-level stages:

1. **Ingest + standardize datasets** → build a unified CSV
2. **Patient-level splitting** → train/val/test with leakage prevention
3. **Train 5 folds** (ConvNeXt-Base) → save checkpoints per fold
4. **Prune ensemble** (select strongest folds) → folds 0/1/4
5. **Evaluate on holdout test** → macro metrics + per-class metrics
6. **Explainability/QC** → attention heatmaps + image QC checks

For the step-by-step Unified V3 pipeline, see [docs/UNIFIED_PIPELINE_V3.md](docs/UNIFIED_PIPELINE_V3.md).

## Quickstart (inference)

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Desktop app (PyQt)

```bash
python desktop_app.py
```

### 3) Web demo (Streamlit Space code)

The Streamlit app used for the Hugging Face Space lives in `huggingface_space/`.

```bash
streamlit run huggingface_space/app.py
```

### 4) FastAPI inference service (optional)

There is also a FastAPI service that supports ensemble inference, uncertainty metrics, and GradCAM overlays.
See `src/inference/service.py` for the server factory.

## Quickstart (training)

### Unified V3 training (recommended)

1) Build unified dataset CSV and patient-level splits:

```bash
python scripts/data_prep/preprocess_dataset.py
python scripts/data_prep/create_unified_splits_v3.py
```

2) Train folds:

```bash
python scripts/training/train.py
```

If you want live logging to a file:

```bash
python start_training_monitored.py
```

Notes:

- `scripts/training/train.py` sets up patient IDs from filenames and creates fold assignments.
- The training loop is optimized for macOS MPS and avoids some Lightning/MPS edge cases.

## Evaluation & audits

### Dataset integrity / leakage

Run the dataset audit:

```bash
python scripts/analysis/audit_dataset.py
```

Read the leakage deep-dive and fixes in [DATA_LEAKAGE_AUDIT.md](DATA_LEAKAGE_AUDIT.md).

### QC and explainability

- Desktop: attention heatmaps per disease are generated in `src/desktop/inference.py`.
- Streamlit Space: QC metrics + attention overlay live in `huggingface_space/app.py`.

## Deployments

This repo includes multiple ways to present the model:

- **Desktop (PyQt6):** clinician-friendly local UI
- **Streamlit Space app:** shareable web demo (see `huggingface_space/`)
- **FastAPI service:** programmatic inference endpoint (see `src/inference/service.py`)

## Benchmark comparison to published work

## Comparison to published work (optional)

This repository contains an explicit comparison write-up for **ODIR-5K** experiments:

- [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) compares an ODIR-5K evaluation to selected published results.
- [STATISTICAL_SUMMARY.md](STATISTICAL_SUMMARY.md) contains detailed per-class statistics for that evaluation.

Important context:

- The ODIR-5K comparison documents a specific experimental setting (dataset + split + thresholding) that is **not identical** to the Unified V3 holdout evaluation.
- When comparing to papers, match the **dataset**, **split protocol** (patient-level vs image-level), and **metric definitions** (macro/micro, thresholding, etc.).

## Repository map

```
ODRV2/
├── configs/                  # Hydra configs (Lightning + inference service)
├── data/                     # Raw/processed data and split CSVs
├── docs/                     # Pipeline and integration docs
├── huggingface_space/        # Streamlit Space app (web demo)
├── models/                   # Checkpoints (local)
├── scripts/
│   ├── data_prep/            # Build unified dataset and splits
│   ├── training/             # Training entrypoint(s)
│   ├── analysis/             # Audits and QC/visualization
│   └── evaluation/           # Evaluation utilities
└── src/
    ├── desktop/              # Desktop UI + desktop inference utilities
    ├── inference/            # FastAPI service + GradCAM
    ├── models/               # Model components
    └── training/             # Training utilities (Lightning)
```

## Documentation index

- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — final methodology and results
- [docs/UNIFIED_PIPELINE_V3.md](docs/UNIFIED_PIPELINE_V3.md) — end-to-end Unified V3 pipeline guide
- [DATA_LEAKAGE_AUDIT.md](DATA_LEAKAGE_AUDIT.md) — leakage findings and fixes
- [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) — ODIR-5K comparison to selected published work
- [STATISTICAL_SUMMARY.md](STATISTICAL_SUMMARY.md) — detailed ODIR-5K stats

## Citation

If you use this repository, please cite the software and include the model version + evaluation setting you used.

```bibtex
@software{odrv2_2025,
  title  = {ODRV2: Unified Multi-Disease Fundus Screening (Research)},
  author = {Brandimarte, Fabian},
  year   = {2025},
  url    = {https://github.com/fdbadmin/ODRV2}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🙏 Acknowledgments

- **ODIR-2019 Challenge** organizers (Peking University) for the dataset
- **ConvNeXt** authors (Liu et al., Meta AI) for the architecture
- **PyTorch Lightning** team for the training framework
- **Albumentations** team for augmentation library

---

## 📧 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/fdbadmin/ODRV2/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/fdbadmin/ODRV2/discussions)

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong><br>
  <sub>Built with ❤️ for advancing medical AI and improving eye care worldwide</sub>
</p>
