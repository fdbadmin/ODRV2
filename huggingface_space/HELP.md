## ODRV2 Web Demo (Help)

### What this does
This web app screens a **retinal fundus image** for 7 conditions using a **3-fold ConvNeXt-Base ensemble** (folds 0/1/4).

### How to use
1. Upload a fundus image (`.jpg`, `.jpeg`, `.png`).
2. The first run may take longer while the models download and load.
3. Review:
   - **Probabilities** per condition
   - A short **Detected / Not detected** summary based on fixed thresholds
   - **QC / Attention Heatmap** (where the model focused)

### Interpreting results
- Probabilities are **not** a diagnosis.
- A “detected” label means the probability exceeded a preset threshold; it does not imply disease severity.
- Heatmaps are **attention visualizations** (GradCAM-style). They indicate regions influencing the output and can be wrong or misleading.

### Limitations
- Performance depends on image quality (blur, glare, poor illumination, small field of view).
- The model can be biased by dataset composition; it may not generalize to all devices/populations.

### Privacy / data handling
- The app performs inference on the server hosting this Space.
- Images are intended to be processed in-memory for inference. However, hosting platforms may keep logs/telemetry—avoid uploading sensitive patient data unless you control the deployment.

### Disclaimer
Research/screening tool only. Results must be verified by a qualified clinician.
