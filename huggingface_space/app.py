"""ODRV2 - Ocular Disease Recognition (Streamlit).

Compatibility note: PyTorch 2.6 changed the default of `torch.load(..., weights_only=...)`
to `True`, which can fail to load older checkpoints unless some classes (e.g. PosixPath)
are allowlisted.
"""
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pathlib
import sys
import types
import hashlib
from PIL import Image
from torchvision import transforms
import timm
from huggingface_hub import hf_hub_download

import torch.nn.functional as F

st.set_page_config(page_title="ODRV2 - Ocular Disease Recognition", page_icon="👁️", layout="wide")

st.markdown(
    """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Layout polish */
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; }

/* “Card” containers */
.odrv2-card {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.65);
}
</style>
""",
    unsafe_allow_html=True,
)


DISEASE_INFO = {
    "Diabetic Retinopathy": {
        "short": "DR",
        "what": "Retinal microvascular damage associated with diabetes.",
        "typical_signs": "Microaneurysms, hemorrhages, hard exudates, neovascularization.",
        "clinical_note": "If detected/suspected, consider grading severity and referral per local guidelines.",
    },
    "Glaucoma": {
        "short": "G",
        "what": "Optic neuropathy often associated with elevated IOP.",
        "typical_signs": "Increased cup-to-disc ratio, RNFL defects, disc hemorrhages.",
        "clinical_note": "If detected/suspected, consider IOP measurement, OCT RNFL, and visual field testing.",
    },
    "Cataract": {
        "short": "C",
        "what": "Lens opacity degrading retinal view and visual acuity.",
        "typical_signs": "Reduced fundus clarity, diffuse haze; may mask retinal pathology.",
        "clinical_note": "Interpret negative findings cautiously if image quality is poor.",
    },
    "AMD": {
        "short": "A",
        "what": "Age-related macular degeneration affecting central retina.",
        "typical_signs": "Drusen, pigmentary changes, geographic atrophy; CNV in wet AMD.",
        "clinical_note": "If detected/suspected, consider OCT and prompt evaluation if wet AMD is possible.",
    },
    "Hypertensive Retinopathy": {
        "short": "H",
        "what": "Retinal vascular changes associated with systemic hypertension.",
        "typical_signs": "AV nicking, arteriolar narrowing, hemorrhages, cotton wool spots.",
        "clinical_note": "Consider blood pressure assessment and systemic risk evaluation.",
    },
    "Myopia": {
        "short": "M",
        "what": "High/pathological myopia with structural retinal/optic changes.",
        "typical_signs": "Peripapillary atrophy, tilted disc, posterior staphyloma, lacquer cracks.",
        "clinical_note": "Monitor for myopic maculopathy and retinal detachment risk as appropriate.",
    },
    "Other": {
        "short": "O",
        "what": "Other ocular pathology not in the main categories.",
        "typical_signs": "Varies; may include scars, detachments, drusen, ERM, etc.",
        "clinical_note": "Use clinical judgement and consider targeted workup based on appearance.",
    },
}

class FundusBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=False, num_classes=0, global_pool='avg')
        self.projection = nn.Linear(self.backbone.num_features, 1024)
    def forward(self, x):
        return self.projection(self.backbone(x))

class FundusModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FundusBackbone()

        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(0.3)
                # Keep attribute name `head` to match checkpoint keys: classifier.head.*
                self.head = nn.Linear(1024, 7)

            def forward(self, x):
                return self.head(self.dropout(x))

        self.classifier = _Classifier()

        # For GradCAM-style attention maps
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target = self.backbone.backbone.stages[-1]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def forward(self, x):
        return self.classifier(self.backbone(x))

THRESHOLDS = [0.48, 0.62, 0.55, 0.50, 0.43, 0.62, 0.43]
DISEASES = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertensive Retinopathy', 'Myopia', 'Other']
THRESHOLDS_BY_DISEASE = dict(zip(DISEASES, THRESHOLDS))

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models():
    # Some training environments pickle PosixPath as `pathlib._local.PosixPath`.
    # Newer Python stdlib doesn't ship that module path, so we provide a small shim.
    if "pathlib._local" not in sys.modules:
        shim = types.ModuleType("pathlib._local")
        shim.PosixPath = pathlib.PosixPath
        shim.Path = pathlib.Path
        sys.modules["pathlib._local"] = shim

    models = []
    for fold in [0, 1, 4]:
        try:
            path = hf_hub_download("fdbprojects/odrv2-models", f"fold_{fold}/best_model.pth")
            model = FundusModel()

            ckpt = None
            # Prefer safe/weights-only load when supported (PyTorch 2.6+), with an allowlist
            # for Path objects that may appear in our training checkpoints.
            try:
                if hasattr(torch, "serialization") and hasattr(torch.serialization, "safe_globals"):
                    with torch.serialization.safe_globals([pathlib.PosixPath, pathlib.Path]):
                        ckpt = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    ckpt = torch.load(path, map_location="cpu")
            except TypeError:
                # Older torch without `weights_only` kwarg.
                ckpt = torch.load(path, map_location="cpu")
            except Exception:
                # Fallback: full load. This is safe here because these are our own checkpoints.
                ckpt = torch.load(path, map_location="cpu", weights_only=False)

            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        except Exception as e:
            st.error(f"Error loading fold {fold}: {e}")
    return models

def predict(image, models):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    tensor = transform(image).unsqueeze(0)
    
    probs_list = []
    with torch.no_grad():
        for m in models:
            probs_list.append(torch.sigmoid(m(tensor)).numpy())
    
    return np.mean(probs_list, axis=0)[0]


def compute_qc_metrics(image: Image.Image) -> tuple[dict, list[str]]:
    """Compute simple QC metrics (not a diagnosis).

    Metrics are heuristic and intended for operator feedback only.
    """
    import cv2

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Focus proxy: variance of Laplacian
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Rough field-of-view proxy: how much of the image is very dark
    dark_frac = float((gray < 10).mean())

    metrics = {
        "Focus (Laplacian var)": lap_var,
        "Brightness (mean)": brightness,
        "Contrast (std)": contrast,
        "Very dark pixels (%)": 100.0 * dark_frac,
    }

    warnings: list[str] = []
    if lap_var < 40:
        warnings.append("Low focus/sharpness detected (image may be blurry).")
    if brightness < 55:
        warnings.append("Low brightness detected (underexposed / dark image).")
    if brightness > 205:
        warnings.append("High brightness detected (overexposed / glare risk).")
    if dark_frac > 0.45:
        warnings.append("Large dark border detected (limited field of view / poor centering).")

    return metrics, warnings


def _make_circular_mask(width: int, height: int) -> np.ndarray:
    import cv2
    # Match desktop app behavior: circle mask, smoothed edge
    center_x, center_y = width // 2, height // 2
    radius = max(1, min(width, height) // 2 - 10)
    y_grid, x_grid = np.ogrid[:height, :width]
    mask = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) <= radius ** 2
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def generate_attention_heatmaps(image: Image.Image, model: FundusModel) -> dict:
    import cv2
    # Use GradCAM on the last ConvNeXt stage (same target as desktop app)
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    circular_mask = _make_circular_mask(width, height)

    img_tensor = transform(image).unsqueeze(0)
    heatmaps = []

    model.eval()
    for class_idx in range(len(DISEASES)):
        img_tensor_copy = img_tensor.clone().requires_grad_(True)
        logits = model(img_tensor_copy)

        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward(retain_graph=True)

        gradients = model.gradients
        activations = model.activations
        if gradients is None or activations is None:
            heatmaps.append(None)
            continue

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (width, height))
        cam_masked = cam_resized * circular_mask

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_masked), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        mask_3ch = np.stack([circular_mask] * 3, axis=-1)
        heatmap_colored = (heatmap_colored * mask_3ch).astype(np.uint8)
        heatmaps.append(heatmap_colored)

    return {DISEASES[i]: heatmaps[i] for i in range(len(DISEASES)) if heatmaps[i] is not None}


def blend_overlay(image: Image.Image, heatmap_rgb: np.ndarray, alpha: float = 0.4) -> Image.Image:
    import cv2
    original = np.array(image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap_rgb, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_resized, alpha, 0)
    return Image.fromarray(overlay)


def render_help_section():
    with st.expander("Help / README", expanded=False):
        help_path = pathlib.Path(__file__).with_name("HELP.md")
        if help_path.exists():
            st.markdown(help_path.read_text(encoding="utf-8"))
        else:
            st.markdown(
                """
### How to use
- Upload a fundus image (`.jpg`, `.jpeg`, `.png`).
- Wait for the ensemble models to load on first use.
- Review the per-condition probabilities and the summary.

### Notes
- This is a research/screening demo. It is not a medical device.
- Images are processed in-memory for inference; no guarantees are made about logging/storage by the hosting platform.
"""
            )

st.markdown("# ODRV2 — Ocular Disease Recognition")
st.caption("3-fold ConvNeXt-Base ensemble (folds 0/1/4). Research/screening demo — not a medical device.")

with st.sidebar:
    st.markdown("## Workflow")
    uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"], label_visibility="visible")
    st.markdown("---")
    st.markdown("## Model")
    with st.spinner("Checking model availability..."):
        models = load_models()
    if models:
        st.success(f"Loaded {len(models)} model(s)")
        st.caption("Models load lazily and are cached per session.")
    else:
        st.error("Models not loaded")
        st.caption("If this persists, the Space may still be restarting.")
    st.markdown("---")
    st.markdown("## Clinical disclaimer")
    st.caption(
        "This tool is for research/screening only. Outputs must be verified by a qualified clinician. "
        "Do not upload patient-identifiable data unless you control the deployment and data handling."
    )


render_help_section()

if not uploaded_file:
    st.markdown(
        """
<div class="odrv2-card">
<b>Get started</b><br/>
Upload a fundus image from the sidebar. The app will show:<br/>
• A QC panel (focus/brightness/contrast + histogram)<br/>
• Per-condition probabilities + threshold-based flags<br/>
• Attention heatmaps (GradCAM-style) to visualize model focus
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
img_hash = hashlib.md5(file_bytes).hexdigest()
image = Image.open(uploaded_file).convert("RGB")

# Inference
with st.spinner("Running inference..."):
    probs = predict(image, models)

prob_by_disease = {DISEASES[i]: float(probs[i]) for i in range(len(DISEASES))}
detected = [d for d in DISEASES if prob_by_disease[d] >= THRESHOLDS_BY_DISEASE[d]]

tabs = st.tabs(["Overview", "Results", "QC", "Heatmap", "Clinical notes"])

with tabs[0]:
    left, right = st.columns([1.0, 1.0])
    with left:
        st.markdown("### Image")
        st.image(image, use_container_width=True)
    with right:
        st.markdown("### Summary")
        if detected:
            st.warning("Detected above threshold: " + ", ".join(detected))
        else:
            st.success("No conditions detected above threshold")

        top = sorted(DISEASES, key=lambda d: prob_by_disease[d], reverse=True)[:3]
        st.markdown("**Top scores**")
        for d in top:
            st.write(f"- {d}: {prob_by_disease[d]:.1%} (threshold {THRESHOLDS_BY_DISEASE[d]:.0%})")

with tabs[1]:
    st.markdown("### Per-condition probabilities")
    st.caption("Threshold flags are fixed and were optimized on the validation set for the 3-fold ensemble.")
    for d in DISEASES:
        p = prob_by_disease[d]
        t = THRESHOLDS_BY_DISEASE[d]
        flag = "⚠️" if p >= t else "✓"
        st.markdown(f"**{flag} {d}** — {p:.1%} (threshold {t:.0%})")
        st.progress(min(max(p, 0.0), 1.0))

with tabs[2]:
    st.markdown("### Image quality control (QC)")
    st.caption("QC is heuristic (operator feedback only). Poor QC can increase false negatives/positives.")
    metrics, warnings = compute_qc_metrics(image)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Focus", f"{metrics['Focus (Laplacian var)']:.1f}")
    m2.metric("Brightness", f"{metrics['Brightness (mean)']:.0f}")
    m3.metric("Contrast", f"{metrics['Contrast (std)']:.0f}")
    m4.metric("Dark border", f"{metrics['Very dark pixels (%)']:.0f}%")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("No major QC warnings detected.")

    # Intensity histogram
    gray = np.array(image.convert("L"))
    hist, bin_edges = np.histogram(gray.flatten(), bins=32, range=(0, 255))
    st.markdown("#### Intensity histogram")
    st.bar_chart(hist)

with tabs[3]:
    st.markdown("### Attention heatmap (GradCAM-style)")
    st.caption(
        "Heatmaps visualize regions influencing the model output. They are not lesion segmentations "
        "and can be misleading; use only as an interpretability aid."
    )

    if "qc" not in st.session_state or st.session_state.get("qc_img_hash") != img_hash:
        st.session_state.qc_img_hash = img_hash
        st.session_state.qc_heatmaps = None

    if st.session_state.qc_heatmaps is None:
        with st.spinner("Generating attention heatmaps..."):
            st.session_state.qc_heatmaps = generate_attention_heatmaps(image, models[0])

    qc_heatmaps = st.session_state.qc_heatmaps or {}
    sorted_diseases = sorted(DISEASES, key=lambda d: prob_by_disease[d], reverse=True)
    selected = st.selectbox("Select condition", options=sorted_diseases, index=0)

    hm = qc_heatmaps.get(selected)
    if hm is None:
        st.info("Heatmap unavailable for this class.")
    else:
        overlay = blend_overlay(image, hm, alpha=0.4)
        st.image(overlay, caption=f"Attention overlay: {selected}", use_container_width=True)

with tabs[4]:
    st.markdown("### Clinician-oriented notes")
    st.caption(
        "These notes are generic and for workflow support only. Final interpretation and management "
        "decisions must follow local clinical guidelines."
    )
    for d in DISEASES:
        info = DISEASE_INFO.get(d, {})
        p = prob_by_disease[d]
        t = THRESHOLDS_BY_DISEASE[d]
        is_pos = p >= t
        heading = f"{info.get('short', '')} — {d}"
        with st.expander(heading, expanded=is_pos):
            st.markdown(f"**Model score:** {p:.1%}  |  **Threshold:** {t:.0%}  |  **Flag:** {'Detected' if is_pos else 'Not detected'}")
            st.markdown(f"**What it is:** {info.get('what', '—')}")
            st.markdown(f"**Typical fundus signs:** {info.get('typical_signs', '—')}")
            st.markdown(f"**Clinical note:** {info.get('clinical_note', '—')}")

st.markdown("---")
st.caption("ODRV2 demo • Ensemble inference on CPU • Results are probabilistic and must be clinically validated")
