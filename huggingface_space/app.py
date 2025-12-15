"""
ODRV2 - Ocular Disease Recognition
Hugging Face Spaces Gradio App

Multi-label fundus image classification for 7 ocular conditions.
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download


# Disease information
DISEASE_INFO = {
    'Diabetic_Retinopathy': {
        'name': 'Diabetic Retinopathy',
        'description': 'Damage to blood vessels in the retina caused by diabetes.',
        'recommendation': 'Refer to ophthalmologist for grading and management.',
        'color': '#E53935'
    },
    'Glaucoma': {
        'name': 'Glaucoma',
        'description': 'Optic nerve damage often associated with elevated intraocular pressure.',
        'recommendation': 'Refer for IOP measurement and visual field testing.',
        'color': '#8E24AA'
    },
    'Cataract': {
        'name': 'Cataract',
        'description': 'Clouding of the natural lens inside the eye.',
        'recommendation': 'Assess visual acuity and refer for surgical evaluation if symptomatic.',
        'color': '#757575'
    },
    'AMD': {
        'name': 'Age-related Macular Degeneration',
        'description': 'Deterioration of the central retina affecting central vision.',
        'recommendation': 'Refer for OCT imaging and anti-VEGF evaluation if wet AMD suspected.',
        'color': '#FB8C00'
    },
    'Hypertension': {
        'name': 'Hypertensive Retinopathy',
        'description': 'Retinal vascular changes caused by high blood pressure.',
        'recommendation': 'Check blood pressure and refer for cardiovascular assessment.',
        'color': '#D32F2F'
    },
    'Myopia': {
        'name': 'Pathological Myopia',
        'description': 'Severe nearsightedness with structural eye changes.',
        'recommendation': 'Monitor for myopic macular degeneration and retinal detachment risk.',
        'color': '#1E88E5'
    },
    'Other': {
        'name': 'Other Pathology',
        'description': 'Other ocular pathologies not in the main categories.',
        'recommendation': 'Further clinical evaluation recommended.',
        'color': '#607D8B'
    }
}

DISEASE_NAMES = ['Diabetic_Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

# Optimized thresholds from training
THRESHOLDS = {
    'Diabetic_Retinopathy': 0.48,
    'Glaucoma': 0.62,
    'Cataract': 0.55,
    'AMD': 0.50,
    'Hypertension': 0.43,
    'Myopia': 0.62,
    'Other': 0.43
}


class FundusBackbone(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=False, feature_dim=1024):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.projection = nn.Linear(self.backbone.num_features, feature_dim)
    
    def forward(self, image):
        features = self.backbone(image)
        return self.projection(features)


class MultiLabelClassifier(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=7):
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features):
        return self.head(self.dropout(features))


class FundusModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FundusBackbone(pretrained=False)
        self.classifier = MultiLabelClassifier()
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        target = self.backbone.backbone.stages[-1]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def load_models():
    """Load the 3-fold ensemble from Hugging Face Hub."""
    import traceback
    models = []
    folds = [0, 1, 4]  # Full 3-fold ensemble
    
    for fold in folds:
        try:
            print(f"Downloading fold {fold} model...", flush=True)
            model_path = hf_hub_download(
                repo_id="fdbprojects/odrv2-models",
                filename=f"fold_{fold}/best_model.pth"
            )
            print(f"Loading fold {fold}...", flush=True)
            
            model = FundusModel()
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            models.append(model)
            print(f"✓ Loaded fold {fold}", flush=True)
            
        except Exception as e:
            print(f"Error loading fold {fold}: {e}", flush=True)
            traceback.print_exc()
    
    print(f"Total models loaded: {len(models)}", flush=True)
    return models


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def generate_heatmap(model, img_tensor, class_idx, original_size):
    """Generate GradCAM heatmap for a specific class."""
    import torch.nn.functional as F
    
    model.eval()
    img_tensor = img_tensor.clone().requires_grad_(True)
    logits = model(img_tensor)
    
    model.zero_grad()
    logits[0, class_idx].backward(retain_graph=True)
    
    gradients = model.gradients
    activations = model.activations
    
    if gradients is None or activations is None:
        return None
    
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()
    
    if cam.max() > 0:
        cam = cam / cam.max()
    
    # Resize and apply circular mask
    width, height = original_size
    cam_resized = cv2.resize(cam, original_size)
    
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 2 - 10
    y_grid, x_grid = np.ogrid[:height, :width]
    circular_mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2
    circular_mask = circular_mask.astype(np.float32)
    circular_mask = cv2.GaussianBlur(circular_mask, (21, 21), 0)
    
    cam_masked = cam_resized * circular_mask
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_masked), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    mask_3ch = np.stack([circular_mask] * 3, axis=-1)
    heatmap = (heatmap * mask_3ch).astype(np.uint8)
    
    return heatmap


def predict(image):
    """Run prediction on uploaded image."""
    if image is None:
        return None, "Please upload an image", None
    
    if not models:
        return None, "⚠️ Models not loaded. Please try again.", None
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')
    
    original_size = image.size
    img_tensor = transform(image).unsqueeze(0)
    
    # Ensemble prediction
    all_probs = []
    with torch.no_grad():
        for model in models:
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.numpy())
    
    avg_probs = np.mean(all_probs, axis=0)[0]
    
    # Build results
    results = []
    detected = []
    confidences = {}
    
    for i, name in enumerate(DISEASE_NAMES):
        prob = float(avg_probs[i])
        threshold = THRESHOLDS[name]
        is_positive = prob >= threshold
        info = DISEASE_INFO[name]
        
        confidences[info['name']] = prob
        
        status = "⚠️ DETECTED" if is_positive else "✓ Not detected"
        results.append(f"**{info['name']}**: {prob:.1%} ({status})")
        
        if is_positive:
            detected.append({
                'name': info['name'],
                'prob': prob,
                'description': info['description'],
                'recommendation': info['recommendation'],
                'class_idx': i
            })
    
    # Generate summary
    if detected:
        summary = f"### ⚠️ {len(detected)} Condition(s) Detected\n\n"
        for d in detected:
            summary += f"**{d['name']}** ({d['prob']:.1%})\n"
            summary += f"- {d['description']}\n"
            summary += f"- 💡 {d['recommendation']}\n\n"
    else:
        summary = "### ✅ No Pathology Detected\n\nAll conditions below threshold."
    
    summary += "\n---\n### All Results:\n" + "\n".join(results)
    summary += "\n\n---\n*⚠️ Research/screening tool only. Results must be verified by a qualified ophthalmologist.*"
    
    # Generate heatmap for highest probability detected condition
    heatmap_overlay = None
    if detected and models:
        top_condition = max(detected, key=lambda x: x['prob'])
        heatmap = generate_heatmap(models[0], img_tensor, top_condition['class_idx'], original_size)
        if heatmap is not None:
            original_np = np.array(image)
            heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
            heatmap_overlay = cv2.addWeighted(original_np, 0.6, heatmap_resized, 0.4, 0)
    
    return confidences, summary, heatmap_overlay


# Load models at startup
print("Loading ODRV2 models...")
models = load_models()
print(f"Loaded {len(models)} models")


# Create Gradio interface
with gr.Blocks(title="ODRV2 - Ocular Disease Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 ODRV2 - Ocular Disease Recognition
    
    Upload a fundus image to screen for **7 ocular conditions**:
    - Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertensive Retinopathy, Pathological Myopia, Other
    
    **Model**: 3-Fold ConvNeXt-Base Ensemble (88.6M parameters × 3) | **Macro F1**: 0.82
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📷 Upload Fundus Image", type="pil")
            analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
            
            gr.Examples(
                examples=[
                    ["examples/normal.jpg"],
                    ["examples/glaucoma.jpg"],
                    ["examples/dr.jpg"],
                ],
                inputs=input_image,
                label="Example Images"
            ) if Path("examples").exists() else None
        
        with gr.Column(scale=1):
            output_labels = gr.Label(label="Disease Probabilities", num_top_classes=7)
            output_text = gr.Markdown(label="Analysis Results")
    
    with gr.Row():
        heatmap_output = gr.Image(label="🔥 Attention Heatmap (Top Detected Condition)")
    
    analyze_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_labels, output_text, heatmap_output]
    )
    
    gr.Markdown("""
    ---
    ### About
    - **Paper**: Multi-label classification of ocular diseases using deep learning
    - **Training Data**: ODIR, APTOS, RFMiD, and other public fundus datasets
    - **Architecture**: ConvNeXt-Base with multi-label classification head
    
    ⚠️ **Disclaimer**: This is a research/screening tool. Results must be verified by a qualified ophthalmologist.
    """)


if __name__ == "__main__":
    demo.launch()
