from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
import cv2  # type: ignore[import]
import base64
from io import BytesIO
from PIL import Image  # type: ignore[import]
from fastapi import FastAPI, UploadFile, Form  # type: ignore[import]
from fastapi.staticfiles import StaticFiles  # type: ignore[import]
from fastapi.responses import RedirectResponse  # type: ignore[import]

from src.preprocessing.pipeline import preprocess_batch
from src.inference.gradcam import GradCAM, overlay_heatmap
from src.training.loop import FundusLightningModule, TrainingConfig
from src.training.cli import _load_with_defaults

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a CHW tensor."""
    mean = torch.tensor(_IMG_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(_IMG_STD, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def load_ensemble(model_dir: Path, config_path: Path, device: torch.device) -> tuple[list[FundusLightningModule], TrainingConfig]:
    """Load all trained fold models for ensembling."""
    cfg = _load_with_defaults(config_path)
    training_cfg = TrainingConfig(
        num_classes=cfg.data.num_classes,
        feature_dim=cfg.model.feature_dim,
        learning_rate=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay,
        epochs=cfg.trainer.max_epochs,
        batch_size=cfg.datamodule.batch_size,
        model_dir=Path(cfg.paths.models),
        pos_weight=cfg.datamodule.get("class_weights", None),
        precision=str(cfg.trainer.get("precision", "32")),
        accelerator=cfg.trainer.get("accelerator", None),
        devices=cfg.trainer.get("devices", None),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 20),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
        image_size=tuple(cfg.datamodule.image_size),
    )
    
    models = []
    # Look for model_fold_0.ckpt, model_fold_1.ckpt, etc.
    # Also include model.ckpt if it exists (as a fallback or part of ensemble)
    checkpoint_files = sorted(model_dir.glob("model_fold_*.ckpt"))
    if not checkpoint_files and (model_dir / "model.ckpt").exists():
        checkpoint_files = [model_dir / "model.ckpt"]
        
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    print(f"Loading {len(checkpoint_files)} models for ensemble...")
    for ckpt in checkpoint_files:
        model = FundusLightningModule.load_from_checkpoint(str(ckpt), config=training_cfg)
        model.to(device)
        model.eval()
        models.append(model)
        
    return models, training_cfg


def _write_temp_image(data: bytes) -> Path:
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = preprocess_batch([path], size)[0]
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor


def _apply_tta_transforms(image: torch.Tensor) -> list[torch.Tensor]:
    """Apply test-time augmentation transforms."""
    transforms = [
        image,  # Original
        torch.flip(image, dims=[2]),  # Horizontal flip
        torch.flip(image, dims=[1]),  # Vertical flip
        torch.rot90(image, k=1, dims=[1, 2]),  # 90° rotation
        torch.rot90(image, k=3, dims=[1, 2]),  # 270° rotation
    ]
    return transforms


def _reverse_tta_transform(probs: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """Reverse TTA transform (for spatial predictions, not needed for classification)."""
    return probs


def _calculate_uncertainty(all_probs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate uncertainty metrics from ensemble predictions.
    
    Returns:
        ensemble_std: Standard deviation across ensemble (disagreement)
        entropy: Predictive entropy for each class
    """
    probs_tensor = torch.stack(all_probs)  # [n_models, n_classes]
    
    # Ensemble disagreement (higher = more uncertain)
    ensemble_std = probs_tensor.std(dim=0)
    
    # Predictive entropy for each class (higher = more uncertain)
    mean_probs = probs_tensor.mean(dim=0)
    epsilon = 1e-10
    entropy = -(mean_probs * torch.log(mean_probs + epsilon) + 
                (1 - mean_probs) * torch.log(1 - mean_probs + epsilon))
    
    return ensemble_std, entropy


def create_app(models: list[FundusLightningModule], config: TrainingConfig, device: torch.device) -> FastAPI:
    """Create FastAPI inference service with ensemble support."""
    app = FastAPI()
    size = config.image_size or (448, 448)
    
    # Optimized thresholds from validation set
    # D, G, C, A, H, M, O
    THRESHOLDS = [0.45, 0.45, 0.40, 0.75, 0.15, 0.30, 0.25]
    CLASS_NAMES = ['Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    
    # Initialize Grad-CAM for the first model (representative)
    # Target the last stage of the ConvNeXt backbone
    gradcam = GradCAM(models[0], target_layer="backbone.backbone.stages.3")

    app.mount("/static", StaticFiles(directory="src/webapp/static"), name="static")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/static/index.html")

    @app.post("/predict")
    async def predict(
        file: UploadFile, 
        age: float = Form(...), 
        sex: int = Form(...),
        use_tta: bool = Form(True)
    ) -> dict[str, Any]:
        image_bytes = await file.read()
        image_path = _write_temp_image(image_bytes)

        try:
            image_tensor = _load_image(image_path, size).to(device)
        finally:
            image_path.unlink(missing_ok=True)

        image_tensor = _normalize(image_tensor)
        
        # Apply TTA if enabled
        if use_tta:
            tta_images = _apply_tta_transforms(image_tensor)
        else:
            tta_images = [image_tensor]
        
        metadata = {
            "age": torch.tensor([age], dtype=torch.float32, device=device),
            "sex": torch.tensor([sex], dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            # Collect predictions from all models and TTA variants
            all_probs = []
            
            for tta_img in tta_images:
                batch = {
                    "image": tta_img.unsqueeze(0),
                    **metadata
                }
                
                for model in models:
                    logits = model(batch)
                    probs = torch.sigmoid(logits)[0]
                    all_probs.append(probs)
            
            # Calculate ensemble statistics
            ensemble_std, entropy = _calculate_uncertainty(all_probs)
            
            # Average predictions
            avg_probs = torch.stack(all_probs).mean(dim=0)
            
            # Calculate overall uncertainty score (0-1, higher = more uncertain)
            uncertainty_score = (ensemble_std.mean() * 0.5 + entropy.mean() * 0.5).item()
            
            # Flag high-uncertainty predictions
            high_uncertainty = uncertainty_score > 0.15
            
        # Generate Grad-CAM heatmaps for detected diseases
        heatmaps = {}
        detected_classes = [i for i, (p, t) in enumerate(zip(avg_probs.cpu().tolist(), THRESHOLDS)) if p >= t]
        
        if detected_classes:
            # Use original (non-TTA) image for visualization
            batch = {
                "image": image_tensor.unsqueeze(0),
                **metadata
            }
            
            try:
                # Generate heatmaps for detected diseases
                class_heatmaps = gradcam.generate_multi_class(batch, detected_classes)
                
                # Convert original image for overlay
                img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
                img_np = ((img_np * 0.229 + 0.485) * 255).clip(0, 255).astype(np.uint8)
                
                for class_idx in detected_classes:
                    heatmap = class_heatmaps[class_idx]
                    overlay = overlay_heatmap(img_np, heatmap, alpha=0.4)
                    
                    # Encode to base64 PNG
                    pil_img = Image.fromarray(overlay)
                    buffer = BytesIO()
                    pil_img.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    heatmaps[CLASS_NAMES[class_idx]] = f"data:image/png;base64,{img_base64}"
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
                heatmaps = {}
        
        predictions = {
            "probabilities": avg_probs.cpu().tolist(),
            "labels": [p >= t for p, t in zip(avg_probs.cpu().tolist(), THRESHOLDS)],
            "thresholds": THRESHOLDS,
            "uncertainty": {
                "overall_score": round(uncertainty_score, 3),
                "per_class_std": ensemble_std.cpu().tolist(),
                "per_class_entropy": entropy.cpu().tolist(),
                "high_uncertainty": high_uncertainty,
                "message": "⚠️ High uncertainty - recommend expert review" if high_uncertainty else "✓ Confident prediction"
            },
            "explainability": {
                "heatmaps": heatmaps,
                "message": f"Generated {len(heatmaps)} heatmap(s) for detected diseases"
            },
            "tta_enabled": use_tta,
            "num_predictions_averaged": len(all_probs)
        }
        return predictions

    return app
