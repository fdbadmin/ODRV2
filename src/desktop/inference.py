"""
Inference module for ODRV2 Desktop Application.
Loads the pruned 3-fold ensemble and runs predictions on fundus images.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Container for prediction results."""
    disease_names: List[str]
    probabilities: Dict[str, float]
    predictions: Dict[str, bool]
    thresholds: Dict[str, float]
    heatmaps: Optional[Dict[str, np.ndarray]] = None  # Per-disease heatmaps
    original_size: Optional[Tuple[int, int]] = None  # Original image size (width, height)
    

# Disease information for clinical explanations
DISEASE_INFO = {
    'Diabetic_Retinopathy': {
        'name': 'Diabetic Retinopathy',
        'short': 'DR',
        'description': 'Damage to blood vessels in the retina caused by diabetes.',
        'clinical_signs': 'Microaneurysms, hemorrhages, exudates, neovascularization.',
        'recommendation': 'Refer to ophthalmologist for grading and management.',
        'color': '#E53935'  # Red
    },
    'Glaucoma': {
        'name': 'Glaucoma',
        'short': 'G',
        'description': 'Optic nerve damage often associated with elevated intraocular pressure.',
        'clinical_signs': 'Increased cup-to-disc ratio, disc hemorrhages, RNFL defects.',
        'recommendation': 'Refer for IOP measurement and visual field testing.',
        'color': '#8E24AA'  # Purple
    },
    'Cataract': {
        'name': 'Cataract',
        'short': 'C',
        'description': 'Clouding of the natural lens inside the eye.',
        'clinical_signs': 'Reduced fundus view clarity, lens opacity visible.',
        'recommendation': 'Assess visual acuity and refer for surgical evaluation if symptomatic.',
        'color': '#757575'  # Gray
    },
    'AMD': {
        'name': 'Age-related Macular Degeneration',
        'short': 'A',
        'description': 'Deterioration of the central retina (macula) affecting central vision.',
        'clinical_signs': 'Drusen, pigmentary changes, geographic atrophy, CNV.',
        'recommendation': 'Refer for OCT imaging and anti-VEGF evaluation if wet AMD suspected.',
        'color': '#FB8C00'  # Orange
    },
    'Hypertension': {
        'name': 'Hypertensive Retinopathy',
        'short': 'H',
        'description': 'Retinal vascular changes caused by high blood pressure.',
        'clinical_signs': 'Arteriovenous nicking, copper/silver wiring, hemorrhages.',
        'recommendation': 'Check blood pressure and refer for cardiovascular assessment.',
        'color': '#D32F2F'  # Dark Red
    },
    'Myopia': {
        'name': 'Pathological Myopia',
        'short': 'M',
        'description': 'Severe nearsightedness with structural eye changes.',
        'clinical_signs': 'Posterior staphyloma, tilted disc, peripapillary atrophy, lacquer cracks.',
        'recommendation': 'Monitor for myopic macular degeneration and retinal detachment risk.',
        'color': '#1E88E5'  # Blue
    },
    'Other': {
        'name': 'Other Pathology',
        'short': 'O',
        'description': 'Other ocular pathologies not in the main categories.',
        'clinical_signs': 'Various findings such as drusen, epiretinal membrane, retinal detachment, or other abnormalities.',
        'recommendation': 'Further clinical evaluation recommended based on specific findings.',
        'color': '#607D8B'  # Blue-gray
    }
}


class FundusBackbone(nn.Module):
    """ConvNeXt backbone for feature extraction."""
    
    def __init__(self, model_name: str = 'convnext_base', pretrained: bool = False, feature_dim: int = 1024):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.projection = nn.Linear(self.backbone.num_features, feature_dim)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.projection(features)


class MultiLabelClassifier(nn.Module):
    """Classification head for multi-label prediction."""
    
    def __init__(self, feature_dim: int = 1024, num_classes: int = 7):
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(features))


class FundusModel(nn.Module):
    """Complete model combining backbone and classifier."""
    
    def __init__(self):
        super().__init__()
        self.backbone = FundusBackbone(pretrained=False)
        self.classifier = MultiLabelClassifier()
        
        # For GradCAM - store activations and gradients
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for GradCAM on the last conv layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Target the last stage of ConvNeXt backbone
        target = self.backbone.backbone.stages[-1]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class FundusPredictor:
    """
    Fundus image predictor using pruned 3-fold ensemble.
    
    Loads models from folds 0, 1, 4 and averages predictions.
    """
    
    # Default thresholds from training
    DEFAULT_THRESHOLDS = {
        'Diabetic_Retinopathy': 0.48,
        'Glaucoma': 0.62,
        'Cataract': 0.55,
        'AMD': 0.50,
        'Hypertension': 0.43,
        'Myopia': 0.62,
        'Other': 0.43
    }
    
    DISEASE_NAMES = ['Diabetic_Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    FOLDS_TO_USE = [0, 1, 4]
    
    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Path to model directory. Defaults to models/unified_v3_retrain/
            device: Device to use ('cpu', 'mps', 'cuda'). Auto-detected if None.
        """
        if model_dir is None:
            # Find project root by going up from src/desktop/
            current = Path(__file__).resolve().parent  # src/desktop
            project_root = current.parent.parent  # Go up to project root
            model_dir = project_root / 'models' / 'unified_v3_retrain'
        
        self.model_dir = Path(model_dir)
        print(f"Model directory: {self.model_dir}")
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.models: List[FundusModel] = []
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._loaded = False
    
    def load_models(self) -> bool:
        """
        Load ensemble models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise.
        """
        self.models = []
        
        for fold in self.FOLDS_TO_USE:
            model_path = self.model_dir / f'fold_{fold}' / 'best_model.pth'
            
            if not model_path.exists():
                print(f"Warning: Model not found at {model_path}")
                continue
            
            try:
                model = FundusModel()
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"Loaded fold {fold} model")
                
            except Exception as e:
                print(f"Error loading fold {fold}: {e}")
                continue
        
        self._loaded = len(self.models) > 0
        
        # Use optimized thresholds from pruned ensemble evaluation
        # These were determined through threshold optimization on validation set
        # with the 3-fold pruned ensemble (folds 0, 1, 4)
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        
        return self._loaded
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded
    
    def predict(self, image: Image.Image, generate_heatmaps: bool = True) -> PredictionResult:
        """
        Run prediction on a fundus image.
        
        Args:
            image: PIL Image (RGB)
            generate_heatmaps: Whether to generate GradCAM heatmaps
            
        Returns:
            PredictionResult with probabilities and predictions
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size  # (width, height)
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Ensemble prediction
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(img_tensor)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
        
        # Average across folds
        avg_probs = np.mean(all_probs, axis=0)[0]
        
        # Build results
        probabilities = {}
        predictions = {}
        
        for i, name in enumerate(self.DISEASE_NAMES):
            prob = float(avg_probs[i])
            threshold = self.thresholds.get(name, 0.5)
            
            probabilities[name] = prob
            predictions[name] = prob >= threshold
        
        # Generate per-disease heatmaps for all diseases
        heatmaps = None
        if generate_heatmaps:
            heatmaps = self._generate_all_heatmaps(img_tensor, original_size)
        
        return PredictionResult(
            disease_names=self.DISEASE_NAMES,
            probabilities=probabilities,
            predictions=predictions,
            thresholds=self.thresholds,
            heatmaps=heatmaps,
            original_size=original_size
        )
    
    def predict_from_path(self, image_path: str) -> PredictionResult:
        """
        Run prediction on an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PredictionResult with probabilities and predictions
        """
        image = Image.open(image_path).convert('RGB')
        return self.predict(image)
    
    def _generate_all_heatmaps(self, img_tensor: torch.Tensor, 
                                original_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Generate GradCAM heatmaps for all disease classes.
        
        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W]
            original_size: Original image size (width, height)
            
        Returns:
            Dictionary mapping disease name to heatmap [H, W, 3] in uint8
        """
        import torch.nn.functional as F
        import cv2
        
        # Use first model for visualization
        model = self.models[0]
        model.eval()
        
        # Create circular mask to exclude black background
        width, height = original_size
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2 - 10  # Slightly smaller than image
        
        y_grid, x_grid = np.ogrid[:height, :width]
        circular_mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2
        circular_mask = circular_mask.astype(np.float32)
        
        # Smooth the mask edges
        circular_mask = cv2.GaussianBlur(circular_mask, (21, 21), 0)
        
        heatmaps = {}
        
        for class_idx, disease_name in enumerate(self.DISEASE_NAMES):
            # Forward pass with gradients enabled
            img_tensor_copy = img_tensor.clone().requires_grad_(True)
            logits = model(img_tensor_copy)
            
            # Backward for this class
            model.zero_grad()
            logits[0, class_idx].backward(retain_graph=True)
            
            # Get gradients and activations
            gradients = model.gradients
            activations = model.activations
            
            if gradients is None or activations is None:
                continue
            
            # Global average pooling of gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            
            # Weighted combination
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().detach().numpy()
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to original image size
            cam_resized = cv2.resize(cam, original_size)
            
            # Apply circular mask to exclude black background
            cam_masked = cam_resized * circular_mask
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_masked), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Make masked areas (outside fundus) dark/transparent
            # Set areas outside the mask to a neutral color
            mask_3ch = np.stack([circular_mask] * 3, axis=-1)
            dark_bg = np.zeros_like(heatmap_colored)
            heatmap_colored = (heatmap_colored * mask_3ch + dark_bg * (1 - mask_3ch)).astype(np.uint8)
            
            heatmaps[disease_name] = heatmap_colored
        
        return heatmaps
    
    @staticmethod
    def get_disease_info(disease_name: str) -> dict:
        """Get clinical information about a disease."""
        return DISEASE_INFO.get(disease_name, {})
