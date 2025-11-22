"""Grad-CAM implementation for explainability."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any


class GradCAM:
    """Generate Grad-CAM heatmaps for model explainability."""
    
    def __init__(self, model: Any, target_layer: str = "backbone.model.stages.3.blocks.26"):
        """
        Args:
            model: The trained model
            target_layer: Name of the layer to visualize (last conv layer by default)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Navigate to the target layer
        target_module = self.model
        for attr in self.target_layer.split('.'):
            target_module = getattr(target_module, attr)
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
    
    def generate(self, batch: dict[str, torch.Tensor], class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for a specific class.
        
        Args:
            batch: Input batch with 'image', 'age', 'sex'
            class_idx: Index of the target class (0-6 for D, G, C, A, H, M, O)
            
        Returns:
            Heatmap as numpy array [H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(batch)
        
        # Backward pass for target class
        self.model.zero_grad()
        class_logit = logits[0, class_idx]
        class_logit.backward()
        
        # Generate CAM
        gradients = self.gradients  # [batch, channels, H, W]
        activations = self.activations  # [batch, channels, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [batch, 1, H, W]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_multi_class(self, batch: dict[str, torch.Tensor], class_indices: list[int]) -> dict[int, np.ndarray]:
        """Generate Grad-CAM for multiple classes.
        
        Args:
            batch: Input batch
            class_indices: List of class indices to visualize
            
        Returns:
            Dictionary mapping class index to heatmap
        """
        heatmaps = {}
        for class_idx in class_indices:
            # Need to run forward pass again for each class
            batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            heatmap = self.generate(batch_copy, class_idx)
            heatmaps[class_idx] = heatmap
        return heatmaps


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] in range [0, 255]
        heatmap: Grad-CAM heatmap [H, W] in range [0, 1]
        alpha: Blending factor
        
    Returns:
        Overlaid image [H, W, 3] in range [0, 255]
    """
    import cv2
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB using a colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Blend
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid
