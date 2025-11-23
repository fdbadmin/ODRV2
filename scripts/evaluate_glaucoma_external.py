"""Evaluate model accuracy on external ACRIMA glaucoma dataset"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.service import load_ensemble, _normalize, _load_image


def main():
    # Setup paths
    root_dir = Path(".")
    models_dir = root_dir / "models"
    config_path = root_dir / "configs/training.yaml"
    
    # Initialize service
    print("Loading ensemble models...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        models, config = load_ensemble(models_dir, config_path, device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Glaucoma threshold (class index 1 = 'G')
    GLAUCOMA_THRESHOLD = 0.45
    GLAUCOMA_IDX = 1  # G is the second class in ['D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    # Find all images
    image_dir = Path("external_data/Database/Images")
    if not image_dir.exists():
        print(f"Error: {image_dir} not found!")
        print("Please run this script from the repository root with external data available.")
        return
    
    # Get all images
    all_images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")))
    print(f"Found {len(all_images)} images")
    
    # Classify images based on filename
    # Images with '_g_' in the filename are glaucoma, others are normal
    normal_images = [img for img in all_images if '_g_' not in img.stem]
    glaucoma_images = [img for img in all_images if '_g_' in img.stem]
    
    print(f"  - Normal: {len(normal_images)}")
    print(f"  - Glaucoma: {len(glaucoma_images)}")
    
    # Run predictions
    print("\nRunning predictions...")
    
    # Track results
    results = {
        'normal': {'correct': 0, 'total': 0, 'predictions': []},
        'glaucoma': {'correct': 0, 'total': 0, 'predictions': []}
    }
    
    # Process normal images
    print("\nProcessing normal images...")
    for img_path in tqdm(normal_images, desc="Normal"):
        try:
            # Load and preprocess image
            img_tensor = _load_image(img_path, config.image_size).to(device)
            img_tensor = _normalize(img_tensor)
            
            # Use default metadata (middle-aged, female)
            batch = {
                "image": img_tensor.unsqueeze(0),
                "age": torch.tensor([50.0], dtype=torch.float32, device=device),
                "sex": torch.tensor([1.0], dtype=torch.float32, device=device),
            }
            
            # Run inference
            with torch.no_grad():
                all_probs = []
                for model in models:
                    logits = model(batch)
                    probs = torch.sigmoid(logits)[0]
                    all_probs.append(probs)
                
                avg_probs = torch.stack(all_probs).mean(dim=0).cpu().numpy()
            
            # Extract glaucoma probability and prediction
            glaucoma_prob = float(avg_probs[GLAUCOMA_IDX])
            has_glaucoma = glaucoma_prob >= GLAUCOMA_THRESHOLD
            
            results['normal']['total'] += 1
            results['normal']['predictions'].append({
                'filename': img_path.name,
                'predicted_glaucoma': has_glaucoma,
                'glaucoma_prob': glaucoma_prob,
            })
            
            # Correct if glaucoma was NOT predicted (true negative)
            if not has_glaucoma:
                results['normal']['correct'] += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Process glaucoma images
    print("\nProcessing glaucoma images...")
    for img_path in tqdm(glaucoma_images, desc="Glaucoma"):
        try:
            # Load and preprocess image
            img_tensor = _load_image(img_path, config.image_size).to(device)
            img_tensor = _normalize(img_tensor)
            
            # Use default metadata (middle-aged, female)
            batch = {
                "image": img_tensor.unsqueeze(0),
                "age": torch.tensor([50.0], dtype=torch.float32, device=device),
                "sex": torch.tensor([1.0], dtype=torch.float32, device=device),
            }
            
            # Run inference
            with torch.no_grad():
                all_probs = []
                for model in models:
                    logits = model(batch)
                    probs = torch.sigmoid(logits)[0]
                    all_probs.append(probs)
                
                avg_probs = torch.stack(all_probs).mean(dim=0).cpu().numpy()
            
            # Extract glaucoma probability and prediction
            glaucoma_prob = float(avg_probs[GLAUCOMA_IDX])
            has_glaucoma = glaucoma_prob >= GLAUCOMA_THRESHOLD
            
            results['glaucoma']['total'] += 1
            results['glaucoma']['predictions'].append({
                'filename': img_path.name,
                'predicted_glaucoma': has_glaucoma,
                'glaucoma_prob': glaucoma_prob,
            })
            
            # Correct if glaucoma WAS predicted (true positive)
            if has_glaucoma:
                results['glaucoma']['correct'] += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS - ACRIMA Glaucoma Dataset")
    print("="*80)
    
    # Normal images (specificity / true negative rate)
    normal_accuracy = results['normal']['correct'] / results['normal']['total'] if results['normal']['total'] > 0 else 0
    print(f"\nNormal Images (should NOT predict glaucoma):")
    print(f"  Total: {results['normal']['total']}")
    print(f"  Correct (True Negatives): {results['normal']['correct']}")
    print(f"  Specificity: {normal_accuracy:.2%}")
    
    # Glaucoma images (sensitivity / true positive rate)
    glaucoma_accuracy = results['glaucoma']['correct'] / results['glaucoma']['total'] if results['glaucoma']['total'] > 0 else 0
    print(f"\nGlaucoma Images (should predict glaucoma):")
    print(f"  Total: {results['glaucoma']['total']}")
    print(f"  Correct (True Positives): {results['glaucoma']['correct']}")
    print(f"  Sensitivity: {glaucoma_accuracy:.2%}")
    
    # Overall accuracy
    total_correct = results['normal']['correct'] + results['glaucoma']['correct']
    total_images = results['normal']['total'] + results['glaucoma']['total']
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_images})")
    
    # Confusion matrix
    true_negatives = results['normal']['correct']
    false_positives = results['normal']['total'] - results['normal']['correct']
    true_positives = results['glaucoma']['correct']
    false_negatives = results['glaucoma']['total'] - results['glaucoma']['correct']
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {true_negatives:4d}  (Normal correctly classified)")
    print(f"  False Positives: {false_positives:4d}  (Normal incorrectly as Glaucoma)")
    print(f"  True Positives:  {true_positives:4d}  (Glaucoma correctly classified)")
    print(f"  False Negatives: {false_negatives:4d}  (Glaucoma incorrectly as Normal)")
    
    # Calculate additional metrics
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"\nPrecision: {precision:.2%}")
    
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall (Sensitivity): {recall:.2%}")
    
    if (true_positives + false_positives + false_negatives) > 0:
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        print(f"F1 Score: {f1:.3f}")
    
    # Show some examples of errors
    print("\n" + "="*80)
    print("Error Analysis")
    print("="*80)
    
    # False positives (normal predicted as glaucoma)
    false_positives_list = [p for p in results['normal']['predictions'] if p['predicted_glaucoma']]
    if false_positives_list:
        print(f"\nFalse Positives (Normal predicted as Glaucoma): {len(false_positives_list)}")
        for i, fp in enumerate(false_positives_list[:5], 1):
            print(f"  {i}. {fp['filename']}: prob={fp['glaucoma_prob']:.3f}")
    
    # False negatives (glaucoma missed)
    false_negatives_list = [p for p in results['glaucoma']['predictions'] if not p['predicted_glaucoma']]
    if false_negatives_list:
        print(f"\nFalse Negatives (Glaucoma missed): {len(false_negatives_list)}")
        for i, fn in enumerate(false_negatives_list[:5], 1):
            print(f"  {i}. {fn['filename']}: prob={fn['glaucoma_prob']:.3f}")
    
    # Probability distribution
    print("\n" + "="*80)
    print("Glaucoma Probability Distribution")
    print("="*80)
    
    normal_probs = [p['glaucoma_prob'] for p in results['normal']['predictions']]
    glaucoma_probs = [p['glaucoma_prob'] for p in results['glaucoma']['predictions']]
    
    print(f"\nNormal images - Glaucoma probability:")
    print(f"  Mean: {np.mean(normal_probs):.3f}")
    print(f"  Median: {np.median(normal_probs):.3f}")
    print(f"  Std: {np.std(normal_probs):.3f}")
    print(f"  Min: {np.min(normal_probs):.3f}, Max: {np.max(normal_probs):.3f}")
    
    print(f"\nGlaucoma images - Glaucoma probability:")
    print(f"  Mean: {np.mean(glaucoma_probs):.3f}")
    print(f"  Median: {np.median(glaucoma_probs):.3f}")
    print(f"  Std: {np.std(glaucoma_probs):.3f}")
    print(f"  Min: {np.min(glaucoma_probs):.3f}, Max: {np.max(glaucoma_probs):.3f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
