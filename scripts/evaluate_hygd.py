"""Evaluate model on Hillel Yaffe Glaucoma Dataset (HYGD)"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.service import load_ensemble, _normalize, _load_image


def main():
    # Setup paths
    root_dir = Path(".")
    hygd_dir = root_dir / "external_data/glaucoma_standard"
    image_dir = hygd_dir / "Images"
    labels_path = hygd_dir / "Labels.csv"
    models_dir = root_dir / "models"
    config_path = root_dir / "configs/training.yaml"
    
    # Load labels
    print("Loading HYGD dataset labels...")
    df = pd.read_csv(labels_path)
    print(f"Total images: {len(df)}")
    print(f"  GON+ (glaucoma): {len(df[df['Label'] == 'GON+'])}")
    print(f"  GON- (normal):   {len(df[df['Label'] == 'GON-'])}")
    
    # Load ensemble
    print("\nLoading ensemble models...")
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
    
    # Track results
    results = {
        'normal': {'correct': 0, 'total': 0, 'predictions': []},
        'glaucoma': {'correct': 0, 'total': 0, 'predictions': []}
    }
    
    # Process all images
    print("\nRunning predictions...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="HYGD"):
        img_path = image_dir / row['Image Name']
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping...")
            continue
        
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
            
            # Determine ground truth
            is_glaucoma = row['Label'] == 'GON+'
            category = 'glaucoma' if is_glaucoma else 'normal'
            
            results[category]['total'] += 1
            results[category]['predictions'].append({
                'filename': row['Image Name'],
                'patient': row['Patient'],
                'predicted_glaucoma': has_glaucoma,
                'glaucoma_prob': glaucoma_prob,
                'quality_score': row['Quality Score']
            })
            
            # Check if correct
            if is_glaucoma and has_glaucoma:
                results['glaucoma']['correct'] += 1
            elif not is_glaucoma and not has_glaucoma:
                results['normal']['correct'] += 1
                
        except Exception as e:
            print(f"Error processing {row['Image Name']}: {e}")
            continue
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS - Hillel Yaffe Glaucoma Dataset (HYGD)")
    print("="*80)
    
    # Normal images (specificity / true negative rate)
    normal_accuracy = results['normal']['correct'] / results['normal']['total'] if results['normal']['total'] > 0 else 0
    print(f"\nNormal Images (GON-, should NOT predict glaucoma):")
    print(f"  Total: {results['normal']['total']}")
    print(f"  Correct (True Negatives): {results['normal']['correct']}")
    print(f"  Specificity: {normal_accuracy:.2%}")
    
    # Glaucoma images (sensitivity / true positive rate)
    glaucoma_accuracy = results['glaucoma']['correct'] / results['glaucoma']['total'] if results['glaucoma']['total'] > 0 else 0
    print(f"\nGlaucoma Images (GON+, should predict glaucoma):")
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
        # Sort by probability to see highest confidence errors
        false_positives_list.sort(key=lambda x: x['glaucoma_prob'], reverse=True)
        for i, fp in enumerate(false_positives_list[:5], 1):
            print(f"  {i}. {fp['filename']}: prob={fp['glaucoma_prob']:.3f}, quality={fp['quality_score']:.2f}")
    
    # False negatives (glaucoma missed)
    false_negatives_list = [p for p in results['glaucoma']['predictions'] if not p['predicted_glaucoma']]
    if false_negatives_list:
        print(f"\nFalse Negatives (Glaucoma missed): {len(false_negatives_list)}")
        # Sort by probability to see lowest confidence predictions
        false_negatives_list.sort(key=lambda x: x['glaucoma_prob'])
        for i, fn in enumerate(false_negatives_list[:5], 1):
            print(f"  {i}. {fn['filename']}: prob={fn['glaucoma_prob']:.3f}, quality={fn['quality_score']:.2f}")
    
    # Probability distribution
    print("\n" + "="*80)
    print("Glaucoma Probability Distribution")
    print("="*80)
    
    normal_probs = [p['glaucoma_prob'] for p in results['normal']['predictions']]
    glaucoma_probs = [p['glaucoma_prob'] for p in results['glaucoma']['predictions']]
    
    print(f"\nNormal images (GON-) - Glaucoma probability:")
    print(f"  Mean: {np.mean(normal_probs):.3f}")
    print(f"  Median: {np.median(normal_probs):.3f}")
    print(f"  Std: {np.std(normal_probs):.3f}")
    print(f"  Min: {np.min(normal_probs):.3f}, Max: {np.max(normal_probs):.3f}")
    
    print(f"\nGlaucoma images (GON+) - Glaucoma probability:")
    print(f"  Mean: {np.mean(glaucoma_probs):.3f}")
    print(f"  Median: {np.median(glaucoma_probs):.3f}")
    print(f"  Std: {np.std(glaucoma_probs):.3f}")
    print(f"  Min: {np.min(glaucoma_probs):.3f}, Max: {np.max(glaucoma_probs):.3f}")
    
    # Quality score analysis
    print("\n" + "="*80)
    print("Image Quality Analysis")
    print("="*80)
    
    normal_quality = [p['quality_score'] for p in results['normal']['predictions']]
    glaucoma_quality = [p['quality_score'] for p in results['glaucoma']['predictions']]
    
    print(f"\nImage Quality Scores:")
    print(f"  Normal:   Mean={np.mean(normal_quality):.2f}, Median={np.median(normal_quality):.2f}")
    print(f"  Glaucoma: Mean={np.mean(glaucoma_quality):.2f}, Median={np.median(glaucoma_quality):.2f}")
    
    # Check if quality affects performance
    if false_positives_list:
        fp_quality = [p['quality_score'] for p in false_positives_list]
        print(f"  False Positives: Mean={np.mean(fp_quality):.2f}")
    
    if false_negatives_list:
        fn_quality = [p['quality_score'] for p in false_negatives_list]
        print(f"  False Negatives: Mean={np.mean(fn_quality):.2f}")
    
    print("\n" + "="*80)
    print("\nNOTE: HYGD contains full fundus images (Deep Fundus Images)")
    print("This is more comparable to the ODIR training set than the ACRIMA crops.")
    print("="*80)


if __name__ == "__main__":
    main()
