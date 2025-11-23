"""Compare ODIR training dataset with ACRIMA external dataset"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import preprocess_batch
from src.inference.service import load_ensemble, _normalize, _load_image


def analyze_image_statistics(image_paths, label):
    """Compute statistics for a set of images"""
    stats = {
        'sizes': [],
        'aspect_ratios': [],
        'mean_brightness': [],
        'std_brightness': [],
        'mean_rgb': [],
        'std_rgb': [],
        'contrast': [],
    }
    
    print(f"\nAnalyzing {len(image_paths)} {label} images...")
    for img_path in tqdm(image_paths[:200], desc=label):  # Sample 200 for speed
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Size and aspect ratio
            w, h = img.size
            stats['sizes'].append((w, h))
            stats['aspect_ratios'].append(w / h)
            
            # Brightness
            gray = np.mean(img_array, axis=2)
            stats['mean_brightness'].append(np.mean(gray))
            stats['std_brightness'].append(np.std(gray))
            
            # RGB channels
            stats['mean_rgb'].append(np.mean(img_array, axis=(0, 1)))
            stats['std_rgb'].append(np.std(img_array, axis=(0, 1)))
            
            # Contrast (std of grayscale)
            stats['contrast'].append(np.std(gray))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return stats


def analyze_preprocessed_features(image_paths, config, device, label):
    """Analyze preprocessed image features"""
    features = {
        'mean_after_preprocess': [],
        'std_after_preprocess': [],
        'min_val': [],
        'max_val': [],
    }
    
    print(f"\nAnalyzing preprocessed features for {label}...")
    for img_path in tqdm(image_paths[:200], desc=f"{label} (preprocessed)"):
        try:
            img_tensor = _load_image(img_path, config.image_size).to(device)
            img_tensor = _normalize(img_tensor)
            
            img_np = img_tensor.cpu().numpy()
            features['mean_after_preprocess'].append(np.mean(img_np))
            features['std_after_preprocess'].append(np.std(img_np))
            features['min_val'].append(np.min(img_np))
            features['max_val'].append(np.max(img_np))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return features


def plot_comparisons(odir_stats, acrima_stats, output_path):
    """Create comparison plots"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('ODIR Training vs ACRIMA External Dataset Comparison', fontsize=16)
    
    # 1. Image sizes
    ax = axes[0, 0]
    odir_sizes = [s[0] for s in odir_stats['sizes']]
    acrima_sizes = [s[0] for s in acrima_stats['sizes']]
    ax.hist([odir_sizes, acrima_sizes], bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Image Width (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Image Width Distribution')
    ax.legend()
    
    # 2. Aspect ratios
    ax = axes[0, 1]
    ax.hist([odir_stats['aspect_ratios'], acrima_stats['aspect_ratios']], 
            bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Aspect Ratio (W/H)')
    ax.set_ylabel('Count')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    
    # 3. Mean brightness
    ax = axes[0, 2]
    ax.hist([odir_stats['mean_brightness'], acrima_stats['mean_brightness']], 
            bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Mean Brightness')
    ax.set_ylabel('Count')
    ax.set_title('Brightness Distribution')
    ax.legend()
    
    # 4. Contrast
    ax = axes[1, 0]
    ax.hist([odir_stats['contrast'], acrima_stats['contrast']], 
            bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Contrast (Std Dev)')
    ax.set_ylabel('Count')
    ax.set_title('Contrast Distribution')
    ax.legend()
    
    # 5. Red channel mean
    ax = axes[1, 1]
    odir_red = [rgb[0] for rgb in odir_stats['mean_rgb']]
    acrima_red = [rgb[0] for rgb in acrima_stats['mean_rgb']]
    ax.hist([odir_red, acrima_red], bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Red Channel Mean')
    ax.set_ylabel('Count')
    ax.set_title('Red Channel Distribution')
    ax.legend()
    
    # 6. Green channel mean
    ax = axes[1, 2]
    odir_green = [rgb[1] for rgb in odir_stats['mean_rgb']]
    acrima_green = [rgb[1] for rgb in acrima_stats['mean_rgb']]
    ax.hist([odir_green, acrima_green], bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Green Channel Mean')
    ax.set_ylabel('Count')
    ax.set_title('Green Channel Distribution')
    ax.legend()
    
    # 7. Blue channel mean
    ax = axes[2, 0]
    odir_blue = [rgb[2] for rgb in odir_stats['mean_rgb']]
    acrima_blue = [rgb[2] for rgb in acrima_stats['mean_rgb']]
    ax.hist([odir_blue, acrima_blue], bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
    ax.set_xlabel('Blue Channel Mean')
    ax.set_ylabel('Count')
    ax.set_title('Blue Channel Distribution')
    ax.legend()
    
    # 8. Preprocessed mean
    ax = axes[2, 1]
    if 'mean_after_preprocess' in odir_stats and 'mean_after_preprocess' in acrima_stats:
        ax.hist([odir_stats['mean_after_preprocess'], acrima_stats['mean_after_preprocess']], 
                bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
        ax.set_xlabel('Mean After Preprocessing')
        ax.set_ylabel('Count')
        ax.set_title('Preprocessed Mean Distribution')
        ax.legend()
    
    # 9. Preprocessed std
    ax = axes[2, 2]
    if 'std_after_preprocess' in odir_stats and 'std_after_preprocess' in acrima_stats:
        ax.hist([odir_stats['std_after_preprocess'], acrima_stats['std_after_preprocess']], 
                bins=30, label=['ODIR', 'ACRIMA'], alpha=0.7)
        ax.set_xlabel('Std After Preprocessing')
        ax.set_ylabel('Count')
        ax.set_title('Preprocessed Std Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")


def print_statistics_summary(odir_stats, acrima_stats):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("DATASET COMPARISON SUMMARY")
    print("="*80)
    
    # Image dimensions
    print("\n1. IMAGE DIMENSIONS:")
    odir_widths = [s[0] for s in odir_stats['sizes']]
    odir_heights = [s[1] for s in odir_stats['sizes']]
    acrima_widths = [s[0] for s in acrima_stats['sizes']]
    acrima_heights = [s[1] for s in acrima_stats['sizes']]
    
    print(f"  ODIR:")
    print(f"    Width:  {np.mean(odir_widths):.1f} ± {np.std(odir_widths):.1f} px")
    print(f"    Height: {np.mean(odir_heights):.1f} ± {np.std(odir_heights):.1f} px")
    print(f"    Aspect Ratio: {np.mean(odir_stats['aspect_ratios']):.3f} ± {np.std(odir_stats['aspect_ratios']):.3f}")
    
    print(f"  ACRIMA:")
    print(f"    Width:  {np.mean(acrima_widths):.1f} ± {np.std(acrima_widths):.1f} px")
    print(f"    Height: {np.mean(acrima_heights):.1f} ± {np.std(acrima_heights):.1f} px")
    print(f"    Aspect Ratio: {np.mean(acrima_stats['aspect_ratios']):.3f} ± {np.std(acrima_stats['aspect_ratios']):.3f}")
    
    # Brightness
    print("\n2. BRIGHTNESS:")
    print(f"  ODIR:   {np.mean(odir_stats['mean_brightness']):.1f} ± {np.std(odir_stats['mean_brightness']):.1f}")
    print(f"  ACRIMA: {np.mean(acrima_stats['mean_brightness']):.1f} ± {np.std(acrima_stats['mean_brightness']):.1f}")
    diff_pct = ((np.mean(acrima_stats['mean_brightness']) - np.mean(odir_stats['mean_brightness'])) / 
                np.mean(odir_stats['mean_brightness']) * 100)
    print(f"  Difference: {diff_pct:+.1f}%")
    
    # Contrast
    print("\n3. CONTRAST:")
    print(f"  ODIR:   {np.mean(odir_stats['contrast']):.1f} ± {np.std(odir_stats['contrast']):.1f}")
    print(f"  ACRIMA: {np.mean(acrima_stats['contrast']):.1f} ± {np.std(acrima_stats['contrast']):.1f}")
    diff_pct = ((np.mean(acrima_stats['contrast']) - np.mean(odir_stats['contrast'])) / 
                np.mean(odir_stats['contrast']) * 100)
    print(f"  Difference: {diff_pct:+.1f}%")
    
    # RGB channels
    print("\n4. RGB CHANNEL MEANS:")
    odir_rgb_mean = np.mean(odir_stats['mean_rgb'], axis=0)
    acrima_rgb_mean = np.mean(acrima_stats['mean_rgb'], axis=0)
    
    print(f"  ODIR:   R={odir_rgb_mean[0]:.1f}, G={odir_rgb_mean[1]:.1f}, B={odir_rgb_mean[2]:.1f}")
    print(f"  ACRIMA: R={acrima_rgb_mean[0]:.1f}, G={acrima_rgb_mean[1]:.1f}, B={acrima_rgb_mean[2]:.1f}")
    print(f"  Difference: R={acrima_rgb_mean[0]-odir_rgb_mean[0]:+.1f}, " +
          f"G={acrima_rgb_mean[1]-odir_rgb_mean[1]:+.1f}, " +
          f"B={acrima_rgb_mean[2]-odir_rgb_mean[2]:+.1f}")
    
    # Preprocessed features
    if 'mean_after_preprocess' in odir_stats and 'mean_after_preprocess' in acrima_stats:
        print("\n5. AFTER PREPROCESSING:")
        print(f"  ODIR Mean:   {np.mean(odir_stats['mean_after_preprocess']):.3f} ± {np.std(odir_stats['mean_after_preprocess']):.3f}")
        print(f"  ACRIMA Mean: {np.mean(acrima_stats['mean_after_preprocess']):.3f} ± {np.std(acrima_stats['mean_after_preprocess']):.3f}")
        print(f"  ODIR Std:    {np.mean(odir_stats['std_after_preprocess']):.3f} ± {np.std(odir_stats['std_after_preprocess']):.3f}")
        print(f"  ACRIMA Std:  {np.mean(acrima_stats['std_after_preprocess']):.3f} ± {np.std(acrima_stats['std_after_preprocess']):.3f}")
    
    print("\n" + "="*80)


def show_sample_images(odir_paths, acrima_paths, output_path):
    """Show sample images from both datasets"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images: ODIR (top) vs ACRIMA (bottom)', fontsize=14)
    
    # ODIR samples
    for i, img_path in enumerate(odir_paths[:5]):
        img = Image.open(img_path).convert('RGB')
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'ODIR {i+1}', fontsize=10)
    
    # ACRIMA samples
    for i, img_path in enumerate(acrima_paths[:5]):
        img = Image.open(img_path).convert('RGB')
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'ACRIMA {i+1}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved sample images to {output_path}")


def main():
    # Setup paths
    root_dir = Path(".")
    odir_img_dir = root_dir / "RAW DATA FULL"
    acrima_img_dir = root_dir / "external_data/Database/Images"
    models_dir = root_dir / "models"
    config_path = root_dir / "configs/training.yaml"
    output_dir = root_dir / "data/processed/dataset_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ODIR data to get glaucoma images
    print("Loading ODIR dataset information...")
    df = pd.read_csv(root_dir / "data/processed/odir_eye_labels.csv")
    
    # Get ODIR glaucoma images
    odir_glaucoma_df = df[df['Label_G'] == 1].sample(n=min(200, len(df[df['Label_G'] == 1])), random_state=42)
    odir_glaucoma_paths = [odir_img_dir / row['filename'] for _, row in odir_glaucoma_df.iterrows() 
                           if (odir_img_dir / row['filename']).exists()]
    
    # Get ODIR normal images (no glaucoma)
    odir_normal_df = df[df['Label_G'] == 0].sample(n=min(200, len(df[df['Label_G'] == 0])), random_state=42)
    odir_normal_paths = [odir_img_dir / row['filename'] for _, row in odir_normal_df.iterrows() 
                         if (odir_img_dir / row['filename']).exists()]
    
    print(f"Found {len(odir_glaucoma_paths)} ODIR glaucoma images")
    print(f"Found {len(odir_normal_paths)} ODIR normal images")
    
    # Get ACRIMA images
    acrima_all = sorted(list(acrima_img_dir.glob("*.jpg")))
    acrima_glaucoma = [img for img in acrima_all if '_g_' in img.stem][:200]
    acrima_normal = [img for img in acrima_all if '_g_' not in img.stem][:200]
    
    print(f"Found {len(acrima_glaucoma)} ACRIMA glaucoma images")
    print(f"Found {len(acrima_normal)} ACRIMA normal images")
    
    # Load models for preprocessing analysis
    print("\nLoading ensemble models for preprocessing analysis...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models, config = load_ensemble(models_dir, config_path, device)
    
    print("\n" + "="*80)
    print("COMPARING GLAUCOMA IMAGES")
    print("="*80)
    
    # Analyze raw statistics
    odir_glaucoma_stats = analyze_image_statistics(odir_glaucoma_paths, "ODIR Glaucoma")
    acrima_glaucoma_stats = analyze_image_statistics(acrima_glaucoma, "ACRIMA Glaucoma")
    
    # Analyze preprocessed features
    odir_glaucoma_preproc = analyze_preprocessed_features(odir_glaucoma_paths, config, device, "ODIR Glaucoma")
    acrima_glaucoma_preproc = analyze_preprocessed_features(acrima_glaucoma, config, device, "ACRIMA Glaucoma")
    
    # Merge stats
    odir_glaucoma_stats.update(odir_glaucoma_preproc)
    acrima_glaucoma_stats.update(acrima_glaucoma_preproc)
    
    # Print summary
    print_statistics_summary(odir_glaucoma_stats, acrima_glaucoma_stats)
    
    # Create plots
    plot_comparisons(odir_glaucoma_stats, acrima_glaucoma_stats, 
                    output_dir / "glaucoma_comparison.png")
    
    # Show sample images
    show_sample_images(odir_glaucoma_paths, acrima_glaucoma, 
                      output_dir / "glaucoma_samples.png")
    
    print("\n" + "="*80)
    print("COMPARING NORMAL IMAGES")
    print("="*80)
    
    # Analyze normal images
    odir_normal_stats = analyze_image_statistics(odir_normal_paths, "ODIR Normal")
    acrima_normal_stats = analyze_image_statistics(acrima_normal, "ACRIMA Normal")
    
    odir_normal_preproc = analyze_preprocessed_features(odir_normal_paths, config, device, "ODIR Normal")
    acrima_normal_preproc = analyze_preprocessed_features(acrima_normal, config, device, "ACRIMA Normal")
    
    odir_normal_stats.update(odir_normal_preproc)
    acrima_normal_stats.update(acrima_normal_preproc)
    
    print_statistics_summary(odir_normal_stats, acrima_normal_stats)
    
    plot_comparisons(odir_normal_stats, acrima_normal_stats, 
                    output_dir / "normal_comparison.png")
    
    show_sample_images(odir_normal_paths, acrima_normal, 
                      output_dir / "normal_samples.png")
    
    print(f"\n✓ All comparison results saved to {output_dir}/")
    print("\nKey files:")
    print(f"  - {output_dir}/glaucoma_comparison.png")
    print(f"  - {output_dir}/glaucoma_samples.png")
    print(f"  - {output_dir}/normal_comparison.png")
    print(f"  - {output_dir}/normal_samples.png")


if __name__ == "__main__":
    main()
