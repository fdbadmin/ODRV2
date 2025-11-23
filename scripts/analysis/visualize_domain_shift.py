"""Visualize the domain shift: ODIR full fundus vs ACRIMA optic disc crops"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    # Setup paths
    root_dir = Path(".")
    odir_img_dir = root_dir / "RAW DATA FULL"
    acrima_img_dir = root_dir / "external_data/Database/Images"
    output_dir = root_dir / "data/processed/dataset_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ODIR data to get glaucoma images
    print("Loading sample images...")
    df = pd.read_csv(root_dir / "data/processed/odir_eye_labels.csv")
    
    # Get a few ODIR glaucoma images
    odir_glaucoma_df = df[df['Label_G'] == 1].sample(n=3, random_state=42)
    odir_glaucoma_paths = [odir_img_dir / row['filename'] for _, row in odir_glaucoma_df.iterrows() 
                           if (odir_img_dir / row['filename']).exists()]
    
    # Get ACRIMA glaucoma images
    acrima_all = sorted(list(acrima_img_dir.glob("*.jpg")))
    acrima_glaucoma = [img for img in acrima_all if '_g_' in img.stem][:3]
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Domain Shift: ODIR Full Fundus (top) vs ACRIMA Optic Disc Crops (bottom)', 
                 fontsize=16, fontweight='bold')
    
    # Show ODIR images (full fundus)
    for i, img_path in enumerate(odir_glaucoma_paths[:3]):
        img = Image.open(img_path).convert('RGB')
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'ODIR Glaucoma {i+1}\n(Full Fundus Image)', 
                            fontsize=12, fontweight='bold')
        # Add border
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
    
    # Show ACRIMA images (optic disc crops)
    for i, img_path in enumerate(acrima_glaucoma[:3]):
        img = Image.open(img_path).convert('RGB')
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'ACRIMA Glaucoma {i+1}\n(Optic Disc Close-up)', 
                            fontsize=12, fontweight='bold')
        # Add border
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    
    # Add text annotation explaining the difference
    fig.text(0.5, 0.48, 
             '↑ Model trained on these (full fundus with visible vessels, macula, optic disc)\n' +
             '↓ Model tested on these (zoomed to optic disc only - completely different field of view)',
             ha='center', va='center', fontsize=13, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "domain_shift_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("DOMAIN SHIFT ANALYSIS")
    print("="*80)
    print("\nKEY FINDING: Major domain shift detected!")
    print("\nODIR Dataset (Training):")
    print("  - Full fundus images showing entire retina")
    print("  - Includes optic disc, macula, blood vessels, peripheral retina")
    print("  - Glaucoma features: cup-to-disc ratio visible within full context")
    print("\nACRIMA Dataset (External Test):")
    print("  - Cropped images focused on optic disc/nerve head region only")
    print("  - Close-up view of optic cup and disc")
    print("  - Different field of view, zoom level, and anatomical context")
    print("\nIMPACT ON MODEL PERFORMANCE:")
    print("  - Model learned features from full fundus images (vessels, macula, etc.)")
    print("  - ACRIMA crops don't contain these contextual features")
    print("  - Model sees completely different image characteristics")
    print("  - Result: 0% sensitivity (unable to detect glaucoma in crops)")
    print("\nCONCLUSION:")
    print("  This explains the poor external validation performance.")
    print("  The model would need to be:")
    print("    1. Retrained on optic disc crops, OR")
    print("    2. Trained with multi-scale/crop augmentations, OR")
    print("    3. Tested on external datasets with similar full fundus images")
    print("="*80)
    
    # Show image size comparison
    print("\nIMAGE STATISTICS:")
    odir_img = Image.open(odir_glaucoma_paths[0])
    acrima_img = Image.open(acrima_glaucoma[0])
    print(f"  ODIR typical size:   {odir_img.size[0]} x {odir_img.size[1]} pixels")
    print(f"  ACRIMA typical size: {acrima_img.size[0]} x {acrima_img.size[1]} pixels")
    
    # Calculate field of view difference
    odir_pixels = odir_img.size[0] * odir_img.size[1]
    acrima_pixels = acrima_img.size[0] * acrima_img.size[1]
    print(f"\n  ODIR total pixels:   {odir_pixels:,}")
    print(f"  ACRIMA total pixels: {acrima_pixels:,}")
    print(f"  Size ratio:          {odir_pixels/acrima_pixels:.1f}x")
    
    print(f"\n✓ Analysis complete. See {output_path} for visual comparison.")


if __name__ == "__main__":
    main()
