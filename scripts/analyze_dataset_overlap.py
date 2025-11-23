"""Analyze disease overlap between ODIR, RFMID1, RFMID2, and HYGD datasets"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("="*80)
    print("MULTI-DATASET DISEASE MAPPING ANALYSIS")
    print("="*80)
    
    # ODIR disease labels
    odir_diseases = {
        'D': 'Diabetic Retinopathy',
        'G': 'Glaucoma',
        'C': 'Cataract',
        'A': 'Age-related Macular Degeneration',
        'H': 'Hypertension',
        'M': 'Myopia',
        'O': 'Other diseases/abnormalities'
    }
    
    print("\n1. ODIR DISEASE CATEGORIES (7 classes):")
    for code, name in odir_diseases.items():
        print(f"   {code}: {name}")
    
    # Load ODIR data
    odir_df = pd.read_csv('data/processed/odir_eye_labels.csv')
    print(f"\n   ODIR Dataset Statistics:")
    for code, name in odir_diseases.items():
        count = odir_df[f'Label_{code}'].sum()
        pct = count / len(odir_df) * 100
        print(f"   - {code} ({name}): {count} ({pct:.1f}%)")
    
    # RFMID1
    print("\n2. RFMID1 DISEASE CATEGORIES (46 classes):")
    rfmid1_df = pd.read_csv('external_data/RFMID1/RFMiD_Training_Labels.csv')
    rfmid1_cols = [col for col in rfmid1_df.columns if col not in ['ID', 'Disease_Risk']]
    print(f"   Total disease columns: {len(rfmid1_cols)}")
    print(f"   Disease categories: {', '.join(rfmid1_cols[:20])}...")
    
    # Count samples per disease
    print(f"\n   RFMID1 Top 10 Diseases by Sample Count:")
    disease_counts = {}
    for col in rfmid1_cols:
        disease_counts[col] = rfmid1_df[col].sum()
    
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    for disease, count in sorted_diseases[:10]:
        pct = count / len(rfmid1_df) * 100
        print(f"   - {disease}: {count} ({pct:.1f}%)")
    
    # RFMID2
    print("\n3. RFMID2 DISEASE CATEGORIES (52 classes):")
    rfmid2_df = pd.read_csv('external_data/RFMID2/Training_set/RFMiD_2_Training_labels.csv', 
                            encoding='latin1')
    rfmid2_cols = [col for col in rfmid2_df.columns if col not in ['ID', 'WNL']]
    print(f"   Total disease columns: {len(rfmid2_cols)}")
    print(f"   Disease categories: {', '.join(rfmid2_cols[:20])}...")
    
    print(f"\n   RFMID2 Top 10 Diseases by Sample Count:")
    disease_counts2 = {}
    for col in rfmid2_cols:
        disease_counts2[col] = rfmid2_df[col].sum()
    
    sorted_diseases2 = sorted(disease_counts2.items(), key=lambda x: x[1], reverse=True)
    for disease, count in sorted_diseases2[:10]:
        pct = count / len(rfmid2_df) * 100
        print(f"   - {disease}: {count} ({pct:.1f}%)")
    
    # HYGD
    print("\n4. HYGD (Hillel Yaffe Glaucoma Dataset):")
    hygd_df = pd.read_csv('external_data/glaucoma_standard/Labels.csv')
    print(f"   Binary classification: Glaucoma (GON+) vs Normal (GON-)")
    print(f"   - GON+ (Glaucoma): {len(hygd_df[hygd_df['Label'] == 'GON+'])}")
    print(f"   - GON- (Normal): {len(hygd_df[hygd_df['Label'] == 'GON-'])}")
    
    # Disease mapping
    print("\n" + "="*80)
    print("POTENTIAL DISEASE MAPPINGS")
    print("="*80)
    
    # Find common diseases
    print("\nLooking for disease overlaps...")
    
    # Check for DR (Diabetic Retinopathy)
    if 'DR' in rfmid1_cols:
        dr_count1 = disease_counts.get('DR', 0)
        dr_count2 = disease_counts2.get('DR', 0) if 'DR' in rfmid2_cols else 0
        odir_d = odir_df['Label_D'].sum()
        print(f"\nDIABETIC RETINOPATHY (D):")
        print(f"  ODIR (D): {odir_d} samples")
        print(f"  RFMID1 (DR): {dr_count1} samples")
        print(f"  RFMID2 (DR): {dr_count2} samples")
        print(f"  ✓ CAN COMBINE: Total {odir_d + dr_count1 + dr_count2} samples")
    
    # Check for ARMD (Age-related Macular Degeneration)
    if 'ARMD' in rfmid1_cols:
        armd_count1 = disease_counts.get('ARMD', 0)
        armd_count2 = disease_counts2.get('ARMD', 0) if 'ARMD' in rfmid2_cols else 0
        odir_a = odir_df['Label_A'].sum()
        print(f"\nAGE-RELATED MACULAR DEGENERATION (A):")
        print(f"  ODIR (A): {odir_a} samples")
        print(f"  RFMID1 (ARMD): {armd_count1} samples")
        print(f"  RFMID2 (ARMD): {armd_count2} samples")
        print(f"  ✓ CAN COMBINE: Total {odir_a + armd_count1 + armd_count2} samples")
    
    # Check for MYA (Myopia)
    if 'MYA' in rfmid1_cols:
        mya_count1 = disease_counts.get('MYA', 0)
        mya_count2 = disease_counts2.get('MYA', 0) if 'MYA' in rfmid2_cols else 0
        odir_m = odir_df['Label_M'].sum()
        print(f"\nMYOPIA (M):")
        print(f"  ODIR (M): {odir_m} samples")
        print(f"  RFMID1 (MYA): {mya_count1} samples")
        print(f"  RFMID2 (MYA): {mya_count2} samples")
        print(f"  ✓ CAN COMBINE: Total {odir_m + mya_count1 + mya_count2} samples")
    
    # Glaucoma
    odir_g = odir_df['Label_G'].sum()
    hygd_g = len(hygd_df[hygd_df['Label'] == 'GON+'])
    print(f"\nGLAUCOMA (G):")
    print(f"  ODIR (G): {odir_g} samples")
    print(f"  HYGD (GON+): {hygd_g} samples")
    
    # Check RFMID for glaucoma-related terms
    glaucoma_terms = [col for col in rfmid1_cols if 'glaucoma' in col.lower() or col in ['GRT', 'ODC', 'ODE', 'ODP']]
    if glaucoma_terms:
        print(f"  RFMID1 potential glaucoma columns: {', '.join(glaucoma_terms)}")
        for term in glaucoma_terms:
            print(f"    - {term}: {disease_counts.get(term, 0)} samples")
    
    glaucoma_terms2 = [col for col in rfmid2_cols if 'glaucoma' in col.lower() or col in ['GRT', 'ODC', 'ODE', 'ODP']]
    if glaucoma_terms2:
        print(f"  RFMID2 potential glaucoma columns: {', '.join(glaucoma_terms2)}")
        for term in glaucoma_terms2:
            print(f"    - {term}: {disease_counts2.get(term, 0)} samples")
    
    print(f"  ✓ CAN COMBINE: At least {odir_g + hygd_g}+ samples")
    
    # Cataract
    if 'C' in odir_diseases:
        odir_c = odir_df['Label_C'].sum()
        print(f"\nCATARACT (C):")
        print(f"  ODIR (C): {odir_c} samples")
        # Look for cataract in RFMID
        cataract_terms = [col for col in rfmid1_cols if 'cataract' in col.lower() or col == 'C']
        if cataract_terms:
            print(f"  RFMID1 potential cataract columns: {', '.join(cataract_terms)}")
        print(f"  Note: May need manual inspection of RFMID categories")
    
    # Hypertension
    odir_h = odir_df['Label_H'].sum()
    htn_count2 = disease_counts2.get('HTN', 0) if 'HTN' in rfmid2_cols else 0
    print(f"\nHYPERTENSION (H):")
    print(f"  ODIR (H): {odir_h} samples")
    print(f"  RFMID2 (HTN): {htn_count2} samples")
    if htn_count2 > 0:
        print(f"  ✓ CAN COMBINE: Total {odir_h + htn_count2} samples")
    
    # Summary
    print("\n" + "="*80)
    print("INTEGRATION SUMMARY")
    print("="*80)
    print("\nConfirmed Overlapping Diseases (can be combined):")
    print("  1. Diabetic Retinopathy (D/DR)")
    print("  2. Age-related Macular Degeneration (A/ARMD)")
    print("  3. Myopia (M/MYA)")
    print("  4. Glaucoma (G/GON+/optic disc abnormalities)")
    print("  5. Hypertension (H/HTN) - RFMID2 only")
    
    print("\nAdditional RFMID-only diseases (can expand model):")
    print("  - Many rare conditions with smaller sample sizes")
    print("  - Could add as additional output classes")
    print("  - Or map to ODIR 'O' (Other) category")
    
    print("\nRecommended Integration Approach:")
    print("  1. Start with confirmed overlaps (D, G, A, M, H)")
    print("  2. Massively increase sample sizes for these core diseases")
    print("  3. Keep ODIR 'C' and 'O' categories for now")
    print("  4. Later: Consider adding high-frequency RFMID diseases")
    
    print("\nExpected Combined Dataset Size:")
    total_images = len(odir_df) + len(hygd_df) + len(rfmid1_df) + len(rfmid2_df)
    print(f"  ODIR: {len(odir_df)}")
    print(f"  HYGD: {len(hygd_df)}")
    print(f"  RFMID1: {len(rfmid1_df)}")
    print(f"  RFMID2: {len(rfmid2_df)}")
    print(f"  TOTAL: ~{total_images} images")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
