# AI-Assisted Fundus Screening Tool
## Clinical Summary for Ophthalmologists

**Document Date**: December 15, 2025  
**Intended Audience**: Ophthalmologists, Optometrists, Clinical Staff  
**System Name**: ODRV2 - Ocular Disease Recognition Version 2

---

## What Is This Tool?

This is an **AI-assisted screening tool** that analyzes standard fundus photographs to flag potential ocular diseases. It is designed to help prioritize patients who may need closer examination, particularly in high-volume screening settings or underserved areas with limited access to specialists.

**The tool can screen for seven conditions:**
1. Diabetic Retinopathy (any grade)
2. Glaucoma
3. Cataract
4. Age-related Macular Degeneration (AMD)
5. Hypertensive Retinopathy
6. Pathological Myopia
7. Normal/Other findings

---

## How Does It Work?

### The Technical Approach (Simplified)

The system uses **deep learning** - a type of artificial intelligence that learns patterns from examples. It was trained by showing it over 32,000 fundus images with known diagnoses, allowing it to learn the visual patterns associated with each disease.

Key points:
- **Training data**: 32,157 images from 20,287 patients
- **Validation**: Tested on 6,282 images the AI never saw during training
- **Patient separation**: Training and test patients were completely separate (no data leakage)
- **Multi-label capable**: Can detect multiple diseases in the same image

### What the AI "Sees"

The model examines the entire fundus image and learns to recognize:
- Microaneurysms, hemorrhages, and exudates (DR)
- Cup-to-disc ratio changes (Glaucoma)
- Lens opacities visible in the fundus (Cataract)
- Drusen and pigmentary changes (AMD)
- Vascular changes (Hypertension)
- Posterior staphyloma and myopic changes (Myopia)

---

## Performance Results

### Overall Accuracy

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Macro F1 Score** | 82% | Overall balanced accuracy across all diseases |
| **AUC-ROC** | 97% | Excellent ability to distinguish disease from normal |

### Per-Disease Performance

| Disease | Detection Rate (Recall) | False Alarm Rate | Clinical Assessment |
|---------|------------------------|------------------|---------------------|
| **Myopia** | 87% | Very low (<1%) | ✅ Highly reliable |
| **Glaucoma** | 81% | Low (8%) | ✅ Clinically useful |
| **Cataract** | 79% | Moderate (20%) | ⚠️ Good, verify findings |
| **AMD** | 76% | Very low (5%) | ✅ Reliable for screening |
| **Diabetic Retinopathy** | 67% | Low (12%) | ⚠️ Good, but misses 1/3 |
| **Hypertension** | 47% | Very low (6%) | ⛔ Limited - misses half |

---

## Clinical Interpretation Guide

### When to Trust the AI

**High Confidence (Reliable for Screening):**
- ✅ **Myopia detection** - 87% detection rate with almost no false positives
- ✅ **Glaucoma flags** - 81% detection, though dilated exam still required
- ✅ **AMD screening** - Catches 76% of cases, good for referral triage

**Moderate Confidence (Use with Clinical Judgment):**
- ⚠️ **Diabetic Retinopathy** - Detects 67% of cases; always examine diabetic patients regardless of AI result
- ⚠️ **Cataract** - Good detection but higher false positive rate

**Low Confidence (Limited Utility):**
- ⛔ **Hypertensive Retinopathy** - Only catches 47% of cases; do not rely on negative result

### Key Clinical Caveats

1. **A negative AI result does NOT rule out disease**
   - The AI misses some cases, especially subtle early disease
   - Clinical history and examination remain essential

2. **This is a screening aid, not a diagnostic tool**
   - All positive flags should be verified by clinical examination
   - The AI cannot grade severity (e.g., DR grading not provided)

3. **Image quality matters**
   - Poor quality images may lead to unreliable results
   - The AI was trained on standard fundus camera images

4. **Rare conditions may be missed**
   - The AI was not trained on rare diseases (e.g., retinitis pigmentosa, choroidal tumors)
   - Unusual presentations may not be flagged

---

## Recommended Use Cases

### Appropriate Uses

| Setting | Use Case | Benefit |
|---------|----------|---------|
| **Primary Care Screening** | Flag diabetic patients for ophthalmology referral | Prioritize limited specialist time |
| **High-Volume Clinics** | Pre-screen patients before examination | Reduce cognitive load |
| **Teleophthalmology** | Triage remote fundus images | Focus expert review on concerning cases |
| **Research Studies** | Automated annotation of large datasets | Consistent baseline classification |

### NOT Recommended For

- ❌ Replacing ophthalmologist diagnosis
- ❌ Final diagnosis without clinical correlation
- ❌ Grading DR severity (ETDRS levels)
- ❌ Monitoring treatment response
- ❌ Medico-legal documentation without human review

---

## Strengths of This System

1. **Trained on diverse data** - Multiple datasets from different populations and cameras
2. **No data leakage** - Training and test patients completely separate
3. **Multi-disease detection** - Can flag multiple conditions simultaneously
4. **High specificity** - Low false positive rates minimize unnecessary referrals
5. **Validated on held-out data** - Performance reflects real-world generalization

---

## Known Limitations

### Limited Training Data for Rare Conditions

| Disease | Training Cases | Limitation |
|---------|---------------|------------|
| Hypertension | 145 | Insufficient data, poor recall |
| AMD | 288 | Small sample, may miss variants |
| Myopia | 275 | Small sample, but performs well |

### Other Limitations

- **Binary detection only** - Does not grade severity
- **Single image analysis** - Does not compare to prior images
- **No OCT integration** - Fundus photos only
- **Population bias** - May perform differently in populations not represented in training data

---

## Technical Validation Summary

### How We Validated the AI

1. **Patient-level data split**: No patient appears in both training and test sets
2. **Stratified sampling**: Disease distribution maintained across splits
3. **Cross-validation**: 5-fold CV during development to tune the model
4. **Held-out test set**: 20% of data (6,282 images) never seen during training
5. **Threshold optimization**: Decision thresholds tuned to balance sensitivity/specificity

### Statistical Confidence

The test set contains:
- 6,282 images from 4,058 unique patients
- Largest classes: Diabetic Retinopathy (1,703), Other (4,299)
- Smallest classes: Hypertension (36), Myopia (54), AMD (55)

Performance on rare classes should be interpreted with caution due to small sample sizes.

---

## Practical Integration

### Suggested Workflow

```
1. Patient has fundus photo taken (standard protocol)
2. Image submitted to AI system
3. AI returns probability for each disease
4. Results flagged for review:
   - Green: Low probability across all diseases
   - Yellow: Moderate probability or mixed signals
   - Red: High probability for one or more diseases
5. Ophthalmologist reviews flagged cases
6. Clinical examination and diagnosis as usual
```

### What the AI Output Looks Like

For each image, the system provides:
- Probability (0-100%) for each of 7 conditions
- Binary flag (Present/Absent) based on optimized thresholds
- Confidence indicator based on probability margins

---

## Questions & Answers

**Q: Can this replace a dilated fundus exam?**
A: No. This is a screening aid for undilated fundus photos. It cannot detect peripheral pathology and should not replace comprehensive examination.

**Q: How should I document AI results in the medical record?**
A: Document that AI-assisted screening was performed and note any flagged conditions. Always include your clinical findings and final diagnosis.

**Q: What if the AI and I disagree?**
A: Trust your clinical judgment. The AI may miss subtle findings or flag artifacts. Document both the AI result and your clinical impression.

**Q: Is this FDA approved?**
A: This is a research system. It has not undergone regulatory approval for clinical use. Use for research and educational purposes only.

**Q: How often is the AI wrong?**
A: Overall, the AI achieves 82% accuracy across all diseases. Individual disease performance varies (see performance table above).

---

## Summary for Clinical Practice

| Aspect | Summary |
|--------|---------|
| **Primary Use** | Screening aid for fundus photo triage |
| **Best Performance** | Myopia, Glaucoma, AMD detection |
| **Limited Performance** | Hypertensive retinopathy |
| **Key Strength** | High specificity, low false positives |
| **Key Weakness** | Misses some cases, especially rare conditions |
| **Bottom Line** | Useful for screening prioritization, not diagnosis |

---

## Contact & Support

For technical questions about the AI system, model training, or integration:
- **Developer**: Fabian De Bie
- **Repository**: ODRV2 (Ocular Disease Recognition Version 2)
- **Documentation**: See TECHNICAL_REPORT.md for full technical details

---

*This document was prepared to provide clinical context for AI-assisted fundus screening. The system described is for research purposes and has not been validated for clinical deployment.*
