# Public Datasets for Rare Retinal Diseases

This document lists publicly available datasets to augment training for rare diseases currently under-represented in Unified Dataset V2.

## Current Status (Unified V2)

| Disease | Cases | Ratio | Status |
|---------|-------|-------|--------|
| **Cataract** | 402 | 1:111 | ⚠️ RARE - Need 1000+ |
| **AMD** | 419 | 1:107 | ⚠️ RARE - Need 1000+ |
| **HTN** | 204 | 1:219 | ⚠️ VERY RARE - Need 500+ |
| **Myopia** | 394 | 1:113 | ⚠️ RARE - Need 1000+ |

---

## 🔍 Cataract Datasets

### 1. Cataract Dataset (Kaggle)
- **Size**: 1,038 images
- **Labels**: Binary cataract classification
- **Access**: Public
- **URL**: https://www.kaggle.com/datasets/jr2ngb/cataractdataset
- **Download**: `kaggle datasets download -d jr2ngb/cataractdataset`
- **Impact**: 402 → 1,440 cases (+258%)

### 2. ZJU-Cataract Dataset
- **Size**: ~500 images
- **Labels**: Cataract grades
- **Access**: Research request required
- **Institution**: Zhejiang University
- **Notes**: Chinese patient population, good diversity

### 3. OIA-ODIR Extended
- **Size**: Additional 2000+ ODIR images (not in 5K subset)
- **Labels**: 8 disease categories including cataract
- **Access**: Available through ODIR challenge organizers
- **URL**: https://odir2019.grand-challenge.org/

---

## 🔍 AMD/ARMD Datasets

### 1. ADAM Challenge Dataset ⭐ RECOMMENDED
- **Size**: 400 training + 400 validation
- **Labels**: AMD severity grades + lesion segmentation
- **Access**: Public registration
- **URL**: https://amd.grand-challenge.org/
- **Format**: High-quality fundus images with expert annotations
- **Impact**: 419 → 1,219 cases (+191%)

### 2. iChallenge-AMD
- **Size**: 400 training + 400 validation + 400 test
- **Labels**: AMD binary + cup/disc segmentation
- **Access**: Challenge registration (may still be available)
- **URL**: https://ai.baidu.com/broad/subordinate?dataset=amd
- **Notes**: Chinese population, good for diversity

### 3. Retinal OCT (Kermany et al.)
- **Size**: 37,206 OCT images (can link to fundus if multimodal)
- **Labels**: AMD, DME, Drusen, Normal
- **Access**: Public
- **URL**: https://data.mendeley.com/datasets/rscbjbr9sj/2
- **Notes**: OCT not fundus, but useful if you have paired data

### 4. UK Biobank
- **Size**: 10,000+ with AMD labels
- **Labels**: AMD diagnosis + genetic data
- **Access**: Research application required (~3 months)
- **URL**: https://www.ukbiobank.ac.uk/
- **Cost**: Application fee
- **Notes**: Gold standard but requires formal research proposal

---

## 🔍 Hypertensive Retinopathy Datasets

### 1. MESSIDOR-2
- **Size**: 1,748 images (subset with HTN annotations)
- **Labels**: DR grades + some HTN labels
- **Access**: Public with agreement
- **URL**: https://www.adcis.net/en/third-party/messidor2/
- **Notes**: Not primary HTN dataset but has some labels

### 2. HRF (High-Resolution Fundus)
- **Size**: 45 images (15 healthy + 15 DR + 15 glaucoma + some HTN)
- **Labels**: HTN retinopathy features
- **Access**: Public
- **URL**: https://www5.cs.fau.de/research/data/fundus-images/
- **Notes**: Small but high quality

### 3. DRIVE Dataset
- **Size**: 40 images (subset with HTN)
- **Labels**: Vessel segmentation + HTN indicators
- **Access**: Public registration
- **URL**: https://drive.grand-challenge.org/
- **Notes**: Primarily for vessel segmentation

### 4. Clinical Partnerships ⭐ RECOMMENDED
- **Source**: Local hospitals, ophthalmology clinics
- **Size**: Variable (aim for 500+ images)
- **Access**: IRB approval + data sharing agreement
- **Notes**: Best option for HTN due to limited public data
- **Action**: Contact medical school ophthalmology departments

---

## 🔍 Myopia Datasets

### 1. PALM Challenge ⭐ RECOMMENDED
- **Size**: 1,200 fundus images
- **Labels**: Pathologic myopia + lesion detection
- **Access**: Challenge registration
- **URL**: https://palm.grand-challenge.org/
- **Format**: High myopia with atrophy labels
- **Impact**: 394 → 1,594 cases (+304%)

### 2. ZOC-Myopia Dataset
- **Size**: 200+ high myopia cases
- **Labels**: Myopia severity grades
- **Access**: Research request
- **Institution**: Zhongshan Ophthalmic Center
- **Notes**: Chinese population

### 3. REFUGE Challenge (Myopia Subset)
- **Size**: ~400 images with myopia
- **Labels**: Glaucoma primary, myopia secondary
- **Access**: Public
- **URL**: https://refuge.grand-challenge.org/
- **Notes**: Overlap with glaucoma cases (multi-label useful)

---

## 🔍 Multi-Disease Datasets (All Rare Diseases)

### 1. RFMiD2 (Validation/Test Sets)
- **Size**: Additional 640 images beyond RFMiD1
- **Labels**: 46 disease categories
- **Access**: Public
- **URL**: https://riadd.grand-challenge.org/
- **Notes**: We only use RFMiD1 training (1,920); add validation/test sets

### 2. DDR (Diabetic Retinopathy Dataset)
- **Size**: 13,673 images
- **Labels**: DR grades + lesion segmentation + other diseases
- **Access**: Public
- **URL**: https://github.com/nkicsl/DDR-dataset
- **Notes**: Chinese population, multi-label annotations

### 3. APTOS 2019 Blindness Detection
- **Size**: 3,662 training images
- **Labels**: DR grades (0-4)
- **Access**: Public (Kaggle)
- **URL**: https://www.kaggle.com/c/aptos2019-blindness-detection
- **Notes**: Already checked, primarily DR focused

### 4. Retinal Fundus Multi-Disease (Mendeley)
- **Size**: Various collections, 5000+ images
- **Labels**: Multiple diseases including rare ones
- **Access**: Public
- **URL**: https://data.mendeley.com/ (search "retinal fundus")
- **Notes**: Quality varies, check individual datasets

---

## 📋 Implementation Priority

### Priority 1: Download Now (Before Training)
These datasets will have immediate impact on rare disease performance:

1. **PALM Challenge** (Myopia)
   - +1,200 images
   - Registration required but usually approved quickly
   - Direct myopia labels

2. **Cataract Dataset** (Kaggle)
   - +1,038 images
   - Immediate download with Kaggle API
   - Simple binary labels

3. **ADAM Challenge** (AMD)
   - +800 images
   - Registration process ~1 week
   - High quality annotations

**Expected Improvement:**
- Cataract: 60% → 75% sensitivity
- Myopia: 60% → 80% sensitivity
- AMD: 60% → 75% sensitivity
- HTN: 40% → 50% (limited data available)

### Priority 2: After Initial Training Results
Evaluate if additional data needed based on validation performance:

4. **RFMiD2** (validation/test sets)
5. **DDR Dataset** (multi-disease)
6. **iChallenge-AMD** (if ADAM insufficient)

### Priority 3: Long-term Enhancement
For production deployment with clinical requirements:

7. **UK Biobank** (requires 3-month application)
8. **Clinical partnerships** (especially for HTN)
9. **ZJU-Cataract** (research request)

---

## 🚀 Quick Start Commands

### Setup Kaggle API
```bash
# Install kaggle
pip install kaggle

# Configure credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download Cataract Dataset
kaggle datasets download -d jr2ngb/cataractdataset -p external_data/
unzip external_data/cataractdataset.zip -d external_data/cataract_kaggle/
```

### Download PALM (Myopia)
```bash
# Register at palm.grand-challenge.org
# After approval, download link provided
# Extract to external_data/PALM/
```

### Download ADAM (AMD)
```bash
# Register at amd.grand-challenge.org
# After approval, download link provided
# Extract to external_data/ADAM/
```

---

## 📊 Expected Dataset After Augmentation

| Disease | Before | After Priority 1 | After Priority 2 | Target |
|---------|--------|------------------|------------------|---------|
| **Cataract** | 402 | 1,440 (+1,038) | 1,940 (+500 ZJU) | 2,000+ |
| **AMD** | 419 | 1,219 (+800) | 2,019 (+800 iChallenge) | 2,000+ |
| **HTN** | 204 | 249 (+45 HRF) | 500+ (clinical) | 1,000+ |
| **Myopia** | 394 | 1,594 (+1,200) | 1,794 (+200 ZOC) | 2,000+ |

**Total Unified V3 Expected:**
- Current V2: 44,673 images
- After Priority 1: ~47,700 images (+6.8%)
- After Priority 2: ~50,000+ images (+11.9%)

---

## ⚠️ Important Notes

1. **Data Quality**: Always verify label quality and imaging modality compatibility
2. **Demographics**: Consider population diversity (Asian vs Caucasian vs African)
3. **Ethics**: Ensure proper data usage agreements and IRB approval if needed
4. **Integration**: Update `create_unified_dataset_v3.py` to incorporate new datasets
5. **Validation**: Keep original external test sets (HYGD, ACRIMA) separate for validation

---

## 📞 Contacts for Research Access

- **ADAM Challenge**: adam-challenge@grand-challenge.org
- **PALM Challenge**: palm-challenge@grand-challenge.org  
- **UK Biobank**: access@ukbiobank.ac.uk
- **ZJU-Cataract**: Contact through research publications
- **Clinical Data**: Work with your institution's medical school

---

Last Updated: 2025-11-23
