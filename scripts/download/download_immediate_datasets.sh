#!/bin/bash

################################################################################
# Download Immediately Available Rare Disease Datasets
# No registration required - all public datasets
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DATA="$PROJECT_ROOT/external_data"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 DOWNLOADING IMMEDIATELY AVAILABLE DATASETS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Create external_data directory if it doesn't exist
mkdir -p "$EXTERNAL_DATA"
cd "$EXTERNAL_DATA"

################################################################################
# 1. HRF Dataset - 45 high-quality images with HTN cases
################################################################################
echo "────────────────────────────────────────────────────────────────────────────────"
echo "📥 [1/5] HRF Dataset (HTN Retinopathy)"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d "HRF" ] && [ "$(ls -A HRF 2>/dev/null)" ]; then
    echo "✓ HRF dataset already exists"
else
    echo "Downloading HRF dataset (45 high-resolution images)..."
    mkdir -p HRF
    
    # Download the complete HRF dataset
    curl -L --progress-bar \
        https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip \
        -o HRF/all.zip
    
    echo "Extracting HRF dataset..."
    unzip -q HRF/all.zip -d HRF/
    rm HRF/all.zip
    
    echo "✅ HRF dataset downloaded: $(find HRF -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l) images"
fi

echo ""

################################################################################
# 2. APTOS 2019 - 3,662 images (Kaggle)
################################################################################
echo "────────────────────────────────────────────────────────────────────────────────"
echo "📥 [2/5] APTOS 2019 Dataset"
echo "────────────────────────────────────────────────────────────────────────────────"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install -q kaggle
fi

if [ -d "APTOS2019" ] && [ "$(ls -A APTOS2019 2>/dev/null)" ]; then
    echo "✓ APTOS 2019 dataset already exists"
else
    echo "Downloading APTOS 2019 dataset (3,662 images)..."
    echo "⚠️  Note: Requires Kaggle API credentials (~/.kaggle/kaggle.json)"
    
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "❌ Kaggle credentials not found!"
        echo "   Please set up Kaggle API: https://www.kaggle.com/docs/api"
        echo "   Download kaggle.json and place it in ~/.kaggle/"
        echo "   Then run: chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "⏭  Skipping APTOS 2019..."
    else
        mkdir -p APTOS2019
        
        # Try to download, but continue on error
        if kaggle competitions download -c aptos2019-blindness-detection -p APTOS2019/ 2>/dev/null; then
            echo "Extracting APTOS 2019..."
            cd APTOS2019
            unzip -q "*.zip" 2>/dev/null || true
            cd ..
            echo "✅ APTOS 2019 downloaded: $(find APTOS2019 -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l) images"
        else
            echo "⚠️  APTOS 2019 requires competition rules acceptance"
            echo "   Visit: https://www.kaggle.com/c/aptos2019-blindness-detection"
            echo "   Click 'Join Competition' and accept rules"
            echo "   Then re-run this script"
            echo ""
            echo "⏭  Skipping APTOS 2019 for now..."
        fi
    fi
fi

echo ""

################################################################################
# 3. RFMiD2 - Validation/Test sets (640 images)
################################################################################
echo "────────────────────────────────────────────────────────────────────────────────"
echo "📥 [3/5] RFMiD2 Validation/Test Sets"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d "RFMiD2" ] && [ "$(ls -A RFMiD2 2>/dev/null)" ]; then
    echo "✓ RFMiD2 dataset already exists"
else
    echo "Downloading RFMiD2 validation/test sets..."
    echo "⚠️  Note: Requires Kaggle API credentials"
    
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "❌ Kaggle credentials not found! Skipping RFMiD2..."
    else
        mkdir -p RFMiD2
        
        # Try Kaggle download (silently, continue on error)
        if kaggle competitions download -c rfmid-2-0 -p RFMiD2/ 2>/dev/null || \
           kaggle datasets download -d andrewmvd/retinal-fundus-multi-disease-image-dataset -p RFMiD2/ 2>/dev/null; then
            echo "Extracting RFMiD2..."
            cd RFMiD2
            unzip -q "*.zip" 2>/dev/null || true
            cd ..
            echo "✅ RFMiD2 downloaded: $(find RFMiD2 -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l) images"
        else
            echo "⏭  RFMiD2 download requires manual intervention"
            echo "   Visit: https://riadd.grand-challenge.org/"
            echo "   Or try: https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-multi-disease-image-dataset"
        fi
    fi
fi

echo ""

################################################################################
# 4. REFUGE Challenge - ~400 myopia cases
################################################################################
echo "────────────────────────────────────────────────────────────────────────────────"
echo "📥 [4/5] REFUGE Challenge Dataset (Myopia)"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d "REFUGE" ] && [ "$(ls -A REFUGE 2>/dev/null)" ]; then
    echo "✓ REFUGE dataset already exists"
else
    echo "Downloading REFUGE challenge dataset..."
    mkdir -p REFUGE
    
    # Try Kaggle source
    if [ -f ~/.kaggle/kaggle.json ]; then
        if kaggle datasets download -d arnavjain1/refuge-dataset -p REFUGE/ 2>/dev/null; then
            echo "Extracting REFUGE..."
            cd REFUGE
            unzip -q "*.zip" 2>/dev/null || true
            cd ..
            echo "✅ REFUGE downloaded: $(find REFUGE -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l) images"
        else
            echo "⏭  REFUGE requires manual download"
            echo "   Visit: https://refuge.grand-challenge.org/"
            echo "   Or: https://www.kaggle.com/datasets/arnavjain1/refuge-dataset"
            echo "   Download and extract to: $EXTERNAL_DATA/REFUGE/"
        fi
    else
        echo "⏭  REFUGE requires Kaggle credentials or manual download"
        echo "   Visit: https://refuge.grand-challenge.org/"
    fi
fi

echo ""

################################################################################
# 5. DDR Dataset - 13,673 images
################################################################################
echo "────────────────────────────────────────────────────────────────────────────────"
echo "📥 [5/5] DDR Dataset (Multi-Disease)"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d "DDR-dataset" ] && [ "$(ls -A DDR-dataset 2>/dev/null)" ]; then
    echo "✓ DDR dataset already exists"
else
    echo "Cloning DDR dataset repository..."
    git clone https://github.com/nkicsl/DDR-dataset.git 2>/dev/null || \
    echo "⚠️  DDR repository already exists or unavailable"
    
    if [ -d "DDR-dataset" ]; then
        echo "✅ DDR repository cloned"
        echo "⚠️  Note: DDR images need to be downloaded separately from:"
        echo "   Baidu Cloud: https://pan.baidu.com/s/1VQ1g2u-IjJSzz0mCMnU1WA (code: d2re)"
        echo "   Google Drive: Check DDR-dataset/README.md for links"
        echo ""
        echo "   After downloading, extract to: $EXTERNAL_DATA/DDR-dataset/images/"
    fi
fi

echo ""

################################################################################
# Summary
################################################################################
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 DOWNLOAD SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Count images in each dataset
count_images() {
    local dir=$1
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')
        echo "$count"
    else
        echo "0"
    fi
}

HRF_COUNT=$(count_images "HRF")
APTOS_COUNT=$(count_images "APTOS2019")
RFMID2_COUNT=$(count_images "RFMiD2")
REFUGE_COUNT=$(count_images "REFUGE")
DDR_COUNT=$(count_images "DDR-dataset")

echo "Downloaded Datasets:"
echo "  • HRF (HTN):           $HRF_COUNT images"
echo "  • APTOS 2019:          $APTOS_COUNT images"
echo "  • RFMiD2:              $RFMID2_COUNT images"
echo "  • REFUGE (Myopia):     $REFUGE_COUNT images"
echo "  • DDR:                 $DDR_COUNT images"
echo ""

TOTAL_NEW=$((HRF_COUNT + APTOS_COUNT + RFMID2_COUNT + REFUGE_COUNT + DDR_COUNT))
echo "Total New Images:      $TOTAL_NEW"
echo ""

# Previously downloaded
CATARACT_COUNT=$(count_images "cataract_kaggle")
if [ $CATARACT_COUNT -gt 0 ]; then
    echo "Previously Downloaded:"
    echo "  • Cataract (Kaggle):   $CATARACT_COUNT images"
    echo ""
fi

################################################################################
# Next Steps
################################################################################
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎯 NEXT STEPS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Verify downloads:"
echo "   ls -lh $EXTERNAL_DATA/*/  | head -20"
echo ""
echo "2. Create Unified Dataset V3:"
echo "   python scripts/create_unified_dataset_v3.py"
echo ""
echo "3. Create train/val/test splits:"
echo "   python scripts/create_unified_splits_v3.py"
echo ""
echo "4. Start training:"
echo "   python scripts/train_unified_v2.py"
echo ""
echo "📋 Manual Downloads Still Needed:"
echo "   • PALM (Myopia): https://palm.grand-challenge.org/"
echo "   • ADAM (AMD): https://amd.grand-challenge.org/"
echo "   • DDR images: Check DDR-dataset/README.md"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
