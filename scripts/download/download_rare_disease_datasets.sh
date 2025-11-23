#!/bin/bash
# Download public datasets for rare diseases
# Run this script to download Cataract, Myopia, and AMD datasets

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DATA="$PROJECT_ROOT/external_data"

echo "================================================================================"
echo "DOWNLOADING RARE DISEASE DATASETS"
echo "================================================================================"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "⚠️  Kaggle credentials not found!"
    echo ""
    echo "Please set up Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Save kaggle.json to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

# Create directories
mkdir -p "$EXTERNAL_DATA/cataract_kaggle"
mkdir -p "$EXTERNAL_DATA/PALM"
mkdir -p "$EXTERNAL_DATA/ADAM"

echo ""
echo "================================================================================"
echo "1. DOWNLOADING CATARACT DATASET (Kaggle)"
echo "================================================================================"
echo "Dataset: jr2ngb/cataractdataset"
echo "Size: 1,038 images"
echo "Impact: Cataract cases 402 → 1,440 (+258%)"
echo ""

if [ -f "$EXTERNAL_DATA/cataract_kaggle/cataract_dataset.zip" ] || [ -d "$EXTERNAL_DATA/cataract_kaggle/train" ]; then
    echo "✓ Cataract dataset already downloaded"
else
    echo "Downloading..."
    kaggle datasets download -d jr2ngb/cataractdataset -p "$EXTERNAL_DATA/cataract_kaggle/"
    
    echo "Extracting..."
    cd "$EXTERNAL_DATA/cataract_kaggle"
    unzip -q cataractdataset.zip
    echo "✓ Cataract dataset downloaded successfully"
fi

echo ""
echo "================================================================================"
echo "2. PALM CHALLENGE DATASET (Myopia)"
echo "================================================================================"
echo "Dataset: Pathologic Myopia Challenge"
echo "Size: 1,200 images"
echo "Impact: Myopia cases 394 → 1,594 (+304%)"
echo ""
echo "⚠️  PALM dataset requires manual registration:"
echo ""
echo "   1. Visit: https://palm.grand-challenge.org/"
echo "   2. Register for the challenge"
echo "   3. Download 'Training' and 'Validation' sets"
echo "   4. Extract to: $EXTERNAL_DATA/PALM/"
echo ""
echo "Expected structure:"
echo "   PALM/"
echo "   ├── Training/"
echo "   │   ├── Training400/"
echo "   │   │   ├── fundus_images/"
echo "   │   │   └── PM_Label_and_Fovea_Location.xlsx"
echo "   └── PALM-Validation400/"
echo ""

if [ -d "$EXTERNAL_DATA/PALM/Training" ]; then
    echo "✓ PALM dataset found"
else
    echo "⏸  Waiting for manual download"
    echo "   Run this script again after downloading PALM dataset"
fi

echo ""
echo "================================================================================"
echo "3. ADAM CHALLENGE DATASET (AMD)"
echo "================================================================================"
echo "Dataset: Automatic Detection challenge on Age-related Macular degeneration"
echo "Size: 400 training + 400 validation"
echo "Impact: AMD cases 419 → 1,219 (+191%)"
echo ""
echo "⚠️  ADAM dataset requires manual registration:"
echo ""
echo "   1. Visit: https://amd.grand-challenge.org/"
echo "   2. Register for the challenge"
echo "   3. Download 'Training' and 'Validation' sets"
echo "   4. Extract to: $EXTERNAL_DATA/ADAM/"
echo ""
echo "Expected structure:"
echo "   ADAM/"
echo "   ├── Training400/"
echo "   │   ├── AMD/"
echo "   │   └── Non-AMD/"
echo "   └── Validation400/"
echo ""

if [ -d "$EXTERNAL_DATA/ADAM/Training400" ]; then
    echo "✓ ADAM dataset found"
else
    echo "⏸  Waiting for manual download"
    echo "   Run this script again after downloading ADAM dataset"
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

CATARACT_STATUS="⏸ Pending"
PALM_STATUS="⏸ Pending"
ADAM_STATUS="⏸ Pending"

if [ -d "$EXTERNAL_DATA/cataract_kaggle/train" ]; then
    CATARACT_STATUS="✓ Ready"
fi

if [ -d "$EXTERNAL_DATA/PALM/Training" ]; then
    PALM_STATUS="✓ Ready"
fi

if [ -d "$EXTERNAL_DATA/ADAM/Training400" ]; then
    ADAM_STATUS="✓ Ready"
fi

echo "Dataset Status:"
echo "  Cataract (Kaggle):  $CATARACT_STATUS"
echo "  PALM (Myopia):      $PALM_STATUS"
echo "  ADAM (AMD):         $ADAM_STATUS"
echo ""

if [ "$CATARACT_STATUS" = "✓ Ready" ] && [ "$PALM_STATUS" = "✓ Ready" ] && [ "$ADAM_STATUS" = "✓ Ready" ]; then
    echo "✅ All datasets ready! Proceed with:"
    echo "   python scripts/create_unified_dataset_v3.py"
else
    echo "⏳ Waiting for manual downloads. Instructions above."
    echo ""
    echo "Once all datasets are downloaded, run:"
    echo "   python scripts/create_unified_dataset_v3.py"
fi

echo ""
echo "================================================================================"
