"""
Download priority datasets for IRFundusSet integration

Downloads:
1. EyePACS (35K DR images) from Kaggle
2. PAPILA (488 glaucoma images) from Figshare  
3. iDRID (516 DR/DME images) from IEEE DataPort

Run with: python scripts/download_irfundus_datasets.py
"""

import subprocess
import urllib.request
import zipfile
import tarfile
import ssl
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    import requests
    
    response = requests.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, DownloadProgressBar(
        unit='B', unit_scale=True, miniters=1, desc=output_path.name, total=total_size
    ) as t:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                t.update(len(chunk))


def download_eyepacs():
    """
    Download EyePACS Diabetic Retinopathy dataset from Kaggle
    
    Size: ~88GB
    Images: 35,108
    Labels: DR grades 0-4 (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)
    """
    print("\n" + "="*80)
    print("DOWNLOADING EYEPACS DATASET (Kaggle)")
    print("="*80)
    
    output_dir = Path("external_data/EyePACS")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nDataset: Diabetic Retinopathy Detection")
    print("Source: Kaggle Competition")
    print("Size: ~88GB (35,108 images)")
    print("Output: external_data/EyePACS/")
    
    # Check if already downloaded
    train_zip = output_dir / "train.zip"
    test_zip = output_dir / "test.zip"
    labels_csv = output_dir / "trainLabels.csv"
    
    if train_zip.exists() and test_zip.exists() and labels_csv.exists():
        print("\n✓ EyePACS already downloaded!")
        
        # Check if extracted
        train_dir = output_dir / "train"
        if train_dir.exists() and len(list(train_dir.glob("*.jpeg"))) > 30000:
            print("✓ Already extracted!")
            return True
        else:
            print("⚠ Not yet extracted. Extracting...")
    else:
        print("\nDownloading via Kaggle API...")
        print("This will take 1-2 hours depending on connection speed...")
        
        try:
            # Download using Kaggle API (use full path in venv)
            import sys
            kaggle_cmd = str(Path(sys.executable).parent / "kaggle")
            subprocess.run([
                kaggle_cmd, "competitions", "download",
                "-c", "diabetic-retinopathy-detection",
                "-p", str(output_dir)
            ], check=True)
            
            print("\n✓ Download complete!")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Kaggle API error: {e}")
            print("\nPlease ensure:")
            print("1. You've accepted competition rules at:")
            print("   https://www.kaggle.com/c/diabetic-retinopathy-detection/rules")
            print("2. Your Kaggle API credentials are valid (~/.kaggle/kaggle.json)")
            return False
        except FileNotFoundError:
            print("\n✗ Kaggle CLI not installed!")
            print("Install with: pip install kaggle")
            return False
    
    # Extract files
    print("\nExtracting train images...")
    if train_zip.exists():
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✓ Train images extracted")
    
    print("\nExtracting test images...")
    if test_zip.exists():
        with zipfile.ZipFile(test_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✓ Test images extracted")
    
    print(f"\n✓ EyePACS ready at: {output_dir}")
    return True


def download_papila():
    """
    Download PAPILA glaucoma dataset from Figshare
    
    Size: ~150MB
    Images: 488
    Labels: Glaucoma (yes/no) + optic disc/cup segmentations
    """
    print("\n" + "="*80)
    print("DOWNLOADING PAPILA DATASET (Figshare)")
    print("="*80)
    
    output_dir = Path("external_data/PAPILA")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nDataset: PAPILA - Optic Disc Segmentation and Glaucoma")
    print("Source: Figshare")
    print("Size: ~150MB (488 images)")
    print("Output: external_data/PAPILA/")
    
    # Check if already downloaded
    fundus_dir = output_dir / "FundusImages"
    if fundus_dir.exists() and len(list(fundus_dir.glob("*.jpg"))) > 400:
        print("\n✓ PAPILA already downloaded and extracted!")
        return True
    
    # Figshare direct download URL
    url = "https://figshare.com/ndownloader/files/28731784"
    zip_file = output_dir / "papila.zip"
    
    print("\nDownloading from Figshare...")
    try:
        download_url(url, zip_file)
        print("\n✓ Download complete!")
        
        print("\nExtracting...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✓ Extraction complete!")
        
        # Clean up zip
        zip_file.unlink()
        
        print(f"\n✓ PAPILA ready at: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nManual download available at:")
        print("https://figshare.com/articles/dataset/PAPILA/14798004")
        return False


def download_idrid():
    """
    Download iDRID dataset from IEEE DataPort
    
    Size: ~2GB
    Images: 516
    Labels: DR grade (0-4), DME grade (0-2), plus segmentations
    
    NOTE: Requires IEEE DataPort account and manual download
    """
    print("\n" + "="*80)
    print("DOWNLOADING IDRID DATASET (IEEE DataPort)")
    print("="*80)
    
    output_dir = Path("external_data/iDRID")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nDataset: Indian Diabetic Retinopathy Image Dataset")
    print("Source: IEEE DataPort")
    print("Size: ~2GB (516 images)")
    print("Output: external_data/iDRID/")
    
    # Check if already downloaded
    grade_dir = output_dir / "B. Disease Grading"
    if grade_dir.exists():
        train_dir = grade_dir / "1. Original Images" / "a. Training Set"
        if train_dir.exists() and len(list(train_dir.glob("*.jpg"))) > 300:
            print("\n✓ iDRID already downloaded!")
            return True
    
    print("\n⚠ IEEE DataPort requires manual download")
    print("\nSteps:")
    print("1. Create account at: https://ieee-dataport.org/")
    print("2. Visit: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid")
    print("3. Click 'Download' and save to: external_data/iDRID/")
    print("4. Extract all ZIP files in external_data/iDRID/")
    print("5. Run this script again to verify")
    
    return False


def verify_downloads():
    """Verify all datasets are downloaded and ready"""
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Check EyePACS
    eyepacs_train = Path("external_data/EyePACS/train")
    if eyepacs_train.exists():
        count = len(list(eyepacs_train.glob("*.jpeg")))
        results['EyePACS'] = count > 30000
        print(f"\n✓ EyePACS: {count} images found")
    else:
        results['EyePACS'] = False
        print("\n✗ EyePACS: Not found")
    
    # Check PAPILA
    papila_fundus = Path("external_data/PAPILA/FundusImages")
    if papila_fundus.exists():
        count = len(list(papila_fundus.glob("*.jpg")))
        results['PAPILA'] = count > 400
        print(f"✓ PAPILA: {count} images found")
    else:
        results['PAPILA'] = False
        print("✗ PAPILA: Not found")
    
    # Check iDRID
    idrid_train = Path("external_data/iDRID/B. Disease Grading/1. Original Images/a. Training Set")
    if idrid_train.exists():
        count = len(list(idrid_train.glob("*.jpg")))
        results['iDRID'] = count > 300
        print(f"✓ iDRID: {count} images found")
    else:
        results['iDRID'] = False
        print("✗ iDRID: Not found")
    
    print("\n" + "="*80)
    if all(results.values()):
        print("✓ ALL DATASETS READY FOR INTEGRATION!")
    else:
        missing = [k for k, v in results.items() if not v]
        print(f"⚠ Missing datasets: {', '.join(missing)}")
    print("="*80)
    
    return all(results.values())


def main():
    print("="*80)
    print("IRFUNDUSSET DATASET DOWNLOADER")
    print("="*80)
    print("\nThis script will download 3 priority datasets:")
    print("1. EyePACS (35K DR images, ~88GB)")
    print("2. PAPILA (488 glaucoma images, ~150MB)")
    print("3. iDRID (516 DR/DME images, ~2GB)")
    print("\nTotal size: ~90GB")
    print("Estimated time: 2-3 hours (depending on connection)")
    
    response = input("\nProceed with download? (y/n): ").lower().strip()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Download datasets
    print("\n" + "="*80)
    print("STARTING DOWNLOADS")
    print("="*80)
    
    # 1. PAPILA (small, fast)
    papila_success = download_papila()
    
    # 2. iDRID (manual download required)
    idrid_success = download_idrid()
    
    # 3. EyePACS (large, slow)
    eyepacs_success = download_eyepacs()
    
    # Verify everything
    verify_downloads()
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Run: python scripts/create_irfundus_config.py")
    print("2. Run: python scripts/harmonize_irfundus_datasets.py")
    print("3. Run: python scripts/create_unified_dataset.py (updated version)")
    print("4. Retrain ensemble on ~45K images")


if __name__ == "__main__":
    main()
