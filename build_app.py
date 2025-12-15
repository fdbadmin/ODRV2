#!/usr/bin/env python3
"""
Build script to create ODRV2.app for macOS distribution.

Usage:
    python build_app.py

This creates a standalone macOS application bundle in dist/ODRV2.app
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / 'models' / 'unified_v3_retrain'
FOLDS = [0, 1, 4]

def check_models():
    """Verify all required model files exist."""
    missing = []
    for fold in FOLDS:
        model_path = MODEL_DIR / f'fold_{fold}' / 'best_model.pth'
        if not model_path.exists():
            missing.append(str(model_path))
    
    if missing:
        print("ERROR: Missing model files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    
    print(f"✓ Found all {len(FOLDS)} model files")

def build_app():
    """Build the macOS app using PyInstaller."""
    
    # Collect data files for models
    add_data_args = []
    for fold in FOLDS:
        src = MODEL_DIR / f'fold_{fold}' / 'best_model.pth'
        dest = f'models/unified_v3_retrain/fold_{fold}'
        add_data_args.extend(['--add-data', f'{src}:{dest}'])
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', 'ODRV2',
        '--windowed',  # No console window
        '--onedir',    # Directory bundle (faster startup than onefile)
        '--clean',
        '--noconfirm',
        
        # Hidden imports for PyTorch/timm
        '--hidden-import', 'timm',
        '--hidden-import', 'timm.models',
        '--hidden-import', 'timm.models.convnext',
        '--hidden-import', 'torch',
        '--hidden-import', 'torchvision',
        '--hidden-import', 'PIL',
        '--hidden-import', 'cv2',
        
        # Exclude unnecessary packages to reduce size
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'tkinter',
        '--exclude-module', 'IPython',
        '--exclude-module', 'jupyter',
        '--exclude-module', 'notebook',
        '--exclude-module', 'pytest',
        '--exclude-module', 'sphinx',
        '--exclude-module', 'tensorboard',
        
        # macOS specific
        '--osx-bundle-identifier', 'com.odrv2.app',
        
        *add_data_args,
        
        'desktop_app_bundled.py'
    ]
    
    print("\n🔧 Building ODRV2.app...")
    print(f"Command: {' '.join(cmd[:10])}...")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode == 0:
        app_path = PROJECT_ROOT / 'dist' / 'ODRV2.app'
        if app_path.exists():
            print(f"\n✅ Successfully built: {app_path}")
            print(f"\nTo run: open {app_path}")
            print("To distribute: zip the ODRV2.app folder")
        else:
            # Check for directory bundle
            dir_path = PROJECT_ROOT / 'dist' / 'ODRV2'
            if dir_path.exists():
                print(f"\n✅ Successfully built: {dir_path}")
    else:
        print("\n❌ Build failed")
        sys.exit(1)

if __name__ == '__main__':
    print("=" * 50)
    print("ODRV2 macOS App Builder")
    print("=" * 50)
    
    check_models()
    build_app()
