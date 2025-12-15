# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ODRV2 Desktop Application.
Creates a macOS .app bundle.

Build with: pyinstaller odrv2.spec
"""

import os
from pathlib import Path

block_cipher = None

# Project root
project_root = Path(SPECPATH)

# Collect model files (the 3-fold ensemble)
model_files = []
model_dir = project_root / 'models' / 'unified_v3_retrain'
for fold in [0, 1, 4]:
    fold_model = model_dir / f'fold_{fold}' / 'best_model.pth'
    if fold_model.exists():
        # (source, destination_folder)
        dest = f'models/unified_v3_retrain/fold_{fold}'
        model_files.append((str(fold_model), dest))

a = Analysis(
    ['desktop_app.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=model_files,
    hiddenimports=[
        'timm',
        'timm.models',
        'timm.models.convnext',
        'torch',
        'torchvision',
        'PIL',
        'cv2',
        'numpy',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ODRV2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,  # Required for macOS
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ODRV2',
)

app = BUNDLE(
    coll,
    name='ODRV2.app',
    icon=None,  # Add icon path here if you have one: 'assets/icon.icns'
    bundle_identifier='com.odrv2.app',
    info_plist={
        'CFBundleName': 'ODRV2',
        'CFBundleDisplayName': 'ODRV2 - Ocular Disease Recognition',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Image Files',
                'CFBundleTypeRole': 'Viewer',
                'LSItemContentTypes': [
                    'public.image',
                    'public.jpeg',
                    'public.png',
                    'public.tiff',
                ],
            }
        ],
    },
)
