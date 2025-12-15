#!/usr/bin/env python3
"""
Bundled app launcher for ODRV2.
This is used by PyInstaller to properly locate resources in the .app bundle.
"""

import sys
from pathlib import Path

# Set up paths for bundled app
if getattr(sys, 'frozen', False):
    # Running as bundled app
    import os
    bundle_dir = Path(sys._MEIPASS)
    os.chdir(bundle_dir)
    
    # Patch the model directory lookup
    import src.desktop.inference as inference
    original_init = inference.FundusPredictor.__init__
    
    def patched_init(self, model_dir=None, device=None):
        if model_dir is None:
            model_dir = bundle_dir / 'models' / 'unified_v3_retrain'
        original_init(self, model_dir, device)
    
    inference.FundusPredictor.__init__ = patched_init

# Now run the main app
from desktop_app import main
main()
