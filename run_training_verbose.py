#!/usr/bin/env python3
"""Wrapper to run training with forced output flushing."""
import sys
import os
import multiprocessing

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("Starting training wrapper...", flush=True)

    # Now import and run the training
    sys.path.insert(0, '.')
    from scripts.training.train import main

    print("About to call main()...", flush=True)
    main()
