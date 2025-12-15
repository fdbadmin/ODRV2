#!/usr/bin/env python3
"""
Start training with live monitoring capability
"""
import subprocess
import sys

# Start training process
process = subprocess.Popen(
    [sys.executable, '-u', 'scripts/training/train.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Write to both file and print to console
with open('training_live.log', 'w') as f:
    for line in process.stdout:
        print(line, end='', flush=True)
        f.write(line)
        f.flush()

process.wait()
print(f"\nTraining completed with exit code: {process.returncode}")
