#!/usr/bin/env python3
"""
Test hardware configuration and optimal settings before training
"""

import torch
import psutil
import time
import numpy as np
from pathlib import Path

def test_mps_availability():
    """Test if MPS is available and working"""
    print("\n" + "="*80)
    print("1. TESTING MPS (METAL) AVAILABILITY")
    print("="*80)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        if not torch.backends.mps.is_built():
            print("   PyTorch was not built with MPS support")
        else:
            print("   MPS device not found (not on Apple Silicon?)")
        return False
    
    print("✅ MPS is available")
    
    # Test basic operations
    try:
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y  # Matrix multiplication
        print("✅ MPS operations working")
        return True
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return False

def test_mixed_precision():
    """Test if mixed precision (FP16) works on MPS"""
    print("\n" + "="*80)
    print("2. TESTING MIXED PRECISION (FP16)")
    print("="*80)
    
    if not torch.backends.mps.is_available():
        print("⏭  Skipping (MPS not available)")
        return False
    
    try:
        device = torch.device("mps")
        
        # Test FP16 operations
        x = torch.randn(1000, 1000, device=device, dtype=torch.float16)
        y = torch.randn(1000, 1000, device=device, dtype=torch.float16)
        z = x @ y
        
        print("✅ Mixed precision (FP16) working")
        return True
    except Exception as e:
        print(f"⚠️  Mixed precision test failed: {e}")
        print("   Use --precision 32 instead")
        return False

def benchmark_batch_sizes():
    """Test different batch sizes to find optimal"""
    print("\n" + "="*80)
    print("3. BENCHMARKING BATCH SIZES")
    print("="*80)
    
    if not torch.backends.mps.is_available():
        print("⏭  Skipping (MPS not available)")
        return
    
    device = torch.device("mps")
    
    # Simulate model (EfficientNet-B3 size)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(64)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(64, 7)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel().to(device)
    
    batch_sizes = [16, 24, 32, 40, 48]
    img_size = 448
    
    print(f"\nTesting with {img_size}x{img_size} images...")
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Memory (GB)':<15} Status")
    print("-" * 60)
    
    for bs in batch_sizes:
        try:
            # Clear cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Measure memory before
            mem_before = psutil.Process().memory_info().rss / (1024**3)
            
            # Create batch
            x = torch.randn(bs, 3, img_size, img_size, device=device)
            
            # Forward pass
            start = time.time()
            with torch.no_grad():
                y = model(x)
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            elapsed = (time.time() - start) * 1000
            
            # Measure memory after
            mem_after = psutil.Process().memory_info().rss / (1024**3)
            mem_used = mem_after - mem_before
            
            total_mem = psutil.virtual_memory().used / (1024**3)
            
            if total_mem < 28:  # Safe limit for 32GB
                status = "✅ Safe"
            elif total_mem < 30:
                status = "⚠️  Caution"
            else:
                status = "❌ Risky"
            
            print(f"{bs:<12} {elapsed:<12.1f} {total_mem:<15.1f} {status}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:<12} {'N/A':<12} {'OOM':<15} ❌ Too large")
                break
            else:
                print(f"{bs:<12} {'Error':<12} {str(e)[:15]:<15}")
        except Exception as e:
            print(f"{bs:<12} {'Error':<12} {str(e)[:15]:<15}")
    
    print("\nRecommendation:")
    total_mem = psutil.virtual_memory().used / (1024**3)
    if total_mem < 20:
        print("  Batch size 40-48: Aggressive (monitor memory)")
    elif total_mem < 24:
        print("  Batch size 32: Optimal (recommended)")
    else:
        print("  Batch size 24: Conservative (safest)")

def check_disk_space():
    """Check if sufficient disk space"""
    print("\n" + "="*80)
    print("4. CHECKING DISK SPACE")
    print("="*80)
    
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    
    print(f"\nDisk usage:")
    print(f"  Total: {disk.total / (1024**3):.1f} GB")
    print(f"  Used:  {disk.used / (1024**3):.1f} GB ({disk.percent}%)")
    print(f"  Free:  {free_gb:.1f} GB")
    
    if free_gb < 15:
        print("\n❌ CRITICAL: Insufficient disk space!")
        print("   Need at least 15-20 GB free for training")
        print("   See optimization guide for cleanup commands")
        return False
    elif free_gb < 25:
        print("\n⚠️  WARNING: Low disk space")
        print("   Recommended to free up more space")
        return True
    else:
        print("\n✅ Sufficient disk space")
        return True

def check_data_availability():
    """Check if training data is ready"""
    print("\n" + "="*80)
    print("5. CHECKING TRAINING DATA")
    print("="*80)
    
    required_files = [
        'data/processed/unified_v3/unified_dataset_v3_filtered.csv',
        'data/processed/unified_v3/unified_train_v3.csv',
        'data/processed/unified_v3/unified_val_v3.csv',
        'data/processed/unified_v3/unified_test_v3.csv'
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✅ {path.name}")
        else:
            print(f"❌ {path.name} - MISSING!")
            all_exist = False
    
    return all_exist

def print_recommendations():
    """Print final recommendations"""
    print("\n" + "="*80)
    print("RECOMMENDED TRAINING COMMAND")
    print("="*80)
    
    print("\n# Set environment variables")
    print("export PYTORCH_ENABLE_MPS_FALLBACK=1")
    print("export OMP_NUM_THREADS=4")
    print("")
    print("# Start training (test with 1 epoch first)")
    print("python scripts/train_unified_v2.py \\")
    print("  --data-dir data/processed/unified_v3 \\")
    print("  --train-csv unified_train_v3.csv \\")
    print("  --val-csv unified_val_v3.csv \\")
    print("  --accelerator mps \\")
    print("  --precision 16-mixed \\")
    print("  --batch-size 32 \\")
    print("  --num-workers 4 \\")
    print("  --max-epochs 1    # Test with 1 epoch first!")
    print("")
    print("# If successful, run full training:")
    print("python scripts/train_unified_v2.py \\")
    print("  --data-dir data/processed/unified_v3 \\")
    print("  --train-csv unified_train_v3.csv \\")
    print("  --val-csv unified_val_v3.csv \\")
    print("  --accelerator mps \\")
    print("  --precision 16-mixed \\")
    print("  --batch-size 32 \\")
    print("  --num-workers 4 \\")
    print("  --max-epochs 30")

def main():
    print("="*80)
    print("HARDWARE OPTIMIZATION TEST FOR TRAINING")
    print("="*80)
    
    # Run tests
    mps_ok = test_mps_availability()
    fp16_ok = test_mixed_precision()
    
    if mps_ok:
        benchmark_batch_sizes()
    
    disk_ok = check_disk_space()
    data_ok = check_data_availability()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nMPS (GPU):          {'✅ Ready' if mps_ok else '❌ Not available'}")
    print(f"Mixed Precision:    {'✅ Supported' if fp16_ok else '⚠️  Use FP32'}")
    print(f"Disk Space:         {'✅ Sufficient' if disk_ok else '❌ Insufficient'}")
    print(f"Training Data:      {'✅ Ready' if data_ok else '❌ Missing files'}")
    
    if mps_ok and disk_ok and data_ok:
        print("\n✅ All systems ready for training!")
        print_recommendations()
    else:
        print("\n⚠️  Please fix issues above before training")
        if not disk_ok:
            print("\nPriority: Free up disk space!")

if __name__ == "__main__":
    main()
