#!/usr/bin/env python3
"""
Inspect the model file to see what format it's in
"""

import os

model_path = 'models/advanced/xgboost_best.pkl'

print("="*60)
print("MODEL FILE INSPECTOR")
print("="*60)

if not os.path.exists(model_path):
    print(f"\n✗ Model file not found at: {model_path}")
    print("\nChecking what files exist in models/advanced/...")
    
    if os.path.exists('models/advanced/'):
        files = os.listdir('models/advanced/')
        print(f"\nFiles found ({len(files)}):")
        for f in files:
            size = os.path.getsize(f'models/advanced/{f}')
            print(f"  - {f} ({size} bytes)")
    else:
        print("✗ Directory models/advanced/ does not exist")
    
    exit(1)

print(f"\n✓ Found: {model_path}")
print(f"  Size: {os.path.getsize(model_path)} bytes")

# Read first few bytes to determine file type
print("\nInspecting file header...")
with open(model_path, 'rb') as f:
    header = f.read(20)
    print(f"  First 20 bytes (hex): {header.hex()}")
    print(f"  First 20 bytes (repr): {repr(header)}")

# Try to detect format
if header.startswith(b'\x80'):
    print("\n  → Looks like a pickle file")
elif header.startswith(b'PK'):
    print("\n  → Looks like a ZIP/joblib file")
elif header.startswith(b'{'):
    print("\n  → Looks like JSON")
elif b'xgboost' in header.lower():
    print("\n  → Looks like XGBoost binary format")
else:
    print("\n  → Unknown format")

print("\n" + "="*60)
print("ALTERNATIVE: CHECK TRAINING OUTPUT")
print("="*60)
print("\nThe easiest way to get feature importance is from your")
print("training script output. Look for lines like:")
print("  Feature X1: 0.2845")
print("  Feature X2: 0.1923")
print("  etc.")
print("\nOr check if you have evaluation results saved in:")
print("  - results/ directory")
print("  - logs/ directory")
print("  - Any .txt or .json files with metrics")
print("\nYou can also check your Gradio dashboard - it should show")
print("the feature importance plot with the actual values!")