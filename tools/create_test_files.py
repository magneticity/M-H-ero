#!/usr/bin/env python3
"""
Simple manual test for multi-dataset export.
Creates test files and loads them, then you can manually test export.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd

# Create test data files
temp_dir = tempfile.mkdtemp(prefix='mhero_export_test_')
print(f"Created test directory: {temp_dir}")

for i in range(1, 4):
    h = np.linspace(-10, 10, 50)
    # Each dataset has slightly different characteristics
    m = np.tanh(h * (1 + i*0.1)) + 0.05 * np.random.randn(50)
    
    df = pd.DataFrame({'H (Oe)': h, 'M (emu)': m})
    filepath = os.path.join(temp_dir, f'test_dataset_{i}.txt')
    df.to_csv(filepath, sep='\t', index=False)
    print(f"Created: {filepath}")

print("\n" + "="*60)
print("Test files created. Now:")
print("1. Run the application: env/bin/python QT_test.py")
print("2. Open multiple files (select all 3 test files)")
print("3. Apply some operations (BG, drift, etc.)")
print("4. Test Export -> Export current loop to TXT")
print("5. Test Export -> Copy current loop to clipboard")
print("="*60)
print(f"\nTest directory: {temp_dir}")
print("(Remember to clean up this directory when done)")
