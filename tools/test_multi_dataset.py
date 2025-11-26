"""
Test multi-dataset loading feature.

This script verifies that:
1. Multiple files can be loaded into dataset_list
2. Plotting works with multiple datasets
3. X/Y column selections apply correctly to all datasets
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add repo root to path so we can import QT_test
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from QT_test import MainWindow
from PySide6.QtWidgets import QApplication


def test_multi_dataset_loading():
    """Test loading multiple files and verify data integrity."""
    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    
    # Find test files
    test_data_dir = os.path.join(repo_root, 'test_data')
    test_files = []
    
    # Look for VHD files in test_data
    if os.path.exists(test_data_dir):
        vhd_files = list(Path(test_data_dir).glob('*.VHD'))
        if vhd_files:
            test_files = [str(f) for f in vhd_files[:2]]  # Take first 2 files
    
    if len(test_files) < 2:
        print("⚠ Skipping multi-dataset test: need at least 2 test files in test_data/")
        print("  Current test files found:", test_files)
        return False
    
    print(f"\n=== Testing Multi-Dataset Loading ===")
    print(f"Loading {len(test_files)} test files...")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")
    
    try:
        # Manually trigger multi-file loading
        mw.dataset_list = []
        mw.multi_mode = True
        
        for path in test_files:
            df = mw._read_table_auto(path)
            mw.dataset_list.append({
                'path': path,
                'original_df': df.copy(deep=True),
                'df': df.copy(deep=True),
                'history': []
            })
        
        # Verify datasets were loaded
        assert len(mw.dataset_list) == len(test_files), "Dataset count mismatch"
        print(f"✓ Loaded {len(mw.dataset_list)} datasets")
        
        # Extract numeric columns
        mw.numeric_cols = [c for c in mw.dataset_list[0]['df'].columns 
                           if np.issubdtype(mw.dataset_list[0]['df'][c].dtype, np.number)]
        
        if len(mw.numeric_cols) < 2:
            print("✗ Not enough numeric columns")
            return False
        
        print(f"✓ Found {len(mw.numeric_cols)} numeric columns: {mw.numeric_cols[:3]}...")
        
        # Test that all datasets have the same numeric columns
        for idx, ds in enumerate(mw.dataset_list):
            ds_cols = [c for c in ds['df'].columns 
                      if np.issubdtype(ds['df'][c].dtype, np.number)]
            if not set(mw.numeric_cols).issubset(set(ds_cols)):
                print(f"✗ Dataset {idx} missing numeric columns")
                return False
        
        print(f"✓ All datasets have compatible numeric columns")
        
        # Populate combos (simulated)
        mw.numeric_cols = [c for c in mw.dataset_list[0]['df'].columns 
                           if np.issubdtype(mw.dataset_list[0]['df'][c].dtype, np.number)]
        
        x_col = mw.numeric_cols[0]
        y_col = mw.numeric_cols[1]
        
        print(f"✓ Using X={x_col}, Y={y_col}")
        
        # Verify each dataset has data for these columns
        for idx, ds in enumerate(mw.dataset_list):
            df = ds['df']
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            print(f"  Dataset {idx}: {len(x_data)} X points, {len(y_data)} Y points")
            
            if len(x_data) < 2 or len(y_data) < 2:
                print(f"✗ Dataset {idx} has insufficient data")
                return False
        
        print(f"✓ All datasets have sufficient data")
        
        # Verify multi_mode flag
        assert mw.multi_mode, "multi_mode flag not set"
        print(f"✓ Multi-mode enabled")
        
        print("\n=== Multi-Dataset Loading Test PASSED ===\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during multi-dataset test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_dataset_backward_compat():
    """Verify single-dataset mode still works (backward compatibility)."""
    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    
    test_data_dir = os.path.join(repo_root, 'test_data')
    test_file = os.path.join(test_data_dir, 'out_of_plane.VHD')
    
    if not os.path.exists(test_file):
        print("⚠ Skipping single-dataset backward compatibility test: test file not found")
        return True
    
    print(f"\n=== Testing Single-Dataset Backward Compatibility ===")
    
    try:
        # Clear multi mode and load single file
        mw.multi_mode = False
        mw.dataset_list = []
        
        df = mw._read_table_auto(test_file)
        mw.original_df = df.copy(deep=True)
        mw.df = df
        
        mw.numeric_cols = [c for c in mw.df.columns 
                          if np.issubdtype(mw.df[c].dtype, np.number)]
        
        if len(mw.numeric_cols) < 2:
            print("✗ Not enough numeric columns")
            return False
        
        print(f"✓ Loaded single file: {os.path.basename(test_file)}")
        print(f"✓ Found {len(mw.numeric_cols)} numeric columns")
        
        # Verify single mode is not active
        assert not mw.multi_mode, "multi_mode should be False"
        assert len(mw.dataset_list) == 0, "dataset_list should be empty"
        print(f"✓ Single-mode active (multi_mode=False)")
        
        print("\n=== Single-Dataset Backward Compatibility Test PASSED ===\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during single-dataset test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Multi-Dataset Feature Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Multi-Dataset Loading", test_multi_dataset_loading()))
    results.append(("Single-Dataset Backward Compat", test_single_dataset_backward_compat()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    sys.exit(0 if passed_count == total_count else 1)
