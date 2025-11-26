#!/usr/bin/env python3
"""
Test loop closure drift correction for multi-dataset mode.
Run with: env/bin/python tools/test_drift_loopclosure_multi.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtWidgets
import QT_test
import numpy as np

def test_drift_loopclosure_multi():
    """Test that drift loop closure works correctly in multi-dataset mode."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create main window
    win = QT_test.MainWindow()
    
    # Load two test files
    test_files = [
        "../test_data/out_of_plane.VHD",
        "../test_data/in_plane.VHD"
    ]
    
    print("Testing loop closure drift correction in multi-dataset mode...")
    print("=" * 60)
    
    # Simulate file loading
    win.dataset_list = []
    win.multi_mode = True
    
    for fname in test_files:
        path = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(path):
            print(f"⚠ Test file not found: {path}")
            continue
        
        df = win._read_table_auto(path)
        win.dataset_list.append({
            'path': path,
            'original_df': df.copy(deep=True),
            'df': df.copy(deep=True),
            'history': []
        })
    
    if not win.dataset_list:
        print("✗ No test files loaded")
        return False
    
    print(f"✓ Loaded {len(win.dataset_list)} datasets")
    
    # Determine numeric columns from first dataset
    win.numeric_cols = [c for c in win.dataset_list[0]['df'].columns 
                        if np.issubdtype(win.dataset_list[0]['df'][c].dtype, np.number)]
    
    if len(win.numeric_cols) < 2:
        print("✗ Need at least 2 numeric columns")
        return False
    
    print(f"✓ Found {len(win.numeric_cols)} numeric columns")
    
    # Populate combos
    win._populate_combos()
    
    # Get X/Y columns
    x_col = win.xCombo.currentText()
    y_col = win.yCombo.currentText()
    print(f"✓ Using X={x_col}, Y={y_col}")
    
    # Capture initial endpoint differences (window means)
    initial_diffs = []
    end_frac = 0.02
    for i, ds in enumerate(win.dataset_list):
        y = ds['df'][y_col].to_numpy()
        n = len(y)
        window = max(1, int(end_frac * n))
        
        finite = np.isfinite(y)
        finite_indices = np.where(finite)[0]
        
        if finite_indices.size < 2:
            print(f"  Dataset {i}: Not enough finite points")
            initial_diffs.append(np.nan)
            continue
        
        start_indices = finite_indices[:window]
        end_indices = finite_indices[-window:]
        
        start_mean = np.nanmean(y[start_indices])
        end_mean = np.nanmean(y[end_indices])
        diff = end_mean - start_mean
        initial_diffs.append(diff)
        print(f"  Dataset {i}: window mean difference = {diff:.6f}")
    
    # Apply loop closure drift correction
    print("\nApplying loop closure drift correction...")
    win._drift_linear_loopclosure_apply()
    
    # Check results
    print("\nChecking results:")
    success = True
    for i, ds in enumerate(win.dataset_list):
        y = ds['df'][y_col].to_numpy()
        n = len(y)
        window = max(1, int(end_frac * n))
        
        finite = np.isfinite(y)
        finite_indices = np.where(finite)[0]
        
        if finite_indices.size < 2:
            print(f"  Dataset {i}: Not enough finite points")
            continue
        
        start_indices = finite_indices[:window]
        end_indices = finite_indices[-window:]
        
        start_mean = np.nanmean(y[start_indices])
        end_mean = np.nanmean(y[end_indices])
        new_diff = end_mean - start_mean
        
        print(f"  Dataset {i}: new window mean difference = {new_diff:.6e} (should be ~0)")
        
        # Tolerance is relative - should be much smaller than original difference
        original_diff = initial_diffs[i] if not np.isnan(initial_diffs[i]) else 1.0
        if abs(new_diff) > abs(original_diff) * 0.1:  # Should reduce diff by at least 90%
            print(f"    ⚠ Window means not sufficiently aligned! (reduction < 90%)")
            success = False
        else:
            print(f"    ✓ Reduced by {100*(1 - abs(new_diff)/abs(original_diff)):.1f}%")
        
        # Check history was recorded
        ds_history = ds.get('history', [])
        if not ds_history or ds_history[-1]['op'] != 'drift_linear_loopclosure':
            print(f"    ✗ Per-dataset history not recorded")
            success = False
    
    # Check global history
    if not win.history or win.history[-1]['op'] != 'drift_linear_loopclosure_multi':
        print("✗ Global history not recorded")
        success = False
    else:
        print("✓ Global history recorded (drift_linear_loopclosure_multi)")
        params = win.history[-1]['params']
        print(f"  - Applied to {params.get('num_datasets')} datasets")
        print(f"  - Slope list: {params.get('slope_list')}")
    
    # Test undo
    print("\nTesting undo...")
    win.undo_last_operation()
    
    # Check endpoint differences restored
    for i, ds in enumerate(win.dataset_list):
        y = ds['df'][y_col].to_numpy()
        n = len(y)
        window = max(1, int(end_frac * n))
        
        finite = np.isfinite(y)
        finite_indices = np.where(finite)[0]
        
        if finite_indices.size < 2:
            continue
        
        start_indices = finite_indices[:window]
        end_indices = finite_indices[-window:]
        
        start_mean = np.nanmean(y[start_indices])
        end_mean = np.nanmean(y[end_indices])
        restored_diff = end_mean - start_mean
        
        print(f"  Dataset {i}: restored difference = {restored_diff:.6f} (original: {initial_diffs[i]:.6f})")
        
        if abs(restored_diff - initial_diffs[i]) > 1e-10:
            print(f"    ✗ Difference not restored correctly!")
            success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Loop closure drift multi-dataset test PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Loop closure drift multi-dataset test FAILED")
        print("=" * 60)
    
    return success

if __name__ == '__main__':
    try:
        success = test_drift_loopclosure_multi()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
