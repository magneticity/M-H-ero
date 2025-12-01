#!/usr/bin/env python3
"""
Test Y-offset (center_y) for multi-dataset mode.
Run with: env/bin/python tools/test_center_y_multi.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtWidgets
import QT_test

def test_center_y_multi():
    """Test that center_y works correctly in multi-dataset mode."""
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
    
    print("Testing Y-offset (center_y) in multi-dataset mode...")
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
    import numpy as np
    win.numeric_cols = [c for c in win.dataset_list[0]['df'].columns 
                        if np.issubdtype(win.dataset_list[0]['df'][c].dtype, np.number)]
    
    if len(win.numeric_cols) < 2:
        print("✗ Need at least 2 numeric columns")
        return False
    
    print(f"✓ Found {len(win.numeric_cols)} numeric columns")
    
    # Populate combos
    win._populate_combos()
    
    # Get Y column
    y_col = win.yCombo.currentText()
    print(f"✓ Using Y column: {y_col}")
    
    # Capture initial means
    initial_means = []
    for i, ds in enumerate(win.dataset_list):
        mean_val = ds['df'][y_col].mean()
        initial_means.append(mean_val)
        print(f"  Dataset {i}: initial mean = {mean_val:.6f}")
    
    # Apply center_y
    print("\nApplying center_y...")
    win.center_y_about_zero()
    
    # Check results
    print("\nChecking results:")
    success = True
    for i, ds in enumerate(win.dataset_list):
        new_mean = ds['df'][y_col].mean()
        print(f"  Dataset {i}: new mean = {new_mean:.6e} (should be ~0)")
        
        if abs(new_mean) > 1e-10:
            print(f"    ⚠ Mean not close to zero!")
            success = False
        
        # Check history was recorded
        ds_history = ds.get('history', [])
        if not ds_history or ds_history[-1]['op'] != 'center_y':
            print(f"    ✗ Per-dataset history not recorded")
            success = False
    
    # Check global history
    if not win.history or win.history[-1]['op'] != 'center_y_multi':
        print("✗ Global history not recorded")
        success = False
    else:
        print("✓ Global history recorded (center_y_multi)")
        params = win.history[-1]['params']
        print(f"  - Applied to {params.get('num_datasets')} datasets")
        print(f"  - Offset list: {params.get('offset_list')}")
    
    # Test undo
    print("\nTesting undo...")
    win.undo_last_operation()
    
    # Check means restored
    for i, ds in enumerate(win.dataset_list):
        restored_mean = ds['df'][y_col].mean()
        print(f"  Dataset {i}: restored mean = {restored_mean:.6f} (original: {initial_means[i]:.6f})")
        
        if abs(restored_mean - initial_means[i]) > 1e-10:
            print(f"    ✗ Mean not restored correctly!")
            success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Y-offset multi-dataset test PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Y-offset multi-dataset test FAILED")
        print("=" * 60)
    
    return success

if __name__ == '__main__':
    try:
        success = test_center_y_multi()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
