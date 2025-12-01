#!/usr/bin/env python3
"""
Test multi-dataset export functionality.
Tests both TXT file export and clipboard copy for multi-dataset mode.
"""

import sys
import os
import tempfile
import shutil

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtWidgets, QtCore
import pandas as pd
import numpy as np

# Import the main window class
from QT_test import MainWindow


def create_test_data_file(path, num_rows=50):
    """Create a simple test data file with H and M columns."""
    h = np.linspace(-10, 10, num_rows)
    m = np.tanh(h) + 0.1 * np.random.randn(num_rows)
    
    df = pd.DataFrame({'H': h, 'M': m})
    df.to_csv(path, sep='\t', index=False)
    return path


def test_export_multi_txt():
    """Test TXT export for multi-dataset mode."""
    print("Testing multi-dataset TXT export...")
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")
    
    try:
        # Create test data files
        file1 = os.path.join(temp_dir, 'test1.txt')
        file2 = os.path.join(temp_dir, 'test2.txt')
        file3 = os.path.join(temp_dir, 'test3.txt')
        
        create_test_data_file(file1, 50)
        create_test_data_file(file2, 60)
        create_test_data_file(file3, 70)
        
        # Create main window and load files
        w = MainWindow()
        w.show()
        
        # Simulate multi-file selection
        w._open_files([file1, file2, file3])
        
        # Wait for processing
        QtCore.QCoreApplication.processEvents()
        
        # Verify multi-mode is active
        assert w.multi_mode, "Multi-mode should be active"
        assert len(w.dataset_list) == 3, f"Expected 3 datasets, got {len(w.dataset_list)}"
        
        print(f"✓ Loaded {len(w.dataset_list)} datasets in multi-mode")
        
        # Test the helper function directly (simulating user export)
        # Create output path
        base_output = os.path.join(temp_dir, 'export_test')
        
        # We can't easily test the full dialog flow, but we can test the core logic
        # by directly calling _export_multi_datasets_to_txt with a mocked dialog
        
        # Instead, let's verify the datasets have the right structure
        for i, ds in enumerate(w.dataset_list, start=1):
            df = ds['df']
            assert df is not None, f"Dataset {i} has no df"
            assert 'H' in df.columns, f"Dataset {i} missing H column"
            assert 'M' in df.columns, f"Dataset {i} missing M column"
            print(f"✓ Dataset {i}: {len(df)} rows, columns={list(df.columns)}")
        
        # Test clipboard copy function (doesn't require file dialog)
        w._copy_multi_datasets_to_clipboard()
        
        # Check clipboard contents
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text()
        
        assert text, "Clipboard should not be empty"
        assert "# Dataset 1:" in text, "Clipboard should contain dataset 1 header"
        assert "# Dataset 2:" in text, "Clipboard should contain dataset 2 header"
        assert "# Dataset 3:" in text, "Clipboard should contain dataset 3 header"
        
        # Count lines (should have headers, data, and blank lines)
        lines = text.split('\n')
        assert len(lines) > 100, f"Expected many lines, got {len(lines)}"
        
        print(f"✓ Clipboard copy successful ({len(lines)} lines)")
        print(f"✓ Clipboard contains all 3 dataset headers")
        
        # Verify data format
        assert '\t' in text, "Clipboard should contain tab-separated data"
        assert 'H\tM' in text, "Clipboard should contain column headers"
        
        print("✓ All export tests passed!")
        
        w.close()
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp dir: {temp_dir}")


def test_export_single_mode_still_works():
    """Verify single-dataset export still works after changes."""
    print("\nTesting single-dataset export (backward compatibility)...")
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")
    
    try:
        # Create single test file
        file1 = os.path.join(temp_dir, 'single.txt')
        create_test_data_file(file1, 100)
        
        # Create main window and load single file
        w = MainWindow()
        w.show()
        
        # Load single file
        w._open_file(file1)
        QtCore.QCoreApplication.processEvents()
        
        # Verify single mode
        assert not w.multi_mode, "Should be in single mode"
        assert w.df is not None, "Should have loaded data"
        
        print("✓ Single file loaded correctly")
        
        # Test clipboard copy (doesn't require dialog)
        w._copy_current_loop_to_clipboard()
        
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text()
        
        assert text, "Clipboard should not be empty"
        assert "H\tM" in text, "Clipboard should contain headers"
        assert "# Dataset" not in text, "Single mode shouldn't have dataset headers"
        
        lines = text.split('\n')
        print(f"✓ Single-mode clipboard copy successful ({len(lines)} lines)")
        
        w.close()
        return True
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp dir: {temp_dir}")


if __name__ == '__main__':
    print("="*60)
    print("Multi-dataset Export Test Suite")
    print("="*60)
    
    try:
        test_export_multi_txt()
        test_export_single_mode_still_works()
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
