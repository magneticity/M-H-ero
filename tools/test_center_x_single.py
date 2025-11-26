#!/usr/bin/env python3
"""Test single-dataset X centering and undo."""
import sys, os, tempfile, shutil
import numpy as np, pandas as pd
from PySide6 import QtWidgets, QtCore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QT_test import MainWindow


def make_file(path, n=50, shift=4.2):
    x = np.linspace(-5, 5, n) + shift
    m = np.tanh(x) + 0.01 * np.random.randn(n)
    pd.DataFrame({'H': x, 'M': m}).to_csv(path, sep='\t', index=False)


def test_center_x_single():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    d = tempfile.mkdtemp(prefix='centerx_single_')
    try:
        f = os.path.join(d, 'd.txt'); make_file(f)
        w = MainWindow(); w.show()

        # Load file via helper pathless open (simulate dialog outcome)
        # Use existing open_file logic but we need to patch QFileDialog? simpler: call _read_table_auto then assign.
        df = w._read_table_auto(f)
        w.original_df = df.copy(deep=True)
        w.df = df.copy(deep=True)
        w.numeric_cols = [c for c in w.df.columns if np.issubdtype(w.df[c].dtype, np.number)]
        w._populate_combos()
        w._replot()
        QtCore.QCoreApplication.processEvents()

        orig_mean = w.df['H'].mean()
        assert abs(orig_mean) > 1.0, 'Original mean should be noticeably offset'

        w.center_x_about_zero()
        QtCore.QCoreApplication.processEvents()
        centered_mean = w.df['H'].mean()
        assert abs(centered_mean) < 1e-9, f'Centered mean not ~0: {centered_mean}'

        # Undo
        w.undo_last_operation()
        QtCore.QCoreApplication.processEvents()
        undone_mean = w.df['H'].mean()
        assert abs(undone_mean - orig_mean) < 1e-9, 'Undo did not restore original mean'
        print('center_x_single test passed')
        w.close()
    finally:
        shutil.rmtree(d)

if __name__ == '__main__':
    test_center_x_single()
