#!/usr/bin/env python3
"""Test multi-dataset X centering (center_x_about_zero and undo)."""
import sys, os, tempfile, shutil
import numpy as np, pandas as pd
from PySide6 import QtWidgets, QtCore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QT_test import MainWindow


def make_file(path, n=40, shift=5.0):
    x = np.linspace(-10, 10, n) + shift  # introduce mean offset
    m = np.tanh(x) + 0.01 * np.random.randn(n)
    pd.DataFrame({'H': x, 'M': m}).to_csv(path, sep='\t', index=False)


def test_center_x_multi():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    d = tempfile.mkdtemp(prefix='centerx_multi_')
    try:
        f1 = os.path.join(d, 'd1.txt'); make_file(f1, shift=3.0)
        f2 = os.path.join(d, 'd2.txt'); make_file(f2, shift=-2.5)
        f3 = os.path.join(d, 'd3.txt'); make_file(f3, shift=1.7)

        w = MainWindow(); w.show()
        w._open_files([f1, f2, f3])
        QtCore.QCoreApplication.processEvents()
        assert w.multi_mode and len(w.dataset_list) == 3

        # Capture original means
        orig_means = [ds['df']['H'].mean() for ds in w.dataset_list]
        assert any(abs(m) > 0.5 for m in orig_means), 'Offsets too small for test'

        # Apply centering
        w.center_x_about_zero()
        QtCore.QCoreApplication.processEvents()

        centered_means = [ds['df']['H'].mean() for ds in w.dataset_list]
        for cm in centered_means:
            assert abs(cm) < 1e-9, f'Centered mean not ~0: {cm}'

        # Undo
        w.undo_last_operation()
        QtCore.QCoreApplication.processEvents()

        undone_means = [ds['df']['H'].mean() for ds in w.dataset_list]
        for om, orig in zip(undone_means, orig_means):
            # Allow tiny numerical drift
            assert abs(om - orig) < 1e-9, f'Undo failed: {om} vs {orig}'

        print('center_x_multi test passed')
        w.close()
    finally:
        shutil.rmtree(d)

if __name__ == '__main__':
    test_center_x_multi()
