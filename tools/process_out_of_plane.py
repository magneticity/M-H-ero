import os
import sys
import numpy as np

# Ensure repo root is on sys.path so we can import QT_test when running from tools/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from QT_test import MainWindow, build_virgin_curve, integrate_HdM
from PySide6 import QtWidgets

# Ensure a QApplication exists for MainWindow
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def naive_virgin_no_smoothing(H, M, n_grid=2000):
    H = np.asarray(H, dtype=float)
    M = np.asarray(M, dtype=float)
    finite = np.isfinite(H) & np.isfinite(M)
    H = H[finite]
    M = M[finite]
    if H.size < 3:
        raise ValueError("Not enough points")
    # simple branch split by sign of derivative
    dh = np.diff(H)
    incr_mask = np.concatenate(([dh[0] >= 0], dh >= 0)) if dh.size else np.ones_like(H, dtype=bool)
    decr_mask = ~incr_mask
    H_incr, M_incr = H[incr_mask], M[incr_mask]
    H_decr, M_decr = H[decr_mask], M[decr_mask]
    if H_incr.size < 2 or H_decr.size < 2:
        order = np.argsort(H)
        Hs, Ms = H[order], M[order]
        mid = len(Hs)//2
        H_incr, M_incr = Hs[:mid], Ms[:mid]
        H_decr, M_decr = Hs[mid:], Ms[mid:]
    si = np.argsort(H_incr)
    sd = np.argsort(H_decr)
    H_incr_s, M_incr_s = H_incr[si], M_incr[si]
    H_decr_s, M_decr_s = H_decr[sd], M_decr[sd]
    H_min = max(np.nanmin(H_incr_s), np.nanmin(H_decr_s))
    H_max = min(np.nanmax(H_incr_s), np.nanmax(H_decr_s))
    if not np.isfinite(H_min) or not np.isfinite(H_max) or H_max <= H_min:
        H_min, H_max = float(np.nanmin(H)), float(np.nanmax(H))
    H_grid = np.linspace(H_min, H_max, n_grid)
    M_incr_grid = np.interp(H_grid, H_incr_s, M_incr_s, left=np.nan, right=np.nan)
    M_decr_grid = np.interp(H_grid, H_decr_s, M_decr_s, left=np.nan, right=np.nan)
    M_vir = np.nanmean(np.vstack([M_incr_grid, M_decr_grid]), axis=0)
    mask_incr_valid = ~np.isnan(M_incr_grid)
    mask_decr_valid = ~np.isnan(M_decr_grid)
    M_vir[mask_incr_valid & ~mask_decr_valid] = M_incr_grid[mask_incr_valid & ~mask_decr_valid]
    M_vir[~mask_incr_valid & mask_decr_valid] = M_decr_grid[~mask_incr_valid & mask_decr_valid]
    valid = np.isfinite(M_vir) & np.isfinite(H_grid)
    H_vir = H_grid[valid]
    M_vir = M_vir[valid]
    return H_vir, M_vir


def process_file(path):
    mw = MainWindow()
    df = mw._read_table_auto(path)
    mw.df = df.copy(deep=True)
    mw.original_df = df.copy(deep=True)
    # Prefer explicit column names if present (user-specified)
    preferred_x = 'Applied_Field'
    preferred_y = 'Signal_perpendicular_to_sample'
    if preferred_x in df.columns and preferred_y in df.columns:
        x_name, y_name = preferred_x, preferred_y
    else:
        # fallback: choose first two numeric columns
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(numeric_cols) < 2:
            raise RuntimeError("Not enough numeric columns and preferred columns not found")
        x_name, y_name = numeric_cols[0], numeric_cols[1]
    x = np.asarray(df[x_name].to_numpy(), dtype=float)
    Hmax = np.nanmax(np.abs(x))
    thr = 0.8 * Hmax if Hmax > 0 else 0.0

    # Apply drift correction using tail method
    try:
        y_corr, info_drift = mw._compute_drift_tails(df, x_name, y_name, thr)
        df_drift = df.copy(deep=True)
        df_drift[y_name] = y_corr
    except Exception as e:
        print('Drift correction failed:', e)
        df_drift = df
        info_drift = None

    # Apply background subtraction
    try:
        y_corr2, info_bg = mw._compute_bg_corrected(df_drift, x_name, y_name, thr)
        df_bg = df_drift.copy(deep=True)
        df_bg[y_name] = y_corr2
    except Exception as e:
        print('BG correction failed:', e)
        df_bg = df_drift
        info_bg = None

    # Build virgin curves using chosen columns
    H = np.asarray(df_bg[x_name].to_numpy(), dtype=float)
    M = np.asarray(df_bg[y_name].to_numpy(), dtype=float)

    H_vir_raw, M_vir_raw = naive_virgin_no_smoothing(H, M, n_grid=2000)
    area_raw = integrate_HdM(H_vir_raw, M_vir_raw, float(np.nanmax(M_vir_raw)))

    H_vir_smooth, M_vir_smooth, Ms = build_virgin_curve(H, M, n_grid=2000)
    area_smooth = integrate_HdM(H_vir_smooth, M_vir_smooth, Ms)

    print('File:', path)
    print('Using columns:', x_name, '-> H,', y_name, '-> M')
    print('Drift info:', info_drift)
    print('BG info:', info_bg)
    print(f'Raw virgin curve points: {len(H_vir_raw)}, area_raw = {area_raw:.6g}')
    print(f'Smoothed virgin curve points: {len(H_vir_smooth)}, area_smooth = {area_smooth:.6g}')


if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(repo_root, 'test_data', 'out_of_plane.VHD')
    process_file(path)
