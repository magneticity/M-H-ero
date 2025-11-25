import sys
import io
import os
import pandas as pd
import matplotlib
import numpy as np

from PySide6 import QtWidgets, QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setParent(parent)

    def plot_xy(self, x, y, xlabel="X", ylabel="Y", title=""):
        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.5)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()


class CalculationWindow(QtWidgets.QWidget):
    """Window showing two subplots for anisotropy-area calculations.

    Usage: create once and call `plot_result(index, H, M, H_vir, M_vir, area, label)`
    where `index` is 0 or 1 to assign to first or second subplot.
    The window persists between file loads; the first subplot is not cleared
    automatically so users can compute a second sample and compare.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FC
        from matplotlib.figure import Figure

        self.setWindowTitle("Anisotropy energy (area method)")
        # Ensure this widget is a top-level window (so .show() creates a separate window)
        try:
            self.setWindowFlag(QtCore.Qt.Window, True)
        except Exception:
            pass
        self.fig = Figure(tight_layout=True, figsize=(10, 4))
        self.canvas = FigureCanvas(self.fig)
        # Add matplotlib navigation toolbar for zoom/pan/home controls
        try:
            self.toolbar = NavigationToolbar(self.canvas, self)
        except Exception:
            self.toolbar = None
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        layout = QtWidgets.QVBoxLayout(self)
        # toolbar (if available) goes above the canvas
        if getattr(self, 'toolbar', None) is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # store last areas and labels
        self.areas = [None, None]
        self.labels = [None, None]
        # store raw results so we can redraw / swap reliably
        # each entry: dict with keys 'H','M','H_vir','M_vir','area','label'
        self.raw_results = [None, None]

        # Controls: clear buttons, swap, diff label
        btn_bar = QtWidgets.QHBoxLayout()
        self.clear1_btn = QtWidgets.QPushButton("Clear 1")
        self.clear2_btn = QtWidgets.QPushButton("Clear 2")
        self.swap_btn = QtWidgets.QPushButton("Swap")
        self.diff_label = QtWidgets.QLabel("Δ area: —")
        btn_bar.addWidget(self.clear1_btn)
        btn_bar.addWidget(self.clear2_btn)
        btn_bar.addWidget(self.swap_btn)
        # Smoothing controls: enable smoothing, window size, isotonic regression
        self.smooth_chk = QtWidgets.QCheckBox("Smooth virgin curve")
        self.smooth_chk.setChecked(True)
        self.smooth_win_spin = QtWidgets.QSpinBox()
        self.smooth_win_spin.setRange(3, 201)
        self.smooth_win_spin.setSingleStep(2)
        self.smooth_win_spin.setValue(11)
        self.smooth_win_spin.setToolTip("Smoothing window (odd integer)")
        self.iso_chk = QtWidgets.QCheckBox("Isotonic")
        self.iso_chk.setChecked(True)
        btn_bar.addWidget(self.smooth_chk)
        btn_bar.addWidget(QtWidgets.QLabel("Window:"))
        btn_bar.addWidget(self.smooth_win_spin)
        btn_bar.addWidget(self.iso_chk)
        btn_bar.addStretch(1)
        btn_bar.addWidget(self.diff_label)
        layout.addLayout(btn_bar)

        self.clear1_btn.clicked.connect(lambda: self._clear_slot(0))
        self.clear2_btn.clicked.connect(lambda: self._clear_slot(1))
        self.swap_btn.clicked.connect(self._swap_slots)
        self.smooth_chk.toggled.connect(self._on_smoothing_changed)
        self.smooth_win_spin.valueChanged.connect(self._on_smoothing_changed)
        self.iso_chk.toggled.connect(self._on_smoothing_changed)

        # internal smoothing state
        self._smoothing_enabled = True
        self._smoothing_window = 11
        self._isotonic_enabled = True

    def _clear_slot(self, idx):
        ax = self.ax1 if idx == 0 else self.ax2
        ax.clear()
        if idx == 0:
            self.areas[0] = None
            self.labels[0] = None
            self.raw_results[0] = None
        else:
            self.areas[1] = None
            self.labels[1] = None
            self.raw_results[1] = None
        self.diff_label.setText("Δ area: —")
        self.canvas.draw_idle()

    def _swap_slots(self):
        # swap stored results and redraw both slots so axes limits update correctly
        self.areas[0], self.areas[1] = self.areas[1], self.areas[0]
        self.labels[0], self.labels[1] = self.labels[1], self.labels[0]
        self.raw_results[0], self.raw_results[1] = self.raw_results[1], self.raw_results[0]

        # redraw both slots from stored data
        self._redraw_slots()
        self._update_diff_label()
        self.canvas.draw_idle()

    def _on_smoothing_changed(self, _=None):
        # Called when smoothing controls change: redraw and update labels
        try:
            self._smoothing_enabled = self.smooth_chk.isChecked()
            self._smoothing_window = int(self.smooth_win_spin.value())
            self._isotonic_enabled = self.iso_chk.isChecked()
        except Exception:
            pass
        self._redraw_slots()
        self.canvas.draw_idle()

    def _draw_slot(self, index):
        """Draw a single slot from self.raw_results[index]."""
        data = self.raw_results[index]
        ax = self.ax1 if index == 0 else self.ax2
        ax.clear()
        if data is None:
            return

        H = np.asarray(data['H'])
        M = np.asarray(data['M'])
        H_vir = np.asarray(data['H_vir'])
        M_vir = np.asarray(data['M_vir'])
        area = data.get('area', 0.0)
        label = data.get('label', None)

        # Plot faint original loop (H on x, M on y)
        try:
            ax.plot(H, M, linewidth=0.8, color='0.6', alpha=0.5, label='raw loop')
        except Exception:
            pass

        # Prepare displayed virgin curve: optionally smooth and enforce isotonicity
        try:
            M_disp = np.asarray(M_vir, dtype=float).copy()
            H_disp = np.asarray(H_vir, dtype=float).copy()
            if getattr(self, 'smooth_chk', None) is not None and self.smooth_chk.isChecked():
                win = int(self.smooth_win_spin.value())
                # ensure odd
                if win % 2 == 0:
                    win = max(3, win - 1)
                if M_disp.size >= 3 and win >= 3:
                    pad = win // 2
                    M_pad = np.pad(M_disp, (pad, pad), mode='reflect')
                    kernel = np.ones(win) / float(win)
                    M_smooth = np.convolve(M_pad, kernel, mode='valid')
                    M_disp = M_smooth
            if getattr(self, 'iso_chk', None) is not None and self.iso_chk.isChecked():
                # isotonic regression (PAV)
                y = M_disp.astype(float)
                n = y.size
                if n > 0:
                    levels = y.copy()
                    weights = np.ones(n, dtype=float)
                    i = 0
                    while i < levels.size - 1:
                        if levels[i] <= levels[i+1]:
                            i += 1
                            continue
                        total = levels[i] * weights[i] + levels[i+1] * weights[i+1]
                        w = weights[i] + weights[i+1]
                        mean = total / w
                        levels[i] = mean
                        weights[i] = w
                        levels = np.delete(levels, i+1)
                        weights = np.delete(weights, i+1)
                        if i > 0:
                            i -= 1
                    # expand
                    try:
                        M_disp = np.repeat(levels, weights.astype(int))
                    except Exception:
                        M_disp = np.interp(np.arange(n), np.linspace(0, n-1, levels.size), levels)
            M_vir_plot = np.maximum(M_disp, 0.0)
        except Exception:
            M_vir_plot = np.maximum(M_vir, 0.0)

        # Prepare H values for plotting: ensure same length as M_vir_plot and non-negative
        try:
            H_disp = np.asarray(H_disp, dtype=float).copy()
        except Exception:
            H_disp = np.asarray(H_vir, dtype=float).copy()

        # If smoothing/isotonic changed M length, resample H_disp to match M_disp length
        if H_disp.size != M_vir_plot.size and H_disp.size >= 2 and M_vir_plot.size >= 2:
            xp = np.linspace(0.0, 1.0, H_disp.size)
            xnew = np.linspace(0.0, 1.0, M_vir_plot.size)
            try:
                H_disp = np.interp(xnew, xp, H_disp)
            except Exception:
                H_disp = np.resize(H_disp, M_vir_plot.size)

        # Clip H to non-negative for display/integration
        H_disp_plot = np.maximum(H_disp, 0.0)

        # Plot virgin curve (H_disp_plot vs M_vir_plot)
        ax.plot(H_disp_plot, M_vir_plot, linewidth=1.5, color='C1', label='virgin (bisected)')

        # Shade area between H and 0 for M in 0..Ms using horizontal fill
        Ms = float(np.nanmax(M_vir_plot) if M_vir_plot.size else 0.0)
        order = np.argsort(M_vir_plot)
        M_sorted = M_vir_plot[order]
        H_sorted = H_disp_plot[order]
        mask = (M_sorted >= 0) & (M_sorted <= Ms)
        if mask.any():
            ax.fill_betweenx(M_sorted[mask], 0.0, H_sorted[mask], color='C1', alpha=0.2)

        # Axis labels: prefer explicit labels stored with the result, else use quantity+system if available
        x_label_pref = None
        y_label_pref = None
        try:
            x_label_pref = data.get('x_label')
            y_label_pref = data.get('y_label')
        except Exception:
            x_label_pref = None
            y_label_pref = None

        def _compose_label(pref_label, q, sys, default):
            if pref_label:
                return pref_label
            if q and sys:
                return f"{q} ({sys})"
            if q:
                return q
            return default

        xlabel = _compose_label(x_label_pref, data.get('xq'), data.get('xsys'), 'H')
        ylabel = _compose_label(y_label_pref, data.get('yq'), data.get('ysys'), 'M')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # If smoothing/isotonic applied, recompute displayed area
        try:
            Ms = float(np.nanmax(M_vir_plot) if M_vir_plot.size else 0.0)
            order = np.argsort(M_vir_plot)
            M_sorted = M_vir_plot[order]
            H_sorted = H_disp_plot[order]
            mask = (M_sorted >= 0) & (M_sorted <= Ms)
            if mask.sum() >= 2:
                applied_area = float(np.trapz(np.maximum(H_sorted[mask], 0.0), M_sorted[mask]))
            else:
                applied_area = area
        except Exception:
            applied_area = area

        # Area units: prefer stored area_units, else show generic H·M
        area_units_pref = None
        try:
            area_units_pref = data.get('area_units')
        except Exception:
            area_units_pref = None
        area_units_str = f" ({area_units_pref})" if area_units_pref else " (units: H·M)"
        ax.annotate(f'Area = {applied_area:.6g}{area_units_str}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=9, verticalalignment='top')

        # update stored displayed area (without overwriting original raw value)
        try:
            self.areas[index] = applied_area
        except Exception:
            pass

        # Autoscale view so limits update nicely
        ax.relim()
        ax.autoscale_view()

    def _redraw_slots(self):
        """Redraw both slots according to stored raw_results."""
        # draw left
        self._draw_slot(0)
        # draw right
        self._draw_slot(1)
        # Update diff label after redraw so smoothed areas are reflected
        self._update_diff_label()

    def apply_axis_swap(self):
        """Apply an axes swap to stored results: swap H<->M and invert area sign.

        This is called when the main window swaps X/Y so previously computed
        results remain consistent with the new axes orientation.
        """
        for i in (0, 1):
            data = self.raw_results[i]
            if data is None:
                continue
            try:
                # swap arrays
                H = np.asarray(data.get('H', []))
                M = np.asarray(data.get('M', []))
                H_vir = np.asarray(data.get('H_vir', []))
                M_vir = np.asarray(data.get('M_vir', []))

                data['H'], data['M'] = M, H
                data['H_vir'], data['M_vir'] = M_vir, H_vir

                # invert area sign
                if 'area' in data and data['area'] is not None:
                    data['area'] = -float(data['area'])
                    self.areas[i] = -self.areas[i] if self.areas[i] is not None else None
            except Exception:
                # best-effort; skip on error
                continue

        # redraw with swapped axes
        self._redraw_slots()

    def _update_diff_label(self):
        # Show a signed difference: left (slot 1) minus right (slot 2).
        # This ensures swapping the two slots will flip the sign of Δ area.
        if self.areas[0] is None or self.areas[1] is None:
            self.diff_label.setText("Δ area: —")
            return

        try:
            diff = float(self.areas[0]) - float(self.areas[1])
        except Exception:
            self.diff_label.setText("Δ area: —")
            return

        # Try to compute a physical energy (K_eff or E_a) when unit metadata is present
        meta0 = self.raw_results[0] or {}
        meta1 = self.raw_results[1] or {}
        xq0 = meta0.get('xq')
        xsys0 = meta0.get('xsys')
        yq0 = meta0.get('yq')
        ysys0 = meta0.get('ysys')

        xq1 = meta1.get('xq')
        xsys1 = meta1.get('xsys')
        yq1 = meta1.get('yq')
        ysys1 = meta1.get('ysys')

        # If metadata is missing or inconsistent between slots, show only area difference
        meta_ok = all([xq0 == xq1, yq0 == yq1, xsys0 == xsys1, ysys0 == ysys1]) and (xq0 is not None and yq0 is not None)

        out_lines = []
        out_lines.append(f"Δ area: {diff:.6g}")

        if not meta_ok:
            out_lines.append("(units: incompatible or unspecified)")
            self.diff_label.setText("\n".join(out_lines))
            return

        # Units handling
        try:
            import math
            mu0 = 4.0 * math.pi * 1e-7  # vacuum permeability H/m
        except Exception:
            mu0 = 4.0 * 3.141592653589793 * 1e-7

        # cgs (emu) case: areas are already in erg/cm^3 (for H·M) or erg (for H·m)
        if xsys0 == 'cgs-emu/Gaussian':
            if xq0 == 'H' and yq0 == 'M':
                # energy density K_eff (erg/cm^3)
                K = diff
                out_lines.append(f"K_eff = {K:.6g} erg/cm³")
            elif yq0 == 'm':
                # energy E_a (erg)
                E = diff
                out_lines.append(f"E_a = {E:.6g} erg")
            else:
                out_lines.append("(units: cgs, unknown quantity pairing)")
        elif xsys0 == 'SI':
            # SI: multiply area by mu0 to get energy (J/m^3 for density, J for energy)
            energy = mu0 * diff
            if xq0 == 'H' and yq0 == 'M':
                out_lines.append(f"K_eff = {energy:.6g} J/m³")
            elif yq0 == 'm':
                out_lines.append(f"E_a = {energy:.6g} J")
            else:
                out_lines.append("(units: SI, unknown quantity pairing)")
        elif xsys0 == 'Heaviside-Lorentz':
            # Heaviside-Lorentz is the rationalized Gaussian system.
            # Using the relations H_HL = H_G / sqrt(4π), M_HL = M_G * sqrt(4π)
            # gives the same numeric ∫ H dM as Gaussian, so `diff` is
            # directly comparable to the cgs result (erg/cm³ or erg).
            if xq0 == 'H' and yq0 == 'M':
                # Report both cgs (erg/cm^3) and converted SI (J/m^3)
                K_cgs = diff
                K_si = 0.1 * diff  # 1 erg/cm^3 = 0.1 J/m^3
                out_lines.append(f"K_eff = {K_cgs:.6g} erg/cm³ ≈ {K_si:.6g} J/m³ (Heaviside–Lorentz)")
            elif yq0 == 'm':
                E_cgs = diff
                E_si = 0.1 * diff
                out_lines.append(f"E_a = {E_cgs:.6g} erg ≈ {E_si:.6g} J (Heaviside–Lorentz)")
            else:
                out_lines.append("(units: HL, unknown quantity pairing)")
        else:
            out_lines.append("(units: unspecified)")

        self.diff_label.setText("\n".join(out_lines))

    def plot_result(self, index, H, M, H_vir, M_vir, area, label,
                    x_label=None, y_label=None, area_units=None,
                    xq=None, xsys=None, yq=None, ysys=None):
        """Plot results into subplot `index` (0 or 1), with optional axis and area units labels."""
        if index not in (0, 1):
            index = 0

        ax_target = self.ax1 if index == 0 else self.ax2

        try:
            # Store axis/units metadata if passed in kwargs (fall back to None)
            self.raw_results[index] = {
                'H': np.asarray(H),
                'M': np.asarray(M),
                'H_vir': np.asarray(H_vir),
                'M_vir': np.asarray(M_vir),
                'area': float(area),
                'label': label,
                'xq': xq,
                'xsys': xsys,
                'yq': yq,
                'ysys': ysys,
            }
        except Exception:
            self.raw_results[index] = None

        # Clear target axis; store raw result and delegate actual drawing to _redraw_slots
        ax_target.clear()

        # store metadata/arrays (raw, unmodified)
        try:
            self.raw_results[index] = {
                'H': np.asarray(H),
                'M': np.asarray(M),
                'H_vir': np.asarray(H_vir),
                'M_vir': np.asarray(M_vir),
                'area': float(area),
                'label': label,
                'x_label': x_label,
                'y_label': y_label,
                'area_units': area_units,
                'xq': xq,
                'xsys': xsys,
                'yq': yq,
                'ysys': ysys,
            }
        except Exception:
            self.raw_results[index] = None

        # Keep label for UI state
        self.labels[index] = label

        # If plotting into left slot, clear the right slot (previous behaviour)
        if index == 0:
            self.ax2.clear()
            self.areas[1] = None
            self.labels[1] = None
            self.raw_results[1] = None

        # Delegate drawing to redraw logic so smoothing/isotonic settings are respected
        self._redraw_slots()
        self.canvas.draw_idle()


def _smooth_sign(x, window=5):
    """Return a smoothed sign of the derivative for branch detection.

    Small helper used by branch detection to reduce noise-induced flips.
    """
    if len(x) < 2:
        return np.array([1])
    d = np.diff(x)
    # simple moving average on derivative magnitude
    w = np.ones(window) / float(window)
    dpad = np.pad(d, (window//2, window-1-window//2), mode='edge')
    ds = np.convolve(dpad, w, mode='valid')
    s = np.sign(ds)
    # replace zeros with previous non-zero or 1
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = s[i-1] if i > 0 else 1
    return s


def build_virgin_curve(H_in, M_in, n_grid=2000):
    """Construct a virgin curve by bisecting increasing and decreasing branches.

    Returns (H_vir, M_vir, Ms_val).
    """
    H = np.asarray(H_in, dtype=float)
    M = np.asarray(M_in, dtype=float)
    finite = np.isfinite(H) & np.isfinite(M)
    H = H[finite]
    M = M[finite]

    if H.size < 3:
        raise ValueError("Not enough points to build virgin curve")

    # Detect monotonic segments using smoothed derivative sign
    s = _smooth_sign(H, window=7)
    # Expand to length of H by repeating last sign
    if s.size < H.size:
        s = np.pad(s, (0, H.size - s.size), mode='edge')

    incr_mask = s >= 0
    decr_mask = ~incr_mask

    H_incr = H[incr_mask]
    M_incr = M[incr_mask]
    H_decr = H[decr_mask]
    M_decr = M[decr_mask]

    # Require at least 2 points in each; fallback to split-by-H if not
    if H_incr.size < 2 or H_decr.size < 2:
        order = np.argsort(H)
        Hs = H[order]
        Ms = M[order]
        mid = len(Hs) // 2
        H_incr, M_incr = Hs[:mid], Ms[:mid]
        H_decr, M_decr = Hs[mid:], Ms[mid:]

    # Sort branch arrays
    si = np.argsort(H_incr)
    sd = np.argsort(H_decr)
    H_incr_s, M_incr_s = H_incr[si], M_incr[si]
    H_decr_s, M_decr_s = H_decr[sd], M_decr[sd]

    # define grid over overlapping H region (if possible)
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
    if valid.sum() < 2:
        raise ValueError("Virgin curve construction failed")

    H_vir = H_grid[valid]
    M_vir = M_vir[valid]
    # Smooth and enforce monotonicity on the virgin curve at high field to
    # avoid zig-zags that make the area unstable. Strategy:
    # 1) Ensure an explicit (0,0) anchor exists (so integration starts at zero)
    # 2) Optionally smooth M_vir with a small moving-average filter
    # 3) Apply isotonic regression (PAV) to enforce non-decreasing M with H
    try:
        # Ensure anchor at (0,0)
        idx0 = np.where(np.isclose(H_vir, 0.0, atol=1e-9))[0]
        if idx0.size > 0:
            M_vir[idx0[0]] = max(0.0, float(M_vir[idx0[0]]))
        else:
            H_vir = np.concatenate(([0.0], H_vir))
            M_vir = np.concatenate(([0.0], M_vir))

        # Keep only non-negative quadrant (we integrate M>=0, H>=0)
        pos_mask = (H_vir >= 0.0) & (M_vir >= 0.0)
        H_vir = H_vir[pos_mask]
        M_vir = M_vir[pos_mask]

        # If too few points after filtering, fail early
        if H_vir.size < 2:
            raise ValueError("Virgin curve contains insufficient positive-quadrant points")

        # Smooth M_vir with a small moving average to suppress local zig-zags.
        # Window should be odd and relatively small compared to n_grid.
        win = 11
        if M_vir.size >= 3:
            win = min(win, M_vir.size if M_vir.size % 2 == 1 else M_vir.size - 1)
            if win >= 3:
                pad = win // 2
                M_pad = np.pad(M_vir, (pad, pad), mode='reflect')
                kernel = np.ones(win) / float(win)
                M_smooth = np.convolve(M_pad, kernel, mode='valid')
                M_vir = M_smooth

        # Isotonic regression (PAV) to enforce non-decreasing M vs H
        def _isotonic_regression(y):
            y = np.asarray(y, dtype=float)
            n = y.size
            if n == 0:
                return y
            # Initialize levels and weights
            levels = y.copy()
            weights = np.ones(n, dtype=float)
            i = 0
            while i < levels.size - 1:
                if levels[i] <= levels[i+1]:
                    i += 1
                    continue
                # merge blocks i and i+1
                total = levels[i] * weights[i] + levels[i+1] * weights[i+1]
                w = weights[i] + weights[i+1]
                mean = total / w
                levels[i] = mean
                weights[i] = w
                # remove i+1
                levels = np.delete(levels, i+1)
                weights = np.delete(weights, i+1)
                if i > 0:
                    i -= 1
            # expand levels back to original length
            out = np.repeat(levels, weights.astype(int))
            # if rounding removed/changed length, fallback to repeating means
            if out.size != n:
                # evenly interpolate back to original length
                out = np.interp(np.arange(n), np.linspace(0, n-1, out.size), out)
            return out

        try:
            M_iso = _isotonic_regression(M_vir)
            # Numerical cleanup: ensure non-negative and same length
            M_vir = np.maximum(M_iso, 0.0)
        except Exception:
            # If isotonic fails, keep smoothed M_vir
            M_vir = np.maximum(M_vir, 0.0)

    except Exception:
        # On any error, fall back to original arrays (caller handles failures)
        pass

    # Require at least two points to form a valid virgin curve in the positive quadrant
    if H_vir.size < 2 or M_vir.size < 2:
        raise ValueError("Virgin curve contains insufficient points in positive H/M quadrant")

    Ms_val = float(np.nanmax(M_vir))
    return H_vir, M_vir, Ms_val


def integrate_HdM(H_vir, M_vir, Ms_val):
    """Integrate area = ∫ H dM from M=0 to M=Ms_val using trapezoidal rule.

    Assumes H_vir, M_vir are arrays with finite values.
    """
    order = np.argsort(M_vir)
    M_sorted = M_vir[order]
    H_sorted = H_vir[order]
    mask = (M_sorted >= 0.0) & (M_sorted <= Ms_val)
    if mask.sum() < 2:
        raise ValueError("Not enough points in 0..Ms for integration")
    M_int = M_sorted[mask]
    H_int = H_sorted[mask]
    # Ensure integration uses field magnitude from 0 to positive saturation
    # i.e. ignore negative H contributions and integrate H >= 0 only.
    H_int_pos = np.maximum(H_int, 0.0)
    return float(np.trapz(H_int_pos, M_int))



class UnitHelpDialog(QtWidgets.QDialog):
    """Small help dialog showing unit conversion guidance and HL explanation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unit Systems Help")
        self.setMinimumWidth(500)

        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("Unit Systems & Conversions")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        title.setFont(title_font)
        layout.addWidget(title)

        # Help text
        help_text = QtWidgets.QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMarkdown(
            "## Supported Unit Systems\n\n"
            "### SI (International System)\n"
            "- Field **H**: A/m\n"
            "- Magnetisation **M**: A/m\n"
            "- Moment **m**: A·m²\n"
            "- Flux density **B**: T (Tesla)\n\n"
            "### cgs-emu/Gaussian\n"
            "- Field **H**: Oe (Oersted)\n"
            "- Magnetisation **M**: emu/cm³\n"
            "- Moment **m**: emu\n"
            "- Flux density **B**: G (Gauss)\n\n"
            "### Heaviside–Lorentz (HL)\n"
            "Rationalized Gaussian system. Same integral values as Gaussian but with scaled field/magnetisation:\n"
            "- **H** and **B** scaled by 1/√(4π) relative to Gaussian\n"
            "- **M** and **m** scaled by √(4π) relative to Gaussian\n"
            "- Use HL when working with rationalized CGS data.\n\n"
            "## Quick Conversions\n"
            "- 1 erg/cm³ = 0.1 J/m³\n"
            "- 1 emu = 10⁻³ A·m²\n"
            "- 1 Oe ≈ 79.577 A/m\n"
        )
        layout.addWidget(help_text)

        # Close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class SetAxesDialog(QtWidgets.QDialog):
    """
    Dialog to manually set what the X and Y axes represent and their unit systems,
    without modifying the data or history.
    """
    def __init__(self, parent=None,
                 current_x_type=None, current_x_system=None,
                 current_y_type=None, current_y_system=None):
        super().__init__(parent)
        self.setWindowTitle("Set axes")

        layout = QtWidgets.QVBoxLayout(self)

        # --- X axis group ---
        groupX = QtWidgets.QGroupBox("X axis quantity")
        x_layout = QtWidgets.QVBoxLayout(groupX)

        self.radioXUnknown = QtWidgets.QRadioButton("Unknown / ambiguous")
        self.radioX_H = QtWidgets.QRadioButton("Field H")
        self.radioX_B = QtWidgets.QRadioButton("Flux density B")

        x_layout.addWidget(self.radioXUnknown)
        x_layout.addWidget(self.radioX_H)
        x_layout.addWidget(self.radioX_B)
        layout.addWidget(groupX)

        # Preselect X type
        if current_x_type == "H":
            self.radioX_H.setChecked(True)
        elif current_x_type == "B":
            self.radioX_B.setChecked(True)
        else:
            self.radioXUnknown.setChecked(True)

        # X unit system
        formX = QtWidgets.QFormLayout()
        self.xSystemCombo = QtWidgets.QComboBox()
        self.xSystemCombo.addItem("Unspecified")
        self.xSystemCombo.addItem("SI")
        self.xSystemCombo.addItem("cgs-emu/Gaussian")
        self.xSystemCombo.addItem("Heaviside-Lorentz")

        if current_x_system in ("SI", "cgs-emu/Gaussian", "Heaviside-Lorentz"):
            idx = self.xSystemCombo.findText(current_x_system)
            if idx >= 0:
                self.xSystemCombo.setCurrentIndex(idx)
        else:
            self.xSystemCombo.setCurrentIndex(0)

        formX.addRow("Unit system for X:", self.xSystemCombo)
        layout.addLayout(formX)

        # --- Y axis group ---
        groupY = QtWidgets.QGroupBox("Y axis quantity")
        y_layout = QtWidgets.QVBoxLayout(groupY)

        self.radioYUnknown = QtWidgets.QRadioButton("Unknown / ambiguous")
        self.radioY_M = QtWidgets.QRadioButton("Magnetisation M")
        self.radioY_m = QtWidgets.QRadioButton("Magnetic moment m")

        y_layout.addWidget(self.radioYUnknown)
        y_layout.addWidget(self.radioY_M)
        y_layout.addWidget(self.radioY_m)
        layout.addWidget(groupY)

        # Preselect Y type
        if current_y_type == "M":
            self.radioY_M.setChecked(True)
        elif current_y_type == "m":
            self.radioY_m.setChecked(True)
        else:
            self.radioYUnknown.setChecked(True)

        # Y unit system
        formY = QtWidgets.QFormLayout()
        self.ySystemCombo = QtWidgets.QComboBox()
        self.ySystemCombo.addItem("Unspecified")
        self.ySystemCombo.addItem("SI")
        self.ySystemCombo.addItem("cgs-emu/Gaussian")
        self.ySystemCombo.addItem("Heaviside-Lorentz")

        if current_y_system in ("SI", "cgs-emu/Gaussian", "Heaviside-Lorentz"):
            idx = self.ySystemCombo.findText(current_y_system)
            if idx >= 0:
                self.ySystemCombo.setCurrentIndex(idx)
        else:
            self.ySystemCombo.setCurrentIndex(0)

        formY.addRow("Unit system for Y:", self.ySystemCombo)
        layout.addLayout(formY)

        # Now wire up the tooltip callbacks (after both combos exist)
        hl_tooltip_short = "Choose unit system (SI, cgs-emu/Gaussian, Heaviside-Lorentz)"
        hl_tooltip_long = (
            "Heaviside–Lorentz (HL) is the rationalized form of Gaussian units.\n"
            "Conversions use SI ↔ Gaussian factors then apply HL rationalization: "
            "H,B scaled by 1/√(4π); M,m scaled by √(4π)."
        )

        def _update_set_axes_tooltips(_=None):
            if self.xSystemCombo.currentText() == "Heaviside-Lorentz" or self.ySystemCombo.currentText() == "Heaviside-Lorentz":
                self.xSystemCombo.setToolTip(hl_tooltip_long)
                self.ySystemCombo.setToolTip(hl_tooltip_long)
            else:
                self.xSystemCombo.setToolTip(hl_tooltip_short)
                self.ySystemCombo.setToolTip(hl_tooltip_short)

        self.xSystemCombo.currentTextChanged.connect(_update_set_axes_tooltips)
        self.ySystemCombo.currentTextChanged.connect(_update_set_axes_tooltips)
        _update_set_axes_tooltips()

        # Info
        info = QtWidgets.QLabel(
            "This changes how axis labels and loop parameters are labelled, "
            "but does not modify the data. Use 'Convert units' to actually "
            "change numerical units."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # OK / Cancel + Help button (bottom row)
        btn_layout = QtWidgets.QHBoxLayout()
        help_btn = QtWidgets.QPushButton("? Help")
        help_btn.setMaximumWidth(60)
        help_btn.clicked.connect(lambda: UnitHelpDialog(self).exec())
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch(1)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

    def get_selection(self):
        """
        Returns:
          x_type: None, 'H', or 'B'
          x_system: None, 'SI', or 'cgs-emu/Gaussian'
          y_type: None, 'M', or 'm'
          y_system: None, 'SI', or 'cgs-emu/Gaussian'
        """
        # X type
        if self.radioX_H.isChecked():
            x_type = "H"
        elif self.radioX_B.isChecked():
            x_type = "B"
        else:
            x_type = None

        # X system
        xs = self.xSystemCombo.currentText()
        if xs == "SI":
            x_system = "SI"
        elif xs == "cgs-emu/Gaussian":
            x_system = "cgs-emu/Gaussian"
        elif xs == "Heaviside-Lorentz":
            x_system = "Heaviside-Lorentz"
        else:
            x_system = None

        # Y type
        if self.radioY_M.isChecked():
            y_type = "M"
        elif self.radioY_m.isChecked():
            y_type = "m"
        else:
            y_type = None

        # Y system
        ys = self.ySystemCombo.currentText()
        if ys == "SI":
            y_system = "SI"
        elif ys == "cgs-emu/Gaussian":
            y_system = "cgs-emu/Gaussian"
        elif ys == "Heaviside-Lorentz":
            y_system = "Heaviside-Lorentz"
        else:
            y_system = None

        return x_type, x_system, y_type, y_system

class SetYQuantityDialog(QtWidgets.QDialog):
    """
    Dialog to manually set what the Y axis represents and its unit system,
    without modifying the data or history.
    """
    def __init__(self, parent=None, current_type=None, current_system=None):
        super().__init__(parent)
        self.setWindowTitle("Set Y quantity")

        layout = QtWidgets.QVBoxLayout(self)

        # Y type group
        groupY = QtWidgets.QGroupBox("Y axis quantity")
        y_layout = QtWidgets.QVBoxLayout(groupY)

        self.radioUnknown = QtWidgets.QRadioButton("Unknown / ambiguous")
        self.radioM = QtWidgets.QRadioButton("Magnetisation M")
        self.radio_m = QtWidgets.QRadioButton("Magnetic moment m")

        y_layout.addWidget(self.radioUnknown)
        y_layout.addWidget(self.radioM)
        y_layout.addWidget(self.radio_m)
        layout.addWidget(groupY)

        # Preselect based on current_type
        if current_type == "M":
            self.radioM.setChecked(True)
        elif current_type == "m":
            self.radio_m.setChecked(True)
        else:
            self.radioUnknown.setChecked(True)

        # Unit system (for Y)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.systemCombo = QtWidgets.QComboBox()
        self.systemCombo.addItem("Unspecified")
        self.systemCombo.addItem("SI")
        self.systemCombo.addItem("cgs-emu/Gaussian")

        if current_system in ("SI", "cgs-emu/Gaussian"):
            idx = self.systemCombo.findText(current_system)
            if idx >= 0:
                self.systemCombo.setCurrentIndex(idx)
        else:
            self.systemCombo.setCurrentIndex(0)  # Unspecified

        form.addRow("Unit system for Y:", self.systemCombo)

        # Info label
        self.infoLabel = QtWidgets.QLabel(
            "This changes how loop parameters and markers are labelled, "
            "but does not modify the data."
        )
        self.infoLabel.setWordWrap(True)
        layout.addWidget(self.infoLabel)

        # OK / Cancel
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self):
        """
        Returns:
          y_type: None, 'M' or 'm'
          y_system: None, 'SI' or 'cgs-emu/Gaussian'
        """
        if self.radioM.isChecked():
            y_type = "M"
        elif self.radio_m.isChecked():
            y_type = "m"
        else:
            y_type = None

        sys_text = self.systemCombo.currentText()
        if sys_text == "SI":
            y_system = "SI"
        elif sys_text == "cgs-emu/Gaussian":
            y_system = "cgs-emu/Gaussian"
        else:
            y_system = None

        return y_type, y_system

class VolumeNormalisationDialog(QtWidgets.QDialog):
    """
    Dialog to convert the Y axis between magnetisation M and moment m
    using a specified sample volume.

    Supports:
      - SI:   M in A/m,        m in A·m^2,  V in m^3
      - cgs:  M in emu/cm^3,   m in emu,    V in cm^3
    """
    def __init__(self, parent=None, known_y_type=None, known_y_system=None):
        super().__init__(parent)
        self.setWindowTitle("Volume normalisation")

        layout = QtWidgets.QVBoxLayout(self)

        # Current Y quantity
        groupY = QtWidgets.QGroupBox("Current Y axis quantity")
        y_layout = QtWidgets.QVBoxLayout(groupY)
        self.radioY_M = QtWidgets.QRadioButton("Magnetisation M")
        self.radioY_m = QtWidgets.QRadioButton("Magnetic moment m")
        self.radioY_M.setChecked(True)
        y_layout.addWidget(self.radioY_M)
        y_layout.addWidget(self.radioY_m)
        layout.addWidget(groupY)

        # Unit system for Y
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.systemCombo = QtWidgets.QComboBox()
        self.systemCombo.addItems(["SI", "cgs-emu/Gaussian"])
        form.addRow("Unit system for Y:", self.systemCombo)

        # Volume input: use a QLineEdit with a regex validator accepting
        # decimal and scientific notation (e.g. 3e-6), since QDoubleSpinBox
        # may not accept exponent entry on all platforms/locales.
        self.volSpin = QtWidgets.QLineEdit()
        self.volSpin.setPlaceholderText("1.0")
        self.volSpin.setText("1.0")

        # Validator for floats with optional exponent
        regex = r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$'
        try:
            validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(regex))
            self.volSpin.setValidator(validator)
        except Exception:
            # Fall back: no validator if not available
            pass

        self.volUnitsCombo = QtWidgets.QComboBox()
        self.volUnitsCombo.addItems(["m^3", "cm^3"])

        vol_hbox = QtWidgets.QHBoxLayout()
        vol_hbox.addWidget(self.volSpin)
        vol_hbox.addWidget(self.volUnitsCombo)

        form.addRow("Sample volume:", vol_hbox)

        # Info label: describes what will happen
        self.infoLabel = QtWidgets.QLabel()
        layout.addWidget(self.infoLabel)

        # React to radio changes to update info
        self.radioY_M.toggled.connect(self._update_info_label)
        self.radioY_m.toggled.connect(self._update_info_label)

        # Apply known Y info, if any
        if known_y_type is not None:
            if known_y_type == "M":
                self.radioY_M.setChecked(True)
            elif known_y_type == "m":
                self.radioY_m.setChecked(True)
            # Lock the choice
            self.radioY_M.setEnabled(False)
            self.radioY_m.setEnabled(False)

        if known_y_system is not None:
            idx = self.systemCombo.findText(known_y_system)
            if idx >= 0:
                self.systemCombo.setCurrentIndex(idx)
            self.systemCombo.setEnabled(False)

        # Now that radios may have changed, set info label text
        self._update_info_label()

        # OK / Cancel
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_info_label(self):
        if self.radioY_M.isChecked():
            text = "Conversion: M → m  (multiply by volume V)"
        else:
            text = "Conversion: m → M  (divide by volume V)"
        self.infoLabel.setText(text)

    def get_selection(self):
        """
        Returns:
          current_quantity: 'M' or 'm'
          system: 'SI' or 'cgs-emu/Gaussian'
          volume_value: float
          volume_units: 'm^3' or 'cm^3'
        """
        current_quantity = "M" if self.radioY_M.isChecked() else "m"
        system = self.systemCombo.currentText()
        # Parse volume value from the line edit; accept scientific notation
        try:
            volume_value = float(self.volSpin.text())
        except Exception:
            volume_value = 1.0
        volume_units = self.volUnitsCombo.currentText()
        return current_quantity, system, volume_value, volume_units


    def _update_info_label(self):
        if self.radioY_M.isChecked():
            text = "Conversion: M → m  (multiply by volume V)"
        else:
            text = "Conversion: m → M  (divide by volume V)"
        self.infoLabel.setText(text)

    def get_selection(self):
        """
        Returns:
          current_quantity: 'M' or 'm'
          system: 'SI' or 'cgs-emu/Gaussian'
          volume_value: float
          volume_units: 'm^3' or 'cm^3'
        """
        current_quantity = "M" if self.radioY_M.isChecked() else "m"
        system = self.systemCombo.currentText()
        try:
            volume_value = float(self.volSpin.text())
        except Exception:
            volume_value = 1.0
        volume_units = self.volUnitsCombo.currentText()
        return current_quantity, system, volume_value, volume_units

class UnitConversionDialog(QtWidgets.QDialog):
    """
    Dialog to choose unit systems and what each axis represents.

    Assumes:
      - X axis = field-like (H or B)
      - Y axis = magnetisation-like (M or m)
    """
    def __init__(self, parent=None, known_y_type=None, known_y_system=None):
        super().__init__(parent)
        self.setWindowTitle("Convert units")

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        # System choices
        self.fromCombo = QtWidgets.QComboBox()
        self.toCombo = QtWidgets.QComboBox()
        systems = ["SI", "cgs-emu/Gaussian", "Heaviside-Lorentz"]
        self.fromCombo.addItems(systems)
        self.toCombo.addItems(systems)

        form.addRow("Current system:", self.fromCombo)
        form.addRow("Target system:", self.toCombo)

        # Context-sensitive tooltip: show detailed HL explanation only when
        # one of the system combos is set to Heaviside-Lorentz.
        hl_tooltip = (
            "Heaviside–Lorentz (HL) is the rationalized form of Gaussian units.\n"
            "Conversions use SI ↔ Gaussian factors then apply HL rationalization:\n"
            "H,B scaled by 1/√(4π); M,m scaled by √(4π)."
        )

        def _update_tooltips(_=None):
            if self.fromCombo.currentText() == "Heaviside-Lorentz" or self.toCombo.currentText() == "Heaviside-Lorentz":
                self.fromCombo.setToolTip(hl_tooltip)
                self.toCombo.setToolTip(hl_tooltip)
            else:
                # keep a short, generic tooltip otherwise
                short = "Choose unit system (SI, cgs-emu/Gaussian, Heaviside-Lorentz)"
                self.fromCombo.setToolTip(short)
                self.toCombo.setToolTip(short)

        self.fromCombo.currentTextChanged.connect(_update_tooltips)
        self.toCombo.currentTextChanged.connect(_update_tooltips)
        _update_tooltips()

        # Axis meanings
        axis_form = QtWidgets.QFormLayout()
        layout.addLayout(axis_form)

        self.xTypeCombo = QtWidgets.QComboBox()
        self.xTypeCombo.addItems(["Field H", "Flux density B"])

        self.yTypeCombo = QtWidgets.QComboBox()
        self.yTypeCombo.addItems(["Magnetisation M", "Magnetic moment m"])

        axis_form.addRow("X axis represents:", self.xTypeCombo)
        axis_form.addRow("Y axis represents:", self.yTypeCombo)

        # What to convert
        self.chkConvertField = QtWidgets.QCheckBox("Convert X axis")
        self.chkConvertMag = QtWidgets.QCheckBox("Convert Y axis")
        self.chkConvertField.setChecked(True)
        self.chkConvertMag.setChecked(True)

        layout.addWidget(self.chkConvertField)
        layout.addWidget(self.chkConvertMag)

        # If we already know Y type / system, preselect + lock them
        if known_y_system is not None:
            idx = self.fromCombo.findText(known_y_system)
            if idx >= 0:
                self.fromCombo.setCurrentIndex(idx)
            self.fromCombo.setEnabled(False)   # current system is fixed

        if known_y_type is not None:
            if known_y_type == "M":
                self.yTypeCombo.setCurrentText("Magnetisation M")
            elif known_y_type == "m":
                self.yTypeCombo.setCurrentText("Magnetic moment m")
            self.yTypeCombo.setEnabled(False)  # Y meaning is fixed

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self):
        """
        Return:
          (from_system, to_system,
           convert_x, x_type_str,
           convert_y, y_type_str)
        """
        return (
            self.fromCombo.currentText(),
            self.toCombo.currentText(),
            self.chkConvertField.isChecked(),
            self.xTypeCombo.currentText(),
            self.chkConvertMag.isChecked(),
            self.yTypeCombo.currentText(),
        )

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("M(H)ero")
        self.resize(1000, 640)

        # --- central layout (controls + plot) ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Controls row
        controls = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(controls)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self.xCombo = QtWidgets.QComboBox()
        self.yCombo = QtWidgets.QComboBox()
        self.swapBtn = QtWidgets.QToolButton()
        self.clearAxesBtn = QtWidgets.QToolButton()
        self.swapBtn.setText("⇄ Swap")
        self.clearAxesBtn.setText("Clear units")
        self.autoRescaleChk = QtWidgets.QCheckBox("Auto-rescale")
        self.autoRescaleChk.setChecked(True)

        self.bgApplyBtn = QtWidgets.QPushButton("Apply BG")
        self.bgCancelBtn = QtWidgets.QPushButton("Cancel BG")
        self.bgApplyBtn.setVisible(False)
        self.bgCancelBtn.setVisible(False)

        h.addWidget(self.bgApplyBtn)
        h.addWidget(self.bgCancelBtn)

        self.bgApplyBtn.clicked.connect(self._bg_commit)
        self.bgCancelBtn.clicked.connect(self._bg_cancel)

        # Track physical quantity types and unit systems for axes
        self.y_quantity_type = None      # None = unknown/ambiguous, 'M' = magnetisation, 'm' = moment
        self.y_unit_system = None        # 'SI' or 'cgs-emu/Gaussian' or None
        self.x_quantity_type = None      # 'H', 'B', or None
        self.x_unit_system = None        # 'SI', 'cgs-emu/Gaussian', or None

        # Background-mode state
        self.bg_mode_active = False
        self._bg_df_before = None   # df snapshot before starting BG mode
        self._bg_x_col = None
        self._bg_y_col = None
        self._bg_threshold = None
        self._bg_vline = None
        self._bg_cid_press = None
        self._bg_cid_motion = None
        self._bg_cid_release = None

        # Drift tails interactive mode state
        self.drift_mode_active = False
        self._drift_df_before = None
        self._drift_x_col = None
        self._drift_y_col = None
        self._drift_threshold = None
        self._drift_vline = None
        self._drift_cid_press = None
        self._drift_cid_motion = None
        self._drift_cid_release = None
        self._drift_dragging = False

        h.addWidget(QtWidgets.QLabel("X:"))
        h.addWidget(self.xCombo, 1)
        h.addWidget(QtWidgets.QLabel("Y:"))
        h.addWidget(self.yCombo, 1)
        h.addWidget(self.swapBtn)
        h.addWidget(self.clearAxesBtn)
        h.addStretch(1)
        h.addWidget(self.autoRescaleChk)

        self.chkShowMarkers = QtWidgets.QCheckBox("Show markers")
        self.chkShowMarkers.setChecked(False)   # default: OFF
        h.addWidget(self.chkShowMarkers)
        self.chkShowMarkers.toggled.connect(self._on_markers_toggled)

        vbox.addWidget(controls)

        # Plot area
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas, 1)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # Physical parameters dock
        self.paramDock = QtWidgets.QDockWidget("Loop Parameters", self)
        self.paramDock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)

        self.lblHc = QtWidgets.QLabel("—")
        self.lblHcPlus = QtWidgets.QLabel("—")
        self.lblHcMinus = QtWidgets.QLabel("—")

        self.lblMr = QtWidgets.QLabel("—")
        self.lblMrPlus = QtWidgets.QLabel("—")
        self.lblMrMinus = QtWidgets.QLabel("—")

        self.lblHs = QtWidgets.QLabel("—")
        self.lblHsPlus = QtWidgets.QLabel("—")
        self.lblHsMinus = QtWidgets.QLabel("—")

        self.lblMs = QtWidgets.QLabel("—")
        self.lblBgSlope = QtWidgets.QLabel("—")

        form.addRow("Hc (avg):", self.lblHc)
        form.addRow("Hc+:", self.lblHcPlus)
        form.addRow("Hc−:", self.lblHcMinus)

        form.addRow("Hs (mag):", self.lblHs)
        form.addRow("Hs+:", self.lblHsPlus)
        form.addRow("Hs−:", self.lblHsMinus)

        # Value labels (numbers) - for M/m so we can change them dynamically
        self.lblMr = QtWidgets.QLabel("—")
        self.lblMrPlus = QtWidgets.QLabel("—")
        self.lblMrMinus = QtWidgets.QLabel("—")
        self.lblMs = QtWidgets.QLabel("—")

        # Caption labels (left side) - for M/m so we can change them dynamically
        self.lblMrCaption = QtWidgets.QLabel()
        self.lblMrPlusCaption = QtWidgets.QLabel()
        self.lblMrMinusCaption = QtWidgets.QLabel()
        self.lblMsCaption = QtWidgets.QLabel()

        form.addRow(self.lblMrCaption, self.lblMr)
        form.addRow(self.lblMrPlusCaption, self.lblMrPlus)
        form.addRow(self.lblMrMinusCaption, self.lblMrMinus)
        form.addRow(self.lblMsCaption, self.lblMs)

        self.paramDock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.paramDock)
        self._update_y_quantity_labels()

        # --- Menu bar ---
        file_menu = self.menuBar().addMenu("&File")
        file_menu = self.menuBar().addMenu("&File")

        self.openAct = QtGui.QAction("&Open…", self)
        self.openAct.setShortcut("Ctrl+O")
        self.openAct.triggered.connect(self.open_file)
        file_menu.addAction(self.openAct)

        self.quitAct = QtGui.QAction("&Quit", self)
        self.quitAct.setShortcut("Ctrl+Q")
        self.quitAct.triggered.connect(self.close)
        file_menu.addAction(self.quitAct)

        process_menu = self.menuBar().addMenu("&Process")

        # --- Set Y quantity ---
        self.setAxesAct = QtGui.QAction("Set &axes", self)
        self.setAxesAct.setStatusTip(
            "Declare what the X and Y axes represent (H/B, M/m) and their unit systems"
        )
        self.setAxesAct.triggered.connect(self._open_set_axes_dialog)
        self.setAxesAct.setShortcut("Ctrl+U")
        process_menu.addAction(self.setAxesAct)

        # --- Centre about Y=0 ---
        self.centerYAct = QtGui.QAction("Center about &Y = 0", self)
        self.centerYAct.setShortcut("Ctrl+Y")
        self.centerYAct.setStatusTip("Subtract mean of current Y column so it is centered at zero")
        self.centerYAct.triggered.connect(self.center_y_about_zero)
        process_menu.addAction(self.centerYAct)
    
        self.bgAct = QtGui.QAction("Linear background subtraction (fitting)", self)
        self.bgAct.setShortcut("Ctrl+B")
        self.bgAct.setStatusTip("Fit a straight line to the high-field region and subtract it")
        self.bgAct.triggered.connect(self._bg_start_mode)
        process_menu.addAction(self.bgAct)

        # --- Drift corrections ---
        self.driftTailsAct = QtGui.QAction("Linear drift correction (fitting)", self)
        self.driftTailsAct.setStatusTip("Estimate drift from high-field tails using equally-spaced time")
        self.driftTailsAct.setShortcut("Ctrl+D")
        self.driftTailsAct.triggered.disconnect()
        self.driftTailsAct.triggered.connect(self._drift_start_tails_mode)
        process_menu.addAction(self.driftTailsAct)

        self.driftLoopAct = QtGui.QAction("Linear drift correction (endpoints)", self)
        self.driftLoopAct.setStatusTip("Estimate drift so that first and last points coincide")
        self.driftLoopAct.triggered.connect(self._drift_linear_loopclosure_apply)
        process_menu.addAction(self.driftLoopAct)

         # --- Convert between systems of units ---
        self.unitConvertAct = QtGui.QAction("Convert &units", self)
        self.unitConvertAct.setStatusTip(
            "Convert current field/magnetisation between SI and cgs/Gaussian units"
        )
        self.unitConvertAct.triggered.connect(self._open_unit_convert_dialog)
        process_menu.addAction(self.unitConvertAct)

        # --- Volume normalisation ---
        self.volNormAct = QtGui.QAction("Volume normalisation", self)
        self.volNormAct.setStatusTip("Convert between moment m and magnetisation M using sample volume")
        self.volNormAct.setShortcut("Ctrl+M")
        self.volNormAct.triggered.connect(self._open_volume_normalisation_dialog)
        process_menu.addAction(self.volNormAct)

        process_menu.addSeparator()

        self.undoAct = QtGui.QAction("&Undo last operation", self)
        self.undoAct.setShortcut("Ctrl+Z")
        self.undoAct.setStatusTip("Undo the last data correction")
        self.undoAct.triggered.connect(self.undo_last_operation)
        process_menu.addAction(self.undoAct)

        export_menu = self.menuBar().addMenu("&Export")

        export_hist_act = QtGui.QAction("Export &History", self)
        export_hist_act.triggered.connect(self.export_history)
        export_menu.addAction(export_hist_act)

        self.exportLoopAct = QtGui.QAction("Export current loop to &TXT", self)
        self.exportLoopAct.setStatusTip("Export the current X/Y loop to a text file")
        self.exportLoopAct.triggered.connect(self._export_current_loop_to_txt)
        export_menu.addAction(self.exportLoopAct)

        self.copyLoopAct = QtGui.QAction("&Copy current loop to clipboard", self)
        self.copyLoopAct.setStatusTip("Copy the current X/Y loop as tab-separated text")
        self.copyLoopAct.setShortcut("Ctrl+C")
        self.copyLoopAct.triggered.connect(self._copy_current_loop_to_clipboard)
        export_menu.addAction(self.copyLoopAct)

        # --- Calculate menu (anisotropy area method) ---
        calc_menu = self.menuBar().addMenu("&Calculate")
        self.calcAnisoAct = QtGui.QAction("Anisotropy energy (area method)", self)
        self.calcAnisoAct.setStatusTip("Estimate anisotropy energy by integrating H dM from 0→Ms")
        self.calcAnisoAct.setShortcut("Ctrl+K")
        self.calcAnisoAct.triggered.connect(self._calculate_anisotropy_area)
        calc_menu.addAction(self.calcAnisoAct)

        # Data
        self.original_df = None    # raw data from file
        self.df = None             # current, modified data
        self.numeric_cols = []
        self.last_path = None
        self.history = []          # list of dicts describing operations
        self._wiring_done = False

        # Wire UI interactions
        self.swapBtn.clicked.connect(self._swap_axes)
        self.clearAxesBtn.clicked.connect(lambda: self._reset_axes_semantics(replot=True))
        # Delay connecting combos until we populate them (prevents spurious plots)

    # ---------- File loading ----------
    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open data file",
            self.last_path or "",
            "Data files (*.csv *.txt *.dat *.tsv *.vhd);;All files (*)",
        )
        if not path:
            return
        
        # Always clear current state on new file
        self.df = None
        self.original_df = None
        self.history = []
        
        # Reset axis semantics, don't replot until combos are set:
        self._reset_axes_semantics(replot=False)

        try:
            self.status.showMessage(f"Loading: {path}")
            df = self._read_table_auto(path)

            # Keep original copy + working copy
            self.original_df = df.copy(deep=True)
            self.df = df

            self.last_path = path

            # Determine numeric columns
            import numpy as np
            self.numeric_cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]

            if len(self.numeric_cols) < 2:
                raise ValueError("Need at least two numeric columns to plot.")

            # Populate combos
            self._populate_combos()

            # Initial plot: Y vs X using first two numeric columns
            self._replot()
            self.status.showMessage(f"Loaded: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            self.status.showMessage("Error")

    def _populate_combos(self):
        self.xCombo.blockSignals(True)
        self.yCombo.blockSignals(True)

        self.xCombo.clear()
        self.yCombo.clear()
        self.xCombo.addItems([str(c) for c in self.numeric_cols])
        self.yCombo.addItems([str(c) for c in self.numeric_cols])

        # Default: first → X, second → Y (or preserve prior selections if still valid)
        if len(self.numeric_cols) >= 2:
            self.xCombo.setCurrentIndex(0)
            self.yCombo.setCurrentIndex(1)

        self.xCombo.blockSignals(False)
        self.yCombo.blockSignals(False)

        # Connect signals once
        if not self._wiring_done:
            self.xCombo.currentIndexChanged.connect(self._replot)
            self.yCombo.currentIndexChanged.connect(self._replot)
            self.autoRescaleChk.toggled.connect(self._on_autorescale_toggled)
            self._wiring_done = True
    
    # ---------- History export ----------
    def export_history(self):
        if not self.history:
            QtWidgets.QMessageBox.information(self, "No history",
                                              "There are no operations to export yet.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save history as JSON",
            self.last_path or "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        import json
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            self.status.showMessage(f"History exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))
            self.status.showMessage("Error exporting history")

    # Updates m/M label texts based on current Y quantity type
    def _update_y_quantity_labels(self):
        """
        Update textual labels for Y-dependent parameters (Mr, Ms) based on
        whether Y is magnetisation M, moment m, or unknown.
        """
        q = self.y_quantity_type

        if q == "M":
            # Magnetisation
            self.lblMrCaption.setText("Mr (mag):")
            self.lblMrPlusCaption.setText("Mr+:")
            self.lblMrMinusCaption.setText("Mr−:")
            self.lblMsCaption.setText("M\u209B (sat):")   # Ms with subscript s
        elif q == "m":
            # Magnetic moment
            self.lblMrCaption.setText("mr (mag):")
            self.lblMrPlusCaption.setText("mr+:")
            self.lblMrMinusCaption.setText("mr−:")
            self.lblMsCaption.setText("m\u209B (sat):")   # ms with subscript s
        else:
            # Unknown / ambiguous
            self.lblMrCaption.setText("Rem (Y):")
            self.lblMrPlusCaption.setText("Rem+ (Y):")
            self.lblMrMinusCaption.setText("Rem− (Y):")
            self.lblMsCaption.setText("Sat (Y):")

    # ---------- Plotting ----------
    def _replot(self):
        if self.df is None or not self.numeric_cols:
            return
        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return
        if x_name == y_name:
            self.status.showMessage("X and Y are the same column; nothing to plot.")
            self.canvas.ax.clear()
            self.canvas.fig.canvas.draw_idle()
            return

        x = self.df[x_name].to_numpy()
        y = self.df[y_name].to_numpy()

        # Drop NaNs together to keep paired data aligned
        import numpy as np
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        # Draw
        self.canvas.ax.clear()
        self.canvas.ax.plot(x, y, linewidth=1.5)
        self._set_axis_labels(self.canvas.ax, x_name, y_name)
        self.canvas.ax.set_title(self.windowTitle())
        self.canvas.ax.grid(True, alpha=0.3)

        if self.autoRescaleChk.isChecked():
            self.canvas.ax.relim()
            self.canvas.ax.autoscale()

                # Draw markers if enabled
        if self.chkShowMarkers.isChecked():
            self._draw_feature_markers(x, y)

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(f"Plotted {y_name} vs {x_name} ({len(x)} points)")
        self._update_parameters()

    def _calculate_anisotropy_area(self):
        """Compute anisotropy area by constructing a virgin curve (bisected
        from increasing/decreasing branches) and integrating H dM from 0→Ms.
        Results are shown in a persistent CalculationWindow with two subplots.
        """
        loop_df = self._get_current_loop_dataframe()
        if loop_df is None:
            QtWidgets.QMessageBox.warning(self, "No loop", "Select valid numeric X/Y columns to calculate anisotropy")
            return

        # Extract H (x) and M (y)
        x_name, y_name = loop_df.columns[0], loop_df.columns[1]
        H = np.asarray(loop_df[x_name].to_numpy(), dtype=float)
        M = np.asarray(loop_df[y_name].to_numpy(), dtype=float)

        finite = np.isfinite(H) & np.isfinite(M)
        H = H[finite]
        M = M[finite]

        if H.size < 3:
            QtWidgets.QMessageBox.warning(self, "Too few points", "Need at least 3 finite points to compute virgin curve")
            return

        # Identify increasing / decreasing branches using local difference of H
        dh = np.diff(H)
        # extend last sign for array length alignment
        last_sign = dh[-1] if dh.size else 0.0
        incr_mask = np.concatenate(([dh[0] >= 0], dh >= 0)) if dh.size else np.ones_like(H, dtype=bool)
        decr_mask = ~incr_mask

        H_incr = H[incr_mask]
        M_incr = M[incr_mask]
        H_decr = H[decr_mask]
        M_decr = M[decr_mask]

        # Need at least two points in each branch to interpolate
        if H_incr.size < 2 or H_decr.size < 2:
            # Fallback: try sorting by H and splitting into two halves
            order = np.argsort(H)
            Hs = H[order]
            Ms = M[order]
            mid = len(Hs) // 2
            H_incr, M_incr = Hs[:mid], Ms[:mid]
            H_decr, M_decr = Hs[mid:], Ms[mid:]

        # Sort branch arrays by H for interpolation
        si = np.argsort(H_incr)
        sd = np.argsort(H_decr)
        H_incr_s, M_incr_s = H_incr[si], M_incr[si]
        H_decr_s, M_decr_s = H_decr[sd], M_decr[sd]

        # Build H grid covering common H range
        H_min = max(np.nanmin(H_incr_s), np.nanmin(H_decr_s))
        H_max = min(np.nanmax(H_incr_s), np.nanmax(H_decr_s))
        if not np.isfinite(H_min) or not np.isfinite(H_max) or H_max <= H_min:
            # widen to overall range
            H_min, H_max = float(np.nanmin(H)), float(np.nanmax(H))

        H_grid = np.linspace(H_min, H_max, 2000)

        # Interpolate using numpy.interp (works on sorted x)
        M_incr_grid = np.interp(H_grid, H_incr_s, M_incr_s, left=np.nan, right=np.nan)
        M_decr_grid = np.interp(H_grid, H_decr_s, M_decr_s, left=np.nan, right=np.nan)

        # Virgin curve: bisect (average) where both defined, otherwise take defined
        M_vir = np.nanmean(np.vstack([M_incr_grid, M_decr_grid]), axis=0)
        # where mean is nan because both are nan, try to fill from one side
        mask_incr_valid = ~np.isnan(M_incr_grid)
        mask_decr_valid = ~np.isnan(M_decr_grid)
        M_vir[mask_incr_valid & ~mask_decr_valid] = M_incr_grid[mask_incr_valid & ~mask_decr_valid]
        M_vir[~mask_incr_valid & mask_decr_valid] = M_decr_grid[~mask_incr_valid & mask_decr_valid]

        valid = np.isfinite(M_vir) & np.isfinite(H_grid)
        if valid.sum() < 2:
            QtWidgets.QMessageBox.warning(self, "Invalid virgin curve", "Could not build a valid virgin curve for this loop")
            return

        H_vir = H_grid[valid]
        M_vir = M_vir[valid]

        # Prefer Ms from the last BG fit for current axes if available
        last_bg = self._get_last_bg_info_for_current_axes()
        try:
            if last_bg is not None and last_bg.get('b_pos') is not None and last_bg.get('b_neg') is not None:
                Ms_bg = 0.5 * (last_bg.get('b_pos') - last_bg.get('b_neg'))
                if np.isfinite(Ms_bg) and Ms_bg > 0:
                    Ms_val = float(Ms_bg)
                else:
                    Ms_val = float(np.nanmax(M_vir))
            else:
                Ms_val = float(np.nanmax(M_vir))
        except Exception:
            Ms_val = float(np.nanmax(M_vir))

        if not np.isfinite(Ms_val) or Ms_val == 0.0:
            QtWidgets.QMessageBox.warning(self, "Bad Ms", "Could not determine a positive saturation Ms")
            return

        try:
            area = integrate_HdM(H_vir, M_vir, Ms_val)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Integration error", str(e))
            return

        # Prepare label info (use filename only to keep titles tidy)
        if self.last_path:
            fname = os.path.basename(self.last_path)
            label = f"{x_name} vs {y_name} ({fname})"
        else:
            label = f"{x_name} vs {y_name}"

        # --- Compose axis and area units from current axis semantics ---
        # Use self.x_quantity_type, self.x_unit_system, self.y_quantity_type, self.y_unit_system
        def _axis_label(qtype, qsys, fallback):
            if qtype == 'H':
                if qsys == 'SI':
                    return 'H (A/m)'
                elif qsys == 'cgs-emu/Gaussian':
                    return 'H (Oe)'
                elif qsys == 'Heaviside-Lorentz':
                    return 'H (HL units)'
                else:
                    return 'H'
            elif qtype == 'B':
                if qsys == 'SI':
                    return 'B (T)'
                elif qsys == 'cgs-emu/Gaussian':
                    return 'B (G)'
                elif qsys == 'Heaviside-Lorentz':
                    return 'B (HL units)'
                else:
                    return 'B'
            elif qtype == 'M':
                if qsys == 'SI':
                    return 'M (A/m)'
                elif qsys == 'cgs-emu/Gaussian':
                    return 'M (emu/cm³)'
                elif qsys == 'Heaviside-Lorentz':
                    return 'M (HL units)'
                else:
                    return 'M'
            elif qtype == 'm':
                if qsys == 'SI':
                    return 'm (A·m²)'
                elif qsys == 'cgs-emu/Gaussian':
                    return 'm (emu)'
                elif qsys == 'Heaviside-Lorentz':
                    return 'm (HL units)'
                else:
                    return 'm'
            else:
                return fallback

        x_label = _axis_label(self.x_quantity_type, self.x_unit_system, x_name)
        y_label = _axis_label(self.y_quantity_type, self.y_unit_system, y_name)

        # Compose area units: e.g. H·M, B·m, etc, with units
        def _area_units(xq, xsys, yq, ysys):
            # If either axis unit system is None/unspecified, use 'arb. units'
            if not xsys or not ysys:
                return 'arb. units'
            # Only handle common cases; fallback to generic
            if xq == 'H' and yq == 'M':
                if xsys == 'SI' and ysys == 'SI':
                    return 'A²/m²'
                elif xsys == 'cgs-emu/Gaussian' and ysys == 'cgs-emu/Gaussian':
                    return 'Oe·emu/cm³'
                elif xsys == 'Heaviside-Lorentz' and ysys == 'Heaviside-Lorentz':
                    return 'HL units'
                else:
                    return 'H·M'
            elif xq == 'B' and yq == 'm':
                if xsys == 'SI' and ysys == 'SI':
                    return 'T·A·m²'
                elif xsys == 'cgs-emu/Gaussian' and ysys == 'cgs-emu/Gaussian':
                    return 'G·emu'
                elif xsys == 'Heaviside-Lorentz' and ysys == 'Heaviside-Lorentz':
                    return 'HL units'
                else:
                    return 'B·m'
            elif xq == 'H' and yq == 'm':
                if xsys == 'SI' and ysys == 'SI':
                    return 'A/m·A·m²'
                elif xsys == 'cgs-emu/Gaussian' and ysys == 'cgs-emu/Gaussian':
                    return 'Oe·emu'
                elif xsys == 'Heaviside-Lorentz' and ysys == 'Heaviside-Lorentz':
                    return 'HL units'
                else:
                    return 'H·m'
            elif xq == 'B' and yq == 'M':
                if xsys == 'SI' and ysys == 'SI':
                    return 'T·A/m'
                elif xsys == 'cgs-emu/Gaussian' and ysys == 'cgs-emu/Gaussian':
                    return 'G·emu/cm³'
                elif xsys == 'Heaviside-Lorentz' and ysys == 'Heaviside-Lorentz':
                    return 'HL units'
                else:
                    return 'B·M'
            else:
                return f'{xq}·{yq}'

        area_units = _area_units(self.x_quantity_type, self.x_unit_system, self.y_quantity_type, self.y_unit_system)

        # Create or reuse calculation window
        if not hasattr(self, 'calc_window') or self.calc_window is None:
            self.calc_window = CalculationWindow(None)

        idx = 0 if self.calc_window.areas[0] is None else 1
        self.calc_window.plot_result(
            idx, H, M, H_vir, M_vir, area, label,
            x_label=x_label, y_label=y_label, area_units=area_units,
            xq=self.x_quantity_type, xsys=self.x_unit_system,
            yq=self.y_quantity_type, ysys=self.y_unit_system,
        )
        self.calc_window.show()
        self.calc_window.raise_()
        self.status.showMessage(f"Calculated anisotropy area = {area:.6g} (plotted in slot {idx+1})")

    def _on_markers_toggled(self, checked):
        # If in background preview mode, redraw the preview;
        # otherwise, just replot the committed data.
        if self.bg_mode_active:
            self._bg_update_preview()
        elif self.drift_mode_active:
            self._drift_update_preview()
        else:
            self._replot()

    def _on_autorescale_toggled(self, checked):
        if self.bg_mode_active:
            self._bg_update_preview()
        elif self.drift_mode_active:
            self._drift_update_preview()
        else:
            self._replot()

    def _swap_axes(self):
        if self.xCombo.count() == 0 or self.yCombo.count() == 0:
            return
        xi = self.xCombo.currentIndex()
        yi = self.yCombo.currentIndex()
        if xi < 0 or yi < 0:
            return
        self.xCombo.blockSignals(True)
        self.yCombo.blockSignals(True)
        self.xCombo.setCurrentIndex(yi)
        self.yCombo.setCurrentIndex(xi)
        self.xCombo.blockSignals(False)
        self.yCombo.blockSignals(False)
        self._replot()
        # If a calculation window exists, update stored results to reflect
        # that the axes have been swapped (H <-> M) so area sign is inverted.
        try:
            if hasattr(self, 'calc_window') and self.calc_window is not None:
                self.calc_window.apply_axis_swap()
        except Exception:
            pass

    def _read_table_auto(self, path):
        import io, re, csv
        import pandas as pd
        import numpy as np

        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()

        # --- 1) Find where numeric data begins ---
        def split_guess(s):
            return [t for t in re.split(r"[,\t; ]+", s.strip()) if t != ""]

        def is_number(tok):
            try:
                float(tok.replace("D", "E").replace(",", "."))  # allow Fortran D and decimal commas
                return True
            except Exception:
                return False

        def numeric_fraction(s):
            toks = split_guess(s)
            if len(toks) < 2: return 0.0
            return sum(is_number(t) for t in toks) / len(toks)

        # Honor explicit markers like [Data], or "new section: section 0:" for the DMS MkII
        data_start = None
        for i, ln in enumerate(lines):
            if ln.strip().lower() == "[data]" or ln.strip().lower() == "new section: section 0:": 
                if ln.strip().lower() == "new section: section 0:":
                    DMS_marker = True # True if DMS MkII style marker found
                print(f"Found data start marker at line {i + 1}")
                data_start = i + 1
                break

        if data_start is None:
            for i, ln in enumerate(lines):
                ls = ln.strip()
                if not ls: 
                    continue
                if ls.startswith(("#", "%", "*", "//", "/*", "@")):
                    continue
                if numeric_fraction(ls) >= 0.8: # If no explicit marker found, use first line with high numeric content
                    print(f"Found data start candidate at line {i}")
                    data_start = i
                    break

        if data_start is None:
            raise ValueError("Could not locate the start of numeric data.")

        # Optional header row just before numeric block?
        header_idx = None
        if data_start > 0:
            if DMS_marker != True: # If not DMK MkII style marker, check previous line for header
                prev = lines[data_start - 1].strip()
                if prev and (numeric_fraction(prev) < 0.5) and not prev.startswith(("#", "%", "*", "//", "/*", "@")):
                    header_idx = data_start - 1
            else: # For DMS MkII style files, use four lines above
                prev = lines[data_start - 4].strip()
                if prev and (numeric_fraction(prev) < 0.5) and not prev.startswith(("#", "%", "*", "//", "/*", "@")):
                    header_idx = data_start - 4

        start_idx = header_idx if header_idx is not None else data_start
        tail = [ln.rstrip("\n") for ln in lines[start_idx:]]

        # --- 2) Pick the best delimiter by consistency over first ~50 data rows ---
        candidates = [",", "\t", ";", r"\s+"]  # treat whitespace as a single delimiter
        def score_sep(sep_regex):
            widths = []
            checked = 0
            for ln in tail:
                s = ln.strip()
                if not s or s.startswith(("#", "%")):
                    continue
                parts = re.split(sep_regex, s)
                # ignore obvious inline comments after a delimiter
                if "#" in parts[-1]:
                    parts[-1] = parts[-1].split("#", 1)[0].strip()
                widths.append(len([p for p in parts if p != ""]))
                checked += 1
                if checked >= 50:
                    break
            if not widths:
                return (0, 0, 0)
            return (np.median(widths), -np.std(widths), -abs(widths[0]-np.median(widths)))  # higher is better, then lower variance

        sep_regex = max(candidates, key=score_sep)
        sep_for_pandas = sep_regex if sep_regex == r"\s+" else sep_regex

        # Heuristic: if we picked ';' and see many commas inside tokens, likely decimal commas → keep ';' as sep
        # If we picked ',' but also see many ';', we might have chosen wrong; the consistency score should have handled this.

        # --- 3) Build text, keep header if detected ---
        text = "\n".join(tail)

        # --- 4) Read robustly: let pandas infer header if we found one; skip malformed lines ---
        # Also trim inline comments and extra spaces after delimiters.
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep_for_pandas,
            engine="python",
            header=0 if header_idx is not None else None,
            comment="#",                # drop trailing inline comments
            skipinitialspace=True,      # "x,  y" → treat like "x,y"
            on_bad_lines="skip",        # skip ragged rows instead of raising
            quoting=csv.QUOTE_MINIMAL
        )

        # Convert decimal commas to dots in object columns that look numeric
        for c in df.columns:
            if df[c].dtype == object:
                # only convert if >=80% of non-null cells look like numbers with comma or dot
                ser = df[c].dropna().astype(str)
                if not ser.empty and (ser.str.contains(r"^[\s\-\+\d\.,EeDd]+$").mean() > 0.8):
                    df[c] = pd.to_numeric(ser.str.replace(",", ".", regex=False), errors="coerce")
            # already numeric → keep
        # Drop empty cols/rows
        df = df.dropna(axis=1, how="all").dropna(how="all")
        if df.empty:
            raise ValueError("Parsed an empty table after handling headers and bad lines.")
        return df
    
    # ---------- Data operations ----------
    def _apply_operation(self, op, params, record=True):
        """
        Apply a single operation to self.df.

        op      : string, e.g. "center_y"
        params  : dict, operation-specific parameters
        record  : if True, also append to self.history
        """
        if self.df is None:
            return

        if op == "center_y":
            col_name = params.get("column")
            if col_name not in self.df.columns:
                return

            col = self.df[col_name]

            if not hasattr(col, "dtype") or not np.issubdtype(col.dtype, np.number):
                return

            # Use given offset if present, otherwise recompute
            if "offset" in params:
                offset = float(params["offset"])
            else:
                offset = float(col.mean(skipna=True))

            self.df[col_name] = col - offset

            if record:
                self._add_history_entry(
                    op="center_y",
                    params={"column": col_name, "offset": offset},
                )

        elif op == "bg_linear_branches":
            xcol = params.get("x_column")
            ycol = params.get("column")
            m_bg = float(params.get("m_bg", 0.0))

            if xcol not in self.df.columns or ycol not in self.df.columns:
                return

            x = np.asarray(self.df[xcol].to_numpy(), dtype=float)
            y = np.asarray(self.df[ycol].to_numpy(), dtype=float)

            finite = np.isfinite(x) & np.isfinite(y)
            y_corr = y.copy()
            y_corr[finite] = y[finite] - m_bg * x[finite]

            self.df[ycol] = y_corr

            if record:
                self._add_history_entry("bg_linear_branches", params)
                
        elif op == "drift_linear_tails":
            ycol = params.get("column")
            xcol = params.get("x_column")
            if ycol not in self.df.columns:
                return

            y_series = self.df[ycol]
            if not np.issubdtype(y_series.dtype, np.number):
                return

            y = np.asarray(y_series.to_numpy(), dtype=float)
            n = len(y)
            if n < 2:
                return

            idx = np.arange(n, dtype=float)
            finite_y = np.isfinite(y)
            if finite_y.sum() < 2:
                return

            # Decide reference mask from high-field tails
            mask_ref = finite_y.copy()

            if xcol is not None and xcol in self.df.columns:
                x_series = self.df[xcol]
                if np.issubdtype(x_series.dtype, np.number):
                    x = np.asarray(x_series.to_numpy(), dtype=float)
                    finite_xy = finite_y & np.isfinite(x)

                    if finite_xy.any():
                        x_abs = np.abs(x[finite_xy])
                        Hmax = np.nanmax(x_abs) if x_abs.size else 0.0

                        # Prefer explicit threshold if provided
                        thr_param = params.get("threshold", None)
                        frac_param = params.get("ref_abs_fraction", None)

                        if thr_param is not None:
                            thr = float(thr_param)
                        else:
                            # Use fraction of max|H|, with a sensible default if None
                            frac = 0.8 if frac_param is None else float(frac_param)
                            thr = frac * Hmax if Hmax > 0 else 0.0

                        tmp = np.zeros_like(mask_ref)
                        tmp[finite_xy] = np.abs(x[finite_xy]) >= thr
                        mask_ref = tmp

            mask_ref = mask_ref & finite_y

            if mask_ref.sum() < 2:
                raise ValueError("Not enough reference points in high-field tails to estimate drift.")

            idx_ref = idx[mask_ref]
            y_ref = y[mask_ref]

            if np.allclose(idx_ref, idx_ref[0]):
                raise ValueError("Reference indices are degenerate; cannot fit drift.")

            # Use stored slope if available (for reproducible replay), else compute
            if "drift_slope" in params:
                slope = float(params["drift_slope"])
            else:
                slope, intercept = np.polyfit(idx_ref, y_ref, 1)
                if record:
                    params["drift_slope"] = float(slope)

            idx_center = idx_ref.mean()
            y_corr = y - slope * (idx - idx_center)

            self.df[ycol] = y_corr

            if record:
                # Normalise what we store so future replays don't hit this issue again
                new_params = {
                    "x_column": xcol,
                    "column": ycol,
                    "drift_slope": float(slope),
                }
                # If we used an explicit threshold, keep it; otherwise store fraction
                thr_param = params.get("threshold", None)
                frac_param = params.get("ref_abs_fraction", None)
                if thr_param is not None:
                    new_params["threshold"] = float(thr_param)
                    new_params["ref_abs_fraction"] = None
                elif frac_param is not None:
                    new_params["ref_abs_fraction"] = float(frac_param)
                else:
                    new_params["ref_abs_fraction"] = 0.8

                self._add_history_entry("drift_linear_tails", new_params)

        elif op == "volume_normalisation":
                col = params.get("column")
                if col not in self.df.columns:
                    return

                s = self.df[col]
                if not np.issubdtype(s.dtype, np.number):
                    return

                direction = params.get("direction")           # 'M_to_m' or 'm_to_M'
                system = params.get("system")                 # 'SI' or 'cgs-emu/Gaussian'
                vol_value = params.get("volume_value")
                vol_units = params.get("volume_units")

                if vol_value is None or vol_value <= 0:
                    raise ValueError("Sample volume must be positive.")

                # Determine effective volume in the natural units of that system
                # SI:   M [A/m],   m [A·m^2],   V [m^3]
                # cgs:  M [emu/cm^3], m [emu],  V [cm^3]
                if system == "SI":
                    if vol_units == "m^3":
                        V_eff = float(vol_value)
                    elif vol_units == "cm^3":
                        V_eff = float(vol_value) * 1.0e-6   # 1 cm^3 = 1e-6 m^3
                    else:
                        raise ValueError(f"Unsupported volume units '{vol_units}' for SI.")
                elif system == "cgs-emu/Gaussian":
                    if vol_units == "cm^3":
                        V_eff = float(vol_value)
                    elif vol_units == "m^3":
                        V_eff = float(vol_value) * 1.0e6    # 1 m^3 = 1e6 cm^3
                    else:
                        raise ValueError(f"Unsupported volume units '{vol_units}' for cgs.")
                else:
                    raise ValueError(f"Unsupported unit system '{system}'.")

                # Compute scale factor: m = M * V, M = m / V
                if "scale_factor" in params and not record:
                    # Replaying from history: use stored factor for reproducibility
                    scale_factor = float(params["scale_factor"])
                else:
                    if direction == "M_to_m":
                        scale_factor = V_eff
                    elif direction == "m_to_M":
                        scale_factor = 1.0 / V_eff
                    else:
                        raise ValueError(f"Unknown direction '{direction}' in volume_normalisation.")

                # Apply scaling
                arr = np.asarray(s.to_numpy(), float)
                self.df[col] = arr * scale_factor

                if record:
                    # Store the scale_factor so replay is exact even if we later
                    # change the volume logic.
                    hist_params = dict(params)
                    hist_params["scale_factor"] = float(scale_factor)
                    self._add_history_entry("volume_normalisation", hist_params)

        elif op == "unit_convert":
            from_sys = params.get("from_system")
            to_sys = params.get("to_system")
            field_col = params.get("field_column")
            mag_col = params.get("mag_column")

            # Legacy factors (if present in history)
            field_factor_param = params.get("field_factor", None)
            mag_factor_param = params.get("mag_factor", None)

            # Axis quantities for new entries
            field_q = params.get("field_quantity", None)
            mag_q = params.get("mag_quantity", None)

            # Determine factors
            if not record and (field_factor_param is not None or mag_factor_param is not None):
                # Replaying an existing entry that already stored its factors
                field_factor = field_factor_param if field_factor_param is not None else 1.0
                mag_factor = mag_factor_param if mag_factor_param is not None else 1.0
            else:
                # Compute from systems + quantities (new-style)
                field_factor, mag_factor = self._unit_conversion_factors_axes(
                    from_sys, to_sys, field_q, mag_q
                )

            # Apply to df
            if field_col is not None and field_col in self.df.columns:
                s = self.df[field_col]
                if np.issubdtype(s.dtype, np.number):
                    self.df[field_col] = np.asarray(s.to_numpy(), float) * field_factor

            if mag_col is not None and mag_col in self.df.columns:
                s = self.df[mag_col]
                if np.issubdtype(s.dtype, np.number):
                    self.df[mag_col] = np.asarray(s.to_numpy(), float) * mag_factor

            if record:
                self._add_history_entry(
                    "unit_convert",
                    {
                        "from_system": from_sys,
                        "to_system": to_sys,
                        "field_column": field_col,
                        "mag_column": mag_col,
                        "field_quantity": field_q,
                        "mag_quantity": mag_q,
                        "field_factor": float(field_factor),
                        "mag_factor": float(mag_factor),
                    },
                )

        # --- Loop-closure drift ---
        elif op == "drift_linear_loopclosure":
            ycol = params.get("column")
            if ycol not in self.df.columns:
                return

            y_series = self.df[ycol]
            if not np.issubdtype(y_series.dtype, np.number):
                return

            y = np.asarray(y_series.to_numpy(), dtype=float)
            n = len(y)
            if n < 2:
                return

            idx = np.arange(n, dtype=float)

            # How big a window (fraction of total points) to use at each end
            end_frac = float(params.get("end_window_fraction", 0.02))
            window = max(1, int(end_frac * n))

            # All finite y indices
            finite = np.isfinite(y)
            finite_indices = np.where(finite)[0]
            if finite_indices.size < 2:
                raise ValueError("Not enough finite points to estimate drift.")

            # Take first/last 'window' finite points
            start_indices = finite_indices[:window]
            end_indices = finite_indices[-window:]

            if start_indices.size == 0 or end_indices.size == 0:
                raise ValueError("No finite points in start/end windows for drift estimate.")

            start_mean = float(np.nanmean(y[start_indices]))
            end_mean = float(np.nanmean(y[end_indices]))

            if "drift_slope" in params:
                slope = float(params["drift_slope"])
            else:
                denom = max(1, n - 1)
                slope = (end_mean - start_mean) / denom
                params["drift_slope"] = float(slope)
                params["end_window_fraction"] = end_frac

            # Subtract linear drift vs index so start/end line up on average
            y_corr = y - slope * idx

            self.df[ycol] = y_corr

            if record:
                self._add_history_entry(
                    "drift_linear_loopclosure",
                    {
                        "x_column": params.get("x_column"),
                        "column": ycol,
                        "end_window_fraction": end_frac,
                        "drift_slope": float(slope),
                    },
                )

        # ... more operations in future
        # elif op == "drift_correct": ...
        # (add more operations here as you implement them)

    def _open_set_axes_dialog(self):
        """Manually set what the X and Y axes represent and their unit systems."""
        dlg = SetAxesDialog(
            self,
            current_x_type=self.x_quantity_type,
            current_x_system=self.x_unit_system,
            current_y_type=self.y_quantity_type,
            current_y_system=self.y_unit_system,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        x_type, x_system, y_type, y_system = dlg.get_selection()

        # Update stored axis meanings
        self.x_quantity_type = x_type
        self.x_unit_system = x_system

        self.y_quantity_type = y_type
        self.y_unit_system = y_system

        # Update Y-related captions (Mr vs mr, Ms vs ms)
        self._update_y_quantity_labels()

        # Replot so axis labels and markers pick up new meanings/units
        self._replot()

        desc_x = x_type or "unknown"
        desc_y = y_type or "unknown"
        if x_system:
            desc_x += f" ({x_system})"
        if y_system:
            desc_y += f" ({y_system})"

        self.status.showMessage(f"Axes set: X = {desc_x}, Y = {desc_y}")

    def _open_set_y_quantity_dialog(self):
        """Manually set what the Y axis represents and its unit system."""
        dlg = SetYQuantityDialog(
            self,
            current_type=self.y_quantity_type,
            current_system=self.y_unit_system,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        y_type, y_system = dlg.get_selection()

        # Update state
        self.y_quantity_type = y_type
        self.y_unit_system = y_system

        # Update captions (Mr/Ms vs mr/ms vs generic)
        self._update_y_quantity_labels()

        # Recompute parameters & markers so labels match the new interpretation
        # (replot usually calls _update_parameters, but we can be explicit)
        self._replot()
        self.status.showMessage(
            f"Y quantity set to: "
            f"{'M' if y_type=='M' else 'm' if y_type=='m' else 'unknown'}"
            f"{', system=' + y_system if y_system else ''}"
        )

    # Shift about y=0
    def center_y_about_zero(self):
        """Shift the currently selected Y column so its mean is 0."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before applying corrections.")
            return

        y_name = self.yCombo.currentText()
        if not y_name:
            QtWidgets.QMessageBox.warning(self, "No Y column", "Select a Y column first.")
            return

        # Delegate to operation dispatcher (record=True by default)
        self._apply_operation("center_y", {"column": y_name}, record=True)

        # Grab offset of last operation for status text (optional)
        if self.history and self.history[-1]["op"] == "center_y":
            offset = self.history[-1]["params"].get("offset", 0.0)
            self.status.showMessage(f"Centered {y_name} about 0 (offset {offset:.4g})")
        else:
            self.status.showMessage(f"Centered {y_name} about 0")

        self._replot()

    def _compute_bg_corrected(self, df, xcol, ycol, threshold):
        """
        Fit a straight line to the high-field branches of a hysteresis loop:

        - Positive branch:   x >=  +threshold
        - Negative branch:   x <=  -threshold

        Each branch is fit separately:
        y_pos ≈ m_pos * x + b_pos
        y_neg ≈ m_neg * x + b_neg

        The background slope m_bg is taken as the average of m_pos and m_neg.
        Only this slope is subtracted:

        y_corr = y - m_bg * x
        """
        x = np.asarray(df[xcol].to_numpy(), dtype=float)
        y = np.asarray(df[ycol].to_numpy(), dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            raise ValueError("Not enough finite points in the data.")

        # Ensure threshold is positive
        thr = float(abs(threshold))

        mask_pos = finite & (x >= thr)
        mask_neg = finite & (x <= -thr)

        if mask_pos.sum() < 2 or mask_neg.sum() < 2:
            raise ValueError("Not enough high-field points on one or both branches.")

        x_pos = x[mask_pos]
        y_pos = y[mask_pos]
        x_neg = x[mask_neg]
        y_neg = y[mask_neg]

        # Guard against nearly constant field region in either branch
        if np.allclose(x_pos, x_pos[0]):
            raise ValueError("Positive high-field branch has almost constant H; cannot fit a line.")
        if np.allclose(x_neg, x_neg[0]):
            raise ValueError("Negative high-field branch has almost constant H; cannot fit a line.")

        m_pos, b_pos = np.polyfit(x_pos, y_pos, 1)
        m_neg, b_neg = np.polyfit(x_neg, y_neg, 1)

        # Background slope: average of the two branch slopes
        m_bg = 0.5 * (m_pos + m_neg)

        # Subtract only the background slope
        y_corr = y - m_bg * x

        info = {
            "m_pos": float(m_pos),
            "b_pos": float(b_pos),
            "m_neg": float(m_neg),
            "b_neg": float(b_neg),
            "m_bg": float(m_bg),
            "threshold": thr,
        }
        return y_corr, info

    def _drift_linear_tails_apply(self):
        """Apply linear drift correction estimated from high-field tails."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before drift correction.")
            return

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            QtWidgets.QMessageBox.warning(self, "Select columns",
                                          "Select X and Y columns before drift correction.")
            return

        try:
            # Use a default high-field fraction to define "saturated" tails
            params = {
                "x_column": x_name,
                "column": y_name,
                "ref_abs_fraction": 0.8,  # use |H| >= 0.8 * max|H|
            }
            self._apply_operation("drift_linear", params, record=True)
            self.status.showMessage("Applied linear drift correction.")
            self._replot()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Drift correction error", str(e))

    def _compute_drift_tails(self, df, xcol, ycol, threshold):
        """
        Given df, field column xcol and signal ycol, and a high-field threshold,
        fit drift from tail points (|H| >= threshold) vs index and return:

            y_corr : drift-corrected signal
            info   : dict with 'slope' and 'threshold'
        """
        y = np.asarray(df[ycol].to_numpy(), dtype=float)
        x = np.asarray(df[xcol].to_numpy(), dtype=float)
        n = len(y)
        idx = np.arange(n, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        mask = finite & (np.abs(x) >= float(abs(threshold)))

        if mask.sum() < 2:
            raise ValueError("Not enough high-field points above the threshold to estimate drift.")

        idx_ref = idx[mask]
        y_ref = y[mask]

        if np.allclose(idx_ref, idx_ref[0]):
            raise ValueError("Reference indices are degenerate; cannot fit drift.")

        # Fit y_ref ~ slope * idx_ref + intercept
        slope, intercept = np.polyfit(idx_ref, y_ref, 1)

        # Subtract drift relative to centre of reference indices (for numerical stability)
        idx_center = idx_ref.mean()
        y_corr = y - slope * (idx - idx_center)

        info = {
            "slope": float(slope),
            "threshold": float(abs(threshold)),
        }
        return y_corr, info

    def _drift_start_tails_mode(self):
        """Enter interactive high-field tails drift-correction mode."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before drift correction.")
            return
        
        # When starting drift mode:
        self.bgApplyBtn.setText("Apply drift")
        self.bgCancelBtn.setText("Cancel")

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            QtWidgets.QMessageBox.warning(self, "Select columns",
                                          "Select X and Y columns before drift correction.")
            return

        # Take a snapshot to work from
        self._drift_df_before = self.df.copy(deep=True)
        self._drift_x_col = x_name
        self._drift_y_col = y_name

        x = np.asarray(self._drift_df_before[x_name].to_numpy(), dtype=float)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            QtWidgets.QMessageBox.warning(self, "No data", "Selected X column has no finite values.")
            return

        # initial threshold: 80% of max |H|
        self._drift_threshold = 0.8 * float(np.nanmax(np.abs(x_finite)))

        # Activate mode
        self.drift_mode_active = True
        self.bg_mode_active = False  # ensure BG mode isn't also active
        self.bgApplyBtn.setVisible(True)
        self.bgCancelBtn.setVisible(True)
        self.bgApplyBtn.setText("Apply drift")
        self.bgCancelBtn.setText("Cancel")

        # disable other editing controls (same helper you use for BG)
        self._set_bg_blocked_enabled(False)

        self.status.showMessage(
            "Drift mode: drag the vertical line to set high-field tails, then click 'Apply drift'."
        )

        # create vertical line & connect events
        ax = self.canvas.ax
        ax.clear()
        self._drift_vline = ax.axvline(self._drift_threshold, linestyle="--")

        canvas = self.canvas
        self._drift_cid_press = canvas.mpl_connect("button_press_event", self._drift_on_press)
        self._drift_cid_motion = canvas.mpl_connect("motion_notify_event", self._drift_on_motion)
        self._drift_cid_release = canvas.mpl_connect("button_release_event", self._drift_on_release)
        self._drift_dragging = False

        # show initial preview
        self._drift_update_preview()

    def _drift_linear_loopclosure_apply(self):
        """Apply linear drift correction so that first and last points coincide."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before drift correction.")
            return

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not y_name:
            QtWidgets.QMessageBox.warning(
                self, "Select columns",
                "Select a Y column before drift correction."
            )
            return

        try:
            params = {
                "x_column": x_name,             # not strictly needed here, but kept for history
                "column": y_name,
                "end_window_fraction": 0.02,    # use first/last 2% of points
            }
            self._apply_operation("drift_linear_loopclosure", params, record=True)
            self.status.showMessage("Applied linear drift correction (loop closure).")
            self._replot()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Drift correction error", str(e))

    def _drift_update_preview(self):
        """Recompute and display drift-corrected loop for current tails threshold."""
        if not self.drift_mode_active or self._drift_df_before is None:
            return

        df = self._drift_df_before
        xcol = self._drift_x_col
        ycol = self._drift_y_col
        thr = self._drift_threshold

        x = np.asarray(df[xcol].to_numpy(), dtype=float)
        y = np.asarray(df[ycol].to_numpy(), dtype=float)

        ax = self.canvas.ax
        ax.clear()

        try:
            y_corr, info = self._compute_drift_tails(df, xcol, ycol, thr)
            slope = info["slope"]
            thr_abs = info["threshold"]
            title_extra = f"  (drift slope={slope:.3g}, |H|≥{thr_abs:.3g})"

            # update parameter panel based on *preview* data
            hc_plus, hc_minus, hc_avg = self._compute_coercivity(x, y_corr)
            Mr_plus, Mr_minus, Mr_mag = self._compute_remanence(x, y_corr)
            # Ms & BG slope unchanged by drift, so leave them as they are from committed state
            self._set_param_labels(
                hc_plus=hc_plus, hc_minus=hc_minus, hc_avg=hc_avg,
                Mr_plus=Mr_plus, Mr_minus=Mr_minus, Mr_mag=Mr_mag,
                Ms=None, m_bg=None,
            )

        except Exception as e:
            # If fit fails, show raw loop and revert parameters to committed state
            ax.plot(x, y, linewidth=1.5, alpha=0.6)
            y_label = self._format_y_axis_label(ycol)
            ax.set_xlabel(self._format_x_axis_label(xcol))
            ax.set_ylabel(y_label + " (drift preview)")
            ax.set_title(f"Drift preview (tails): fit invalid: {e}")
            ax.grid(True, alpha=0.3)
            self._update_parameters()
            self.canvas.fig.canvas.draw_idle()
            return

        # 1) Raw loop (faint)
        ax.plot(x, y, linewidth=1.0, alpha=0.3, label="raw loop")

        # 2) Drift-corrected loop
        ax.plot(x, y_corr, linewidth=1.5, label="drift-corrected (preview)")

        # 3) Highlight tail points used for the fit
        finite = np.isfinite(x) & np.isfinite(y)
        thr_abs = float(abs(thr_abs))
        mask_tail = finite & (np.abs(x) >= thr_abs)
        if mask_tail.sum() >= 2:
            ax.scatter(x[mask_tail], y_corr[mask_tail], s=15, marker=".", alpha=0.7, label="tails used")

        # 4) Threshold guides & shading
        self._drift_vline = ax.axvline(+thr_abs, linestyle="--")
        ax.axvline(-thr_abs, linestyle="--")

        x_finite = x[finite]
        if x_finite.size > 0:
            x_max = float(np.nanmax(np.abs(x_finite)))
            ax.axvspan(+thr_abs, +x_max, alpha=0.05)
            ax.axvspan(-x_max, -thr_abs, alpha=0.05)

        y_label = self._format_y_axis_label(ycol)
        ax.set_xlabel(self._format_x_axis_label(xcol))
        ax.set_ylabel(y_label + " (drift preview)")
        ax.set_title("Preview: linear drift (high-field tails)" + title_extra)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # 5) Feature markers on the preview loop
        if self.chkShowMarkers.isChecked():
            self._draw_feature_markers(x, y_corr, bg_info=None, use_history_if_none=True)

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(
            f"Drift mode: |{xcol}| ≥ {thr_abs:.4g}; drag line, then 'Apply drift'"
        )

    def _drift_on_press(self, event):
        if not self.drift_mode_active or event.inaxes != self.canvas.ax:
            return
        if event.xdata is None:
            return
        x_line = self._drift_threshold
        tol = 0.02 * (self.canvas.ax.get_xlim()[1] - self.canvas.ax.get_xlim()[0])
        if abs(event.xdata - x_line) < tol:
            self._drift_dragging = True

    def _drift_on_motion(self, event):
        if not self.drift_mode_active or not self._drift_dragging:
            return
        if event.inaxes != self.canvas.ax or event.xdata is None:
            return

        # Optionally clamp to data range
        x = np.asarray(self._drift_df_before[self._drift_x_col].to_numpy(), dtype=float)
        x_finite = x[np.isfinite(x)]
        if x_finite.size:
            x_min, x_max = float(x_finite.min()), float(x_finite.max())
            self._drift_threshold = min(max(event.xdata, x_min), x_max)
        else:
            self._drift_threshold = event.xdata

        self._drift_update_preview()

    def _drift_on_release(self, event):
        if not self.drift_mode_active:
            return
        self._drift_dragging = False

    def _drift_disconnect_events(self):
        if self._drift_cid_press is not None:
            self.canvas.mpl_disconnect(self._drift_cid_press)
        if self._drift_cid_motion is not None:
            self.canvas.mpl_disconnect(self._drift_cid_motion)
        if self._drift_cid_release is not None:
            self.canvas.mpl_disconnect(self._drift_cid_release)
        self._drift_cid_press = self._drift_cid_motion = self._drift_cid_release = None

    def _drift_exit_mode(self):
        self.drift_mode_active = False
        self._drift_disconnect_events()

        self._drift_df_before = None
        self._drift_x_col = None
        self._drift_y_col = None
        self._drift_threshold = None

        if self._drift_vline is not None:
            try:
                self._drift_vline.remove()
            except Exception:
                pass
            self._drift_vline = None

        self.bgApplyBtn.setVisible(False)
        self.bgCancelBtn.setVisible(False)
        self._set_bg_blocked_enabled(True)

    def _bg_start_mode(self):
        """Enter interactive background-subtraction mode."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before background subtraction.")
            return

        # When starting BG mode:
        self.bgApplyBtn.setText("Apply BG")
        self.bgCancelBtn.setText("Cancel")

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            QtWidgets.QMessageBox.warning(self, "Select columns",
                                          "Select X and Y columns before background subtraction.")
            return

        # Snapshot current df (this is the state we'll operate on)
        self._bg_df_before = self.df.copy(deep=True)
        self._bg_x_col = x_name
        self._bg_y_col = y_name

        x = self._bg_df_before[x_name].to_numpy()
        if x.size == 0:
            QtWidgets.QMessageBox.warning(self, "No data", "Selected X column is empty.")
            return

        # Choose an initial threshold: e.g. 70% of max |x|
        self._bg_threshold = 0.7 * float(np.nanmax(np.abs(x)))
        
        self.bg_mode_active = True
        self.bgApplyBtn.setVisible(True)
        self.bgCancelBtn.setVisible(True)

        # Disable other loop-editing controls while in BG mode
        self._set_bg_blocked_enabled(False)

        self.status.showMessage("Background mode: drag the vertical line to set high-field threshold, then click 'Apply BG'.")

        # Create vertical line
        ax = self.canvas.ax
        if self._bg_vline is not None:
            self._bg_vline.remove()
        self._bg_vline = ax.axvline(self._bg_threshold, linestyle="--")

        # Connect matplotlib events for dragging
        canvas = self.canvas
        self._bg_cid_press = canvas.mpl_connect("button_press_event", self._bg_on_press)
        self._bg_cid_motion = canvas.mpl_connect("motion_notify_event", self._bg_on_motion)
        self._bg_cid_release = canvas.mpl_connect("button_release_event", self._bg_on_release)
        self._bg_dragging = False

        # Draw initial preview
        self._bg_update_preview()

    def _bg_on_press(self, event):
        if not self.bg_mode_active or event.inaxes != self.canvas.ax:
            return
        if event.xdata is None:
            return

        # Check if click is near the vertical line
        x_line = self._bg_threshold
        tol = 0.02 * (self.canvas.ax.get_xlim()[1] - self.canvas.ax.get_xlim()[0])
        if abs(event.xdata - x_line) < tol:
            self._bg_dragging = True

    def _bg_on_motion(self, event):
        if not self.bg_mode_active or not self._bg_dragging:
            return
        if event.inaxes != self.canvas.ax or event.xdata is None:
            return

        # Update threshold and preview
        self._bg_threshold = float(event.xdata)
        self._bg_update_preview()

    def _bg_on_release(self, event):
        if not self.bg_mode_active:
            return
        self._bg_dragging = False

    def _bg_update_preview(self):
        """Recompute and display BG-subtracted data."""
        if not self.bg_mode_active or self._bg_df_before is None:
            return

        df = self._bg_df_before
        xcol = self._bg_x_col
        ycol = self._bg_y_col
        thr = self._bg_threshold

        x = np.asarray(df[xcol].to_numpy(), dtype=float)
        y = np.asarray(df[ycol].to_numpy(), dtype=float)

        ax = self.canvas.ax
        ax.clear()

        try:
            y_corr, info = self._compute_bg_corrected(df, xcol, ycol, thr)
            m_pos = info["m_pos"]
            b_pos = info["b_pos"]
            m_neg = info["m_neg"]
            b_neg = info["b_neg"]
            m_bg  = info["m_bg"]
            thr_abs = info["threshold"]

            title_extra = (
                f"  (m_bg={m_bg:.3g}, m+={m_pos:.3g}, m-={m_neg:.3g}, |H|≥{thr_abs:.3g})"
            )
            # ---- Update parameter panel based on PREVIEW data ----
            hc_plus, hc_minus, hc_avg = self._compute_coercivity(x, y_corr)
            Mr_plus, Mr_minus, Mr_mag = self._compute_remanence(x, y_corr)
            Ms = 0.5 * (b_pos - b_neg)

             # noise + Hs from preview BG tails:
            noise_std = self._estimate_noise_from_bg_tails(x, y_corr, info, Ms)
            if noise_std is not None and noise_std > 0:
                Hs_plus, Hs_minus, Hs_mag = self._compute_saturation_field_noise_based(
                    x, y_corr, Ms, noise_std, k=3
                )
            else:
                Hs_plus = Hs_minus = Hs_mag = None

            self._set_param_labels(
                        hc_plus=hc_plus, hc_minus=hc_minus, hc_avg=hc_avg,
                        Mr_plus=Mr_plus, Mr_minus=Mr_minus, Mr_mag=Mr_mag,
                        Ms=Ms, m_bg=m_bg,
                        hs_plus=Hs_plus, hs_minus=Hs_minus, hs_mag=Hs_mag,
                    )

        except Exception as e:
            # If fit fails (e.g. silly threshold), just show raw loop and
            # revert to parameters from committed data
            ax.plot(x, y, linewidth=1.5, alpha=0.6)
            y_label = self._format_y_axis_label(ycol)
            ax.set_xlabel(self._format_x_axis_label(xcol))
            ax.set_ylabel(y_label + " (BG preview)")
            ax.set_title(f"Preview: background subtraction (fit invalid: {e})")
            ax.grid(True, alpha=0.3)

            # revert parameter panel to committed values
            self._update_parameters()
            self.canvas.fig.canvas.draw_idle()
            return

        # 1) Raw loop (faint, in the background)
        ax.plot(x, y, linewidth=1.0, alpha=0.3, label="raw loop")

        # 2) BG-subtracted loop (preview)
        ax.plot(x, y_corr, linewidth=1.5, label="BG-subtracted (preview)")

        # 3) Dashed fit lines on ± high-field branches (over raw data)
        finite = np.isfinite(x) & np.isfinite(y)
        thr_abs = float(abs(thr_abs))

        mask_pos = finite & (x >= thr_abs)
        mask_neg = finite & (x <= -thr_abs)

        if mask_pos.sum() >= 2:
            x_pos = x[mask_pos]
            # Draw fit line only over the region actually used for the fit
            x_pos_line = np.linspace(x_pos.min(), x_pos.max(), 100)
            y_pos_fit = m_pos * x_pos_line + b_pos
            ax.plot(x_pos_line, y_pos_fit, linestyle="--", linewidth=1.0, label="fit (+H)")

        if mask_neg.sum() >= 2:
            x_neg = x[mask_neg]
            x_neg_line = np.linspace(x_neg.min(), x_neg.max(), 100)
            y_neg_fit = m_neg * x_neg_line + b_neg
            ax.plot(x_neg_line, y_neg_fit, linestyle="--", linewidth=1.0, label="fit (−H)")

        # 4) Symmetric vertical threshold guides + optional shading
        self._bg_vline = ax.axvline(+thr_abs, linestyle="--")
        ax.axvline(-thr_abs, linestyle="--")

        x_finite = x[finite]
        if x_finite.size > 0:
            x_max = float(np.nanmax(np.abs(x_finite)))
            ax.axvspan(+thr_abs, +x_max, alpha=0.05)
            ax.axvspan(-x_max, -thr_abs, alpha=0.05)

        y_label = self._format_y_axis_label(ycol)
        ax.set_xlabel(self._format_x_axis_label(xcol))
        ax.set_ylabel(y_label + " (BG preview)")
        ax.set_title("Preview: background subtraction" + title_extra)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if self.chkShowMarkers.isChecked():
            self._draw_feature_markers(x, y_corr, bg_info=info, use_history_if_none=False)

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(
            f"Background mode: |{xcol}| ≥ {thr_abs:.4g}; drag line, then 'Apply BG'"
        )

    def _bg_disconnect_events(self):
        if self._bg_cid_press is not None:
            self.canvas.mpl_disconnect(self._bg_cid_press)
        if self._bg_cid_motion is not None:
            self.canvas.mpl_disconnect(self._bg_cid_motion)
        if self._bg_cid_release is not None:
            self.canvas.mpl_disconnect(self._bg_cid_release)
        self._bg_cid_press = self._bg_cid_motion = self._bg_cid_release = None

    def _bg_commit(self):
        #if (not self.bg_mode_active or self._bg_df_before is None) and (not self.drift_mode_active or self._drift_df_before is None):
         #   return
        # if BG mode is active → commit BG
        if self.bg_mode_active and self._bg_df_before is not None:
            df_before = self._bg_df_before
            xcol = self._bg_x_col
            ycol = self._bg_y_col
            thr = self._bg_threshold

            try:
                y_corr, info = self._compute_bg_corrected(df_before, xcol, ycol, thr)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Cannot apply background", str(e))
                return

            # Commit: replace current df with BG-subtracted version
            self.df = df_before.copy(deep=True)
            self.df[ycol] = y_corr

            # Log operation so undo/replay works
            self._add_history_entry(
                op="bg_linear_branches",
                params={
                    "x_column": xcol,
                    "column": ycol,
                    "threshold": info["threshold"],
                    "m_pos": info["m_pos"],
                    "b_pos": info["b_pos"],
                    "m_neg": info["m_neg"],
                    "b_neg": info["b_neg"],
                    "m_bg": info["m_bg"],
                },
            )
            self._bg_exit_mode()
            self.status.showMessage(
                f"Applied BG: |{xcol}|>={info['threshold']:.4g}, m_bg={info['m_bg']:.3g}"
            )
            self._replot()
            return

    def _bg_commit(self):
        """
        Handler for the 'Apply' button.

        - If background mode is active, commit the BG subtraction.
        - If drift-tails mode is active, commit the drift correction.
        """
        # --- Background subtraction commit ---
        if self.bg_mode_active:
            if self._bg_df_before is None:
                return

            df_before = self._bg_df_before
            xcol = self._bg_x_col
            ycol = self._bg_y_col
            thr = self._bg_threshold

            try:
                # Uses branch-based BG function:
                #   y_corr, info = _compute_bg_corrected(df, xcol, ycol, threshold)
                # where info = {
                #   "m_pos", "b_pos", "m_neg", "b_neg", "m_bg", "threshold"
                # }
                y_corr, info = self._compute_bg_corrected(df_before, xcol, ycol, thr)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Cannot apply background", str(e))
                return

            # Commit: replace working df with BG-subtracted version
            self.df = df_before.copy(deep=True)
            self.df[ycol] = y_corr

            # Record operation so undo/rebuild works
            self._add_history_entry(
                op="bg_linear_branches",
                params={
                    "x_column": xcol,
                    "column": ycol,
                    "threshold": info.get("threshold"),
                    "m_pos": info.get("m_pos"),
                    "b_pos": info.get("b_pos"),
                    "m_neg": info.get("m_neg"),
                    "b_neg": info.get("b_neg"),
                    "m_bg": info.get("m_bg"),
                },
            )

            # Exit BG mode and redraw
            self._bg_exit_mode()
            self.status.showMessage(
                f"Applied linear background subtraction "
                f"(|{xcol}| ≥ {info.get('threshold', 0):.4g}, "
                f"m_bg = {info.get('m_bg', 0):.3g})"
            )
            self._replot()
            return

        # --- Drift (high-field tails) commit ---
        if self.drift_mode_active:
            if self._drift_df_before is None:
                return

            df_before = self._drift_df_before
            xcol = self._drift_x_col
            ycol = self._drift_y_col
            thr = self._drift_threshold

            try:
                # Uses your drift helper:
                #   y_corr, info = _compute_drift_tails(df, xcol, ycol, threshold)
                # where info = { "slope", "threshold" }
                y_corr, info = self._compute_drift_tails(df_before, xcol, ycol, thr)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Cannot apply drift", str(e))
                return

            # Commit: replace working df with drift-corrected version
            self.df = df_before.copy(deep=True)
            self.df[ycol] = y_corr

            # Record operation (so undo/rebuild can re-apply with same slope/threshold)
            self._add_history_entry(
                op="drift_linear_tails",
                params={
                    "x_column": xcol,
                    "column": ycol,
                    # explicit threshold used in interactive mode
                    "threshold": info.get("threshold"),
                    "drift_slope": info.get("slope"),
                },
            )

            # Exit drift mode and redraw
            self._drift_exit_mode()
            self.status.showMessage(
                f"Applied linear drift correction (tails): "
                f"slope = {info.get('slope', 0):.3g}, "
                f"|{xcol}| ≥ {info.get('threshold', 0):.4g}"
            )
            self._replot()
            return

        # If neither mode is active, do nothing
        return

    def _bg_cancel(self):
        if self.bg_mode_active:
            if not self.bg_mode_active:
                return
            # Just restore df_before and exit
            if self._bg_df_before is not None:
                self.df = self._bg_df_before

            self._bg_exit_mode()
            self.status.showMessage("Background subtraction canceled.")
            self._replot()
            return

        if self.drift_mode_active:
            if self._drift_df_before is not None:
                self.df = self._drift_df_before
            self._drift_exit_mode()
            self.status.showMessage("Drift correction canceled.")
            self._replot()

    def _bg_exit_mode(self):
        """Common cleanup for leaving background mode."""
        self.bg_mode_active = False
        self.bgApplyBtn.setVisible(False)
        self.bgCancelBtn.setVisible(False)
        self._bg_disconnect_events()

        self._bg_df_before = None
        self._bg_x_col = None
        self._bg_y_col = None
        self._bg_threshold = None

        if self._bg_vline is not None:
            try:
                self._bg_vline.remove()
            except Exception:
                pass
            self._bg_vline = None

        self._set_bg_blocked_enabled(True)

    def _set_bg_blocked_enabled(self, enabled: bool):
        """
        Enable/disable controls that shouldn't be usable while background
        preview mode is active (y-offset, undo, axis swap, X/Y selection, etc.).
        """
        # Actions that modify data or history
        for act in [
            getattr(self, "centerYAct", None),
            getattr(self, "undoAct", None),
            getattr(self, "bgAct", None),         # no nested BG inside BG mode
            getattr(self, "driftTailsAct", None),
            getattr(self, "driftLoopAct", None),
            getattr(self, "openAct", None),       # optional: block changing file mid-preview
            getattr(self, "exportLoopAct", None), 
            getattr(self, "copyLoopAct", None),
            getattr(self, "exportHistAct", None), 
            getattr(self, "unitConvertAct", None),
            getattr(self, "volNormAct", None),
            getattr(self, "setAxesAct", None),
        ]:
            if act is not None:
                act.setEnabled(enabled)

        # Widgets that affect axes / columns directly
        for w in [
            getattr(self, "swapBtn", None),
            getattr(self, "xCombo", None),
            getattr(self, "yCombo", None),
        ]:
            if w is not None:
                w.setEnabled(enabled)

    def _compute_coercivity(self, x, y):
        """
        Estimate coercive fields from zero-crossings of y(x):

          - Hc+ : smallest positive field where M crosses zero
          - Hc− : largest negative field where M crosses zero
          - Hc  : average half-width, ~ (Hc+ - Hc−)/2

        Uses simple linear interpolation between sign changes.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if x.size < 2:
            return None, None, None

        hc_candidates = []

        for i in range(len(x) - 1):
            y0, y1 = y[i], y[i + 1]
            x0, x1 = x[i], x[i + 1]

            # Exact zero at a point
            if y0 == 0:
                hc_candidates.append(x0)
                continue
            if y1 == 0:
                hc_candidates.append(x1)
                continue

            # Sign change between points
            if y0 * y1 < 0:
                # Linear interpolation to M=0
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                hc_candidates.append(x_zero)

        if not hc_candidates:
            return None, None, None

        hc_candidates = np.array(hc_candidates, dtype=float)
        pos = hc_candidates[hc_candidates > 0]
        neg = hc_candidates[hc_candidates < 0]

        hc_plus = float(pos.min()) if pos.size else None
        hc_minus = float(neg.max()) if neg.size else None

        hc_avg = None
        if hc_plus is not None and hc_minus is not None:
            hc_avg = 0.5 * (hc_plus - hc_minus)
        elif hc_plus is not None:
            hc_avg = abs(hc_plus)
        elif hc_minus is not None:
            hc_avg = abs(hc_minus)

        return hc_plus, hc_minus, hc_avg

    def _compute_remanence(self, x, y, fallback_window_fraction=0.02):
        """
        Estimate remanent magnetisations using branch-specific interpolation
        at H = 0.

        Strategy:
          - Scan consecutive points for sign changes in H (x).
          - For each pair that brackets H = 0, linearly interpolate M at H = 0.
          - Use dH = x[i+1] - x[i] to decide branch:
                dH < 0  → descending branch (from +H to -H) → Mr+
                dH > 0  → ascending branch (from -H to +H) → Mr−
          - Mr (mag) ≈ (Mr+ - Mr−)/2 for a roughly symmetric loop.

        If no zero-crossings are found, falls back to a simple |H| window
        around 0 with width set by fallback_window_fraction.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if x.size < 2:
            return None, None, None

        # Ensure we actually span H = 0
        if not (np.nanmin(x) <= 0.0 <= np.nanmax(x)):
            # No zero field in range → nothing meaningful to do
            return None, None, None

        Mr_plus_candidates = []   # descending branch (dH < 0)
        Mr_minus_candidates = []  # ascending branch (dH > 0)

        for i in range(len(x) - 1):
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]

            # Skip degenerate segment
            if not np.isfinite(x0) or not np.isfinite(x1):
                continue
            if x1 == x0:
                continue

            # Does this segment bracket H=0?
            # (includes exact zeros and sign changes)
            if x0 == 0.0:
                x_zero = 0.0
                y_zero = float(y0)
            elif x1 == 0.0:
                x_zero = 0.0
                y_zero = float(y1)
            elif x0 * x1 < 0.0:
                # Linear interpolation to H=0
                x_zero = 0.0
                y_zero = float(y0 - x0 * (y1 - y0) / (x1 - x0))
            else:
                continue

            dH = x1 - x0
            if dH < 0:
                Mr_plus_candidates.append(y_zero)   # descending branch (from +H)
            elif dH > 0:
                Mr_minus_candidates.append(y_zero)  # ascending branch (from -H)
            # dH==0 already excluded above

        Mr_plus = np.mean(Mr_plus_candidates) if Mr_plus_candidates else None
        Mr_minus = np.mean(Mr_minus_candidates) if Mr_minus_candidates else None

        # If for some reason we didn’t find both branches, fall back to window method
        if Mr_plus is None or Mr_minus is None:
            max_abs_H = np.nanmax(np.abs(x))
            if max_abs_H > 0:
                h_window = fallback_window_fraction * max_abs_H
                mask = np.abs(x) <= h_window
                if mask.sum() >= 2:
                    y_window = y[mask]
                    Mr_plus_fb = float(np.nanmax(y_window))
                    Mr_minus_fb = float(np.nanmin(y_window))
                    if Mr_plus is None:
                        Mr_plus = Mr_plus_fb
                    if Mr_minus is None:
                        Mr_minus = Mr_minus_fb

        if Mr_plus is None and Mr_minus is None:
            return None, None, None

        Mr_mag = None
        if Mr_plus is not None and Mr_minus is not None:
            Mr_mag = 0.5 * (Mr_plus - Mr_minus)
        elif Mr_plus is not None:
            Mr_mag = abs(Mr_plus)
        elif Mr_minus is not None:
            Mr_mag = abs(Mr_minus)

        return (
            float(Mr_plus) if Mr_plus is not None else None,
            float(Mr_minus) if Mr_minus is not None else None,
            float(Mr_mag) if Mr_mag is not None else None,
        )

    def _compute_saturation_field_noise_based(self, x, y, Ms, noise_std, k=3):
        """
        Hs: first fields past which |M - Ms| <= k*noise on each branch.
        """
        if noise_std is None or noise_std <= 0:
            return None, None, None

        x = np.asarray(x, float)
        y = np.asarray(y, float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 2:
            return None, None, None

        Ms_abs = abs(Ms)
        band = k * noise_std

        # Positive branch: M -> +Ms
        mask_pos = x > 0
        x_pos = x[mask_pos]
        y_pos = y[mask_pos]
        Hs_plus = None
        if x_pos.size:
            sat_mask_pos = np.abs(y_pos - Ms_abs) <= band
            if sat_mask_pos.any():
                Hs_plus = float(x_pos[sat_mask_pos].min())

        # Negative branch: M -> -Ms
        mask_neg = x < 0
        x_neg = x[mask_neg]
        y_neg = y[mask_neg]
        Hs_minus = None
        if x_neg.size:
            sat_mask_neg = np.abs(y_neg + Ms_abs) <= band
            if sat_mask_neg.any():
                Hs_minus = float(x_neg[sat_mask_neg].max())

        Hs_mag = None
        if Hs_plus is not None and Hs_minus is not None:
            Hs_mag = 0.5 * (abs(Hs_plus) + abs(Hs_minus))
        elif Hs_plus is not None:
            Hs_mag = abs(Hs_plus)
        elif Hs_minus is not None:
            Hs_mag = abs(Hs_minus)

        return Hs_plus, Hs_minus, Hs_mag

    def _unit_conversion_factors(self, from_sys, to_sys):
        """
        Return (field_factor, mag_factor) for H and M when converting
        between unit systems.

        - Field H: SI uses A/m, cgs-emu/Gaussian uses Oe.
          1 Oe = 1000 / (4π) A/m  => 1 A/m = 4π/1000 Oe

        - Magnetisation M: SI uses A/m, cgs-emu uses emu/cm^3.
          1 emu/cm^3 = 1000 A/m  => 1 A/m = 1e-3 emu/cm^3
        """
        # Supported systems: SI, cgs-emu/Gaussian, Heaviside-Lorentz (HL).
        # We'll compute conversions by using SI <-> Gaussian factors (legacy)
        # and applying the rationalization factors for HL where required.
        def _canon(sys):
            # Normalise common names (no aliasing to cgs-esu here)
            return sys

        from_sys_c = _canon(from_sys)
        to_sys_c = _canon(to_sys)

        if from_sys_c == to_sys_c:
            return 1.0, 1.0

        four_pi = 4.0 * np.pi

        # Helper: Gaussian <-> HL scale for each quantity
        def _gauss_to_hl_scale(q):
            # Gaussian -> Heaviside-Lorentz:
            #  - H, B : divide by sqrt(4π)
            #  - M, m : multiply by sqrt(4π)
            if q in ("H", "B"):
                return 1.0 / np.sqrt(four_pi)
            if q in ("M", "m"):
                return np.sqrt(four_pi)
            return 1.0

        # SI <-> Gaussian (existing behavior)
        def si_to_gauss_factors():
            return (four_pi / 1000.0, 1.0e-3)

        def gauss_to_si_factors():
            return (1000.0 / four_pi, 1.0e3)

        # Case: SI <-> cgs-emu/Gaussian
        if from_sys_c == "SI" and to_sys_c == "cgs-emu/Gaussian":
            return si_to_gauss_factors()
        if from_sys_c == "cgs-emu/Gaussian" and to_sys_c == "SI":
            return gauss_to_si_factors()

        # Case: SI <-> Heaviside-Lorentz: compose SI->Gaussian and Gaussian->HL
        if from_sys_c == "SI" and to_sys_c == "Heaviside-Lorentz":
            f_field, f_mag = si_to_gauss_factors()
            # apply Gaussian->HL scale
            f_field *= _gauss_to_hl_scale("H")
            f_mag *= _gauss_to_hl_scale("M")
            return f_field, f_mag

        if from_sys_c == "Heaviside-Lorentz" and to_sys_c == "SI":
            # HL -> Gaussian -> SI
            f_field, f_mag = gauss_to_si_factors()
            # HL -> Gaussian scale is inverse of Gaussian->HL
            f_field /= _gauss_to_hl_scale("H")
            f_mag /= _gauss_to_hl_scale("M")
            return f_field, f_mag

        # Case: Gaussian <-> Heaviside-Lorentz
        if from_sys_c == "cgs-emu/Gaussian" and to_sys_c == "Heaviside-Lorentz":
            return (_gauss_to_hl_scale("H"), _gauss_to_hl_scale("M"))

        if from_sys_c == "Heaviside-Lorentz" and to_sys_c == "cgs-emu/Gaussian":
            return (1.0 / _gauss_to_hl_scale("H"), 1.0 / _gauss_to_hl_scale("M"))

        # For other combinations not yet implemented, raise a clear error.
        raise ValueError(
            f"Unit conversion between '{from_sys}' and '{to_sys}' is not implemented yet."
        )

    def _update_parameters(self):
        """
        Recompute and display physical parameters for the selected
        X/Y columns from the current df and history.

        - Coercivity (Hc+, Hc−, Hc) whenever possible.
        - Background slope and M_sat only after a BG correction
          (last bg_linear_branches op for these columns).
        """
        # Defaults when nothing is available
        for lbl in [self.lblHc, self.lblHcPlus, self.lblHcMinus,
                    self.lblMr, self.lblMrPlus, self.lblMrMinus,
                    self.lblHs, self.lblHsPlus, self.lblHsMinus,
                    self.lblMs, self.lblBgSlope]:
            lbl.setText("—")

        if self.df is None:
            return

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return

        x_series = self.df.get(x_name)
        y_series = self.df.get(y_name)
        if x_series is None or y_series is None:
            return

        if not (np.issubdtype(x_series.dtype, np.number) and
                np.issubdtype(y_series.dtype, np.number)):
            return

        x = np.asarray(x_series.to_numpy(), dtype=float)
        y = np.asarray(y_series.to_numpy(), dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if x.size < 2:
            return

        # --- 1) Coercivity from current loop ---
        hc_plus, hc_minus, hc_avg = self._compute_coercivity(x, y)
        if hc_plus is not None:
            self.lblHcPlus.setText(f"{hc_plus:.4g}")
        if hc_minus is not None:
            self.lblHcMinus.setText(f"{hc_minus:.4g}")
        if hc_avg is not None:
            self.lblHc.setText(f"{hc_avg:.4g}")

        # --- 2) Remanence from current loop ---
        Mr_plus, Mr_minus, Mr_mag = self._compute_remanence(x, y)
        if Mr_plus is not None:
            self.lblMrPlus.setText(f"{Mr_plus:.4g}")
        if Mr_minus is not None:
            self.lblMrMinus.setText(f"{Mr_minus:.4g}")
        if Mr_mag is not None:
            self.lblMr.setText(f"{Mr_mag:.4g}")

        # --- 3) Background slope and M_sat from the last BG operation ---
        last_bg = self._get_last_bg_info_for_current_axes()
        if last_bg is None:
            # No BG correction yet for this X/Y → leave Ms, BG slope as "—"
            return
        # Background slope m_bg
        m_bg = last_bg.get("m_bg", None)
        if m_bg is not None:
            self.lblBgSlope.setText(f"{m_bg:.4g}")

        # --- 4) M_sat: from high-field intercepts of BG fits
        b_pos = last_bg.get("b_pos", None)
        b_neg = last_bg.get("b_neg", None)
        if b_pos is not None and b_neg is not None:
            # For a symmetric loop, M_sat ≈ (b_pos - b_neg) / 2
            Ms = 0.5 * (b_pos - b_neg)
            self.lblMs.setText(f"{Ms:.4g}")

        # --- 5) Saturation field from Ms and current loop (only if Ms known) ---
        last_bg = self._get_last_bg_info_for_current_axes()
        if last_bg is None:
            return

        if last_bg is not None and Ms is not None:
            noise_std = self._estimate_noise_from_bg_tails(x, y, last_bg, Ms)
            if noise_std is not None:
                Hs_plus, Hs_minus, Hs_mag = self._compute_saturation_field_noise_based(
                    x, y, Ms, noise_std, k=2
                )
            if Hs_plus is not None:
                self.lblHsPlus.setText(f"{Hs_plus:.4g}")
            else:
                self.lblHsPlus.setText("—")

            if Hs_minus is not None:
                self.lblHsMinus.setText(f"{Hs_minus:.4g}")
            else:
                self.lblHsMinus.setText("—")

            if Hs_mag is not None:
                self.lblHs.setText(f"{Hs_mag:.4g}")
            else:
                self.lblHs.setText("—")
        else:
            # No Ms → no meaningful Hs
            self.lblHs.setText("—")
            self.lblHsPlus.setText("—")
            self.lblHsMinus.setText("—")

    def _get_last_bg_info_for_current_axes(self, scaled=True):
        """
        Find the most recent bg_linear_branches operation that matches
        the currently selected X/Y columns.

        If scaled=True, return a COPY of its params with threshold and
        magnetisation-related values scaled into the *current* units,
        taking into account any unit_convert and volume_normalisation
        operations that occurred AFTER the BG fit.

        If scaled=False, return the raw params dict stored in history.
        """
        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return None

        if not self.history:
            return None

        last_idx = None
        last_raw = None

        # 1) Find the last BG op for these axes
        for i, entry in enumerate(self.history):
            if entry.get("op") == "bg_linear_branches":
                params = entry.get("params", {})
                if (params.get("x_column") == x_name and
                        params.get("column") == y_name):
                    last_idx = i
                    last_raw = params

        if last_raw is None:
            return None
        if not scaled:
            return last_raw

        # 2) Compute net scaling from *later* ops
        field_factor = 1.0
        mag_factor = 1.0

        for entry in self.history[last_idx + 1:]:
            op = entry.get("op")
            p = entry.get("params", {})

            if op == "unit_convert":
                # Use stored factors from that op
                f_fac = p.get("field_factor", None)
                m_fac = p.get("mag_factor", None)

                # Only apply if that conversion touched our current axes
                if p.get("field_column") == x_name and f_fac is not None:
                    field_factor *= float(f_fac)
                if p.get("mag_column") == y_name and m_fac is not None:
                    mag_factor *= float(m_fac)

            elif op == "volume_normalisation":
                # Volume normalisation scales ONLY the Y column
                if p.get("column") == y_name:
                    sf = p.get("scale_factor", None)
                    if sf is not None:
                        mag_factor *= float(sf)

        # 3) Build a scaled copy of the BG fit
        scaled_params = dict(last_raw)  # shallow copy is fine (all scalars)

        # Threshold is in field units
        if "threshold" in scaled_params and scaled_params["threshold"] is not None:
            scaled_params["threshold"] = scaled_params["threshold"] * field_factor

        # Magnetisation-related params (Ms etc.) are in mag units
        for key in ("m_bg", "m_pos", "m_neg", "b_pos", "b_neg"):
            if key in scaled_params and scaled_params[key] is not None:
                scaled_params[key] = scaled_params[key] * mag_factor

        return scaled_params

    def _estimate_noise_from_bg_tails(self, x, y, last_bg, Ms):
        """
        Estimate noise from the BG-corrected high-field tails:

        - use |H| >= threshold from last_bg
        - subtract +Ms / -Ms on each branch
        - compute std of residuals
        """
        if last_bg is None or Ms is None:
            return None

        x = np.asarray(x, float)
        y = np.asarray(y, float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            return None

        thr = float(abs(last_bg.get("threshold", 0.0)))

        mask_pos = finite & (x >= thr)
        mask_neg = finite & (x <= -thr)

        residuals = []

        if mask_pos.sum() >= 2:
            # positive tail ~ +Ms
            res_pos = y[mask_pos] - Ms
            residuals.append(res_pos)

        if mask_neg.sum() >= 2:
            # negative tail ~ -Ms
            res_neg = y[mask_neg] + Ms
            residuals.append(res_neg)

        if not residuals:
            return None

        all_res = np.concatenate(residuals)
        if all_res.size < 2:
            return None

        noise_std = float(np.nanstd(all_res))
        return noise_std

    def _open_volume_normalisation_dialog(self):
        """Open the volume normalisation dialog and apply if accepted."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before volume normalisation.")
            return

        y_name = self.yCombo.currentText()
        if not y_name or y_name not in self.df.columns:
            QtWidgets.QMessageBox.warning(
                self, "Select Y column",
                "Select a Y column (moment or magnetisation) before volume normalisation."
            )
            return

        dlg = VolumeNormalisationDialog(
            self,
            known_y_type=self.y_quantity_type,
            known_y_system=self.y_unit_system,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        current_quantity, system, vol_value, vol_units = dlg.get_selection()

        if vol_value <= 0:
            QtWidgets.QMessageBox.warning(
                self, "Invalid volume",
                "Sample volume must be positive."
            )
            return

        target_quantity = "m" if current_quantity == "M" else "M"
        direction = "M_to_m" if current_quantity == "M" else "m_to_M"

        params = {
            "column": y_name,
            "current_quantity": current_quantity,  # 'M' or 'm'
            "target_quantity": target_quantity,
            "direction": direction,                # 'M_to_m' or 'm_to_M'
            "system": system,                      # 'SI' or 'cgs-emu/Gaussian'
            "volume_value": vol_value,
            "volume_units": vol_units,             # 'm^3' or 'cm^3'
        }

        try:
            # Apply the numeric scaling
            self._apply_operation("volume_normalisation", params, record=True)

            # Update semantic state before replotting, so labels use it
            self.y_quantity_type = target_quantity   # 'M' or 'm' after conversion
            self.y_unit_system = system              # 'SI' or 'cgs-emu/Gaussian'
            self._update_y_quantity_labels()

            # Now replot with the new semantics
            self._replot()

            self.status.showMessage(
                f"Volume normalisation: {current_quantity} → {target_quantity} "
                f"(V={vol_value:g} {vol_units}, system={system})"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Volume normalisation error", str(e))

    def _open_unit_convert_dialog(self):
        """Open the unit conversion dialog and apply conversion if accepted."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before converting units.")
            return

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            QtWidgets.QMessageBox.warning(
                self, "Select columns",
                "Select X and Y columns before converting units."
            )
            return

        dlg = UnitConversionDialog(
            self,
            known_y_type=self.y_quantity_type,
            known_y_system=self.y_unit_system,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        (from_sys, to_sys,
        convert_x, x_type_str,
        convert_y, y_type_str) = dlg.get_selection()

        if from_sys == to_sys:
            self.status.showMessage("Unit conversion: source and target systems are the same.")
            return

        # Implement SI <-> cgs-emu/Gaussian and Heaviside-Lorentz (rationalized Gaussian)
        impl_sys = {"SI", "cgs-emu/Gaussian", "Heaviside-Lorentz"}
        if from_sys not in impl_sys or to_sys not in impl_sys:
            QtWidgets.QMessageBox.information(
                self,
                "Not implemented",
                (
                    "Unit conversion is currently implemented for SI, cgs-emu/Gaussian, "
                    "and Heaviside-Lorentz (rationalized Gaussian) for H, B, M, and m.\n\n"
                    "Other systems are not yet supported."
                ),
            )
            return

        if not convert_x and not convert_y:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to convert",
                "Please select at least one of 'Convert X' or 'Convert Y'."
            )
            return

        # Map UI strings → quantity codes ONCE
        x_q = None
        if convert_x:
            if x_type_str == "Field H":
                x_q = "H"
            elif x_type_str == "Flux density B":
                x_q = "B"

        y_q = None
        if convert_y:
            if y_type_str == "Magnetisation M":
                y_q = "M"
            elif y_type_str == "Magnetic moment m":
                y_q = "m"

        params = {
            "from_system": from_sys,
            "to_system": to_sys,
            "field_column": x_name if convert_x else None,
            "mag_column": y_name if convert_y else None,
            "field_quantity": x_q,
            "mag_quantity": y_q,
        }

        try:
            # Apply numeric conversion first
            self._apply_operation("unit_convert", params, record=True)

            # ✅ NOW update axis meaning/state BEFORE replotting
            if convert_x and x_q is not None:
                self.x_quantity_type = x_q
                self.x_unit_system = to_sys

            if convert_y and y_q is not None:
                self.y_quantity_type = y_q
                self.y_unit_system = to_sys
                self._update_y_quantity_labels()

            # Replot with updated state so labels use H/B/M/m + units
            self._replot()

            self.status.showMessage(
                f"Converted units: {from_sys} → {to_sys} "
                f"(X: {x_q or '-'}, Y: {y_q or '-'})"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Unit conversion error", str(e))

    def _unit_conversion_factor_for_quantity(self, quantity, from_sys, to_sys):
        """
        Return multiplicative factor for a given physical quantity when
        converting values from from_sys → to_sys.

        Supported systems: SI <-> cgs-emu/Gaussian
        Quantities:
          H : magnetic field strength
          B : flux density
          M : magnetisation
          m : magnetic moment
        """
        if quantity is None or from_sys == to_sys:
            return 1.0

        # No aliasing to cgs-esu; support SI, cgs-emu/Gaussian, Heaviside-Lorentz.
        def _canon(sys):
            return sys

        from_sys_c = _canon(from_sys)
        to_sys_c = _canon(to_sys)

        four_pi = 4.0 * np.pi

        def _gauss_to_hl_scale(q):
            if q in ("H", "B"):
                return 1.0 / np.sqrt(four_pi)
            if q in ("M", "m"):
                return np.sqrt(four_pi)
            return 1.0

        # SI -> Gaussian (legacy factors)
        if from_sys_c == "SI" and to_sys_c == "cgs-emu/Gaussian":
            if quantity == "H":
                return four_pi / 1000.0
            elif quantity == "B":
                return 1.0e4
            elif quantity == "M":
                return 1.0e-3
            elif quantity == "m":
                return 1.0e3

        # Gaussian -> SI
        if from_sys_c == "cgs-emu/Gaussian" and to_sys_c == "SI":
            if quantity == "H":
                return 1000.0 / four_pi
            elif quantity == "B":
                return 1.0e-4
            elif quantity == "M":
                return 1.0e3
            elif quantity == "m":
                return 1.0e-3

        # SI <-> Heaviside-Lorentz via Gaussian
        if from_sys_c == "SI" and to_sys_c == "Heaviside-Lorentz":
            base = None
            if quantity == "H":
                base = four_pi / 1000.0
            elif quantity == "B":
                base = 1.0e4
            elif quantity == "M":
                base = 1.0e-3
            elif quantity == "m":
                base = 1.0e3
            if base is None:
                raise ValueError("Unknown quantity")
            return base * _gauss_to_hl_scale(quantity)

        if from_sys_c == "Heaviside-Lorentz" and to_sys_c == "SI":
            # HL -> Gaussian -> SI
            base = None
            if quantity == "H":
                base = 1000.0 / four_pi
            elif quantity == "B":
                base = 1.0e-4
            elif quantity == "M":
                base = 1.0e3
            elif quantity == "m":
                base = 1.0e-3
            if base is None:
                raise ValueError("Unknown quantity")
            # HL->Gaussian scaling is inverse of Gaussian->HL
            return base / _gauss_to_hl_scale(quantity)

        # Gaussian <-> HL direct
        if from_sys_c == "cgs-emu/Gaussian" and to_sys_c == "Heaviside-Lorentz":
            if quantity in ("H", "B"):
                return 1.0 / np.sqrt(four_pi)
            if quantity in ("M", "m"):
                return np.sqrt(four_pi)

        if from_sys_c == "Heaviside-Lorentz" and to_sys_c == "cgs-emu/Gaussian":
            if quantity in ("H", "B"):
                return np.sqrt(four_pi)
            if quantity in ("M", "m"):
                return 1.0 / np.sqrt(four_pi)

        raise ValueError(
            f"Unit conversion for quantity '{quantity}' between '{from_sys}' "
            f"and '{to_sys}' is not implemented."
        )

    def _unit_conversion_factors_axes(self, from_sys, to_sys, field_quantity, mag_quantity):
        """
        Convenience: get factors for the X axis ('field_column') and
        Y axis ('mag_column') given their physical meanings.
        """
        field_factor = self._unit_conversion_factor_for_quantity(field_quantity, from_sys, to_sys) \
            if field_quantity is not None else 1.0
        mag_factor = self._unit_conversion_factor_for_quantity(mag_quantity, from_sys, to_sys) \
            if mag_quantity is not None else 1.0
        return field_factor, mag_factor

    def _get_x_units_string(self):
        """Return a unit string for the X axis based on x_quantity_type & x_unit_system."""
        q = self.x_quantity_type
        s = self.x_unit_system

        if q == "H":
            if s == "SI":
                return "A/m"
            elif s == "cgs-emu/Gaussian":
                return "Oe"
            elif s == "Heaviside-Lorentz":
                return "Oe (HL)"
        elif q == "B":
            if s == "SI":
                return "T"
            elif s == "cgs-emu/Gaussian":
                return "G"
            elif s == "Heaviside-Lorentz":
                return "G (HL)"

        return None  # unknown

    def _get_y_units_string(self):
        """Return a unit string for the Y axis based on y_quantity_type & y_unit_system."""
        q = self.y_quantity_type
        s = self.y_unit_system

        if q == "M":
            if s == "SI":
                return "A/m"
            elif s == "cgs-emu/Gaussian":
                return "emu/cm^3"
            elif s == "Heaviside-Lorentz":
                return "emu/cm^3 (HL)"
        elif q == "m":
            if s == "SI":
                return "A·m^2"
            elif s == "cgs-emu/Gaussian":
                return "emu"
            elif s == "Heaviside-Lorentz":
                return "emu (HL)"

        return None  # unknown

    def _format_x_axis_label(self, col_name: str) -> str:
        """
        Compose the X axis label from the known quantity/unit info and/or
        the raw column name.
        """
        q = self.x_quantity_type
        units = self._get_x_units_string()

        if q in ("H", "B"):
            base = q
            if units:
                return f"{base} ({units})"
            else:
                return base
        else:
            # Unknown physical meaning → fall back to column name
            return col_name if col_name else "X"

    def _format_y_axis_label(self, col_name: str) -> str:
        """
        Compose the Y axis label from the known quantity/unit info and/or
        the raw column name.
        """
        q = self.y_quantity_type
        units = self._get_y_units_string()

        if q in ("M", "m"):
            base = q
            if units:
                return f"{base} ({units})"
            else:
                return base
        else:
            # Unknown physical meaning → fall back to column name
            return col_name if col_name else "Y"

    def _set_axis_labels(self, ax, x_col_name: str, y_col_name: str):
        """Convenience: set both axis labels on a given Axes."""
        ax.set_xlabel(self._format_x_axis_label(x_col_name))
        ax.set_ylabel(self._format_y_axis_label(y_col_name))

    def _reset_axes_semantics(self, replot: bool = True):
        """
        Forget the physical meaning and unit systems of X and Y.

        This does NOT change any data or history; it only resets:
        - x_quantity_type / x_unit_system
        - y_quantity_type / y_unit_system
        and updates labels accordingly.
        """
        self.x_quantity_type = None
        self.x_unit_system = None
        self.y_quantity_type = None
        self.y_unit_system = None

        # Reset parameter captions back to generic "Sat (Y)", "Rem (Y)", etc.
        self._update_y_quantity_labels()

        # Replot so axis labels revert to column names
        if replot and self.df is not None:
            self._replot()

    def _draw_feature_markers(self, x, y, bg_info=None, use_history_if_none=True):
        """
        Draw feature markers (Mr, Hc, Ms, Hs, etc.) on the current axes.

        - x, y: arrays for the currently plotted loop (committed or preview)
        - bg_info: optional BG info dict for this specific loop (e.g. preview)
        - use_history_if_none: if True and bg_info is None, fall back to the
        most recent bg_linear_branches op from history for the current axes.
        """
        ax = self.canvas.ax

        # Make sure we work only with finite points
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        q = self.y_quantity_type

        if q == "M":
            label_Ms = "Ms"
            label_Mr_plus = "Mr+"
            label_Mr_minus = "Mr−"
        elif q == "m":
            label_Ms = "ms"
            label_Mr_plus = "mr+"
            label_Mr_minus = "mr−"
        else:
            label_Ms = "Ys"
            label_Mr_plus = "Yr+"
            label_Mr_minus = "Yr−"

        if x.size < 2:
            return

        # 1) Remanence markers
        Mr_plus, Mr_minus, Mr_mag = self._compute_remanence(x, y)

        if Mr_plus is not None:
            ax.scatter([0.0], [Mr_plus], s=30, marker="o")
            ax.annotate(
                label_Mr_plus,
                xy=(0.0, Mr_plus),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        if Mr_minus is not None:
            ax.scatter([0.0], [Mr_minus], s=30, marker="o")
            ax.annotate(
                label_Mr_minus,
                xy=(0.0, Mr_minus),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
            )

        # 2) Coercivity markers
        hc_plus, hc_minus, hc_avg = self._compute_coercivity(x, y)

        if hc_plus is not None:
            ax.scatter([hc_plus], [0.0], s=30, marker="s")
            ax.annotate(
                "Hc+",
                xy=(hc_plus, 0.0),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        if hc_minus is not None:
            ax.scatter([hc_minus], [0.0], s=30, marker="s")
            ax.annotate(
                "Hc−",
                xy=(hc_minus, 0.0),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
            )

        # 3) Ms & Hs (only if BG info exists)
        Ms = None  

        last_bg = bg_info
        if last_bg is None and use_history_if_none:
            # This returns a *scaled* copy of the last BG info in current units
            last_bg = self._get_last_bg_info_for_current_axes()

        if last_bg is None:
            # No BG fit: nothing more to draw (Mr/Hc only)
            return

        # Compute Ms from BG intercepts
        b_pos = last_bg.get("b_pos", None)
        b_neg = last_bg.get("b_neg", None)
        if b_pos is None or b_neg is None:
            # BG exists but missing intercepts → can't define Ms sensibly
            return

        Ms = 0.5 * (b_pos - b_neg)

        # If Ms is NaN or not finite, bail out of Ms/Hs part
        if not np.isfinite(Ms):
            return

        # --- Ms horizontal lines ---
        x_min, x_max = float(x.min()), float(x.max())
        ax.hlines(Ms,  x_min, x_max, linestyles="--")
        ax.hlines(-Ms, x_min, x_max, linestyles="--")
        ax.annotate(
            label_Ms,
            xy=(x_min, Ms),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

        # --- Hs vertical lines (noise-based) ---
        # Need noise estimate from BG tails; function should safely return None
        noise_std = self._estimate_noise_from_bg_tails(x, y, last_bg, Ms)
        if noise_std is None or not np.isfinite(noise_std) or noise_std <= 0:
            return

        Hs_plus, Hs_minus, Hs_mag = self._compute_saturation_field_noise_based(
            x, y, Ms, noise_std, k=3
        )

        if Hs_plus is None and Hs_minus is None:
            return

        y_min, y_max = float(y.min()), float(y.max())

        if Hs_plus is not None and np.isfinite(Hs_plus):
            ax.vlines(Hs_plus, y_min, y_max, linestyles="--")
            ax.annotate(
                "Hs+",
                xy=(Hs_plus, y_max),
                xytext=(3, -15),
                textcoords="offset points",
                fontsize=8,
            )

        if Hs_minus is not None and np.isfinite(Hs_minus):
            ax.vlines(Hs_minus, y_min, y_max, linestyles="--")
            ax.annotate(
                "Hs−",
                xy=(Hs_minus, y_max),
                xytext=(3, -15),
                textcoords="offset points",
                fontsize=8,
            )


    # --- Hs vertical markers (noise-based) ---
        noise_std = self._estimate_noise_from_bg_tails(x, y, last_bg, Ms)
        if noise_std is not None and noise_std > 0:
            Hs_plus, Hs_minus, Hs_mag = self._compute_saturation_field_noise_based(
                x, y, Ms, noise_std, k=3
            )

            y_min, y_max = float(y.min()), float(y.max())

            if Hs_plus is not None:
                ax.vlines(Hs_plus, y_min, y_max, linestyles="--")
                ax.annotate(
                    "Hs+",
                    xy=(Hs_plus, y_max),
                    xytext=(3, -15),
                    textcoords="offset points",
                    fontsize=8,
                )

            if Hs_minus is not None:
                ax.vlines(Hs_minus, y_min, y_max, linestyles="--")
                ax.annotate(
                    "Hs−",
                    xy=(Hs_minus, y_max),
                    xytext=(3, -15),
                    textcoords="offset points",
                    fontsize=8,
                )

    def _set_param_labels(self,
                        hc_plus=None, hc_minus=None, hc_avg=None,
                        Mr_plus=None, Mr_minus=None, Mr_mag=None,
                        Ms=None, m_bg=None,
                        hs_plus=None, hs_minus=None, hs_mag=None):
        """Update the parameter panel labels, leaving None as '—'."""
        def set_lbl(lbl, val):
            lbl.setText("—" if val is None else f"{val:.4g}")

        set_lbl(self.lblHcPlus, hc_plus)
        set_lbl(self.lblHcMinus, hc_minus)
        set_lbl(self.lblHc, hc_avg)

        set_lbl(self.lblHsPlus, hs_plus)
        set_lbl(self.lblHsMinus, hs_minus)
        set_lbl(self.lblHs, hs_mag)

        set_lbl(self.lblMrPlus, Mr_plus)
        set_lbl(self.lblMrMinus, Mr_minus)
        set_lbl(self.lblMr, Mr_mag)

        set_lbl(self.lblMs, Ms)
        set_lbl(self.lblBgSlope, m_bg)

    def _add_history_entry(self, op, params):
        """Append an operation to the history log."""
        from datetime import datetime

        entry = {
            "op": op,
            "params": params or {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.history.append(entry)
        # optional: print to console for debugging
        # print("HISTORY:", entry)

    def undo_last_operation(self):
        """Remove the last operation from history and rebuild df from scratch."""
        if not self.history:
            QtWidgets.QMessageBox.information(self, "Nothing to undo",
                                              "There are no operations to undo.")
            return

        # Remove last operation
        last = self.history.pop()

        # Rebuild df from original + remaining history
        self._rebuild_df_from_history()

        # Update status + plot
        self.status.showMessage(f"Undid: {last['op']}")
        self._replot()

    def _rebuild_df_from_history(self):
        """Reset df to original_df and replay all history entries."""
        if self.original_df is None:
            return

        # Start from a clean copy of the raw data
        self.df = self.original_df.copy(deep=True)

        # Recompute numeric columns
        self.numeric_cols = [
            c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)
        ]

        # Try to keep current X/Y choices if they still exist
        old_x = self.xCombo.currentText() if self.xCombo.count() > 0 else None
        old_y = self.yCombo.currentText() if self.yCombo.count() > 0 else None

        # Re-apply all operations without re-recording them
        for entry in self.history:
            self._apply_operation(entry["op"], entry["params"], record=False)

        # Repopulate combos if needed
        # (e.g. after operations that add/remove columns in the future)
        if not self.numeric_cols:
            return

        # Ensure combos list the right columns
        self._populate_combos()

        # Try to restore previous x/y selection when still valid
        def restore_combo(combo, name):
            if name and combo.findText(name) != -1:
                combo.setCurrentText(name)

        restore_combo(self.xCombo, old_x)
        restore_combo(self.yCombo, old_y)

    def _get_current_loop_dataframe(self):
        """
        Return a 2-column DataFrame with the currently selected X and Y
        columns, or None if not available/valid.
        """
        if self.df is None:
            return None

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return None

        if x_name not in self.df.columns or y_name not in self.df.columns:
            return None

        x_series = self.df[x_name]
        y_series = self.df[y_name]

        # Require numeric columns for export
        if not (np.issubdtype(x_series.dtype, np.number) and
                np.issubdtype(y_series.dtype, np.number)):
            return None

        # Keep as-is (including NaNs) so export matches what you are plotting
        loop_df = pd.DataFrame({
            x_name: x_series.to_numpy(),
            y_name: y_series.to_numpy(),
        })

        return loop_df
    
    def _export_current_loop_to_txt(self):
        """Export the current X/Y loop to a tab-separated .txt file."""
        loop_df = self._get_current_loop_dataframe()
        if loop_df is None:
            QtWidgets.QMessageBox.warning(
                self, "Cannot export",
                "No valid numeric X/Y columns selected to export."
            )
            return

        # File dialog
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current loop to TXT",
            "",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return

        try:
            # Tab-separated, header included, no index
            loop_df.to_csv(
                path,
                sep="\t",
                index=False,
                float_format="%.10g",
            )
            self.status.showMessage(f"Exported current loop to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export error",
                f"Failed to export data:\n{e}"
            )
    
    def _copy_current_loop_to_clipboard(self):
        """Copy the current X/Y loop as tab-separated text to the clipboard."""
        loop_df = self._get_current_loop_dataframe()
        if loop_df is None:
            QtWidgets.QMessageBox.warning(
                self, "Cannot copy",
                "No valid numeric X/Y columns selected to copy."
            )
            return

        # Convert DataFrame to TSV string
        from io import StringIO
        buf = StringIO()
        loop_df.to_csv(
            buf,
            sep="\t",
            index=False,
            float_format="%.10g",
        )
        text = buf.getvalue()

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)

        self.status.showMessage(
            f"Copied {len(loop_df)} rows of {loop_df.columns[0]} vs {loop_df.columns[1]} to clipboard"
        )

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
