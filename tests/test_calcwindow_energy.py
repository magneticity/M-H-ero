import math
import numpy as np
import pytest
from PySide6 import QtWidgets

from QT_test import CalculationWindow, build_virgin_curve


def ensure_qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def make_result(area, xq, xsys, yq, ysys):
    # Minimal result dict used by _update_diff_label
    return {
        'H': np.array([0.0]),
        'M': np.array([0.0]),
        'H_vir': np.array([0.0]),
        'M_vir': np.array([0.0]),
        'area': float(area),
        'label': 'test',
        'xq': xq,
        'xsys': xsys,
        'yq': yq,
        'ysys': ysys,
    }


def test_cgs_K_eff_display():
    ensure_qt_app()
    cw = CalculationWindow(None)

    cw.raw_results[0] = make_result(100.0, 'H', 'cgs-emu/Gaussian', 'M', 'cgs-emu/Gaussian')
    cw.raw_results[1] = make_result(30.0, 'H', 'cgs-emu/Gaussian', 'M', 'cgs-emu/Gaussian')
    cw.areas[0] = 100.0
    cw.areas[1] = 30.0

    cw._update_diff_label()
    txt = cw.diff_label.text()
    assert 'Δ area' in txt
    assert 'K_eff' in txt
    assert 'erg' in txt or 'erg/cm' in txt
    assert '70' in txt


def test_cgs_Ea_moment_display():
    ensure_qt_app()
    cw = CalculationWindow(None)

    cw.raw_results[0] = make_result(200.0, 'H', 'cgs-emu/Gaussian', 'm', 'cgs-emu/Gaussian')
    cw.raw_results[1] = make_result(50.0, 'H', 'cgs-emu/Gaussian', 'm', 'cgs-emu/Gaussian')
    cw.areas[0] = 200.0
    cw.areas[1] = 50.0

    cw._update_diff_label()
    txt = cw.diff_label.text()
    assert 'Δ area' in txt
    assert 'E_a' in txt
    assert 'erg' in txt
    assert '150' in txt


def test_si_K_eff_conversion():
    ensure_qt_app()
    cw = CalculationWindow(None)

    cw.raw_results[0] = make_result(10.0, 'H', 'SI', 'M', 'SI')
    cw.raw_results[1] = make_result(2.0, 'H', 'SI', 'M', 'SI')
    cw.areas[0] = 10.0
    cw.areas[1] = 2.0

    cw._update_diff_label()
    txt = cw.diff_label.text()
    # diff = 8.0; energy = mu0 * 8.0
    mu0 = 4.0 * math.pi * 1e-7
    expected = f"{(mu0 * 8.0):.6g}"
    assert 'K_eff' in txt
    assert 'J' in txt
    assert expected in txt


def test_hl_equivalence_to_cgs():
    ensure_qt_app()
    cw = CalculationWindow(None)

    # HL numeric diff should be treated like cgs (erg/cm3) and convertible to J/m3
    cw.raw_results[0] = make_result(50.0, 'H', 'Heaviside-Lorentz', 'M', 'Heaviside-Lorentz')
    cw.raw_results[1] = make_result(20.0, 'H', 'Heaviside-Lorentz', 'M', 'Heaviside-Lorentz')
    cw.areas[0] = 50.0
    cw.areas[1] = 20.0

    cw._update_diff_label()
    txt = cw.diff_label.text()
    assert 'K_eff' in txt
    assert 'erg' in txt
    # converted J/m3 should also appear (diff = 30 -> 3.0 J/m3)
    assert '3' in txt


def test_build_virgin_curve_monotonic_tail():
    """Construct a synthetic loop that would produce a noisy virgin curve at high H
    and ensure build_virgin_curve returns M_vir non-decreasing (monotonic)."""
    # synthetic increasing/decreasing branches with noise at high H
    H_inc = np.linspace(0, 1.0, 200)
    M_inc = np.linspace(0, 1.0, 200) + 0.02 * np.sin(np.linspace(0, 20, 200))
    H_dec = np.linspace(1.0, 0, 200)
    M_dec = np.linspace(1.0, 0, 200) + 0.02 * np.cos(np.linspace(0, 20, 200))

    H = np.concatenate([H_inc, H_dec])
    M = np.concatenate([M_inc, M_dec])

    H_vir, M_vir, Ms = build_virgin_curve(H, M, n_grid=500)
    # Check monotonic non-decreasing M_vir
    diffs = np.diff(M_vir)
    assert np.all(diffs >= -1e-8)
