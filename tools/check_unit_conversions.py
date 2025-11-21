import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from QT_test import MainWindow
import math

# Use the unbound method to avoid creating a QApplication
conv = MainWindow._unit_conversion_factor_for_quantity

quantities = ["H", "B", "M", "m"]

def approx_eq(a, b, rel=1e-9, abs_tol=1e-12):
    return abs(a-b) <= max(rel * max(abs(a), abs(b)), abs_tol)

print("Checking unit conversion factors...")
for q in quantities:
    f_si_hl = conv(None, q, 'SI', 'Heaviside-Lorentz')
    f_hl_si = conv(None, q, 'Heaviside-Lorentz', 'SI')
    prod = f_si_hl * f_hl_si
    print(f"{q}: SI->HL = {f_si_hl:.12g}, HL->SI = {f_hl_si:.12g}, product = {prod:.12g}")
    if not approx_eq(prod, 1.0, rel=1e-9, abs_tol=1e-9):
        raise SystemExit(f"Round-trip SI<->{q} HL failed (product={prod})")

# Also check SI <-> Gaussian round-trip
for q in quantities:
    f_si_g = conv(None, q, 'SI', 'cgs-emu/Gaussian')
    f_g_si = conv(None, q, 'cgs-emu/Gaussian', 'SI')
    prod = f_si_g * f_g_si
    print(f"{q}: SI->G = {f_si_g:.12g}, G->SI = {f_g_si:.12g}, product = {prod:.12g}")
    if not approx_eq(prod, 1.0, rel=1e-9, abs_tol=1e-9):
        raise SystemExit(f"Round-trip SI<->Gaussian failed (product={prod})")

print("All conversion sanity checks passed.")
