"""
Simple runnable checks for anisotropy helpers in QT_test.py.
Run with: env/bin/python tools/run_anisotropy_checks.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from QT_test import build_virgin_curve, integrate_HdM
import numpy as np

# Create a synthetic hysteresis loop: H from -1..1 and back, M is tanh(H*5)
H1 = np.linspace(-1.0, 1.0, 201)
M1 = np.tanh(H1 * 5.0)
H2 = np.linspace(1.0, -1.0, 201)
M2 = np.tanh(H2 * 5.0)
# concatenate to make a loop (increasing then decreasing)
H_loop = np.concatenate([H1, H2])
M_loop = np.concatenate([M1, M2])

print('Building virgin curve from synthetic loop...')
H_vir, M_vir, Ms = build_virgin_curve(H_loop, M_loop)
print(f'Virgin Ms = {Ms:.6g}, points = {len(H_vir)}')
area = integrate_HdM(H_vir, M_vir, Ms)
print(f'Integrated area (synthetic) = {area:.6g}')

# Basic checks: area should be finite and positive
assert np.isfinite(area), 'Area is not finite'
assert area > 0, 'Area should be positive for this synthetic loop'
print('Anisotropy helper checks passed.')
