#!/usr/bin/env python3
"""
Analyze high-field behavior to check saturation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ipr_file = os.path.join(base_dir, 'test_data', 'IPR.csv')
opr_file = os.path.join(base_dir, 'test_data', 'OPR.csv')

# Load raw data
df_ip = pd.read_csv(ipr_file)
df_oop = pd.read_csv(opr_file)

H_ip = df_ip.iloc[:, 0].values  # Oe
m_ip = df_ip.iloc[:, 1].values  # emu
H_oop = df_oop.iloc[:, 0].values
m_oop = df_oop.iloc[:, 1].values

print("=== High-field analysis (raw data) ===\n")

# Analyze high-field region (H > 4000 Oe)
mask_ip = H_ip > 4000
mask_oop = H_oop > 4000

H_ip_hf = H_ip[mask_ip]
m_ip_hf = m_ip[mask_ip]
H_oop_hf = H_oop[mask_oop]
m_oop_hf = m_oop[mask_oop]

print("In-plane high-field (H > 4000 Oe):")
print(f"  H range: {H_ip_hf.min():.0f} to {H_ip_hf.max():.0f} Oe")
print(f"  m range: {m_ip_hf.min():.6e} to {m_ip_hf.max():.6e} emu")
print(f"  Δm over high field: {(m_ip_hf.max() - m_ip_hf.min()):.6e} emu")
print(f"  Relative change: {100*(m_ip_hf.max() - m_ip_hf.min())/m_ip_hf.max():.2f}%")

# Linear fit to high-field region
p_ip = np.polyfit(H_ip_hf, m_ip_hf, 1)
print(f"  Linear fit: m = {p_ip[0]:.6e} * H + {p_ip[1]:.6e}")
print(f"  Slope (susceptibility): χ = {p_ip[0]:.6e} emu/Oe")
print()

print("Out-of-plane high-field (H > 4000 Oe):")
print(f"  H range: {H_oop_hf.min():.0f} to {H_oop_hf.max():.0f} Oe")
print(f"  m range: {m_oop_hf.min():.6e} to {m_oop_hf.max():.6e} emu")
print(f"  Δm over high field: {(m_oop_hf.max() - m_oop_hf.min()):.6e} emu")
print(f"  Relative change: {100*(m_oop_hf.max() - m_oop_hf.min())/m_oop_hf.max():.2f}%")

p_oop = np.polyfit(H_oop_hf, m_oop_hf, 1)
print(f"  Linear fit: m = {p_oop[0]:.6e} * H + {p_oop[1]:.6e}")
print(f"  Slope (susceptibility): χ = {p_oop[0]:.6e} emu/Oe")
print()

print("=== Interpretation ===")
print(f"In-plane slope:      {p_ip[0]:.6e} emu/Oe  ({'POSITIVE → NOT SATURATED' if p_ip[0] > 0 else 'negative'})")
print(f"Out-of-plane slope:  {p_oop[0]:.6e} emu/Oe  ({'positive' if p_oop[0] > 0 else 'NEGATIVE → saturated'})")
print()
print("Difference in slopes: {:.6e} emu/Oe".format(p_ip[0] - p_oop[0]))
print()

# Estimate how much extra field needed for in-plane
# Assuming we need the slope to match the OOP slope
print("=== Saturation estimate ===")
# The excess slope is ferromagnetic contribution
chi_excess = p_ip[0] - p_oop[0]
print(f"Excess susceptibility (FM): {chi_excess:.6e} emu/Oe")

# Current unsaturated moment at H_max
m_at_6kOe = m_ip_hf[H_ip_hf == H_ip_hf.max()][0]
# Estimate true Ms by extrapolating when slope would match OOP
# This is approximate - assumes linear approach to saturation
print(f"\nCurrent moment at {H_ip_hf.max():.0f} Oe: {m_at_6kOe:.6e} emu")
print(f"If we need to reach where slope = {p_oop[0]:.6e}, and excess slope is {chi_excess:.6e}...")
print(f"Very rough estimate: need ~{chi_excess*1e4:.1e} emu more moment")
print(f"At current rate, that's ~{1e4*(chi_excess/chi_excess):.0f} Oe = ~1 Tesla more field")
print()

# Volume normalized comparison
V_cm3 = 4e-7
Ms_ip_bg = 0.000394  # from background fit
Ms_oop_bg = 0.0004094

print("=== After volume normalization ===")
print(f"Volume: {V_cm3:.2e} cm³")
print(f"In-plane Ms (BG fit): {Ms_ip_bg:.6e} emu → {Ms_ip_bg/V_cm3:.6e} emu/cm³")
print(f"Out-of-plane Ms (BG fit): {Ms_oop_bg:.6e} emu → {Ms_oop_bg/V_cm3:.6e} emu/cm³")
print(f"Actual at max field IP: {m_at_6kOe:.6e} emu → {m_at_6kOe/V_cm3:.6e} emu/cm³")
print(f"Difference: {100*(m_at_6kOe - Ms_ip_bg)/Ms_ip_bg:.2f}% above BG fit value")
