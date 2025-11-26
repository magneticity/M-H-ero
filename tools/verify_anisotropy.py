#!/usr/bin/env python3
"""
Verify anisotropy calculation from exported processed data.
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import from QT_test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QT_test import integrate_HdM

def load_and_calculate(in_plane_file, out_of_plane_file):
    """Load processed data and calculate anisotropy energy."""
    
    # Load data
    df_ip = pd.read_csv(in_plane_file)
    df_oop = pd.read_csv(out_of_plane_file)
    
    print("=== In-plane data ===")
    print(f"Columns: {list(df_ip.columns)}")
    print(f"Number of points: {len(df_ip)}")
    print(f"H range: {df_ip.iloc[:, 0].min():.2e} to {df_ip.iloc[:, 0].max():.2e} A/m")
    print(f"M range: {df_oop.iloc[:, 1].min():.2e} to {df_ip.iloc[:, 1].max():.2e} A/m")
    print()
    
    print("=== Out-of-plane data ===")
    print(f"Columns: {list(df_oop.columns)}")
    print(f"Number of points: {len(df_oop)}")
    print(f"H range: {df_oop.iloc[:, 0].min():.2e} to {df_oop.iloc[:, 0].max():.2e} A/m")
    print(f"M range: {df_oop.iloc[:, 1].min():.2e} to {df_oop.iloc[:, 1].max():.2e} A/m")
    print()
    
    # Extract H and M
    H_ip = df_ip.iloc[:, 0].values
    M_ip = df_ip.iloc[:, 1].values
    H_oop = df_oop.iloc[:, 0].values
    M_oop = df_oop.iloc[:, 1].values
    
    # Build virgin curves using the same method as the app
    def build_virgin_curve(H, M):
        """Build virgin curve from hysteresis data."""
        finite = np.isfinite(H) & np.isfinite(M)
        H = H[finite]
        M = M[finite]
        
        # Identify increasing / decreasing branches
        dh = np.diff(H)
        incr_mask = np.concatenate(([dh[0] >= 0], dh >= 0)) if dh.size else np.ones_like(H, dtype=bool)
        decr_mask = ~incr_mask
        
        H_incr = H[incr_mask]
        M_incr = M[incr_mask]
        H_decr = H[decr_mask]
        M_decr = M[decr_mask]
        
        # Sort by H
        si = np.argsort(H_incr)
        sd = np.argsort(H_decr)
        H_incr_s, M_incr_s = H_incr[si], M_incr[si]
        H_decr_s, M_decr_s = H_decr[sd], M_decr[sd]
        
        # Build H grid
        H_min = max(np.nanmin(H_incr_s), np.nanmin(H_decr_s))
        H_max = min(np.nanmax(H_incr_s), np.nanmax(H_decr_s))
        H_grid = np.linspace(H_min, H_max, 2000)
        
        # Interpolate
        M_incr_grid = np.interp(H_grid, H_incr_s, M_incr_s, left=np.nan, right=np.nan)
        M_decr_grid = np.interp(H_grid, H_decr_s, M_decr_s, left=np.nan, right=np.nan)
        
        # Virgin curve: average
        M_vir = np.nanmean(np.vstack([M_incr_grid, M_decr_grid]), axis=0)
        mask_incr_valid = ~np.isnan(M_incr_grid)
        mask_decr_valid = ~np.isnan(M_decr_grid)
        M_vir[mask_incr_valid & ~mask_decr_valid] = M_incr_grid[mask_incr_valid & ~mask_decr_valid]
        M_vir[~mask_incr_valid & mask_decr_valid] = M_decr_grid[~mask_incr_valid & mask_decr_valid]
        
        valid = np.isfinite(M_vir) & np.isfinite(H_grid)
        return H_grid[valid], M_vir[valid]
    
    print("Building virgin curves...")
    H_vir_ip, M_vir_ip = build_virgin_curve(H_ip, M_ip)
    H_vir_oop, M_vir_oop = build_virgin_curve(H_oop, M_oop)
    
    # Determine Ms (use max of virgin curves)
    Ms_ip = float(np.nanmax(M_vir_ip))
    Ms_oop = float(np.nanmax(M_vir_oop))
    Ms_avg = 0.5 * (Ms_ip + Ms_oop)
    
    print(f"Ms (in-plane): {Ms_ip:.6e} A/m")
    print(f"Ms (out-of-plane): {Ms_oop:.6e} A/m")
    print(f"Ms (average): {Ms_avg:.6e} A/m")
    print()
    
    # Calculate areas
    print("Calculating anisotropy areas...")
    area_ip = integrate_HdM(H_vir_ip, M_vir_ip, Ms_avg)
    area_oop = integrate_HdM(H_vir_oop, M_vir_oop, Ms_avg)
    
    print(f"Area (in-plane): {area_ip:.6e} A²/m²")
    print(f"Area (out-of-plane): {area_oop:.6e} A²/m²")
    print(f"Area difference (IP - OOP): {area_ip - area_oop:.6e} A²/m²")
    print()
    
    # Calculate energy
    mu0 = 4 * np.pi * 1e-7  # H/m
    diff = area_ip - area_oop
    K_eff = mu0 * diff  # J/m³
    
    print(f"μ₀ = {mu0:.6e} H/m")
    print(f"K_eff = μ₀ × Δarea = {K_eff:.6e} J/m³")
    print()
    
    # Calculate H_K
    H_K = 2.0 * K_eff / (mu0 * Ms_avg)
    print(f"H_K = 2K_eff/(μ₀Ms) = {H_K:.6e} A/m")
    print()
    
    # Expected values for comparison
    print("=== Analysis ===")
    # Estimate saturation field from in-plane data
    # Look at field where M reaches ~95% of Ms
    threshold = 0.95 * Ms_avg
    sat_indices = np.where(M_vir_ip >= threshold)[0]
    if len(sat_indices) > 0:
        H_sat_estimate = H_vir_ip[sat_indices[0]]
        print(f"In-plane saturation field (95% Ms): {H_sat_estimate:.6e} A/m")
        print(f"Ratio H_sat/H_K: {H_sat_estimate/H_K:.2f}")
        print()
    
    # Expected K_eff if H_K ~ H_sat
    if len(sat_indices) > 0:
        K_expected = H_sat_estimate * mu0 * Ms_avg / 2.0
        print(f"If H_K ≈ H_sat, expected K_eff: {K_expected:.6e} J/m³")
        print(f"Ratio of expected/measured K_eff: {K_expected/K_eff:.2f}")
        print()
    
    # Check if the issue is with the virgin curve construction
    print("=== Virgin curve statistics ===")
    print(f"In-plane virgin curve: {len(H_vir_ip)} points")
    print(f"  H range: {H_vir_ip.min():.2e} to {H_vir_ip.max():.2e} A/m")
    print(f"  M range: {M_vir_ip.min():.2e} to {M_vir_ip.max():.2e} A/m")
    print(f"Out-of-plane virgin curve: {len(H_vir_oop)} points")
    print(f"  H range: {H_vir_oop.min():.2e} to {H_vir_oop.max():.2e} A/m")
    print(f"  M range: {M_vir_oop.min():.2e} to {M_vir_oop.max():.2e} A/m")
    
    return {
        'Ms': Ms_avg,
        'area_ip': area_ip,
        'area_oop': area_oop,
        'K_eff': K_eff,
        'H_K': H_K
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(base_dir, 'test_data')
    in_plane = os.path.join(test_data_dir, 'IPP.csv')
    out_of_plane = os.path.join(test_data_dir, 'OPP.csv')
    
    if not os.path.exists(in_plane) or not os.path.exists(out_of_plane):
        print("Error: Could not find IPP.csv or OPP.csv in test_data directory")
        sys.exit(1)
    
    print(f"Using in-plane file: {os.path.basename(in_plane)}")
    print(f"Using out-of-plane file: {os.path.basename(out_of_plane)}")
    print()
    
    results = load_and_calculate(in_plane, out_of_plane)
