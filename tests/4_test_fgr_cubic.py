#!/usr/bin/env python3
"""
Fermi's Golden Rule: perturbative acoustic emission rate.

With cubic anharmonicity V₃ = (α/6) Σ_edges (δr)³, a belt mode p
can decay into two phonons via:
  Γ_p = π α² Σ_{m,n} |V_{mnp}|² δ(ω_p - ω_m - ω_n)

The rate is expressed as Γ/ω = ε² f, where ε = α·ℓ₀/k_L and f is
a dimensionless factor decomposed by acoustic content.

NOTES:
  - f_ac = 0 at Γ is trivial, not a deep result. At k=0 the only modes
    below ω_edge are the 3 translations (ω=0), which have H[m,e] = 0
    because e_m^j = e_m^i (uniform displacement). The coupling vertex
    V_{mnp} vanishes whenever m or n is a translation. A nontrivial
    acoustic fraction would require BZ integration (finite-k modes with
    ω < ω_edge have nonzero H). The tests document this Γ-point result
    honestly; the physically meaningful test is test_total_rate_positive
    (2-phonon decay exists).
  - BZ-integrated FGR rate is not computed here. The MD test (test 5)
    provides indirect all-k validation: it simulates a real supercell where
    all k-modes interact via anharmonicity, confirming E_low ∝ ε² scaling.
    The Γ-point FGR shows the mechanism exists; the MD shows it works at
    all k collectively.
  - Acoustic fraction at finite k is NOT small (~25-60% at k=0.05..0.3).
    The selection rule breaks at k≠0 (source projection grows as k²,
    see test 3) and acoustic channels become O(1). This is expected:
    at finite k, protection comes from the kinematic gap (energy
    conservation), not from coupling being small.
  - f is independent of α: the FGR formula has Γ ∝ α² and ε² ∝ α²,
    so f = Γ/(ωε²) cancels α. A test of "rate vs alpha²" would be a
    tautology. The MD test (test 5) tests something genuinely different:
    that the nonlinear dynamics remains perturbative (E_low ∝ ε²).

RAW OUTPUT (pytest -v -s, Mar 2026):
  test_one_phonon_forbidden    ω_belt = 1.2112, ω_edge = 0.8022, ratio = 1.510
  test_ac_ac_negligible        f_ac_ac/f_total = 0.000000
  test_low_omega_fraction_small f_ac/f_total = 0.000 (acoustic fraction)
  test_total_rate_positive     f_total = 7.5020e-05
  test_optical_dominates       f_optical = 7.5020e-05, f_ac = 0.0000e+00, ratio = inf (f_ac=0)
  test_omega_edge_computed     ω_edge = 0.8022 (computed from BZ scan)
  test_rate_linear_in_eta      f/η = [1.50e-3, 1.50e-3, 1.50e-3], spread = 0.0012
  test_lifetime_estimate       τ = 1.33e+08 periods at ε=0.01 (rate table: ε=0.001→1.33e+10, ε=0.1→1.33e+06)
  test_belt_mode_coverage      c² in ω > ω_edge: 1.0000, top-5 belt: 0.3729

References:
  physics_ai/ST_11/wip/w_15_kelvin_no_drag/05_fgr_cubic.py
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/w14_13_beyond_harmonic.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.analysis.no_drag import (
    build_enriched_belt_vectors, compute_acoustic_ceiling
)
from physics.bloch import DisplacementBloch


@pytest.fixture(scope="module")
def fgr_data():
    """Build Kelvin N=2, compute FGR channel decomposition."""
    N = 2
    L = 4.0 * N
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    nv = len(v)
    n_dof = 3 * nv

    # Diagonalize D(k=0) — must be real at k=0
    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)
    D0_complex = bloch.build_dynamical_matrix(np.zeros(3))
    assert np.allclose(D0_complex.imag, 0, atol=1e-14), (
        f"D(k=0) has imaginary part: max|imag| = {np.max(np.abs(D0_complex.imag)):.2e}"
    )
    D0 = np.real(D0_complex)
    evals, evecs = np.linalg.eigh(D0)
    evecs = np.real(evecs)
    omega = np.sqrt(np.maximum(evals, 0))

    # Belt source
    box_center = np.array([L/2, L/2, L/2])
    dists_to_center = np.linalg.norm(centers - box_center, axis=1)
    source_cell = np.argmin(dists_to_center)
    _, old_vec = build_enriched_belt_vectors(
        source_cell, centers, v, f, cfi, L)
    u0 = old_vec / np.linalg.norm(old_vec)

    coeffs = evecs.T @ u0
    c2 = np.abs(coeffs)**2
    order = np.argsort(c2)[::-1]

    # Edge geometry
    v_arr = np.array(v)
    idx_i = np.array([edge[0] for edge in e])
    idx_j = np.array([edge[1] for edge in e])
    delta_edges = v_arr[idx_j] - v_arr[idx_i]
    delta_edges = delta_edges - L * np.round(delta_edges / L)
    edge_lengths = np.linalg.norm(delta_edges, axis=1)
    edge_dirs = delta_edges / edge_lengths[:, np.newaxis]

    # H[m, e] = (e_m^j - e_m^i) · ê / sqrt(2·omega_m)
    coord_idx = np.arange(3)[None, :]
    ei = evecs[3 * idx_i[:, None] + coord_idx, :]
    ej = evecs[3 * idx_j[:, None] + coord_idx, :]
    H = np.einsum('ed,edm->me', edge_dirs, ej - ei)

    omega_min = 0.01
    valid = omega > omega_min
    omega_safe = np.where(valid, omega, 1.0)
    H = H / np.sqrt(2 * omega_safe)[:, np.newaxis]
    H[~valid, :] = 0.0

    ell0 = np.mean(edge_lengths)
    assert np.std(edge_lengths) / ell0 < 1e-10, (
        f"Edge lengths not uniform: std/mean = {np.std(edge_lengths)/ell0:.2e}"
    )

    # Acoustic ceiling (computed, not hardcoded)
    res_ceil = compute_acoustic_ceiling(bloch, L)
    omega_edge = res_ceil['omega_edge']

    # Frequency masks
    ac_mask_strict = omega < omega_edge

    # Belt modes: above acoustic ceiling, sorted by c² weight
    belt_modes = [m for m in order[:20] if omega[m] > omega_edge]
    belt_calc = belt_modes[:5]
    eta = 0.05
    k_L = 2.0

    rates = {'total': 0, 'ac_strict': 0, 'ac_ac': 0, 'n_belt': 0}
    c2_weights = []

    for p in belt_calc:
        omega_p = omega[p]
        g_p = H[p, :]
        H_w = H * g_p[np.newaxis, :]
        C_matrix = H_w @ H.T
        C2 = C_matrix**2

        omega_sum = omega[np.newaxis, :] + omega[:, np.newaxis]
        delta_fn = (eta / np.pi) / ((omega_p - omega_sum)**2 + eta**2)

        rt = np.pi * np.sum(C2 * delta_fn)
        ac_2d = ac_mask_strict[np.newaxis, :] | ac_mask_strict[:, np.newaxis]
        ras = np.pi * np.sum(C2 * delta_fn * ac_2d)
        ac_both = ac_mask_strict[np.newaxis, :] & ac_mask_strict[:, np.newaxis]
        raa = np.pi * np.sum(C2 * delta_fn * ac_both)

        w = c2[p]
        c2_weights.append(w)
        rates['total'] += w * rt
        rates['ac_strict'] += w * ras
        rates['ac_ac'] += w * raa

    # Normalize by total weight
    total_weight = sum(c2_weights)
    for key in rates:
        if key != 'n_belt':
            rates[key] /= total_weight

    # c²-weighted belt frequency
    omega_belt = sum(w * omega[p] for w, p in zip(c2_weights, belt_calc)) / total_weight

    # Convert to dimensionless f
    scale = (k_L / ell0)**2 / omega_belt
    f_total = scale * rates['total']
    f_ac_strict = scale * rates['ac_strict']
    f_ac_ac = scale * rates['ac_ac']

    return {
        'omega_edge': omega_edge,
        'omega_belt': omega_belt,
        'f_total': f_total,
        'f_ac_strict': f_ac_strict,
        'f_ac_ac': f_ac_ac,
        'n_belt_modes': len(belt_calc),
        'omega': omega,
        'c2': c2, 'belt_calc': belt_calc,
        'H': H, 'evecs': evecs, 'ell0': ell0,
        'scale': scale, 'bloch': bloch, 'L': L,
    }


# =========================================================================
# Tests
# =========================================================================

def test_one_phonon_forbidden(fgr_data):
    """1-phonon belt → acoustic is forbidden by kinematic gap."""
    gap_ratio = fgr_data['omega_belt'] / fgr_data['omega_edge']
    print(f"ω_belt = {fgr_data['omega_belt']:.4f}, ω_edge = {fgr_data['omega_edge']:.4f}, ratio = {gap_ratio:.3f}")
    assert gap_ratio > 1.0, (
        f"No kinematic gap: ω_belt/ω_edge = {gap_ratio:.3f}"
    )


def test_ac_ac_negligible(fgr_data):
    """ac+ac channel is negligible (both final states acoustic).

    At Γ this is trivially zero: the only acoustic modes are translations
    with H = 0 (see module NOTES).
    """
    f_ac_ac = fgr_data['f_ac_ac']
    f_total = fgr_data['f_total']

    # ac+ac should be < 1% of total
    if f_total > 0:
        ratio = f_ac_ac / f_total
        print(f"f_ac_ac/f_total = {ratio:.6f}")
        assert ratio < 0.01, (
            f"ac+ac not negligible: {ratio:.4f} of total"
        )


def test_low_omega_fraction_small(fgr_data):
    """Low-ω (acoustic) fraction is small relative to total.

    At Γ this is trivially zero: see module NOTES.
    """
    f_ac = fgr_data['f_ac_strict']
    f_total = fgr_data['f_total']

    assert f_total > 0, "Total rate is zero"
    ratio = f_ac / f_total
    print(f"f_ac/f_total = {ratio:.3f} (acoustic fraction)")
    # Acoustic fraction should be < 30% — most emission is optical
    assert ratio < 0.30, (
        f"Acoustic fraction too large: {ratio:.3f}"
    )


def test_total_rate_positive(fgr_data):
    """Total 2-phonon rate is finite and positive."""
    f_total = fgr_data['f_total']
    print(f"f_total = {f_total:.4e}")
    assert f_total > 1e-8, (
        f"Total rate suspiciously small: f = {f_total:.2e}"
    )
    assert f_total < 1.0, (
        f"Total rate suspiciously large: f = {f_total:.2e}"
    )


def test_optical_dominates(fgr_data):
    """Optical emission dominates over acoustic emission.

    At Γ this is trivially true (f_ac = 0): see module NOTES.
    """
    f_ac = fgr_data['f_ac_strict']
    f_total = fgr_data['f_total']
    f_optical = f_total - f_ac

    ratio_str = f"{f_optical/f_ac:.1f}x" if f_ac > 0 else "inf (f_ac=0)"
    print(f"f_optical = {f_optical:.4e}, f_ac = {f_ac:.4e}, ratio = {ratio_str}")
    assert f_optical > 0.70 * f_total, (
        f"Optical does not dominate: f_opt/f_total = {f_optical/f_total:.3f}"
    )


def test_omega_edge_computed(fgr_data):
    """Acoustic ceiling is computed from BZ scan, not hardcoded."""
    omega_edge = fgr_data['omega_edge']
    print(f"ω_edge = {omega_edge:.4f} (computed from BZ scan)")
    assert 0.7 < omega_edge < 0.9, (
        f"omega_edge out of expected range: {omega_edge:.4f}"
    )


def test_rate_linear_in_eta(fgr_data):
    """f ∝ η (Lorentzian on discrete spectrum).

    On a finite system the Lorentzian δ-function broadening captures a
    number of (m,n) pairs proportional to η, so f/η = const. This is
    expected behavior, not a bug. The test verifies the FGR machinery is
    internally consistent across broadening values.
    """
    d = fgr_data
    omega, H, c2 = d['omega'], d['H'], d['c2']
    belt_calc, scale = d['belt_calc'], d['scale']
    c2_w = np.array([c2[p] for p in belt_calc])
    total_w = c2_w.sum()
    k_L = 2.0

    f_over_eta = []
    for eta in [0.02, 0.05, 0.10]:
        rate = 0.0
        for ip, p in enumerate(belt_calc):
            omega_p = omega[p]
            g_p = H[p, :]
            H_w = H * g_p[np.newaxis, :]
            C2 = (H_w @ H.T)**2
            omega_sum = omega[np.newaxis, :] + omega[:, np.newaxis]
            delta_fn = (eta / np.pi) / ((omega_p - omega_sum)**2 + eta**2)
            rate += c2_w[ip] * np.pi * np.sum(C2 * delta_fn)
        rate /= total_w
        f_val = scale * rate
        f_over_eta.append(f_val / eta)

    spread = (max(f_over_eta) - min(f_over_eta)) / np.mean(f_over_eta)
    print(f"f/η = {[f'{x:.4e}' for x in f_over_eta]}, spread = {spread:.4f}")
    assert spread < 0.05, (
        f"f/η not constant: spread = {spread:.4f}"
    )


def test_lifetime_estimate(fgr_data):
    """Lifetime τ = 1/(ε²·f·ω) at reference anharmonicity ε = 0.01.

    The total 2-phonon rate Γ/ω = ε² × f_total gives the belt mode
    lifetime in units of oscillation periods. At ε = 0.01 (weak cubic
    anharmonicity), τ > 10⁷ periods — the belt excitation is
    kinematically long-lived.

    NOTE: This is the TOTAL 2-phonon lifetime (optical + acoustic
    channels). At Γ the acoustic channel is trivially zero (see module
    NOTES), so this is entirely optical decay. The acoustic drag
    lifetime at finite k is longer (kinematic gap + dipolar suppression).
    """
    f_total = fgr_data['f_total']
    omega_belt = fgr_data['omega_belt']

    eps_ref = 0.01
    tau_ref = 1.0 / (eps_ref**2 * f_total)

    # Rate table
    print(f"\n  Rate table (Γ/ω = ε² × f, f_total = {f_total:.4e}):")
    print(f"  {'ε':>10s}  {'Γ/ω':>12s}  {'τ (periods)':>15s}")
    for eps in [0.001, 0.01, 0.03, 0.1]:
        gamma_over_w = eps**2 * f_total
        tau = 1.0 / gamma_over_w
        print(f"  {eps:10.3f}  {gamma_over_w:12.2e}  {tau:15.2e}")

    print(f"\n  At ε = {eps_ref}: τ = {tau_ref:.2e} oscillation periods")
    # Belt excitation should survive > 10^6 periods at weak anharmonicity
    assert tau_ref > 1e6, (
        f"Lifetime too short at ε={eps_ref}: τ = {tau_ref:.2e}"
    )


def test_belt_mode_coverage(fgr_data):
    """Belt source projects dominantly onto belt-frequency modes.

    > 95% of c² should be in modes with ω > ω_edge (belt region).
    Top 5 belt modes capture a substantial fraction (~37%).
    """
    d = fgr_data
    c2, omega = d['c2'], d['omega']
    omega_edge = d['omega_edge']

    c2_belt_region = c2[omega > omega_edge].sum()
    c2_total = c2.sum()
    belt_frac = c2_belt_region / c2_total

    c2_top5 = sum(c2[p] for p in d['belt_calc'])
    top5_frac = c2_top5 / c2_total

    print(f"c² in ω > ω_edge: {belt_frac:.4f}, top-5 belt: {top5_frac:.4f}")
    assert belt_frac > 0.95, (
        f"Belt source not dominated by belt modes: {belt_frac:.4f}"
    )
    assert top5_frac > 0.20, (
        f"Top-5 belt modes capture too little: {top5_frac:.4f}"
    )
