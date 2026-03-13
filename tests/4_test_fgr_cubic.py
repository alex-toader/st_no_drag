#!/usr/bin/env python3
"""
Fermi's Golden Rule: perturbative acoustic emission rate.

Tests:
  1. 1-phonon emission is forbidden (kinematic gap)
  2. ac+ac channel is negligible
  3. Low-ω fraction is small (< 30% of total)
  4. Total rate is finite and positive (mechanism works)
  5. Rate structure: optical dominates over acoustic

With cubic anharmonicity V₃ = (α/6) Σ_edges (δr)³, a belt mode p
can decay into two phonons via:
  Γ_p = π α² Σ_{m,n} |V_{mnp}|² δ(ω_p - ω_m - ω_n)

The rate is expressed as Γ/ω = ε² f, where ε = α·ℓ₀/k_L and f is
a dimensionless factor decomposed by acoustic content.

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
from core_math.analysis.no_drag import build_enriched_belt_vectors
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
    edges_list = [list(edge) for edge in e]
    n_edges = len(edges_list)

    # Diagonalize D(k=0)
    bloch = DisplacementBloch(v, edges_list, L, k_L=2.0, k_T=1.0, mass=1.0)
    D0 = np.real(bloch.build_dynamical_matrix(np.zeros(3)))
    evals, evecs = np.linalg.eigh(D0)
    evecs = np.real(evecs)
    omega = np.sqrt(np.maximum(evals, 0))

    # Belt source
    box_center = np.array([L/2, L/2, L/2])
    dists_to_center = np.linalg.norm(centers - box_center, axis=1)
    source_cell = np.argmin(dists_to_center)
    _, old_vec = build_enriched_belt_vectors(
        source_cell, centers, v, f, cfi, L)
    u0 = np.real(old_vec)
    u0 = u0 / np.linalg.norm(u0)

    coeffs = evecs.T @ u0
    c2 = np.abs(coeffs)**2
    order = np.argsort(c2)[::-1]
    belt_modes = [m for m in order[:20] if omega[m] > 1.0]

    # Edge geometry
    v_arr = np.array(v)
    idx_i = np.array([edge[0] for edge in edges_list])
    idx_j = np.array([edge[1] for edge in edges_list])
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

    # Acoustic ceiling (Kelvin primitive-cell value)
    omega_edge = 0.8022

    # Frequency masks
    ac_mask_strict = omega < omega_edge

    # FGR for top 5 belt modes (or fewer if not enough)
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
    }


# =========================================================================
# Tests
# =========================================================================

def test_one_phonon_forbidden(fgr_data):
    """1-phonon belt → acoustic is forbidden by kinematic gap."""
    gap_ratio = fgr_data['omega_belt'] / fgr_data['omega_edge']
    assert gap_ratio > 1.0, (
        f"No kinematic gap: ω_belt/ω_edge = {gap_ratio:.3f}"
    )


def test_ac_ac_negligible(fgr_data):
    """ac+ac channel is negligible (both final states acoustic)."""
    f_ac_ac = fgr_data['f_ac_ac']
    f_total = fgr_data['f_total']

    # ac+ac should be < 1% of total
    if f_total > 0:
        ratio = f_ac_ac / f_total
        assert ratio < 0.01, (
            f"ac+ac not negligible: {ratio:.4f} of total"
        )


def test_low_omega_fraction_small(fgr_data):
    """Low-ω (acoustic) fraction is small relative to total."""
    f_ac = fgr_data['f_ac_strict']
    f_total = fgr_data['f_total']

    assert f_total > 0, "Total rate is zero"
    ratio = f_ac / f_total
    # Acoustic fraction should be < 30% — most emission is optical
    assert ratio < 0.30, (
        f"Acoustic fraction too large: {ratio:.3f}"
    )


def test_total_rate_positive(fgr_data):
    """Total 2-phonon rate is finite and positive."""
    f_total = fgr_data['f_total']
    assert f_total > 1e-8, (
        f"Total rate suspiciously small: f = {f_total:.2e}"
    )
    assert f_total < 1.0, (
        f"Total rate suspiciously large: f = {f_total:.2e}"
    )


def test_optical_dominates(fgr_data):
    """Optical emission dominates over acoustic emission."""
    f_ac = fgr_data['f_ac_strict']
    f_total = fgr_data['f_total']
    f_optical = f_total - f_ac

    assert f_optical > f_ac, (
        f"Optical does not dominate: f_opt = {f_optical:.2e}, "
        f"f_ac = {f_ac:.2e}"
    )
