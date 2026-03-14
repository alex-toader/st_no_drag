#!/usr/bin/env python3
"""
Dipolar scaling: hop source radiates as k² not k⁰.

Tests:
  1. All Z12-Z12 hop sources have M₀ = 0 (monopole moment machine zero)
  2. Dipole moment M₁ is O(1) — channel exists but is suppressed
  3. Acoustic overlap scales as k² on [100], [110], [111] (power law fit)
  4. Constant-force source (monopole) gives k⁰ scaling (negative control)
  5. k² slope consistent across 5 pairs × 3 BZ directions (universality)
  6. Z16-Z16 hop sources have M₀ ≠ 0 (negative control)
  7. Z12-Z16 hop: M₀ ≠ 0, slope ≈ 0 (dipolar scaling is Z12-Z12 specific)
  8. k² slope = 2.0 at L_cell = 2, 4, 8 (geometric, not discretization)

The hop source S = F_belt(B) - F_belt(A) represents a particle moving
from cell A to cell B. M₀ = 0 means no net force → no monopole radiation.
The leading multipole is a dipole (M₁ ~ O(1)), giving acoustic overlap
~ k² instead of k⁰. Schematic emission rate scales as k⁴ vs k² for
monopole — suppression by (k₀)².

RAW OUTPUT (pytest -v -s, Mar 2026):
  test_monopole_zero_all_pairs    Z12-Z12 pairs: 48, max |M₀| = 1.71e-15
  test_dipole_nonzero             |M₁| min = 0.7787, mean = 1.6350
  test_hop_source_k_squared       [100]: slope = 2.00, [110]: slope = 2.00, [111]: slope = 2.00
  test_monopole_source_k_zero     [100]: slope = -0.00, [110]: slope = -0.00, [111]: slope = -0.00
  test_k_squared_multiple_pairs   [100]: 2.000×5, [110]: 2.000,2.001,2.000×3, [111]: 2.000×5
  test_z16_monopole_nonzero       Z16-Z16 pairs: 16, M₀ > 0.5: 14/16, max |M₀| = 1.0496
  test_z12_z16_hop_monopole       Z12-Z16 pairs: 96, |M₀| = 0.5248, slope = -0.0003
  test_dipolar_scaling_vs_L_cell  L=2: 2.000, L=4: 2.000, L=8: 2.000

References:
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/07_finite_k_radiation.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)
from core_math.analysis.no_drag import get_belt_vertex_forces
from physics.bloch import DisplacementBloch


def _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L):
    """Build hop source S = F_belt(B) - F_belt(A) on all vertices."""
    vf_a = get_belt_vertex_forces(ci_a, centers, v, f, cfi, L)
    vf_b = get_belt_vertex_forces(ci_b, centers, v, f, cfi, L)
    if vf_a is None or vf_b is None:
        return None, None

    V_count = len(v)
    all_verts = set(vf_a.keys()) | set(vf_b.keys())
    S_vec = np.zeros(3 * V_count)
    for vi in all_verts:
        si = vf_b.get(vi, np.zeros(3)) - vf_a.get(vi, np.zeros(3))
        S_vec[3 * vi:3 * vi + 3] = si
    return S_vec, all_verts


def _compute_multipole(S_vec, v, all_verts, L):
    """Compute monopole (M0) and dipole (M1) moments of source.

    NOTE: M₁ depends on origin r0 when M₀ ≠ 0 (shifts by outer(M₀, δ)).
    When M₀ = 0 (Z12-Z12 hop sources), M₁ is origin-independent.
    """
    verts_list = sorted(all_verts)  # deterministic ordering
    r_ref = v[verts_list[0]]
    unwrapped = []
    for vi in verts_list:
        delta = v[vi] - r_ref
        delta -= L * np.round(delta / L)
        unwrapped.append(r_ref + delta)
    r0 = np.mean(unwrapped, axis=0)

    M0 = np.zeros(3)
    M1 = np.zeros((3, 3))
    for vi in all_verts:
        si = S_vec[3 * vi:3 * vi + 3]
        ri = v[vi] - r0
        ri -= L * np.round(ri / L)
        M0 += si
        M1 += np.outer(si, ri)
    return M0, M1


def _acoustic_overlap(S_vec, bloch, k_vec, norm_S_sq):
    """Project source onto lowest 3 bands at given k.

    Returns normalized overlap: sum_lam |<S|u_lam>|^2 / ||S||^2.
    """
    D = bloch.build_dynamical_matrix(k_vec)
    w2, modes = np.linalg.eigh(D)
    order = np.argsort(w2)
    assert w2[order[0]] > -1e-10, (
        f"Soft mode detected: min eigenvalue = {w2[order[0]]:.2e}"
    )
    modes = modes[:, order]

    total = 0.0
    for branch in range(3):
        mode = modes[:, branch]
        total += np.abs(np.dot(S_vec, np.conj(mode)))**2
    return total / norm_S_sq if norm_S_sq > 0 else 0.0


@pytest.fixture(scope="module")
def dipolar_data():
    """Build C15, find Z12-Z12 pairs, compute multipoles and overlaps."""
    L_cell = 4.0
    v, e, f, cfi = build_c15_supercell_periodic(1, L_cell)
    centers = np.array(get_c15_points(1, L_cell))
    L = L_cell
    n_cells = len(cfi)
    cell_type = np.array([12 if len(cfi[ci]) == 12 else 16
                          for ci in range(n_cells)])

    # Find all Z12-Z12 neighbor pairs
    face_to_cells = {}
    for ci in range(n_cells):
        for f_idx, _ in cfi[ci]:
            face_to_cells.setdefault(f_idx, []).append(ci)

    pairs = []
    seen = set()
    for f_idx, cells in face_to_cells.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if cell_type[a] != 12 or cell_type[b] != 12:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    pairs.sort()

    # Multipole moments for all pairs
    M0_mags = []
    M1_norms = []
    for ci_a, ci_b in pairs:
        S_vec, all_verts = _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L)
        if S_vec is None:
            continue
        M0, M1 = _compute_multipole(S_vec, v, all_verts, L)
        M0_mags.append(np.linalg.norm(M0))
        M1_norms.append(np.linalg.norm(M1))

    # Z16-Z16 pairs (negative control: M₀ can be nonzero)
    z16_pairs = []
    z16_seen = set()
    for f_idx, cells in face_to_cells.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if cell_type[a] != 16 or cell_type[b] != 16:
            continue
        key = (min(a, b), max(a, b))
        if key in z16_seen:
            continue
        z16_seen.add(key)
        z16_pairs.append((a, b))
    z16_pairs.sort()

    z16_M0_mags = []
    for ci_a, ci_b in z16_pairs:
        S_vec, av = _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L)
        if S_vec is None:
            continue
        M0, _ = _compute_multipole(S_vec, v, av, L)
        z16_M0_mags.append(np.linalg.norm(M0))

    # Z12-Z16 mixed pairs (M₀ ≠ 0: Z16 contributes monopole)
    mixed_pairs = []
    mixed_seen = set()
    for f_idx, cells in face_to_cells.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if {cell_type[a], cell_type[b]} != {12, 16}:
            continue
        key = (min(a, b), max(a, b))
        if key in mixed_seen:
            continue
        mixed_seen.add(key)
        mixed_pairs.append((a, b))
    mixed_pairs.sort()

    mixed_M0_mags = []
    for ci_a, ci_b in mixed_pairs:
        S_vec, av = _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L)
        if S_vec is None:
            continue
        M0, _ = _compute_multipole(S_vec, v, av, L)
        mixed_M0_mags.append(np.linalg.norm(M0))

    # k-scaling on representative pair
    ci_a, ci_b = pairs[0]
    S_real, all_verts = _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L)
    norm_real = np.dot(S_real, S_real)

    # Constant-force source (monopole, M0 != 0)
    # _acoustic_overlap normalizes by norm_S_sq, so S_mono magnitude is irrelevant
    S_mono = np.zeros(3 * len(v))
    np.random.seed(123)
    F_mono = np.random.randn(3)
    F_mono /= np.linalg.norm(F_mono)
    for vi in all_verts:
        S_mono[3 * vi:3 * vi + 3] = F_mono
    norm_mono = np.dot(S_mono, S_mono)

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0)

    # Long-wavelength condition: k * ell0 << 1
    edge_lens = np.array([np.linalg.norm(
        (v[e[i][1]] - v[e[i][0]]) - L * np.round((v[e[i][1]] - v[e[i][0]]) / L)
    ) for i in range(len(e))])
    ell0 = edge_lens.mean()

    k_BZ = np.pi / L
    k_fracs = [0.005, 0.01, 0.02, 0.05]  # fit range: k <= 0.05 k_BZ
    assert max(k_fracs) * k_BZ * ell0 < 0.1, (
        f"Long-wavelength condition violated: max(k*ell0) = "
        f"{max(k_fracs) * k_BZ * ell0:.4f}"
    )

    k_dirs = {
        '[100]': np.array([1, 0, 0], dtype=float),
        '[110]': np.array([1, 1, 0], dtype=float) / np.sqrt(2),
        '[111]': np.array([1, 1, 1], dtype=float) / np.sqrt(3),
    }

    slopes_real = {}
    slopes_mono = {}
    for dir_name, k_hat in k_dirs.items():
        data_real = []
        data_mono = []
        for kf in k_fracs:
            k_vec = kf * k_BZ * k_hat
            data_real.append(_acoustic_overlap(S_real, bloch, k_vec, norm_real))
            data_mono.append(_acoustic_overlap(S_mono, bloch, k_vec, norm_mono))

        log_k = np.log(k_fracs)
        log_real = np.log([max(d, 1e-30) for d in data_real])
        log_mono = np.log([max(d, 1e-30) for d in data_mono])
        slopes_real[dir_name] = np.polyfit(log_k, log_real, 1)[0]
        slopes_mono[dir_name] = np.polyfit(log_k, log_mono, 1)[0]

    # k² slopes on [100] and [111] for multiple Z12-Z12 pairs
    log_k = np.log(k_fracs)
    multi_slopes = {}
    for dir_label in ['[100]', '[110]', '[111]']:
        k_hat = k_dirs[dir_label]
        slopes_dir = []
        for ci_a, ci_b in pairs[:5]:
            S_p, _ = _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L)
            if S_p is None:
                continue
            norm_p = np.dot(S_p, S_p)
            data_p = []
            for kf in k_fracs:
                k_vec = kf * k_BZ * k_hat
                data_p.append(_acoustic_overlap(S_p, bloch, k_vec, norm_p))
            log_p = np.log([max(d, 1e-30) for d in data_p])
            slopes_dir.append(np.polyfit(log_k, log_p, 1)[0])
        multi_slopes[dir_label] = np.array(slopes_dir)

    # Z12-Z16 slope on [100] (expect ~0: monopole radiation)
    ci_a_m, ci_b_m = mixed_pairs[0]
    S_mixed, _ = _build_hop_source(ci_a_m, ci_b_m, centers, v, f, cfi, L)
    norm_mixed = np.dot(S_mixed, S_mixed)
    data_mixed = []
    k_hat_100 = k_dirs['[100]']
    for kf in k_fracs:
        k_vec = kf * k_BZ * k_hat_100
        data_mixed.append(_acoustic_overlap(S_mixed, bloch, k_vec, norm_mixed))
    log_mixed = np.log([max(d, 1e-30) for d in data_mixed])
    slope_mixed = np.polyfit(log_k, log_mixed, 1)[0]

    return {
        'M0_mags': np.array(M0_mags),
        'M1_norms': np.array(M1_norms),
        'n_pairs': len(pairs),
        'slopes_real': slopes_real,
        'slopes_mono': slopes_mono,
        'z16_M0_mags': np.array(z16_M0_mags),
        'n_z16_pairs': len(z16_pairs),
        'multi_slopes': multi_slopes,
        'mixed_M0_mags': np.array(mixed_M0_mags),
        'n_mixed_pairs': len(mixed_pairs),
        'slope_mixed': slope_mixed,
    }


# =========================================================================
# Tests
# =========================================================================

def test_monopole_zero_all_pairs(dipolar_data):
    """M₀ = 0 (machine zero) on all Z12-Z12 hop sources."""
    M0_max = dipolar_data['M0_mags'].max()
    print(f"Z12-Z12 pairs: {dipolar_data['n_pairs']}, max |M₀| = {M0_max:.2e}")
    assert M0_max < 1e-12, (
        f"Hop source M₀ not zero: max |M₀| = {M0_max:.2e}"
    )


def test_dipole_nonzero(dipolar_data):
    """M₁ ~ O(1): dipole channel exists (leading multipole).

    NOTE: M₁ scales as L³ (force ~ L², displacement ~ L). Thresholds
    calibrated for L_cell = 4.0. At L_cell = 2.0, M₁ ~ 0.19.
    """
    M1_min = dipolar_data['M1_norms'].min()
    M1_mean = dipolar_data['M1_norms'].mean()
    print(f"|M₁| min = {M1_min:.4f}, mean = {M1_mean:.4f}")
    assert M1_min > 0.5, (
        f"Dipole too small: min |M₁| = {M1_min:.4f}"
    )
    assert M1_mean > 1.0, (
        f"Dipole mean too small: {M1_mean:.4f}"
    )


def test_hop_source_k_squared(dipolar_data):
    """Acoustic overlap scales as k² (dipolar) on all 3 BZ directions."""
    for dir_name, slope in dipolar_data['slopes_real'].items():
        print(f"Hop source {dir_name}: slope = {slope:.2f}")
        assert 1.8 < slope < 2.2, (
            f"Hop source slope on {dir_name}: {slope:.2f}, "
            f"expected ~2.0 (dipolar)"
        )


def test_monopole_source_k_zero(dipolar_data):
    """Constant-force source (M₀ != 0) gives k⁰ scaling (flat)."""
    for dir_name, slope in dipolar_data['slopes_mono'].items():
        print(f"Monopole {dir_name}: slope = {slope:.2f}")
        assert abs(slope) < 0.3, (
            f"Monopole source slope on {dir_name}: {slope:.2f}, "
            f"expected ~0.0 (flat)"
        )


def test_k_squared_multiple_pairs(dipolar_data):
    """k² slope consistent across multiple Z12-Z12 pairs on all 3 BZ dirs."""
    for dir_label, slopes in dipolar_data['multi_slopes'].items():
        print(f"Z12-Z12 {dir_label} slopes ({len(slopes)} pairs): "
              f"{', '.join(f'{s:.3f}' for s in slopes)}")
        for i, s in enumerate(slopes):
            assert 1.8 < s < 2.2, (
                f"Pair {i} slope on {dir_label} = {s:.3f}, expected ~2.0"
            )


def test_z16_monopole_nonzero(dipolar_data):
    """Z16-Z16 hop sources: M₀ not always zero (negative control)."""
    M0 = dipolar_data['z16_M0_mags']
    n_nonzero = np.sum(M0 > 0.5)
    print(f"Z16-Z16 pairs: {dipolar_data['n_z16_pairs']}, "
          f"M₀ > 0.5: {n_nonzero}/{len(M0)}, "
          f"max |M₀| = {M0.max():.4f}")
    assert n_nonzero > 0, (
        f"No Z16-Z16 pair with M₀ > 0.5, max = {M0.max():.4f}"
    )


def test_z12_z16_hop_monopole(dipolar_data):
    """Z12-Z16 hop: M₀ ≠ 0 and slope ≈ 0 (monopole radiation, not dipolar)."""
    M0 = dipolar_data['mixed_M0_mags']
    slope = dipolar_data['slope_mixed']
    M0_mean = M0.mean()
    M0_spread = (M0.max() - M0.min()) / M0_mean if M0_mean > 0 else 0
    print(f"Z12-Z16 pairs: {dipolar_data['n_mixed_pairs']}, "
          f"|M₀| mean = {M0_mean:.4f}, spread = {100*M0_spread:.1f}%, "
          f"[100] slope = {slope:.4f}")
    assert M0_mean > 0.3, (
        f"Z12-Z16 M₀ too small: mean = {M0_mean:.4f}"
    )
    assert abs(slope) < 0.3, (
        f"Z12-Z16 slope = {slope:.4f}, expected ~0 (monopole)"
    )


def test_dipolar_scaling_vs_L_cell():
    """k² slope = 2.0 at L_cell = 2, 4, 8 (geometric, not discretization)."""
    k_fracs = [0.005, 0.01, 0.02, 0.05]
    log_k = np.log(k_fracs)
    k_hat = np.array([1, 0, 0], dtype=float)

    slopes = {}
    for L_cell in [2.0, 4.0, 8.0]:
        v, e, f, cfi = build_c15_supercell_periodic(1, L_cell)
        centers = np.array(get_c15_points(1, L_cell))
        L = L_cell
        n_cells = len(cfi)
        cell_type = np.array([12 if len(cfi[ci]) == 12 else 16
                              for ci in range(n_cells)])

        # Find first Z12-Z12 pair
        face_to_cells = {}
        for ci in range(n_cells):
            for f_idx, _ in cfi[ci]:
                face_to_cells.setdefault(f_idx, []).append(ci)
        pair = None
        for f_idx, cells in face_to_cells.items():
            if len(cells) == 2:
                a, b = cells
                if cell_type[a] == 12 and cell_type[b] == 12:
                    pair = (a, b)
                    break

        S_vec, _ = _build_hop_source(pair[0], pair[1], centers, v, f, cfi, L)
        norm_sq = np.dot(S_vec, S_vec)
        bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0)
        k_BZ = np.pi / L

        data = []
        for kf in k_fracs:
            data.append(_acoustic_overlap(S_vec, bloch, kf * k_BZ * k_hat, norm_sq))
        log_d = np.log([max(d, 1e-30) for d in data])
        slopes[L_cell] = np.polyfit(log_k, log_d, 1)[0]

    for L_cell, s in slopes.items():
        print(f"L_cell={L_cell}: slope = {s:.4f}")
        assert 1.8 < s < 2.2, (
            f"L_cell={L_cell}: slope = {s:.4f}, expected ~2.0"
        )
