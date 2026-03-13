#!/usr/bin/env python3
"""
Dipolar scaling: hop source radiates as k² not k⁰.

Tests:
  1. All Z12-Z12 hop sources have M₀ = 0 (monopole moment machine zero)
  2. Dipole moment M₁ is O(1) — channel exists but is suppressed
  3. Acoustic overlap scales as k² on [100], [110], [111] (power law fit)
  4. Constant-force source (monopole) gives k⁰ scaling (negative control)

The hop source S = F_belt(B) - F_belt(A) represents a particle moving
from cell A to cell B. M₀ = 0 means no net force → no monopole radiation.
The leading multipole is a dipole (M₁ ~ O(1)), giving acoustic overlap
~ k² instead of k⁰. Schematic emission rate scales as k⁴ vs k² for
monopole — suppression by (k₀)².

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
    """Compute monopole (M0) and dipole (M1) moments of source."""
    verts_list = list(all_verts)
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

    # k-scaling on representative pair
    ci_a, ci_b = pairs[0]
    S_real, all_verts = _build_hop_source(ci_a, ci_b, centers, v, f, cfi, L)
    norm_real = np.dot(S_real, S_real)

    # Constant-force source (monopole, M0 != 0)
    S_mono = np.zeros(3 * len(v))
    np.random.seed(123)
    F_mono = np.random.randn(3)
    F_mono /= np.linalg.norm(F_mono)
    for vi in all_verts:
        S_mono[3 * vi:3 * vi + 3] = F_mono
    S_mono *= np.sqrt(norm_real / np.dot(S_mono, S_mono))
    norm_mono = np.dot(S_mono, S_mono)

    edges = [list(edge) for edge in e]
    bloch = DisplacementBloch(v, edges, L, k_L=2.0, k_T=1.0)

    k_BZ = np.pi / L
    k_fracs = [0.005, 0.01, 0.02, 0.05]  # fit range: k <= 0.05 k_BZ

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

    return {
        'M0_mags': np.array(M0_mags),
        'M1_norms': np.array(M1_norms),
        'n_pairs': len(pairs),
        'slopes_real': slopes_real,
        'slopes_mono': slopes_mono,
    }


# =========================================================================
# Tests
# =========================================================================

def test_monopole_zero_all_pairs(dipolar_data):
    """M₀ = 0 (machine zero) on all Z12-Z12 hop sources."""
    M0_max = dipolar_data['M0_mags'].max()
    assert M0_max < 1e-12, (
        f"Hop source M₀ not zero: max |M₀| = {M0_max:.2e}"
    )


def test_dipole_nonzero(dipolar_data):
    """M₁ ~ O(1): dipole channel exists (leading multipole)."""
    M1_min = dipolar_data['M1_norms'].min()
    M1_mean = dipolar_data['M1_norms'].mean()
    assert M1_min > 0.5, (
        f"Dipole too small: min |M₁| = {M1_min:.4f}"
    )
    assert M1_mean > 1.0, (
        f"Dipole mean too small: {M1_mean:.4f}"
    )


def test_hop_source_k_squared(dipolar_data):
    """Acoustic overlap scales as k² (dipolar) on all 3 BZ directions."""
    for dir_name, slope in dipolar_data['slopes_real'].items():
        assert 1.8 < slope < 2.2, (
            f"Hop source slope on {dir_name}: {slope:.2f}, "
            f"expected ~2.0 (dipolar)"
        )


def test_monopole_source_k_zero(dipolar_data):
    """Constant-force source (M₀ != 0) gives k⁰ scaling (flat)."""
    for dir_name, slope in dipolar_data['slopes_mono'].items():
        assert abs(slope) < 0.3, (
            f"Monopole source slope on {dir_name}: {slope:.2f}, "
            f"expected ~0.0 (flat)"
        )
