#!/usr/bin/env python3
"""
Selection rule M₀ = 0 for belt m=2 mode.

Tests:
  1. C15: M₀ = 0 on all 16 Z12 cells (exact by symmetry)
  2. C15: M₀ ≠ 0 on Z16 cells (negative control)
  3. Kelvin: M₀ = 0 on all cells (all identical truncated octahedra)
  4. Chain complex d₁d₀ = 0 does not explain M₀ = 0 (negative control)

The selection rule M₀ = Σ_j P_j A_j n̂_j is the net force from m=2
belt pressure cos(2θ) distributed over belt faces. M₀ = 0 means no
monopole coupling between the belt mode and uniform translations.

MECHANISM (verified in tests below):
  Z12 belt (N=6): n_ax[i] = -n_ax[i+N/2] (antipodal tilt symmetry).
    → Fourier spectrum n_ax has only odd harmonics (m=1,3,5).
    → cos(mθ) pressure at even m cannot beat with odd n_ax → M₀ = 0.
    → Protected modes: all even m (including m=2, the belt mode).
  Z16 belt (N=8): n_ax[i] = +n_ax[i+N/2] (symmetric, not antipodal).
    → n_ax has even harmonics (m=2,6) → cos(2θ) beats → M₀ ≠ 0.
  Kelvin belt (N=6): n_ax ≡ 0 (flat belt, all normals ⊥ belt axis).
    → Oh includes inversion → all even m protected (same as Z12).
    → Odd m NOT protected (radial forces survive despite flat belt).

NOTES:
  - cell_type is determined by face count (12 or 16 faces). This is a
    geometric property of C15 Voronoi, not an approximation. Cross-checked
    by belt size (Z12→N=6, Z16→N=8) in test_c15_belt_counts.
  - Circuit ordering from find_best_belt is deterministic (verified).
    The i↔i+N/2 antipodal pairing relies on this cyclic ordering.

RAW OUTPUT (pytest -v -s, Mar 2026):
  test_c15_z12_selection_rule       Z12 cells: 16, max |M₀| = 8.75e-17
  test_c15_z16_nonzero              Z16 cells: 8, |M₀| range = [0.0328, 0.0328]
  test_c15_z12_no_m2_normal_tilt    Z12 max |n_ax[m=2]| = 7.58e-16, Z16 min = 1.3615
  test_kelvin_selection_rule        Kelvin cells: 16, max |M₀| = 1.78e-15
  test_c15_belt_counts              Belt sizes: Z12 = {6}, Z16 = {8}
  test_c15_z16_m0_scales_as_area    M₀(L=2)/M₀(L=1) = 4.00, expected L² = 4.0
  test_c15_z12_all_m_selection_rule m=1: 1.49e-01, m=2: 8.75e-17, m=3: 6.91e-02,
                                     m=4: 2.74e-16, m=5: 1.35e-01 → even m protected
  test_c15_z12_antipodal_tilt       max |n_ax[i] + n_ax[i+N/2]| = 8.33e-16
  test_c15_z16_symmetric_tilt       max |n_ax[i] - n_ax[i+N/2]| = 6.11e-16
  test_kelvin_flat_belt             max |n_ax| = 7.28e-17 (flat belt → all m protected)
  test_c15_m0_linear_in_jitter      M₀/δ ratios: [1.78, 1.79, 1.85], spread = 3.7%
  test_chain_complex_does_not_explain_m0  d₁d₀=0 universal; Z12 ||D₀u₀||=1.059, Z16=1.024;
                                     M₀ differs by 15 orders, edge ext ratio = 1.034

References:
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/05_belt_phonon_coupling.py
  physics_ai/ST_11/wip/w_15_kelvin_no_drag/01_kelvin_no_drag.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)
from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.analysis.no_drag import (
    compute_selection_rule, build_enriched_belt_vectors
)
from core_math.operators.incidence import build_d0, build_d1


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def c15_mesh():
    """Build C15 N=1 supercell (24 cells: 16 Z12 + 8 Z16)."""
    N, L_cell = 1, 1.0
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = np.array(get_c15_points(N, L_cell))
    L = N * L_cell
    n_cells = len(cfi)
    cell_type = np.array([12 if len(cfi[ci]) == 12 else 16
                          for ci in range(n_cells)])
    return v, e, f, cfi, centers, L, cell_type


@pytest.fixture(scope="module")
def kelvin_mesh():
    """Build Kelvin N=2 supercell (16 identical truncated octahedra)."""
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4.0 * N
    return v, f, cfi, centers, L


# =========================================================================
# Tests
# =========================================================================

def test_c15_z12_selection_rule(c15_mesh):
    """M₀ = 0 exactly on all 16 Z12 cells."""
    v, e, f, cfi, centers, L, cell_type = c15_mesh

    z12_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    assert len(z12_cells) == 16, f"Expected 16 Z12 cells, got {len(z12_cells)}"

    m0_values = []
    for ci in z12_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Z12 cell {ci} has no belt"
        m0_values.append(sr['M0_mag'])

    m0_max = max(m0_values)
    print(f"Z12 cells: {len(z12_cells)}, max |M₀| = {m0_max:.2e}")
    assert m0_max < 1e-12, (
        f"M₀ not zero on Z12: max |M₀| = {m0_max:.2e}"
    )


def test_c15_z16_nonzero(c15_mesh):
    """M₀ ≠ 0 on Z16 cells (negative control)."""
    v, e, f, cfi, centers, L, cell_type = c15_mesh

    z16_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 16]
    assert len(z16_cells) == 8, f"Expected 8 Z16 cells, got {len(z16_cells)}"

    m0_values = []
    for ci in z16_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Z16 cell {ci} has no belt"
        m0_values.append(sr['M0_mag'])

    m0_min = min(m0_values)
    m0_max = max(m0_values)
    print(f"Z16 cells: {len(z16_cells)}, |M₀| range = [{m0_min:.4f}, {m0_max:.4f}]")
    # At L_cell=1.0, Z16 M₀ ≈ 0.033 (scales as L_cell²)
    assert m0_min > 0.01, (
        f"Z16 M₀ should be nonzero: min |M₀| = {m0_min:.6f}"
    )


def test_c15_z12_no_m2_normal_tilt(c15_mesh):
    """Z12 normal-tilt Fourier spectrum has no m=2 component.

    This is WHY M₀ = 0: the n_ax spectrum at m=2 is zero, so
    cos(2θ) pressure cannot beat with normal tilt to produce force.
    On Z16, n_ax[m=2] is O(1) — confirming the mechanism.
    """
    v, e, f, cfi, centers, L, cell_type = c15_mesh

    z12_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    z16_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 16]

    max_m2_z12 = 0.0
    for ci in z12_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        max_m2_z12 = max(max_m2_z12, abs(sr['n_ax_spectrum'][2]))

    min_m2_z16 = np.inf
    for ci in z16_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        min_m2_z16 = min(min_m2_z16, abs(sr['n_ax_spectrum'][2]))

    print(f"Z12 max |n_ax[m=2]| = {max_m2_z12:.2e}, Z16 min |n_ax[m=2]| = {min_m2_z16:.4f}")
    assert max_m2_z12 < 1e-12, (
        f"Z12 should have no m=2 normal tilt: max |n_ax[2]| = {max_m2_z12:.2e}"
    )
    assert min_m2_z16 > 0.1, (
        f"Z16 should have m=2 normal tilt: min |n_ax[2]| = {min_m2_z16:.4f}"
    )


def test_kelvin_selection_rule(kelvin_mesh):
    """M₀ = 0 on all 16 Kelvin cells."""
    v, f, cfi, centers, L = kelvin_mesh
    n_cells = len(centers)

    n_belt = 0
    m0_values = []
    for ci in range(n_cells):
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        if sr is not None:
            n_belt += 1
            m0_values.append(sr['M0_mag'])

    assert n_belt == n_cells, (
        f"Not all Kelvin cells have belts: {n_belt}/{n_cells}"
    )

    m0_max = max(m0_values)
    print(f"Kelvin cells: {n_belt}, max |M₀| = {m0_max:.2e}")
    assert m0_max < 1e-12, (
        f"M₀ not zero on Kelvin: max |M₀| = {m0_max:.2e}"
    )


def test_c15_belt_counts(c15_mesh):
    """All 24 C15 cells have belts. Z12 has N=6, Z16 has N=8."""
    v, e, f, cfi, centers, L, cell_type = c15_mesh

    belt_n = {}
    for ci in range(len(centers)):
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Cell {ci} has no belt"
        belt_n[ci] = sr['n_belt']

    z12_n = set(belt_n[ci] for ci in range(len(centers))
                if cell_type[ci] == 12)
    z16_n = set(belt_n[ci] for ci in range(len(centers))
                if cell_type[ci] == 16)

    print(f"Belt sizes: Z12 = {z12_n}, Z16 = {z16_n}")
    assert z12_n == {6}, f"Z12 belt sizes: {z12_n} (expected {{6}})"
    assert z16_n == {8}, f"Z16 belt sizes: {z16_n} (expected {{8}})"


def test_c15_z16_m0_scales_as_area(c15_mesh):
    """Z16 M₀ scales as L_cell² (dimensional check: force = pressure × area)."""
    L_cell_2 = 2.0
    v2, e2, f2, cfi2 = build_c15_supercell_periodic(1, L_cell_2)
    centers2 = np.array(get_c15_points(1, L_cell_2))
    L2 = L_cell_2
    cell_type2 = np.array([12 if len(cfi2[ci]) == 12 else 16
                           for ci in range(len(cfi2))])

    z16_2 = [ci for ci in range(len(centers2)) if cell_type2[ci] == 16]

    m0_L1 = []
    m0_L2 = []
    v1, e1, f1, cfi1, centers1, L1, cell_type1 = c15_mesh
    for ci in [ci for ci in range(len(centers1)) if cell_type1[ci] == 16]:
        sr = compute_selection_rule(ci, centers1, v1, f1, cfi1, L1)
        m0_L1.append(sr['M0_mag'])
    for ci in z16_2:
        sr = compute_selection_rule(ci, centers2, v2, f2, cfi2, L2)
        m0_L2.append(sr['M0_mag'])

    ratio = np.mean(m0_L2) / np.mean(m0_L1)
    expected = (L_cell_2 / 1.0)**2
    print(f"M₀(L=2)/M₀(L=1) = {ratio:.2f}, expected L² = {expected:.1f}")
    assert abs(ratio - expected) / expected < 0.05, (
        f"M₀ scaling: ratio = {ratio:.3f}, expected {expected:.1f}"
    )


def test_c15_z12_all_m_selection_rule(c15_mesh):
    """M₀ = 0 on Z12 for ALL azimuthal modes m, not just m=2.

    Computes net force for cos(m·θ) pressure at m=1,2,3,4,5.
    If M₀=0 for all m, the protection is stronger than the m=2 claim.
    """
    v, e, f, cfi, centers, L, cell_type = c15_mesh

    z12_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    from core_math.analysis.no_drag import _belt_geometry, geometric_normal

    m_values = [1, 2, 3, 4, 5]
    max_m0_per_m = {}

    for m in m_values:
        max_m0 = 0.0
        for ci in z12_cells:
            bg = _belt_geometry(ci, centers, v, f, cfi, L)
            circuit, theta, normals, areas, bn, fd = bg
            N = len(circuit)
            pressure = np.cos(m * theta)
            M0_vec = np.zeros(3)
            for idx in range(N):
                M0_vec += pressure[idx] * areas[idx] * normals[idx]
            max_m0 = max(max_m0, np.linalg.norm(M0_vec))
        max_m0_per_m[m] = max_m0

    for m, val in max_m0_per_m.items():
        print(f"Z12 m={m}: max |M₀| = {val:.2e}")

    # m=2 must be zero (paper claim)
    assert max_m0_per_m[2] < 1e-12, f"m=2 failed: {max_m0_per_m[2]:.2e}"
    # Even m protected, odd m not (antipodal symmetry of n_ax)
    for m in [2, 4]:
        assert max_m0_per_m[m] < 1e-12, f"m={m} should be protected: {max_m0_per_m[m]:.2e}"
    for m in [1, 3, 5]:
        assert max_m0_per_m[m] > 0.01, f"m={m} should NOT be protected: {max_m0_per_m[m]:.2e}"


def test_c15_z12_antipodal_tilt(c15_mesh):
    """Z12: n_ax[i] = -n_ax[i + N/2] (antipodal symmetry in circuit order).

    The normal axial tilt is odd under half-circuit rotation.
    This is WHY even-m modes are protected: cos(mθ) is even under
    π-rotation for even m, so ∫ cos(mθ) · n_ax(θ) = 0 by parity.
    """
    v, e, f, cfi, centers, L, cell_type = c15_mesh
    from core_math.analysis.no_drag import _belt_geometry

    z12_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    max_sum = 0.0

    for ci in z12_cells:
        bg = _belt_geometry(ci, centers, v, f, cfi, L)
        circuit, theta, normals, areas, bn, fd = bg
        N = len(circuit)
        n_ax = np.array([np.dot(normals[i], bn) for i in range(N)])

        for i in range(N // 2):
            j = i + N // 2
            max_sum = max(max_sum, abs(n_ax[i] + n_ax[j]))

    print(f"Z12 antipodal: max |n_ax[i] + n_ax[i+N/2]| = {max_sum:.2e}")
    assert max_sum < 1e-12, (
        f"Z12 antipodal symmetry broken: {max_sum:.2e}"
    )


def test_c15_z16_symmetric_tilt(c15_mesh):
    """Z16: n_ax[i] = +n_ax[i + N/2] (symmetric, NOT antipodal).

    The normal axial tilt is even under half-circuit rotation → even-m
    Fourier components are nonzero → cos(2θ) beats with n_ax → M₀ ≠ 0.
    """
    v, e, f, cfi, centers, L, cell_type = c15_mesh
    from core_math.analysis.no_drag import _belt_geometry

    z16_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 16]
    max_diff = 0.0

    for ci in z16_cells:
        bg = _belt_geometry(ci, centers, v, f, cfi, L)
        circuit, theta, normals, areas, bn, fd = bg
        N = len(circuit)
        n_ax = np.array([np.dot(normals[i], bn) for i in range(N)])

        for i in range(N // 2):
            j = i + N // 2
            max_diff = max(max_diff, abs(n_ax[i] - n_ax[j]))

    print(f"Z16 symmetric: max |n_ax[i] - n_ax[i+N/2]| = {max_diff:.2e}")
    assert max_diff < 1e-12, (
        f"Z16 symmetric tilt broken: {max_diff:.2e}"
    )


def test_kelvin_flat_belt(kelvin_mesh):
    """Kelvin: n_ax ≡ 0 (belt is flat, all normals perpendicular to belt axis).

    Oh includes inversion, so all even m are protected (same mechanism
    as Z12). The flat belt is a consequence of the higher Oh symmetry.
    Odd m (1,3,5) are NOT protected: their radial forces survive.
    """
    v, f, cfi, centers, L = kelvin_mesh
    from core_math.analysis.no_drag import _belt_geometry

    max_n_ax = 0.0
    for ci in range(len(centers)):
        bg = _belt_geometry(ci, centers, v, f, cfi, L)
        if bg is None:
            continue
        circuit, theta, normals, areas, bn, fd = bg
        N = len(circuit)
        n_ax = np.array([np.dot(normals[i], bn) for i in range(N)])
        max_n_ax = max(max_n_ax, np.max(np.abs(n_ax)))

    print(f"Kelvin max |n_ax| = {max_n_ax:.2e} (flat belt)")
    assert max_n_ax < 1e-12, (
        f"Kelvin belt not flat: max |n_ax| = {max_n_ax:.2e}"
    )


def test_kelvin_m_scan(kelvin_mesh):
    """Kelvin per-m scan: even m protected (Oh inversion), odd m not.

    M0 has two components: axial (along belt axis) and radial (in belt
    plane). Flat belt (n_ax=0) kills the axial component for all m.
    Inversion (Oh ⊃ Ci) kills the radial component at even m only:
      n_{j+N/2} = -n_j  and  cos(m*(theta+pi)) = (-1)^m cos(m*theta)
      Even m: radial forces cancel in pairs => M0 = 0
      Odd m: radial forces reinforce => M0 != 0
    """
    v, f, cfi, centers, L = kelvin_mesh
    from core_math.analysis.no_drag import _belt_geometry

    m_values = [1, 2, 3, 4, 5]
    max_m0_per_m = {}

    for m in m_values:
        max_m0 = 0.0
        for ci in range(len(centers)):
            bg = _belt_geometry(ci, centers, v, f, cfi, L)
            if bg is None:
                continue
            circuit, theta, normals, areas, bn, fd = bg
            N = len(circuit)
            pressure = np.cos(m * theta)
            M0_vec = np.zeros(3)
            for idx in range(N):
                M0_vec += pressure[idx] * areas[idx] * normals[idx]
            max_m0 = max(max_m0, np.linalg.norm(M0_vec))
        max_m0_per_m[m] = max_m0

    for m, val in max_m0_per_m.items():
        status = "= 0 (protected)" if val < 1e-12 else "> 0 (NOT protected)"
        print(f"Kelvin m={m}: max |M₀| = {val:.2e}  {status}")

    # Even m protected by Oh inversion
    assert max_m0_per_m[2] < 1e-12, (
        f"Kelvin m=2 should be protected: {max_m0_per_m[2]:.2e}"
    )
    assert max_m0_per_m[4] < 1e-12, (
        f"Kelvin m=4 should be protected: {max_m0_per_m[4]:.2e}"
    )
    # Odd m NOT protected (radial forces survive despite flat belt)
    assert max_m0_per_m[1] > 0.1, (
        f"Kelvin m=1 should NOT be protected: {max_m0_per_m[1]:.2e}"
    )
    assert max_m0_per_m[3] > 0.1, (
        f"Kelvin m=3 should NOT be protected: {max_m0_per_m[3]:.2e}"
    )


def test_c15_m0_linear_in_jitter():
    """M₀ scales linearly with geometric jitter δ (perturbative protection).

    Selection rule is symmetry-exact (M₀=0 at δ=0) with linear degradation
    M₀ ∝ δ. The coefficient M₀/δ ≈ 1.78 is O(1) and constant across 3
    decades of δ. This confirms the protection is by symmetry, not topology.

    N=6 belt on Z12: m=0..5 are the complete Fourier modes (m≥6 aliases
    to m mod 6 by Nyquist).
    """
    N, L_cell = 1, 1.0
    points_exact = get_c15_points(N, L_cell)
    L = N * L_cell

    np.random.seed(42)
    noise = np.random.randn(*points_exact.shape)

    deltas = [1e-4, 1e-3, 1e-2]
    ratios = []

    for delta in deltas:
        points_j = points_exact + delta * noise
        vj, ej, fj, cfij = build_c15_supercell_periodic(N, L_cell, points=points_j)
        centersj = np.array(points_j)
        ctj = np.array([12 if len(cfij[ci]) == 12 else 16
                         for ci in range(len(cfij))])
        z12_j = [ci for ci in range(len(centersj)) if ctj[ci] == 12]

        m0_vals = []
        for ci in z12_j:
            sr = compute_selection_rule(ci, centersj, vj, fj, cfij, L)
            if sr is not None:
                m0_vals.append(sr['M0_mag'])

        m0_max = max(m0_vals)
        ratios.append(m0_max / delta)

    ratios = np.array(ratios)
    spread = (ratios.max() - ratios.min()) / ratios.mean()
    print(f"M₀/δ ratios: {ratios}, spread = {100*spread:.1f}%")
    # Linear scaling: M₀/δ should be constant (spread < 10%)
    assert spread < 0.10, (
        f"M₀ not linear in δ: spread = {100*spread:.1f}%"
    )
    # Coefficient is O(1)
    assert 0.5 < ratios.mean() < 5.0, (
        f"M₀/δ coefficient unexpected: {ratios.mean():.2f}"
    )


def test_c15_z16_m4_jitter():
    """Z16 m=4 protection is by S4 symmetry, not Nyquist aliasing.

    Under vertex jitter δ, M₀(m=4) grows linearly (symmetry-broken).
    Nyquist aliasing would survive perturbation. The thetas are NOT
    equally spaced (55.6° and 34.4° alternating), confirming this is
    genuine S4 protection.

    Coefficient M₀/δ ≈ 3.8 at small δ, comparable to Z12 m=2 (≈1.8).
    """
    from core_math.analysis.no_drag import _belt_geometry

    N, L_cell = 1, 1.0
    points_exact = get_c15_points(N, L_cell)
    L = N * L_cell

    np.random.seed(42)
    noise = np.random.randn(*points_exact.shape)

    deltas = [1e-4, 5e-4, 1e-3]
    ratios = []

    for delta in deltas:
        points_j = points_exact + delta * noise
        vj, ej, fj, cfij = build_c15_supercell_periodic(N, L_cell, points=points_j)
        centersj = np.array(points_j)
        ctj = np.array([12 if len(cfij[ci]) == 12 else 16
                         for ci in range(len(cfij))])
        z16_j = [ci for ci in range(len(centersj)) if ctj[ci] == 16]

        m0_vals = []
        for ci in z16_j:
            bg = _belt_geometry(ci, centersj, vj, fj, cfij, L)
            circuit, theta, normals, areas, bn, fd = bg
            pressure = np.cos(4 * theta)
            M0_vec = np.zeros(3)
            for idx in range(len(circuit)):
                M0_vec += pressure[idx] * areas[idx] * normals[idx]
            m0_vals.append(np.linalg.norm(M0_vec))

        m0_max = max(m0_vals)
        ratios.append(m0_max / delta)

    ratios = np.array(ratios)
    spread = (ratios.max() - ratios.min()) / ratios.mean()
    print(f"Z16 m=4 M₀/δ ratios: {ratios}, spread = {100*spread:.1f}%")
    # Linear scaling: M₀/δ should be approximately constant
    assert spread < 0.15, (
        f"M₀(m=4) not linear in δ: spread = {100*spread:.1f}%"
    )
    # Coefficient is O(1)
    assert 0.5 < ratios.mean() < 10.0, (
        f"M₀/δ coefficient unexpected: {ratios.mean():.2f}"
    )


def test_c15_z16_m_scan(c15_mesh):
    """Z16 per-m scan: m=2 not protected (no inversion), m=4 protected (S4).

    Z16 has Td symmetry (no inversion). For m=2: M₀ ≠ 0 (Td does not
    protect m=2). For m=4 = N/2 (N=8 belt faces): M₀ = 0 because
    the 8 belt normals split into two S4 orbits (4 faces each) under
    the improper rotation S4: (x,y,z) → (y,−x,−z). Each orbit's
    area-weighted normals sum to zero (Σ area·n̂ = 0 per orbit).
    At m=4, cos(4θ) is constant within each orbit (+1.000 on A,
    −0.737 on B), so M₀ = c_A·sum_A + c_B·sum_B = 0.
    At m=2, cos(2θ) varies within orbits (±1 on A), breaking
    factorization → M₀ ≠ 0.  NOT Nyquist aliasing: thetas are
    unequally spaced (55.6° and 34.4° alternating).
    """
    v, e, f, cfi, centers, L, cell_type = c15_mesh
    from core_math.analysis.no_drag import _belt_geometry

    z16_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 16]
    m_values = [1, 2, 3, 4, 5, 6, 7]
    max_m0_per_m = {}

    for m in m_values:
        max_m0 = 0.0
        for ci in z16_cells:
            bg = _belt_geometry(ci, centers, v, f, cfi, L)
            circuit, theta, normals, areas, bn, fd = bg
            N = len(circuit)
            pressure = np.cos(m * theta)
            M0_vec = np.zeros(3)
            for idx in range(N):
                M0_vec += pressure[idx] * areas[idx] * normals[idx]
            max_m0 = max(max_m0, np.linalg.norm(M0_vec))
        max_m0_per_m[m] = max_m0

    for m, val in max_m0_per_m.items():
        status = "= 0 (protected)" if val < 1e-12 else "> 0 (NOT protected)"
        print(f"Z16 m={m}: max |M₀| = {val:.2e}  {status}")

    # m=4 = N/2 protected by rotational averaging (not inversion)
    assert max_m0_per_m[4] < 1e-12, (
        f"Z16 m=4 should be protected: {max_m0_per_m[4]:.2e}"
    )
    # m=2 NOT protected (paper's negative control)
    assert max_m0_per_m[2] > 0.01, (
        f"Z16 m=2 should NOT be protected: {max_m0_per_m[2]:.2e}"
    )


def test_chain_complex_does_not_explain_m0(c15_mesh):
    """d₁d₀ = 0 holds universally and does not explain M₀ = 0.

    The chain complex identity d₁d₀ = 0 is a topological property of ANY
    cell complex (boundary-of-boundary = 0). It holds on Z12 (where M₀ = 0)
    and Z16 (where M₀ ≠ 0) equally. Belt sources are not floppy modes:
    edge extensions ||D₀u₀|| are O(1) on both cell types.

    This rules out chain-complex explanations of the selection rule.
    The actual mechanism is geometric (centrosymmetry from Wyckoff site
    symmetry), not topological.
    """
    v, e, f, cfi, centers, L, cell_type = c15_mesh
    nv = len(v)
    ne = len(e)

    # 1. d₁d₀ = 0 on the full complex (universal identity)
    d0 = build_d0(v, e)
    d1 = build_d1(v, e, f)
    assert np.max(np.abs(d1 @ d0)) == 0, "d₁d₀ ≠ 0 (broken complex)"

    # 2. Build vector edge-extension operator D₀: R^{3V} → R^E
    #    Extension of edge (i,j) = (u_j - u_i) · r̂_ij
    D0_vec = np.zeros((ne, 3 * nv))
    for ei, (i, j) in enumerate(e):
        r_ij = v[j] - v[i]
        r_ij -= L * np.round(r_ij / L)
        length = np.linalg.norm(r_ij)
        if length > 1e-15:
            r_hat = r_ij / length
            for d in range(3):
                D0_vec[ei, 3 * i + d] = -r_hat[d]
                D0_vec[ei, 3 * j + d] = +r_hat[d]

    # 3. Edge extensions from belt source on Z12 vs Z16
    z12 = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    z16 = [ci for ci in range(len(centers)) if cell_type[ci] == 16]

    z12_ext, z16_ext = [], []
    z12_m0, z16_m0 = [], []

    for ci in z12:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        vecs, old_vec = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        assert old_vec is not None, f"Z12 cell {ci}: no belt vectors"
        z12_m0.append(sr['M0_mag'])
        z12_ext.append(np.linalg.norm(D0_vec @ old_vec))

    for ci in z16:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        vecs, old_vec = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        assert old_vec is not None, f"Z16 cell {ci}: no belt vectors"
        z16_m0.append(sr['M0_mag'])
        z16_ext.append(np.linalg.norm(D0_vec @ old_vec))

    z12_ext_mean = np.mean(z12_ext)
    z16_ext_mean = np.mean(z16_ext)
    z12_m0_mean = np.mean(z12_m0)
    z16_m0_mean = np.mean(z16_m0)

    print(f"d₁d₀ = 0: universal (max entry = 0)")
    print(f"Z12: M₀ = {z12_m0_mean:.2e}, ||D₀u₀|| = {z12_ext_mean:.4f}")
    print(f"Z16: M₀ = {z16_m0_mean:.2e}, ||D₀u₀|| = {z16_ext_mean:.4f}")
    print(f"Edge extension ratio Z12/Z16 = {z12_ext_mean/z16_ext_mean:.3f}")

    # Key assertions:
    # a) Belt sources are NOT floppy: ||D₀u₀|| = O(1) on both cell types.
    #    If belt sources were in ker(D₀), edge extensions would be ~0.
    #    Measured: ~1.06 (Z12) and ~1.02 (Z16) — both O(1).
    assert z12_ext_mean > 0.1, f"Z12 belt unexpectedly floppy: {z12_ext_mean}"
    assert z16_ext_mean > 0.1, f"Z16 belt unexpectedly floppy: {z16_ext_mean}"

    # b) Yet M₀ differs by >12 orders of magnitude
    assert z12_m0_mean < 1e-12
    assert z16_m0_mean > 0.01
    assert z16_m0_mean / max(z12_m0_mean, 1e-20) > 1e10, (
        f"M₀ gap too small: Z16/Z12 = {z16_m0_mean/max(z12_m0_mean,1e-20):.0e}"
    )
