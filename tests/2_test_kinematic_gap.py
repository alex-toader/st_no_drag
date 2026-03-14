#!/usr/bin/env python3
"""
Kinematic gap: belt frequencies above acoustic ceiling.

The kinematic gap is the primary structural protection for no-drag.
Belt excitations live at optical frequencies where no sound-like mode
exists. Even beyond the harmonic model, decay into acoustic phonons
requires energy-conserving final states — the gap prevents this.

The gap is structural (survives at k_L = k_T = 1, ratio 1.503) and
isotropic (ratio > 1.7 in all 9 BZ directions, worst-case [111]).
Both C15 and Kelvin have identical belt basis dimensions (96D/16D).
Character-weighted centroid velocity is significantly more subsonic
than the worst-case maximum (0.335 vs 0.711 on C15).

NOTES:
  - Belt basis dimensions (96D enriched, 16D particle) are specific to
    N=1 C15 (16 Z12 cells × 6 vectors/cell) and N=2 Kelvin (16 cells × 6).
    For other supercell sizes the numbers scale with n_belt_cells.
  - vT uses avg(omega[0], omega[1]) at small k. The two transverse modes
    are degenerate to < 0.6% on C15 (exact on [100], [111]; 0.62% on [110]).
  - omega[2] ceiling (0.694) vs acousticness ceiling at 10% (0.935): the
    latter includes optical modes with residual COM content. Belt floor
    (1.179) is above both, so the gap claim holds under either definition.

RAW OUTPUT (pytest -v -s, Mar 2026):
  test_c15_gap_exists              C15 gap ratio = 1.699
  test_c15_belt_above_acoustic     C15 ω_edge = 0.6935, ω_belt_min = 1.1785
  test_kelvin_gap_exists           Kelvin gap ratio = 1.374
  test_c15_belt_basis_dimensions   C15 belt basis: 96D enriched, 16D particle (16 belt cells)
  test_c15_subsonic                C15 v_g/v_T = 0.711
  test_kelvin_subsonic             Kelvin v_g/v_T = 0.541
  test_c15_gap_survives_isotropy   C15 gap ratio at k_L=k_T=1: 1.503
  test_c15_gap_all_bz_directions   [111] worst-case: 1.736; all 9 dirs > 1.7
  test_kelvin_belt_basis_dimensions Kelvin belt basis: 96D enriched, 16D particle (16 belt cells)
  test_c15_centroid_more_subsonic   v_g_max/v_T = 0.711, v_g_centroid/v_T = 0.334
  test_c15_gap_convergence         ω_edge(40)=ω_edge(80)=0.693536, Δ=0.00
  test_gap_vs_kL_kT_ratio          min gap = 1.503 at k_L/k_T=1.0; >1 for all [1,3]
  test_acousticness_ceiling_consistent ω[2]=0.694 < ac_10%=0.935 < belt=1.179; strict gap 1.260
  test_c15_gap_survives_jitter   δ=0.01: gap ratio = 1.693 (ω_edge=0.6935, ω_belt=1.1737)
  test_belt_floor_above_acoustic C15: 1.478, Kelvin: 1.363, WP: 1.327 (all belt modes above edge)
  test_hop_source_zero_weight_below_edge  All 16 Z12 cells: zero source weight below ω_edge
  test_floppy_mode_count         Kelvin: 94 floppy (M=96); WP: 44 floppy (M=46); E=2V, M=V

References:
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/14_fermi_golden_rule.py (Parts 2b-2c)
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/17_bz_3d_gap_scan.py
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
    build_belt_basis,
    compute_acoustic_ceiling,
    compute_acousticness_ceiling,
    compute_particle_floor,
    compute_projected_velocities,
    _DEFAULT_BZ_DIRS,
    _DENSE_BZ_DIRS,
)
from physics.bloch import DisplacementBloch


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def c15_gap_data():
    """Build C15 and compute gap quantities + velocities."""
    N, L_cell = 1, 4.0
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = np.array(get_c15_points(N, L_cell))
    L = N * L_cell
    n_cells = len(cfi)
    cell_type = np.array([12 if len(cfi[ci]) == 12 else 16
                          for ci in range(n_cells)])

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    z12_cells = [ci for ci in range(n_cells) if cell_type[ci] == 12]
    Q_belt, Q_particle, M_transfer = build_belt_basis(
        z12_cells, centers, v, f, cfi, L)
    n_particle = Q_particle.shape[1]

    # Acoustic ceiling (3 BZ directions for speed)
    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    # Particle floor
    pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    # Velocities (computed once, reused by subsonic + centroid tests)
    vel = compute_projected_velocities(
        bloch, Q_belt, M_transfer, n_particle,
        L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    return {
        'omega_edge': ac['omega_edge'],
        'omega_belt_min': pf['omega_belt_min_global'],
        'omega_belt_gamma': pf['omega_belt_min_gamma'],
        'gap_ratio': pf['omega_belt_min_global'] / ac['omega_edge'],
        'bloch': bloch,
        'Q_belt': Q_belt,
        'Q_particle': Q_particle,
        'M_transfer': M_transfer,
        'n_particle': n_particle,
        'n_belt_cells': len(z12_cells),
        'vel': vel,
        'L': L,
        'vertices': v, 'edges': e,
    }


@pytest.fixture(scope="module")
def kelvin_gap_data():
    """Build Kelvin and compute gap quantities."""
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4.0 * N
    n_cells = len(centers)

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    all_cells = list(range(n_cells))
    Q_belt, Q_particle, M_transfer = build_belt_basis(
        all_cells, centers, v, f, cfi, L)
    n_particle = Q_particle.shape[1]

    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    vel = compute_projected_velocities(
        bloch, Q_belt, M_transfer, n_particle,
        L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    return {
        'omega_edge': ac['omega_edge'],
        'omega_belt_min': pf['omega_belt_min_global'],
        'gap_ratio': pf['omega_belt_min_global'] / ac['omega_edge'],
        'bloch': bloch,
        'Q_belt': Q_belt,
        'Q_particle': Q_particle,
        'M_transfer': M_transfer,
        'n_particle': n_particle,
        'n_belt_cells': n_cells,
        'vel': vel,
        'L': L,
    }


# =========================================================================
# Tests
# =========================================================================

def test_c15_gap_exists(c15_gap_data):
    """C15 gap ratio > 1: belt frequencies above acoustic ceiling."""
    gap = c15_gap_data['gap_ratio']
    print(f"C15 gap ratio = {gap:.3f}")
    assert gap > 1.0, (
        f"No kinematic gap on C15: ratio = {gap:.3f}"
    )
    assert gap > 1.2, (
        f"C15 gap ratio unexpectedly small: {gap:.3f}"
    )


def test_c15_belt_above_acoustic(c15_gap_data):
    """C15 particle floor and acoustic ceiling have correct values."""
    omega_edge = c15_gap_data['omega_edge']
    omega_belt = c15_gap_data['omega_belt_min']

    print(f"C15 ω_edge = {omega_edge:.4f}, ω_belt_min = {omega_belt:.4f}")
    assert 0.5 < omega_edge < 0.8, (
        f"ω_edge = {omega_edge:.4f}, expected ~0.69"
    )
    assert 1.0 < omega_belt < 1.4, (
        f"ω_belt_min = {omega_belt:.4f}, expected ~1.18"
    )


def test_kelvin_gap_exists(kelvin_gap_data):
    """Kelvin gap ratio > 1: belt frequencies above acoustic ceiling."""
    gap = kelvin_gap_data['gap_ratio']
    print(f"Kelvin gap ratio = {gap:.3f}")
    assert gap > 1.0, (
        f"No kinematic gap on Kelvin: ratio = {gap:.3f}"
    )


def test_c15_belt_basis_dimensions(c15_gap_data):
    """C15 belt basis: 6 vectors/cell × n_belt_cells, QR rank-reduced."""
    Q_belt = c15_gap_data['Q_belt']
    n_particle = c15_gap_data['n_particle']
    n_belt_cells = c15_gap_data['n_belt_cells']

    # 6 enriched vectors per cell (3 directions × 2 phases), all independent
    expected_enriched = 6 * n_belt_cells

    print(f"C15 belt basis: {Q_belt.shape[1]}D enriched, {n_particle}D particle"
          f" ({n_belt_cells} belt cells)")
    assert Q_belt.shape[1] == expected_enriched, (
        f"Belt basis rank = {Q_belt.shape[1]}, expected {expected_enriched}"
    )
    assert n_particle == n_belt_cells, (
        f"Particle subspace dim = {n_particle}, expected {n_belt_cells}"
    )


def test_c15_subsonic(c15_gap_data):
    """C15 particle velocity v_g < v_T (subsonic)."""
    ratio = c15_gap_data['vel']['vg_max_over_vT']
    print(f"C15 v_g/v_T = {ratio:.3f}")
    assert ratio < 1.0, (
        f"C15 particle supersonic: v_g/v_T = {ratio:.3f}"
    )
    assert ratio > 0.3, (
        f"C15 v_g/v_T suspiciously small: {ratio:.3f}"
    )


def test_kelvin_subsonic(kelvin_gap_data):
    """Kelvin particle velocity v_g < v_T (subsonic)."""
    ratio = kelvin_gap_data['vel']['vg_max_over_vT']
    print(f"Kelvin v_g/v_T = {ratio:.3f}")
    assert ratio < 1.0, (
        f"Kelvin particle supersonic: v_g/v_T = {ratio:.3f}"
    )
    assert ratio > 0.1, (
        f"Kelvin v_g/v_T suspiciously small: {ratio:.3f}"
    )


def test_kelvin_centroid_subsonic(kelvin_gap_data):
    """Kelvin centroid velocity subsonic: v_g_centroid < v_T."""
    vel = kelvin_gap_data['vel']
    ratio_max = vel['vg_max_over_vT']
    ratio_cen = vel['vg_centroid_over_vT']
    print(f"Kelvin v_g_max/v_T = {ratio_max:.3f}, v_g_centroid/v_T = {ratio_cen:.3f}")

    assert ratio_cen < ratio_max, (
        f"Centroid {ratio_cen:.3f} >= max {ratio_max:.3f}"
    )
    assert ratio_cen < 0.5, (
        f"Centroid velocity ratio unexpectedly large: {ratio_cen:.3f}"
    )


def test_c15_gap_survives_isotropy(c15_gap_data):
    """Gap survives at k_L = k_T (isotropic springs): structural, not parameter artifact."""
    d = c15_gap_data
    bloch_iso = DisplacementBloch(d['vertices'], d['edges'], d['L'],
                                  k_L=1.0, k_T=1.0, mass=1.0)
    ac = compute_acoustic_ceiling(bloch_iso, d['L'],
                                  bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    pf = compute_particle_floor(bloch_iso, d['Q_belt'], d['M_transfer'],
                                d['n_particle'], d['L'],
                                bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    gap_iso = pf['omega_belt_min_global'] / ac['omega_edge']
    print(f"C15 gap ratio at k_L=k_T=1: {gap_iso:.3f}")
    assert gap_iso > 1.0, (
        f"Gap closes at isotropy: ratio = {gap_iso:.3f}"
    )
    assert gap_iso > 1.2, (
        f"Gap unexpectedly small at isotropy: {gap_iso:.3f}"
    )


def test_c15_gap_all_bz_directions(c15_gap_data):
    """Gap ratio > 1 in all 9 BZ directions (isotropy check)."""
    d = c15_gap_data
    # Ceiling on _DENSE_BZ_DIRS — per_dir keys match the loop below
    ac = compute_acoustic_ceiling(d['bloch'], d['L'],
                                  bz_dirs=_DENSE_BZ_DIRS, n_k=40)
    min_ratio = np.inf
    min_dir = ""
    for dname, dhat in _DENSE_BZ_DIRS.items():
        pf_dir = compute_particle_floor(
            d['bloch'], d['Q_belt'], d['M_transfer'], d['n_particle'],
            d['L'], bz_dirs={dname: dhat}, n_k=40)
        ratio = pf_dir['omega_belt_min_global'] / ac['per_dir'][dname]
        print(f"  {dname}: gap ratio = {ratio:.3f}")
        if ratio < min_ratio:
            min_ratio = ratio
            min_dir = dname

    print(f"C15 worst-case gap: {min_ratio:.3f} ({min_dir})")
    assert min_ratio > 1.0, (
        f"Gap closes in direction {min_dir}: ratio = {min_ratio:.3f}"
    )


def test_kelvin_belt_basis_dimensions(kelvin_gap_data):
    """Kelvin belt basis: 6 vectors/cell × n_belt_cells, QR rank-reduced."""
    Q_belt = kelvin_gap_data['Q_belt']
    n_particle = kelvin_gap_data['n_particle']
    n_belt_cells = kelvin_gap_data['n_belt_cells']

    expected_enriched = 6 * n_belt_cells

    print(f"Kelvin belt basis: {Q_belt.shape[1]}D enriched, {n_particle}D particle"
          f" ({n_belt_cells} belt cells)")
    assert Q_belt.shape[1] == expected_enriched, (
        f"Belt basis rank = {Q_belt.shape[1]}, expected {expected_enriched}"
    )
    assert n_particle == n_belt_cells, (
        f"Particle subspace dim = {n_particle}, expected {n_belt_cells}"
    )


def test_c15_centroid_more_subsonic(c15_gap_data):
    """v_g centroid < v_g max: character-weighted velocity more subsonic."""
    vel = c15_gap_data['vel']
    ratio_max = vel['vg_max_over_vT']
    ratio_cen = vel['vg_centroid_over_vT']
    print(f"C15 v_g_max/v_T = {ratio_max:.3f}, v_g_centroid/v_T = {ratio_cen:.3f}")

    assert ratio_cen < ratio_max, (
        f"Centroid {ratio_cen:.3f} >= max {ratio_max:.3f}"
    )
    assert ratio_cen < 0.5, (
        f"Centroid velocity ratio unexpectedly large: {ratio_cen:.3f}"
    )


def test_c15_gap_convergence(c15_gap_data):
    """Acoustic ceiling converged: delta < 1% between n_k=40 and n_k=80."""
    d = c15_gap_data
    ac40 = compute_acoustic_ceiling(d['bloch'], d['L'],
                                    bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    ac80 = compute_acoustic_ceiling(d['bloch'], d['L'],
                                    bz_dirs=_DEFAULT_BZ_DIRS, n_k=80)
    delta = abs(ac80['omega_edge'] - ac40['omega_edge'])
    rel = delta / ac40['omega_edge']
    print(f"Convergence: ω_edge(40)={ac40['omega_edge']:.6f}, "
          f"ω_edge(80)={ac80['omega_edge']:.6f}, Δ={delta:.2e} ({rel*100:.2f}%)")
    assert rel < 0.01, (
        f"Acoustic ceiling not converged: Δ/ω = {rel:.4f}"
    )


def test_gap_vs_kL_kT_ratio(c15_gap_data):
    """Gap ratio > 1 for k_L/k_T from 1 to 3 (parametric robustness)."""
    d = c15_gap_data
    ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    min_gap = np.inf
    for r in ratios:
        bloch_r = DisplacementBloch(d['vertices'], d['edges'], d['L'],
                                    k_L=r, k_T=1.0, mass=1.0)
        ac = compute_acoustic_ceiling(bloch_r, d['L'],
                                      bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
        pf = compute_particle_floor(bloch_r, d['Q_belt'], d['M_transfer'],
                                    d['n_particle'], d['L'],
                                    bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
        gap = pf['omega_belt_min_global'] / ac['omega_edge']
        print(f"  k_L/k_T={r:.1f}: gap ratio = {gap:.3f}")
        min_gap = min(min_gap, gap)

    print(f"Min gap across k_L/k_T scan: {min_gap:.3f}")
    assert min_gap > 1.0, (
        f"Gap closes at some k_L/k_T: min ratio = {min_gap:.3f}"
    )


def test_acousticness_ceiling_consistent(c15_gap_data):
    """Acousticness ceiling hierarchy: omega[2] < 10% < belt floor.

    omega[2] = ceiling of 3rd acoustic branch (purely acoustic modes).
    Acousticness 10% = max ω where any mode has > 10% COM content
    (wider: includes optical modes with residual translation character).

    The paper's gap claim holds under BOTH definitions: belt floor 1.18
    is above even the 10% acousticness ceiling 0.94.
    """
    d = c15_gap_data
    ac = compute_acoustic_ceiling(d['bloch'], d['L'],
                                  bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    ac_com = compute_acousticness_ceiling(d['bloch'], d['L'], n_k=40,
                                          bz_dirs=_DEFAULT_BZ_DIRS)

    omega_edge = ac['omega_edge']
    ceiling_10 = ac_com['results'][0]['ceiling']
    omega_belt = d['omega_belt_min']

    print(f"ω_edge (omega[2]):          {omega_edge:.4f}")
    print(f"Acousticness ceiling (10%): {ceiling_10:.4f}")
    print(f"ω_belt_min:                 {omega_belt:.4f}")

    # Hierarchy: omega[2] < acousticness_10% < belt_min
    # omega[2] is the 3rd branch ceiling; acousticness_10% is higher because
    # optical modes can retain partial COM content above the 3rd branch.
    assert omega_edge < ceiling_10, (
        f"omega[2] = {omega_edge:.4f} >= acousticness ceiling "
        f"{ceiling_10:.4f} — expected omega[2] to be lower"
    )
    # Key claim: belt floor above EVEN the wider acousticness ceiling
    assert omega_belt > ceiling_10, (
        f"Belt floor {omega_belt:.4f} below acousticness ceiling "
        f"{ceiling_10:.4f} — gap claim fails under strict definition"
    )
    gap_strict = omega_belt / ceiling_10
    print(f"Gap ratio (strict, 10% acousticness): {gap_strict:.3f}")


def test_c15_gap_survives_jitter():
    """Gap > 1 under random vertex jitter δ=0.01.

    The kinematic gap is spectral (not symmetry-exact), so it survives
    geometric disorder. At δ=0.01, the gap ratio decreases slightly
    but remains well above 1. This confirms the gap is structural.

    Uses fixed seed for reproducibility.
    """
    N, L_cell = 1, 4.0
    L = N * L_cell
    points_exact = get_c15_points(N, L_cell)

    np.random.seed(42)
    delta = 0.01
    noise = np.random.randn(*points_exact.shape)
    points_j = points_exact + delta * noise

    vj, ej, fj, cfij = build_c15_supercell_periodic(N, L_cell, points=points_j)
    centersj = np.array(points_j)

    z12_j = [ci for ci in range(len(cfij)) if len(cfij[ci]) == 12]

    Q_belt, Q_particle, M_transfer = build_belt_basis(
        z12_j, centersj, vj, fj, cfij, L)
    n_particle = Q_particle.shape[1]

    bloch = DisplacementBloch(vj, ej, L, k_L=2.0, k_T=1.0, mass=1.0)

    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    gap = pf['omega_belt_min_global'] / ac['omega_edge']

    print(f"δ={delta}: gap ratio = {gap:.3f} "
          f"(ω_edge={ac['omega_edge']:.4f}, ω_belt={pf['omega_belt_min_global']:.4f})")
    print(f"  (unperturbed: 1.699)")
    assert gap > 1.0, f"Gap < 1 under jitter δ={delta}: ratio = {gap:.3f}"


def test_belt_floor_above_acoustic():
    """All belt eigenvalues above acoustic ceiling, on all 3 structures.

    belt_floor = min eigenvalue of H_belt = Q_belt^T D Q_belt across BZ.
    This is the strongest gap claim: every belt mode (particle + deformation)
    sits above omega_edge. The hop source excites the full belt spectrum
    (not just particle modes), so belt_floor is the physically relevant
    quantity for the kinematic gap.
    """
    from core_math.builders.wp_periodic import build_wp_supercell_periodic, get_a15_points

    configs = []
    # C15
    v, e, f, cfi = build_c15_supercell_periodic(1, 4.0)
    c = np.array(get_c15_points(1, 4.0))
    z12 = [ci for ci in range(len(cfi)) if len(cfi[ci]) == 12]
    configs.append(('C15', v, e, f, cfi, c, z12, 4.0))
    # Kelvin
    vk, ek, fk, cfik = build_bcc_supercell_periodic(2)
    ck = np.array(generate_bcc_centers(2))
    configs.append(('Kelvin', vk, ek, fk, cfik, ck, list(range(len(ck))), 8.0))
    # WP
    vw, ew, fw, cfiw = build_wp_supercell_periodic(1, 4.0)
    cw = np.array(get_a15_points(1, 4.0))
    ta = [ci for ci in range(len(cfiw)) if len(cfiw[ci]) == 12]
    configs.append(('WP', vw, ew, fw, cfiw, cw, ta, 4.0))

    for name, vs, es, fs, cfis, cs, belt_cells, Ls in configs:
        Q_belt = build_belt_basis(belt_cells, cs, vs, fs, cfis, Ls)[0]
        bloch = DisplacementBloch(vs, es, Ls, k_L=2.0, k_T=1.0, mass=1.0)
        ac = compute_acoustic_ceiling(bloch, Ls, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
        edge = ac['omega_edge']

        belt_min = np.inf
        for d_vec in _DEFAULT_BZ_DIRS.values():
            k_max = np.pi / (Ls * np.max(np.abs(d_vec)))
            for ik in range(41):
                kk = (ik / 40) * k_max
                Dk = bloch.build_dynamical_matrix(kk * d_vec)
                H_belt = Q_belt.T @ Dk @ Q_belt
                w = np.sqrt(max(np.linalg.eigvalsh(H_belt)[0], 0))
                if w < belt_min:
                    belt_min = w

        gap = belt_min / edge
        print(f"  {name}: belt_floor={belt_min:.4f}, edge={edge:.4f}, gap={gap:.3f}")
        assert gap > 1.0, f"{name}: belt_floor below acoustic ceiling ({gap:.3f})"
        assert gap > 1.2, f"{name}: gap unexpectedly small ({gap:.3f})"


def test_hop_source_zero_weight_below_edge():
    """Hop source has zero spectral weight below acoustic ceiling.

    The hop excites belt eigenmodes across the full belt spectrum.
    The kinematic gap guarantees that none of these modes fall below
    omega_edge. This test verifies that directly: decompose the hop
    source into belt eigenmodes and check that zero weight sits below
    the acoustic ceiling. Tested on C15 (all 16 Z12 cells).
    """
    from core_math.analysis.no_drag import get_belt_vertex_forces

    N, L_cell = 1, 4.0
    L = N * L_cell
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = np.array(get_c15_points(N, L_cell))
    z12 = [ci for ci in range(len(cfi)) if len(cfi[ci]) == 12]
    Q_belt = build_belt_basis(z12, centers, v, f, cfi, L)[0]

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)
    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    edge = ac['omega_edge']

    # Belt eigenmodes at Gamma
    Dk = bloch.build_dynamical_matrix(np.zeros(3))
    H_belt = Q_belt.T @ Dk @ Q_belt
    evals, evecs = np.linalg.eigh(H_belt)
    omega = np.sqrt(np.maximum(evals, 0))

    for ci in z12:
        u0 = np.zeros(3 * len(v))
        vf = get_belt_vertex_forces(ci, centers, v, f, cfi, L)
        for vi, force in vf.items():
            u0[3 * vi:3 * vi + 3] = force

        u0_belt = Q_belt.T @ u0
        c_eig = evecs.T @ u0_belt
        c2 = np.real(c_eig ** 2)
        total = np.sum(c2)
        if total < 1e-30:
            continue
        c2 /= total

        weight_below = np.sum(c2[omega < edge])
        print(f"  cell {ci}: weight below edge = {weight_below:.2e}, "
              f"lowest excited mode = {omega[c2 > 0.001].min():.4f}")
        assert weight_below < 1e-12, (
            f"Cell {ci}: source weight {weight_below:.2e} below edge"
        )

    print(f"  All {len(z12)} cells: zero source weight below ω_edge = {edge:.4f}")

    # Also verify on hop-pair source S = F_belt(B) - F_belt(A)
    ci_a, ci_b = z12[0], z12[1]
    S_hop = np.zeros(3 * len(v))
    vf_a = get_belt_vertex_forces(ci_a, centers, v, f, cfi, L)
    vf_b = get_belt_vertex_forces(ci_b, centers, v, f, cfi, L)
    for vi in set(vf_a.keys()) | set(vf_b.keys()):
        S_hop[3 * vi:3 * vi + 3] = (vf_b.get(vi, np.zeros(3))
                                     - vf_a.get(vi, np.zeros(3)))
    s_belt = Q_belt.T @ S_hop
    c_hop = evecs.T @ s_belt
    c2_hop = np.real(c_hop ** 2)
    c2_hop /= np.sum(c2_hop)
    w_hop = np.sum(c2_hop[omega < edge])
    print(f"  Hop pair ({ci_a},{ci_b}): weight below edge = {w_hop:.2e}")
    assert w_hop < 1e-12, f"Hop source weight {w_hop:.2e} below edge"


def test_floppy_mode_count():
    """At k_T=0 (central-force), ~V floppy modes exist.

    Maxwell count M = 3V - E = V for Plateau foams (deg=4, E=2V).
    Numerically: n_zero = V + 3 - S where S = states of self-stress.
    Adding k_T > 0 lifts these into the optical band → kinematic gap.
    """
    results = []
    # Kelvin N=2
    v_k, e_k, f_k, cfi_k = build_bcc_supercell_periodic(2)
    L_k = 8.0
    bloch_k = DisplacementBloch(v_k, e_k, L_k, k_L=1.0, k_T=0.0, mass=1.0)
    Dk = bloch_k.build_dynamical_matrix(np.zeros(3))
    evals_k = np.linalg.eigvalsh(Dk)
    n_zero_k = int(np.sum(np.abs(evals_k) < 1e-10))
    nv_k = len(v_k)
    results.append(("Kelvin", nv_k, len(e_k), n_zero_k))

    # WP N=1
    from core_math.builders.wp_periodic import build_wp_supercell_periodic
    res_wp = build_wp_supercell_periodic(1, 4.0)
    v_w, e_w = res_wp[0], res_wp[1]
    L_w = 4.0
    bloch_w = DisplacementBloch(v_w, e_w, L_w, k_L=1.0, k_T=0.0, mass=1.0)
    Dw = bloch_w.build_dynamical_matrix(np.zeros(3))
    evals_w = np.linalg.eigvalsh(Dw)
    n_zero_w = int(np.sum(np.abs(evals_w) < 1e-10))
    nv_w = len(v_w)
    results.append(("WP", nv_w, len(e_w), n_zero_w))

    for name, nv, ne, nz in results:
        maxwell = 3 * nv - ne
        floppy = nz - 3
        print(f"{name}: V={nv}, E={ne}, M=3V-E={maxwell}, "
              f"zero modes={nz} (3 transl + {floppy} floppy)")
        # E = 2V for Plateau foams → M = V
        assert ne == 2 * nv, f"{name}: E={ne} ≠ 2V={2*nv}"
        assert maxwell == nv, f"{name}: M={maxwell} ≠ V={nv}"
        # Floppy count close to V (within ~5 from self-stress)
        assert abs(floppy - nv) < 10, (
            f"{name}: floppy={floppy}, expected ~V={nv}"
        )
