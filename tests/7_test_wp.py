#!/usr/bin/env python3
"""
Weaire-Phelan foam: selection rule, gap, wavepacket, dipolar scaling.

WP structure: A15 lattice (space group Pm3n, No. 223), 8 cells per domain.
  - 2 Type A (pentagonal dodecahedra, 12 faces, Wyckoff 2a)
  - 6 Type B (tetrakaidecahedra, 14 faces, Wyckoff 6d)

Type A cells are pentagonal dodecahedra with N=6 belt, identical protection
mechanism to C15 Z12: antipodal normal tilt n_ax[i] = -n_ax[i+N/2] kills
even-m Fourier modes (m=2,4), giving M₀ = 0 (machine zero).

Type B cells have mixed pentagon/hexagon faces, variable areas (2:1 ratio),
and no antipodal symmetry. M₀ ≠ 0 on Type B (negative control).

RAW OUTPUT (pytest -v -s, Mar 2026):
  test_wp_topology                 V=46, E=92, F=54, C=8, chi3=0; 48 pentagons + 6 hexagons
  test_wp_type_a_selection_rule    Type A (2 cells): max M₀ = 1.12e-15
  test_wp_type_b_nonzero           Type B (6 cells): M₀ range [0.669, 0.824]
  test_wp_type_a_antipodal_tilt    max |n_ax[i]+n_ax[i+3]| = 5.55e-17
  test_wp_type_a_even_m_protected  m=2: ZERO, m=4: ZERO; m=1,3,5: nonzero
  test_wp_gap_exists               WP gap ratio = 1.694 (ω_edge=1.0014, ω_belt_min=1.6964)
  test_wp_gap_all_bz_directions    worst [111]: 1.727; all 9 dirs > 1.7
  test_wp_gap_survives_isotropy    gap ratio at k_L=k_T=1: 1.283
  test_wp_gap_convergence          ω_edge(40)=1.001360, ω_edge(80)=1.001360, Δ=0.00
  test_wp_subsonic_centroid        v_g_max/v_T=1.016, v_g_centroid/v_T=0.397
  test_wp_gap_vs_spring_ratio      gap > 1 for all k_L/k_T ∈ [1,3] (min 1.283 at isotropy)
  test_wp_wavepacket_gamma         max ||Q_trans^T u(t)||² = 2.58e-31 over 4 periods
  test_wp_belt_covers_source       residual = 5.89e-17
  test_wp_source_cell_independence max ||Q_trans^T u₀||² over 2 Type A cells = 1.90e-32
  test_wp_type_b_loophole_bound   Modes selected: 17; max COM = 1.93e-02 at [111] k/k_max=1.0 (< 10%)
  test_wp_hop_source_monopole_zero M₀ = 6.26e-16 (machine zero)
  test_wp_hop_source_dipole_exists |M₁| = 6.2631
  test_wp_hop_source_k_squared     [100]: slope=2.00, [110]: slope=1.99, [111]: slope=1.98

Date: Mar 2026
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core_math.builders.wp_periodic import (
    build_wp_supercell_periodic, get_a15_points
)
from core_math.analysis.no_drag import (
    compute_selection_rule,
    get_belt_vertex_forces,
    build_belt_basis,
    build_enriched_belt_vectors,
    compute_acoustic_ceiling,
    compute_particle_floor,
    _belt_geometry,
    _DEFAULT_BZ_DIRS,
    _DENSE_BZ_DIRS,
)
from physics.bloch import DisplacementBloch


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def wp_mesh():
    """Build WP N=1 supercell (8 cells: 2 Type A + 6 Type B)."""
    N, L_cell = 1, 4.0
    v, e, f, cfi = build_wp_supercell_periodic(N, L_cell)
    centers = get_a15_points(N, L_cell)
    L = N * L_cell
    n_cells = len(cfi)
    cell_nfaces = np.array([len(cfi[ci]) for ci in range(n_cells)])
    type_a = [ci for ci in range(n_cells) if cell_nfaces[ci] == 12]
    type_b = [ci for ci in range(n_cells) if cell_nfaces[ci] == 14]
    return v, e, f, cfi, centers, L, type_a, type_b


@pytest.fixture(scope="module")
def wp_gap_data(wp_mesh):
    """Compute gap, velocity, and belt basis for WP."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    Q_belt, Q_particle, M_transfer = build_belt_basis(
        type_a, centers, v, f, cfi, L)
    n_particle = Q_particle.shape[1]

    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    return {
        'omega_edge': ac['omega_edge'],
        'omega_belt_min': pf['omega_belt_min_global'],
        'gap_ratio': pf['omega_belt_min_global'] / ac['omega_edge'],
        'bloch': bloch,
        'Q_belt': Q_belt,
        'M_transfer': M_transfer,
        'n_particle': n_particle,
        'L': L,
        'v': v, 'e': e, 'f': f, 'cfi': cfi,
        'centers': centers, 'type_a': type_a,
    }


@pytest.fixture(scope="module")
def wp_wavepacket(wp_mesh):
    """Diagonalize D(k=0), prepare belt source for wavepacket test."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh
    nv = len(v)
    n_dof = 3 * nv

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    # Diagonalize D(k=0)
    D0_complex = bloch.build_dynamical_matrix(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(D0_complex.imag, 0, atol=1e-14)
    D0 = np.real(D0_complex)
    evals, evecs = np.linalg.eigh(D0)
    evecs = np.real(evecs)
    omega = np.sqrt(np.maximum(evals, 0))

    # Translation basis
    Q_trans = np.zeros((n_dof, 3))
    for d in range(3):
        q = np.zeros(n_dof)
        for i in range(nv):
            q[3 * i + d] = 1.0
        Q_trans[:, d] = q / np.linalg.norm(q)

    # Belt source on Type A cell 0
    source_cell = type_a[0]
    _, old_vec = build_enriched_belt_vectors(
        source_cell, centers, v, f, cfi, L)
    u0 = old_vec / np.linalg.norm(old_vec)

    # Mode decomposition
    coeffs = evecs.T @ u0
    mode_energy = 0.5 * coeffs**2 * omega**2

    omega_eff = np.sqrt(np.sum(coeffs**2 * omega**2) / np.sum(coeffs**2))
    T_period = 2 * np.pi / omega_eff

    # Belt basis
    Q_belt, _, _ = build_belt_basis(type_a, centers, v, f, cfi, L)

    return {
        'evecs': evecs, 'omega': omega, 'coeffs': coeffs,
        'Q_trans': Q_trans, 'u0': u0, 'T_period': T_period,
        'Q_belt': Q_belt, 'bloch': bloch,
        'nv': nv, 'n_dof': n_dof, 'L': L,
        'v': v, 'e': e, 'f': f, 'cfi': cfi,
        'centers': centers, 'type_a': type_a,
    }


# =========================================================================
# Topology
# =========================================================================

def test_wp_topology(wp_mesh):
    """WP N=1: V=46, E=92, F=54, C=8, χ₃=0; 2 Type A + 6 Type B."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh
    V, E, F, C = len(v), len(e), len(f), len(cfi)
    chi3 = V - E + F - C

    # Face polygon counts
    from collections import Counter
    face_sizes = Counter(len(face) for face in f)
    n_pent, n_hex = face_sizes.get(5, 0), face_sizes.get(6, 0)

    print(f"V={V}, E={E}, F={F}, C={C}, chi3={chi3}")
    print(f"Faces: {n_pent} pentagons + {n_hex} hexagons")
    assert V == 46
    assert E == 92
    assert F == 54
    assert C == 8
    assert chi3 == 0
    assert len(type_a) == 2, f"Expected 2 Type A cells, got {len(type_a)}"
    assert len(type_b) == 6, f"Expected 6 Type B cells, got {len(type_b)}"
    assert set(face_sizes.keys()) == {5, 6}, f"Unexpected face sizes: {face_sizes}"
    assert n_pent == 48 and n_hex == 6, f"Expected 48+6, got {n_pent}+{n_hex}"


# =========================================================================
# Selection rule
# =========================================================================

def test_wp_type_a_selection_rule(wp_mesh):
    """M₀ = 0 on all Type A (dodecahedral) cells."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    m0_values = []
    for ci in type_a:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Type A cell {ci} has no belt"
        assert sr['n_belt'] == 6, f"Expected N=6 belt, got {sr['n_belt']}"
        m0_values.append(sr['M0_mag'])

    m0_max = max(m0_values)
    print(f"Type A ({len(type_a)} cells): max M₀ = {m0_max:.2e}")
    assert m0_max < 1e-12, f"M₀ not zero on Type A: max = {m0_max:.2e}"


def test_wp_type_b_nonzero(wp_mesh):
    """M₀ ≠ 0 on Type B cells (negative control)."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    m0_values = []
    for ci in type_b:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Type B cell {ci} has no belt"
        m0_values.append(sr['M0_mag'])

    m0_min = min(m0_values)
    m0_max = max(m0_values)
    print(f"Type B ({len(type_b)} cells): M₀ range [{m0_min:.3f}, {m0_max:.3f}]")
    assert m0_min > 0.5, f"Type B M₀ too small: min = {m0_min:.4f}"


def test_wp_type_a_antipodal_tilt(wp_mesh):
    """Type A: n_ax[i] = -n_ax[i+N/2] (antipodal symmetry)."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    max_asym = 0.0
    for ci in type_a:
        bg = _belt_geometry(ci, centers, v, f, cfi, L)
        assert bg is not None
        circuit, theta, normals, areas, bn, fd = bg
        N = len(circuit)
        assert N == 6

        n_ax = np.array([np.dot(normals[i], bn) for i in range(N)])
        for i in range(N // 2):
            asym = abs(n_ax[i] + n_ax[i + N // 2])
            max_asym = max(max_asym, asym)

    print(f"max |n_ax[i]+n_ax[i+3]| = {max_asym:.2e}")
    assert max_asym < 1e-12, f"Antipodal symmetry broken: {max_asym:.2e}"


def test_wp_type_a_even_m_protected(wp_mesh):
    """Type A: even m (2,4) protected, odd m (1,3,5) not."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    ci = type_a[0]
    bg = _belt_geometry(ci, centers, v, f, cfi, L)
    circuit, theta, normals, areas, bn, fd = bg
    N = len(circuit)

    protected = []
    unprotected = []
    for m in range(1, N):
        pressure = np.cos(m * theta)
        M0 = np.zeros(3)
        for idx in range(N):
            M0 += pressure[idx] * areas[idx] * normals[idx]
        M0_mag = np.linalg.norm(M0)

        if m % 2 == 0:
            protected.append((m, M0_mag))
        else:
            unprotected.append((m, M0_mag))

    for m, val in protected:
        print(f"  m={m}: M₀ = {val:.2e} (ZERO)")
        assert val < 1e-12, f"Even m={m} not protected: M₀ = {val:.2e}"

    for m, val in unprotected:
        print(f"  m={m}: M₀ = {val:.4f}")
        assert val > 0.1, f"Odd m={m} unexpectedly protected: M₀ = {val:.2e}"


# =========================================================================
# Kinematic gap
# =========================================================================

def test_wp_gap_exists(wp_gap_data):
    """WP gap ratio > 1: belt above acoustic ceiling."""
    gap = wp_gap_data['gap_ratio']
    omega_edge = wp_gap_data['omega_edge']
    omega_belt = wp_gap_data['omega_belt_min']

    print(f"WP gap ratio = {gap:.3f} "
          f"(ω_edge={omega_edge:.4f}, ω_belt_min={omega_belt:.4f})")
    assert gap > 1.0, f"No kinematic gap: ratio = {gap:.3f}"
    assert gap > 1.5, f"Gap unexpectedly small: ratio = {gap:.3f}"


def test_wp_gap_all_bz_directions(wp_gap_data):
    """Gap > 1 in all 9 BZ directions."""
    bloch = wp_gap_data['bloch']
    Q_belt = wp_gap_data['Q_belt']
    M_transfer = wp_gap_data['M_transfer']
    n_particle = wp_gap_data['n_particle']
    L = wp_gap_data['L']

    worst_name, worst_gap = None, np.inf
    for name, dhat in _DENSE_BZ_DIRS.items():
        ac = compute_acoustic_ceiling(bloch, L, bz_dirs={name: dhat}, n_k=40)
        pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                    L, bz_dirs={name: dhat}, n_k=40)
        gap = pf['omega_belt_min_global'] / ac['omega_edge']
        if gap < worst_gap:
            worst_gap = gap
            worst_name = name

    print(f"worst {worst_name}: {worst_gap:.3f}; all 9 dirs > {worst_gap:.1f}")
    assert worst_gap > 1.0, f"Gap < 1 in direction {worst_name}: {worst_gap:.3f}"


def test_wp_gap_survives_isotropy(wp_mesh):
    """Gap > 1 at isotropy k_L = k_T = 1."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    bloch = DisplacementBloch(v, e, L, k_L=1.0, k_T=1.0, mass=1.0)
    Q_belt, Q_particle, M_transfer = build_belt_basis(
        type_a, centers, v, f, cfi, L)
    n_particle = Q_particle.shape[1]

    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    gap = pf['omega_belt_min_global'] / ac['omega_edge']

    print(f"WP gap ratio at k_L=k_T=1: {gap:.3f}")
    assert gap > 1.0, f"Gap < 1 at isotropy: {gap:.3f}"
    assert gap > 1.2, f"Gap unexpectedly small at isotropy: {gap:.3f}"


def test_wp_gap_convergence(wp_gap_data):
    """Acoustic ceiling converged: delta < 1% between n_k=40 and n_k=80."""
    bloch = wp_gap_data['bloch']
    L = wp_gap_data['L']

    ac40 = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
    ac80 = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=80)
    delta = abs(ac80['omega_edge'] - ac40['omega_edge'])
    rel = delta / ac40['omega_edge']

    print(f"ω_edge(40)={ac40['omega_edge']:.6f}, "
          f"ω_edge(80)={ac80['omega_edge']:.6f}, Δ={delta:.2e} ({rel*100:.2f}%)")
    assert rel < 0.01, f"Acoustic ceiling not converged: Δ/ω = {rel:.4f}"


def test_wp_subsonic_centroid(wp_gap_data):
    """Particle centroid velocity subsonic: v_g_centroid < v_T."""
    from core_math.analysis.no_drag import compute_projected_velocities

    bloch = wp_gap_data['bloch']
    Q_belt = wp_gap_data['Q_belt']
    M_transfer = wp_gap_data['M_transfer']
    n_particle = wp_gap_data['n_particle']
    L = wp_gap_data['L']

    vel = compute_projected_velocities(
        bloch, Q_belt, M_transfer, n_particle, L,
        bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    ratio_max = vel['vg_max_over_vT']
    ratio_centroid = vel['vg_centroid_over_vT']

    print(f"v_g_max/v_T = {ratio_max:.3f}, v_g_centroid/v_T = {ratio_centroid:.3f}")
    # WP is marginally supersonic at k_L/k_T=2 (v_g_max/v_T ~ 1.016)
    # but centroid is always subsonic
    assert ratio_centroid < 1.0, (
        f"Centroid supersonic: v_g_centroid/v_T = {ratio_centroid:.3f}")
    assert ratio_centroid < 0.5, (
        f"Centroid unexpectedly fast: v_g_centroid/v_T = {ratio_centroid:.3f}")


def test_wp_gap_vs_spring_ratio(wp_mesh):
    """Gap > 1 for k_L/k_T in [1, 3]."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh

    ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    min_gap = np.inf

    for r in ratios:
        bloch = DisplacementBloch(v, e, L, k_L=r, k_T=1.0, mass=1.0)
        Q_belt, Q_particle, M_transfer = build_belt_basis(
            type_a, centers, v, f, cfi, L)
        n_particle = Q_particle.shape[1]

        ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
        pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                                    L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
        gap = pf['omega_belt_min_global'] / ac['omega_edge']
        print(f"  k_L/k_T={r:.1f}: gap = {gap:.3f}")
        min_gap = min(min_gap, gap)

    assert min_gap > 1.0, f"Gap < 1 at some k_L/k_T: min = {min_gap:.3f}"


# =========================================================================
# Wavepacket
# =========================================================================

def test_wp_wavepacket_gamma(wp_wavepacket):
    """||Q_trans^T u(t)||² stays at machine zero at Γ."""
    d = wp_wavepacket
    evecs, omega, coeffs = d['evecs'], d['omega'], d['coeffs']
    Q_trans, T = d['Q_trans'], d['T_period']

    n_steps = 80
    times = np.linspace(0, 4 * T, n_steps)
    max_trans = 0.0

    for t in times:
        u_t = evecs @ (coeffs * np.cos(omega * t))
        trans_sq = np.linalg.norm(Q_trans.T @ u_t)**2
        max_trans = max(max_trans, trans_sq)

    print(f"max ||Q_trans^T u(t)||² = {max_trans:.2e} over 4 periods")
    assert max_trans < 1e-25, (
        f"Translation content not zero: max = {max_trans:.2e}"
    )


def test_wp_belt_covers_source(wp_wavepacket):
    """Belt basis covers the source: ||u₀ - Q Q^T u₀|| ~ machine zero."""
    d = wp_wavepacket
    u0, Q_belt = d['u0'], d['Q_belt']

    proj = Q_belt @ (Q_belt.T @ u0)
    residual = np.linalg.norm(u0 - proj)

    print(f"residual = {residual:.2e}")
    assert residual < 1e-12, f"Belt basis doesn't cover source: {residual:.2e}"


def test_wp_source_cell_independence(wp_wavepacket):
    """Both Type A cells give machine zero acoustic content at Γ."""
    d = wp_wavepacket
    v, f, cfi, centers, L = d['v'], d['f'], d['cfi'], d['centers'], d['L']
    type_a = d['type_a']
    evecs, omega = d['evecs'], d['omega']
    Q_trans = d['Q_trans']

    max_proj = 0.0
    for ci in type_a:
        _, old_vec = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        u_ci = old_vec / np.linalg.norm(old_vec)
        trans_sq = np.linalg.norm(Q_trans.T @ u_ci)**2
        max_proj = max(max_proj, trans_sq)

    print(f"max ||Q_trans^T u₀||² over {len(type_a)} Type A cells = {max_proj:.2e}")
    assert max_proj < 1e-25, (
        f"Source→translation not zero: {max_proj:.2e}"
    )


def test_wp_type_b_loophole_bound(wp_mesh):
    """Type B loophole bound: particle-band COM content < 10% at all k.

    WP has 6 Type B cells (75%) with M₀ ≠ 0. At finite k, hybridization
    could mix belt and acoustic character through the Type B channel.
    This test bounds the leakage: particle-like eigenstates of D(k)
    have COM-coherent translation content < 10% across the BZ.

    Analog of test_c15_particle_com_content_finite_k (test 3) but
    stronger test: 75% unprotected cells (vs 33% Z16 on C15).
    """
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh
    nv = len(v)
    n_dof = 3 * nv

    # Belt basis (Type A cells only)
    Q_belt, Q_particle, M_transfer = build_belt_basis(
        type_a, centers, v, f, cfi, L)
    n_particle = Q_particle.shape[1]

    # Q_trans (uniform displacement)
    Q_trans = np.zeros((n_dof, 3))
    for d in range(3):
        q = np.zeros(n_dof)
        for i in range(nv):
            q[3 * i + d] = 1.0
        Q_trans[:, d] = q / np.linalg.norm(q)

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    # omega_cut from particle floor
    res_floor = compute_particle_floor(
        bloch, Q_belt, M_transfer, n_particle, L)
    omega_cut = res_floor['omega_belt_min_global']

    # bp_cut: belt projection threshold (above random baseline)
    bp_random = Q_belt.shape[1] / n_dof
    bp_cut = 0.25

    print(f"omega_cut = {omega_cut:.4f} (particle floor from BZ scan)")
    print(f"bp_cut = {bp_cut:.3f} (random baseline {bp_random:.3f})")

    # BZ scan: 7 directions, 4 k-fractions
    dirs = {
        '[100]': np.array([1, 0, 0], dtype=float),
        '[010]': np.array([0, 1, 0], dtype=float),
        '[001]': np.array([0, 0, 1], dtype=float),
        '[110]': np.array([1, 1, 0], dtype=float) / np.sqrt(2),
        '[101]': np.array([1, 0, 1], dtype=float) / np.sqrt(2),
        '[011]': np.array([0, 1, 1], dtype=float) / np.sqrt(2),
        '[111]': np.array([1, 1, 1], dtype=float) / np.sqrt(3),
    }
    k_fracs = [0.2, 0.5, 0.8, 1.0]

    max_com = 0.0
    worst_info = ""
    n_selected = 0

    for dname, d_hat in dirs.items():
        k_max = np.pi / (L * np.max(np.abs(d_hat)))
        for kf in k_fracs:
            k_vec = kf * k_max * d_hat
            Dk = bloch.build_dynamical_matrix(k_vec)
            evals, evecs = np.linalg.eigh(Dk)
            omega = np.sqrt(np.maximum(evals, 0))

            for n in range(len(omega)):
                if omega[n] < omega_cut:
                    continue
                bp = np.sum(np.abs(Q_belt.T @ evecs[:, n])**2)
                if bp < bp_cut:
                    continue
                n_selected += 1
                com = np.sum(np.abs(Q_trans.T @ evecs[:, n])**2)
                if com > max_com:
                    max_com = com
                    worst_info = (f"{dname} k/k_max={kf}, "
                                  f"ω={omega[n]:.4f}, bp={bp:.3f}")

    print(f"Modes selected: {n_selected}")
    print(f"Max COM content: {max_com:.4e}")
    if worst_info:
        print(f"  worst case: {worst_info}")
    assert n_selected > 0, "No particle-like modes found (bp_cut too high?)"
    assert max_com < 0.10, (
        f"Type B loophole: COM content {max_com:.4f} >= 10% at {worst_info}"
    )


# =========================================================================
# Hop source (dipolar scaling)
# =========================================================================

def test_wp_hop_source_monopole_zero(wp_mesh):
    """Hop source (Type A → Type A) has M₀ = 0."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh
    nV = len(v)

    ci_a, ci_b = type_a[0], type_a[1]
    vf_a = get_belt_vertex_forces(ci_a, centers, v, f, cfi, L)
    vf_b = get_belt_vertex_forces(ci_b, centers, v, f, cfi, L)
    assert vf_a is not None and vf_b is not None

    hop = np.zeros(3 * nV)
    for vi, force in vf_b.items():
        hop[3*vi:3*vi+3] += force
    for vi, force in vf_a.items():
        hop[3*vi:3*vi+3] -= force

    M0 = np.zeros(3)
    for vi in range(nV):
        M0 += hop[3*vi:3*vi+3]
    M0_mag = np.linalg.norm(M0)

    print(f"M₀ = {M0_mag:.2e} (machine zero)")
    assert M0_mag < 1e-12, f"Hop source M₀ not zero: {M0_mag:.2e}"


def test_wp_hop_source_dipole_exists(wp_mesh):
    """Hop source has M₁ ~ O(1): dipole channel exists."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh
    nV = len(v)

    ci_a, ci_b = type_a[0], type_a[1]
    vf_a = get_belt_vertex_forces(ci_a, centers, v, f, cfi, L)
    vf_b = get_belt_vertex_forces(ci_b, centers, v, f, cfi, L)

    hop = np.zeros(3 * nV)
    all_verts = set()
    for vi, force in vf_b.items():
        hop[3*vi:3*vi+3] += force
        all_verts.add(vi)
    for vi, force in vf_a.items():
        hop[3*vi:3*vi+3] -= force
        all_verts.add(vi)

    # Dipole moment (origin at centroid of active vertices)
    verts_list = sorted(all_verts)
    r_ref = v[verts_list[0]]
    unwrapped = []
    for vi in verts_list:
        delta = v[vi] - r_ref
        delta -= L * np.round(delta / L)
        unwrapped.append(r_ref + delta)
    r0 = np.mean(unwrapped, axis=0)

    M1 = np.zeros((3, 3))
    for vi in all_verts:
        si = hop[3*vi:3*vi+3]
        ri = v[vi] - r0
        ri -= L * np.round(ri / L)
        M1 += np.outer(si, ri)

    M1_norm = np.linalg.norm(M1)
    print(f"|M₁| = {M1_norm:.4f}")
    assert M1_norm > 0.1, f"Dipole too small: |M₁| = {M1_norm:.4f}"


def test_wp_hop_source_k_squared(wp_mesh):
    """Acoustic overlap of hop source scales as k² (single pair, N=1 has only 2 Type A)."""
    v, e, f, cfi, centers, L, type_a, type_b = wp_mesh
    nV = len(v)

    ci_a, ci_b = type_a[0], type_a[1]
    vf_a = get_belt_vertex_forces(ci_a, centers, v, f, cfi, L)
    vf_b = get_belt_vertex_forces(ci_b, centers, v, f, cfi, L)

    hop = np.zeros(3 * nV)
    for vi, force in vf_b.items():
        hop[3*vi:3*vi+3] += force
    for vi, force in vf_a.items():
        hop[3*vi:3*vi+3] -= force

    norm_S_sq = np.dot(hop, hop)
    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    bz_dirs = {
        '[100]': np.array([1, 0, 0], dtype=float),
        '[110]': np.array([1, 1, 0], dtype=float) / np.sqrt(2),
        '[111]': np.array([1, 1, 1], dtype=float) / np.sqrt(3),
    }

    for name, dhat in bz_dirs.items():
        k_max = np.pi / (L * np.max(np.abs(dhat)))
        k_vals = np.logspace(-2.0, -0.7, 10) * k_max

        log_k, log_p = [], []
        for kk in k_vals:
            kvec = kk * dhat
            D = bloch.build_dynamical_matrix(kvec)
            w2, modes = np.linalg.eigh(D)
            order = np.argsort(w2)
            modes = modes[:, order]

            overlap = 0.0
            for branch in range(3):
                mode = modes[:, branch]
                overlap += np.abs(np.dot(hop, np.conj(mode)))**2
            overlap /= norm_S_sq

            if abs(overlap) > 1e-30:
                log_k.append(np.log(kk))
                log_p.append(np.log(abs(overlap)))

        if len(log_k) >= 2:
            slope = np.polyfit(log_k, log_p, 1)[0]
            print(f"  {name}: slope = {slope:.2f}")
            assert 1.8 < slope < 2.2, (
                f"Slope not ~2 in {name}: {slope:.2f}"
            )
