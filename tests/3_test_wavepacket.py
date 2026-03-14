#!/usr/bin/env python3
"""
Wavepacket test: zero acoustic emission from belt excitation.

A localized belt excitation (cos2θ×n̂ on one cell) is evolved
harmonically using normal mode decomposition at Γ. In the harmonic
model, Bloch eigenmodes are exact stationary states — energy cannot
transfer irreversibly between bands. The test verifies that the belt
initial state has zero overlap with acoustic modes.

This is a Γ-point test. It confirms zero acoustic emission and
spreading kinematics but does not test directional transport (which
requires a finite-k₀ wavepacket).

Both displacement u(t) and velocity v(t) have zero acoustic content.
Result is independent of which cell sources the belt excitation.

NOTES:
  - Threshold difference (1e-25 for translation content vs 1e-40 for
    acoustic energy): translation content ||Q^T u(t)||² accumulates
    floating-point noise from ~285 modes (each contributing ~1e-32),
    giving ~1e-29. Acoustic energy E = Σ c²ₙ ω²ₙ/2 over 3 modes where
    c ~ 1e-16 and ω ~ 1e-8 (numerical zero), giving ~1e-46. Both are
    machine zero, different mechanisms.
  - D(k=0) is exactly real (no phase factors at k=0). The assert verifies
    this rather than silently truncating with np.real().
  - Energy conservation is not tested: normal-mode evolution u(t) = Σ cₙ eₙ cos(ωₙt)
    is analytically exact (dE/E = 0.00 verified). A test would just check numpy's cos.
  - C15 wavepacket not tested: Kelvin is sufficient for the harmonic theorem
    claim. C15 would be much slower without new insight.
  - Finite-k wavepacket NOT tested here. D(k=0) of the N=2 supercell has
    only 3 acoustic modes (translations at ω=0) — folded k-points at BZ edge
    have zero acousticness (not COM-coherent). A finite-k₀ modulation just
    changes which optical modes are excited, not the acoustic overlap.
    Finite-k protection comes from the kinematic gap (test 2), not from
    eigenmode orthogonality. Belt-acoustic orthogonality (selection rule)
    holds only at k=0; at k≠0, acoustic modes have O(1) overlap with Q_belt.

RAW OUTPUT (pytest -v -s, Mar 2026):
  test_zero_translation_content       max ||Q_trans^T u(t)||² = 6.65e-31 over 4 periods
  test_zero_acoustic_energy           E_acoustic (acousticness > 10%) = 1.39e-46
  test_zero_mode_energy               Zero modes: 3, E_zero = 1.39e-46
  test_drift_smaller_than_spreading   drift = 2.3918, sigma = 3.9207, drift/sigma = 0.610
  test_zero_velocity_acoustic_content max ||Q_trans^T v(t)||² = 1.99e-31 over 4 periods
  test_source_cell_independence       max E_acoustic over 16 source cells = 1.39e-46
  test_belt_basis_covers_source       ||u0 - Q_belt Q_belt^T u0|| = 6.64e-16
  test_belt_char_vs_source_projection belt_char mean = 0.0721, max source proj = 2.99e-31
  test_source_projection_scales_k_squared proj/k² = 0.027 (spread 1.2%), scaling k² confirmed
  test_c15_particle_com_content_finite_k  max COM = 8.11e-02 at [011] k/k_max=1.0 (Z16 loophole < 10%)

References:
  physics_ai/ST_11/wip/w_15_kelvin_no_drag/03_wavepacket.py
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/w14_09_enriched_wavepacket.py
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/w14_14_z16_admixture.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)
from core_math.analysis.no_drag import build_enriched_belt_vectors, build_belt_basis
from physics.bloch import DisplacementBloch


@pytest.fixture(scope="module")
def kelvin_wavepacket():
    """Build Kelvin N=2, diagonalize D(k=0), prepare belt source."""
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4.0 * N
    nv = len(v)
    n_dof = 3 * nv
    n_cells = len(centers)

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    # Diagonalize D(k=0) — must be real at k=0
    D0_complex = bloch.build_dynamical_matrix(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(D0_complex.imag, 0, atol=1e-14), (
        f"D(k=0) has imaginary part: max|imag| = {np.max(np.abs(D0_complex.imag)):.2e}"
    )
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

    acousticness = np.sum(np.abs(Q_trans.T @ evecs)**2, axis=0)

    # Belt source on cell closest to box center
    box_center = np.array([L/2, L/2, L/2])
    dists_to_center = np.linalg.norm(centers - box_center, axis=1)
    source_cell = np.argmin(dists_to_center)

    _, old_vec = build_enriched_belt_vectors(
        source_cell, centers, v, f, cfi, L)
    u0 = old_vec / np.linalg.norm(old_vec)

    # Remove COM component (defensive — should already be orthogonal)
    u0 = u0 - Q_trans @ (Q_trans.T @ u0)
    u0 = u0 / np.linalg.norm(u0)

    # Mode decomposition
    coeffs = evecs.T @ u0
    mode_energy = 0.5 * coeffs**2 * omega**2
    E_total = mode_energy.sum()

    # Effective frequency and period
    omega_eff = np.sqrt(np.sum(coeffs**2 * omega**2) / np.sum(coeffs**2))
    T_period = 2 * np.pi / omega_eff

    # Per-cell belt probes (shared by drift + source_cell_independence)
    cell_Q_local = {}
    for ci in range(n_cells):
        vecs, _ = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        if vecs:
            B = np.column_stack(vecs)
            Q, R = np.linalg.qr(B)
            rd = np.abs(np.diag(R))
            rk = np.sum(rd > 1e-10 * rd[0])
            cell_Q_local[ci] = Q[:, :rk]

    # Global belt basis
    all_cells = list(range(n_cells))
    Q_belt, _, _ = build_belt_basis(all_cells, centers, v, f, cfi, L)

    return {
        'evecs': evecs, 'omega': omega, 'coeffs': coeffs,
        'Q_trans': Q_trans, 'acousticness': acousticness,
        'mode_energy': mode_energy, 'E_total': E_total,
        'u0': u0, 'T_period': T_period,
        'nv': nv, 'n_dof': n_dof, 'n_cells': n_cells,
        'centers': centers, 'L': L, 'source_cell': source_cell,
        'v': v, 'e': e, 'f': f, 'cfi': cfi,
        'cell_Q_local': cell_Q_local,
        'bloch': bloch, 'Q_belt': Q_belt,
    }


def test_zero_translation_content(kelvin_wavepacket):
    """||Q_trans^T u(t)||² stays at machine zero for all t."""
    d = kelvin_wavepacket
    evecs, omega, coeffs = d['evecs'], d['omega'], d['coeffs']
    Q_trans, T = d['Q_trans'], d['T_period']

    # Evolve over 4 periods, sample at 40 points per period
    n_steps = 160
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


def test_zero_acoustic_energy(kelvin_wavepacket):
    """Energy in acoustic modes (acousticness > 10%) is machine zero."""
    d = kelvin_wavepacket
    acoustic_mask = d['acousticness'] > 0.10
    E_acoustic = d['mode_energy'][acoustic_mask].sum()

    print(f"E_acoustic (acousticness > 10%) = {E_acoustic:.2e}")
    assert E_acoustic < 1e-40, (
        f"Acoustic energy not zero: {E_acoustic:.2e}"
    )


def test_zero_mode_energy(kelvin_wavepacket):
    """Energy in zero-frequency modes (translations) is machine zero."""
    d = kelvin_wavepacket
    zero_mask = d['omega'] < 0.01
    E_zero = d['mode_energy'][zero_mask].sum()
    n_zero = int(zero_mask.sum())

    print(f"Zero modes: {n_zero}, E_zero = {E_zero:.2e}")
    assert n_zero == 3, f"Expected 3 zero modes, got {n_zero}"
    assert E_zero < 1e-40, (
        f"Zero-mode energy not zero: {E_zero:.2e}"
    )


def test_drift_smaller_than_spreading(kelvin_wavepacket):
    """Drift << spreading at Γ-point (no directed transport)."""
    d = kelvin_wavepacket
    evecs, omega, coeffs = d['evecs'], d['omega'], d['coeffs']
    centers, L = d['centers'], d['L']
    source_cell = d['source_cell']
    T = d['T_period']
    n_cells = d['n_cells']
    cell_Q_local = d['cell_Q_local']

    # Displacement vectors from source
    dr_vecs = np.zeros((n_cells, 3))
    for j in range(n_cells):
        delta = centers[j] - centers[source_cell]
        delta = delta - L * np.round(delta / L)
        dr_vecs[j] = delta

    # Measure at t = 2T
    t_2T = 2 * T
    u_t = evecs @ (coeffs * np.cos(omega * t_2T))

    w = np.zeros(n_cells)
    for j in range(n_cells):
        if j in cell_Q_local:
            proj = cell_Q_local[j].T @ u_t
            w[j] = np.dot(proj, proj)

    total_w = w.sum()
    cv = np.array([np.sum(w * dr_vecs[:, d]) / total_w for d in range(3)])
    drift = np.linalg.norm(cv)
    dev = dr_vecs - cv[np.newaxis, :]
    sigma = np.sqrt(np.sum(w * np.sum(dev**2, axis=1)) / total_w)

    print(f"drift = {drift:.4f}, sigma = {sigma:.4f}, drift/sigma = {drift/sigma:.3f}")
    # Minimum spreading: half the nearest-neighbour distance in a BCC of n_cells
    sigma_min = L / (2 * n_cells**(1./3))
    assert sigma > sigma_min, (
        f"Spreading too small: σ = {sigma:.4f} < L/(2·N) = {sigma_min:.4f}"
    )
    assert drift / sigma < 1.0, (
        f"Drift too large: drift/σ = {drift/sigma:.3f}"
    )


def test_zero_velocity_acoustic_content(kelvin_wavepacket):
    """Momentum-space check: ||Q_trans^T v(t)||² stays at machine zero."""
    d = kelvin_wavepacket
    evecs, omega, coeffs = d['evecs'], d['omega'], d['coeffs']
    Q_trans, T = d['Q_trans'], d['T_period']

    n_steps = 160
    times = np.linspace(0, 4 * T, n_steps)
    max_v_trans = 0.0

    for t in times:
        v_t = evecs @ (-coeffs * omega * np.sin(omega * t))
        v_trans_sq = np.linalg.norm(Q_trans.T @ v_t)**2
        max_v_trans = max(max_v_trans, v_trans_sq)

    print(f"max ||Q_trans^T v(t)||² = {max_v_trans:.2e} over 4 periods")
    assert max_v_trans < 1e-25, (
        f"Velocity translation content not zero: max = {max_v_trans:.2e}"
    )


def test_source_cell_independence(kelvin_wavepacket):
    """E_acoustic is machine zero regardless of which cell sources the belt."""
    d = kelvin_wavepacket
    evecs, omega = d['evecs'], d['omega']
    Q_trans, acousticness = d['Q_trans'], d['acousticness']
    centers, v, f, cfi, L = d['centers'], d['v'], d['f'], d['cfi'], d['L']
    n_cells = d['n_cells']

    ac_mask = acousticness > 0.10

    max_E_ac = 0.0
    for ci in range(n_cells):
        _, old_vec = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        if old_vec is None:
            continue
        u0_ci = old_vec / np.linalg.norm(old_vec)
        u0_ci = u0_ci - Q_trans @ (Q_trans.T @ u0_ci)
        u0_ci = u0_ci / np.linalg.norm(u0_ci)
        c_ci = evecs.T @ u0_ci
        E_ac = 0.5 * np.sum(c_ci[ac_mask]**2 * omega[ac_mask]**2)
        max_E_ac = max(max_E_ac, E_ac)

    print(f"max E_acoustic over {n_cells} source cells = {max_E_ac:.2e}")
    assert max_E_ac < 1e-40, (
        f"Acoustic energy nonzero for some source cell: {max_E_ac:.2e}"
    )


def test_belt_basis_covers_source(kelvin_wavepacket):
    """Belt source u0 lies entirely in Q_belt subspace."""
    d = kelvin_wavepacket
    centers, v, f, cfi, L = d['centers'], d['v'], d['f'], d['cfi'], d['L']
    n_cells = d['n_cells']
    u0 = d['u0']

    all_cells = list(range(n_cells))
    Q_belt, _, _ = build_belt_basis(all_cells, centers, v, f, cfi, L)

    u0_proj = Q_belt @ (Q_belt.T @ u0)
    residual = np.linalg.norm(u0 - u0_proj)

    print(f"||u0 - Q_belt Q_belt^T u0|| = {residual:.2e}")
    assert residual < 1e-12, (
        f"Belt source not in Q_belt subspace: residual = {residual:.2e}"
    )


def test_belt_char_vs_source_projection(kelvin_wavepacket):
    """Acoustic modes overlap belt subspace (~7%) but all belt sources project to zero.

    Belt character |Q_belt^T e_ac|² measures geometric overlap between the
    acoustic eigenvector and the 96D belt subspace. Source projection
    |e_ac^T u0|² measures whether a specific belt excitation pumps acoustic
    modes. These are different: e_ac can have nonzero belt subspace component
    while being orthogonal to every belt source (selection rule).
    """
    d = kelvin_wavepacket
    evecs, omega = d['evecs'], d['omega']
    Q_trans, Q_belt = d['Q_trans'], d['Q_belt']
    centers, v, f, cfi, L = d['centers'], d['v'], d['f'], d['cfi'], d['L']
    n_cells = d['n_cells']

    ac_mask = omega < 0.01
    ac_indices = np.where(ac_mask)[0]

    # Belt character of acoustic modes
    belt_char = np.sum(np.abs(Q_belt.T @ evecs[:, ac_indices])**2, axis=0)
    mean_belt_char = np.mean(belt_char)

    # All 16 belt sources: projection onto acoustic modes
    max_source_proj = 0.0
    for ci in range(n_cells):
        _, vec_ci = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        if vec_ci is None:
            continue
        u_ci = vec_ci / np.linalg.norm(vec_ci)
        u_ci = u_ci - Q_trans @ (Q_trans.T @ u_ci)
        u_ci = u_ci / np.linalg.norm(u_ci)
        proj_ac = sum(np.dot(evecs[:, m], u_ci)**2 for m in ac_indices)
        max_source_proj = max(max_source_proj, proj_ac)

    print(f"belt_char of acoustic modes: {belt_char} (mean {mean_belt_char:.4f})")
    print(f"max source projection onto acoustic: {max_source_proj:.2e}")

    # Belt character is O(1) — acoustic modes see belt DOFs
    assert mean_belt_char > 0.01, (
        f"Belt character unexpectedly small: {mean_belt_char:.4f}"
    )
    # But all source projections are machine zero — selection rule
    assert max_source_proj < 1e-25, (
        f"Source projection not zero: {max_source_proj:.2e}"
    )


def test_source_projection_scales_k_squared(kelvin_wavepacket):
    """At finite k, belt source projection onto acoustic modes grows as k².

    At k=0, selection rule gives zero projection (machine zero).
    At k≠0, the selection rule breaks and projection grows as k² —
    this is the dipolar scaling (M₀ = 0, leading multipole is dipole ~ k).
    The kinematic gap provides protection at finite k (test 2).
    """
    d = kelvin_wavepacket
    bloch, u0 = d['bloch'], d['u0']

    k_values = [0.01, 0.02, 0.05, 0.1]
    proj_over_k2 = []

    for k_mag in k_values:
        k = np.array([k_mag, 0.0, 0.0])
        Dk = bloch.build_dynamical_matrix(k)
        evalsk, evecsk = np.linalg.eigh(Dk)
        omegak = np.sqrt(np.maximum(evalsk, 0))

        # Lowest 3 modes = acoustic branch at this k
        sorted_idx = np.argsort(omegak)
        low3 = sorted_idx[:3]

        coeffs_k = evecsk.T.conj() @ u0
        sum_proj = np.sum(np.abs(coeffs_k[low3])**2)
        proj_over_k2.append(sum_proj / k_mag**2)

    # proj/k² should be approximately constant (dipolar scaling)
    ratios = np.array(proj_over_k2)
    spread = (ratios.max() - ratios.min()) / np.mean(ratios)

    print(f"proj/k² at k={k_values}: {[f'{r:.4e}' for r in ratios]}")
    print(f"mean = {np.mean(ratios):.4e}, spread = {spread:.3f}")

    assert spread < 0.20, (
        f"proj/k² not constant (spread {spread:.3f}): scaling is not k²"
    )
    # Coefficient should be O(0.01) — not zero, not huge
    assert np.mean(ratios) > 1e-4, (
        f"proj/k² too small: {np.mean(ratios):.2e}"
    )
    assert np.mean(ratios) < 1.0, (
        f"proj/k² too large: {np.mean(ratios):.2e}"
    )


def test_c15_particle_com_content_finite_k():
    """Z16 loophole bound: particle-band COM content < 10% at all k.

    C15 has Z16 cells with M₀ ≠ 0 (test 1). At finite k, hybridization
    could mix belt and acoustic character through the Z16 channel.
    This test bounds the leakage: particle-like eigenstates of D(k)
    have COM-coherent translation content < 10% across the BZ.

    COM content = Σ_d |⟨ê_d|ψ_n(k)⟩|² (uniform displacement probe,
    exact for cell-periodic Bloch convention).

    Thresholds computed from data:
      omega_cut = omega_belt_min from particle floor scan (belt modes start here)
      bp_cut = 0.25 (belt projection; random baseline ~ rank(Q_belt)/n_dof)

    BZ convention: k_max = π/(L·max|d̂|) per direction, consistent with
    compute_acoustic_ceiling and compute_particle_floor.

    Source: physics_ai/ST_11/src/1_foam/tests/physics/no_drag/w14_14_z16_admixture.py
    """
    from core_math.analysis.no_drag import compute_particle_floor

    N = 1
    L_cell = 4.0
    L = L_cell * N
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell=L_cell)
    centers = np.array(get_c15_points(N, L_cell))
    nv = len(v)
    n_dof = 3 * nv

    # Belt basis (Z12 cells only — Z16 cells have different belt structure)
    n_cells = len(cfi)
    z12_cells = [ci for ci in range(n_cells)
                 if len(cfi[ci]) == 12]
    Q_belt, Q_particle, M_transfer = build_belt_basis(
        z12_cells, centers, v, f, cfi, L)
    n_particle = Q_particle.shape[1]

    # Q_trans (uniform displacement — 3 orthonormal vectors)
    Q_trans = np.zeros((n_dof, 3))
    for d in range(3):
        q = np.zeros(n_dof)
        for i in range(nv):
            q[3 * i + d] = 1.0
        Q_trans[:, d] = q / np.linalg.norm(q)

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    # omega_cut from particle floor (belt modes above this frequency)
    res_floor = compute_particle_floor(
        bloch, Q_belt, M_transfer, n_particle, L)
    omega_cut = res_floor['omega_belt_min_global']

    # bp_cut: belt projection threshold
    bp_random = Q_belt.shape[1] / n_dof
    bp_cut = 0.25  # just above random for Z12-only belt basis (~0.235)

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
                # Belt projection (Q_belt is real, evec may be complex)
                bp = np.sum(np.abs(Q_belt.T @ evecs[:, n])**2)
                if bp < bp_cut:
                    continue
                # COM content
                com = np.sum(np.abs(Q_trans.T @ evecs[:, n])**2)
                if com > max_com:
                    max_com = com
                    worst_info = (f"{dname} k/k_max={kf}, "
                                  f"ω={omega[n]:.4f}, bp={bp:.3f}")

    print(f"Max COM content: {max_com:.4e}")
    print(f"  worst case: {worst_info}")
    assert max_com < 0.10, (
        f"Z16 loophole: COM content {max_com:.4f} >= 10% at {worst_info}"
    )
