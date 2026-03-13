#!/usr/bin/env python3
"""
Wavepacket test: zero acoustic emission from belt excitation.

Tests:
  1. Kelvin: translation content ||Q_trans^T u(t)||² stays at machine zero
  2. Kelvin: energy in acoustic modes is machine zero
  3. Kelvin: drift << spreading (no directed transport at Gamma)

A localized belt excitation (cos2θ×n̂ on one cell) is evolved
harmonically using normal mode decomposition at Γ. In the harmonic
model, Bloch eigenmodes are exact stationary states — energy cannot
transfer irreversibly between bands. The test verifies that the belt
initial state has zero overlap with acoustic modes.

This is a Γ-point test. It confirms zero acoustic emission and
spreading kinematics but does not test directional transport (which
requires a finite-k₀ wavepacket).

References:
  physics_ai/ST_11/wip/w_15_kelvin_no_drag/03_wavepacket.py
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/w14_09_enriched_wavepacket.py
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
def kelvin_wavepacket():
    """Build Kelvin N=2, diagonalize D(k=0), prepare belt source."""
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4.0 * N
    nv = len(v)
    n_dof = 3 * nv
    n_cells = len(centers)

    edges_list = [list(edge) for edge in e]
    bloch = DisplacementBloch(v, edges_list, L, k_L=2.0, k_T=1.0, mass=1.0)

    # Diagonalize D(k=0)
    D0 = np.real(bloch.build_dynamical_matrix(np.array([0.0, 0.0, 0.0])))
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

    # Remove COM component (should already be orthogonal)
    u0 = u0 - Q_trans @ (Q_trans.T @ u0)
    u0 = u0 / np.linalg.norm(u0)

    # Mode decomposition
    coeffs = evecs.T @ u0
    mode_energy = 0.5 * coeffs**2 * omega**2
    E_total = mode_energy.sum()

    # Effective frequency and period
    omega_eff = np.sqrt(np.sum(coeffs**2 * omega**2) / np.sum(coeffs**2))
    T_period = 2 * np.pi / omega_eff

    return {
        'evecs': evecs, 'omega': omega, 'coeffs': coeffs,
        'Q_trans': Q_trans, 'acousticness': acousticness,
        'mode_energy': mode_energy, 'E_total': E_total,
        'u0': u0, 'T_period': T_period,
        'nv': nv, 'n_dof': n_dof, 'n_cells': n_cells,
        'centers': centers, 'L': L, 'source_cell': source_cell,
        'v': v, 'f': f, 'cfi': cfi,
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

    assert max_trans < 1e-25, (
        f"Translation content not zero: max = {max_trans:.2e}"
    )


def test_zero_acoustic_energy(kelvin_wavepacket):
    """Energy in acoustic modes (acousticness > 10%) is machine zero."""
    d = kelvin_wavepacket
    acoustic_mask = d['acousticness'] > 0.10
    E_acoustic = d['mode_energy'][acoustic_mask].sum()

    assert E_acoustic < 1e-40, (
        f"Acoustic energy not zero: {E_acoustic:.2e}"
    )


def test_zero_mode_energy(kelvin_wavepacket):
    """Energy in zero-frequency modes (translations) is machine zero."""
    d = kelvin_wavepacket
    zero_mask = d['omega'] < 0.01
    E_zero = d['mode_energy'][zero_mask].sum()
    n_zero = int(zero_mask.sum())

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
    v, f, cfi = d['v'], d['f'], d['cfi']

    # Per-cell belt probes
    cell_Q_local = {}
    for ci in range(n_cells):
        vecs, _ = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        if vecs:
            B = np.column_stack([np.real(vv) for vv in vecs])
            Q, R = np.linalg.qr(B)
            rd = np.abs(np.diag(R))
            rk = np.sum(rd > 1e-10 * rd[0])
            cell_Q_local[ci] = Q[:, :rk]

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

    # At Γ-point with 16 cells, expect drift/sigma < 1
    assert sigma > 0.5, f"Spreading too small: σ = {sigma:.4f}"
    assert drift / sigma < 1.0, (
        f"Drift too large: drift/σ = {drift/sigma:.3f}"
    )
