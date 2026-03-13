#!/usr/bin/env python3
"""
MD validation: cubic anharmonicity energy transfer.

Tests:
  1. Harmonic baseline: E_low stays at machine zero
  2. Cubic: E_low/ε² scaling (dimensionless ratio constant across ε)
  3. COM drift stays at machine zero
  4. Energy conservation

Time-domain Verlet integration of a belt excitation under cubic
anharmonicity V₃ = (α/6) Σ_edges (δr)³. Measures energy transfer
into low-ω modes (proxy for acoustic sector).

Uses Kelvin N=3 (972 DOF, 18 low-omega modes). N=2 has 0 low-omega
modes at Γ, making the test vacuous.

References:
  physics_ai/ST_11/wip/w_15_kelvin_no_drag/04_md_cubic.py
  physics_ai/ST_11/src/1_foam/tests/physics/no_drag/16_md_drag_measurement.py
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
from core_math.dynamics.md_foam import (
    harmonic_force_spring, cubic_force, verlet_step, prepare_edges,
    sector_energy, harmonic_energy_spring, cubic_energy, remove_com
)


def _build_Q_trans(nv):
    n_dof = 3 * nv
    Q = np.zeros((n_dof, 3))
    for d in range(3):
        q = np.zeros(n_dof)
        for i in range(nv):
            q[3 * i + d] = 1.0
        Q[:, d] = q / np.linalg.norm(q)
    return Q


def _purge_low_omega(u, evecs, omega, omega_cut):
    low_mask = (omega > 0.01) & (omega < omega_cut)
    evecs_low = evecs[:, low_mask]
    proj = evecs_low @ (evecs_low.T @ u)
    return u - proj


def _com_drift(u, v, Q_trans):
    pu = Q_trans.T @ u
    pv = Q_trans.T @ v
    return np.dot(pu, pu) + np.dot(pv, pv)


def _run_md(u0, v0, force_fn, dt, n_steps, evecs, omega,
            Q_trans, omega_cut, check_every):
    u, v = u0.copy(), v0.copy()
    a = force_fn(u)
    n_checks = n_steps // check_every + 2
    E_low = np.zeros(n_checks)
    com = np.zeros(n_checks)
    E_low[0] = sector_energy(u, v, evecs, omega, omega_cut)
    com[0] = _com_drift(u, v, Q_trans)
    idx = 1
    for step in range(n_steps):
        u, v, a = verlet_step(u, v, a, force_fn, dt)
        if (step + 1) % check_every == 0 and idx < n_checks:
            E_low[idx] = sector_energy(u, v, evecs, omega, omega_cut)
            com[idx] = _com_drift(u, v, Q_trans)
            idx += 1
    return E_low[:idx], com[:idx], u, v


@pytest.fixture(scope="module")
def md_system():
    """Build Kelvin N=3, eigendecompose, prepare belt init."""
    N = 3
    L = N * 4.0
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    nv = len(v)
    n_dof = 3 * nv

    edge_info = prepare_edges(v, e, L)
    k_L, k_T = 2.0, 1.0

    edges_list = [list(edge) for edge in e]
    bloch = DisplacementBloch(v, edges_list, L, k_L=k_L, k_T=k_T, mass=1.0)
    D = bloch.build_dynamical_matrix(np.zeros(3)).real
    evals, evecs = np.linalg.eigh(D)
    omega = np.sqrt(np.maximum(evals, 0))

    Q_trans = _build_Q_trans(nv)
    omega_cut = 0.8022

    # Source belt vector
    box_center = np.array([L/2, L/2, L/2])
    dists = np.linalg.norm(centers - box_center, axis=1)
    source_cell = np.argmin(dists)
    _, old_vec = build_enriched_belt_vectors(
        source_cell, centers, v, f, cfi, L)
    u0 = 0.1 * old_vec / np.linalg.norm(old_vec)
    v0 = np.zeros(n_dof)
    u0, v0 = remove_com(u0, v0, nv)
    u0 = _purge_low_omega(u0, evecs, omega, omega_cut)
    u0, v0 = remove_com(u0, v0, nv)

    # MD parameters
    omega_belt_min = omega[omega > 1.0].min()
    T_belt = 2 * np.pi / omega_belt_min
    dt = 0.001
    n_periods = 8
    n_steps = int(n_periods * T_belt / dt)
    check_every = int(0.2 * T_belt / dt)

    return {
        'u0': u0, 'v0': v0, 'evecs': evecs, 'omega': omega,
        'Q_trans': Q_trans, 'omega_cut': omega_cut,
        'edge_info': edge_info, 'k_L': k_L, 'k_T': k_T,
        'dt': dt, 'n_steps': n_steps, 'check_every': check_every,
        'nv': nv, 'ell0': edge_info['edge_lengths'].mean(),
    }


def test_harmonic_baseline(md_system):
    """Harmonic: E_low stays at machine noise."""
    d = md_system

    def force_harm(u):
        return harmonic_force_spring(u, d['edge_info'], d['k_L'], d['k_T'])

    E_low, com, _, _ = _run_md(
        d['u0'], d['v0'], force_harm, d['dt'], d['n_steps'],
        d['evecs'], d['omega'], d['Q_trans'], d['omega_cut'],
        d['check_every'])

    assert E_low.max() < 1e-25, (
        f"Harmonic E_low not zero: max = {E_low.max():.2e}"
    )


def test_eps_squared_scaling(md_system):
    """E_low/ε² is constant across ε values (scaling confirmed)."""
    d = md_system
    k_L, ell0 = d['k_L'], d['ell0']

    eps_values = [0.03, 0.1]
    sat_over_eps2 = []

    for eps in eps_values:
        alpha = eps * k_L / ell0

        def make_force(a):
            def fn(u):
                return harmonic_force_spring(u, d['edge_info'], d['k_L'], d['k_T']) + \
                       cubic_force(u, d['edge_info'], a)
            return fn

        E_low, _, _, _ = _run_md(
            d['u0'], d['v0'], make_force(alpha), d['dt'], d['n_steps'],
            d['evecs'], d['omega'], d['Q_trans'], d['omega_cut'],
            d['check_every'])

        # Period-average the last half
        n_half = len(E_low) // 2
        E_sat = E_low[n_half:].mean()
        sat_over_eps2.append(E_sat / eps**2)

    # Check scaling: spread < 30%
    ratios = np.array(sat_over_eps2)
    spread = (ratios.max() - ratios.min()) / ratios.mean()
    assert spread < 0.30, (
        f"ε² scaling fails: spread = {100*spread:.1f}%, "
        f"ratios = {ratios}"
    )


def test_com_zero(md_system):
    """COM drift stays at machine zero under cubic anharmonicity."""
    d = md_system
    alpha = 0.1 * d['k_L'] / d['ell0']

    def force_fn(u):
        return harmonic_force_spring(u, d['edge_info'], d['k_L'], d['k_T']) + \
               cubic_force(u, d['edge_info'], alpha)

    _, com, _, _ = _run_md(
        d['u0'], d['v0'], force_fn, d['dt'], d['n_steps'],
        d['evecs'], d['omega'], d['Q_trans'], d['omega_cut'],
        d['check_every'])

    assert com.max() < 1e-20, (
        f"COM drift not zero: max = {com.max():.2e}"
    )


def test_energy_conservation(md_system):
    """Energy is conserved to < 10⁻⁵ relative drift."""
    d = md_system
    eps = 0.1
    alpha = eps * d['k_L'] / d['ell0']

    def force_fn(u):
        return harmonic_force_spring(u, d['edge_info'], d['k_L'], d['k_T']) + \
               cubic_force(u, d['edge_info'], alpha)

    _, _, u_final, v_final = _run_md(
        d['u0'], d['v0'], force_fn, d['dt'], d['n_steps'],
        d['evecs'], d['omega'], d['Q_trans'], d['omega_cut'],
        d['check_every'])

    E0 = harmonic_energy_spring(d['u0'], d['v0'], d['edge_info'],
                                 d['k_L'], d['k_T']) + \
         cubic_energy(d['u0'], d['edge_info'], alpha)
    Ef = harmonic_energy_spring(u_final, v_final, d['edge_info'],
                                 d['k_L'], d['k_T']) + \
         cubic_energy(u_final, d['edge_info'], alpha)

    rel_drift = abs(Ef - E0) / abs(E0)
    assert rel_drift < 1e-5, (
        f"Energy not conserved: relative drift = {rel_drift:.2e}"
    )
