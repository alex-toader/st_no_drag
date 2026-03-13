#!/usr/bin/env python3
"""
Kinematic gap: belt frequencies above acoustic ceiling.

Tests:
  1. C15: gap ratio ω_belt_min / ω_acoustic_ceiling > 1
  2. C15: acousticness ceiling with COM-coherent threshold
  3. Kelvin: gap ratio > 1
  4. Both foams: subsonic particle velocity v_g < v_T

The kinematic gap is the primary structural protection for no-drag.
Belt excitations live at optical frequencies where no sound-like mode
exists. Even beyond the harmonic model, decay into acoustic phonons
requires energy-conserving final states — the gap prevents this.

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
)
from physics.bloch import DisplacementBloch


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def c15_gap_data():
    """Build C15 and compute gap quantities."""
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
    from core_math.analysis.no_drag import _DEFAULT_BZ_DIRS
    ac = compute_acoustic_ceiling(bloch, L, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    # Particle floor
    pf = compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
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
        'L': L,
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

    from core_math.analysis.no_drag import _DEFAULT_BZ_DIRS
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
    }


# =========================================================================
# Tests
# =========================================================================

def test_c15_gap_exists(c15_gap_data):
    """C15 gap ratio > 1: belt frequencies above acoustic ceiling."""
    gap = c15_gap_data['gap_ratio']
    assert gap > 1.0, (
        f"No kinematic gap on C15: ratio = {gap:.3f}"
    )
    # Expected ~1.6 with 3 BZ dirs (release value 1.21 uses tighter
    # COM-coherent ceiling from 12³ scan)
    assert gap > 1.2, (
        f"C15 gap ratio unexpectedly small: {gap:.3f}"
    )


def test_c15_belt_above_acoustic(c15_gap_data):
    """C15 particle floor and acoustic ceiling have correct values."""
    omega_edge = c15_gap_data['omega_edge']
    omega_belt = c15_gap_data['omega_belt_min']

    # Acoustic ceiling: ω₃ ≈ 0.69 (from release doc)
    assert 0.5 < omega_edge < 0.8, (
        f"ω_edge = {omega_edge:.4f}, expected ~0.69"
    )
    # Particle floor: ω_belt_min ≈ 1.14-1.16 (from release doc)
    assert 1.0 < omega_belt < 1.4, (
        f"ω_belt_min = {omega_belt:.4f}, expected ~1.15"
    )


def test_kelvin_gap_exists(kelvin_gap_data):
    """Kelvin gap ratio > 1: belt frequencies above acoustic ceiling."""
    gap = kelvin_gap_data['gap_ratio']
    assert gap > 1.0, (
        f"No kinematic gap on Kelvin: ratio = {gap:.3f}"
    )


def test_c15_belt_basis_dimensions(c15_gap_data):
    """C15 belt basis: 96D enriched, 16D particle (cos2θ×n̂)."""
    Q_belt = c15_gap_data['Q_belt']
    n_particle = c15_gap_data['n_particle']

    assert Q_belt.shape[1] == 96, (
        f"Belt basis rank = {Q_belt.shape[1]}, expected 96"
    )
    assert n_particle == 16, (
        f"Particle subspace dim = {n_particle}, expected 16"
    )


def test_c15_subsonic(c15_gap_data):
    """C15 particle velocity v_g < v_T (subsonic)."""
    d = c15_gap_data
    from core_math.analysis.no_drag import _DEFAULT_BZ_DIRS
    vel = compute_projected_velocities(
        d['bloch'], d['Q_belt'], d['M_transfer'], d['n_particle'],
        d['L'], bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    ratio = vel['vg_max_over_vT']
    assert ratio < 1.0, (
        f"C15 particle supersonic: v_g/v_T = {ratio:.3f}"
    )
    # Expected ~0.89 from release doc
    assert ratio > 0.3, (
        f"C15 v_g/v_T suspiciously small: {ratio:.3f}"
    )


def test_kelvin_subsonic(kelvin_gap_data):
    """Kelvin particle velocity v_g < v_T (subsonic)."""
    d = kelvin_gap_data
    from core_math.analysis.no_drag import _DEFAULT_BZ_DIRS
    vel = compute_projected_velocities(
        d['bloch'], d['Q_belt'], d['M_transfer'], d['n_particle'],
        d['L'], bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)

    ratio = vel['vg_max_over_vT']
    assert ratio < 1.0, (
        f"Kelvin particle supersonic: v_g/v_T = {ratio:.3f}"
    )
    # Expected ~0.55 from release doc
    assert ratio > 0.1, (
        f"Kelvin v_g/v_T suspiciously small: {ratio:.3f}"
    )
