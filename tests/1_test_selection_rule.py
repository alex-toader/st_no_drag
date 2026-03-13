#!/usr/bin/env python3
"""
Selection rule M₀ = 0 for belt m=2 mode.

Tests:
  1. C15: M₀ = 0 on all 16 Z12 cells (exact by symmetry)
  2. C15: M₀ ≠ 0 on Z16 cells (negative control)
  3. Kelvin: M₀ = 0 on all cells (all identical truncated octahedra)

The selection rule M₀ = Σ_j P_j A_j n̂_j is the net force from m=2
belt pressure cos(2θ) distributed over belt faces. M₀ = 0 means no
monopole coupling between the belt mode and uniform translations.

On Z12 cells: the normal-tilt Fourier spectrum n_ax has no m=2
component, so m=2 pressure × normal tilt cannot produce a net force.
On Z16 cells: n_ax has an m=2 component → beats with pressure → M₀ ≠ 0.
Z16 cells are not on the transport corridor (Z12→Z12 hopping dominates).

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
from core_math.analysis.no_drag import compute_selection_rule


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
    return v, f, cfi, centers, L, cell_type


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
    v, f, cfi, centers, L, cell_type = c15_mesh

    z12_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    assert len(z12_cells) == 16, f"Expected 16 Z12 cells, got {len(z12_cells)}"

    m0_values = []
    for ci in z12_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Z12 cell {ci} has no belt"
        m0_values.append(sr['M0_mag'])

    m0_max = max(m0_values)
    assert m0_max < 1e-12, (
        f"M₀ not zero on Z12: max |M₀| = {m0_max:.2e}"
    )


def test_c15_z16_nonzero(c15_mesh):
    """M₀ ≠ 0 on Z16 cells (negative control)."""
    v, f, cfi, centers, L, cell_type = c15_mesh

    z16_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 16]
    assert len(z16_cells) == 8, f"Expected 8 Z16 cells, got {len(z16_cells)}"

    m0_values = []
    for ci in z16_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Z16 cell {ci} has no belt"
        m0_values.append(sr['M0_mag'])

    m0_min = min(m0_values)
    # At L_cell=1.0, Z16 M₀ ≈ 0.033 (scales as L_cell²)
    assert m0_min > 0.01, (
        f"Z16 M₀ should be nonzero: min |M₀| = {m0_min:.6f}"
    )


def test_c15_z12_no_m2_normal_tilt(c15_mesh):
    """Z12 normal-tilt Fourier spectrum has no m=2 component.

    This is WHY M₀ = 0: the n_ax spectrum at m=2 is zero, so
    cos(2θ) pressure cannot beat with normal tilt to produce force.
    """
    v, f, cfi, centers, L, cell_type = c15_mesh

    z12_cells = [ci for ci in range(len(centers)) if cell_type[ci] == 12]
    max_m2 = 0.0

    for ci in z12_cells:
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        # n_ax_spectrum[2] is the m=2 Fourier component of normal axial tilt
        m2_mag = abs(sr['n_ax_spectrum'][2])
        max_m2 = max(max_m2, m2_mag)

    assert max_m2 < 1e-12, (
        f"Z12 should have no m=2 normal tilt: max |n_ax[2]| = {max_m2:.2e}"
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
    assert m0_max < 1e-12, (
        f"M₀ not zero on Kelvin: max |M₀| = {m0_max:.2e}"
    )


def test_c15_belt_counts(c15_mesh):
    """All 24 C15 cells have belts. Z12 has N=6, Z16 has N=8."""
    v, f, cfi, centers, L, cell_type = c15_mesh

    belt_n = {}
    for ci in range(len(centers)):
        sr = compute_selection_rule(ci, centers, v, f, cfi, L)
        assert sr is not None, f"Cell {ci} has no belt"
        belt_n[ci] = sr['n_belt']

    z12_n = set(belt_n[ci] for ci in range(len(centers))
                if cell_type[ci] == 12)
    z16_n = set(belt_n[ci] for ci in range(len(centers))
                if cell_type[ci] == 16)

    assert z12_n == {6}, f"Z12 belt sizes: {z12_n} (expected {{6}})"
    assert z16_n == {8}, f"Z16 belt sizes: {z16_n} (expected {{8}})"
