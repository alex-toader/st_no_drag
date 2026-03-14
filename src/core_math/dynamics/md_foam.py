"""
Molecular dynamics for foam lattices (harmonic + cubic anharmonicity).

Convention: all force functions return ACCELERATION (force/mass).
All energy functions return ENERGY PER UNIT MASS.
This is consistent with D = K/m (dynamical matrix = stiffness/mass).

Provides:
  - harmonic_force_spring: a from tensorial springs on edges (O(n_edges))
  - cubic_force: a from V₃ = (α/6) Σ_edges (δr_e)³
  - verlet_step: one velocity Verlet step
  - modal_energy: energy decomposition onto eigenmodes
  - prepare_edges: precompute edge geometry for force evaluation

NOTE: edge_dirs are FIXED at equilibrium geometry. This is a Taylor expansion
model (harmonic + cubic), not a geometrically nonlinear MD.

NOTE: V₃ ∝ (δr)³ is odd → unbounded below for one sign of δr.
Safe in perturbative regime (small amplitudes). Monitor energy conservation
and max(|δr|) to detect runaway.

DESIGN NOTES:
  - cubic_force sign: F_i = +(α/2)(δr)²ê is correct (-dV₃/du_i with
    δr = u_j - u_i). Consistency verified by test_energy_conservation
    (|dE/E| = 6.91e-07 over ~40k Verlet steps).
  - harmonic_force_spring ≡ -D·u verified indirectly: test_harmonic_baseline
    shows E_low = 4.26e-32 (machine zero), which requires exact consistency
    between spring forces and the eigendecomposition from D(k=0).
  - remove_com modifies arrays in-place via reshape views. Callers use
    the returned ravel. This is intentional, not a bug.
  - sector_energy omega_cut is set by the caller (test), not by this module.
    The value 0.8022 comes from compute_acoustic_ceiling and is validated
    by test_harmonic_baseline (zero leakage confirms correct separation).

Feb 2026
"""

import numpy as np


# =========================================================================
# Edge geometry (precomputed, reusable)
# =========================================================================

def prepare_edges(vertices, edges, L):
    """Precompute edge geometry for force evaluation.

    Returns dict with:
      idx_i, idx_j: (n_edges,) source/target vertex indices
      edge_dirs: (n_edges, 3) unit vectors along edges (minimum image)
      edge_lengths: (n_edges,) rest lengths
    """
    v_arr = np.array(vertices)
    idx_i = np.array([edge[0] for edge in edges])
    idx_j = np.array([edge[1] for edge in edges])
    delta = v_arr[idx_j] - v_arr[idx_i]
    delta = delta - L * np.round(delta / L)
    lengths = np.linalg.norm(delta, axis=1)
    assert lengths.min() > 1e-10, (
        f"Degenerate edge: min length = {lengths.min():.2e}")
    dirs = delta / lengths[:, np.newaxis]
    return {
        'idx_i': idx_i,
        'idx_j': idx_j,
        'edge_dirs': dirs,
        'edge_lengths': lengths,
        'n_edges': len(edges),
    }


# =========================================================================
# Forces
# =========================================================================

def harmonic_force_spring(u, edge_info, k_L, k_T, mass=1.0):
    """Harmonic force from tensorial springs on edges (O(n_edges)).

    Spring stiffness: K_ab = k_T δ_ab + (k_L - k_T) ê_a ê_b
    Force on vertex i from edge (i,j): F_i = (K/m) · (u_j - u_i)

    Equivalent to F = -D·u but O(n_edges) instead of O(n²).
    Use this for large supercells where D is expensive.

    Args:
      u: (n_dof,) displacement vector
      edge_info: from prepare_edges()
      k_L: longitudinal spring constant
      k_T: transverse spring constant
      mass: vertex mass

    Returns: (n_dof,) acceleration
    """
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']  # (n_edges, 3) unit vectors

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)

    # du = u_j - u_i for each edge
    du = u3[idx_j] - u3[idx_i]  # (n_edges, 3)

    # Tensorial spring: K · du = k_T * du + (k_L - k_T) * (du · ê) ê
    du_par = np.sum(du * dirs, axis=1, keepdims=True) * dirs  # longitudinal
    f_edge = k_T * du + (k_L - k_T) * du_par  # (n_edges, 3)
    f_edge /= mass

    # Distribute: F_i += f_edge, F_j -= f_edge
    F = np.zeros_like(u)
    F3 = F.reshape(nv, 3)
    np.add.at(F3, idx_i, f_edge)
    np.add.at(F3, idx_j, -f_edge)

    return F


def cubic_force(u, edge_info, alpha, mass=1.0):
    """Cubic anharmonic acceleration from V₃ = (α/6) Σ_edges (δr_e)³.

    δr_e = (u_j - u_i) · ê_e  (edge extension, scalar)

    Force on vertex i from edge (i,j): F_i = +(α/2)(δr_e)² ê_e
    Force on vertex j from edge (i,j): F_j = -(α/2)(δr_e)² ê_e

    Returns: (n_dof,) acceleration (force / mass).
    """
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)

    # Edge extensions: δr_e = (u_j - u_i) · ê_e
    du = u3[idx_j] - u3[idx_i]  # (n_edges, 3)
    dr = np.sum(du * dirs, axis=1)  # (n_edges,)

    # Acceleration magnitude per edge: (α/2)(δr)² / mass
    a_mag = 0.5 * alpha * dr**2 / mass  # (n_edges,)

    # Distribute to vertices
    F = np.zeros_like(u)
    F3 = F.reshape(nv, 3)

    # a_i += +(α/2m)(δr)² ê, a_j += -(α/2m)(δr)² ê
    a_vec = a_mag[:, np.newaxis] * dirs  # (n_edges, 3)
    np.add.at(F3, idx_i, a_vec)
    np.add.at(F3, idx_j, -a_vec)

    return F


def cubic_energy(u, edge_info, alpha, mass=1.0):
    """Cubic potential energy per unit mass: V₃/m = (α/6m) Σ_edges (δr_e)³."""
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']
    nv = len(u) // 3
    u3 = u.reshape(nv, 3)
    du = u3[idx_j] - u3[idx_i]
    dr = np.sum(du * dirs, axis=1)
    return (alpha / (6.0 * mass)) * np.sum(dr**3)


# =========================================================================
# Integrator
# =========================================================================

def verlet_step(u, v, a, force_fn, dt):
    """One velocity Verlet step.

    Args:
      u: (n_dof,) positions
      v: (n_dof,) velocities
      a: (n_dof,) accelerations (from previous step)
      force_fn: callable(u) -> (n_dof,) acceleration
      dt: time step

    Returns: (u_new, v_new, a_new)
    """
    u_new = u + dt * v + 0.5 * dt**2 * a
    a_new = force_fn(u_new)
    v_new = v + 0.5 * dt * (a + a_new)
    return u_new, v_new, a_new


# =========================================================================
# Diagnostics
# =========================================================================

def modal_energy(u, v, evecs, omega):
    """Energy per eigenmode.

    E_n = (1/2) [ω_n² c_n² + ċ_n²]

    where c_n = e_n^T u, ċ_n = e_n^T v.

    Args:
      u: (n_dof,) displacement
      v: (n_dof,) velocity
      evecs: (n_dof, n_modes) eigenvectors (columns)
      omega: (n_modes,) eigenfrequencies

    Returns: (n_modes,) energy per mode
    """
    c = evecs.T @ u   # (n_modes,)
    cdot = evecs.T @ v  # (n_modes,)
    return 0.5 * (omega**2 * c**2 + cdot**2)


def sector_energy(u, v, evecs, omega, omega_cut):
    """Total energy in modes with ω < omega_cut.

    Returns scalar.
    """
    E = modal_energy(u, v, evecs, omega)
    mask = omega < omega_cut
    return np.sum(E[mask])


def harmonic_energy_spring(u, v, edge_info, k_L, k_T, mass=1.0):
    """Total energy per unit mass from springs (O(n_edges)).

    V/m = (1/2m) Σ_edges du^T K du = (1/2m) Σ [k_T |du|² + (k_L-k_T)(du·ê)²]
    T/m = (1/2) Σ_i |v_i|²

    Consistent with total_energy_harmonic(u, v, D) where D = K/m.
    """
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)

    du = u3[idx_j] - u3[idx_i]
    du_sq = np.sum(du**2, axis=1)
    du_par = np.sum(du * dirs, axis=1)

    V_over_m = 0.5 * np.sum(k_T * du_sq + (k_L - k_T) * du_par**2) / mass
    T_over_m = 0.5 * np.dot(v, v)
    return V_over_m + T_over_m


def remove_com(u, v, nv):
    """Project out center-of-mass translation from u and v.

    Removes the uniform-translation component (3 zero-modes).
    """
    u3 = u.reshape(nv, 3)
    v3 = v.reshape(nv, 3)
    u3 -= u3.mean(axis=0)
    v3 -= v3.mean(axis=0)
    return u3.ravel(), v3.ravel()
