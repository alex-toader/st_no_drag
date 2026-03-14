"""
No-drag analysis for foam lattices.

Functions for computing:
  - Selection rule M₀ (net force from m=2 belt traction)
  - Enriched belt basis Q_belt (96D for C15)
  - Acoustic ceiling ω_edge (max ω₃ along BZ)
  - Particle floor ω_belt_min (min particle-band ω from H_eff)
  - Projected group velocities v_g from H_eff bands

All functions are foam-agnostic (work on C15, Kelvin, WP).

DESIGN NOTES:
  - compute_acoustic_ceiling uses omega[2] (3rd eigenvalue) directly.
    No spurious zero modes exist at k≠0 in our geometries, so omega[2]
    is always the 3rd acoustic branch. compute_acousticness_ceiling
    provides an independent cross-check via translation projection.
  - compute_projected_velocities identifies particle bands at Γ and
    tracks them through BZ without re-identification. This is correct
    because H_eff = Q_belt^T D Q_belt projects out acoustic modes —
    particle bands in H_eff cannot cross with acoustic bands.
  - vT is measured from avg(omega[0], omega[1]) at small k. The two
    transverse modes are degenerate to < 0.6% (verified on C15
    [100]/[110]/[111]), so the average differs from omega[0] by < 0.1%.
  - _belt_geometry recomputes get_cell_geometry + find_best_belt on each
    call. find_best_belt is deterministic (smallest eigenvalue, stable
    sort), so repeated calls return identical results.

SELECTION RULE — ANTIPODAL SYMMETRY:
  The selection rule M₀ = 0 for even-m modes on Z12 cells is a consequence
  of the antipodal tilt symmetry: n_ax[i] = -n_ax[i + N/2] in circuit order.
  This means the normal axial tilt has only odd Fourier harmonics, so it
  cannot beat with even-m pressure cos(mθ) to produce a net force.
  - Z12 (N=6): antipodal → even m protected (m=2,4)
  - Z16 (N=8): symmetric (n_ax[i] = +n_ax[i+N/2]) → odd m protected (m=1,3,5,7)
  - Kelvin (N=6): n_ax ≡ 0 (flat belt) → all m protected

Mar 2026
"""

import numpy as np
from ..analysis.cell_topology import get_cell_geometry, find_best_belt


# =========================================================================
# Geometry helpers
# =========================================================================

def geometric_normal(verts, cell_center):
    """Face normal from vertex cross products, oriented outward."""
    cf = verts.mean(axis=0)
    total_cross = np.zeros(3)
    n = len(verts)
    for k in range(n):
        v1 = verts[k] - cf
        v2 = verts[(k + 1) % n] - cf
        total_cross += np.cross(v1, v2)
    nrm = total_cross / np.linalg.norm(total_cross)
    if np.dot(nrm, cf - cell_center) < 0:
        nrm = -nrm
    return nrm


def _belt_geometry(ci, centers, v, f, cfi, L):
    """Compute belt geometry for one cell.

    Returns (circuit, theta, normals, areas, belt_normal, fd) or None.
    """
    cc = centers[ci]
    fd, adj = get_cell_geometry(ci, cc, v, f, cfi, L)
    result = find_best_belt(fd, adj, cc)
    if result is None:
        return None

    circuit, omega = result
    N = len(circuit)

    belt_centers = np.array([fd[circuit[i]]['center'] for i in range(N)])
    rel = belt_centers - cc
    centered = rel - rel.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    bn = Vt[-1]
    proj = rel - np.outer(rel @ bn, bn)
    norms_p = np.linalg.norm(proj, axis=1)
    u_idx = np.argmax(norms_p)
    u = proj[u_idx] / norms_p[u_idx]
    w = np.cross(bn, u)
    w = w / np.linalg.norm(w)
    theta = np.arctan2(proj @ w, proj @ u)

    areas = np.zeros(N)
    normals = np.zeros((N, 3))
    for idx in range(N):
        face = fd[circuit[idx]]
        verts_f = face['vertices']
        cf = verts_f.mean(axis=0)
        tc = np.zeros(3)
        for k in range(len(verts_f)):
            tc += np.cross(verts_f[k] - cf,
                           verts_f[(k + 1) % len(verts_f)] - cf)
        areas[idx] = np.linalg.norm(tc) / 2
        normals[idx] = geometric_normal(verts_f, cc)

    return circuit, theta, normals, areas, bn, fd


# =========================================================================
# Selection rule
# =========================================================================

def compute_selection_rule(ci, centers, v, f, cfi, L):
    """Compute M₀ = Σ F_j from m=2 belt traction on one cell.

    Returns dict with:
      M0_vec: net force vector (3,)
      M0_mag: |M₀|
      n_ax_spectrum: Fourier spectrum of normal axial tilt (N,) complex
      n_belt: number of belt faces
    Returns None if cell has no belt.
    """
    bg = _belt_geometry(ci, centers, v, f, cfi, L)
    if bg is None:
        return None

    circuit, theta, normals, areas, bn, fd = bg
    N = len(circuit)
    pressure = np.cos(2 * theta)

    # Net force
    M0_vec = np.zeros(3)
    for idx in range(N):
        M0_vec += pressure[idx] * areas[idx] * normals[idx]

    # Normal-tilt Fourier spectrum
    n_ax = np.array([np.dot(normals[idx], bn) for idx in range(N)])
    n_ax_spectrum = np.zeros(N, dtype=complex)
    for m in range(N):
        n_ax_spectrum[m] = np.sum(n_ax * np.exp(-1j * m * theta))

    return {
        'M0_vec': M0_vec,
        'M0_mag': np.linalg.norm(M0_vec),
        'n_ax_spectrum': n_ax_spectrum,
        'n_belt': N,
    }


# =========================================================================
# Belt vertex forces
# =========================================================================

def get_belt_vertex_forces(ci, centers, v, f, cfi, L):
    """Compute m=2 belt forces distributed to vertices.

    Returns dict {vertex_index: force_vector (3,)} or None.
    """
    bg = _belt_geometry(ci, centers, v, f, cfi, L)
    if bg is None:
        return None

    circuit, theta, normals, areas, bn, fd = bg
    N = len(circuit)
    pressure = np.cos(2 * theta)

    vertex_forces = {}
    for idx in range(N):
        face = fd[circuit[idx]]
        F_face = pressure[idx] * areas[idx] * normals[idx]
        n_v = len(face['vertices'])
        F_per_vert = F_face / n_v
        for vi in face['vertex_ids']:
            if vi not in vertex_forces:
                vertex_forces[vi] = np.zeros(3)
            vertex_forces[vi] += F_per_vert

    return vertex_forces


# =========================================================================
# Enriched belt basis
# =========================================================================

def build_enriched_belt_vectors(ci, centers, v, f, cfi, L):
    """Build 6 trial vectors per cell: 3 directions x 2 phases of m=2.

    Directions per face j:
      n_hat = outward face normal
      r_hat = cell->face radial, projected into face plane
      t_hat = n_hat x r_hat (tangent)

    Phases: cos(2*theta), sin(2*theta)

    Returns list of normalized (3*nv,) vectors (up to 6).
    Also returns the cos(2*theta) x n_hat vector separately (first in list).
    """
    cc = centers[ci]
    fd, adj = get_cell_geometry(ci, cc, v, f, cfi, L)
    result = find_best_belt(fd, adj, cc)
    if result is None:
        return [], None

    circuit, omega = result
    N = len(circuit)
    nv = len(v)

    # Belt geometry
    belt_centers = np.array([fd[circuit[i]]['center'] for i in range(N)])
    rel = belt_centers - cc
    centered = rel - rel.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    bn = Vt[-1]
    proj = rel - np.outer(rel @ bn, bn)
    norms_p = np.linalg.norm(proj, axis=1)
    u_idx = np.argmax(norms_p)
    u_hat = proj[u_idx] / norms_p[u_idx]
    w_hat = np.cross(bn, u_hat)
    w_hat /= np.linalg.norm(w_hat)
    theta = np.arctan2(proj @ w_hat, proj @ u_hat)

    # Per-face direction triads
    normals_dir = np.zeros((N, 3))
    radials = np.zeros((N, 3))
    tangents = np.zeros((N, 3))

    for idx in range(N):
        face = fd[circuit[idx]]
        verts_f = face['vertices']
        fc = verts_f.mean(axis=0)

        n_hat = geometric_normal(verts_f, cc)
        normals_dir[idx] = n_hat

        r_raw = fc - cc
        r_in = r_raw - np.dot(r_raw, n_hat) * n_hat
        rn = np.linalg.norm(r_in)
        if rn > 1e-15:
            radials[idx] = r_in / rn
        else:
            alt = bn - np.dot(bn, n_hat) * n_hat
            radials[idx] = alt / max(np.linalg.norm(alt), 1e-15)

        t_hat = np.cross(n_hat, radials[idx])
        tn = np.linalg.norm(t_hat)
        tangents[idx] = t_hat / max(tn, 1e-15)

    # Build 6 displacement vectors
    vectors = []
    old_vector = None  # cos(2θ) x n_hat

    for phase_idx, amplitude in enumerate(
            [np.cos(2 * theta), np.sin(2 * theta)]):
        for dir_idx, directions in enumerate(
                [normals_dir, radials, tangents]):
            u_vec = np.zeros(3 * nv)
            for idx in range(N):
                face = fd[circuit[idx]]
                n_v = len(face['vertices'])
                disp = amplitude[idx] * directions[idx] / n_v
                for vi in face['vertex_ids']:
                    u_vec[3 * vi: 3 * vi + 3] += disp

            norm = np.linalg.norm(u_vec)
            if norm > 1e-15:
                u_vec_n = u_vec / norm
                vectors.append(u_vec_n)
                if phase_idx == 0 and dir_idx == 0:
                    old_vector = u_vec_n

    return vectors, old_vector


def build_belt_basis(belt_cells, centers, v, f, cfi, L):
    """Build Q_belt and Q_particle from enriched belt vectors.

    Args:
      belt_cells: list of cell indices with belts (e.g. Z12 cells in C15)

    Returns:
      Q_belt: (3*nv, n_belt) orthonormalized belt basis
      Q_particle: (3*nv, n_particle) cos2θ×n̂ subspace
      M_transfer: (n_particle, n_belt) transfer matrix
    """
    all_raw = []
    old_vectors = {}

    for ci in belt_cells:
        vecs, old_vec = build_enriched_belt_vectors(ci, centers, v, f, cfi, L)
        if old_vec is not None:
            old_vectors[ci] = old_vec
        all_raw.extend(vecs)

    missing = [ci for ci in belt_cells if ci not in old_vectors]
    if missing:
        raise ValueError(f"Cells without belt: {missing}")

    if not all_raw:
        return None, None, None

    V_raw = np.column_stack(all_raw)
    Q, R = np.linalg.qr(V_raw, mode='reduced')
    r_diag = np.abs(np.diag(R))
    keep = r_diag > 1e-10 * r_diag.max()
    Q_belt = Q[:, keep]

    # Particle subspace (cos2θ×n̂ only)
    U_old = np.column_stack([old_vectors[ci] for ci in belt_cells
                             if ci in old_vectors])
    Q_old_qr, R_old_qr = np.linalg.qr(U_old, mode='reduced')
    keep_old = np.abs(np.diag(R_old_qr)) > 1e-10 * np.abs(
        np.diag(R_old_qr)).max()
    Q_particle = Q_old_qr[:, keep_old]

    M_transfer = Q_particle.T @ Q_belt

    return Q_belt, Q_particle, M_transfer


# =========================================================================
# Acoustic ceiling
# =========================================================================

_DEFAULT_BZ_DIRS = {
    '[100]': np.array([1, 0, 0], dtype=float),
    '[110]': np.array([1, 1, 0], dtype=float) / np.sqrt(2),
    '[111]': np.array([1, 1, 1], dtype=float) / np.sqrt(3),
}

_DENSE_BZ_DIRS = {
    '[100]': np.array([1, 0, 0], dtype=float),
    '[010]': np.array([0, 1, 0], dtype=float),
    '[001]': np.array([0, 0, 1], dtype=float),
    '[110]': np.array([1, 1, 0], dtype=float) / np.sqrt(2),
    '[101]': np.array([1, 0, 1], dtype=float) / np.sqrt(2),
    '[011]': np.array([0, 1, 1], dtype=float) / np.sqrt(2),
    '[111]': np.array([1, 1, 1], dtype=float) / np.sqrt(3),
    '[112]': np.array([1, 1, 2], dtype=float) / np.sqrt(6),
    '[122]': np.array([1, 2, 2], dtype=float) / 3.0,
}


def compute_acoustic_ceiling(bloch, L, bz_dirs=None, n_k=80):
    """Compute ω_edge = max ω₃(k) along BZ high-symmetry lines.

    Returns dict with:
      omega_edge: acoustic ceiling
      omega_edge_dir: direction where max occurs
      omega_edge_2x: value at 2x resolution (convergence check)
      per_dir: {dir_name: omega_edge_in_dir}
    """
    if bz_dirs is None:
        bz_dirs = _DENSE_BZ_DIRS

    omega_edge = 0.0
    omega_edge_dir = ""
    per_dir = {}

    for dname, dhat in bz_dirs.items():
        k_max = np.pi / (L * np.max(np.abs(dhat)))
        k_vals = np.linspace(0, k_max, n_k)
        w3_max = 0.0

        for ik in range(1, n_k):
            Dk = bloch.build_dynamical_matrix(k_vals[ik] * dhat)
            evals = np.sort(np.linalg.eigvalsh(Dk))
            omega = np.sqrt(np.maximum(evals, 0))
            if omega[2] > w3_max:
                w3_max = omega[2]

        per_dir[dname] = w3_max
        if w3_max > omega_edge:
            omega_edge = w3_max
            omega_edge_dir = dname

    # Convergence check at 2x resolution
    omega_edge_2x = 0.0
    for dhat in bz_dirs.values():
        k_max = np.pi / (L * np.max(np.abs(dhat)))
        k_vals = np.linspace(0, k_max, 2 * n_k)
        for ik in range(1, len(k_vals)):
            Dk = bloch.build_dynamical_matrix(k_vals[ik] * dhat)
            evals = np.sort(np.linalg.eigvalsh(Dk))
            omega = np.sqrt(np.maximum(evals, 0))
            omega_edge_2x = max(omega_edge_2x, omega[2])

    return {
        'omega_edge': omega_edge,
        'omega_edge_dir': omega_edge_dir,
        'omega_edge_2x': omega_edge_2x,
        'convergence_delta': abs(omega_edge_2x - omega_edge),
        'per_dir': per_dir,
    }


def compute_acousticness_ceiling(bloch, L, n_k=80, bz_dirs=None,
                                  thresholds=(0.10, 0.05)):
    """Compute COM-coherent acoustic ceiling using translation projection.

    For each threshold, finds the max omega at which any mode has
    acousticness (= sum of squared projections onto 3 uniform-displacement
    vectors, i.e. COM-coherent content) above the threshold.

    Uses 9 BZ directions by default for dense coverage.

    Args:
      bloch: DisplacementBloch object
      L: supercell side length
      n_k: k-points per direction
      bz_dirs: dict of direction vectors (default: 9 directions)
      thresholds: tuple of acousticness thresholds to scan

    Returns dict with:
      results: list of {threshold, ceiling, ceiling_dir} per threshold
      random_level: 3/n_dof (baseline for random vector)
    """
    if bz_dirs is None:
        bz_dirs = _DENSE_BZ_DIRS

    nv = bloch.n_vertices if hasattr(bloch, 'n_vertices') else len(bloch.vertices)
    n_dof = 3 * nv

    # Build translation basis (3 orthonormal uniform-displacement vectors)
    Q_trans = np.zeros((n_dof, 3))
    for d in range(3):
        q = np.zeros(n_dof)
        for i in range(nv):
            q[3 * i + d] = 1.0
        Q_trans[:, d] = q / np.linalg.norm(q)

    # Single BZ scan, filter per threshold afterwards
    # Store (omega, acousticness, dir_name) for each mode at each k
    ceilings = {th: (0.0, '') for th in thresholds}

    for dname, dhat in bz_dirs.items():
        k_max = np.pi / (L * np.max(np.abs(dhat)))
        k_vals = np.linspace(0, k_max, n_k)
        for ik in range(1, n_k):
            kvec = k_vals[ik] * dhat
            Dk = bloch.build_dynamical_matrix(kvec)
            evals, evecs = np.linalg.eigh(Dk)
            omega = np.sqrt(np.maximum(evals, 0))
            proj = np.abs(Q_trans.T @ evecs) ** 2
            acousticness = proj.sum(axis=0)

            for th in thresholds:
                mask = acousticness > th
                if mask.any():
                    om = omega[mask].max()
                    if om > ceilings[th][0]:
                        ceilings[th] = (om, dname)

    results = []
    for th in thresholds:
        results.append({
            'threshold': th,
            'ceiling': ceilings[th][0],
            'ceiling_dir': ceilings[th][1],
        })

    return {
        'results': results,
        'random_level': 3.0 / n_dof,
    }


# =========================================================================
# Particle floor
# =========================================================================

def compute_particle_floor(bloch, Q_belt, M_transfer, n_particle,
                           L, bz_dirs=None, n_k=80):
    """Compute ω_belt_min_global from H_eff(k) scan.

    Particle bands re-identified at each k by maximal cos2θ×n̂ character,
    using tolerance-based selection (includes all modes with pc >= cutoff - tol
    to handle near-degeneracies stably).

    Returns dict with:
      omega_belt_min_global: global particle floor
      omega_belt_min_gamma: particle floor at Γ
      floor_pc: particle character of the mode realizing the global floor
      floor_n_selected: number of modes in the tolerance-based particle set
    """
    if bz_dirs is None:
        bz_dirs = _DENSE_BZ_DIRS

    omega_min_global = np.inf
    omega_min_gamma = np.inf
    floor_pc = 0.0
    floor_n_selected = 0

    for dhat in bz_dirs.values():
        k_max = np.pi / (L * np.max(np.abs(dhat)))
        k_vals = np.linspace(0, k_max, n_k)

        for ik, kk in enumerate(k_vals):
            Dk = bloch.build_dynamical_matrix(kk * dhat)
            H_belt = Q_belt.T @ Dk @ Q_belt
            evals, evecs = np.linalg.eigh(H_belt)
            omega = np.sqrt(np.maximum(evals, 0))

            Mc = M_transfer @ evecs
            pc = np.sum(np.abs(Mc)**2, axis=0)

            # Tolerance-based selection
            pc_sorted = np.sort(pc)[::-1]
            pc_cut = pc_sorted[min(n_particle - 1, len(pc_sorted) - 1)]
            p_mask = pc >= pc_cut - 1e-6

            w_min = omega[p_mask].min()
            if w_min < omega_min_global:
                omega_min_global = w_min
                floor_idx = np.where(p_mask)[0][np.argmin(omega[p_mask])]
                floor_pc = pc[floor_idx]
                floor_n_selected = int(p_mask.sum())

            if ik == 0:
                omega_min_gamma = min(omega_min_gamma, w_min)

    return {
        'omega_belt_min_global': omega_min_global,
        'omega_belt_min_gamma': omega_min_gamma,
        'floor_pc': floor_pc,
        'floor_n_selected': floor_n_selected,
    }


# =========================================================================
# Projected group velocities
# =========================================================================

def max_group_velocity(k_vals, omega_bands):
    """Maximum |dω/dk| across all bands via central differences."""
    n_k, n_orb = omega_bands.shape
    max_vg = 0.0
    max_band = 0
    for band in range(n_orb):
        for ik in range(1, n_k - 1):
            dk = k_vals[ik + 1] - k_vals[ik - 1]
            domega = omega_bands[ik + 1, band] - omega_bands[ik - 1, band]
            vg = abs(domega / dk)
            if vg > max_vg:
                max_vg = vg
                max_band = band
    return max_vg, max_band


def centroid_velocity(k_vals, omega_bands, score_bands):
    """Weighted average |dω/dk| across bands, weighted by score at each k."""
    n_k, n_orb = omega_bands.shape
    total_w = 0.0
    weighted_vg = 0.0
    for band in range(n_orb):
        for ik in range(1, n_k - 1):
            dk = k_vals[ik + 1] - k_vals[ik - 1]
            domega = omega_bands[ik + 1, band] - omega_bands[ik - 1, band]
            vg = abs(domega / dk)
            s = score_bands[ik, band]
            weighted_vg += s * vg
            total_w += s
    return weighted_vg / max(total_w, 1e-30)


def compute_projected_velocities(bloch, Q_belt, M_transfer, n_particle,
                                 L, bz_dirs=None, n_k=60):
    """Compute v_g_max and v_g_centroid from projected H_eff bands.

    Returns dict with:
      vg_max: max group velocity of particle bands
      vg_centroid: character-weighted centroid group velocity
      vT_avg: average transverse sound speed
      vg_max_over_vT: ratio
      vg_centroid_over_vT: ratio
    """
    if bz_dirs is None:
        bz_dirs = _DENSE_BZ_DIRS

    n_belt = Q_belt.shape[1]

    # Measure v_T
    all_vT = []
    for dhat in bz_dirs.values():
        k_small = 0.05 * dhat
        Dk = bloch.build_dynamical_matrix(k_small)
        evals = np.sort(np.linalg.eigvalsh(Dk))
        omega = np.sqrt(np.maximum(evals, 0))
        # Two transverse modes (lowest two): average for direction-dependent splitting
        vT = 0.5 * (omega[0] + omega[1]) / np.linalg.norm(k_small)
        all_vT.append(vT)
    vT_avg = np.mean(all_vT)

    vg_max_all = 0.0
    cvg_max_all = 0.0

    for dhat in bz_dirs.values():
        k_max = np.pi / (L * np.max(np.abs(dhat)))
        k_vals = np.linspace(0, k_max, n_k)

        omega_proj = np.zeros((n_k, n_belt))
        pc_proj = np.zeros((n_k, n_belt))

        for ik, kk in enumerate(k_vals):
            Dk = bloch.build_dynamical_matrix(kk * dhat)
            H_eff = Q_belt.T @ Dk @ Q_belt
            evals, evecs = np.linalg.eigh(H_eff)
            omega_proj[ik] = np.sqrt(np.maximum(evals, 0))
            Mc = M_transfer @ evecs
            pc_proj[ik] = np.sum(np.abs(Mc)**2, axis=0)

        # Identify particle bands at Γ
        pc_gamma = pc_proj[0]
        particle_bands = np.sort(np.argsort(-pc_gamma)[:n_particle])

        omega_particle = omega_proj[:, particle_bands]
        pc_particle = pc_proj[:, particle_bands]

        vg_part, _ = max_group_velocity(k_vals, omega_particle)
        cvg_part = centroid_velocity(k_vals, omega_particle, pc_particle)

        vg_max_all = max(vg_max_all, vg_part)
        cvg_max_all = max(cvg_max_all, cvg_part)

    return {
        'vg_max': vg_max_all,
        'vg_centroid': cvg_max_all,
        'vT_avg': vT_avg,
        'vg_max_over_vT': vg_max_all / vT_avg,
        'vg_centroid_over_vT': cvg_max_all / vT_avg,
    }
