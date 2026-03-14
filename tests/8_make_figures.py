#!/usr/bin/env python3
"""
Generate publication figures for the paper.

Figures generated:
  - fig2_dispersion.pdf   (Fig 2: C15 band structure along [100])
  - fig3_k_squared.pdf    (Fig 3: k² dipolar scaling, log-log)
  - fig4_gap_ratio.pdf    (Fig 4: gap ratio vs k_L/k_T, 3 structures)

Usage:
    cd /path/to/st_no_drag
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 tests/8_make_figures.py

Mar 2026
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)
from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.builders.wp_periodic import build_wp_supercell_periodic, get_a15_points
from core_math.analysis.no_drag import (
    build_belt_basis, get_belt_vertex_forces,
    compute_acoustic_ceiling,
)
from physics.bloch import DisplacementBloch

PAPER_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper')


# =========================================================================
# Fig 2: Band structure (C15, [100] direction)
# =========================================================================

def make_fig2():
    """C15 band structure along [100] showing acoustic bands, gap, belt band."""
    print("\n--- Figure 2: C15 band structure [100] ---")

    N, L_cell = 1, 4.0
    L = N * L_cell
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = np.array(get_c15_points(N, L_cell))

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0, mass=1.0)

    z12 = [ci for ci in range(len(cfi)) if len(cfi[ci]) == 12]
    Q_belt = build_belt_basis(z12, centers, v, f, cfi, L)[0]

    # Scan along [100]
    k_hat = np.array([1, 0, 0], dtype=float)
    k_max = np.pi / L
    n_k = 80
    k_vals = np.linspace(0, k_max, n_k)

    n_dof = 3 * len(v)
    n_show = 80  # must cover all bands below ylim=1.5 (need ≥76)
    omega_full = np.zeros((n_k, n_show))

    # Belt character for coloring
    belt_char = np.zeros((n_k, n_show))
    # Random baseline: if belt character > this, mode has significant belt content
    bc_random = Q_belt.shape[1] / n_dof  # ~0.235 for C15

    for ik in range(n_k):
        kvec = k_vals[ik] * k_hat
        Dk = bloch.build_dynamical_matrix(kvec)
        evals, evecs = np.linalg.eigh(Dk)
        omega = np.sqrt(np.maximum(evals, 0))
        omega_full[ik] = omega[:n_show]

        for n in range(n_show):
            belt_char[ik, n] = np.sum(np.abs(Q_belt.T @ evecs[:, n])**2)

    # Compute omega_edge and omega_belt_min for reference lines.
    # belt_min = lowest eigenvalue of H_belt across BZ (same definition as fig4).
    ac = compute_acoustic_ceiling(bloch, L, n_k=80)
    omega_edge = ac['omega_edge']
    from core_math.analysis.no_drag import _DEFAULT_BZ_DIRS
    omega_belt_min = np.inf
    for d_vec in _DEFAULT_BZ_DIRS.values():
        km = np.pi / (L * np.max(np.abs(d_vec)))
        for ik in range(81):
            kk = (ik / 80) * km
            Dk_scan = bloch.build_dynamical_matrix(kk * d_vec)
            H_belt_scan = Q_belt.T @ Dk_scan @ Q_belt
            ev = np.linalg.eigvalsh(H_belt_scan)
            w = np.sqrt(max(ev[0], 0))
            if w < omega_belt_min:
                omega_belt_min = w

    # Normalize k to BZ edge
    k_norm = k_vals / k_max

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    # Three visual categories:
    #   acoustic (bc < 0.5*bc_random): gray
    #   mixed optical (0.5*bc_random <= bc < 2*bc_random): light coral
    #   belt-like (bc >= 2*bc_random): red
    for n in range(n_show):
        bc_avg = belt_char[:, n].mean()
        if bc_avg < 0.5 * bc_random:
            color, lw, alpha = '#4a4a4a', 0.7, 0.4
        elif bc_avg < 2 * bc_random:
            color, lw, alpha = '#d4816b', 0.8, 0.5
        else:
            color, lw, alpha = '#c0392b', 1.2, 0.9
        ax.plot(k_norm, omega_full[:, n], color=color, linewidth=lw, alpha=alpha)

    # Highlight 3 acoustic branches
    for n in range(3):
        ax.plot(k_norm, omega_full[:, n], color='#2c3e50', linewidth=1.4, alpha=0.8)

    # Reference lines
    ax.axhline(omega_edge, color='#7f8c8d', linestyle='--', linewidth=1.0,
               label=fr'$\omega_{{edge}}$ = {omega_edge:.3f}')
    ax.axhline(omega_belt_min, color='#c0392b', linestyle='--', linewidth=1.0,
               label=fr'$\omega_{{belt,min}}$ = {omega_belt_min:.3f}')

    # Gap annotation
    gap_mid = 0.5 * (omega_edge + omega_belt_min)
    ax.annotate('', xy=(0.93, omega_belt_min), xytext=(0.93, omega_edge),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(0.96, gap_mid, f'gap\n{omega_belt_min/omega_edge:.2f}x',
            ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel(r'$k / k_{BZ}$ along [100]')
    ax.set_ylabel(r'$\omega$ (natural units)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    path = os.path.join(PAPER_DIR, 'fig2_dispersion.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  {n_k} k-points, {n_show} bands")
    print(f"  ω_edge = {omega_edge:.4f}, ω_belt_min = {omega_belt_min:.4f}")
    print(f"  Saved {path}")


# =========================================================================
# Fig 3: k² scaling (dipolar, log-log)
# =========================================================================

def make_fig3():
    """Log-log plot of acoustic overlap vs k for hop source (slope = 2)."""
    print("\n--- Figure 3: k² dipolar scaling ---")

    L_cell = 4.0
    v, e, f, cfi = build_c15_supercell_periodic(1, L_cell)
    centers = np.array(get_c15_points(1, L_cell))
    L = L_cell
    n_cells = len(cfi)
    cell_type = np.array([12 if len(cfi[ci]) == 12 else 16
                          for ci in range(n_cells)])

    # Find first Z12-Z12 pair
    face_to_cells = {}
    for ci in range(n_cells):
        for f_idx, _ in cfi[ci]:
            face_to_cells.setdefault(f_idx, []).append(ci)

    z12_pair = None
    for f_idx, cells in face_to_cells.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if cell_type[a] == 12 and cell_type[b] == 12:
            z12_pair = (a, b)
            break

    # Also find Z12-Z16 pair (monopole control)
    mixed_pair = None
    for f_idx, cells in face_to_cells.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if {cell_type[a], cell_type[b]} == {12, 16}:
            mixed_pair = (a, b)
            break

    bloch = DisplacementBloch(v, e, L, k_L=2.0, k_T=1.0)

    k_BZ = np.pi / L
    k_fracs = np.array([0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05])

    k_dirs = {
        '[100]': np.array([1, 0, 0], dtype=float),
        '[110]': np.array([1, 1, 0], dtype=float) / np.sqrt(2),
        '[111]': np.array([1, 1, 1], dtype=float) / np.sqrt(3),
    }

    def _acoustic_overlap(S_vec, k_vec, norm_sq):
        D = bloch.build_dynamical_matrix(k_vec)
        w2, modes = np.linalg.eigh(D)
        order = np.argsort(w2)
        modes = modes[:, order]
        total = 0.0
        for branch in range(3):
            total += np.abs(np.dot(S_vec, np.conj(modes[:, branch])))**2
        return total / norm_sq if norm_sq > 0 else 0.0

    # Build sources
    vf_a = get_belt_vertex_forces(z12_pair[0], centers, v, f, cfi, L)
    vf_b = get_belt_vertex_forces(z12_pair[1], centers, v, f, cfi, L)
    S_dip = np.zeros(3 * len(v))
    for vi in set(vf_a.keys()) | set(vf_b.keys()):
        S_dip[3 * vi:3 * vi + 3] = vf_b.get(vi, np.zeros(3)) - vf_a.get(vi, np.zeros(3))
    norm_dip = np.dot(S_dip, S_dip)

    vf_ma = get_belt_vertex_forces(mixed_pair[0], centers, v, f, cfi, L)
    vf_mb = get_belt_vertex_forces(mixed_pair[1], centers, v, f, cfi, L)
    S_mono = np.zeros(3 * len(v))
    for vi in set(vf_ma.keys()) | set(vf_mb.keys()):
        S_mono[3 * vi:3 * vi + 3] = vf_mb.get(vi, np.zeros(3)) - vf_ma.get(vi, np.zeros(3))
    norm_mono = np.dot(S_mono, S_mono)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    colors = {'[100]': '#d62728', '[110]': '#2ca02c', '[111]': '#1f77b4'}
    markers = {'[100]': 'o', '[110]': 's', '[111]': '^'}

    # Save first direction's data for reference lines
    ref_dip_0 = None
    ref_mono_0 = None

    for dir_name, k_hat in k_dirs.items():
        overlaps_dip = []
        overlaps_mono = []
        for kf in k_fracs:
            k_vec = kf * k_BZ * k_hat
            overlaps_dip.append(_acoustic_overlap(S_dip, k_vec, norm_dip))
            overlaps_mono.append(_acoustic_overlap(S_mono, k_vec, norm_mono))

        if ref_dip_0 is None:
            ref_dip_0 = overlaps_dip[0]
            ref_mono_0 = overlaps_mono[0]

        ax.loglog(k_fracs, overlaps_dip,
                  marker=markers[dir_name], color=colors[dir_name],
                  markersize=5, linewidth=1.2,
                  label=f'Z12-Z12 {dir_name}')
        ax.loglog(k_fracs, overlaps_mono,
                  marker=markers[dir_name], color=colors[dir_name],
                  markersize=4, linewidth=0.8, linestyle=':',
                  alpha=0.5, fillstyle='none',
                  label=f'Z12-Z16 {dir_name}')

        # Fit slope for dipolar
        log_k = np.log(k_fracs)
        log_ov = np.log([max(d, 1e-30) for d in overlaps_dip])
        slope = np.polyfit(log_k, log_ov, 1)[0]
        print(f"  {dir_name}: dipolar slope = {slope:.3f}")

    # Reference lines (anchored to first direction)
    k_ref = np.array([k_fracs[0], k_fracs[-1]])
    ref_val = ref_dip_0 * (k_ref / k_fracs[0])**2
    ax.loglog(k_ref, ref_val, 'k--', linewidth=0.8, alpha=0.5, label=r'$\propto k^2$')
    ref_val0 = ref_mono_0 * np.ones_like(k_ref)
    ax.loglog(k_ref, ref_val0, 'k:', linewidth=0.8, alpha=0.5, label=r'$\propto k^0$')

    ax.set_xlabel(r'$k / k_{BZ}$')
    ax.set_ylabel(r'$|\langle S | \psi_{ac} \rangle|^2 / \|S\|^2$')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    # Title goes in LaTeX caption, not in figure

    fig.tight_layout()
    path = os.path.join(PAPER_DIR, 'fig3_k_squared.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Fig 4: Gap ratio vs k_L/k_T (3 structures)
# =========================================================================

def make_fig4():
    """Gap ratio vs spring ratio k_L/k_T for C15, Kelvin, WP."""
    print("\n--- Figure 4: Gap ratio vs k_L/k_T ---")

    kL_kT_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]

    structures = {}

    # C15
    print("  Building C15...")
    N, L_cell = 1, 4.0
    L = N * L_cell
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = np.array(get_c15_points(N, L_cell))
    z12 = [ci for ci in range(len(cfi)) if len(cfi[ci]) == 12]
    structures['C15'] = (v, e, f, cfi, centers, z12, L)

    # Kelvin
    print("  Building Kelvin...")
    N_k = 2
    vk, ek, fk, cfik = build_bcc_supercell_periodic(N_k)
    ck = np.array(generate_bcc_centers(N_k))
    Lk = 4.0 * N_k
    all_k = list(range(len(ck)))
    structures['Kelvin'] = (vk, ek, fk, cfik, ck, all_k, Lk)

    # WP
    print("  Building WP...")
    vw, ew, fw, cfiw = build_wp_supercell_periodic(1, 4.0)
    cw = np.array(get_a15_points(1, 4.0))
    Lw = 4.0
    type_a = [ci for ci in range(len(cfiw)) if len(cfiw[ci]) == 12]
    structures['WP'] = (vw, ew, fw, cfiw, cw, type_a, Lw)

    colors = {'C15': '#d62728', 'Kelvin': '#2ca02c', 'WP': '#1f77b4'}
    markers = {'C15': 'o', 'Kelvin': 's', 'WP': '^'}

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    from core_math.analysis.no_drag import _DEFAULT_BZ_DIRS

    for name, (vs, es, fs, cfis, cs, belt_cells, Ls) in structures.items():
        # Belt basis depends on geometry only, not spring constants
        Q_belt = build_belt_basis(belt_cells, cs, vs, fs, cfis, Ls)[0]

        gaps = []
        for ratio in kL_kT_values:
            bloch = DisplacementBloch(vs, es, Ls, k_L=ratio, k_T=1.0, mass=1.0)

            ac = compute_acoustic_ceiling(bloch, Ls, bz_dirs=_DEFAULT_BZ_DIRS, n_k=40)
            # Belt floor = lowest eigenvalue of H_belt = Q_belt^T D Q_belt
            # across BZ.  This is the strongest gap claim: ALL belt modes
            # (particle + deformation) sit above omega_edge.
            # Note: compute_particle_floor selects only modes with high
            # particle character, which gives a higher (weaker) bound but
            # is numerically unstable on WP (2 particle modes out of 12
            # belt DOF, similar pc values cause selection jumps).
            belt_min = np.inf
            for d_vec in _DEFAULT_BZ_DIRS.values():
                k_max = np.pi / (Ls * np.max(np.abs(d_vec)))
                for ik in range(41):
                    kk = (ik / 40) * k_max
                    Dk = bloch.build_dynamical_matrix(kk * d_vec)
                    H_belt = Q_belt.T @ Dk @ Q_belt
                    evals = np.linalg.eigvalsh(H_belt)
                    w = np.sqrt(max(evals[0], 0))
                    if w < belt_min:
                        belt_min = w
            gap = belt_min / ac['omega_edge']
            gaps.append(gap)
            print(f"  {name} k_L/k_T={ratio:.2f}: gap = {gap:.3f}")

        ax.plot(kL_kT_values, gaps,
                marker=markers[name], color=colors[name],
                markersize=6, linewidth=1.5, label=name)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$k_L / k_T$')
    ax.set_ylabel(r'Gap ratio $\omega_{belt,min} / \omega_{edge}$')
    ax.legend(fontsize=9)
    ax.set_xlim(0.9, 3.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    path = os.path.join(PAPER_DIR, 'fig4_gap_ratio.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FIGURE GENERATION")
    print("Paper figures 2, 3, 4")
    print("=" * 60)

    os.makedirs(PAPER_DIR, exist_ok=True)
    make_fig2()
    make_fig3()
    make_fig4()

    print("\n" + "=" * 60)
    print("Done. All figures saved to paper/")
    print("=" * 60)
