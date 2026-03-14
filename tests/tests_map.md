# Tests Map

## Test Files (82 tests, ~10 min total)

### Infrastructure (inline in src, runs at build time)

Structural invariants are verified inside the builders, not in separate test files.
A reviewer reading the source sees the checks directly.

- **structures.py** -- canonical_face self-test at import (rotations, orientation, numpy.int64, degenerate rejection)
- **kelvin.py** (strict=True) -- V=24, E=36, F=14, χ=2, deg=3, Σface=2E, edge∈2faces, d₁d₀=0
- **multicell_periodic.py** -- V=12N³, χ₂=C, χ₃=0, deg=4, 3faces/edge, 2cells/face, d₁d₀=0

### 1_test_selection_rule.py (15 tests)

Selection rule M₀ = 0 for belt m=2 mode. Antipodal symmetry of normal tilt.

- **test_c15_z12_selection_rule** -- M₀ = 0 on all 16 Z12 cells (exact by symmetry)
- **test_c15_z16_nonzero** -- M₀ != 0 on Z16 cells (negative control)
- **test_c15_z12_no_m2_normal_tilt** -- Z12 n_ax[m=2] = 0 (WHY M₀ = 0); Z16 n_ax[m=2] ~ 1.36 (mechanism)
- **test_kelvin_selection_rule** -- M₀ = 0 on all 16 Kelvin cells
- **test_c15_belt_counts** -- All 24 C15 cells have belts; Z12 N=6, Z16 N=8
- **test_c15_z16_m0_scales_as_area** -- M₀(L=2)/M₀(L=1) = 4.00 (force ~ L²)
- **test_c15_z12_all_m_selection_rule** -- Even m (2,4) protected, odd m (1,3,5) not
- **test_c15_z12_antipodal_tilt** -- n_ax[i] = -n_ax[i+N/2] on Z12 (WHY even m protected)
- **test_c15_z16_symmetric_tilt** -- n_ax[i] = +n_ax[i+N/2] on Z16 (WHY even m NOT protected)
- **test_kelvin_flat_belt** -- n_ax ≡ 0 on Kelvin (flat belt; even m protected by Oh inversion)
- **test_kelvin_m_scan** -- Kelvin even m (2,4) protected (Oh inversion); odd m (1,3,5) NOT protected (radial forces survive)
- **test_c15_z16_m_scan** -- Z16 m=4 (=N/2) protected by S4 symmetry (not inversion, not Nyquist); m=2 not protected
- **test_c15_m0_linear_in_jitter** -- M₀ ∝ δ with coefficient ~1.78 (symmetry, not topology)
- **test_c15_z16_m4_jitter** -- Z16 m=4: M₀ ∝ δ with coefficient ~3.76 (S4 symmetry, not Nyquist aliasing)
- **test_chain_complex_does_not_explain_m0** -- d₁d₀=0 universal (Z12 AND Z16); ||D₀u₀|| ≈ 1.06 vs 1.02 (not floppy); M₀ differs by 15 orders (negative control)

### 2_test_kinematic_gap.py (18 tests)

Kinematic gap: belt frequencies above acoustic ceiling. Subsonic particle velocity.
Gap is structural (survives at isotropy k_L=k_T) and isotropic (>1 in all 9 BZ dirs).
Holds under both omega[2] and 10% acousticness definitions of acoustic ceiling.

- **test_c15_gap_exists** -- C15 gap ratio omega_belt_min / omega_edge > 1.2
- **test_c15_belt_above_acoustic** -- omega_edge ~ 0.69, omega_belt_min ~ 1.18 (expected ranges)
- **test_kelvin_gap_exists** -- Kelvin gap ratio > 1
- **test_c15_belt_basis_dimensions** -- C15 belt basis: 6×n_belt_cells D enriched, n_belt_cells D particle
- **test_c15_subsonic** -- C15 v_g_max / v_T < 1 (measured 0.711)
- **test_kelvin_subsonic** -- Kelvin v_g_max / v_T < 1 (measured 0.541)
- **test_c15_gap_survives_isotropy** -- Gap ratio 1.503 at k_L=k_T=1 (structural, not parameter artifact)
- **test_c15_gap_all_bz_directions** -- Gap ratio > 1 in all 9 BZ directions (worst-case [111]: 1.736)
- **test_kelvin_belt_basis_dimensions** -- Kelvin belt basis: same formula as C15 (96D/16D for N=2)
- **test_c15_centroid_more_subsonic** -- v_g_centroid/v_T = 0.334 << v_g_max/v_T = 0.711
- **test_kelvin_centroid_subsonic** -- Kelvin v_g_centroid/v_T = 0.150 (subsonic)
- **test_c15_gap_convergence** -- ω_edge converged: Δ=0 between n_k=40 and n_k=80
- **test_gap_vs_kL_kT_ratio** -- Gap > 1 for all k_L/k_T ∈ [1,3] (min 1.503 at isotropy)
- **test_acousticness_ceiling_consistent** -- ω[2]=0.694 < ac_10%=0.935 < belt=1.179; strict gap 1.260
- **test_c15_gap_survives_jitter** -- Gap 1.693 under δ=0.01 vertex jitter (unperturbed: 1.699, drop 0.4%)
- **test_belt_floor_above_acoustic** -- min eigenvalue of H_belt > ω_edge on C15 (1.478), Kelvin (1.363), WP (1.327)
- **test_hop_source_zero_weight_below_edge** -- Hop source spectral weight below ω_edge = 0 on all 16 Z12 cells
- **test_floppy_mode_count** -- E=2V, M=3V-E=V; Kelvin 94 floppy, WP 44 floppy at k_T=0 (Maxwell count verified)

### 3_test_wavepacket.py (10 tests)

Wavepacket: zero acoustic emission from belt excitation (Kelvin, Γ-point).
Both displacement and velocity have zero acoustic content. Source-cell independent.
Belt source fully covered by Q_belt. Finite-k protection is via kinematic gap (test 2).
Belt character of acoustic modes (~7%) is distinct from source projection (machine zero).
At finite k, source projection grows as k² (dipolar scaling, connects to test 6).
Z16 loophole bound: particle-band COM content < 10% across BZ (C15).

- **test_zero_translation_content** -- ||Q_trans^T u(t)||^2 < 1e-25 over 4 periods
- **test_zero_acoustic_energy** -- Energy in acoustic modes (acousticness > 10%) is machine zero
- **test_zero_mode_energy** -- 3 zero-frequency modes (translations), energy < 1e-40
- **test_drift_smaller_than_spreading** -- Drift << spreading at Gamma (no directed transport)
- **test_zero_velocity_acoustic_content** -- ||Q_trans^T v(t)||^2 < 1e-25 (momentum-space check)
- **test_source_cell_independence** -- E_acoustic machine zero for all 16 source cells
- **test_belt_basis_covers_source** -- ||u0 - Q_belt Q_belt^T u0|| = 6.64e-16 (u0 ∈ belt subspace)
- **test_belt_char_vs_source_projection** -- Acoustic modes have belt_char ~7% but all 16 source projections = machine zero
- **test_source_projection_scales_k_squared** -- At finite k, source→acoustic projection / k² = 0.027 (constant, dipolar scaling)
- **test_c15_particle_com_content_finite_k** -- Z16 loophole: particle-band COM content < 8.1% across 7 BZ dirs (C15, Z12-only belt basis)

### 4_test_fgr_cubic.py (9 tests)

Fermi's Golden Rule: perturbative acoustic emission rate (Kelvin, cubic V₃).
f_ac = 0 at Γ is trivial (translations have H = 0); the meaningful result
is test_total_rate_positive (2-phonon decay exists). f ∝ η is expected
(Lorentzian on discrete spectrum).

- **test_one_phonon_forbidden** -- 1-phonon emission forbidden (kinematic gap: omega_belt > omega_edge)
- **test_ac_ac_negligible** -- ac+ac channel < 1% of total (trivially zero at Γ: see NOTES)
- **test_low_omega_fraction_small** -- Acoustic fraction < 30% of total rate (trivially zero at Γ)
- **test_total_rate_positive** -- Total 2-phonon rate finite and positive (mechanism works)
- **test_optical_dominates** -- Optical emission dominates over acoustic emission (trivially at Γ)
- **test_omega_edge_computed** -- ω_edge = 0.8022 computed from BZ scan (not hardcoded)
- **test_rate_linear_in_eta** -- f/η = 1.50e-3 constant (spread 0.1%): Lorentzian on discrete spectrum
- **test_lifetime_estimate** -- τ = 1.33×10⁸ periods at ε=0.01; rate table ε=0.001→10¹⁰, ε=0.1→10⁶
- **test_belt_mode_coverage** -- 100% of c² in ω > ω_edge; top-5 belt modes capture 37%

### 5_test_md_cubic.py (4 tests)

MD validation: cubic anharmonicity energy transfer (Kelvin N=3, 972 DOF).

- **test_harmonic_baseline** -- Harmonic: E_low stays at machine zero (no leakage)
- **test_eps_squared_scaling** -- E_low/eps^2 constant across 3 eps values [0.03, 0.06, 0.1] (spread 0.0%)
- **test_com_zero** -- COM drift stays at machine zero under cubic anharmonicity
- **test_energy_conservation** -- Energy conserved to < 1e-5 relative drift

### 6_test_dipolar_scaling.py (8 tests)

Dipolar scaling: hop source radiates as k^2 not k^0 (C15).

- **test_monopole_zero_all_pairs** -- M₀ = 0 (machine zero) on all Z12-Z12 hop sources
- **test_dipole_nonzero** -- M₁ ~ O(1): dipole channel exists (leading multipole)
- **test_hop_source_k_squared** -- Acoustic overlap scales as k^2 on [100]/[110]/[111]
- **test_monopole_source_k_zero** -- Constant-force source (monopole) gives k^0 scaling (negative control)
- **test_k_squared_multiple_pairs** -- k^2 slope = 2.000 on 5 pairs × 3 BZ dirs (universality)
- **test_z16_monopole_nonzero** -- Z16-Z16 hop sources: M₀ > 0.5 on 14/16 pairs (negative control)
- **test_z12_z16_hop_monopole** -- Z12-Z16 hop: M₀ = 0.52, slope ≈ 0 (monopole, not dipolar)
- **test_dipolar_scaling_vs_L_cell** -- slope = 2.0 at L=2,4,8 (geometric, not discretization)

### 7_test_wp.py (18 tests)

Weaire-Phelan foam: all claims verified on third structure. WP has 2 Type A
(dodecahedra, 12 pentagon faces) + 6 Type B (tetrakaidecahedra, 14 faces).
Type A has same antipodal tilt mechanism as C15 Z12.

- **test_wp_topology** -- V=46, E=92, F=54, C=8, χ₃=0; 2 Type A + 6 Type B; 48 pentagons + 6 hexagons
- **test_wp_type_a_selection_rule** -- M₀ = 0 on both Type A cells (machine zero, ~1e-15)
- **test_wp_type_b_nonzero** -- M₀ ∈ [0.669, 0.824] on Type B (negative control)
- **test_wp_type_a_antipodal_tilt** -- n_ax[i] = −n_ax[i+3] on Type A (same as C15 Z12)
- **test_wp_type_a_even_m_protected** -- Even m (2,4) protected, odd m (1,3,5) not
- **test_wp_gap_exists** -- Gap ratio = 1.694 (ω_edge=1.001, ω_belt_min=1.696)
- **test_wp_gap_all_bz_directions** -- Gap > 1 in all 9 BZ directions (worst [111]: 1.727)
- **test_wp_gap_survives_isotropy** -- Gap ratio 1.283 at k_L=k_T=1 (structural)
- **test_wp_gap_convergence** -- ω_edge converged: Δ=0 between n_k=40 and n_k=80
- **test_wp_subsonic_centroid** -- v_g_centroid/v_T = 0.397 (subsonic); v_g_max/v_T = 1.016 (marginal)
- **test_wp_gap_vs_spring_ratio** -- Gap > 1 for all k_L/k_T ∈ [1,3] (min 1.283 at isotropy)
- **test_wp_wavepacket_gamma** -- ||Q_trans^T u(t)||² < 1e-25 over 4 periods
- **test_wp_belt_covers_source** -- Belt basis residual = 5.89e-17
- **test_wp_source_cell_independence** -- Both Type A cells give machine zero acoustic content
- **test_wp_type_b_loophole_bound** -- Type B loophole: particle-band COM content < 1.94% across 7 BZ dirs (Type A-only belt basis, 75% unprotected)
- **test_wp_hop_source_monopole_zero** -- Hop source M₀ = 6.26e-16 (machine zero)
- **test_wp_hop_source_dipole_exists** -- |M₁| = 6.26 (dipole channel exists)
- **test_wp_hop_source_k_squared** -- Acoustic overlap ~ k² on [100]/[110]/[111] (slope ≈ 2.0)

## Paper Mapping

| Paper claim | Tests |
|-------------|-------|
| Part 1a: Harmonic theorem (Bloch modes exact) | 3 (wavepacket) |
| Part 1b: Kinematic gap (belt above acoustic ceiling) | 2 (gap exists, values, basis dims) |
| Part 1c: Subsonic particle velocity v_g < v_T | 2 (explicit: C15 0.88, Kelvin 0.54) |
| Part 1d: Selection rule M₀ = 0 | 1 (Z12, Z16 control, Fourier, Kelvin) |
| Part 1f: Beyond harmonic (FGR cubic) | 4 (1-phonon forbidden, channel decomposition) |
| Part 1g: MD validation (eps^2 scaling) | 5 (harmonic baseline, scaling, COM, energy) |
| Part 2a: Dipolar scaling |M(q)|^2 ~ q^2 | 6 (monopole zero, dipole O(1), k^2 fit) |
| Part 2b: Kinematic gap (rotation) | 2 (same gap, shared claim) |
| Part 2c: Harmonic theorem (rotation) | 3 (same theorem, shared claim) |

## Source Modules

| Module | Path | Description |
|--------|------|-------------|
| no_drag | core_math/analysis/no_drag.py | Belt basis, selection rule, gap, velocities |
| cell_topology | core_math/analysis/cell_topology.py | Cell geometry, circuit finding, caps |
| c15_periodic | core_math/builders/c15_periodic.py | C15 supercell builder |
| wp_periodic | core_math/builders/wp_periodic.py | WP (Weaire-Phelan) supercell builder |
| multicell_periodic | core_math/builders/multicell_periodic.py | Kelvin/BCC supercell builder |
| kelvin | core_math/builders/kelvin.py | Base Kelvin cell |
| md_foam | core_math/dynamics/md_foam.py | MD integrator (Verlet, forces, energies) |
| incidence | core_math/operators/incidence.py | d₀, d₁ operators |
| structures | core_math/spec/structures.py | canonical_face ordering |
| constants (spec) | core_math/spec/constants.py | Numeric constants |
| bloch | physics/bloch.py | DisplacementBloch class |
| constants (physics) | physics/constants.py | Physical constants |
