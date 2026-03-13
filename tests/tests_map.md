# Tests Map

## Test Files (28 tests, ~3 min total)

### 1_test_selection_rule.py (5 tests)

Selection rule M₀ = 0 for belt m=2 mode.

- **test_c15_z12_selection_rule** -- M₀ = 0 on all 16 Z12 cells (exact by symmetry)
- **test_c15_z16_nonzero** -- M₀ != 0 on Z16 cells (negative control)
- **test_c15_z12_no_m2_normal_tilt** -- Z12 normal-tilt Fourier spectrum has no m=2 component (WHY M₀ = 0)
- **test_kelvin_selection_rule** -- M₀ = 0 on all 16 Kelvin cells (all identical truncated octahedra)
- **test_c15_belt_counts** -- All 24 C15 cells have belts; Z12 has N=6, Z16 has N=8

### 2_test_kinematic_gap.py (6 tests)

Kinematic gap: belt frequencies above acoustic ceiling. Subsonic particle velocity.

- **test_c15_gap_exists** -- C15 gap ratio omega_belt_min / omega_edge > 1.2
- **test_c15_belt_above_acoustic** -- omega_edge ~ 0.69, omega_belt_min ~ 1.15 (expected ranges)
- **test_kelvin_gap_exists** -- Kelvin gap ratio > 1
- **test_c15_belt_basis_dimensions** -- C15 belt basis: 96D enriched, 16D particle (cos2theta x n_hat)
- **test_c15_subsonic** -- C15 v_g_max / v_T < 1 (measured 0.88)
- **test_kelvin_subsonic** -- Kelvin v_g_max / v_T < 1 (measured 0.54)

### 3_test_wavepacket.py (4 tests)

Wavepacket: zero acoustic emission from belt excitation (Kelvin, Gamma-point).

- **test_zero_translation_content** -- ||Q_trans^T u(t)||^2 < 1e-25 over 4 periods
- **test_zero_acoustic_energy** -- Energy in acoustic modes (acousticness > 10%) is machine zero
- **test_zero_mode_energy** -- 3 zero-frequency modes (translations), energy < 1e-40
- **test_drift_smaller_than_spreading** -- Drift << spreading at Gamma (no directed transport)

### 4_test_fgr_cubic.py (5 tests)

Fermi's Golden Rule: perturbative acoustic emission rate (Kelvin, cubic V₃).

- **test_one_phonon_forbidden** -- 1-phonon emission forbidden (kinematic gap: omega_belt > omega_edge)
- **test_ac_ac_negligible** -- ac+ac channel < 1% of total (both final states acoustic)
- **test_low_omega_fraction_small** -- Acoustic fraction < 30% of total rate
- **test_total_rate_positive** -- Total 2-phonon rate finite and positive (mechanism works)
- **test_optical_dominates** -- Optical emission dominates over acoustic emission

### 5_test_md_cubic.py (4 tests)

MD validation: cubic anharmonicity energy transfer (Kelvin N=3, 972 DOF).

- **test_harmonic_baseline** -- Harmonic: E_low stays at machine zero (no leakage)
- **test_eps_squared_scaling** -- E_low/eps^2 is constant across eps values (scaling confirmed)
- **test_com_zero** -- COM drift stays at machine zero under cubic anharmonicity
- **test_energy_conservation** -- Energy conserved to < 1e-5 relative drift

### 6_test_dipolar_scaling.py (4 tests)

Dipolar scaling: hop source radiates as k^2 not k^0 (C15).

- **test_monopole_zero_all_pairs** -- M₀ = 0 (machine zero) on all Z12-Z12 hop sources
- **test_dipole_nonzero** -- M₁ ~ O(1): dipole channel exists (leading multipole)
- **test_hop_source_k_squared** -- Acoustic overlap scales as k^2 on [100]/[110]/[111]
- **test_monopole_source_k_zero** -- Constant-force source (monopole) gives k^0 scaling (negative control)

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
| multicell_periodic | core_math/builders/multicell_periodic.py | Kelvin/BCC supercell builder |
| kelvin | core_math/builders/kelvin.py | Base Kelvin cell |
| md_foam | core_math/dynamics/md_foam.py | MD integrator (Verlet, forces, energies) |
| incidence | core_math/operators/incidence.py | d₀, d₁ operators |
| structures | core_math/spec/structures.py | Mesh contract |
| constants (spec) | core_math/spec/constants.py | Numeric constants |
| bloch | physics/bloch.py | DisplacementBloch class |
| constants (physics) | physics/constants.py | Physical constants |
