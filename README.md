# st_no_drag

> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19035807.svg)](https://doi.org/10.5281/zenodo.19035807)
No-drag protection for belt-mode excitations in 3D foam lattices.

**Author:** Alexandru Toader (toader_alexandru@yahoo.com)

## Result

The m=2 belt mode (shear deformation on equatorial faces) propagates between foam cells without losing energy to the acoustic sector. Two geometric conditions:

1. **Force monopole cancellation M₀ = 0** — centrosymmetric cells produce antipodal force cancellation. At finite k, coupling is dipolar (|M(k)|² ~ k²), not monopolar.
2. **Kinematic gap ω_belt > ω_edge** — belt frequencies sit above the acoustic ceiling. Single-phonon decay is forbidden by energy conservation.

When both hold: exact zero emission at the zone center, k² suppression at finite k, and anharmonic lifetime of ~10⁸ oscillation periods at ε = 0.01.

Verified on three foam geometries:
- **C15 Laves** (Fd3̄m, 24 cells: 16 Z12 + 8 Z16)
- **Kelvin BCC** (Im3̄m, 16 truncated octahedra)
- **Weaire–Phelan** (Pm3̄n, 8 cells: 2 Type A + 6 Type B)

## Tests

82 tests across 7 files (~10 min total). Each test file targets one claim.

See `tests/tests_map.md` for the complete inventory with per-test descriptions.

```
tests/
├── 1_test_selection_rule.py     (15 tests)  M₀ = 0 selection rule + mechanism
├── 2_test_kinematic_gap.py      (18 tests)  Belt above acoustic ceiling + gap robustness
├── 3_test_wavepacket.py         (10 tests)  Zero acoustic emission at Gamma
├── 4_test_fgr_cubic.py          ( 9 tests)  FGR channel decomposition (Kelvin)
├── 5_test_md_cubic.py           ( 4 tests)  MD validation: ε² scaling, energy conservation
├── 6_test_dipolar_scaling.py    ( 8 tests)  Hop source radiates as k² not k⁰ (C15)
└── 7_test_wp.py                 (18 tests)  Weaire–Phelan: all claims on third structure
```

## Paper

`paper/main.tex` — full manuscript draft with 4 figures.

## Source code

```
src/
├── core_math/
│   ├── analysis/
│   │   ├── no_drag.py           Belt basis, selection rule, gap, velocities
│   │   └── cell_topology.py     Cell geometry, circuit finding, caps
│   ├── builders/
│   │   ├── c15_periodic.py      C15 supercell builder
│   │   ├── wp_periodic.py       Weaire–Phelan supercell builder
│   │   ├── multicell_periodic.py  Kelvin/BCC supercell builder
│   │   └── kelvin.py            Base Kelvin cell
│   ├── dynamics/
│   │   └── md_foam.py           MD integrator (Verlet, forces, energies)
│   ├── operators/
│   │   └── incidence.py         d₀, d₁ operators
│   └── spec/
│       ├── structures.py        Mesh contract
│       └── constants.py         Numeric constants
└── physics/
    ├── bloch.py                 DisplacementBloch class
    └── constants.py             Physical constants
```

## Running tests

```bash
# All tests (~10 min)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/ -v

# Single test file
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/1_test_selection_rule.py -v
```

## Requirements

- Python 3.9+
- NumPy, SciPy

## License

MIT
