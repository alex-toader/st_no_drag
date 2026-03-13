# st_no_drag

No-drag protection for belt-mode excitations in 3D foam lattices.

**Author:** Alexandru Toader (toader_alexandru@yahoo.com)

## Result

The m=2 belt mode (shear deformation on hexagonal faces) propagates between foam cells without losing energy to the acoustic sector. Three independent protections:

1. **Selection rule M₀ = 0** -- the net force from belt m=2 pressure is exactly zero on Z12 cells (C15) and all Kelvin cells. At finite q, coupling is dipolar (|M(q)|² ~ q²), not monopolar.
2. **Kinematic gap** -- belt frequencies sit above the acoustic ceiling (gap ratio 1.21 on C15 with COM-coherent definition, 1.33 on Kelvin). No sound-like mode exists at belt frequencies.
3. **Harmonic theorem** -- Bloch eigenmodes are exact stationary states. No irreversible energy transfer between bands.

Beyond harmonic: cubic anharmonicity (Fermi's Golden Rule) gives a perturbative decay rate Gamma/omega = eps² * f with f ~ 10⁻⁵ (Kelvin). MD validation confirms eps² scaling.

Verified on two foam geometries: C15 (Laves phase, 16 Z12 + 8 Z16 cells) and Kelvin (BCC Voronoi, truncated octahedra).

## Tests

28 tests across 6 files (~3 min total). Each test file targets one claim of the no-drag mechanism.

See `tests/tests_map.md` for the complete inventory with per-test descriptions and paper mappings.

```
tests/
├── 1_test_selection_rule.py     (5 tests)  M₀ = 0 on Z12/Kelvin, M₀ != 0 on Z16
├── 2_test_kinematic_gap.py      (6 tests)  Belt above acoustic ceiling + subsonic velocity
├── 3_test_wavepacket.py         (4 tests)  Zero acoustic emission at Gamma (Kelvin)
├── 4_test_fgr_cubic.py          (5 tests)  FGR channel decomposition (Kelvin)
├── 5_test_md_cubic.py           (4 tests)  MD validation: eps² scaling, energy conservation
└── 6_test_dipolar_scaling.py    (4 tests)  Hop source radiates as k² not k⁰ (C15)
```

## Source code

```
src/
├── core_math/
│   ├── analysis/
│   │   ├── no_drag.py           Belt basis, selection rule, gap, velocities
│   │   └── cell_topology.py     Cell geometry, circuit finding, caps
│   ├── builders/
│   │   ├── c15_periodic.py      C15 supercell builder
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
# All tests (~3 min)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/ -v

# Single test file
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/1_test_selection_rule.py -v
```

## Requirements

- Python 3.9+
- NumPy, SciPy

## License

MIT
