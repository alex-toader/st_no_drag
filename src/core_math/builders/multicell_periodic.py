"""
Periodic BCC Multicell
======================

N×N×N BCC supercell with PERIODIC boundary conditions (3-torus T³).

GEOMETRY:
    Period: L = 4N in each direction
    All coordinates wrapped: x → x mod L

TOPOLOGY (verified Mar 2026, checked at build time):
    This is a FOAM (Plateau structure), NOT a 2-manifold!

    - V = 12N³ (24 vertices/cell, each shared by 4 cells)
    - χ(2-skeleton) = V - E + F = C (number of 3-cells)
    - χ(3-complex) = V - E + F - C = 0 (correct for T³)
    - Every vertex has degree 4 (Plateau vertex)
    - Every edge bounds exactly 3 faces (Plateau border)
    - Every face shared by exactly 2 cells
    - d₁ @ d₀ = 0 (orientations consistent)

    Key insight: BCC Kelvin is space-filling foam where each edge
    is a Plateau border (junction of 3 soap films), not 2.

COMPARISON with non-periodic (multicell.py):
    - multicell.py: open boundary, boundary effects
    - This module: periodic, proper k-vector, no boundary

PARITY (verified Jan 2026):
    - Parity P: x → (L - x) mod L is FREE action (no fixed vertices/edges)
    - Tr(P|_H) = 1 (Lefschetz holds)
    - BUT: P anti-commutes with d₁ᵀ in single cell, NOT in periodic
    - Result: antipodal currents → bridge (single cell) vs ring (periodic)

Date: Jan 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Set
from itertools import product

from .kelvin import build_kelvin_cell
from ..spec.structures import canonical_face
from ..spec.constants import WRAP_DECIMALS, EPS_CLOSE


def wrap_coord(x: float, L: float) -> float:
    """Wrap coordinate to [0, L) with tolerance for numerical precision."""
    result = x % L
    # Snap to 0 if very close to 0 (from rounding noise on exact multiples).
    # The abs(result - L) check is defensive: % should return [0, L), but
    # some edge cases with float precision can produce values very close to L.
    if abs(result) < EPS_CLOSE or abs(result - L) < EPS_CLOSE:
        result = 0.0
    return result


def wrap_position(pos: np.ndarray, L: float) -> tuple:
    """Wrap 3D position to canonical form in [0, L)³."""
    wrapped = np.array([wrap_coord(x, L) for x in pos])
    return tuple(np.round(wrapped, WRAP_DECIMALS))


def generate_bcc_centers(N: int) -> List[Tuple[float, float, float]]:
    """
    Generate BCC cell centers for N×N×N supercell.

    SET A: (4i, 4j, 4k) for i,j,k ∈ [0, N)
    SET B: (4i+2, 4j+2, 4k+2) for i,j,k ∈ [0, N)

    Total: 2N³ cells
    """
    centers = []

    for i, j, k in product(range(N), repeat=3):
        centers.append((4.0*i, 4.0*j, 4.0*k))
        centers.append((4.0*i + 2.0, 4.0*j + 2.0, 4.0*k + 2.0))

    return centers


def build_bcc_supercell_periodic(N: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Build N×N×N BCC supercell with periodic boundary conditions.

    Args:
        N: supercell size (2N³ cells total)

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists (CCW orientation)
        cell_face_incidence: list of (face_idx, orientation) per cell

    TOPOLOGY:
        χ(2-skeleton) = V - E + F = C (number of cells)
        χ(3-complex) = V - E + F - C = 0 (3-torus)
        Every edge bounds exactly 3 faces (Plateau border)
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    L = 4.0 * N  # period

    # Get base Kelvin cell
    base_v, base_e, base_f, _ = build_kelvin_cell()
    base_v = np.array(base_v)

    # Step 1: Collect all vertices with periodic identification
    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []

    centers = generate_bcc_centers(N)

    # Map: (cell_idx, local_v_idx) -> global_v_idx
    cell_vertex_map: List[List[int]] = []

    for center in centers:
        center = np.array(center)
        cell_map = []

        for local_v in base_v:
            pos = center + local_v
            canonical = wrap_position(pos, L)

            if canonical not in vertex_dict:
                vertex_dict[canonical] = len(vertices)
                vertices.append(np.array(canonical))

            cell_map.append(vertex_dict[canonical])

        cell_vertex_map.append(cell_map)

    # Step 2: Collect edges with deduplication
    edge_set: Set[Tuple[int, int]] = set()

    for cell_idx, cell_map in enumerate(cell_vertex_map):
        for i, j in base_e:
            gi, gj = cell_map[i], cell_map[j]
            edge = (min(gi, gj), max(gi, gj))
            edge_set.add(edge)

    edges = sorted(edge_set)

    # Step 3: Collect faces with deduplication using canonical ordering
    # This preserves orientation information for d₂
    #
    # face_data[canonical] = {
    #     'face': list (canonical vertex order),
    #     'cells': {cell_idx: orientation}
    # }
    # where orientation = +1 if cell sees canonical direction as outward, -1 if inward
    face_data: Dict[tuple, dict] = {}

    for cell_idx, cell_map in enumerate(cell_vertex_map):
        for local_face in base_f:
            global_face = [cell_map[v] for v in local_face]
            canonical, rel_orient = canonical_face(global_face)

            if canonical not in face_data:
                face_data[canonical] = {
                    'face': list(canonical),
                    'cells': {}
                }

            # Orientation from rel_orient:
            # - Kelvin cell faces have CCW orientation when viewed from outside the cell
            # - rel_orient = +1 means global_face has same winding as canonical
            #   → canonical normal points outward from this cell → orientation = +1
            # - rel_orient = -1 means global_face has opposite winding
            #   → canonical normal points inward to this cell → orientation = -1
            #
            # This works because two cells sharing a face see it from opposite sides,
            # so their local face windings are reversed, giving opposite rel_orient values.
            face_data[canonical]['cells'][cell_idx] = rel_orient

    # Build face list and cell-face incidence
    faces = []
    canonical_to_face_idx = {}
    for canonical, data in face_data.items():
        canonical_to_face_idx[canonical] = len(faces)
        faces.append(data['face'])

    # Build cell_face_incidence: for each cell, list of (face_idx, orientation)
    n_cells_total = len(centers)
    cell_face_incidence = [[] for _ in range(n_cells_total)]
    for canonical, data in face_data.items():
        face_idx = canonical_to_face_idx[canonical]
        for cell_idx, orientation in data['cells'].items():
            cell_face_incidence[cell_idx].append((face_idx, orientation))

    # Verify foam invariants
    V, E, F = len(vertices), len(edges), len(faces)
    C = 2 * N**3

    # V = 12N³ (24 vertices per cell, each shared by 4 cells)
    if V != 12 * N**3:
        raise ValueError(f"Expected V={12*N**3}, got {V}")

    # χ(2-skeleton) = C, χ(3-complex) = 0
    if V - E + F != C:
        raise ValueError(f"χ₂ = {V-E+F}, expected C={C}")

    # Vertex degree = 4 (Plateau vertex: 4 borders meet)
    from collections import Counter
    deg = Counter()
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    bad_deg = {v: d for v, d in deg.items() if d != 4}
    if bad_deg:
        raise ValueError(f"{len(bad_deg)} vertices not degree 4")

    # Each edge bounds exactly 3 faces (Plateau border)
    edge_face_count = Counter()
    for face in faces:
        n = len(face)
        for k in range(n):
            e = (min(face[k], face[(k+1)%n]), max(face[k], face[(k+1)%n]))
            edge_face_count[e] += 1
    bad_ef = {e: c for e, c in edge_face_count.items() if c != 3}
    if bad_ef:
        raise ValueError(f"{len(bad_ef)} edges not in exactly 3 faces")

    # Each face shared by exactly 2 cells
    for canonical, data in face_data.items():
        nc = len(data['cells'])
        if nc != 2:
            raise ValueError(f"Face has {nc} cells, expected 2")

    # d₁d₀ = 0 (exactness: consistent orientations)
    from ..operators.incidence import build_d0, build_d1
    verts_arr = np.array(vertices)
    d0 = build_d0(verts_arr, edges)
    d1 = build_d1(verts_arr, edges, faces)
    product = d1 @ d0
    if product.any():
        raise ValueError(f"d₁d₀ ≠ 0: max entry = {abs(product).max()}")

    return verts_arr, edges, faces, cell_face_incidence
