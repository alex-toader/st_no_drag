"""
Periodic Weaire-Phelan Supercell via Voronoi
=============================================

N×N×N Weaire-Phelan foam with PERIODIC boundary conditions (3-torus T³).

CONSTRUCTION:
    1. Generate A15 lattice points (space group Pm3n, No. 223)
    2. Compute Voronoi tessellation with periodic images
    3. Order ridge vertices cyclically in face plane
    4. Extract vertices, edges, faces with wrapping

STRUCTURE:
    8 cells per fundamental domain:
        - 2 Type A (dodecahedra) at Wyckoff 2a
        - 6 Type B (tetrakaidecahedra) at Wyckoff 6d

TOPOLOGY (verified at build time):
    - χ(2-skeleton) = V - E + F = C (number of cells)
    - χ(3-complex) = V - E + F - C = 0 (3-torus T³)
    - Every vertex has degree 4 (Plateau vertex)
    - Every edge bounds exactly 3 faces (Plateau border)
    - Every face shared by exactly 2 cells
    - d₁ @ d₀ = 0 (orientations consistent)
    - Faces: pentagons and hexagons only
    - Type A (2a): 12 faces, Type B (6d): 14 faces

Date: Mar 2026
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import Tuple, List, Dict
from itertools import product

from ..spec.structures import canonical_face as canonical_face_with_orient
from ..spec.constants import WRAP_DECIMALS, EPS_CLOSE


def wrap_coord(x: float, L: float) -> float:
    """Wrap coordinate to [0, L)."""
    result = x % L
    if abs(result) < EPS_CLOSE or abs(result - L) < EPS_CLOSE:
        result = 0.0
    return result


def wrap_pos(pos: np.ndarray, L: float) -> tuple:
    """Wrap 3D position to canonical form in [0, L)³."""
    return tuple(round(wrap_coord(x, L), WRAP_DECIMALS) for x in pos)


def unwrap_coords_to_reference(coords: np.ndarray, L: float) -> np.ndarray:
    """
    Unwrap periodic coordinates to same image.

    For faces crossing periodic boundary, vertices may be in different
    periodic images (offset by ±L). This function brings them all to
    the same image by using the first vertex as reference.
    """
    if len(coords) == 0:
        return coords

    unwrapped = coords.copy()
    ref = unwrapped[0]

    for i in range(1, len(unwrapped)):
        for j in range(3):
            diff = unwrapped[i, j] - ref[j]
            if diff > L/2:
                unwrapped[i, j] -= L
            elif diff < -L/2:
                unwrapped[i, j] += L

    return unwrapped


def order_ridge_vertices(ridge_coords: np.ndarray, site1: np.ndarray, site2: np.ndarray) -> List[int]:
    """
    Order ridge vertices cyclically in the face plane.

    Ridge vertices from Voronoi are NOT guaranteed to be in cyclic order.
    This function projects them onto the face plane and sorts by angle.
    """
    n = len(ridge_coords)
    if n < 3:
        return list(range(n))

    # Face normal: direction from site1 to site2
    normal = site2 - site1
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return list(range(n))
    normal = normal / norm_len

    # Centroid of ridge vertices
    centroid = np.mean(ridge_coords, axis=0)

    # Build orthonormal basis in face plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = arbitrary - np.dot(arbitrary, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project vertices onto (u, v) plane and compute angles
    angles = []
    for i, coord in enumerate(ridge_coords):
        rel = coord - centroid
        proj_u = np.dot(rel, u)
        proj_v = np.dot(rel, v)
        angle = np.arctan2(proj_v, proj_u)
        angles.append((angle, i))

    angles.sort(key=lambda x: x[0])
    return [idx for _, idx in angles]


def get_a15_points(N: int, L_cell: float = 1.0) -> np.ndarray:
    """
    Generate A15 lattice points for N×N×N supercell.

    A15 Wyckoff positions (space group Pm3n, No. 223):
        2a: (0,0,0), (1/2,1/2,1/2)
        6d: (1/4,0,1/2), (3/4,0,1/2), (1/2,1/4,0),
            (1/2,3/4,0), (0,1/2,1/4), (0,1/2,3/4)

    Total: 8 sites per unit cell
    """
    frac = [
        [0, 0, 0], [0.5, 0.5, 0.5],           # 2a
        [0.25, 0, 0.5], [0.75, 0, 0.5],        # 6d
        [0.5, 0.25, 0], [0.5, 0.75, 0],
        [0, 0.5, 0.25], [0, 0.5, 0.75],
    ]
    points = []
    for i, j, k in product(range(N), repeat=3):
        for f in frac:
            p = [(i + f[0]) * L_cell, (j + f[1]) * L_cell, (k + f[2]) * L_cell]
            points.append(p)
    return np.array(points)


def build_wp_supercell_periodic(
    N: int,
    L_cell: float = 4.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Build N×N×N Weaire-Phelan supercell with periodic boundary conditions.

    Uses Voronoi tessellation of A15 lattice.

    Args:
        N: supercell size (8N³ cells total)
        L_cell: side of fundamental domain

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists
        cell_face_incidence: list of [(face_idx, orientation)] per cell

    TOPOLOGY:
        - χ(3-complex) = 0 (3-torus T³)
        - Every edge bounds exactly 3 faces (Plateau foam)
        - Every vertex has degree 4
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if L_cell <= 0:
        raise ValueError(f"L_cell must be > 0, got {L_cell}")

    L = N * L_cell
    points = get_a15_points(N, L_cell)
    n_pts = len(points)

    # Create 3×3×3 periodic images for Voronoi
    images = []
    image_offset = []
    for di, dj, dk in product([-1, 0, 1], repeat=3):
        offset = np.array([di, dj, dk]) * L
        images.append(points + offset)
        image_offset.append((di, dj, dk))

    all_points = np.vstack(images)
    central_idx = image_offset.index((0, 0, 0))
    central_start = central_idx * n_pts
    central_end = central_start + n_pts

    # Compute Voronoi
    vor = Voronoi(all_points)

    # Collect unique vertices and faces with wrapping
    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []

    # face_data[canonical] = {
    #     'face': list (canonical vertex order),
    #     'cells': {cell_idx: orientation}
    # }
    face_data: Dict[tuple, dict] = {}

    def get_vertex_idx(pos: np.ndarray) -> int:
        wrapped = wrap_pos(pos, L)
        if wrapped not in vertex_dict:
            vertex_dict[wrapped] = len(vertices)
            vertices.append(np.array(wrapped))
        return vertex_dict[wrapped]

    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        ridge_verts = vor.ridge_vertices[ridge_idx]

        # Skip unbounded ridges
        if -1 in ridge_verts:
            continue

        # Only process if at least one point is in central cell
        in_c1 = central_start <= p1 < central_end
        in_c2 = central_start <= p2 < central_end

        if not (in_c1 or in_c2):
            continue

        # Get ridge vertex coordinates
        ridge_coords = np.array([vor.vertices[v_idx] for v_idx in ridge_verts])

        # Unwrap coordinates to same periodic image before ordering
        ridge_coords_unwrapped = unwrap_coords_to_reference(ridge_coords, L)

        # Order vertices cyclically in face plane
        # Normal direction: site1 → site2
        site1 = all_points[p1]
        site2 = all_points[p2]
        ordered_indices = order_ridge_vertices(ridge_coords_unwrapped, site1, site2)

        # Map vertices with wrapping, in cyclic order
        face = []
        for local_idx in ordered_indices:
            pos = ridge_coords_unwrapped[local_idx]
            new_idx = get_vertex_idx(pos)
            face.append(new_idx)

        # Skip degenerate faces
        unique_verts = []
        for v in face:
            if not unique_verts or v != unique_verts[-1]:
                unique_verts.append(v)
        if len(unique_verts) > 1 and unique_verts[0] == unique_verts[-1]:
            unique_verts = unique_verts[:-1]
        face = unique_verts

        if len(face) < 3:
            continue

        # Skip faces with repeated vertices (non-simple cycles after wrapping)
        if len(set(face)) != len(face):
            continue

        # Canonicalize with orientation tracking
        try:
            canon, rel_orient = canonical_face_with_orient(face)
        except ValueError:
            continue

        # Cell indices: Voronoi point index mod n_pts gives cell in fundamental domain
        cell_p1 = p1 % n_pts
        cell_p2 = p2 % n_pts

        if canon not in face_data:
            face_data[canon] = {
                'face': list(canon),
                'cells': {}
            }

        # Orientation convention:
        # - Face vertices ordered with normal from site1 → site2
        # - rel_orient = +1: canonical winding matches face → normal points p1→p2
        #   → Cell p1: outward (+rel_orient), Cell p2: inward (-rel_orient)
        # - rel_orient = -1: canonical winding reversed
        #   → Cell p1: outward (+rel_orient), Cell p2: inward (-rel_orient)
        cells = face_data[canon]['cells']
        if in_c1:
            if cell_p1 in cells and cells[cell_p1] != rel_orient:
                raise ValueError(
                    f"Orientation inconsistency: cell {cell_p1}, face {canon[:3]}...")
            cells[cell_p1] = rel_orient
        if in_c2:
            if cell_p2 in cells and cells[cell_p2] != -rel_orient:
                raise ValueError(
                    f"Orientation inconsistency: cell {cell_p2}, face {canon[:3]}...")
            cells[cell_p2] = -rel_orient

    # Build face list and canonical-to-index mapping
    faces = []
    canonical_to_face_idx = {}
    for canonical, data in face_data.items():
        canonical_to_face_idx[canonical] = len(faces)
        faces.append(data['face'])

    # Build cell_face_incidence: for each cell, list of (face_idx, orientation)
    n_cells = 8 * N**3
    cell_face_incidence: List[List[Tuple[int, int]]] = [[] for _ in range(n_cells)]
    for canonical, data in face_data.items():
        face_idx = canonical_to_face_idx[canonical]
        for cell_idx, orientation in data['cells'].items():
            cell_face_incidence[cell_idx].append((face_idx, orientation))

    # Build edges from faces
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k+1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_set.add(edge)

    edges = sorted(edge_set)

    # Verify foam invariants
    V, E, F = len(vertices), len(edges), len(faces)
    C = 8 * N**3

    # χ(2-skeleton) = C ⟹ χ(3-complex) = V - E + F - C = 0
    if V - E + F != C:
        raise ValueError(f"χ₂ = {V-E+F}, expected C={C}")

    # Vertex degree = 4 (Plateau vertex)
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

    # Faces: pentagons and hexagons only (A15 / Weaire-Phelan)
    face_sizes = Counter(len(f) for f in faces)
    if not set(face_sizes.keys()).issubset({5, 6}):
        raise ValueError(f"Unexpected face sizes: {face_sizes}")

    # Cell face counts: 2a (dodecahedra) = 12 faces, 6d (tetrakaidecahedra) = 14
    for ci, cfi_ci in enumerate(cell_face_incidence):
        expected = 12 if ci % 8 < 2 else 14
        if len(cfi_ci) != expected:
            raise ValueError(f"Cell {ci}: {len(cfi_ci)} faces, expected {expected}")

    # d₁d₀ = 0 (exactness)
    from ..operators.incidence import build_d0, build_d1
    verts_arr = np.array(vertices)
    d0 = build_d0(verts_arr, edges)
    d1 = build_d1(verts_arr, edges, faces)
    d1d0 = d1 @ d0
    if d1d0.any():
        raise ValueError(f"d₁d₀ ≠ 0: max entry = {abs(d1d0).max()}")

    return verts_arr, edges, faces, cell_face_incidence
