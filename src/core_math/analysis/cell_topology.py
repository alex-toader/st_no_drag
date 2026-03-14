"""
Cell Topology Utilities
========================

Topological analysis of foam cells: belt finding and cap assignment.

Groups:
  1. Periodic geometry — periodic_delta
  2. Cell extraction — get_cell_geometry
  3. Circuit finding — find_simple_cycles, equatorial_test,
                       circuit_holonomy, find_best_belt
  4. Cap assignment — assign_caps_bfs

Feb 2026
"""

import numpy as np
from collections import defaultdict, deque


# ═══════════════════════════════════════════════════════════════
# 1. PERIODIC GEOMETRY
# ═══════════════════════════════════════════════════════════════

def periodic_delta(a, b, L):
    """
    Minimum-image displacement b - a under periodic boundary conditions.

    Args:
        a, b: points (arrays or scalars)
        L: box size (scalar for cubic box)

    Returns:
        displacement vector (b - a) wrapped to [-L/2, L/2]
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return d - L * np.round(d / L)


# ═══════════════════════════════════════════════════════════════
# 2. CELL EXTRACTION
# ═══════════════════════════════════════════════════════════════

def get_cell_geometry(cell_idx, cell_center, vertices, faces, cfi, L):
    """
    Extract geometry for one cell: unwrap vertices, compute face centers,
    build face adjacency graph.

    Args:
        cell_idx: index into cfi
        cell_center: 3D center of cell
        vertices: global vertex array
        faces: global face list (each face = list of vertex indices)
        cfi: cell-face-incidence (list of lists of (face_idx, orient))
        L: box size

    Returns:
        face_data: list of dicts per face:
            'f_idx': global face index
            'center': 3D face center (unwrapped)
            'n_sides': number of vertices
            'edges': dict of edge_key → edge midpoint
            'vertices': Nx3 unwrapped vertex positions
            'vertex_ids': list of global vertex indices
        adj: dict of dict, face adjacency via shared edges
            adj[i][j] = edge_key for adjacent faces i, j
    """
    face_list = cfi[cell_idx]
    face_data = []
    edge_to_faces = defaultdict(list)

    for local_i, (f_idx, orient) in enumerate(face_list):
        fv = faces[f_idx]
        unwrapped = np.array([
            cell_center + periodic_delta(cell_center, vertices[v], L)
            for v in fv
        ])
        fc = unwrapped.mean(axis=0)

        edges_of_face = {}
        for j in range(len(fv)):
            v1_idx, v2_idx = fv[j], fv[(j + 1) % len(fv)]
            e_key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            mid = (unwrapped[j] + unwrapped[(j + 1) % len(unwrapped)]) / 2
            edges_of_face[e_key] = mid
            edge_to_faces[e_key].append(local_i)

        face_data.append({
            'f_idx': f_idx,
            'center': fc,
            'n_sides': len(fv),
            'edges': edges_of_face,
            'vertices': unwrapped,
            'vertex_ids': list(fv),
        })

    adj = defaultdict(dict)
    for e_key, fl in edge_to_faces.items():
        if len(fl) == 2:
            i, j = fl
            adj[i][j] = e_key
            adj[j][i] = e_key

    return face_data, dict(adj)


# ═══════════════════════════════════════════════════════════════
# 3. CIRCUIT FINDING
# ═══════════════════════════════════════════════════════════════

def find_simple_cycles(adj, n_faces, max_length=8):
    """
    Enumerate all simple cycles on face adjacency graph up to max_length.

    Uses DFS with canonical rotation to avoid duplicates.

    Args:
        adj: face adjacency dict (from get_cell_geometry)
        n_faces: total number of faces
        max_length: maximum cycle length

    Returns:
        list of tuples, each a cycle of face indices (canonically ordered)
    """
    cycles_found = set()

    def dfs(path, start, current, visited, max_len):
        if len(path) > max_len:
            return
        for neighbor in adj.get(current, {}):
            if neighbor == start and len(path) >= 3:
                cycle = tuple(path)
                min_idx = cycle.index(min(cycle))
                rotated = cycle[min_idx:] + cycle[:min_idx]
                if len(rotated) > 1 and rotated[-1] < rotated[1]:
                    rotated = (rotated[0],) + tuple(reversed(rotated[1:]))
                cycles_found.add(rotated)
            elif neighbor not in visited and neighbor > start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(path, start, neighbor, visited, max_len)
                path.pop()
                visited.remove(neighbor)

    for start in range(n_faces):
        visited = {start}
        dfs([start], start, start, visited, max_length)

    return sorted(cycles_found, key=lambda c: len(c))


def equatorial_test(circuit, face_data, adj, cell_center, n_total_faces):
    """
    Heuristic filter: test whether a circuit is equatorial (passes near
    cell center and splits remaining faces into balanced caps). Used for
    candidate selection in find_best_belt; holonomy is computed separately
    by circuit_holonomy.

    Args:
        circuit: tuple of local face indices forming a cycle
        face_data: from get_cell_geometry
        adj: face adjacency
        cell_center: 3D cell center
        n_total_faces: total faces on cell

    Returns:
        dict with:
            'is_equatorial': bool (offset_ratio < 0.3 and balance > 0.3)
            'offset_ratio': float
            'balance': float
        or None if circuit edges not connected
    """
    n = len(circuit)
    points = []
    for k in range(n):
        fi = circuit[k]
        fj = circuit[(k + 1) % n]
        if fj not in adj.get(fi, {}):
            return None
        e_key = adj[fi][fj]
        points.append(face_data[fi]['edges'][e_key])

    points = np.array(points)
    centroid = points.mean(axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    plane_normal = Vt[-1]

    offset = abs(np.dot(cell_center - centroid, plane_normal))
    R_cell = np.mean([
        np.linalg.norm(fd['center'] - cell_center) for fd in face_data
    ])
    offset_ratio = offset / R_cell if R_cell > 0 else 999

    circuit_set = set(circuit)
    remaining = [i for i in range(n_total_faces) if i not in circuit_set]
    cap1, cap2 = [], []
    for fi in remaining:
        proj = np.dot(face_data[fi]['center'] - centroid, plane_normal)
        if proj >= 0:
            cap1.append(fi)
        else:
            cap2.append(fi)

    n1, n2 = len(cap1), len(cap2)
    balance = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 0

    return {
        'is_equatorial': (offset_ratio < 0.3) and (balance > 0.3),
        'offset_ratio': offset_ratio,
        'balance': balance,
    }


def circuit_holonomy(circuit, face_data, adj):
    """
    Compute Gauss-Bonnet holonomy Ω for one cap of a circuit.

    Uses T8b method: BFS cap assignment, vertex deficit angles,
    boundary vertex assignment via Jacobi neighbor voting.

    Note on convention dependence: for Ω≈2π circuits, the boundary
    vertex assignment is unambiguous (Jacobi voting and face-incidence
    counting produce identical results on all tested cell types).
    For non-2π circuits, the numeric Ω value depends on the assignment
    convention. Only Ω≈2π membership is used for physics claims.
    See ST_9/wip/w_6_release_infra/03_all_circuits_A_vs_B.py.

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        adj: face adjacency

    Returns:
        float Ω (holonomy angle) or None if circuit doesn't split cell
    """
    cap1_faces, cap2_faces, belt_faces = assign_caps_bfs(
        face_data, circuit, adj)

    if not cap1_faces or not cap2_faces:
        return None

    # Vertex angle sums — keyed by global vertex ID
    all_verts = {}
    for fi, fd in enumerate(face_data):
        verts = fd['vertices']
        vids = fd['vertex_ids']
        for j, vid in enumerate(vids):
            v = verts[j]
            if vid not in all_verts:
                all_verts[vid] = []
            prev_v = verts[(j - 1) % len(verts)]
            next_v = verts[(j + 1) % len(verts)]
            e1 = prev_v - v
            e2 = next_v - v
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
            all_verts[vid].append(np.arccos(cos_a))

    # Assign vertices to caps — keyed by vertex ID
    vert_to_cap = {}
    belt_vids = set()
    cap1_vids = set()
    cap2_vids = set()

    for fi in cap1_faces:
        for vid in face_data[fi]['vertex_ids']:
            cap1_vids.add(vid)
    for fi in cap2_faces:
        for vid in face_data[fi]['vertex_ids']:
            cap2_vids.add(vid)
    for fi in belt_faces:
        for vid in face_data[fi]['vertex_ids']:
            belt_vids.add(vid)

    for vid in cap1_vids - belt_vids:
        vert_to_cap[vid] = 1
    for vid in cap2_vids - belt_vids:
        vert_to_cap[vid] = 2

    # Jacobi voting for boundary vertices (shared with belt).
    # Collect all new assignments per round, apply at end of round.
    # This makes the result independent of iteration order.
    boundary = belt_vids & (cap1_vids | cap2_vids)
    all_edges_adj = defaultdict(set)
    for fi, fd_item in enumerate(face_data):
        vids = fd_item['vertex_ids']
        for j in range(len(vids)):
            v1 = vids[j]
            v2 = vids[(j + 1) % len(vids)]
            all_edges_adj[v1].add(v2)
            all_edges_adj[v2].add(v1)

    changed = True
    while changed:
        new_assignments = {}
        for vid in boundary:
            if vid in vert_to_cap:
                continue
            neighbors = all_edges_adj.get(vid, set())
            c1 = sum(1 for nb in neighbors if vert_to_cap.get(nb) == 1)
            c2 = sum(1 for nb in neighbors if vert_to_cap.get(nb) == 2)
            if c1 > c2:
                new_assignments[vid] = 1
            elif c2 > c1:
                new_assignments[vid] = 2
        vert_to_cap.update(new_assignments)
        changed = len(new_assignments) > 0

    for vid in boundary:
        if vid not in vert_to_cap:
            vert_to_cap[vid] = 1

    # Vertex deficits
    deficits = {}
    for vid, angles in all_verts.items():
        deficits[vid] = 2 * np.pi - sum(angles)

    cap1_all = {vid for vid, c in vert_to_cap.items() if c == 1}
    omega = sum(deficits.get(vid, 0) for vid in cap1_all)
    return omega


def find_best_belt(face_data, adj, cell_center, hol_threshold=0.3):
    """
    Find the best Ω≈2π equatorial circuit for one cell.

    Searches all simple cycles up to length 8, filters for equatorial
    geometry, selects the one closest to Ω = 2π.

    Args:
        face_data: from get_cell_geometry
        adj: face adjacency
        cell_center: 3D cell center
        hol_threshold: max |Ω| - 2π error (default 0.3)

    Returns:
        tuple (circuit, holonomy) or None if no Ω≈2π belt found
    """
    n_f = len(face_data)
    cycles = find_simple_cycles(adj, n_f, max_length=8)

    best = None
    best_err = 999

    for cyc in cycles:
        eq = equatorial_test(cyc, face_data, adj, cell_center, n_f)
        if not (eq and eq['is_equatorial']):
            continue
        hol = circuit_holonomy(cyc, face_data, adj)
        if hol is None:
            continue
        err = abs(abs(hol) - 2 * np.pi)
        if err < hol_threshold and err < best_err:
            best_err = err
            best = (cyc, hol)

    return best


# ═══════════════════════════════════════════════════════════════
# 4. CAP ASSIGNMENT
# ═══════════════════════════════════════════════════════════════

def assign_caps_bfs(face_data, circuit, adj):
    """
    Assign non-belt faces to cap1 or cap2 using BFS.

    Starting from the first non-belt face, BFS finds all reachable
    faces without crossing the belt → cap1. Remaining non-belt faces
    → cap2. The belt separates the cell into two caps.

    Args:
        face_data: from get_cell_geometry
        circuit: tuple of local face indices forming the belt
        adj: face adjacency

    Returns:
        cap1: set of local face indices
        cap2: set of local face indices
        belt: set of local face indices (= set(circuit))
    """
    circuit_set = set(circuit)
    n_faces = len(face_data)
    cap_faces = [i for i in range(n_faces) if i not in circuit_set]

    if not cap_faces:
        return set(), set(), circuit_set

    visited = set()
    cap1_faces = set()
    queue = deque([cap_faces[0]])
    visited.add(cap_faces[0])
    cap1_faces.add(cap_faces[0])

    while queue:
        f = queue.popleft()
        for nb in adj.get(f, {}):
            if nb not in visited and nb not in circuit_set:
                visited.add(nb)
                cap1_faces.add(nb)
                queue.append(nb)

    cap2_faces = set(cap_faces) - cap1_faces
    return cap1_faces, cap2_faces, circuit_set


