"""
Incidence matrices d₀ and d₁ for DEC on graphs.

d₀: E × V  "gradient" - oriented edge-vertex incidence
d₁: F × E  "curl" - oriented face-edge incidence
"""

import numpy as np
from typing import List, Tuple


def build_d0(vertices: np.ndarray,
             edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build gradient operator d₀: C⁰ → C¹.

    d₀[e, v] = -1 if v is the source of edge e
    d₀[e, v] = +1 if v is the target of edge e

    Convention: for edge (i, j) with i < j, i is source, j is target.
    """
    V = len(vertices)
    E = len(edges)
    d0 = np.zeros((E, V), dtype=int)

    for e_idx, (i, j) in enumerate(edges):
        d0[e_idx, i] = -1
        d0[e_idx, j] = +1

    return d0


def build_d1(vertices: np.ndarray,
             edges: List[Tuple[int, int]],
             faces: List[List[int]]) -> np.ndarray:
    """
    Build curl operator d₁: C¹ → C².

    d₁[f, e] = ±1 according to face boundary orientation.
    """
    V = len(vertices)
    E = len(edges)
    F = len(faces)

    edge_dict = {}
    for e_idx, (i, j) in enumerate(edges):
        edge_dict[(i, j)] = (e_idx, +1)
        edge_dict[(j, i)] = (e_idx, -1)

    d1 = np.zeros((F, E), dtype=int)

    for f_idx, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            v1 = face[k]
            v2 = face[(k + 1) % n]
            if (v1, v2) not in edge_dict:
                raise ValueError(f"Face {f_idx} segment ({v1},{v2}) not in edge list")
            e_idx, sign = edge_dict[(v1, v2)]
            if d1[f_idx, e_idx] != 0:
                raise ValueError(f"Face {f_idx} uses edge {e_idx} twice")
            d1[f_idx, e_idx] += sign

    return d1
