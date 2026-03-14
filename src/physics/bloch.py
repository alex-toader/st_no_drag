"""
Displacement-Based Bloch Formulation
=====================================

Phonon band structure via tensorial spring network on periodic structures.

MODEL:
    Energy = (1/2) Σ_e [ k_L [(u_j - u_i)·ê]² + k_T |u_j - u_i - [(u_j-u_i)·ê]ê|² ]

    The coupling matrix for edge e is: K_e = k_L (ê⊗ê) + k_T (I - ê⊗ê)

    When k_L = k_T: isotropic springs (degenerate T/L branches)
    When k_L ≠ k_T: proper T/L separation with v_L ≠ v_T

DESIGN NOTES:
  - D(k) is Hermitian by construction: each edge contributes K_e with
    conjugate phases to the (i,j) and (j,i) blocks. eigvalsh downstream
    assumes Hermitian input; this is guaranteed, not checked at runtime.
  - ZERO_EIGENVALUE_THRESHOLD (physics) vs EPS_ZERO (spec/constants):
    different contexts. EPS_ZERO is for coordinate snapping (topology).
    ZERO_EIGENVALUE_THRESHOLD is for eigenvalue classification (physics).

Jan 2026
"""

import numpy as np
from typing import Tuple, List

from .constants import (
    ZERO_EIGENVALUE_THRESHOLD,
    COEFFICIENT_THRESHOLD,
)


def compute_edge_geometry(vertices: np.ndarray,
                          edges: List[Tuple[int, int]],
                          L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute edge unit vectors and boundary crossings for all edges.

    This is the unified function for computing edge geometry with periodic
    boundary conditions. Use this instead of duplicating the logic.

    Args:
        vertices: (V, 3) vertex positions
        edges: list of (i, j) tuples
        L: period of the cubic cell

    Returns:
        edge_vectors: (E, 3) unit vectors along edges (unwrapped)
        crossings: (E, 3) boundary crossing vectors
    """
    edge_vectors = []
    crossings = []

    for (i, j) in edges:
        delta = vertices[j] - vertices[i]
        n = np.zeros(3, dtype=int)

        # Unwrap periodic boundary
        for axis in range(3):
            if delta[axis] < -L/2:
                delta[axis] += L
                n[axis] = +1
            elif delta[axis] > L/2:
                delta[axis] -= L
                n[axis] = -1

        length = np.linalg.norm(delta)
        if length > ZERO_EIGENVALUE_THRESHOLD:
            edge_vectors.append(delta / length)
        else:
            edge_vectors.append(np.zeros(3))
        crossings.append(n)

    return np.array(edge_vectors), np.array(crossings)


class DisplacementBloch:
    """
    Displacement-based Bloch formulation for elastic phonon bands.

    Unlike DEC 1-forms (1 DOF per edge), this uses displacement vectors (3 DOF per vertex).
    This is the standard solid-state physics formulation for phonon band structure.

    MODEL: Tensorial spring network
        Energy = (1/2) Σ_e [ k_L [(u_j - u_i)·ê]² + k_T |u_j - u_i - [(u_j-u_i)·ê]ê|² ]

    where:
        ê = unit vector along edge
        k_L = longitudinal stiffness (compression along edge)
        k_T = transverse stiffness (shear perpendicular to edge)

    The coupling matrix for edge e is: K_e = k_L (ê⊗ê) + k_T (I - ê⊗ê)

    When k_L = k_T: isotropic springs (degenerate T/L branches)
    When k_L ≠ k_T: proper T/L separation with v_L ≠ v_T
    """

    def __init__(self, vertices: np.ndarray,
                 edges: List[Tuple[int, int]],
                 L: float,
                 spring_k: float = 1.0,
                 mass: float = 1.0,
                 k_L: float = None,
                 k_T: float = None):
        """
        Initialize displacement-based Bloch system.

        Args:
            vertices: (V, 3) positions
            edges: list of (i, j) tuples
            L: period
            spring_k: spring constant (uniform, used if k_L/k_T not specified)
            mass: vertex mass (uniform)
            k_L: longitudinal spring constant (default: spring_k)
            k_T: transverse spring constant (default: spring_k)

        For proper T/L separation, set k_L ≠ k_T.
        Typical values for elastic medium with bulk modulus K and shear modulus G:
            k_L ∝ K + 4G/3 (longitudinal modulus)
            k_T ∝ G (shear modulus)
        """
        self.vertices = vertices
        self.edges = edges
        self.L = L
        self.spring_k = spring_k
        self.mass = mass

        # Tensorial spring constants
        self.k_L = k_L if k_L is not None else spring_k
        self.k_T = k_T if k_T is not None else spring_k

        self.V = len(vertices)
        self.E = len(edges)

        # Precompute edge vectors and crossings using unified function
        self.edge_vectors, self.crossings = compute_edge_geometry(vertices, edges, L)

    def build_dynamical_matrix(self, k: np.ndarray) -> np.ndarray:
        """
        Build dynamical matrix D(k) for wave vector k using tensorial springs.

        D(k) = (1/m) K(k)

        where K(k) is the Bloch-twisted stiffness matrix.

        The eigenvalue problem is: ω² u = D(k) u

        For each edge with direction ê, the coupling matrix is:
            K_e = k_L (ê⊗ê) + k_T (I - ê⊗ê)

        This gives:
            K_ab = k_L * ê_a * ê_b + k_T * (δ_ab - ê_a * ê_b)
                 = k_T * δ_ab + (k_L - k_T) * ê_a * ê_b

        Args:
            k: (3,) wave vector

        Returns:
            D: (3V, 3V) complex Hermitian matrix

        Physics:
            - When k_L = k_T: isotropic, all branches degenerate
            - When k_L > k_T: longitudinal branch faster (v_L > v_T)
            - When k_L < k_T: transverse branch faster (v_T > v_L)
        """
        D = np.zeros((3*self.V, 3*self.V), dtype=complex)

        for e_idx, (i, j) in enumerate(self.edges):
            e_hat = self.edge_vectors[e_idx]
            n = self.crossings[e_idx]

            # Phase factor for edge crossing boundary
            phase = np.exp(1j * np.dot(k, n * self.L))

            # Tensorial coupling: K_ab = k_T δ_ab + (k_L - k_T) ê_a ê_b
            for a in range(3):
                for b in range(3):
                    # K_ab = k_T * δ_ab + (k_L - k_T) * ê_a * ê_b
                    coeff = (self.k_L - self.k_T) * e_hat[a] * e_hat[b]
                    if a == b:
                        coeff += self.k_T

                    if abs(coeff) < COEFFICIENT_THRESHOLD:
                        continue

                    # Diagonal blocks (self-interaction)
                    D[3*i + a, 3*i + b] += coeff
                    D[3*j + a, 3*j + b] += coeff

                    # Off-diagonal blocks (interaction i-j)
                    D[3*i + a, 3*j + b] -= coeff * phase
                    D[3*j + a, 3*i + b] -= coeff * np.conj(phase)

        # Include mass (D = K/m)
        D /= self.mass

        return D
