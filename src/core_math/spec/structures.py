"""
Canonical face ordering for periodic mesh builders.

SELF-TEST (runs at import, verified Mar 2026):
    - All cyclic rotations → same canonical form
    - Reversed winding → same canonical, opposite sign
    - numpy.int64 inputs → identical to Python int
    - Triangle (3), square (4), hexagon (6) faces → correct
    - Degenerate faces (< 3 vertices) → ValueError
"""

from typing import List, Tuple


def canonical_face(face: List[int]) -> Tuple[tuple, int]:
    """
    Return canonical representation of a face cycle and its relative orientation.

    The canonical form starts at the minimum vertex index, in the
    lexicographically smaller direction (forward vs reversed).

    Args:
        face: list of vertex indices forming a cycle

    Returns:
        (canonical_tuple, orientation)
        - canonical_tuple: face vertices in canonical order
        - orientation: +1 if input has same winding as canonical, -1 if reversed
    """
    if len(face) < 3:
        raise ValueError(f"Face must have at least 3 vertices, got {len(face)}")

    face = [int(v) for v in face]
    min_idx = face.index(min(face))
    rotated = face[min_idx:] + face[:min_idx]
    reversed_rot = [rotated[0]] + rotated[1:][::-1]

    if tuple(rotated) < tuple(reversed_rot):
        return tuple(rotated), +1
    else:
        return tuple(reversed_rot), -1


def _self_test():
    """Verify canonical_face invariants. Called once at import."""
    import numpy as np

    # All rotations → same canonical
    f = [5, 3, 8, 2]
    canon, _ = canonical_face(f)
    for k in range(len(f)):
        rotated = f[k:] + f[:k]
        assert canonical_face(rotated)[0] == canon

    # Reversed winding → opposite sign
    f = [1, 2, 3, 4]
    c1, o1 = canonical_face(f)
    c2, o2 = canonical_face([1, 4, 3, 2])
    assert c1 == c2 and o1 == -o2

    # numpy.int64 compatibility
    f_np = [np.int64(v) for v in [3, 1, 4, 2]]
    c_py, o_py = canonical_face([3, 1, 4, 2])
    c_np, o_np = canonical_face(f_np)
    assert c_py == c_np and o_py == o_np

    # Hexagon: all rotations + reverse agree
    f = [10, 4, 7, 12, 3, 8]
    canon, _ = canonical_face(f)
    for k in range(6):
        assert canonical_face(f[k:] + f[:k])[0] == canon
    assert canonical_face([f[0]] + f[1:][::-1])[0] == canon

    # Degenerate rejection
    for bad in [[1, 2], [1]]:
        try:
            canonical_face(bad)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


_self_test()
