"""
Physics layer constants for DisplacementBloch.
"""

# Zero detection for wave vector magnitude |k|
ZERO_K_THRESHOLD = 1e-12

# Zero detection for eigenvalues and norms
ZERO_EIGENVALUE_THRESHOLD = 1e-10

# Threshold for skipping negligible matrix coefficients
COEFFICIENT_THRESHOLD = 1e-15

# Minimum k magnitude for dispersion analysis
DISPERSION_K_MIN = 1e-3
