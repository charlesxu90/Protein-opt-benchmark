"""
Multi-Objective Optimization Metrics

Hypervolume indicator and Pareto-front coverage for multi-property protein
optimization (Task 3 of the refined benchmark plan).

Conventions
-----------
- All objectives are treated as **maximize**. Pass negated values to minimize.
- Points are 2D arrays of shape (n_points, n_objectives).
- Reference points must be dominated by every point in the set
  (i.e., reference[i] <= min(point[i]) for maximization).

Backends
--------
- Hypervolume: prefers `pymoo.indicators.hv.HV` when available; otherwise
  falls back to a simple O(n^2) sweep that is exact in 2D.
- Pareto front extraction: pure NumPy, O(n^2). Fine for the tens-of-thousands
  scale we use in benchmarks.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


# =============================================================================
# Pareto front extraction
# =============================================================================

def pareto_front_mask(points: ArrayLike) -> np.ndarray:
    """Return a boolean mask of the non-dominated (Pareto-optimal) points.

    A point p is non-dominated if no other point q strictly dominates it,
    i.e., q[i] >= p[i] for all i and q[j] > p[j] for at least one j.

    Args:
        points: array of shape (n, m), maximize.

    Returns:
        Boolean mask of shape (n,) marking Pareto-optimal points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points must be 2D, got shape {pts.shape}")
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # Any q that dominates p[i]?
        dominates = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if dominates.any():
            is_pareto[i] = False
    return is_pareto


def pareto_front(points: ArrayLike) -> np.ndarray:
    """Return the Pareto-optimal subset (maximize convention)."""
    pts = np.asarray(points, dtype=float)
    return pts[pareto_front_mask(pts)]


# =============================================================================
# Hypervolume
# =============================================================================

def _hypervolume_2d_max(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2D hypervolume for a maximize problem.

    Standard sweep: sort Pareto front by first objective descending, then
    accumulate the area of the rectangles above ref.
    """
    front = points[pareto_front_mask(points)]
    if front.size == 0:
        return 0.0
    # Sort by first objective descending
    order = np.argsort(-front[:, 0])
    front = front[order]
    # Sweep
    hv = 0.0
    prev_y = ref[1]
    for x, y in front:
        if x <= ref[0] or y <= ref[1]:
            continue
        hv += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return float(hv)


def hypervolume(points: ArrayLike, ref: ArrayLike) -> float:
    """Hypervolume indicator for a maximize problem.

    Uses pymoo when available (handles arbitrary dimensions), otherwise
    falls back to an exact 2D sweep.

    Args:
        points: array of shape (n, m). Objective values to evaluate.
        ref: reference point of shape (m,). Must be dominated by all points
            (i.e., ref[i] <= min(points[:, i]) for maximize).

    Returns:
        Hypervolume of the dominated region above the reference point.
    """
    pts = np.asarray(points, dtype=float)
    r = np.asarray(ref, dtype=float)
    if pts.ndim != 2 or r.ndim != 1 or pts.shape[1] != r.shape[0]:
        raise ValueError(
            f"Shape mismatch: points {pts.shape}, ref {r.shape}"
        )
    if pts.shape[0] == 0:
        return 0.0

    # Try pymoo (handles m >= 2 with WFG algorithm)
    try:
        from pymoo.indicators.hv import HV
        # pymoo uses minimize; negate
        hv_calc = HV(ref_point=-r)
        return float(hv_calc(-pts))
    except ImportError:
        pass

    if pts.shape[1] == 2:
        return _hypervolume_2d_max(pts, r)
    raise NotImplementedError(
        f"Hypervolume for m={pts.shape[1]} objectives requires pymoo. "
        "Install with `pip install pymoo`."
    )


# =============================================================================
# Pareto front coverage
# =============================================================================

def pareto_front_coverage(
    discovered: ArrayLike,
    reference_front: ArrayLike,
    tolerance: float = 1e-6,
) -> float:
    """Fraction of the reference Pareto front that the discovered set matches.

    A reference-front point r is "covered" if some discovered point d weakly
    dominates it: d[i] >= r[i] - tolerance for all i.

    Args:
        discovered: array (n_d, m). The variants the method has queried.
        reference_front: array (n_r, m). The true Pareto front (typically
            extracted from the full DMS landscape).
        tolerance: numerical slack for "matches".

    Returns:
        Coverage in [0, 1].
    """
    d = np.asarray(discovered, dtype=float)
    r = np.asarray(reference_front, dtype=float)
    if r.shape[0] == 0:
        return 1.0
    if d.shape[0] == 0:
        return 0.0
    # For each ref point, check if any discovered point weakly dominates it
    covered = 0
    for ref_pt in r:
        if np.any(np.all(d >= ref_pt - tolerance, axis=1)):
            covered += 1
    return covered / r.shape[0]


# =============================================================================
# Convenience: extract reference Pareto front from a DMS landscape
# =============================================================================

def reference_front_from_landscape(
    objectives: ArrayLike,
) -> np.ndarray:
    """Extract the reference Pareto front from a full multi-objective landscape.

    Args:
        objectives: array (n, m) of objective values for every variant in the
            landscape (maximize).

    Returns:
        Pareto-optimal subset, shape (n_pareto, m).
    """
    return pareto_front(objectives)


def auto_reference_point(
    points: ArrayLike,
    margin: float = 0.05,
) -> np.ndarray:
    """Pick a reference point dominated by every point (maximize).

    Returns min(points, axis=0) - margin * range(points, axis=0).
    """
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0)
    rng = pts.max(axis=0) - lo
    return lo - margin * np.where(rng > 0, rng, 1.0)


__all__ = [
    "pareto_front_mask",
    "pareto_front",
    "hypervolume",
    "pareto_front_coverage",
    "reference_front_from_landscape",
    "auto_reference_point",
]
