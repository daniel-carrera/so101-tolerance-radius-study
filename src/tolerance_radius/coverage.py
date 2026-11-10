"""The "lookup" prediction that the paper tests.

Suppose a policy does nothing more than recall its demonstrations: a placement succeeds or fails according to how far
it is from the nearest demonstration, and nothing else. Then the response curve measured in a threshold block (success
against distance) predicts success anywhere on the table, and the layout of the demonstrations decides how much of the
table is within reach:

* `covered_fraction` is the share of a region that lies within a given radius of a demonstration: the predicted
  success rate of placements drawn evenly from the region, if success were certain inside the radius and impossible
  outside it;
* `expected_success` applies any response curve to the placements themselves.

The paired block compares these predictions with what the two policies do at the same placements. Distances are to
measured starts of trained demonstrations of one set (see `geometry`).

Standard library and NumPy only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tolerance_radius.geometry import Grid, nearest_distances_xy, trained_positions
from tolerance_radius.schema import DemoStart, Point


def covered_fraction(
    demos: Sequence[DemoStart], grid: Grid, radius_mm: float, region: str = "trained_cells", step_mm: float = 0.5
) -> float:
    """The share of a region that is closer than `radius_mm` to a demonstration, from 0 to 1.

    A region is a part of the test region (`geometry.REGIONS`); the default is the part in trained cells, where the
    placements of the paired block are drawn. The share is measured on sample points `step_mm` apart: with the default
    it is good to about a thousandth of the region. A radius of 0 covers nothing.
    """
    if not (math.isfinite(radius_mm) and radius_mm >= 0):
        raise ValueError(f"a radius is a length of 0 or more, got {radius_mm!r}")
    xs, ys, weights = grid.region_lattice(region, step_mm)
    within = nearest_distances_xy(xs, ys, trained_positions(demos)) < radius_mm
    return float(weights[within].sum() / weights.sum())


def expected_success(points: Any, demos: Sequence[DemoStart], curve: Callable[[Any], Any]) -> np.ndarray:
    """The success probability the lookup prediction gives each point: `curve(distance to the nearest demonstration)`.

    `points` is a sequence of `Point` or an `(n, 2)` array of x and y in mm. `curve` maps a distance in mm to a
    probability; it may take an array of distances and return an array, or take one distance at a time. Returns one
    probability per point.
    """
    xy = _as_xy(points)
    distances = nearest_distances_xy(xy[:, 0], xy[:, 1], trained_positions(demos))
    probabilities = _apply(curve, distances)
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):  # also catches a result that is not a number
        raise ValueError("the curve must give probabilities: numbers from 0 to 1")
    return probabilities


def _as_xy(points: Any) -> np.ndarray:
    if not isinstance(points, np.ndarray):
        points = [(p.x_mm, p.y_mm) if isinstance(p, Point) else p for p in points]
    xy = np.asarray(points, dtype=float)
    if xy.size == 0:
        return xy.reshape(0, 2)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"points are Point objects or an (n, 2) array of x and y, got shape {xy.shape}")
    return xy


def _apply(curve: Callable[[Any], Any], distances: np.ndarray) -> np.ndarray:
    """The curve at every distance, whether it works on a whole array or on one number at a time."""
    try:
        values = np.asarray(curve(distances), dtype=float)
    except (TypeError, ValueError):
        values = None
    if values is None or values.shape != distances.shape:
        values = np.array([float(curve(float(distance))) for distance in distances.tolist()], dtype=float)
    return values
