"""Turning a request into a place to put the cube.

The planner asks for one of three things, and each has its own way of becoming a target on the table:

* *a distance from the demonstrations* (a policy trial of a threshold block): a point of the test region whose
  nearest demonstration is that far away, drawn evenly from all such points (`placement_at_distance`);
* *a distance from one demonstration* (a replay trial, where that demonstration's recorded actions are played back):
  a point on the circle of that radius around its start, in a direction drawn evenly from those that stay inside the
  test region (`placement_around`);
* *anywhere in a region* (the paired block): a point drawn evenly from the region (`uniform_placements`).

Every target lies in the test region. Distances are to measured starts of trained demonstrations (see `geometry`).
Randomness comes only from the generator passed in, and only through `Generator.random`.

Standard library and NumPy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from tolerance_radius.geometry import Grid, Rect, largest_nearest_distance, nearest_distances_xy, trained_positions
from tolerance_radius.schema import DemoStart, Point

#: `placement_at_distance` first sorts the test region into tiles of about this size, then draws inside the tiles that
#: can hold a point at the requested distance.
_TILE_MM = 1.0
#: Points drawn at a time, and how many times, before concluding that the points at the requested distance are too few
#: to hit by chance.
_BATCH = 256
_BATCHES = 40


def max_feasible_level(demos: Sequence[DemoStart], grid: Grid) -> float:
    """The largest distance to the nearest demonstration that any point of the test region has, in mm.

    A requested distance beyond this (by more than the match tolerance) cannot be placed.
    """
    return largest_nearest_distance(trained_positions(demos), [grid.test_region])[0]


def placement_at_distance(
    level_mm: float, demos: Sequence[DemoStart], grid: Grid, rng: np.random.Generator, match_mm: float
) -> Point | None:
    """A point of the test region whose nearest demonstration is `level_mm` away, to within `match_mm`.

    The point is drawn evenly (by area) from all the points of the test region that qualify. Returns None when no
    point of the test region qualifies. When the qualifying points are too few to be hit by chance (a requested
    distance at the very end of what the table offers), one of them is constructed instead of drawn.
    """
    _check_level(level_mm)
    if not (math.isfinite(match_mm) and match_mm > 0):
        raise ValueError(f"the match tolerance must be a positive length, got {match_mm!r}")
    positions = trained_positions(demos)
    region = grid.test_region

    # Tiles of equal size covering the test region. A tile can hold a qualifying point only if the distance at its
    # centre is within the tolerance plus half a tile diagonal of the request, because the distance to the nearest
    # demonstration changes by at most 1 mm per mm moved.
    nx = max(1, math.ceil(region.width_mm / _TILE_MM - 1e-9))
    ny = max(1, math.ceil(region.height_mm / _TILE_MM - 1e-9))
    tile_x, tile_y = region.width_mm / nx, region.height_mm / ny
    corner_x, corner_y = np.meshgrid(
        region.x_min_mm + np.arange(nx) * tile_x, region.y_min_mm + np.arange(ny) * tile_y, indexing="ij"
    )
    corner_x, corner_y = corner_x.ravel(), corner_y.ravel()
    at_centre = nearest_distances_xy(corner_x + tile_x / 2.0, corner_y + tile_y / 2.0, positions)
    reach = match_mm + 0.5 * math.hypot(tile_x, tile_y) + 1e-9
    tiles = np.flatnonzero(np.abs(at_centre - level_mm) <= reach)

    # Draw a tile, then a point in it, and keep the first point that qualifies. All tiles have the same area, so the
    # points kept are spread evenly over the qualifying area.
    for _ in range(_BATCHES if len(tiles) else 0):
        draws = rng.random((_BATCH, 3))
        tile = tiles[np.minimum((draws[:, 0] * len(tiles)).astype(np.int64), len(tiles) - 1)]
        xs = np.minimum(corner_x[tile] + draws[:, 1] * tile_x, region.x_max_mm)
        ys = np.minimum(corner_y[tile] + draws[:, 2] * tile_y, region.y_max_mm)
        hits = np.flatnonzero(np.abs(nearest_distances_xy(xs, ys, positions) - level_mm) <= match_mm)
        if len(hits):
            return Point(float(xs[hits[0]]), float(ys[hits[0]]))
    return _constructed_placement(level_mm, positions, region, match_mm)


def placement_around(demo: DemoStart, level_mm: float, grid: Grid, rng: np.random.Generator) -> Point | None:
    """A point `level_mm` from one demonstration's measured start, inside the test region.

    The direction is drawn evenly from the directions in which the point stays in the test region. Returns None when
    there is no such direction. At distance 0 the point is the start itself.
    """
    _check_level(level_mm)
    centre, region = demo.measured, grid.test_region
    if level_mm == 0:
        return centre if region.contains(centre) else None
    arcs = _arcs_inside(centre, level_mm, region)
    total = sum(end - start for start, end in arcs)
    if total <= 0.0:
        return None
    along = rng.random() * total
    angle = arcs[-1][1]
    for start, end in arcs:
        if along < end - start:
            angle = start + along
            break
        along -= end - start
    # Rounding can leave a point on the edge a hair outside: put it back on the edge.
    x = min(max(centre.x_mm + level_mm * math.cos(angle), region.x_min_mm), region.x_max_mm)
    y = min(max(centre.y_mm + level_mm * math.sin(angle), region.y_min_mm), region.y_max_mm)
    return Point(x, y)


def uniform_placements(grid: Grid, n: int, rng: np.random.Generator, region: str = "trained_cells") -> list[Point]:
    """`n` points drawn evenly (by area) from a region of the table.

    A region is a part of the test region (`geometry.REGIONS`): "trained_cells" is the part of the trained cells
    inside the margin, so that a placement is never closer to the grid's outer edge than a test placement may be.
    """
    if n < 0:
        raise ValueError(f"the number of placements cannot be negative, got {n!r}")
    rects = grid.region_rects(region)
    upto = np.cumsum([rect.area_mm2 for rect in rects])
    draws = rng.random((n, 3))
    # A rectangle in proportion to its area, then a point in it.
    which = np.minimum(np.searchsorted(upto, draws[:, 0] * upto[-1], side="right"), len(rects) - 1)
    points = []
    for index, across, up in zip(which.tolist(), draws[:, 1].tolist(), draws[:, 2].tolist(), strict=True):
        rect = rects[index]
        points.append(Point(rect.x_min_mm + across * rect.width_mm, rect.y_min_mm + up * rect.height_mm))
    return points


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _check_level(level_mm: float) -> None:
    if not (math.isfinite(level_mm) and level_mm >= 0):
        raise ValueError(f"a requested distance is a length of 0 or more, got {level_mm!r}")


def _constructed_placement(level_mm: float, positions: np.ndarray, region: Rect, match_mm: float) -> Point | None:
    """A point of the region at the requested distance without drawing, or None when the region has none.

    The distance to the nearest position takes every value between its smallest and its largest over the region, the
    region being in one piece. So the request can be met exactly when it is within the tolerance of that range, and a
    point that meets it lies on the straight line from where the distance is smallest to where it is largest.
    """
    largest, far = largest_nearest_distance(positions, [region])
    inside_x = np.clip(positions[:, 0], region.x_min_mm, region.x_max_mm)
    inside_y = np.clip(positions[:, 1], region.y_min_mm, region.y_max_mm)
    to_region = np.sqrt((positions[:, 0] - inside_x) ** 2 + (positions[:, 1] - inside_y) ** 2)
    closest = int(np.argmin(to_region))
    smallest, near = float(to_region[closest]), Point(float(inside_x[closest]), float(inside_y[closest]))
    if level_mm - match_mm > largest or level_mm + match_mm < smallest:
        return None
    if level_mm >= largest:
        return far
    if level_mm <= smallest:
        return near

    def point_at(fraction: float) -> Point:
        return Point(near.x_mm + fraction * (far.x_mm - near.x_mm), near.y_mm + fraction * (far.y_mm - near.y_mm))

    # Halve the stretch of the line on which the distance passes the request, until it is a point.
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        point = point_at(middle)
        if float(nearest_distances_xy(point.x_mm, point.y_mm, positions)) < level_mm:
            low = middle
        else:
            high = middle
    return point_at(high)


def _arcs_inside(centre: Point, radius_mm: float, region: Rect) -> list[tuple[float, float]]:
    """The parts of a circle that lie in a rectangle, as `(start, end)` angles in radians with `end > start`."""
    crossings: set[float] = set()
    for x in (region.x_min_mm, region.x_max_mm):
        cosine = (x - centre.x_mm) / radius_mm
        if abs(cosine) < 1.0:
            crossings.update((math.acos(cosine), 2.0 * math.pi - math.acos(cosine)))
    for y in (region.y_min_mm, region.y_max_mm):
        sine = (y - centre.y_mm) / radius_mm
        if abs(sine) < 1.0:
            crossings.update((math.asin(sine) % (2.0 * math.pi), (math.pi - math.asin(sine)) % (2.0 * math.pi)))

    def inside(angle: float) -> bool:
        x = centre.x_mm + radius_mm * math.cos(angle)
        y = centre.y_mm + radius_mm * math.sin(angle)
        return region.x_min_mm <= x <= region.x_max_mm and region.y_min_mm <= y <= region.y_max_mm

    if not crossings:  # the circle never meets an edge: it is wholly inside or wholly outside
        return [(0.0, 2.0 * math.pi)] if inside(0.0) else []
    angles = sorted(crossings)
    ends = [*angles[1:], angles[0] + 2.0 * math.pi]
    arcs = zip(angles, ends, strict=True)
    return [(start, end) for start, end in arcs if end > start and inside((start + end) / 2.0)]
