"""The grid on the table, its regions, and distances from a point to the demonstrations.

The table frame is in millimetres with its origin at a corner of the grid: x runs along the columns (A, B, ...), y along
the rows (1, 2, ...). Cell "B1" is column B, row 1.

Words used here
---------------
* A *trained cell* is a cell in which demonstrations start; a *held-out cell* is one in which none does.
* The *test region* is the grid shrunk by a margin on every side: no test placement goes outside it.
* A *region* (`REGIONS`) is a part of the test region: the part in trained cells, the part in held-out cells, or all of
  it. Placements are drawn from regions, so a region never reaches into the margin.
* A distance to "the demonstrations" is to the measured start of the nearest demonstration that was trained on, in one
  demonstration set. The functions here refuse a list that mixes the two sets and skip episodes marked `trained=False`,
  so that a distance cannot quietly be taken to the wrong thing.

Standard library and NumPy only.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tolerance_radius.schema import DATASETS, DemoStart, Point

#: The parts of the test region that placements are drawn from and coverage is computed over.
REGIONS = ("trained_cells", "held_out_cells", "test_region")

_LABEL = re.compile(r"([A-Z])([1-9][0-9]*)")


class GeometryError(ValueError):
    """A grid, a label, a region or a set of demonstrations is not something the geometry can work with."""


@dataclass(frozen=True, slots=True)
class Rect:
    """A rectangle on the table with sides along the axes, in millimetres. Its edges belong to it."""

    x_min_mm: float
    y_min_mm: float
    x_max_mm: float
    y_max_mm: float

    def __post_init__(self) -> None:
        if not (self.x_min_mm < self.x_max_mm and self.y_min_mm < self.y_max_mm):
            raise GeometryError(f"a rectangle has a positive width and height, got {self!r}")

    @property
    def width_mm(self) -> float:
        """Its extent along x."""
        return self.x_max_mm - self.x_min_mm

    @property
    def height_mm(self) -> float:
        """Its extent along y."""
        return self.y_max_mm - self.y_min_mm

    @property
    def area_mm2(self) -> float:
        """Its area, in square millimetres."""
        return self.width_mm * self.height_mm

    def contains(self, point: Point) -> bool:
        """True for a point inside the rectangle or on its edge."""
        return self.x_min_mm <= point.x_mm <= self.x_max_mm and self.y_min_mm <= point.y_mm <= self.y_max_mm


@dataclass(frozen=True, slots=True)
class Grid:
    """The grid of cells on the table: which cells exist, which are trained, and where test placements may go."""

    cols: int
    rows: int
    cell_mm: float
    #: Cells in which no demonstration starts.
    held_out_cells: tuple[str, ...]
    cube_edge_mm: float
    #: Test placements keep the cube's centre at least this far inside the grid's outer edge.
    test_region_margin_mm: float

    def __post_init__(self) -> None:
        for name in ("cols", "rows"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise GeometryError(f"{name} must be a whole number of at least 1, got {value!r}")
        if self.cols > 26:
            raise GeometryError("columns are named A to Z: at most 26")
        for name in ("cell_mm", "cube_edge_mm"):
            value = getattr(self, name)
            if not _is_number(value) or value <= 0:
                raise GeometryError(f"{name} must be a positive length, got {value!r}")
        margin = self.test_region_margin_mm
        if not _is_number(margin) or margin < 0 or 2 * margin >= min(self.width_mm, self.height_mm):
            raise GeometryError(f"test_region_margin_mm must leave a test region inside the grid, got {margin!r}")
        held_out = tuple(self.held_out_cells)
        for label in held_out:
            self._col_row(label)
        if len(set(held_out)) != len(held_out):
            raise GeometryError(f"a held-out cell is listed twice: {held_out}")
        if len(held_out) == self.cols * self.rows:
            raise GeometryError("every cell is held out: there is no trained cell")
        # Stored in the grid's own order, so that two grids with the same cells are equal.
        object.__setattr__(self, "held_out_cells", tuple(label for label in self.labels if label in held_out))

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any]) -> Grid:
        """The grid described by the `[table]` section of the protocol."""
        table = protocol.get("table")
        if not isinstance(table, Mapping):
            raise GeometryError("the protocol has no [table] section")
        try:
            return cls(
                cols=table["cols"],
                rows=table["rows"],
                cell_mm=float(table["cell_mm"]),
                held_out_cells=tuple(table["held_out_cells"]),
                cube_edge_mm=float(table["cube_edge_mm"]),
                test_region_margin_mm=float(table["test_region_margin_mm"]),
            )
        except KeyError as error:
            raise GeometryError(f"[table] is missing {error.args[0]!r}") from error

    # ------------------------------------------------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------------------------------------------------

    @property
    def width_mm(self) -> float:
        """The extent of the grid along x: its columns."""
        return self.cols * self.cell_mm

    @property
    def height_mm(self) -> float:
        """The extent of the grid along y: its rows."""
        return self.rows * self.cell_mm

    @property
    def labels(self) -> tuple[str, ...]:
        """Every cell, column by column: A1, A2, A3, B1, ..."""
        return tuple(_label(col, row) for col in range(self.cols) for row in range(self.rows))

    @property
    def trained_cells(self) -> tuple[str, ...]:
        """The cells in which demonstrations start, column by column."""
        return tuple(label for label in self.labels if label not in self.held_out_cells)

    def is_trained(self, label: str) -> bool:
        """True for a cell in which demonstrations start."""
        self._col_row(label)
        return label not in self.held_out_cells

    def cell_rect(self, label: str) -> Rect:
        """The square a cell covers."""
        col, row = self._col_row(label)
        return Rect(col * self.cell_mm, row * self.cell_mm, (col + 1) * self.cell_mm, (row + 1) * self.cell_mm)

    def cell_centre(self, label: str) -> Point:
        """The centre of a cell."""
        col, row = self._col_row(label)
        return Point((col + 0.5) * self.cell_mm, (row + 0.5) * self.cell_mm)

    def cell_of(self, point: Point) -> str | None:
        """The cell a point is in, or None when it is off the grid.

        A point on the border between two cells belongs to the one that starts there (the next column or row); the
        grid's far edges belong to the last column and row.
        """
        if not self.outline.contains(point):
            return None
        col = min(int(point.x_mm // self.cell_mm), self.cols - 1)
        row = min(int(point.y_mm // self.cell_mm), self.rows - 1)
        return _label(col, row)

    # ------------------------------------------------------------------------------------------------------------
    # Regions
    # ------------------------------------------------------------------------------------------------------------

    @property
    def outline(self) -> Rect:
        """The whole grid."""
        return Rect(0.0, 0.0, self.width_mm, self.height_mm)

    @property
    def test_region(self) -> Rect:
        """The grid shrunk by the margin: where test placements may go."""
        margin = self.test_region_margin_mm
        return Rect(margin, margin, self.width_mm - margin, self.height_mm - margin)

    def in_test_region(self, point: Point) -> bool:
        """True for a point in the test region or on its edge."""
        return self.test_region.contains(point)

    def in_trained_cells(self, point: Point) -> bool:
        """True for a point in a trained cell or on the border of one (the margin included)."""
        return any(self.cell_rect(label).contains(point) for label in self.trained_cells)

    def trained_cells_mask(self, xs: Any, ys: Any) -> np.ndarray:
        """`in_trained_cells` for arrays of x and y: an array of booleans of their common shape."""
        xs, ys = np.broadcast_arrays(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        mask = np.zeros(xs.shape, dtype=bool)
        for label in self.trained_cells:
            rect = self.cell_rect(label)
            mask |= (xs >= rect.x_min_mm) & (xs <= rect.x_max_mm) & (ys >= rect.y_min_mm) & (ys <= rect.y_max_mm)
        return mask

    def region_rects(self, region: str) -> tuple[Rect, ...]:
        """A region as rectangles that do not overlap: its cells, cut off at the edge of the test region."""
        _check_region(region)
        if region == "test_region":
            return (self.test_region,)
        cells = self.trained_cells if region == "trained_cells" else self.held_out_cells
        clipped = (_intersection(self.cell_rect(label), self.test_region) for label in cells)
        return tuple(rect for rect in clipped if rect is not None)

    def region_area_mm2(self, region: str) -> float:
        """The area of a region."""
        return sum(rect.area_mm2 for rect in self.region_rects(region))

    def in_region(self, point: Point, region: str) -> bool:
        """True for a point of the region. A border between a trained and a held-out cell counts as trained."""
        _check_region(region)
        if not self.in_test_region(point):
            return False
        if region == "test_region":
            return True
        return self.in_trained_cells(point) == (region == "trained_cells")

    def region_mask(self, xs: Any, ys: Any, region: str) -> np.ndarray:
        """`in_region` for arrays of x and y: an array of booleans of their common shape."""
        _check_region(region)
        xs, ys = np.broadcast_arrays(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        box = self.test_region
        inside = (xs >= box.x_min_mm) & (xs <= box.x_max_mm) & (ys >= box.y_min_mm) & (ys <= box.y_max_mm)
        if region == "test_region":
            return inside
        trained = self.trained_cells_mask(xs, ys)
        return inside & (trained if region == "trained_cells" else ~trained)

    def region_lattice(self, region: str, step_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample points that cover a region evenly, for sums over it: `(xs, ys, weights)`.

        Each rectangle of the region is cut into equal tiles no larger than `step_mm` on a side. A sample point is the
        centre of a tile and its weight is the tile's area in square millimetres, so the weights add up to the
        region's area.
        """
        if not _is_number(step_mm) or step_mm <= 0:
            raise GeometryError(f"the lattice step must be a positive length, got {step_mm!r}")
        xs, ys, weights = [], [], []
        for rect in self.region_rects(region):
            nx = max(1, math.ceil(rect.width_mm / step_mm - 1e-9))
            ny = max(1, math.ceil(rect.height_mm / step_mm - 1e-9))
            tile_x, tile_y = rect.width_mm / nx, rect.height_mm / ny
            centres_x = rect.x_min_mm + (np.arange(nx) + 0.5) * tile_x
            centres_y = rect.y_min_mm + (np.arange(ny) + 0.5) * tile_y
            grid_x, grid_y = np.meshgrid(centres_x, centres_y, indexing="ij")
            xs.append(grid_x.ravel())
            ys.append(grid_y.ravel())
            weights.append(np.full(nx * ny, tile_x * tile_y))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)

    def _col_row(self, label: str) -> tuple[int, int]:
        match = _LABEL.fullmatch(label) if isinstance(label, str) else None
        if match is not None:
            col, row = ord(match.group(1)) - ord("A"), int(match.group(2)) - 1
            if col < self.cols and row < self.rows:
                return col, row
        raise GeometryError(f"{label!r} is not a cell of this {self.cols} x {self.rows} grid")


# --------------------------------------------------------------------------------------------------------------------
# Distances to the demonstrations
# --------------------------------------------------------------------------------------------------------------------


def trained_demos(demo_starts: Iterable[DemoStart], dataset: str) -> list[DemoStart]:
    """The demonstrations of one set that were trained on, in episode order."""
    if dataset not in DATASETS:
        raise GeometryError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    demos = sorted((d for d in demo_starts if d.dataset == dataset and d.trained), key=lambda d: d.episode_index)
    for first, second in itertools.pairwise(demos):
        if first.episode_index == second.episode_index:
            raise GeometryError(f"episode {first.episode_index} of the {dataset} set appears twice")
    return demos


def trained_positions(demos: Sequence[DemoStart]) -> np.ndarray:
    """The positions distances are taken to: the measured starts of the trained episodes, as an `(n, 2)` array.

    A list that mixes the two demonstration sets, or holds no trained episode, is refused.
    """
    return _trained_positions(demos)[1]


def nearest(point: Point, demos: Sequence[DemoStart]) -> tuple[int, float]:
    """The demonstration nearest to a point: `(its position in demos, the distance in mm)`.

    The distance is to the measured start. Episodes left out of training are skipped; between two equally near
    demonstrations the earlier one in the list is returned.
    """
    indices, positions = _trained_positions(demos)
    squared = (positions[:, 0] - point.x_mm) ** 2 + (positions[:, 1] - point.y_mm) ** 2
    best = int(np.argmin(squared))
    return indices[best], math.sqrt(float(squared[best]))


def nearest_distance_field(xs: Any, ys: Any, demos: Sequence[DemoStart]) -> np.ndarray:
    """The distance from each `(x, y)` to its nearest demonstration, in mm, for arrays of x and y.

    The result has the common shape of `xs` and `ys`. The same rules as `nearest`: measured starts, trained episodes
    only, one demonstration set.
    """
    return nearest_distances_xy(xs, ys, _trained_positions(demos)[1])


def nearest_distances_xy(xs: Any, ys: Any, positions_xy: Any) -> np.ndarray:
    """The distance from each `(x, y)` to the nearest of some positions, given as an `(n, 2)` array of x and y in mm."""
    return np.sqrt(_nearest_squared(xs, ys, _as_positions(positions_xy)))


def in_convex_hull(point: Point, demos: Sequence[DemoStart], tolerance_mm: float = 1e-9) -> bool:
    """True when a point lies among the demonstrations: inside the convex hull of their starts, or on its edge.

    A placement inside the hull asks a policy to interpolate between demonstrations; one outside asks it to
    extrapolate. The same rules as `nearest`: measured starts, trained episodes only, one demonstration set.
    """
    hull = _convex_hull([(float(x), float(y)) for x, y in _trained_positions(demos)[1]])
    p = (point.x_mm, point.y_mm)
    if len(hull) == 1:
        return math.dist(p, hull[0]) <= tolerance_mm
    if len(hull) == 2:
        return _distance_to_segment(p, hull[0], hull[1]) <= tolerance_mm
    # The hull runs counter-clockwise, so the inside is on the left of every edge.
    edges = zip(hull, hull[1:] + hull[:1], strict=True)
    return all(_cross(start, end, p) / math.dist(start, end) >= -tolerance_mm for start, end in edges)


def largest_nearest_distance(positions_xy: Any, rects: Sequence[Rect]) -> tuple[float, Point]:
    """The largest distance from any point of some rectangles to its nearest position, and a point where it occurs.

    This is the largest gap that the positions leave in the rectangles. It is exact, not sampled: the farthest point
    is always a corner of a rectangle, a point of an edge at equal distance from two positions, or a point at equal
    distance from three, so every such point is tried. Meant for the tens of positions of a demonstration set (the
    work grows with the cube of their number).

    When several points are equally far (the corners of cells around a regular layout, say), the one nearest the
    middle of the rectangles is returned, then the one with the lowest x, then the lowest y: the answer does not
    depend on the order of the search or on the last digit of a rounding.
    """
    positions = np.unique(_as_positions(positions_xy), axis=0)
    if len(rects) == 0:
        raise GeometryError("there is no rectangle to measure the gap in")
    centres = _circumcentres(positions)
    normals, offsets = _bisectors(positions)
    candidates = np.concatenate([_gap_candidates(rect, centres, normals, offsets) for rect in rects])
    distances = np.sqrt(_nearest_squared(candidates[:, 0], candidates[:, 1], positions))
    largest = float(distances.max())
    middle_x = (min(rect.x_min_mm for rect in rects) + max(rect.x_max_mm for rect in rects)) / 2.0
    middle_y = (min(rect.y_min_mm for rect in rects) + max(rect.y_max_mm for rect in rects)) / 2.0

    def preference(point: list[float]) -> tuple[float, float, float]:
        to_middle = math.hypot(point[0] - middle_x, point[1] - middle_y)
        return round(to_middle, 6), round(point[0], 6), round(point[1], 6)

    x, y = min(candidates[distances >= largest - 1e-9].tolist(), key=preference)
    return largest, Point(x, y)


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _label(col: int, row: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def _check_region(region: str) -> None:
    if region not in REGIONS:
        raise GeometryError(f"region must be one of {REGIONS}, got {region!r}")


def _intersection(first: Rect, second: Rect) -> Rect | None:
    x_min, x_max = max(first.x_min_mm, second.x_min_mm), min(first.x_max_mm, second.x_max_mm)
    y_min, y_max = max(first.y_min_mm, second.y_min_mm), min(first.y_max_mm, second.y_max_mm)
    return Rect(x_min, y_min, x_max, y_max) if x_min < x_max and y_min < y_max else None


def _trained_positions(demos: Sequence[DemoStart]) -> tuple[list[int], np.ndarray]:
    """Where the trained demonstrations of a list are in it, and their measured starts as an `(n, 2)` array."""
    datasets = {demo.dataset for demo in demos}
    if len(datasets) > 1:
        raise GeometryError(f"distances are taken within one demonstration set, got {sorted(datasets)}")
    indices = [index for index, demo in enumerate(demos) if demo.trained]
    if not indices:
        raise GeometryError("there is no trained demonstration to measure a distance to")
    positions = np.array([[demos[i].measured.x_mm, demos[i].measured.y_mm] for i in indices], dtype=float)
    return indices, positions


def _as_positions(positions_xy: Any) -> np.ndarray:
    positions = np.asarray(positions_xy, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise GeometryError(f"positions are an (n, 2) array with at least one row, got shape {positions.shape}")
    if not np.all(np.isfinite(positions)):
        raise GeometryError("positions must be finite numbers")
    return positions


def _nearest_squared(xs: Any, ys: Any, positions: np.ndarray) -> np.ndarray:
    """Squared distance from each `(x, y)` to the nearest position. One pass per position keeps the memory small."""
    xs, ys = np.broadcast_arrays(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    best = np.full(xs.shape, np.inf)
    for x, y in positions.tolist():
        dx, dy = xs - x, ys - y
        np.minimum(best, dx * dx + dy * dy, out=best)
    return best


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The corners of the convex hull, counter-clockwise (Andrew's monotone chain). Points in a line give two."""
    points = sorted(set(points))
    if len(points) <= 2:
        return points
    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _distance_to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    along = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)
    along = min(max(along, 0.0), 1.0)
    return math.dist(p, (a[0] + along * ab[0], a[1] + along * ab[1]))


def _circumcentres(positions: np.ndarray) -> np.ndarray:
    """The point at equal distance from each three positions that are not in a line, as an `(m, 2)` array."""
    triples = np.array(list(itertools.combinations(range(len(positions)), 3)), dtype=int).reshape(-1, 3)
    if len(triples) == 0:
        return np.zeros((0, 2))
    a = positions[triples[:, 0]]
    b = positions[triples[:, 1]] - a
    c = positions[triples[:, 2]] - a
    twice_area = 2.0 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    keep = np.abs(twice_area) > 1e-12
    a, b, c, twice_area = a[keep], b[keep], c[keep], twice_area[keep]
    b_squared = b[:, 0] ** 2 + b[:, 1] ** 2
    c_squared = c[:, 0] ** 2 + c[:, 1] ** 2
    x = a[:, 0] + (c[:, 1] * b_squared - b[:, 1] * c_squared) / twice_area
    y = a[:, 1] + (b[:, 0] * c_squared - c[:, 0] * b_squared) / twice_area
    centres = np.stack([x, y], axis=1)
    return centres[np.all(np.isfinite(centres), axis=1)]


def _bisectors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each two positions, the line of points at equal distance from both, as `normal . (x, y) = offset`."""
    pairs = np.array(list(itertools.combinations(range(len(positions)), 2)), dtype=int).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    p, q = positions[pairs[:, 0]], positions[pairs[:, 1]]
    return q - p, ((q**2).sum(axis=1) - (p**2).sum(axis=1)) / 2.0


def _gap_candidates(rect: Rect, centres: np.ndarray, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Every point of a rectangle that can be the farthest from the positions (see `largest_nearest_distance`)."""
    x_min, x_max, y_min, y_max = rect.x_min_mm, rect.x_max_mm, rect.y_min_mm, rect.y_max_mm
    candidates = [np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])]
    inside = (centres[:, 0] >= x_min) & (centres[:, 0] <= x_max) & (centres[:, 1] >= y_min) & (centres[:, 1] <= y_max)
    candidates.append(centres[inside])
    crosses_x = normals[:, 1] != 0.0  # the bisector is not parallel to an edge x = constant
    crosses_y = normals[:, 0] != 0.0
    for x in (x_min, x_max):
        y = (offsets[crosses_x] - normals[crosses_x, 0] * x) / normals[crosses_x, 1]
        y = y[(y >= y_min) & (y <= y_max)]
        candidates.append(np.stack([np.full(y.shape, x), y], axis=1))
    for y in (y_min, y_max):
        x = (offsets[crosses_y] - normals[crosses_y, 1] * y) / normals[crosses_y, 0]
        x = x[(x >= x_min) & (x <= x_max)]
        candidates.append(np.stack([x, np.full(x.shape, y)], axis=1))
    return np.concatenate(candidates)
