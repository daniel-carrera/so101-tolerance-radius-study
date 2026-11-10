"""Where each demonstration starts: two layouts of the same budget, and how evenly each covers the trained cells.

Both demonstration sets hold the same number of episodes in each trained cell. They differ in one design choice:

* **clustered**: every episode of a cell starts at the cell's centre (the "several episodes per location" recipe);
* **spread**: every episode starts at a position of its own, and the positions cover the trained cells evenly.

A layout is the list of planned episodes in recording order. Recording runs in rounds (each trained cell once, in a
shuffled order) grouped into sittings that alternate the two sets (`[datasets].sittings` in the protocol). An episode's
index is its position in its own set's recording order.

How the spread positions are chosen
-----------------------------------
Independent random positions leave clumps and holes, so the positions are placed in three steps, all on a lattice of
whole millimetres so that the same seed gives the same layout on any machine:

1. *Best candidate*: five passes over the trained cells in a shuffled order; each time, of a number of random
   candidates in the cell, keep the one farthest from every position placed so far.
2. *Relaxation*: move each position to the middle of the part of the trained cells that is nearest to it, and repeat.
   This evens out the spacing.
3. *Closing gaps*: find the point of the trained cells farthest from every position (the largest gap) and move a
   neighbouring position one millimetre if that makes the largest gap smaller; repeat until no such move is left.

Throughout, a position stays in its own cell, out of the margin at the grid's outer edge, at least the placement
tolerance away from the borders of its cell (so that a cube accepted within the tolerance is still measured in its own
cell), and no closer to another position than the minimum spacing.

Standard library and NumPy only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius import config
from tolerance_radius.config import ConfigError
from tolerance_radius.geometry import Grid, largest_nearest_distance
from tolerance_radius.schema import DATASETS, SCHEMA_VERSION, DemoStart, Point
from tolerance_radius.seeds import derived_rng, pick_index, shuffled

#: Best-candidate sampling: how many random candidates are compared for each position.
_CANDIDATES = 30
#: The relaxation stops when no position moves, or after this many passes.
_RELAXATION_PASSES = 40
#: Closing gaps stops when no move helps, or after this many moves.
_GAP_MOVES = 400
#: Closing gaps: how many of the positions nearest to the gap may be moved, and how many equally large gaps are tried.
_NEIGHBOURS = 4
_GAPS_TRIED = 8
#: The eight moves of one millimetre.
_UNIT_MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


class LayoutError(ValueError):
    """A layout cannot be made, written or read as asked."""


@dataclass(frozen=True, slots=True)
class PlannedEpisode:
    """One demonstration episode as planned: which set it belongs to, when it is recorded and where the cube starts."""

    dataset: str
    #: Position of the episode in its own set's recording order, from 0.
    episode_index: int
    #: The round it is recorded in, from 1. A round visits every trained cell once.
    round: int
    #: The sitting it is recorded in, from 1, in the order of `[datasets].sittings`.
    sitting: int
    cell: str
    #: Where the centre of the cube should be when the episode starts.
    target: Point

    def __post_init__(self) -> None:
        _require(self.dataset in DATASETS, f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        for name, least in (("episode_index", 0), ("round", 1), ("sitting", 1)):
            value = getattr(self, name)
            _require(_is_count(value, least), f"{name} must be a whole number from {least}, got {value!r}")
        _require(isinstance(self.cell, str) and self.cell != "", "cell must be a non-empty label")
        _require(isinstance(self.target, Point), f"target must be a Point, got {self.target!r}")


@dataclass(frozen=True, slots=True)
class LayoutQuality:
    """How evenly a layout's start positions cover the trained cells."""

    episodes: int
    distinct_positions: int
    #: Smallest distance between the planned starts of two episodes (0 when two episodes share a position).
    min_spacing_mm: float
    #: Average, over the episodes, of the distance to the nearest other episode's planned start.
    mean_nearest_neighbour_mm: float
    #: Largest distance from any point of the trained cells to its nearest planned start, and where it occurs.
    largest_gap_mm: float
    largest_gap_at: Point


# --------------------------------------------------------------------------------------------------------------------
# The two layouts
# --------------------------------------------------------------------------------------------------------------------


def layout(dataset: str, protocol: Mapping[str, Any], seed: int | None = None) -> list[PlannedEpisode]:
    """The planned episodes of one demonstration set, by its name."""
    if dataset == "clustered":
        return clustered_layout(protocol, seed)
    if dataset == "spread":
        return spread_layout(protocol, seed)
    raise LayoutError(f"there is no layout called {dataset!r}: the sets are {DATASETS}")


def clustered_layout(protocol: Mapping[str, Any], seed: int | None = None) -> list[PlannedEpisode]:
    """The clustered set: every episode of a cell starts at the cell's centre.

    Returns the episodes in recording order. The seed (by default `[datasets].layout_seed`) only decides the order of
    the cells within each round.
    """
    grid = Grid.from_protocol(protocol)
    return _episodes("clustered", protocol, grid, _seed(protocol, seed), lambda cell, _round: grid.cell_centre(cell))


def spread_layout(protocol: Mapping[str, Any], seed: int | None = None) -> list[PlannedEpisode]:
    """The spread set: every episode starts at a position of its own, and the positions cover the trained cells evenly.

    Returns the episodes in recording order. The seed (by default `[datasets].layout_seed`) decides the positions,
    which of a cell's positions is used in which round, and the order of the cells within each round.
    """
    grid = Grid.from_protocol(protocol)
    seed = _seed(protocol, seed)
    datasets = protocol["datasets"]
    positions = spread_positions(
        grid,
        per_cell=datasets["episodes_per_cell"],
        min_spacing_mm=datasets["spread_min_spacing_mm"],
        cell_inset_mm=protocol["placement"]["tolerance_mm"],
        rng=derived_rng(seed, "spread", "positions"),
    )
    # Which of a cell's positions is used in which round: a shuffle of its own for each cell.
    by_round = {}
    for cell, points in positions.items():
        by_round[cell] = shuffled(points, derived_rng(seed, "spread", "round-of-position", cell))
    return _episodes("spread", protocol, grid, seed, lambda cell, round_number: by_round[cell][round_number - 1])


def spread_positions(
    grid: Grid,
    per_cell: int,
    min_spacing_mm: float,
    cell_inset_mm: float,
    rng: np.random.Generator,
    *,
    relaxation_passes: int = _RELAXATION_PASSES,
    gap_moves: int = _GAP_MOVES,
) -> dict[str, list[Point]]:
    """`per_cell` start positions in each trained cell, covering the trained cells evenly (see the module's notes).

    Positions are whole millimetres. No two are closer than `min_spacing_mm`; each is in the test region and at least
    `cell_inset_mm` inside the borders of its own cell. `relaxation_passes` and `gap_moves` limit the second and third
    step (0 switches a step off); the layouts of the experiment use the defaults.
    """
    cells = grid.trained_cells
    boxes = [_allowed_box(grid, cell, cell_inset_mm) for cell in cells]
    min_squared = math.ceil(min_spacing_mm**2 - 1e-9)
    sites, site_box = _best_candidates(boxes, per_cell, min_squared, rng)
    sample_x, sample_y = _sample_points(grid)
    _relax(sites, site_box, boxes, min_squared, sample_x, sample_y, relaxation_passes)
    _close_gaps(sites, site_box, boxes, min_squared, sample_x, sample_y, gap_moves)
    positions: dict[str, list[Point]] = {cell: [] for cell in cells}
    for (x, y), box_index in zip(sites.tolist(), site_box, strict=True):
        positions[cells[box_index]].append(Point(float(x), float(y)))
    check_spread_positions(positions, grid, per_cell, min_spacing_mm, cell_inset_mm)
    return positions


def check_spread_positions(
    positions: Mapping[str, Sequence[Point]], grid: Grid, per_cell: int, min_spacing_mm: float, cell_inset_mm: float
) -> None:
    """Raise `LayoutError`, naming every broken rule, unless the positions keep the rules of the spread set.

    The rules: `per_cell` positions in each trained cell and nowhere else; each in the test region (out of the margin)
    and at least `cell_inset_mm` inside the borders of its own cell; no two closer than `min_spacing_mm`.
    `spread_positions` runs this on its own result, so it cannot hand back a layout that breaks a rule.
    """
    problems: list[str] = []
    if set(positions) != set(grid.trained_cells):
        problems.append(f"the cells {sorted(positions)} are not the trained cells {sorted(grid.trained_cells)}")
    for cell, points in positions.items():
        if len(points) != per_cell:
            problems.append(f"{cell} has {len(points)} positions, not {per_cell}")
        rect = grid.cell_rect(cell) if cell in grid.labels else None
        for point in points:
            where = f"({point.x_mm:g}, {point.y_mm:g})"
            if not grid.in_test_region(point):
                problems.append(f"{where} is in the margin at the grid's outer edge")
            depth = -math.inf  # how far inside the borders of its own cell the position is
            if rect is not None and grid.cell_of(point) == cell:
                depth_x = min(point.x_mm - rect.x_min_mm, rect.x_max_mm - point.x_mm)
                depth_y = min(point.y_mm - rect.y_min_mm, rect.y_max_mm - point.y_mm)
                depth = min(depth_x, depth_y)
            if depth < cell_inset_mm - 1e-9:
                problems.append(f"{where} is not at least {cell_inset_mm:g} mm inside cell {cell}")
    everything = [point for points in positions.values() for point in points]
    for index, first in enumerate(everything):
        for second in everything[index + 1 :]:
            if first.distance_to(second) < min_spacing_mm - 1e-9:
                problems.append(
                    f"({first.x_mm:g}, {first.y_mm:g}) and ({second.x_mm:g}, {second.y_mm:g}) are less than "
                    f"{min_spacing_mm:g} mm apart"
                )
    if problems:
        raise LayoutError("; ".join(problems))


def planned_demo_starts(episodes: Sequence[PlannedEpisode]) -> list[DemoStart]:
    """A layout as start-state records in which every cube sits exactly on its target.

    For rehearsals and tests before the camera audit exists. The analysis never uses this: its distances are to
    measured starts.
    """
    return [DemoStart(e.dataset, e.episode_index, e.cell, measured=e.target, planned=e.target) for e in episodes]


# --------------------------------------------------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------------------------------------------------


def layout_quality(episodes: Sequence[PlannedEpisode], grid: Grid) -> LayoutQuality:
    """Spacing and coverage figures of a layout's planned starts (see `LayoutQuality`).

    The largest gap is taken over the whole of the trained cells, the margin at the grid's outer edge included.
    """
    if len(episodes) == 0:
        raise LayoutError("a layout with no episodes has no quality figures")
    xy = np.array([[e.target.x_mm, e.target.y_mm] for e in episodes], dtype=float)
    if len(xy) > 1:
        dx = xy[:, None, 0] - xy[None, :, 0]
        dy = xy[:, None, 1] - xy[None, :, 1]
        distances = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(distances, np.inf)
        nearest_other = distances.min(axis=1)
        min_spacing, mean_nearest = float(nearest_other.min()), float(nearest_other.mean())
    else:
        min_spacing = mean_nearest = math.inf
    gap, where = largest_nearest_distance(xy, [grid.cell_rect(cell) for cell in grid.trained_cells])
    return LayoutQuality(
        episodes=len(episodes),
        distinct_positions=len({e.target for e in episodes}),
        min_spacing_mm=min_spacing,
        mean_nearest_neighbour_mm=mean_nearest,
        largest_gap_mm=gap,
        largest_gap_at=where,
    )


# --------------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------------


def layout_path(dataset: str, root: str | Path | None = None) -> Path:
    """Where a set's layout is kept: `config/layouts/<dataset>.json` under the repository (or under `root`)."""
    _require(dataset in DATASETS, f"dataset must be one of {DATASETS}, got {dataset!r}")
    return Path(root if root is not None else config.REPO_ROOT) / "config" / "layouts" / f"{dataset}.json"


def write_layout(
    path: str | Path, episodes: Sequence[PlannedEpisode], *, seed: int, protocol_version: str, replace: bool = False
) -> bool:
    """Write a layout as one JSON document: the set, the seed and protocol version it was made from, its episodes.

    Returns True when the file was written and False when it already held these episodes (it is then left as it was
    made, with the seed and protocol version of that time). A file that holds anything else is left as it is and
    reported with `LayoutError`, unless `replace` is true: recording may have started from the file on disk, and a
    layout must not change under a set that is half recorded.
    """
    dataset = _check_recording_order(episodes)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "seed": seed,
        "protocol_version": protocol_version,
        "episodes": [asdict(episode) for episode in episodes],
    }
    path = Path(path)
    state = layout_file_state(path, episodes)
    if state == "same":
        return False
    if state == "different" and not replace:
        raise LayoutError(f"{path} holds other episodes than the layout to be written: it was not replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def layout_file_state(path: str | Path, episodes: Sequence[PlannedEpisode]) -> str:
    """How a layout file compares with a layout: "absent", "same" (it holds these episodes) or "different"."""
    path = Path(path)
    if not path.exists():
        return "absent"
    try:
        return "same" if read_layout(path) == list(episodes) else "different"
    except (LayoutError, UnicodeDecodeError):
        return "different"


def read_layout(path: str | Path) -> list[PlannedEpisode]:
    """The planned episodes of a layout file, in recording order. A file that is not a whole layout is an error."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise LayoutError("a layout file holds one JSON object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise LayoutError(f"unsupported schema_version {payload.get('schema_version')!r}")
        if not isinstance(payload.get("episodes"), list):
            raise LayoutError("a layout file lists its episodes under 'episodes'")
        episodes = [_episode_from_json(item) for item in payload["episodes"]]
        if _check_recording_order(episodes) != payload.get("dataset"):
            raise LayoutError(f"the episodes are not of the set the file names ({payload.get('dataset')!r})")
    except (LayoutError, json.JSONDecodeError) as error:
        raise LayoutError(f"{path}: {error}") from error
    return episodes


# --------------------------------------------------------------------------------------------------------------------
# Recording order
# --------------------------------------------------------------------------------------------------------------------


def _seed(protocol: Mapping[str, Any], seed: int | None) -> int:
    return protocol["datasets"]["layout_seed"] if seed is None else seed


def _recording_order(protocol: Mapping[str, Any], dataset: str) -> list[tuple[int, int]]:
    """A set's rounds in the order they are recorded, each with the number of its sitting: `(round, sitting)`."""
    datasets = protocol["datasets"]
    order = [
        (round_number, sitting_number)
        for sitting_number, sitting in enumerate(datasets["sittings"], start=1)
        if sitting["dataset"] == dataset
        for round_number in sitting["rounds"]
    ]
    expected = list(range(1, datasets["episodes_per_cell"] + 1))
    if sorted(round_number for round_number, _ in order) != expected:
        raise ConfigError(f"[datasets].sittings must give the {dataset} set each of the rounds {expected} once")
    return order


def _episodes(
    dataset: str, protocol: Mapping[str, Any], grid: Grid, seed: int, position: Callable[[str, int], Point]
) -> list[PlannedEpisode]:
    """The episodes of a set in recording order, given where the cube starts in each cell and round."""
    episodes: list[PlannedEpisode] = []
    for round_number, sitting_number in _recording_order(protocol, dataset):
        # Each round has a stream of its own, so its order does not depend on when the round is recorded.
        for cell in shuffled(grid.trained_cells, derived_rng(seed, dataset, "round-order", round_number)):
            episodes.append(
                PlannedEpisode(dataset, len(episodes), round_number, sitting_number, cell, position(cell, round_number))
            )
    return episodes


def _check_recording_order(episodes: Sequence[PlannedEpisode]) -> str:
    """The set a list of episodes belongs to, after checking that it is one set, numbered from 0 without gaps."""
    _require(len(episodes) > 0, "a layout has at least one episode")
    datasets = {episode.dataset for episode in episodes}
    _require(len(datasets) == 1, f"a layout holds the episodes of one set, got {sorted(datasets)}")
    indices = [episode.episode_index for episode in episodes]
    _require(indices == list(range(len(episodes))), "episodes must be numbered from 0 in recording order, without gaps")
    return datasets.pop()


def _episode_from_json(data: Any) -> PlannedEpisode:
    names = {f.name for f in fields(PlannedEpisode)}
    _require(isinstance(data, dict) and set(data) == names, f"an episode has exactly the fields {sorted(names)}")
    target = data["target"]
    _require(isinstance(target, dict) and set(target) == {"x_mm", "y_mm"}, "a target has exactly x_mm and y_mm")
    try:
        return PlannedEpisode(**(data | {"target": Point(target["x_mm"], target["y_mm"])}))
    except ValueError as error:  # a Point that is not a pair of finite numbers
        raise LayoutError(str(error)) from error


# --------------------------------------------------------------------------------------------------------------------
# Spread positions: whole millimetres, so every comparison below is exact
# --------------------------------------------------------------------------------------------------------------------


def _allowed_box(grid: Grid, cell: str, inset_mm: float) -> tuple[int, int, int, int]:
    """The whole-millimetre positions a start may take in a cell: `(x_lo, x_hi, y_lo, y_hi)`, ends included."""
    rect, region = grid.cell_rect(cell), grid.test_region
    x_lo = math.ceil(max(rect.x_min_mm + inset_mm, region.x_min_mm) - 1e-9)
    x_hi = math.floor(min(rect.x_max_mm - inset_mm, region.x_max_mm) + 1e-9)
    y_lo = math.ceil(max(rect.y_min_mm + inset_mm, region.y_min_mm) - 1e-9)
    y_hi = math.floor(min(rect.y_max_mm - inset_mm, region.y_max_mm) + 1e-9)
    # A position on the border with the next cell would belong to that cell.
    if x_lo <= x_hi and y_lo <= y_hi and grid.cell_of(Point(float(x_hi), float(y_lo))) != cell:
        x_hi -= 1
    if x_lo <= x_hi and y_lo <= y_hi and grid.cell_of(Point(float(x_lo), float(y_hi))) != cell:
        y_hi -= 1
    if x_lo > x_hi or y_lo > y_hi:
        raise LayoutError(f"cell {cell} has no room for a start position inside the margin and the inset")
    return x_lo, x_hi, y_lo, y_hi


def _best_candidates(
    boxes: Sequence[tuple[int, int, int, int]], per_cell: int, min_squared: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    """Step 1. Returns the positions as an `(n, 2)` array of whole millimetres and, for each, the index of its box."""
    placed: list[tuple[int, int, int]] = []  # (box index, x, y)

    def clearance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Squared distance from each candidate to the nearest position placed so far."""
        nearest = np.full(xs.shape, np.iinfo(np.int64).max, dtype=np.int64)
        for _, x, y in placed:
            np.minimum(nearest, (xs - x) ** 2 + (ys - y) ** 2, out=nearest)
        return nearest

    for _ in range(per_cell):
        for box_index in shuffled(range(len(boxes)), rng):
            x_lo, x_hi, y_lo, y_hi = boxes[box_index]
            xs = np.array([x_lo + pick_index(x_hi - x_lo + 1, rng) for _ in range(_CANDIDATES)], dtype=np.int64)
            ys = np.array([y_lo + pick_index(y_hi - y_lo + 1, rng) for _ in range(_CANDIDATES)], dtype=np.int64)
            free = clearance(xs, ys)
            if int(free.max()) < min_squared:
                # No candidate keeps the spacing: look at every position of the box before giving up.
                whole_box = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1), indexing="ij")
                xs, ys = whole_box[0].ravel(), whole_box[1].ravel()
                free = clearance(xs, ys)
                if int(free.max()) < min_squared:
                    raise LayoutError(
                        f"could not place {per_cell} start positions in every cell at least "
                        f"{math.sqrt(min_squared):.1f} mm apart: the first placement ran out of room"
                    )
            best = int(np.argmax(free))
            placed.append((box_index, int(xs[best]), int(ys[best])))
    placed.sort(key=lambda entry: entry[0])
    return np.array([[x, y] for _, x, y in placed], dtype=np.int64), [box_index for box_index, _, _ in placed]


def _sample_points(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Every whole-millimetre point of the trained cells, borders included: where coverage is measured in steps 2, 3."""
    xs, ys = np.meshgrid(
        np.arange(0, math.floor(grid.width_mm) + 1), np.arange(0, math.floor(grid.height_mm) + 1), indexing="ij"
    )
    inside = grid.trained_cells_mask(xs, ys)
    return xs[inside].astype(np.int64), ys[inside].astype(np.int64)


def _nearest_two(
    sites: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each sample point: its nearest position, the squared distance to it, and to the second nearest."""
    far = np.iinfo(np.int64).max
    owner = np.zeros(sample_x.shape, dtype=np.int64)
    nearest = np.full(sample_x.shape, far, dtype=np.int64)
    second = np.full(sample_x.shape, far, dtype=np.int64)
    for index, (x, y) in enumerate(sites.tolist()):
        squared = (sample_x - x) ** 2 + (sample_y - y) ** 2
        closer = squared < nearest
        second = np.where(closer, nearest, np.minimum(second, squared))
        nearest = np.where(closer, squared, nearest)
        owner[closer] = index
    return owner, nearest, second


def _keeps_spacing(sites: np.ndarray, index: int, x: int, y: int, min_squared: int) -> bool:
    """True when position `index` could sit at `(x, y)` without coming too close to another position."""
    squared = (sites[:, 0] - x) ** 2 + (sites[:, 1] - y) ** 2
    squared[index] = min_squared
    return int(squared.min()) >= min_squared


def _relax(
    sites: np.ndarray,
    site_box: Sequence[int],
    boxes: Sequence[tuple[int, int, int, int]],
    min_squared: int,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    passes: int,
) -> None:
    """Step 2, in place: move each position to the middle of the sample points nearest to it, until nothing moves."""
    for _ in range(passes):
        owner, _, _ = _nearest_two(sites, sample_x, sample_y)
        count = np.bincount(owner, minlength=len(sites)).tolist()
        # Sums of whole numbers far below 2**53: exact.
        sum_x = np.bincount(owner, weights=sample_x, minlength=len(sites)).tolist()
        sum_y = np.bincount(owner, weights=sample_y, minlength=len(sites)).tolist()
        moved = False
        for index in range(len(sites)):
            if count[index] == 0:
                continue
            x_lo, x_hi, y_lo, y_hi = boxes[site_box[index]]
            # The mean, rounded to the nearest millimetre in whole-number arithmetic, kept inside the box.
            x = min(max((2 * int(sum_x[index]) + count[index]) // (2 * count[index]), x_lo), x_hi)
            y = min(max((2 * int(sum_y[index]) + count[index]) // (2 * count[index]), y_lo), y_hi)
            stays = (x, y) == (int(sites[index, 0]), int(sites[index, 1]))
            if not stays and _keeps_spacing(sites, index, x, y, min_squared):
                sites[index] = (x, y)
                moved = True
        if not moved:
            return


def _close_gaps(
    sites: np.ndarray,
    site_box: Sequence[int],
    boxes: Sequence[tuple[int, int, int, int]],
    min_squared: int,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    moves: int,
) -> None:
    """Step 3, in place: move a position beside the largest gap by one millimetre while that makes the gap smaller.

    A layout is scored by its largest squared distance from a sample point to its nearest position, then by how many
    sample points are that far, then by the sum over all sample points. A move is made only if it lowers the score,
    so the largest gap never grows and the loop ends.
    """

    def score(squared: np.ndarray) -> tuple[int, int, int]:
        largest = int(squared.max())
        return largest, int(np.count_nonzero(squared == largest)), int(squared.sum())

    for _ in range(moves):
        owner, nearest, second = _nearest_two(sites, sample_x, sample_y)
        current = score(nearest)
        best_move: tuple[tuple[int, int, int], int, int, int] | None = None
        for sample in np.flatnonzero(nearest == current[0])[:_GAPS_TRIED].tolist():
            to_gap = (sites[:, 0] - sample_x[sample]) ** 2 + (sites[:, 1] - sample_y[sample]) ** 2
            for index in np.argsort(to_gap, kind="stable")[:_NEIGHBOURS].tolist():
                x_lo, x_hi, y_lo, y_hi = boxes[site_box[index]]
                without = np.where(owner == index, second, nearest)  # the distances if this position were absent
                for step_x, step_y in _UNIT_MOVES:
                    x, y = int(sites[index, 0]) + step_x, int(sites[index, 1]) + step_y
                    in_box = x_lo <= x <= x_hi and y_lo <= y <= y_hi
                    if not in_box or not _keeps_spacing(sites, index, x, y, min_squared):
                        continue
                    moved = score(np.minimum(without, (sample_x - x) ** 2 + (sample_y - y) ** 2))
                    if moved < current and (best_move is None or moved < best_move[0]):
                        best_move = (moved, index, x, y)
            if best_move is not None:
                break
        if best_move is None:
            return
        sites[best_move[1]] = (best_move[2], best_move[3])


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayoutError(message)


def _is_count(value: Any, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least
