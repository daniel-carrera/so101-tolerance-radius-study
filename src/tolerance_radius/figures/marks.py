"""Where marks go: the geometry the figures share, with no drawing in it.

* `stack_heights`: trials shown as marks along a distance axis must not hide one another, and may not move sideways,
  because a mark's place along the axis is a measurement. They are piled up instead;
* `untested_stretches`: the parts of a distance axis in which nothing was tested, so that a figure can say so;
* `curve_pieces`: the parts of a distance axis over which a fitted curve has trials to stand on;
* `spread_apart`: marks along a line that would cover each other, moved apart as little as it takes;
* `free_slot`: where along a line there is still room for a label;
* `hatch_segments`: the lines of a hatching, as plain line segments.

Standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tolerance_radius.geometry import Rect


def stack_heights(x_mm: Sequence[float], diameter_mm: float) -> list[float]:
    """How far above a line each of a row of round marks is drawn, so that no mark covers another.

    `x_mm` holds where each mark belongs along the line and `diameter_mm` the room one mark takes, all in the same
    unit (millimetres on the page). The marks are dropped one at a time, from the smallest x to the largest, each as
    low as it can go: on the line if nothing is in the way, otherwise resting on the marks already there. No mark
    moves sideways. The result is the height of each mark's centre above the line, in the order of `x_mm`.
    """
    if not (math.isfinite(diameter_mm) and diameter_mm > 0.0):
        raise ValueError(f"a mark has a positive diameter, got {diameter_mm!r}")
    xs = [float(x) for x in x_mm]
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("a mark's place is a finite number")
    heights = [0.0] * len(xs)
    placed: list[tuple[float, float]] = []
    clear = diameter_mm**2 - 1e-9  # two marks may touch
    for index in sorted(range(len(xs)), key=lambda i: (xs[i], i)):
        x = xs[index]
        near = [(px, py) for px, py in placed if abs(px - x) < diameter_mm]
        # The lowest free place is on the line or resting on one of the marks within reach.
        resting = sorted({0.0, *(py + math.sqrt(diameter_mm**2 - (px - x) ** 2) for px, py in near)})
        height = next(y for y in resting if all((px - x) ** 2 + (py - y) ** 2 >= clear for px, py in near))
        heights[index] = height
        placed.append((x, height))
    return heights


def untested_stretches(
    distances_mm: Sequence[float], *, upto_mm: float, longer_than_mm: float
) -> list[tuple[float, float]]:
    """The stretches of a distance axis from 0 to `upto_mm` in which no trial lies, where longer than a given length.

    Each is `(start, end)`: the trial below it and the trial above it, or an end of the axis where there is no trial
    on that side. With no trial at all the whole axis is one stretch.
    """
    if not upto_mm > 0.0:
        raise ValueError(f"the axis must reach beyond 0, got {upto_mm!r}")
    inside = sorted({float(d) for d in distances_mm if 0.0 <= d <= upto_mm})
    marks = [0.0, *inside, float(upto_mm)]
    return [(low, high) for low, high in zip(marks, marks[1:], strict=False) if high - low > longer_than_mm]


def curve_pieces(distances_mm: Sequence[float], *, longer_than_mm: float) -> list[tuple[float, float]]:
    """The stretches of a distance axis over which a curve fitted to these trials may be drawn.

    A curve is drawn from the nearest trial to the farthest and no further, and it is interrupted wherever two
    neighbouring trials are more than `longer_than_mm` apart: across such a gap the curve would show the model and
    not the data. Each piece is `(start, end)` with `start < end`; a trial that stands alone gives no piece.
    """
    tested = sorted({float(d) for d in distances_mm})
    pieces: list[tuple[float, float]] = []
    start = previous = None
    for d in tested:
        if start is None:
            start = d
        elif d - previous > longer_than_mm:
            if previous > start:
                pieces.append((start, previous))
            start = d
        previous = d
    if start is not None and previous > start:
        pieces.append((start, previous))
    return pieces


def spread_apart(places_mm: Sequence[float], least_mm: float) -> list[float]:
    """Places along a line, moved as little as it takes for no two to be closer than `least_mm`.

    Places that have room stay where they are. Places that would crowd each other are set `least_mm` apart, in their
    order, about the mean of where they belong, so that the group as a whole does not move. The result is in the
    order of `places_mm`.
    """
    if not (math.isfinite(least_mm) and least_mm >= 0.0):
        raise ValueError(f"the least distance between two places is 0 or more, got {least_mm!r}")
    order = sorted(range(len(places_mm)), key=lambda index: (places_mm[index], index))
    # Groups of neighbours, each a list of indices; a group is laid out evenly about the mean of its places.
    groups: list[list[int]] = []
    for index in order:
        groups.append([index])
        while len(groups) > 1 and _low_end(groups[-1], places_mm, least_mm) - _high_end(
            groups[-2], places_mm, least_mm
        ) < least_mm - 1e-12:
            last = groups.pop()
            groups[-1] += last
    moved = [0.0] * len(places_mm)
    for group in groups:
        start = _low_end(group, places_mm, least_mm)
        for position, index in enumerate(group):
            moved[index] = start + position * least_mm
    return moved


def _low_end(group: Sequence[int], places_mm: Sequence[float], least_mm: float) -> float:
    centre = sum(places_mm[index] for index in group) / len(group)
    return centre - least_mm * (len(group) - 1) / 2.0


def _high_end(group: Sequence[int], places_mm: Sequence[float], least_mm: float) -> float:
    return _low_end(group, places_mm, least_mm) + least_mm * (len(group) - 1)


def free_slot(
    taken_mm: Sequence[tuple[float, float]], width_mm: float, *, low_mm: float, high_mm: float, clear_mm: float = 0.0
) -> float | None:
    """Where something `width_mm` wide can go between `low_mm` and `high_mm` without coming within `clear_mm` of a
    stretch that is taken: the start of the place nearest the high end, or None when there is no room.
    """
    blocked = sorted((start - clear_mm, end + clear_mm) for start, end in taken_mm)
    # The place nearest the high end either ends at the high end or ends where a taken stretch begins.
    candidates = [high_mm - width_mm] + [start - width_mm for start, _ in blocked]
    for start in sorted(candidates, reverse=True):
        if start < low_mm - 1e-12 or start + width_mm > high_mm + 1e-12:
            continue
        end = start + width_mm
        if all(end <= taken_start + 1e-12 or start >= taken_end - 1e-12 for taken_start, taken_end in blocked):
            return start
    return None


def hatch_segments(
    rect: Rect, spacing_mm: float, *, rising: bool = True
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The lines of a diagonal hatching over a rectangle, as segments cut off at its edges.

    The lines run at 45 degrees (rising to the right, or falling) and are `spacing_mm` apart, measured across them.
    They are laid out from the origin of the frame and not from the rectangle, so that the hatchings of two
    rectangles that touch run through from one into the other.
    """
    if not (math.isfinite(spacing_mm) and spacing_mm > 0.0):
        raise ValueError(f"hatch lines are a positive distance apart, got {spacing_mm!r}")
    # A rising line is y = x + c and a falling one y = -x + c; lines `spacing_mm` apart differ in c by its sqrt(2).
    step = spacing_mm * math.sqrt(2.0)
    if rising:
        lowest, highest = rect.y_min_mm - rect.x_max_mm, rect.y_max_mm - rect.x_min_mm
    else:
        lowest, highest = rect.y_min_mm + rect.x_min_mm, rect.y_max_mm + rect.x_max_mm
    segments = []
    for k in range(math.ceil(lowest / step - 1e-9), math.floor(highest / step + 1e-9) + 1):
        c = k * step
        if rising:
            x_start, x_end = max(rect.x_min_mm, rect.y_min_mm - c), min(rect.x_max_mm, rect.y_max_mm - c)
            ends = ((x_start, x_start + c), (x_end, x_end + c))
        else:
            x_start, x_end = max(rect.x_min_mm, c - rect.y_max_mm), min(rect.x_max_mm, c - rect.y_min_mm)
            ends = ((x_start, c - x_start), (x_end, c - x_end))
        if x_end - x_start > 1e-9:
            segments.append(ends)
    return segments


def stack_extent(heights: Sequence[float], diameter_mm: float) -> float:
    """The room a pile of marks takes above its line: from the line to the top of the highest mark. 0 for no mark."""
    return (max(heights) + diameter_mm) if len(heights) else 0.0
