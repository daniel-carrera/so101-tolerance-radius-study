"""Figure 1: the earlier evaluation, by distance.

Each of the earlier evaluation's trials is one mark: its outcome (success above the rate axis, failure below) against
the distance from the cube's start position to the nearest demonstration the policy was trained on. Trials in cells
with demonstrations and trials in held-out cells have marks of their own; a trial whose start position comes from a
still and not from its video has an open mark. The stretch in which no trial started is shaded and says so. The
success rate of each of the two groups is drawn with its Wilson interval, over the distances the group covers.

No curve is drawn: the data have nothing between the two groups for a curve to stand on.

The data are `analysis/pilot_august/trials_with_distance.csv`, whose columns that folder's README explains. Every
number in the figure (the counts, the intervals, the ends of the shaded stretch) is computed here from that file.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.lines import Line2D

from tolerance_radius.figures import HEIGHTS_MM, marks, style
from tolerance_radius.figures.ratechart import MarkGroup, RateChart, mark_look
from tolerance_radius.figures.style import FigureError, Page, Series
from tolerance_radius.intervals import wilson_interval

#: The columns of the CSV this figure reads.
COLUMNS = ("trial", "d_nearest_trained49_mm", "success", "kind", "position_source")
#: `kind` as logged in the earlier evaluation: a cell with demonstrations, a held-out cell.
KIND_TRAINED, KIND_HELD_OUT = "id", "ood"
#: `position_source` of a trial measured from the first frame of its own video.
FROM_VIDEO = "video frame"

#: The two groups of trials. Both are trials of one policy, trained on a dataset laid out like the clustered one, so
#: both wear that layout's colour; the marker says which kind of cell a trial started in.
IN_TRAINED_CELL = Series("cell with demonstrations", style.BLUE, style.BLUE_TINT, "o")
IN_HELD_OUT_CELL = Series("held-out cell", style.BLUE, style.BLUE_TINT, "^")
FROM_STILL_LABEL = "same, from a still image"

X_LABEL = "Distance to the nearest trained demonstration, $d$ (mm)"
#: A stretch of distances with no trial is shaded when it is longer than this: one and a half of the 10 mm steps in
#: which distances are binned and staircases start.
UNTESTED_LONGER_THAN_MM = 15.0
LEVEL = 0.95
#: The mark of a group's rate, larger than a trial's.
RATE_MARK_PT = 4.4


@dataclass(frozen=True, slots=True)
class EarlierTrial:
    """One trial of the earlier evaluation, as far as the figure needs it."""

    trial: int
    #: From the cube's start position to the nearest of the demonstrations the policy was trained on.
    distance_mm: float
    success: bool
    #: True for a trial in a cell with no demonstrations.
    held_out: bool
    #: True when the start position was measured from a still and not from the trial's own video.
    from_still: bool


@dataclass(frozen=True, slots=True)
class GroupRate:
    """The success rate of one group of trials, and the distances the group covers."""

    trials: int
    successes: int
    #: The Wilson interval of the rate.
    low: float
    high: float
    nearest_mm: float
    farthest_mm: float

    @property
    def rate(self) -> float:
        return self.successes / self.trials


def read_trials(path: str | Path) -> list[EarlierTrial]:
    """The trials of `trials_with_distance.csv`, in the file's order. A row the figure cannot use is an error."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise FigureError(f"{path.name} has no column {missing}")
        trials = []
        for number, row in enumerate(reader, start=2):
            try:
                distance = float(row["d_nearest_trained49_mm"])
                success = {"1": True, "0": False}[row["success"]]
                held_out = {KIND_TRAINED: False, KIND_HELD_OUT: True}[row["kind"]]
                trial = int(row["trial"])
            except (KeyError, ValueError) as error:
                raise FigureError(f"{path.name}, line {number}: cannot read {error}") from error
            if not (math.isfinite(distance) and distance >= 0.0):
                raise FigureError(f"{path.name}, line {number}: the distance is not a number of 0 or more")
            trials.append(EarlierTrial(trial, distance, success, held_out, row["position_source"] != FROM_VIDEO))
    if not trials:
        raise FigureError(f"{path.name} holds no trial")
    return trials


def group_rate(trials: Sequence[EarlierTrial], level: float = LEVEL) -> GroupRate | None:
    """The success rate of some trials with its Wilson interval; None for no trials."""
    if not trials:
        return None
    successes = sum(trial.success for trial in trials)
    low, high = wilson_interval(successes, len(trials), level)
    distances = [trial.distance_mm for trial in trials]
    return GroupRate(len(trials), successes, low, high, min(distances), max(distances))


def stretch_note(start_mm: float, end_mm: float, upto_mm: float) -> list[list[str]]:
    """Ways of saying that no trial lies in a stretch, the fullest first, each as lines of text."""
    if start_mm <= 0.0:
        return [["no trial", f"below {end_mm:.1f} mm"], ["no trial"]]
    if end_mm >= upto_mm:
        return [["no trial", f"beyond {start_mm:.1f} mm"], ["no trial"]]
    return [
        ["no trial between", f"{start_mm:.1f} and {end_mm:.1f} mm"],
        ["no trial", "between", f"{start_mm:.1f} and", f"{end_mm:.1f} mm"],
        ["no trial"],
    ]


def figure(trials: Sequence[EarlierTrial]) -> Page:
    """Draw Figure 1 from the trials of the earlier evaluation."""
    if not trials:
        raise FigureError("the earlier evaluation has no trial to draw")
    with style.settings():
        page = Page(HEIGHTS_MM[1])
        groups, members = [], []
        for series, held_out in ((IN_TRAINED_CELL, False), (IN_HELD_OUT_CELL, True)):
            chosen = [trial for trial in trials if trial.held_out == held_out]
            members.append(chosen)
            groups.append(
                MarkGroup(
                    series,
                    tuple(trial.distance_mm for trial in chosen),
                    tuple(trial.success for trial in chosen),
                    tuple(trial.from_still for trial in chosen),
                )
            )
        farthest = max(trial.distance_mm for trial in trials)
        x_max = 10.0 * max(1, math.ceil(farthest / 10.0 - 1e-9))
        chart = RateChart(page, groups, x_max_mm=x_max, x_label=X_LABEL, separate_lanes=False)

        # What is drawn in the rate band, as boxes on the page, so that the legend can keep clear of it.
        taken: list[Box] = []
        distances = [trial.distance_mm for trial in trials]
        for start, end in marks.untested_stretches(distances, upto_mm=x_max, longer_than_mm=UNTESTED_LONGER_THAN_MM):
            room = chart.x(end) - chart.x(start) - 1.0
            fitting = [
                lines
                for lines in stretch_note(start, end, x_max)
                if max(style.text_width_mm(line) for line in lines) <= room
            ]
            chart.shade(start, end, fitting[0] if fitting else [])
            taken.append((chart.x(start), chart.band_low_mm, chart.x(end), chart.band_high_mm))
        for series, chosen in zip((IN_TRAINED_CELL, IN_HELD_OUT_CELL), members, strict=True):
            rate = group_rate(chosen)
            if rate is not None:
                taken += _draw_rate(chart, series, rate)
        _draw_legend(chart, any(trial.from_still for trial in trials), taken)
    return page


#: A rectangle on the page: `(x_min, y_min, x_max, y_max)` in millimetres.
Box = tuple[float, float, float, float]


def _draw_rate(chart: RateChart, series: Series, rate: GroupRate) -> list[Box]:
    """A group's success rate: a line over the distances the group covers, its Wilson interval, and its count.

    Returns the room these take on the page.
    """
    axes = chart.page.axes
    y = chart.y(rate.rate)
    x_near, x_far = chart.x(rate.nearest_mm), chart.x(rate.farthest_mm)
    x_mid = (x_near + x_far) / 2.0
    axes.add_line(Line2D([x_near, x_far], [y, y], color=series.color, linewidth=1.3, solid_capstyle="butt",
                         zorder=3.0, gid=f"rate-span:{series.label}"))  # fmt: skip
    axes.add_line(Line2D([x_mid, x_mid], [chart.y(rate.low), chart.y(rate.high)], color=series.color,
                         linewidth=style.INTERVAL_PT, solid_capstyle="butt", zorder=3.0,
                         gid=f"rate-interval:{series.label}"))  # fmt: skip
    axes.add_line(Line2D([x_mid], [y], linestyle="none", marker=series.marker, zorder=3.5,
                         gid=f"rate:{series.label}", **mark_look(series, RATE_MARK_PT)))  # fmt: skip
    # The count goes beside the interval: beyond the end of the group's line where the frame has room for it there,
    # so that it does not straddle the edge of a shaded stretch, and next to the interval otherwise.
    label = f"{rate.successes} of {rate.trials}"
    width = style.text_width_mm(label)
    baseline = style.centred_baseline_mm(chart.y((rate.low + rate.high) / 2.0))
    left = x_far + 1.2 if x_far + 1.2 + width <= chart.right_mm else x_mid + 1.3
    if left + width > chart.right_mm:
        left = x_mid - 1.3 - width
    chart.page.text(left, baseline, label, gid=f"rate-count:{series.label}")
    return [
        (x_near, min(y, chart.y(rate.low)) - 0.9, x_far, max(y, chart.y(rate.high)) + 0.9),
        (left, baseline - 0.6, left + width, baseline + 2.0),
    ]


def _draw_legend(chart: RateChart, any_from_still: bool, taken: Sequence[Box]) -> None:
    """Name the marks, in a corner of the rate band that nothing else uses.

    The legend stays out of the strips, where its marks would read as trials, and out of a shaded stretch, where no
    trial lies.
    """
    entries: list[tuple[Series, bool, str]] = [(IN_TRAINED_CELL, False, IN_TRAINED_CELL.label)]
    if any_from_still:
        entries.append((IN_TRAINED_CELL, True, FROM_STILL_LABEL))
    entries.append((IN_HELD_OUT_CELL, False, IN_HELD_OUT_CELL.label))
    pitch, mark_room, margin = 2.7, 2.3, 1.2
    width = mark_room + max(style.text_width_mm(label) for _, _, label in entries)
    height = pitch * len(entries)
    lefts = (chart.right_mm - width, chart.left_mm + margin)
    tops = (chart.band_high_mm - margin, chart.band_low_mm + margin + height)
    for left, top in ((left, top) for top in tops for left in lefts):
        box = (left - 0.5, top - height, left + width, top)
        if not any(box[0] < x1 and x0 < box[2] and box[1] < y1 and y0 < box[3] for x0, y0, x1, y1 in taken):
            break
    else:
        raise FigureError("the legend of the earlier evaluation has no free corner of the rate band")
    for index, (series, open_mark, label) in enumerate(entries):
        y = top - (index + 0.5) * pitch
        look = mark_look(series, chart.mark_pt, open_mark=open_mark)
        chart.page.axes.add_line(Line2D([left + 0.8], [y], linestyle="none", marker=series.marker, zorder=6.0,
                                        gid="legend-mark", **look))  # fmt: skip
        chart.page.text(left + mark_room, style.centred_baseline_mm(y), label, gid="legend-text")
