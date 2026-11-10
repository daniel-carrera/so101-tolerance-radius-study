"""Figure 3: the response curves of the clustered policy and of the replay.

Success against the distance to the nearest demonstration, for two conditions on one axis. For each condition:

* every scored trial of its threshold block as a mark at its measured distance, successes above the rate axis and
  failures below;
* the success rate in 10 mm stretches of distance with its Wilson interval, drawn at the mean distance of the
  stretch's trials;
* the posterior-mean curve of the fitted model with its band;
* the tolerance radius in a row above the distance axis: the posterior median as a mark, its interval as a bar, and a
  line from the mark up to the curve.

What the figure will not draw
-----------------------------
* **An interval that is open on a side** is drawn as a bar that runs off to that end of the axis and ends in an
  arrow. It then carries no mark for the median and no line to the curve: the trials do not locate the radius, and
  the protocol reports "at least" or "at most" (`protocol/protocol.md`, section 10).
* **A curve where nothing was tested.** A condition's curve runs from its nearest trial to its farthest and is
  interrupted wherever two neighbouring trials are more than a bin and a half apart.
* **A rate where no trial is.** A stretch's rate stands at the mean distance of its trials, not at the middle of the
  stretch: four trials just beyond 40 mm do not put a mark at 45.
* **A stretch that no condition tested** is shaded and says so: between trials when it is longer than a bin and a
  half, and beyond the farthest trial when the axis runs on for more than half a bin (it does when an interval's end
  lies out there, which then rests on the model and not on trials).

Which fit, and which trials
---------------------------
Each condition's **block fit** is drawn, with the trials of that block and no others: the clustered policy's
threshold block and the replay's ran interleaved in one session, the comparison of the two radii uses these fits, and
the discs of Figure 4 have this radius. A condition without a block fit is not drawn, whatever other trials it has.
`fits="pooled"` draws the clustered policy's pooled fit instead, with all the trials that fit was made from (its
trials of the workspace test included); the replay has no pooled fit and keeps its block fit.

The rates are counted here from the trials that are drawn, by the protocol's rule and the project's own Wilson
interval, so that they always belong to the marks beside them. The same counting is first applied to all of a
condition's trials and compared with the bins the analysis wrote: if it does not give them, the figure and the
analysis count differently, and the figure is refused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from tolerance_radius.figures import HEIGHTS_MM, marks, results, style
from tolerance_radius.figures.ratechart import MarkGroup, RateChart, mark_look
from tolerance_radius.figures.results import Bin, Fit, ResultsError, ScoredTrial
from tolerance_radius.figures.style import FigureError, Page, Series

#: The two conditions of the figure, the first next to the rate band, with the names the figure prints.
CONDITIONS: tuple[tuple[str, Series, str], ...] = (
    ("act_clustered", style.CLUSTERED, "policy"),
    ("replay", style.REPLAY, "replay"),
)
#: The label of the distance axis says whether the distances were divided by the instrument's scale factor.
X_LABELS = {
    True: "Distance to the nearest demonstration, $d$ (mm, corrected for scale)",
    False: "Distance to the nearest demonstration, $d$ (mm, as read)",
}
#: The distance axis ends at a multiple of this, as near beyond the farthest thing drawn as that allows.
AXIS_END_STEP_MM = 5.0
#: Two neighbouring trials farther apart than this many bins leave a stretch that counts as untested, and so does an
#: axis that runs on for more than `UNTESTED_END_BINS` beyond the farthest trial.
UNTESTED_BINS = 1.5
UNTESTED_END_BINS = 0.5
UNTESTED_NOTE = "not tested"
#: The marks of two rates are at least this far apart on the page, so that neither covers the other.
RATES_APART_MM = 1.75
#: The opacity of a curve's band, by the colour of its series: ink needs less to look as light as a colour.
BAND_ALPHA = {style.INK: 0.11}
DEFAULT_BAND_ALPHA = 0.2
#: The rate band holds two curves, their bands and their rates: the trials' marks are made smaller, if need be, to
#: leave it this height.
LEAST_BAND_MM = 17.5
RATE_MARK_PT = 4.0
RADIUS_MARK_PT = 4.6
RADIUS_BAR_PT = 1.7
#: Half the height of the mark that ends an interval's bar, and the length of the arrowhead of an open end.
END_MM = 0.75
ARROW_MM = 1.7
#: What the bars in the rows above the distance axis are, and the room the words keep from a bar.
ROWS_NAME = "tolerance radius"
ROWS_NAME_CLEAR_MM = 1.5


@dataclass(frozen=True, slots=True)
class Shown:
    """What the figure draws of one condition."""

    series: Series
    name: str
    fit: Fit
    #: The trials drawn as marks: those the fit was made from.
    trials: tuple[ScoredTrial, ...]
    #: Their rates by stretch of distance.
    bins: tuple[Bin, ...]


def shown(results_: Mapping[str, Any], condition: str, series: Series, name: str, fits: str = "block") -> Shown | None:
    """What to draw of a condition: its fit, the trials the fit was made from, and their rates. None when the
    condition has no fit to draw, which is when its threshold block was not run.
    """
    everything = results.trials(results_, condition)
    width, level = results.bin_width_mm(results_), results.interval_level(results_)
    if everything:
        _check_bins(results_, condition, everything, width, level)
    fit = results.fit(results_, condition, fits) or results.fit(results_, condition, "block")
    if fit is None:
        return None
    drawn = [trial for trial in everything if trial.block in fit.blocks]
    if not drawn:
        blocks = list(fit.blocks)
        raise ResultsError(f"conditions.{condition}: the fit names the blocks {blocks} and no trial is in them")
    return Shown(series, name, fit, tuple(drawn), tuple(results.binned(drawn, width, level)))


def _check_bins(
    results_: Mapping[str, Any], condition: str, everything: Sequence[ScoredTrial], width: float, level: float
) -> None:
    """Refuse a file whose bins this code's counting does not give: the figure's rates would not be the analysis's."""
    written = results.file_bins(results_, condition)
    counted = results.binned(everything, width, level)
    if results.same_bins(written, counted):
        return
    for ours, theirs in zip(counted, written, strict=False):
        if not results.same_bins([ours], [theirs]):
            break
    else:
        ours, theirs = (counted[len(written)], None) if len(counted) > len(written) else (None, written[len(counted)])
    raise ResultsError(
        f"conditions.{condition}.bins are not the counts of the trials listed, in stretches of {width:g} mm closed "
        f"below, with Wilson intervals at {level:g}: the file has {_said(theirs)} where this code counts "
        f"{_said(ours)}. The figure would show other rates than the analysis"
    )


def _said(bin_: Bin | None) -> str:
    if bin_ is None:
        return "no further bin"
    return (f"{bin_.k} of {bin_.n} in {bin_.lo_mm:g} to {bin_.hi_mm:g} mm "
            f"(interval {bin_.wilson_lo:.4f} to {bin_.wilson_hi:.4f})")  # fmt: skip


def conditions_shown(results_: Mapping[str, Any], fits: str = "block") -> list[Shown]:
    """What the figure draws, condition by condition; empty when neither threshold block was run."""
    if fits not in results.FITS:
        raise FigureError(f"fits is one of {results.FITS}, got {fits!r}")
    return [found for condition, series, name in CONDITIONS
            if (found := shown(results_, condition, series, name, fits)) is not None]  # fmt: skip


def has_something_to_draw(results_: Mapping[str, Any]) -> bool:
    """Whether the results hold the threshold block of one of the figure's two conditions."""
    return any(results.fit(results_, condition, "block") is not None for condition, _, _ in CONDITIONS)


def axis_end_mm(drawn: Sequence[Shown], bin_mm: float) -> float:
    """Where the distance axis ends: at the first multiple of `AXIS_END_STEP_MM` that holds every trial drawn and
    every end of an interval that the trials close. An open end runs off the axis and needs no room on it.
    """
    reach = max(trial.d_mm for item in drawn for trial in item.trials)
    for item in drawn:
        closed = [bound for bound, is_open in ((item.fit.r50_lo_mm, item.fit.open_below),
                                               (item.fit.r50_hi_mm, item.fit.open_above)) if not is_open]  # fmt: skip
        reach = max([reach, *closed])
    return max(bin_mm, AXIS_END_STEP_MM * math.ceil(reach / AXIS_END_STEP_MM - 1e-9))


def untested_stretches(drawn: Sequence[Shown], x_max_mm: float, bin_mm: float) -> list[tuple[float, float]]:
    """The stretches of the axis that no condition drawn has a trial in, and that are long enough to say so."""
    everything = [trial.d_mm for item in drawn for trial in item.trials]
    stretches = marks.untested_stretches(everything, upto_mm=x_max_mm, longer_than_mm=UNTESTED_BINS * bin_mm)
    farthest = max(everything)
    if x_max_mm - farthest > UNTESTED_END_BINS * bin_mm and (farthest, x_max_mm) not in stretches:
        stretches.append((farthest, x_max_mm))
    return stretches


def figure(results_: Mapping[str, Any], *, fits: str = "block") -> Page:
    """Draw Figure 3 from the analysis results. `fits` is "block" (the paper's figure) or "pooled"."""
    results.check_header(results_)
    drawn = conditions_shown(results_, fits)
    if not drawn:
        raise FigureError("the results hold no threshold block of the clustered policy or of the replay")
    bin_mm = results.bin_width_mm(results_)
    with style.settings():
        page = Page(HEIGHTS_MM[3])
        x_max = axis_end_mm(drawn, bin_mm)
        groups = [MarkGroup(item.series, tuple(t.d_mm for t in item.trials), tuple(t.success for t in item.trials))
                  for item in drawn]  # fmt: skip
        chart = RateChart(page, groups, x_max_mm=x_max, x_label=X_LABELS[results.scale_applied(results_)],
                          rows=len(drawn), tick_step_mm=bin_mm, least_band_mm=LEAST_BAND_MM)  # fmt: skip

        untested = untested_stretches(drawn, x_max, bin_mm)
        for start, end in untested:
            chart.shade(start, end, [UNTESTED_NOTE])
        longer_than = UNTESTED_BINS * bin_mm
        for item in drawn:
            _draw_curve(chart, item, longer_than)
        _draw_rates(chart, drawn, untested)
        taken: list[tuple[float, float]] = []
        for index, item in enumerate(drawn):
            taken.append(_draw_radius(chart, item, chart.row_y_mm[index], longer_than))
        _name_rows(chart, taken)
        for condition, _, name in CONDITIONS:
            if name not in [item.name for item in drawn]:
                has_trials = bool(results.trials(results_, condition))
                page.notes.append(
                    f"the {name} has trials and no block fit in the results: none of them is drawn"
                    if has_trials
                    else f"the {name} has no trial in the results: the figure shows one condition only"
                )
        if results.is_draft(results_):
            page.stamp("DRAFT")
    return page


def _draw_curve(chart: RateChart, item: Shown, longer_than_mm: float) -> None:
    """The posterior-mean curve and its band, over the stretches the condition's trials cover."""
    fit, series, axes = item.fit, item.series, chart.page.axes
    tested = [trial.d_mm for trial in item.trials]
    for start, end in marks.curve_pieces(tested, longer_than_mm=longer_than_mm):
        start, end = max(start, fit.curve_d_mm[0]), min(end, fit.curve_d_mm[-1])
        if end <= start:
            continue
        distances = [start, *(d for d in fit.curve_d_mm if start < d < end), end]
        xs = [chart.x(d) for d in distances]

        def on_page(values: Sequence[float]) -> list[float]:
            return [chart.y(results.interpolate(fit.curve_d_mm, values, d)) for d in distances]  # noqa: B023

        axes.fill_between(xs, on_page(fit.curve_lo), on_page(fit.curve_hi), facecolor=series.color, linewidth=0.0,
                          alpha=BAND_ALPHA.get(series.color, DEFAULT_BAND_ALPHA), zorder=1.2,
                          gid=f"band:{item.name}")  # fmt: skip
        line = Line2D(xs, on_page(fit.curve_mean), color=series.color, linewidth=style.CURVE_PT, zorder=2.6,
                      solid_capstyle="butt", dash_capstyle="butt", gid=f"curve:{item.name}")  # fmt: skip
        if series.dashes is not None:
            line.set_dashes(series.dashes)
        axes.add_line(line)


def rate_places(
    chart: RateChart, drawn: Sequence[Shown], untested: Sequence[tuple[float, float]] = ()
) -> list[list[float]]:
    """Page x of every rate, condition by condition and bin by bin: the mean distance of the bin's trials, moved
    aside by as little as it takes where two marks would cover each other, and never into a stretch that is shaded
    as untested (`untested`, in millimetres of distance).
    """
    wanted = [chart.x(bin_.mean_d_mm) for item in drawn for bin_ in item.bins]
    places = marks.spread_apart(wanted, RATES_APART_MM)
    for start, end in untested:
        low, high = chart.x(start), chart.x(end)
        # The marks that belong below the stretch stay below it, the nearest one first, each keeping its distance
        # from the one before; likewise above.
        for side, edge, away in ((lambda x: x <= low + 1e-9, low, -1.0), (lambda x: x >= high - 1e-9, high, 1.0)):
            limit = edge
            for index in sorted((i for i, x in enumerate(wanted) if side(x)), key=lambda i: away * places[i]):
                if away * (places[index] - limit) < 0.0:
                    places[index] = limit
                limit = places[index] + away * RATES_APART_MM
    moved = iter(places)
    return [[next(moved) for _ in item.bins] for item in drawn]


def _draw_rates(chart: RateChart, drawn: Sequence[Shown], untested: Sequence[tuple[float, float]]) -> None:
    """The success rate of each stretch of distance with its Wilson interval."""
    axes = chart.page.axes
    for item, places in zip(drawn, rate_places(chart, drawn, untested), strict=True):
        series = item.series
        for bin_, x in zip(item.bins, places, strict=True):
            axes.add_line(Line2D([x, x], [chart.y(bin_.wilson_lo), chart.y(bin_.wilson_hi)], color=series.color,
                                 linewidth=style.INTERVAL_PT, solid_capstyle="butt", zorder=3.0,
                                 gid=f"rate-interval:{item.name}"))  # fmt: skip
            axes.add_line(Line2D([x], [chart.y(bin_.rate)], linestyle="none", marker=series.marker, zorder=3.2,
                                 gid=f"rate:{item.name}", **mark_look(series, RATE_MARK_PT)))  # fmt: skip


def _name_rows(chart: RateChart, taken: Sequence[tuple[float, float]]) -> None:
    """Say what the bars in the rows are, in the free place nearest the right end of the rows; nowhere when the bars
    leave no room. `taken` holds the stretches of page x the bars cover.
    """
    if not chart.row_y_mm:
        return
    width = style.text_width_mm(ROWS_NAME)
    left = marks.free_slot(taken, width, low_mm=chart.left_mm + 1.0, high_mm=chart.right_mm,
                           clear_mm=ROWS_NAME_CLEAR_MM)  # fmt: skip
    if left is not None:
        baseline = style.centred_baseline_mm((chart.row_y_mm[0] + chart.row_y_mm[-1]) / 2.0)
        chart.page.text(left, baseline, ROWS_NAME, color=style.INK_SECONDARY, gid="rows-name")


def _draw_radius(chart: RateChart, item: Shown, y: float, longer_than_mm: float) -> tuple[float, float]:
    """A condition's row above the distance axis: its name in the margin, and its tolerance radius as a bar over
    the interval. A side the trials leave open ends in an arrow at that end of the axis.

    Returns the stretch of page x the bar covers.
    """
    page, axes, series, fit = chart.page, chart.page.axes, item.series, item.fit
    # The name, with the condition's mark before it: this is where a reader learns which marks are whose.
    name_left = chart.margin_x_mm - style.text_width_mm(item.name)
    baseline = style.centred_baseline_mm(y) + 0.25
    page.text(chart.margin_x_mm, baseline, item.name, ha="right", gid=f"row-name:{item.name}")
    axes.add_line(Line2D([name_left - 1.3], [y], linestyle="none", marker=series.marker, zorder=6.0,
                         gid=f"row-mark:{item.name}", **mark_look(series, chart.mark_pt + 0.4)))  # fmt: skip
    start = chart.left_mm if fit.open_below else chart.x(fit.r50_lo_mm)
    end = chart.right_mm if fit.open_above else chart.x(fit.r50_hi_mm)
    # The bar stops short of an open end, where an arrowhead takes over and points off the axis.
    bar = [start + (ARROW_MM if fit.open_below else 0.0), end - (ARROW_MM if fit.open_above else 0.0)]
    axes.add_line(Line2D(bar, [y, y], color=series.color, linewidth=RADIUS_BAR_PT, solid_capstyle="butt",
                         zorder=3.0, gid=f"radius-interval:{item.name}"))  # fmt: skip
    for x, is_open, inward in ((start, fit.open_below, 1.0), (end, fit.open_above, -1.0)):
        if is_open:
            back = x + inward * ARROW_MM
            axes.add_patch(Polygon([(x, y), (back, y + END_MM), (back, y - END_MM)], closed=True,
                                   facecolor=series.color, edgecolor="none", zorder=3.0,
                                   gid=f"radius-open:{item.name}"))  # fmt: skip
        else:
            axes.add_line(Line2D([x, x], [y - END_MM, y + END_MM], color=series.color, linewidth=0.9, zorder=3.0,
                                 solid_capstyle="butt", gid=f"radius-end:{item.name}"))  # fmt: skip
    if not fit.bounded:
        return start, end
    x_median = chart.x(fit.r50_median_mm)
    # The line from the radius up to the curve, where the curve is drawn there.
    pieces = marks.curve_pieces([trial.d_mm for trial in item.trials], longer_than_mm=longer_than_mm)
    if any(low <= fit.r50_median_mm <= high for low, high in pieces):
        drop = Line2D([x_median, x_median], [y, chart.y(fit.mean_at(fit.r50_median_mm))], color=series.color,
                      linewidth=0.7, zorder=2.2, dash_capstyle="butt", solid_capstyle="butt",
                      gid=f"radius-line:{item.name}")  # fmt: skip
        if series.dashes is not None:
            drop.set_dashes(series.dashes)
        axes.add_line(drop)
    axes.add_line(Line2D([x_median], [y], linestyle="none", marker=series.marker, zorder=3.6,
                         gid=f"radius:{item.name}", **mark_look(series, RADIUS_MARK_PT)))  # fmt: skip
    return start, end
