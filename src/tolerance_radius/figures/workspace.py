"""Figure 4: the workspace test and the lookup prediction.

The grid twice, one picture for each policy. In each picture:

* that policy's demonstrations, as small dots where they were measured;
* a disc around every demonstration, of the clustered policy's tolerance radius (its block fit, which has not seen
  the workspace trials). The discs of one picture are drawn as one area, with one outline;
* the placements of the workspace test, the same for both policies: a filled mark where the policy succeeded, an
  open mark where it failed. A mark in a hatched, held-out cell is drawn over the hatching, with a white ring.

Below each picture, the counts: how many placements the policy succeeded and failed at, and how many successes the
lookup prediction expects, with the interval of the predicted count. Where the placements are counted by region
(trained cells, held-out cells), each region has a line of its own, with its expected count and, when the file gives
one, the interval for that region; this as long as it leaves the pictures a useful size and the lines fit, otherwise
the totals stand alone and the page's notes say so. A last line across the figure says what the marks and the discs
are.

The discs are a picture of the prediction, not the prediction. The prediction is the fitted curve at each
placement's distance; the discs show where that curve passes its half.

When the trials do not bound the radius
---------------------------------------
The protocol then reports "at least" or "at most" and no radius, and the discs follow it. With an interval open
above they have the radius of its lower end, the smallest radius the trials allow: what lies inside them is inside
whatever the radius is. With an interval open below they have the radius of its upper end, the largest the trials
allow: what lies outside them is outside whatever the radius is. Their outline is then dashed and the last line says
which of the two is drawn. With an interval open on both sides no disc is drawn, and the line says so.

The figure names no cell: the letters and numbers of Figure 2 are left out here, for the room, and because the
paper never speaks of a cell by its name.

Positions and distances
-----------------------
The file's positions are as the instrument reads them, while its distances may have been divided by the
instrument's scale factor (`protocol/data-contracts.md`). On the page a disc therefore has the radius times that
divisor, so that a placement drawn inside a disc is one whose distance, as the analysis used it, is below the
radius. The figure checks this placement by placement and adds a note for each one that disagrees (`notes` of the
page): the caller prints them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from tolerance_radius.figures import HEIGHTS_MM, gridmap, results, style
from tolerance_radius.figures.gridmap import Panel
from tolerance_radius.figures.results import Count, Demonstration, Fit, Lookup, Placement
from tolerance_radius.figures.style import FigureError, Page, Series
from tolerance_radius.geometry import Grid, nearest_distances_xy

#: The two policies, left and right, with the title of each picture.
POLICIES: tuple[tuple[str, Series, str], ...] = (
    ("act_clustered", style.CLUSTERED, "Clustered policy"),
    ("act_spread", style.SPREAD, "Spread policy"),
)
#: The condition whose block fit gives the radius of the discs, in both pictures.
RADIUS_FROM = "act_clustered"
#: What the radius of the discs is: the fitted radius, or an end of its interval when the other end is open.
RADIUS, SMALLEST, LARGEST = "radius", "smallest", "largest"
#: What the last line of the figure says the discs are, and what it says when the trials allow none to be drawn.
DISC_NAMES = {
    RADIUS: "within the radius of a demonstration",
    SMALLEST: "within the smallest radius the trials allow",
    LARGEST: "within the largest radius the trials allow",
}
NO_DISC_NAME = "no disc: the trials do not bound the radius"
OUTCOME_NAMES = {True: "success", False: "failure"}
#: The names of the lines of counts below a picture.
ROW_NAMES = {"all": "all", "trained_cells": "trained", "held_out_cells": "held-out"}
HEADINGS = ("cells", "observed", "predicted")
#: Sizes in points: a placement's mark, a demonstration's dot.
PLACEMENT_PT = 3.5
DEMO_PT = 1.9
#: The height of a line of counts, the clear space above the first one, and the space between two columns.
LINE_MM = 2.65
ABOVE_COUNTS_MM = 0.9
COLUMN_GAP_MM = 2.0
#: The clear space above the last line (the key), and between two of its entries.
ABOVE_KEY_MM = 0.5
KEY_GAP_MM = 3.2
#: In a line of text, a small mark stands this far from where its entry begins, and its number or word this far.
MARK_AT_MM = 0.9
AFTER_MARK_MM = 2.25
TOP_MM = 0.3
BOTTOM_MM = 0.4
#: With the counts by region below them, the pictures must still be this wide; otherwise only the totals are shown.
LEAST_PANEL_MM = 32.0
#: The outline of the discs, in points: a line, and the dashes of one that marks an end of an interval.
OUTLINE_PT = 0.55
OUTLINE_DASHES = (2.2, 1.4)
#: The step of the lattice on which the outline of the discs is found, in table millimetres.
OUTLINE_STEP_MM = 1.0
#: A placement's distance in the file may differ from the distance between the positions drawn by this much before
#: the figure says so: more when only the pair's shared position is known, which is the mean of two placements.
OWN_POSITION_TOLERANCE_MM = 0.5
PAIR_POSITION_TOLERANCE_MM = 4.5


@dataclass(frozen=True, slots=True)
class Discs:
    """The discs drawn around the demonstrations."""

    #: Their radius, in the distances of the analysis.
    radius_mm: float
    #: `RADIUS` for the fitted radius; `SMALLEST` or `LARGEST` for the end of an interval whose other end is open.
    kind: str


def discs_of(fit: Fit | None) -> Discs | None:
    """The discs a fit allows to be drawn; None without a fit, and for an interval that is open on both sides."""
    if fit is None or (fit.open_below and fit.open_above):
        return None
    if fit.open_above:
        return Discs(fit.r50_lo_mm, SMALLEST)
    if fit.open_below:
        return Discs(fit.r50_hi_mm, LARGEST)
    return Discs(fit.r50_median_mm, RADIUS)


def count_rows(results_: Mapping[str, Any], policy: str) -> tuple[list[tuple[str, Count, Lookup | None]], list[str]]:
    """The lines of counts below a policy's picture: all placements, then each region when the placements lie in
    more than one. Each line is its name, the successes among the placements with what the lookup prediction
    expects, and for the line of all placements the prediction with its interval.

    The second value holds what the reader of the results found to say of the counts by region.
    """
    rows = [("all", results.success_count(results_, policy), results.lookup(results_, policy))]
    regions, notes = results.region_counts(results_, policy)
    if regions and len(regions) > 1:
        rows += [(region, count, None) for region, count in regions.items()]
    return rows, notes


def figure(results_: Mapping[str, Any], grid: Grid) -> Page:
    """Draw Figure 4 from the analysis results and the grid."""
    results.check_header(results_)
    if not results.has_workspace(results_):
        raise FigureError("the results hold no workspace test: there is nothing to draw")
    fit = results.fit(results_, RADIUS_FROM, "block")
    discs, divisor = discs_of(fit), results.distance_divisor(results_)
    rows, placed, demos, notes = {}, {}, {}, []
    for policy, _, _ in POLICIES:
        notes += results.pair_checks(results_, policy)
        rows[policy], said = count_rows(results_, policy)
        notes += said
        placed[policy] = results.placements(results_, policy)
        dataset = results.DATASET_OF_POLICY[policy]
        demos[policy] = [demo for demo in results.demonstrations(results_, dataset) if demo.trained]
        if not demos[policy]:
            raise FigureError(f"the results hold no demonstration the {dataset} policy was trained on")
        for where in (*placed[policy], *demos[policy]):
            if not (0.0 <= where.x_mm <= grid.width_mm and 0.0 <= where.y_mm <= grid.height_mm):
                raise FigureError(f"a position of {policy} ({where.x_mm:.1f}, {where.y_mm:.1f}) is off the grid")
    with style.settings():
        page = Page(HEIGHTS_MM[4])
        panels = _lay_out(page, grid, rows)
        if not _fits(page, panels, rows) and any(len(found) > 1 for found in rows.values()):
            # The counts by region take more room than the figure has: the totals stay.
            rows = {policy: found[:1] for policy, found in rows.items()}
            panels = _lay_out(page, grid, rows)
            notes.append("the counts by region do not fit below the pictures at this size: only the totals are shown")
        if not _fits(page, panels, rows):
            raise FigureError("the counts of the workspace test do not fit below the pictures")
        key_y = BOTTOM_MM + LINE_MM / 2.0
        for panel, (policy, series, title) in zip(panels, POLICIES, strict=True):
            if discs is not None:
                _draw_discs(page, panel, demos[policy], discs, divisor, series)
            gridmap.draw_grid(page, panel, title=title, names=False)
            _draw_demonstrations(page, panel, demos[policy], series)
            _draw_placements(page, panel, placed[policy])
            _draw_counts(page, panel, rows[policy], panel.y0_mm - ABOVE_COUNTS_MM)
            notes += _disagreements(policy, placed[policy], demos[policy], discs, divisor)
        _draw_key(page, panels[0].x0_mm, key_y, discs, fit is not None)
        page.notes += list(dict.fromkeys(notes))  # a note that holds for both policies is said once
        if results.is_draft(results_):
            page.stamp("DRAFT")
    return page


def _lay_out(page: Page, grid: Grid, rows: Mapping[str, Sequence[tuple[str, Count, Lookup | None]]]) -> list[Panel]:
    """Where the two pictures go, with room below them for the lines of counts and for the last line."""
    lines = 1 + max(len(found) for found in rows.values())
    below = BOTTOM_MM + LINE_MM + ABOVE_KEY_MM + ABOVE_COUNTS_MM + lines * LINE_MM
    return gridmap.side_by_side(page, grid, top_mm=page.height_mm - TOP_MM, bottom_mm=below, names=False)


def _fits(page: Page, panels: Sequence[Panel], rows: Mapping[str, Sequence[tuple[str, Count, Lookup | None]]]) -> bool:
    """Whether the pictures are still of a useful size and every table of counts ends before the next picture's
    table begins, or before the page does.
    """
    if panels[0].right_mm - panels[0].x0_mm < LEAST_PANEL_MM:
        return False
    limits = [panel.x0_mm - 1.0 for panel in panels[1:]] + [page.width_mm - gridmap.SIDE_MM]
    return all(panel.x0_mm + _counts_columns(rows[policy])[-1] <= limit
               for panel, (policy, _, _), limit in zip(panels, POLICIES, limits, strict=True))  # fmt: skip


def _counts_columns(rows: Sequence[tuple[str, Count, Lookup | None]]) -> tuple[float, float, float, float]:
    """Where the columns of a table of counts begin, from the table's left edge: the names, what was observed, what
    is predicted; and, last, where the table ends.
    """
    names = [ROW_NAMES.get(name, name.replace("_", " ")) for name, _, _ in rows]
    digits = max(style.text_width_mm(str(max(count.k, count.n - count.k))) for _, count, _ in rows)
    observed_mm = 2.0 * (AFTER_MARK_MM + digits) + 1.6
    second = max(style.text_width_mm(text) for text in (HEADINGS[0], *names)) + COLUMN_GAP_MM
    third = second + max(observed_mm, style.text_width_mm(HEADINGS[1])) + COLUMN_GAP_MM
    predicted = [text for text in (_predicted_text(count, lookup) for _, count, lookup in rows) if text is not None]
    if not predicted:
        return 0.0, second, third, third - COLUMN_GAP_MM
    widest = max(style.text_width_mm(text) for text in predicted)
    heading = style.text_width_mm(HEADINGS[2]) + 1.5 + 1.0  # the word, and the mark after it
    return 0.0, second, third, third + max(widest, heading)


def _draw_discs(
    page: Page, panel: Panel, demos: Sequence[Demonstration], discs: Discs, divisor: float, series: Series
) -> None:
    """The discs around the demonstrations, as one tinted area with one outline, cut off at the grid's edge. The
    outline is dashed when the radius is an end of an interval and not the fitted radius.
    """
    axes, grid = page.axes, panel.grid
    radius_read_mm = discs.radius_mm * divisor  # positions are as read; the radius is in the analysis's millimetres
    if not radius_read_mm > 0.0:
        return
    clip = panel.outline(page)
    for demo in demos:
        disc = Circle((demo.x_mm, demo.y_mm), radius_read_mm, transform=panel.transform(page), facecolor=series.tint,
                      edgecolor="none", zorder=0.6, gid=f"disc:{series.label}")  # fmt: skip
        axes.add_patch(disc)
        disc.set_clip_path(clip)  # after it is added: adding a patch to the axes sets its clipping to the page
    # One outline around all the discs: the line on which the nearest demonstration is exactly a radius away.
    xs = np.arange(0.0, grid.width_mm + OUTLINE_STEP_MM / 2.0, OUTLINE_STEP_MM)
    ys = np.arange(0.0, grid.height_mm + OUTLINE_STEP_MM / 2.0, OUTLINE_STEP_MM)
    table_x, table_y = np.meshgrid(xs, ys)
    nearest = nearest_distances_xy(table_x, table_y, np.array([[demo.x_mm, demo.y_mm] for demo in demos]))
    if nearest.min() < radius_read_mm < nearest.max():
        outline = axes.contour(panel.x0_mm + panel.scale * table_x, panel.y0_mm + panel.scale * table_y, nearest,
                               levels=[radius_read_mm], colors=[series.color], linewidths=[OUTLINE_PT], zorder=1.2,
                               linestyles=["solid" if discs.kind == RADIUS else (0, OUTLINE_DASHES)])  # fmt: skip
        outline.set_gid(f"disc-outline:{series.label}:{discs.kind}")


def _draw_demonstrations(page: Page, panel: Panel, demos: Sequence[Demonstration], series: Series) -> None:
    for demo in demos:
        page.axes.add_line(Line2D([panel.x(demo.x_mm)], [panel.y(demo.y_mm)], linestyle="none", marker=series.marker,
                                  markersize=style.marker_size(series.marker, DEMO_PT + 0.25),
                                  markerfacecolor=series.color, markeredgecolor="white", markeredgewidth=0.25,
                                  zorder=3.0, gid=f"demonstration:{series.label}"))  # fmt: skip


def _placement_look(success: bool, size_pt: float = PLACEMENT_PT) -> dict[str, Any]:
    """A placement's mark: filled where the policy succeeded, open where it failed, in ink, with white around or
    inside it so that it stands clear of a tint, a hatching or another mark below it.
    """
    if success:
        return {"marker": "o", "markersize": size_pt + 0.5, "markerfacecolor": style.INK, "markeredgecolor": "white",
                "markeredgewidth": 0.5}  # fmt: skip
    return {"marker": "o", "markersize": size_pt - 0.3, "markerfacecolor": "white", "markeredgecolor": style.INK,
            "markeredgewidth": 0.8}  # fmt: skip


def _draw_placements(page: Page, panel: Panel, placed: Sequence[Placement]) -> None:
    for placement in placed:
        outcome = OUTCOME_NAMES[placement.success]
        page.axes.add_line(Line2D([panel.x(placement.x_mm)], [panel.y(placement.y_mm)], linestyle="none", zorder=5.0,
                                  gid=f"placement:{outcome}", **_placement_look(placement.success)))  # fmt: skip


def _draw_counts(page: Page, panel: Panel, rows: Sequence[tuple[str, Count, Lookup | None]], top_mm: float) -> None:
    """The counts below a picture, as a small table: which cells, what was observed, what the lookup predicts."""
    predicted = [_predicted_text(count, lookup) for _, count, lookup in rows]
    with_prediction = any(text is not None for text in predicted)
    names = [ROW_NAMES.get(name, name.replace("_", " ")) for name, _, _ in rows]
    digits = max(style.text_width_mm(str(max(count.k, count.n - count.k))) for _, count, _ in rows)
    first, second, third, _ = (panel.x0_mm + column for column in _counts_columns(rows))

    def baseline(line: int) -> float:
        return style.centred_baseline_mm(top_mm - (line + 0.5) * LINE_MM)

    page.text(first, baseline(0), HEADINGS[0], color=style.INK_SECONDARY, gid="counts-heading")
    page.text(second, baseline(0), HEADINGS[1], color=style.INK_SECONDARY, gid="counts-heading")
    if with_prediction:
        page.text(third, baseline(0), HEADINGS[2], color=style.INK_SECONDARY, gid="counts-heading")
        mark_x = third + style.text_width_mm(HEADINGS[2]) + 1.5
        page.axes.add_line(Line2D([mark_x], [top_mm - 0.5 * LINE_MM], linestyle="none", zorder=6.0,
                                  gid="counts-heading-mark", **_placement_look(True, 3.0)))  # fmt: skip
    lines = enumerate(zip(rows, names, predicted, strict=True), start=1)
    for line, ((name, count, _), shown_name, prediction) in lines:
        y = top_mm - (line + 0.5) * LINE_MM
        page.text(first, baseline(line), shown_name, gid=f"counts-name:{name}")
        x = second
        for success, number in ((True, count.k), (False, count.n - count.k)):
            page.axes.add_line(Line2D([x + MARK_AT_MM], [y], linestyle="none", zorder=6.0, gid=f"counts-mark:{name}",
                                      **_placement_look(success, 3.0)))  # fmt: skip
            page.text(x + AFTER_MARK_MM, baseline(line), str(number),
                      gid=f"counts-{OUTCOME_NAMES[success]}:{name}")  # fmt: skip
            x += AFTER_MARK_MM + digits + 1.6
        if prediction is not None:
            page.text(third, baseline(line), prediction, gid=f"counts-predicted:{name}")


def _predicted_text(count: Count, lookup: Lookup | None) -> str | None:
    """What the lookup prediction expects for a line of counts, as text; None when the file has no prediction."""
    if lookup is not None:
        return f"{lookup.expected:.1f} ({lookup.lo} to {lookup.hi})"
    if count.lookup_expected is None:
        return None
    if count.lookup_lo is not None and count.lookup_hi is not None:
        return f"{count.lookup_expected:.1f} ({count.lookup_lo} to {count.lookup_hi})"
    return f"{count.lookup_expected:.1f}"


def _draw_key(page: Page, left_mm: float, y_mm: float, discs: Discs | None, fitted: bool) -> None:
    """The last line of the figure: what a filled mark, an open mark and the tinted discs are.

    The discs have an entry when there is a fit to draw them from; it says which radius they have, or that the
    trials allow none to be drawn.
    """
    x, baseline = left_mm, style.centred_baseline_mm(y_mm)
    for success in (True, False):
        page.axes.add_line(Line2D([x + MARK_AT_MM], [y_mm], linestyle="none", zorder=6.0, gid="key-mark",
                                  **_placement_look(success, 3.0)))  # fmt: skip
        page.text(x + AFTER_MARK_MM, baseline, OUTCOME_NAMES[success], gid="key-text")
        x += AFTER_MARK_MM + style.text_width_mm(OUTCOME_NAMES[success]) + KEY_GAP_MM
    if not fitted:
        return
    if discs is None:
        page.text(x, baseline, NO_DISC_NAME, gid="key-text")
        return
    for _, series, _ in POLICIES:  # a swatch in each picture's tint, outlined as the discs are
        swatch = Rectangle((x, y_mm - 0.85), 2.4, 1.7, facecolor=series.tint, edgecolor=series.color, zorder=6.0,
                           linewidth=OUTLINE_PT, gid=f"key-disc:{discs.kind}")  # fmt: skip
        if discs.kind != RADIUS:
            swatch.set_linestyle((0, (1.6, 1.0)))
        page.axes.add_patch(swatch)
        x += 2.4 + 0.7
    page.text(x + 0.4, baseline, DISC_NAMES[discs.kind], gid="key-text")


def _disagreements(policy: str, placed: Sequence[Placement], demos: Sequence[Demonstration], discs: Discs | None,
                   divisor: float) -> list[str]:  # fmt: skip
    """Placements whose distance in the file is not the distance between the positions drawn, and placements drawn
    on the other side of the discs' edge than their distance says. Each as a sentence for the person who runs the
    figures; none when the picture and the numbers agree.
    """
    positions = np.array([[demo.x_mm, demo.y_mm] for demo in demos])
    notes = []
    for index, placement in enumerate(placed, start=1):
        tolerance = OWN_POSITION_TOLERANCE_MM if placement.own_position else PAIR_POSITION_TOLERANCE_MM
        drawn = float(nearest_distances_xy(placement.x_mm, placement.y_mm, positions)) / divisor
        if abs(drawn - placement.d_mm) > tolerance:
            notes.append(
                f"figure 4, {policy}, placement {index}: the file gives d = {placement.d_mm:.1f} mm, the positions "
                f"drawn are {drawn:.1f} mm apart"
            )
        elif discs is not None and (drawn <= discs.radius_mm) != (placement.d_mm <= discs.radius_mm):
            notes.append(
                f"figure 4, {policy}, placement {index}: drawn {'inside' if drawn <= discs.radius_mm else 'outside'}"
                f" a disc, while its distance in the file ({placement.d_mm:.1f} mm) is on the other side of the "
                f"discs' radius ({discs.radius_mm:.1f} mm)"
            )
    return notes
