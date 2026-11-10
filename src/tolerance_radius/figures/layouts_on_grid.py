"""Figure 2: the two layouts on the grid.

The grid twice, side by side: the start positions of the clustered set on the left, those of the spread set on the
right, each demonstration one dot where the camera measured the cube to be. Held-out cells are hatched, the columns
are lettered and the rows numbered, a bar gives the scale, and the cube's footprint is drawn once, to scale.

Starts that fall on one point are drawn as they were measured, on top of one another, and counted beside the point
("x5", with a multiplication sign): a dot's place is a measurement and is not moved to make room.

The cube's footprint is drawn beside the grids and not around a start. A start position is the centre of the cube's
visible outline as the camera maps it onto the table, which is several millimetres from the centre of its footprint,
so a square drawn around a dot would claim to know where the cube stood.

The data are the start-state audits (`data/audit/<dataset>.json`). The same function draws the layouts as planned,
for a look before recording; that picture carries the word PLANNED and is not the paper's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from tolerance_radius.figures import HEIGHTS_MM, gridmap, style
from tolerance_radius.figures.gridmap import Panel
from tolerance_radius.figures.style import FigureError, Page, Series
from tolerance_radius.geometry import Grid
from tolerance_radius.schema import DATASETS, DemoStart

TITLES = {"clustered": "Clustered", "spread": "Spread"}
SERIES: dict[str, Series] = {"clustered": style.CLUSTERED, "spread": style.SPREAD}
#: Two starts closer than this were placed on one point. The spread layout keeps its points at least 12 mm apart and
#: every start is set within 2 mm of its point, so starts of one point are within 4 mm of each other and starts of
#: two points are more than 8 mm apart.
SAME_POINT_MM = 6.0
#: The length of the scale bar on the table: one cell.
SCALE_BAR_MM = 50.0
#: A dot's size in points, and the white ring around it.
DOT_PT = 2.5
RING_PT = 0.3
#: From the lowest dot of a point down to the baseline of its count, and the sign the count is written with.
COUNT_BELOW_MM = 2.6
TIMES = "\N{MULTIPLICATION SIGN}"
#: Room below the grids for the scale bar and the names of what is drawn beside the grids.
FOOT_MM = 2.9
TOP_MM = 0.3
BOTTOM_MM = 0.3


def points_of(starts: Sequence[DemoStart], same_point_mm: float = SAME_POINT_MM) -> list[list[DemoStart]]:
    """The demonstrations sorted into the points they start at: starts closer than `same_point_mm`, directly or
    through a chain of others, are one point. The points come in the order of their first episode.
    """
    remaining = sorted(starts, key=lambda start: start.episode_index)
    points: list[list[DemoStart]] = []
    while remaining:
        point = [remaining.pop(0)]
        grown = True
        while grown:
            grown = False
            for other in list(remaining):
                if any(other.measured.distance_to(member.measured) < same_point_mm for member in point):
                    point.append(other)
                    remaining.remove(other)
                    grown = True
        points.append(point)
    return points


def summary(starts: Sequence[DemoStart]) -> str:
    """One layout in a few words: how many demonstrations were trained on, and at how many points they start."""
    trained = [start for start in starts if start.trained]
    points = len(points_of(trained))
    return f"{len(trained)} start{'s' if len(trained) != 1 else ''} at {points} point{'s' if points != 1 else ''}"


def figure(starts: Mapping[str, Sequence[DemoStart]], grid: Grid, *, stamp: str | None = None) -> Page:
    """Draw Figure 2 from the demonstrations' start positions, one list for each of the two sets.

    `stamp` writes a word across the figure: "PLANNED" when the positions are the plan and not the audit.
    """
    if set(starts) != set(DATASETS):
        raise FigureError(f"the figure needs the starts of both sets {DATASETS}, got {sorted(starts)}")
    for dataset, demos in starts.items():
        if not demos:
            raise FigureError(f"the {dataset} set has no demonstration to draw")
        wrong = sorted({demo.dataset for demo in demos} - {dataset})
        if wrong:
            raise FigureError(f"the starts given for the {dataset} set hold demonstrations of {wrong}")
        outside = [demo.episode_index for demo in demos if grid.cell_of(demo.measured) is None]
        if outside:
            raise FigureError(f"episodes {outside} of the {dataset} set start off the grid")
    with style.settings():
        page = Page(HEIGHTS_MM[2])
        left, right = gridmap.side_by_side(
            page,
            grid,
            top_mm=page.height_mm - TOP_MM,
            bottom_mm=BOTTOM_MM + FOOT_MM,
            gap_mm=2.0,
            gap_table_mm=grid.cube_edge_mm,
        )
        for panel, dataset in zip((left, right), DATASETS, strict=True):
            gridmap.draw_grid(page, panel, title=TITLES[dataset], note=summary(starts[dataset]))
            _draw_starts(page, panel, starts[dataset], SERIES[dataset])
        foot_y = left.y0_mm - 0.5 - FOOT_MM / 2.0
        gridmap.draw_scale_bar(page, left, SCALE_BAR_MM, x_table_mm=0.0, y_mm=foot_y)
        _draw_cube(page, left, right, foot_y)
        if any(not demo.trained for demos in starts.values() for demo in demos):
            x = right.x0_mm
            page.axes.add_line(Line2D([x + 0.8], [foot_y], linestyle="none", marker="o", markersize=DOT_PT,
                                      markerfacecolor="white", markeredgecolor=style.INK, markeredgewidth=0.6,
                                      zorder=5.0, gid="not-trained-key"))  # fmt: skip
            page.text(x + 2.0, style.centred_baseline_mm(foot_y), "recorded, not trained on", gid="not-trained-label")
        if stamp:
            page.stamp(stamp)
    return page


def _draw_starts(page: Page, panel: Panel, starts: Sequence[DemoStart], series: Series) -> None:
    """Every demonstration's measured start as a dot, and beside each point where several start, how many."""
    axes = page.axes
    for demo in starts:
        x, y = panel.x(demo.measured.x_mm), panel.y(demo.measured.y_mm)
        if demo.trained:
            look = {"markerfacecolor": series.color, "markeredgecolor": "white", "markeredgewidth": RING_PT,
                    "markersize": style.marker_size(series.marker, DOT_PT + RING_PT)}  # fmt: skip
        else:
            look = {"markerfacecolor": "white", "markeredgecolor": series.color, "markeredgewidth": 0.6,
                    "markersize": style.marker_size(series.marker, DOT_PT)}  # fmt: skip
        gid = f"start:{series.label}:{'trained' if demo.trained else 'not-trained'}"
        axes.add_line(Line2D([x], [y], linestyle="none", marker=series.marker, zorder=4.0, gid=gid, **look))
    for point in points_of([demo for demo in starts if demo.trained]):
        if len(point) < 2:
            continue
        x = sum(panel.x(demo.measured.x_mm) for demo in point) / len(point)
        lowest = min(panel.y(demo.measured.y_mm) for demo in point)
        page.text(x, lowest - COUNT_BELOW_MM, f"{TIMES}{len(point)}", ha="center", gid=f"count:{series.label}")


def _draw_cube(page: Page, left: Panel, right: Panel, foot_y: float) -> None:
    """The cube's footprint, to the scale of the grids, between them, with its name below."""
    edge = left.length(left.grid.cube_edge_mm)
    middle = (left.right_mm + right.x0_mm - gridmap.NUMBERS_MM) / 2.0
    page.axes.add_patch(Rectangle((middle - edge / 2.0, left.y0_mm), edge, edge, facecolor=style.SHADE,
                                  edgecolor=style.INK, linewidth=0.7, zorder=3.0, gid="cube-footprint"))  # fmt: skip
    page.text(middle, style.centred_baseline_mm(foot_y), "cube", ha="center", gid="cube-label")
