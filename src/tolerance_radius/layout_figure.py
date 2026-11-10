"""A picture of the demonstration layouts on the grid, to look at before recording.

One panel for each layout, all in the same window and scale: the grid with its columns lettered and its rows numbered,
the held-out cells shaded, the test region framed, a dot at every planned start with the outline of the cube's
footprint around it, and the circle that the largest gap leaves empty.

This is the only module of the package that needs Matplotlib. It draws through Matplotlib's object interface, so it
opens no window and changes no global setting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from tolerance_radius.geometry import Grid, Rect
from tolerance_radius.layouts import LayoutQuality, PlannedEpisode, layout_quality

# One colour for the data (the starts); everything else in greys that step back from it.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_HAIRLINE = "#d9d8d0"
_FRAME = "#a9a79e"
_SHADE = "#eeede8"
_START = "#2a78d6"
_TEXT = {"fontfamily": ["Arial", "Helvetica Neue", "DejaVu Sans"]}

_TITLE = "Planned start positions of the demonstrations: the same episodes, laid out in different ways"
_FOOTNOTE = (
    "Dot: where the centre of the cube starts. Faint square: the cube's footprint there. "
    "Grey cell: held out, no demonstration starts in it.\n"
    "Inner frame: the test region, inside the margin at the grid's edge. "
    "Circle: the largest gap, around the point of the trained cells farthest from every start."
)


def layout_preview(layouts: Mapping[str, Sequence[PlannedEpisode]], grid: Grid) -> Figure:
    """A figure with one panel for each layout, in the order given. The keys name the panels."""
    qualities = {name: layout_quality(episodes, grid) for name, episodes in layouts.items()}
    window = _window(grid, qualities.values())
    figure = Figure(figsize=(6.3 * len(layouts), 5.0), facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, len(layouts), squeeze=False)[0]
    for panel, (name, episodes) in zip(panels, layouts.items(), strict=True):
        _draw_panel(panel, name, episodes, qualities[name], grid, window)
    figure.subplots_adjust(left=0.085 / len(layouts) + 0.01, right=0.985, top=0.8, bottom=0.2, wspace=0.14)
    figure.text(0.015, 0.955, _TITLE, fontsize=12, color=_INK, fontweight="bold", va="center", **_TEXT)
    figure.text(0.015, 0.055, _FOOTNOTE, fontsize=8.5, color=_INK_SECONDARY, va="center", linespacing=1.5, **_TEXT)
    return figure


def save_layout_preview(path: str | Path, layouts: Mapping[str, Sequence[PlannedEpisode]], grid: Grid) -> None:
    """Draw the layouts (see `layout_preview`) and save the picture as a PNG, making its folder if need be.

    The file carries no note of the drawing library's version, so the same picture drawn again is the same file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = layout_preview(layouts, grid)
    figure.savefig(path, format="png", dpi=160, facecolor=_SURFACE, metadata={"Software": None})


def _window(grid: Grid, qualities: Sequence[LayoutQuality]) -> Rect:
    """The part of the table every panel shows: the grid, the cubes that overhang it, and every largest-gap circle."""
    reach = grid.cube_edge_mm / 2.0
    for quality in qualities:
        where, gap = quality.largest_gap_at, quality.largest_gap_mm
        reach = max(reach, gap - where.x_mm, where.x_mm + gap - grid.width_mm)
        reach = max(reach, gap - where.y_mm, where.y_mm + gap - grid.height_mm)
    pad = reach + 0.06 * grid.cell_mm
    return Rect(-pad, -pad, grid.width_mm + pad, grid.height_mm + pad)


def _draw_panel(
    panel: Axes, name: str, episodes: Sequence[PlannedEpisode], quality: LayoutQuality, grid: Grid, window: Rect
) -> None:
    starts = sorted({(episode.target.x_mm, episode.target.y_mm) for episode in episodes})
    half = grid.cube_edge_mm / 2.0

    # The table: held-out cells shaded and named as such, every cell outlined, the grid and the test region framed.
    inset = 0.07 * grid.cell_mm
    for cell in grid.labels:
        rect = grid.cell_rect(cell)
        if not grid.is_trained(cell):
            panel.add_patch(_frame(rect, f"held-out:{cell}", facecolor=_SHADE, edgecolor="none", zorder=0.5))
            corner = (rect.x_min_mm + inset, rect.y_max_mm - inset)
            panel.text(*corner, "held out", ha="left", va="top", fontsize=8, color=_INK_MUTED, **_TEXT)
        panel.add_patch(_frame(rect, f"cell:{cell}", edgecolor=_HAIRLINE, zorder=1))
    panel.add_patch(_frame(grid.outline, "grid", edgecolor=_FRAME, zorder=1.5))
    panel.add_patch(_frame(grid.test_region, "test-region", edgecolor=_INK_MUTED, zorder=1.5))

    # The starts: the cube's footprint, faint, and a dot at its centre with a ring of the background around it.
    for x, y in starts:
        footprint = Rectangle((x - half, y - half), 2 * half, 2 * half, facecolor="none", edgecolor=_START, alpha=0.3)
        footprint.set(linewidth=0.7, zorder=2, gid="footprint")
        panel.add_patch(footprint)
    xs, ys = [x for x, _ in starts], [y for _, y in starts]
    panel.scatter(xs, ys, s=30, color=_START, edgecolors=_SURFACE, linewidths=1.0, zorder=3)

    # The largest gap: the empty circle around the point farthest from every start.
    where = (quality.largest_gap_at.x_mm, quality.largest_gap_at.y_mm)
    circle = Circle(where, quality.largest_gap_mm, facecolor="none", edgecolor=_INK_SECONDARY, linewidth=0.9)
    circle.set(zorder=2.5, gid="largest-gap")
    panel.add_patch(circle)
    panel.plot(*where, marker="+", markersize=6, color=_INK_SECONDARY, markeredgewidth=0.9, zorder=2.5)

    facts = [f"{quality.episodes} episodes at {quality.distinct_positions} positions"]
    if quality.min_spacing_mm > 0:
        facts.append(f"closest two {quality.min_spacing_mm:.1f} mm apart")
    facts.append(f"largest gap {quality.largest_gap_mm:.1f} mm")
    title = f"{name}\n{', '.join(facts)}"
    panel.set_title(title, loc="left", fontsize=10.5, color=_INK, linespacing=1.4, pad=22, **_TEXT)

    # Millimetres along the bottom and the left; the letters of the columns along the top, the numbers of the rows
    # along the right, each at the middle of its cells.
    panel.set_aspect("equal")
    panel.set_xlim(window.x_min_mm, window.x_max_mm)
    panel.set_ylim(window.y_min_mm, window.y_max_mm)
    panel.set_xticks([column * grid.cell_mm for column in range(grid.cols + 1)])
    panel.set_yticks([row * grid.cell_mm for row in range(grid.rows + 1)])
    panel.tick_params(colors=_INK_MUTED, labelsize=8, length=3, width=0.8, labelfontfamily=_TEXT["fontfamily"])
    panel.set_xlabel("x (mm)", fontsize=8.5, color=_INK_SECONDARY, **_TEXT)
    panel.set_ylabel("y (mm)", fontsize=8.5, color=_INK_SECONDARY, **_TEXT)
    panel.set_facecolor(_SURFACE)
    columns = panel.secondary_xaxis("top")
    columns.set_xticks([(column + 0.5) * grid.cell_mm for column in range(grid.cols)])
    columns.set_xticklabels([chr(ord("A") + column) for column in range(grid.cols)])
    rows = panel.secondary_yaxis("right")
    rows.set_yticks([(row + 0.5) * grid.cell_mm for row in range(grid.rows)])
    rows.set_yticklabels([str(row + 1) for row in range(grid.rows)])
    for edge in (columns, rows):
        edge.tick_params(colors=_INK_SECONDARY, labelsize=9, length=0, labelfontfamily=_TEXT["fontfamily"])
    for axes in (panel, columns, rows):
        for spine in axes.spines.values():
            spine.set_visible(False)


def _frame(rect: Rect, gid: str, **style: object) -> Rectangle:
    """A rectangle of the table as a patch: an outline unless a face colour is given."""
    style = {"facecolor": "none", "linewidth": 0.8} | style
    return Rectangle((rect.x_min_mm, rect.y_min_mm), rect.width_mm, rect.height_mm, gid=gid, **style)
