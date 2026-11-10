"""The grid on the table as a picture on the page: Figures 2 and 4 are both two such pictures side by side.

A `Panel` says where one picture of the grid sits and at what scale, and turns table millimetres into page
millimetres. `side_by_side` lays out pictures of equal scale as large as the page allows; `draw_grid` draws the
cells, hatches the held-out ones, and names the columns along the top edge and the rows along the left edge. Both
can leave the names out, for a figure that needs the room and never speaks of a cell by its name.

x runs to the right along the columns A, B, ... and y runs up along the rows 1, 2, ..., as in the table frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D, Transform

from tolerance_radius.figures import marks, style
from tolerance_radius.figures.style import FigureError, Page
from tolerance_radius.geometry import Grid, Rect

#: Room at the left of a picture for the numbers of the rows, and above it for the letters of the columns.
NUMBERS_MM = 2.2
LETTERS_MM = 2.5
#: Room above the letters for a picture's title.
TITLE_MM = 3.0
#: The lines of the hatching on a held-out cell are this far apart on the table.
HATCH_SPACING_MM = 6.5
#: From the page's left and right edges to the outermost pictures.
SIDE_MM = 0.4


@dataclass(frozen=True, slots=True)
class Panel:
    """One picture of the grid on the page: where the table frame's origin is, and the scale."""

    #: Page position of the table frame's origin, the outer corner of cell A1.
    x0_mm: float
    y0_mm: float
    #: Page millimetres for one table millimetre.
    scale: float
    grid: Grid

    def x(self, table_x_mm: float) -> float:
        """Page x of a table x."""
        return self.x0_mm + self.scale * table_x_mm

    def y(self, table_y_mm: float) -> float:
        """Page y of a table y."""
        return self.y0_mm + self.scale * table_y_mm

    def length(self, table_mm: float) -> float:
        """A length on the table as a length on the page."""
        return self.scale * table_mm

    @property
    def right_mm(self) -> float:
        """Page x of the grid's right edge."""
        return self.x(self.grid.width_mm)

    @property
    def top_mm(self) -> float:
        """Page y of the grid's top edge."""
        return self.y(self.grid.height_mm)

    def transform(self, page: Page) -> Transform:
        """For Matplotlib: draws an artist given in table millimetres at its place in this picture."""
        return Affine2D().scale(self.scale).translate(self.x0_mm, self.y0_mm) + page.axes.transData

    def outline(self, page: Page) -> Rectangle:
        """The grid's outline as an invisible patch, for cutting off what reaches beyond the grid."""
        grid = self.grid
        return Rectangle((0.0, 0.0), grid.width_mm, grid.height_mm, transform=self.transform(page))


def side_by_side(
    page: Page,
    grid: Grid,
    *,
    count: int = 2,
    top_mm: float,
    bottom_mm: float,
    gap_mm: float = 1.5,
    gap_table_mm: float = 0.0,
    names: bool = True,
) -> list[Panel]:
    """Pictures of the grid next to each other at one scale, as large as the room allows.

    The pictures, with the room for their titles and (unless `names` is false) for the letters of their columns and
    the numbers of their rows, go between the page's sides and between `bottom_mm` and `top_mm`. Two neighbours are
    `gap_mm` apart plus the length of `gap_table_mm` on the table at the pictures' own scale (room for something
    drawn to scale between them). Whatever width is left over is shared out between the gaps and the sides.
    """
    if count < 1:
        raise FigureError("at least one picture of the grid is needed")
    numbers_mm, letters_mm = (NUMBERS_MM, LETTERS_MM) if names else (0.0, 0.0)
    fixed = 2.0 * SIDE_MM + count * numbers_mm + (count - 1) * gap_mm
    by_width = (page.width_mm - fixed) / (count * grid.width_mm + (count - 1) * gap_table_mm)
    by_height = (top_mm - bottom_mm - TITLE_MM - letters_mm) / grid.height_mm
    scale = min(by_width, by_height)
    if not scale > 0.0:
        raise FigureError("there is no room on the page for the pictures of the grid")
    gap = gap_mm + gap_table_mm * scale
    used = fixed + count * grid.width_mm * scale + (count - 1) * gap_table_mm * scale
    spare = (page.width_mm - used) / (count + 1)
    y0 = top_mm - TITLE_MM - letters_mm - grid.height_mm * scale
    panels = []
    x = SIDE_MM + spare
    for _ in range(count):
        panels.append(Panel(x + numbers_mm, y0, scale, grid))
        x += numbers_mm + grid.width_mm * scale + gap + spare
    return panels


def draw_grid(
    page: Page, panel: Panel, *, title: str | None = None, note: str | None = None, names: bool = True
) -> None:
    """Draw the grid: its cells, the hatching of the held-out cells and, unless `names` is false, the letters of the
    columns along the top and the numbers of the rows along the left. A title goes above the grid (and its letters)
    at the left, a note at the right.
    """
    grid, axes = panel.grid, page.axes
    # Held-out cells: hatched, under everything else that is drawn on the grid.
    segments = []
    for cell in grid.held_out_cells:
        for (xa, ya), (xb, yb) in marks.hatch_segments(grid.cell_rect(cell), HATCH_SPACING_MM):
            segments.append([(panel.x(xa), panel.y(ya)), (panel.x(xb), panel.y(yb))])
    axes.add_collection(LineCollection(segments, colors=style.HATCH, linewidths=style.HAIRLINE_PT, zorder=1.0,
                                       capstyle="butt", gid="held-out-hatching"))  # fmt: skip
    # The borders between the cells, then the grid's outline over their ends.
    for column in range(1, grid.cols):
        x = panel.x(column * grid.cell_mm)
        axes.add_line(Line2D([x, x], [panel.y0_mm, panel.top_mm], color=style.GRIDLINE, zorder=1.5,
                             linewidth=style.HAIRLINE_PT, gid="cell-border"))  # fmt: skip
    for row in range(1, grid.rows):
        y = panel.y(row * grid.cell_mm)
        axes.add_line(Line2D([panel.x0_mm, panel.right_mm], [y, y], color=style.GRIDLINE, zorder=1.5,
                             linewidth=style.HAIRLINE_PT, gid="cell-border"))  # fmt: skip
    axes.add_patch(Rectangle((panel.x0_mm, panel.y0_mm), panel.right_mm - panel.x0_mm, panel.top_mm - panel.y0_mm,
                             facecolor="none", edgecolor=style.AXIS, linewidth=style.AXIS_PT, zorder=1.6,
                             gid="grid-outline"))  # fmt: skip
    # Names of the columns and rows, each at the middle of its cells.
    for column in range(grid.cols if names else 0):
        x = panel.x((column + 0.5) * grid.cell_mm)
        page.text(x, panel.top_mm + 0.75, chr(ord("A") + column), ha="center", gid="column-name")
    for row in range(grid.rows if names else 0):
        y = panel.y((row + 0.5) * grid.cell_mm)
        page.text(panel.x0_mm - 0.7, style.centred_baseline_mm(y), str(row + 1), ha="right", gid="row-name")
    title_y = panel.top_mm + (LETTERS_MM if names else 0.0) + 0.75
    if title is not None:
        page.text(panel.x0_mm, title_y, title, size_pt=style.LABEL_PT, bold=True, gid="panel-title")
    if note is not None:
        page.text(panel.right_mm, title_y, note, ha="right", gid="panel-note")


def cell_rects(panel: Panel, cells: Sequence[str]) -> list[Rect]:
    """The squares some cells cover on the page."""
    rects = []
    for cell in cells:
        rect = panel.grid.cell_rect(cell)
        low, high = (panel.x(rect.x_min_mm), panel.y(rect.y_min_mm)), (panel.x(rect.x_max_mm), panel.y(rect.y_max_mm))
        rects.append(Rect(*low, *high))
    return rects


def draw_scale_bar(page: Page, panel: Panel, table_mm: float, *, x_table_mm: float, y_mm: float) -> None:
    """A bar that is `table_mm` long on the table, starting at a table x, at a page height, with its length written
    after it.
    """
    start, end = panel.x(x_table_mm), panel.x(x_table_mm + table_mm)
    axes = page.axes
    axes.add_line(Line2D([start, end], [y_mm, y_mm], color=style.INK, linewidth=0.8, solid_capstyle="butt",
                         zorder=3.0, gid="scale-bar"))  # fmt: skip
    for x in (start, end):
        axes.add_line(Line2D([x, x], [y_mm - 0.6, y_mm + 0.6], color=style.INK, linewidth=0.8, zorder=3.0,
                             solid_capstyle="butt", gid="scale-bar-end"))  # fmt: skip
    page.text(end + 1.0, style.centred_baseline_mm(y_mm), f"{table_mm:g} mm", gid="scale-bar-label")
