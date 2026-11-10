"""The frame Figures 1 and 3 share: success against distance, with every trial shown as a mark.

From top to bottom the frame has

* the **success strip**: one mark for every trial that succeeded, at its distance, piled away from the rate axis
  where several fall at one distance;
* the **rate band**: an axis of success rate from 0 to 1, for rates of groups of trials and for fitted curves;
* the **failure strip**: one mark for every trial that failed, piled downwards;
* optional **rows** just above the distance axis, for what a figure reads off that axis (Figure 3 puts each
  condition's tolerance radius there);
* the **distance axis**.

A mark's place along the distance axis is its measured distance and is never changed; only its height in a strip is
chosen, so that no mark covers another (`marks.stack_heights`). The strips take the room their piles need and the rate
band takes what is left of the figure's fixed height.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from tolerance_radius.figures import marks, style
from tolerance_radius.figures.style import MM_PER_PT, FigureError, Page, Series

#: From the page's left edge to the rate axis, and from the right end of the distance axis to the page's right edge.
LEFT_MM = 10.2
RIGHT_MM = 1.4
#: From the page's bottom edge to the distance axis (room for its numbers and its label), and above the top strip.
AXIS_Y_MM = 7.3
TOP_MM = 0.5
#: How far inside the ends of the distance axis the distances 0 and `x_max_mm` lie, so that a mark at 0 is whole.
INSET_MM = 1.3
#: Clear space between the rate band and the nearest mark of a strip, between two lanes of a strip, and between the
#: failure strip and what is below it.
BAND_GAP_MM = 1.05
LANE_GAP_MM = 0.45
FOOT_GAP_MM = 0.6
#: The height of one row above the distance axis.
ROW_MM = 2.2
#: A trial's mark: its size, the white ring around it, and the room one mark takes in a pile. In points.
MARK_PT = 3.3
RING_PT = 0.45
PILE_PT = MARK_PT + 0.75
#: The marks are made smaller, down to this size, when their piles would leave the rate band less than its least
#: height.
SMALLEST_MARK_PT = 2.4
LEAST_BAND_MM = 13.0
#: An empty strip still takes this room, so that it can be seen to be empty.
EMPTY_STRIP_MM = 1.6
TICK_MM = 0.9
#: A note in a shaded stretch keeps this much of the stretch's width free; a stretch narrower than `UPRIGHT_NOTE_MM`
#: carries no note at all, not even an upright one.
NOTE_MARGIN_MM = 1.0
UPRIGHT_NOTE_MM = 3.2


@dataclass(frozen=True, slots=True)
class MarkGroup:
    """The trials drawn with one kind of mark: one condition in Figure 3, one group of cells in Figure 1."""

    series: Series
    distances_mm: tuple[float, ...]
    successes: tuple[bool, ...]
    #: Trials drawn with an open mark (white inside), by position; empty for none.
    open_marks: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if len(self.distances_mm) != len(self.successes):
            raise FigureError("a group of trials has one outcome for every distance")
        if self.open_marks and len(self.open_marks) != len(self.distances_mm):
            raise FigureError("open_marks, when given, says for every trial whether its mark is open")
        if any(not (d >= 0.0) for d in self.distances_mm):
            raise FigureError("a distance to the nearest demonstration is a number of 0 or more")

    def is_open(self, index: int) -> bool:
        """Whether a trial's mark is drawn open."""
        return bool(self.open_marks) and self.open_marks[index]


@dataclass(slots=True)
class RateChart:
    """The frame, laid out on a page for some groups of trials. See the module's notes for its parts.

    `separate_lanes` gives every group a lane of its own in each strip, the first group next to the rate band; without
    it the groups share one pile. `rows` is the number of rows kept free just above the distance axis.
    """

    page: Page
    groups: Sequence[MarkGroup]
    x_max_mm: float
    x_label: str
    separate_lanes: bool = True
    rows: int = 0
    tick_step_mm: float = 10.0
    #: The marks are made smaller, as far as they may be, until the rate band is at least this high.
    least_band_mm: float = LEAST_BAND_MM

    mark_pt: float = field(init=False)
    #: Page y of the bottom and the top of the rate band (rate 0 and rate 1).
    band_low_mm: float = field(init=False)
    band_high_mm: float = field(init=False)
    #: Page y of the centre of each row above the axis, the top row first.
    row_y_mm: list[float] = field(init=False)
    #: Where every trial's mark was drawn: `(group index, trial index) -> (page x, page y)`.
    mark_places: dict[tuple[int, int], tuple[float, float]] = field(init=False)
    #: Page y of the outer edge of each strip: the top of the success strip, the bottom of the failure strip.
    strip_edge_mm: dict[bool, float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.x_max_mm > 0.0:
            raise FigureError("the distance axis must reach beyond 0")
        farthest = max((d for group in self.groups for d in group.distances_mm), default=0.0)
        if farthest > self.x_max_mm:
            raise FigureError(f"a trial at {farthest:.1f} mm lies beyond the axis, which ends at {self.x_max_mm:g} mm")
        self._lay_out()
        self._draw_frame()
        self._draw_marks()

    # ------------------------------------------------------------------------------------------------------------
    # Places
    # ------------------------------------------------------------------------------------------------------------

    @property
    def left_mm(self) -> float:
        """Page x of the rate axis: the left end of the distance axis."""
        return LEFT_MM

    @property
    def right_mm(self) -> float:
        """Page x of the right end of the distance axis."""
        return self.page.width_mm - RIGHT_MM

    @property
    def top_mm(self) -> float:
        """Page y of the top of the frame."""
        return self.page.height_mm - TOP_MM

    @property
    def margin_x_mm(self) -> float:
        """Page x at which the texts in the left margin end: the numbers of the rate axis, the names of the strips."""
        return self.left_mm - TICK_MM - 0.7

    def x(self, distance_mm: float) -> float:
        """Page x of a distance."""
        scale = (self.right_mm - self.left_mm - 2.0 * INSET_MM) / self.x_max_mm
        return self.left_mm + INSET_MM + distance_mm * scale

    def y(self, rate: float) -> float:
        """Page y of a success rate between 0 and 1."""
        return self.band_low_mm + rate * (self.band_high_mm - self.band_low_mm)

    # ------------------------------------------------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------------------------------------------------

    def _piles(self, mark_pt: float) -> dict[bool, list[dict[tuple[int, int], float]]]:
        """For successes and for failures: the lanes, each a map from a trial to the height of its mark's centre
        above the lane's base, with marks of the given size.
        """
        pitch = (mark_pt + PILE_PT - MARK_PT) * MM_PER_PT
        piles: dict[bool, list[dict[tuple[int, int], float]]] = {}
        for outcome in (True, False):
            members = [
                [(g, t) for t, success in enumerate(group.successes) if success == outcome]
                for g, group in enumerate(self.groups)
            ]
            if not self.separate_lanes:
                members = [[key for lane in members for key in lane]]
            lanes = []
            for lane in members:
                heights = marks.stack_heights([self.x(self.groups[g].distances_mm[t]) for g, t in lane], pitch)
                lanes.append({key: pitch / 2.0 + height for key, height in zip(lane, heights, strict=True)})
            piles[outcome] = lanes
        return piles

    def _lay_out(self) -> None:
        rows_mm = self.rows * ROW_MM + (0.5 if self.rows else 0.0)
        self.row_y_mm = [AXIS_Y_MM + 0.5 + (self.rows - 0.5 - index) * ROW_MM for index in range(self.rows)]
        available = self.top_mm - AXIS_Y_MM - rows_mm

        def strip_mm(lanes: list[dict[tuple[int, int], float]], pitch: float) -> float:
            extents = [max(lane.values()) + pitch / 2.0 for lane in lanes if lane]
            return sum(extents) + LANE_GAP_MM * max(len(extents) - 1, 0) if extents else EMPTY_STRIP_MM

        mark_pt = MARK_PT
        while True:
            pitch = (mark_pt + PILE_PT - MARK_PT) * MM_PER_PT
            piles = self._piles(mark_pt)
            heights = {outcome: strip_mm(piles[outcome], pitch) for outcome in (True, False)}
            band = available - heights[True] - heights[False] - 2.0 * BAND_GAP_MM - FOOT_GAP_MM
            if band >= self.least_band_mm or mark_pt <= SMALLEST_MARK_PT:
                break
            mark_pt = max(SMALLEST_MARK_PT, mark_pt - 0.3)
        if band < LEAST_BAND_MM / 2.0:
            raise FigureError("the piles of marks leave no room for the rate band at this figure's height")
        self.mark_pt = mark_pt
        self.band_low_mm = AXIS_Y_MM + rows_mm + FOOT_GAP_MM + heights[False] + BAND_GAP_MM
        self.band_high_mm = self.band_low_mm + band
        self.strip_edge_mm = {
            True: self.band_high_mm + BAND_GAP_MM + heights[True],
            False: self.band_low_mm - BAND_GAP_MM - heights[False],
        }
        # Marks are piled away from the rate band: successes upwards, failures downwards, lane after lane.
        self.mark_places = {}
        for outcome, direction in ((True, 1.0), (False, -1.0)):
            base = (self.band_high_mm if outcome else self.band_low_mm) + direction * BAND_GAP_MM
            for lane in piles[outcome]:
                if not lane:
                    continue
                for (g, t), height in lane.items():
                    x = self.x(self.groups[g].distances_mm[t])
                    self.mark_places[(g, t)] = (x, base + direction * height)
                base += direction * (max(lane.values()) + pitch / 2.0 + LANE_GAP_MM)

    # ------------------------------------------------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------------------------------------------------

    def _draw_frame(self) -> None:
        page, axes = self.page, self.page.axes
        label_x = self.margin_x_mm
        # The levels of the rate axis, as hairlines across the frame, and the axis itself.
        for rate, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            y = self.y(rate)
            axes.add_line(Line2D([self.left_mm, self.right_mm], [y, y], color=style.GRIDLINE, zorder=1.0,
                                 linewidth=style.HAIRLINE_PT, gid=f"level:{label}"))  # fmt: skip
            axes.add_line(Line2D([self.left_mm - TICK_MM, self.left_mm], [y, y], color=style.AXIS, zorder=2.0,
                                 linewidth=style.AXIS_PT))  # fmt: skip
            page.text(label_x, style.centred_baseline_mm(y), label, ha="right", gid=f"rate-tick:{label}")
        axes.add_line(Line2D([self.left_mm, self.left_mm], [self.band_low_mm, self.band_high_mm], color=style.AXIS,
                             linewidth=style.AXIS_PT, zorder=2.0, gid="rate-axis"))  # fmt: skip
        middle = (self.band_low_mm + self.band_high_mm) / 2.0
        page.text(2.9, middle, "Success rate", size_pt=style.LABEL_PT, ha="center", rotation=90.0, gid="rate-label")
        # The strips are named where the rate axis's numbers are.
        for outcome, name in ((True, "success"), (False, "failure")):
            inner = (self.band_high_mm + BAND_GAP_MM) if outcome else (self.band_low_mm - BAND_GAP_MM)
            centre = (inner + self.strip_edge_mm[outcome]) / 2.0
            page.text(label_x, style.centred_baseline_mm(centre) + 0.25, name, ha="right", gid=f"strip:{name}")
        # The distance axis.
        axes.add_line(Line2D([self.left_mm, self.right_mm], [AXIS_Y_MM, AXIS_Y_MM], color=style.AXIS, zorder=2.0,
                             linewidth=style.AXIS_PT, gid="distance-axis"))  # fmt: skip
        steps = int(self.x_max_mm / self.tick_step_mm + 1e-9)
        for step in range(steps + 1):
            distance = step * self.tick_step_mm
            x = self.x(distance)
            axes.add_line(Line2D([x, x], [AXIS_Y_MM - TICK_MM, AXIS_Y_MM], color=style.AXIS, zorder=2.0,
                                 linewidth=style.AXIS_PT))  # fmt: skip
            page.text(x, AXIS_Y_MM - TICK_MM - 2.35, f"{distance:g}", ha="center", gid=f"distance-tick:{distance:g}")
        page.text((self.left_mm + self.right_mm) / 2.0, 0.95, self.x_label, size_pt=style.LABEL_PT, ha="center",
                  gid="distance-label")  # fmt: skip

    def _draw_marks(self) -> None:
        axes = self.page.axes
        for (g, t), (x, y) in self.mark_places.items():
            group = self.groups[g]
            series = group.series
            look = mark_look(series, self.mark_pt, open_mark=group.is_open(t))
            outcome = "success" if group.successes[t] else "failure"
            axes.add_line(Line2D([x], [y], linestyle="none", marker=series.marker, zorder=4.0,
                                 gid=f"trial:{series.label}:{outcome}", **look))  # fmt: skip

    def shade(self, start_mm: float, end_mm: float, lines: Sequence[str]) -> None:
        """Shade a stretch of distances over the whole height of the frame and write a note in the middle of the
        rate band: for a stretch in which nothing was tested.

        The note is written across the stretch when it fits. A note of one line that does not fit is written
        upright instead, when the stretch is wide enough for that and the band high enough; otherwise the stretch
        is shaded and carries no note.
        """
        low, high = self.x(start_mm), self.x(end_mm)
        self.page.axes.add_patch(Rectangle((low, AXIS_Y_MM), high - low, self.top_mm - AXIS_Y_MM, facecolor=style.SHADE,
                                           edgecolor="none", zorder=0.5, gid="untested-stretch"))  # fmt: skip
        if not lines:
            return
        centre, middle = (low + high) / 2.0, (self.band_low_mm + self.band_high_mm) / 2.0
        widest = max(style.text_width_mm(line) for line in lines)
        if widest <= high - low - NOTE_MARGIN_MM:
            pitch = 2.75
            top = middle + pitch * (len(lines) - 1) / 2.0
            for index, line in enumerate(lines):
                self.page.text(centre, style.centred_baseline_mm(top - index * pitch), line, ha="center",
                               gid="untested-note")  # fmt: skip
        elif len(lines) == 1 and high - low >= UPRIGHT_NOTE_MM and widest <= self.band_high_mm - self.band_low_mm - 1.0:
            # Turned a quarter, a line of text has its baseline at its right: half a letter's height right of centre.
            self.page.text(centre + (middle - style.centred_baseline_mm(middle)), middle, lines[0], ha="center",
                           rotation=90.0, gid="untested-note")  # fmt: skip


def mark_look(series: Series, size_pt: float, *, open_mark: bool = False) -> dict[str, object]:
    """How a trial's mark is drawn, as Matplotlib's marker properties: filled with the series' colour inside a white
    ring that keeps it apart from its neighbours, or open (white inside a ring of the colour).
    """
    if open_mark:
        return {"markerfacecolor": "white", "markeredgecolor": series.color, "markeredgewidth": 0.9,
                "markersize": style.marker_size(series.marker, size_pt - 0.4)}  # fmt: skip
    return {"markerfacecolor": series.color, "markeredgecolor": "white", "markeredgewidth": RING_PT,
            "markersize": style.marker_size(series.marker, size_pt + RING_PT)}  # fmt: skip
