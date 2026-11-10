"""Registering the table: where the two floor tags are, in the grid's own millimetres.

The grid is never drawn on the work surface. Its place is fixed once, by seating a printed sheet where the grid should
be, and from then on by two tags taped to the floor. This module measures those two tags against the sheet.

The sheet (`scripts/make_registration_sheet.py`) carries six small tags at known places inside the grid. A run of
camera pictures that show the sheet and both floor tags is enough:

1. each sheet tag's outline is measured from the picture (`outline.py`): the tag detector's own corners sit about half
   a pixel inside the outline at the rig's resolution, which is half a millimetre there;
2. the sheet's outline corners, whose places are known in grid millimetres, give the conversion from pixels to the
   grid: the table frame;
3. the floor tags' outlines, measured the same way, are put through that conversion and stored: this is the table
   geometry. Every later session measures the floor tags by their outlines too. (At first the detector's own corners
   were stored and used. They sit a few tenths of a pixel off the true corners by an amount that depends on where
   the pixels fall, so a picture that had shifted by 0.1 px could put a solve from the two tags more than a
   millimetre from the frame: a watchman that cries out at nothing. Through outlines the same solve follows the
   picture's real shift to within about 0.3 mm. Both measured on synthetic pictures of the rig's view, 2026-10-06.)

The sheet is then taken away. While the camera stays where it is, a session that solves the conversion again from the
two floor tags gets the frame of step 2 back, whatever the floor tags' true size or flatness: the stored corners were
made through that very frame. So nothing a registration computes from its own pictures through the floor
tags can test the frame; an earlier version of this module printed such a figure as a check, and it could only repeat
the sheet's fit. What a registration can and does say about itself:

* **the sheet's fit**: how far the sheet's outline corners are from one flat plane seen through an ideal lens. A
  sheet that is not flat shows here: a millimetre of height moves what the camera sees by most of a millimetre;
* **leave one out**: each sheet tag in turn is left out of the fit and its corners are predicted from the others: how
  well the frame holds at a place that no tag pinned;
* **the sheet tags' size**: each tag's own size, read through the frame, against its printed size. A tag that reads
  short one way lies on a slope; all of them off together is an outline that is not being measured rightly (an
  over-exposed picture, a blur);
* **the floor tags' size** by their outline, read through the frame, against their caliper size. This is the only
  figure here that does not rest on the sheet alone, and so the only one that can see a sheet lying tilted as one
  rigid plane, or a sheet whose size was measured wrongly. It cannot say which is wrong, the sheet or the floor tag;
* **what one frame is worth**: the conversion solved from single frames' floor tags, as a session's check solves it,
  against the frame: the wander that the detector's noise causes over the grid. One pixel on one corner of a floor
  tag moves the far side of the grid by up to 2.5 mm (measured on the first sitting's geometry), which is why
  sessions measure through one stored conversion and not through a fresh solve.

The frame itself and the floor tags' pixel corners are kept with the result, so that the calibration made afterwards
can be held against them: a floor tag pressed flat, or a camera setting changed, between the registration and the
calibration would otherwise put the grid somewhere else without a word.

The independent check of the whole instrument is not here: it is the cube moved along a printed scale
(`protocol/protocol.md`, block V0). An independent review of the first sitting's frames measured the sheet's printed grid
lines, which no fit uses: they lay 0.35 mm (root mean square) and up to 1.2 mm from the frame, when the sheet's fit
read 0.35 mm on average and 0.79 at worst. That sheet was not flat, and its figures are where the first limits
below come from.

A printer rarely prints at exactly 100 % in both directions. The sheet's outline is therefore measured with calipers
and the registration is told its size (`sheet_scale_from_outline`): the sheet's tags are then taken to be where they
really are on the paper, and the table frame comes out in true millimetres.

The table frame: origin at the outer corner of cell A1, x along columns A to E, y along rows 1 to 3. Seated as the
sheet says, row 1 is nearest the arm and column A is on the left as seen from the open side of the light box.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from tolerance_radius.rig import lerobot101, outline

#: Width of a sheet tag's black border, as designed.
SHEET_TAG_MM = 36.0
#: The grid's outline on the sheet, as designed: its size between the centres of its lines, and the line's width
#: (0.6 point). Calipers on the outline's outer edges read the size plus one line width.
SHEET_OUTLINE_MM = (250.0, 150.0)
SHEET_OUTLINE_LINE_MM = 0.6 * 25.4 / 72.0
#: A measured outline further than this from its design is not this sheet printed at 100 %.
SHEET_SCALE_LIMIT = 0.02
#: The sheet's tags and where their centres are in the table frame: the centres of cells A1, C1, E1, A3, C3 and E3.
SHEET_TAGS: dict[int, tuple[float, float]] = {
    10: (25.0, 25.0),
    11: (125.0, 25.0),
    12: (225.0, 25.0),
    13: (25.0, 125.0),
    14: (125.0, 125.0),
    15: (225.0, 125.0),
}
#: The two tags that stay on the floor.
FLOOR_TAG_IDS = (0, 1)
FLOOR_TAG_NOMINAL_MM = 60.0
#: Every tag of the sheet must be in view with a clean outline. With four of the six the frame of the first sitting
#: moved by up to 1.9 mm (an independent review's trial, 2026-10-05), and a tag that cannot be measured is hidden, in
#: glare or bent: something to put right, not to leave out.
MIN_SHEET_TAGS = len(SHEET_TAGS)
#: A tag must be found in at least this share of the frames to be used.
MIN_SHARE = 0.5
#: The limits below are set by what the experiment needs of the table frame (`docs/decisions.md`, 2026-10-05): a
#: cube is accepted within 2 mm of its target, and the largest distance a staircase asks for is 50 mm. Each has two
#: steps: beyond the first the registration says so and is still stored (a NOTE), beyond the second it is refused.
#:
#: The sheet's outline corners against one flat plane through an ideal lens, mean and largest misfit. A flat sheet
#: gives about 0.2 and 0.45 mm on synthetic pictures of the rig's view. The first sitting's sheet read 0.35 and 0.79
#: and was not flat: its printed lines lay up to 1.2 mm from the frame, one and a half times the largest misfit. So
#: the note is where a flat sheet ends, and the refusal where the frame can no longer be within about 1.5 mm.
SHEET_FIT_NOTE_MM = (0.3, 0.6)
SHEET_FIT_LIMIT_MM = (0.5, 1.0)
#: The sheet's printed grid lines against the frame: root mean square over all scans, and the largest mean over 10 mm
#: of one line. No fit uses the lines, so this is the frame's error between the tags, measured directly, and when
#: the lines can be read it is what judges the sheet: the frame must be within a millimetre everywhere on the grid.
#: Seen at the rig: 0.35 and 1.19 mm for the first sitting's sheet (an edge stood about 2 mm off the floor), 0.37
#: and 0.95 for a sheet lying loose, 0.23 to 0.25 and 0.48 to 0.59 for the same sheet taped flat (2026-10-06).
LINES_NOTE_MM = (0.35, 0.7)
LINES_LIMIT_MM = (0.5, 1.0)
#: Fewer scans than this that show a line (of about 1070): the lines cannot be read, and the tags judge alone.
MIN_LINE_SCANS = 400
#: With the lines as judge, the tags' own figures stop a registration only when they are far off: a tag covered,
#: in glare, or on a lifted corner. (At 27 to 31 px across, a taped sheet whose lines lay within 0.6 mm of the frame
#: read 0.39 and 0.88 mm for the fit and 1.9 mm for a tag left out: the small tags' outlines are noisier in the
#: camera's compressed picture than on synthetic ones, and limits close above a flat sheet's synthetic figures
#: refused or noted a good sheet.)
GROSS_FIT_LIMIT_MM = (1.0, 2.0)
GROSS_LEAVE_ONE_OUT_LIMIT_MM = 4.0
#: A sheet tag left out of the fit, its corners predicted from the others, largest error. For a corner tag this is an
#: extrapolation by a whole tag spacing, which no point of the grid ever is, so it overstates the error on the grid.
LEAVE_ONE_OUT_NOTE_MM = 1.0
LEAVE_ONE_OUT_LIMIT_MM = 2.0
#: The sheet tags' size read through the frame against their printed size. The mean over the tags: 2 % is 1 mm in
#: 50, and all tags off together means the outlines are not measured rightly. One tag alone: said beyond this, since
#: one tag's size is known only to about a percent at this view.
SHEET_TAG_SCALE_LIMIT = 0.02
SHEET_TAG_SCALE_NOTE = 0.03
#: A floor tag's size by its outline against its stated size. Beyond the first figure the registration says so:
#: the sheet and the floor tag are not in one plane, or one of their sizes is wrong. Beyond the second it is not this
#: tag, or it is bent.
FLOOR_TAG_SIZE_NOTE = 0.025
FLOOR_TAG_SIZE_LIMIT = 0.06
#: The conversion solved from one frame's floor tags, against the frame: the largest gap anywhere on the grid,
#: averaged over the frames. A session solves from 20 frames and may be 1.5 mm from the stored conversion; single
#: frames that are a millimetre off on average leave too little of that.
WANDER_MEAN_LIMIT_MM = 1.0
#: A printed scale further than this from 1 is unusual for a printer: the registration asks for the number to be
#: checked. (Within `SHEET_SCALE_LIMIT` it is still taken.)
SHEET_SCALE_NOTE = 0.006


class RegistrationError(RuntimeError):
    """The pictures do not allow the table to be registered."""


def _checked_scale(scale: Sequence[float]) -> tuple[float, float]:
    """A printed scale as two numbers near 1: across the sheet (x) and up it (y)."""
    values = tuple(float(v) for v in scale)
    if len(values) != 2 or not all(abs(v - 1.0) <= SHEET_SCALE_LIMIT for v in values):
        raise RegistrationError(
            f"a printed scale is two numbers within {SHEET_SCALE_LIMIT:.0%} of 1 (across the sheet, up the sheet), "
            f"got {scale!r}: a sheet further off than that was not printed at 100 %"
        )
    return values


def sheet_scale_from_outline(width_mm: float, height_mm: float) -> tuple[float, float]:
    """The sheet's printed scale, from the grid's outline measured with calipers on the outer edges of its lines.

    `width_mm` is the long side (columns A to E, the table's x), `height_mm` the short side (rows 1 to 3, y).
    """
    scale = tuple((float(measured) - SHEET_OUTLINE_LINE_MM) / designed
                  for measured, designed in zip((width_mm, height_mm), SHEET_OUTLINE_MM, strict=True))
    return _checked_scale(scale)


def sheet_corners_mm(tag_id: int, scale: Sequence[float] = (1.0, 1.0)) -> np.ndarray:
    """A sheet tag's four corners in the table frame, in the detector's order.

    The detector reports a tag's corners clockwise from the tag's own top-left. On the sheet every tag is printed
    upright with the page, and the page's upward direction is the table's +y. `scale` is the sheet's printed scale
    across and up: everything on the paper, the tags' places and their sizes, is that much larger or smaller than
    designed.
    """
    cx, cy = SHEET_TAGS[tag_id]
    half = SHEET_TAG_MM / 2.0
    designed = np.array([(cx - half, cy + half), (cx + half, cy + half),
                         (cx + half, cy - half), (cx - half, cy - half)])
    return designed * np.asarray(scale, dtype=float)


def mean_corners(detections: Sequence[Mapping[int, Any]], min_share: float = MIN_SHARE) -> dict[int, np.ndarray]:
    """Each tag's corners averaged over the frames it was found in; tags found too rarely are left out."""
    if not detections:
        raise RegistrationError("no frames to register from")
    seen: dict[int, list[np.ndarray]] = {}
    for found in detections:
        for tag_id, quad in found.items():
            seen.setdefault(int(tag_id), []).append(np.asarray(quad, dtype=float).reshape(4, 2))
    needed = max(1, int(np.ceil(min_share * len(detections))))
    return {tag_id: np.mean(np.stack(quads), axis=0) for tag_id, quads in seen.items() if len(quads) >= needed}


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _sides(corners: np.ndarray) -> tuple[float, ...]:
    return tuple(float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])) for i in range(4))


@dataclass(frozen=True)
class Registration:
    """What one registration found."""

    #: The table geometry document, in the form the rest of the instrument reads.
    geometry: dict[str, Any]
    #: The table frame: the conversion from pixels to table millimetres, fitted through the sheet tags' outlines.
    px_to_mm: np.ndarray
    #: The sheet tags the conversion was fitted through.
    sheet_ids: tuple[int, ...]
    frames: int
    #: How far the sheet's outline corners are from one flat plane through an ideal lens: mean and largest misfit.
    sheet_fit_mean_mm: float
    sheet_fit_max_mm: float
    #: Each sheet tag left out in turn and predicted from the others: mean and largest error of its corners.
    leave_one_out_mean_mm: float
    leave_one_out_max_mm: float
    #: The sheet tags' size read through the frame, as a share of their printed size: across the sheet, up the sheet.
    sheet_tag_scale: tuple[float, float]
    #: How far inside a tag's outline the detector puts its corners, on average over every tag in view, pixels.
    detector_inset_px: float
    #: Each floor tag's four sides through the frame: by its outline (what is stored), and by the detector's corners.
    floor_sides_mm: dict[int, tuple[float, ...]]
    floor_detector_sides_mm: dict[int, tuple[float, ...]]
    #: Each floor tag's size by its outline as a share of its stated size: across the tag, up the tag.
    floor_tag_scale: dict[int, tuple[float, float]]
    #: The conversion solved from single frames' floor tags against the frame, over the grid: mean and largest gap.
    wander_mean_mm: float
    wander_max_mm: float
    #: The sheet's printed grid lines against the frame (`sheet_lines`), or None when they were not or could not be
    #: read.
    lines: dict[str, Any] | None = None
    #: Plain sentences. `problems` is empty when the registration may be stored; `notes` are stored with it.
    problems: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems


def _fit(pixels: np.ndarray, table: np.ndarray) -> np.ndarray | None:
    """The conversion from pixels to table millimetres that fits the given points best (least squares)."""
    import cv2

    matrix, _ = cv2.findHomography(pixels.astype(np.float64), table.astype(np.float64), 0)
    return None if matrix is None else np.asarray(matrix, dtype=np.float64)


def _size(sides: Sequence[float]) -> tuple[float, float]:
    """A tag's size from its four sides: across the tag (its top and bottom), up the tag (its right and left)."""
    return (sides[0] + sides[2]) / 2.0, (sides[1] + sides[3]) / 2.0


def grid_lattice_mm(grid: Mapping[str, Any], step_mm: float = 25.0) -> np.ndarray:
    """Points all over the grid, outline included, every `step_mm`: where two conversions are compared."""
    width, height = grid["cols"] * grid["cell_mm"], grid["rows"] * grid["cell_mm"]
    xs, ys = np.meshgrid(np.arange(0.0, width + 1e-9, step_mm), np.arange(0.0, height + 1e-9, step_mm))
    return np.column_stack([xs.ravel(), ys.ravel()])


def frames_apart_mm(first_px_to_mm: Any, second_px_to_mm: Any, grid: Mapping[str, Any]) -> float:
    """The largest distance, anywhere on the grid, between where two conversions put the same pixels."""
    first = np.asarray(first_px_to_mm, dtype=float)
    points_mm = grid_lattice_mm(grid)
    points_px = _apply(np.linalg.inv(first), points_mm)
    return float(np.linalg.norm(_apply(np.asarray(second_px_to_mm, dtype=float), points_px) - points_mm, axis=1).max())


def sheet_line_stretches() -> list[tuple[str, float, float, float]]:
    """The stretches of the sheet's printed grid lines that are clear of crossings, labels, small print and the
    ruler's ticks: (axis, value, from, to) in design millimetres. `("y", 50, 10, 40)` is the line y = 50 from x = 10
    to x = 40."""
    stretches: list[tuple[str, float, float, float]] = []
    for y in (0.0, 50.0, 100.0, 150.0):
        spans = [(60, 90), (110, 140), (160, 170)] if y == 0.0 else [(a + 10, a + 40) for a in range(0, 250, 50)]
        stretches += [("y", y, float(lo), float(hi)) for lo, hi in spans]
    for x in (0.0, 50.0, 100.0, 150.0, 200.0, 250.0):
        spans = ([(10, 21), (29, 40), (60, 71), (79, 90), (110, 121), (129, 140)] if x == 250.0
                 else [(a + 10, a + 40) for a in range(0, 150, 50)])
        stretches += [("x", x, float(lo), float(hi)) for lo, hi in spans]
    return stretches


def sheet_lines(picture_gray: np.ndarray, px_to_mm: np.ndarray, sheet_scale: Sequence[float]) -> dict[str, Any] | None:
    """How far the sheet's printed grid lines are seen from where the frame puts them, or None when they cannot be read.

    Every millimetre along every clear stretch the picture is scanned across the line (3 px either way); the line is
    where the centre of its dip in brightness is. Returned: the number of scans that showed a line, the root mean
    square of their offsets in millimetres (all, and for the lines of constant x and of constant y apart), and the
    largest mean offset over 10 mm of one line with where it is. Lines of constant y test the frame along the
    camera's line of sight, which is where a sheet that is not flat shows.
    """
    import cv2

    half_px, step_px = 3.0, 0.25
    mm_to_px = np.linalg.inv(np.asarray(px_to_mm, dtype=float))
    image = np.asarray(picture_gray, dtype=np.float32)
    scan = np.arange(-half_px, half_px + 1e-9, step_px)
    ends = max(2, int(round(0.75 / step_px)))
    rows: list[tuple[str, float, float, float]] = []
    for axis, value, lo, hi in sheet_line_stretches():
        along = np.linspace(lo, hi, int(hi - lo) + 1)
        design = (np.column_stack([along, np.full(len(along), value)]) if axis == "y"
                  else np.column_stack([np.full(len(along), value), along]))
        table = design * np.asarray(sheet_scale, dtype=float)
        pixels = _apply(mm_to_px, table)
        normal = _apply(mm_to_px, table + ((0.0, 1.0) if axis == "y" else (1.0, 0.0))) - pixels
        px_per_mm = np.linalg.norm(normal, axis=1)
        normal = normal / px_per_mm[:, None]
        points = pixels[:, None, :] + scan[None, :, None] * normal[:, None, :]
        inside = ((points[..., 0] >= 0) & (points[..., 0] <= image.shape[1] - 1)
                  & (points[..., 1] >= 0) & (points[..., 1] <= image.shape[0] - 1)).all(axis=1)
        profile = cv2.remap(image, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32),
                            cv2.INTER_CUBIC)
        left, right = profile[:, :ends].mean(axis=1), profile[:, -ends:].mean(axis=1)
        base = left[:, None] + (right - left)[:, None] * (scan[None, :] - scan[:ends].mean()) / (
            scan[-ends:].mean() - scan[:ends].mean())
        dark = np.clip(base - profile, 0.0, None)
        area = dark.sum(axis=1) * step_px
        centre = (dark * scan[None, :]).sum(axis=1) / np.maximum(dark.sum(axis=1), 1e-9)
        shown = inside & (area > 0.5 * float(np.median(area))) & (area > 0.3)
        rows += [(axis, value, float(a), float(c / s)) for a, c, s, ok in zip(along, centre, px_per_mm, shown,
                                                                              strict=True) if ok]
    if len(rows) < MIN_LINE_SCANS:
        return None
    offsets = np.array([row[3] for row in rows])
    by_axis = {axis: np.array([row[3] for row in rows if row[0] == axis]) for axis in ("x", "y")}
    bins: dict[tuple[str, float, int], list[float]] = {}
    for axis, value, along_mm, offset in rows:
        bins.setdefault((axis, value, int(along_mm // 10)), []).append(offset)
    full = {key: values for key, values in bins.items() if len(values) >= 5}
    worst = max(full, key=lambda key: abs(float(np.mean(full[key]))))
    return {
        "scans": len(rows),
        "rms_mm": float(np.sqrt(np.mean(offsets**2))),
        "rms_x_mm": float(np.sqrt(np.mean(by_axis["x"] ** 2))) if len(by_axis["x"]) else 0.0,
        "rms_y_mm": float(np.sqrt(np.mean(by_axis["y"] ** 2))) if len(by_axis["y"]) else 0.0,
        "worst_mm": float(abs(np.mean(full[worst]))),
        "worst_where": f"the line {worst[0]} = {worst[1]:g} mm, {10 * worst[2]} to {10 * worst[2] + 10} mm along it",
    }


def single_frame_wander(
    geometry: Mapping[str, Any], outlines: Sequence[Mapping[int, Any]], px_to_mm: np.ndarray
) -> tuple[float, float]:
    """How far the conversion solved from one frame's floor tags puts the grid from the frame.

    Each frame in which both floor tags' outlines were measured is solved alone, as a session's check solves it, and
    its largest gap from the frame anywhere on the grid is taken. Returned: the mean of that over the frames, and the largest. With no
    noise in the detector the gap is zero by construction; what is reported is what the detector's noise on eight
    corners does over the grid. From a handful of frames it says little.
    """
    floor = {int(tag_id): corners for tag_id, corners in geometry["tags"].items()}
    solve = lerobot101.load("placement").homography_from_tag_geometry
    gaps: list[float] = []
    for found in outlines:
        if not all(tag_id in found for tag_id in floor):
            continue
        quads = {tag_id: np.asarray(found[tag_id], dtype=float).reshape(4, 2) for tag_id in floor}
        matrix = solve(quads, floor, geometry["tag_mm"])
        if matrix is None:
            continue
        gaps.append(frames_apart_mm(px_to_mm, matrix, geometry["grid"]))
    if not gaps:
        raise RegistrationError("the two floor tags do not determine the table frame in any single frame")
    return float(np.mean(gaps)), float(np.max(gaps))


def register(
    detections: Sequence[Mapping[int, Any]],
    outlines: Sequence[Mapping[int, Any]],
    *,
    grid: Mapping[str, Any],
    note: str = "",
    created: str | None = None,
    sheet_scale: Sequence[float] = (1.0, 1.0),
    sheet_outline_measured_mm: Sequence[float] | None = None,
    floor_tags_mm: Mapping[int, Sequence[float]] | None = None,
    picture: np.ndarray | None = None,
) -> Registration:
    """Measure the two floor tags against the sheet, from the tags found in a run of frames.

    `detections` holds, for each frame, the tags as the detector reports them, `{id: 4 x 2 corners in pixels}`;
    `outlines` holds, for the same frames, the tags' outlines measured from the picture (`outline.tag_outlines`).
    `grid` is the grid's size (`cols`, `rows`, `cell_mm`), written into the geometry so that the drawing tools agree
    with the protocol. `sheet_scale` is the sheet's printed scale across and up (`sheet_scale_from_outline`);
    `sheet_outline_measured_mm` is the caliper reading it came from, kept with the result. `floor_tags_mm` gives each
    floor tag's size by calipers, across and up the tag as printed; without it the nominal size is taken.
    `picture` is a grey picture of the scene (best the mean of the frames): with it the sheet's printed grid lines
    are held against the frame (`sheet_lines`), and they, not the tags' own fit, judge whether the sheet lies flat.
    Raises `RegistrationError` when the pictures cannot give a registration at all; what makes a registration
    doubtful is returned in its `problems`.
    """
    sheet_scale = _checked_scale(sheet_scale)
    if len(outlines) != len(detections):
        raise RegistrationError("the detector's tags and the tags' outlines must come from the same frames")
    corners_px = mean_corners(detections)
    outline_px = mean_corners(outlines)
    sheet_ids = tuple(sorted(set(outline_px) & set(SHEET_TAGS)))
    if len(sheet_ids) < MIN_SHEET_TAGS:
        seen = sorted(set(corners_px) & set(SHEET_TAGS))
        lacking = sorted(set(SHEET_TAGS) - set(sheet_ids))
        raise RegistrationError(
            f"all six of the sheet's tags {sorted(SHEET_TAGS)} must be in view with a clean outline: the detector "
            f"found {seen}, and an outline could be measured for {list(sheet_ids)}. Missing: {lacking}. Is the sheet "
            "seated, flat, wholly inside the picture, uncovered and free of glare?"
        )
    missing = [tag_id for tag_id in FLOOR_TAG_IDS if tag_id not in corners_px]
    if missing:
        raise RegistrationError(
            f"floor tag(s) {missing} were found in fewer than half of the frames, or not at all: both must be in the "
            "picture with the sheet, with nothing over them"
        )
    unmeasured = [tag_id for tag_id in FLOOR_TAG_IDS if tag_id not in outline_px]
    if unmeasured:
        raise RegistrationError(
            f"floor tag(s) {unmeasured} are in the picture but their outline could not be measured: is something "
            "lying across them, are they in glare, or curled?"
        )
    stated = {tag_id: (FLOOR_TAG_NOMINAL_MM, FLOOR_TAG_NOMINAL_MM) for tag_id in FLOOR_TAG_IDS}
    for tag_id, size in (floor_tags_mm or {}).items():
        if int(tag_id) not in stated or len(tuple(size)) != 2:
            raise RegistrationError(f"a floor tag's size is given for tag 0 or 1 as (across, up) in mm, got {tag_id}: "
                                    f"{size!r}")
        stated[int(tag_id)] = (float(size[0]), float(size[1]))

    pixels = np.concatenate([outline_px[tag_id] for tag_id in sheet_ids])
    table = np.concatenate([sheet_corners_mm(tag_id, sheet_scale) for tag_id in sheet_ids])
    px_to_mm = _fit(pixels, table)
    if px_to_mm is None:
        raise RegistrationError("the sheet's tags do not determine a conversion from the picture to the table")
    misfit = np.linalg.norm(_apply(px_to_mm, pixels) - table, axis=1)
    lines = None if picture is None else sheet_lines(picture, px_to_mm, sheet_scale)

    left_out: list[float] = []
    for tag_id in sheet_ids:
        others = [other for other in sheet_ids if other != tag_id]
        without = _fit(np.concatenate([outline_px[other] for other in others]),
                       np.concatenate([sheet_corners_mm(other, sheet_scale) for other in others]))
        if without is None:
            raise RegistrationError(f"without tag {tag_id} the sheet's other tags do not determine a conversion")
        predicted = _apply(without, outline_px[tag_id])
        left_out.extend(np.linalg.norm(predicted - sheet_corners_mm(tag_id, sheet_scale), axis=1).tolist())

    printed = (SHEET_TAG_MM * sheet_scale[0], SHEET_TAG_MM * sheet_scale[1])
    sheet_sizes = np.array([_size(_sides(_apply(px_to_mm, outline_px[tag_id]))) for tag_id in sheet_ids])
    sheet_tag_scale = (float(sheet_sizes[:, 0].mean() / printed[0]), float(sheet_sizes[:, 1].mean() / printed[1]))
    each_tag_scale = {tag_id: (float(size[0] / printed[0]), float(size[1] / printed[1]))
                      for tag_id, size in zip(sheet_ids, sheet_sizes, strict=True)}
    both = sorted(set(corners_px) & set(outline_px))
    inset = float(np.mean([outline.inset_px(corners_px[tag_id], outline_px[tag_id]) for tag_id in both]))

    tags: dict[str, Any] = {}
    sides: dict[int, tuple[float, ...]] = {}
    detector_sides: dict[int, tuple[float, ...]] = {}
    floor_scale: dict[int, tuple[float, float]] = {}
    problems: list[str] = []
    notes: list[str] = []
    for tag_id in FLOOR_TAG_IDS:
        corners = _apply(px_to_mm, outline_px[tag_id])
        sides[tag_id] = _sides(corners)
        detector_sides[tag_id] = _sides(_apply(px_to_mm, corners_px[tag_id]))
        tags[str(tag_id)] = {
            "centre": [round(float(v), 2) for v in corners.mean(axis=0)],
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in corners],
        }
        across, up = _size(sides[tag_id])
        floor_scale[tag_id] = (across / stated[tag_id][0], up / stated[tag_id][1])
        worst = max(abs(value - 1.0) for value in floor_scale[tag_id])
        said = (f"floor tag {tag_id} reads {across:.1f} x {up:.1f} mm by its outline (across x up the tag) against "
                f"{stated[tag_id][0]:g} x {stated[tag_id][1]:g} stated, {100 * (floor_scale[tag_id][0] - 1.0):+.1f} % "
                f"and {100 * (floor_scale[tag_id][1] - 1.0):+.1f} %")
        if worst > FLOOR_TAG_SIZE_LIMIT:
            problems.append(
                f"{said} (limit {100 * FLOOR_TAG_SIZE_LIMIT:g} %): this is not the 60 mm tag, a size was mistyped "
                "(the sheet's outline or the tag's), the sheet lies tilted, or the tag is bent"
            )
        elif worst > FLOOR_TAG_SIZE_NOTE:
            notes.append(
                f"{said}: the sheet and this tag do not lie in one plane, or a size is wrong. This is the only figure "
                "here that can see a sheet lying tilted, so do not pass over it: tape the sheet flat along all four "
                "edges, press this tag flat and look along the floor at it, check the two sizes typed, and register "
                "again. If it stays with both plainly flat, say so: the cause is then the floor or the lens, and "
                "the cube moved along a printed scale (runbook 1, step 4) is what shows whether the grid is right"
            )

    geometry: dict[str, Any] = {
        "tag_mm": FLOOR_TAG_NOMINAL_MM,
        "tags": tags,
        "grid": {"cols": int(grid["cols"]), "rows": int(grid["rows"]), "cell_mm": float(grid["cell_mm"])},
        "note": note,
        "created": created or datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    wander_mean, wander_max = single_frame_wander(geometry, outlines, px_to_mm)
    geometry["registration"] = {
        "method": "printed sheet with six 36 mm tags; the frame is fitted through the sheet tags' outlines, and the "
                  "floor tags' outlines are stored through it",
        "sheet_tags": list(sheet_ids),
        "sheet_scale": [round(v, 5) for v in sheet_scale],
        "sheet_outline_measured_mm": (None if sheet_outline_measured_mm is None
                                      else [float(v) for v in sheet_outline_measured_mm]),
        "frames": len(detections),
        "sheet_fit_mean_mm": round(float(misfit.mean()), 3),
        "sheet_fit_max_mm": round(float(misfit.max()), 3),
        "leave_one_out_mean_mm": round(float(np.mean(left_out)), 3),
        "leave_one_out_max_mm": round(float(np.max(left_out)), 3),
        "sheet_tag_scale": [round(v, 4) for v in sheet_tag_scale],
        "detector_inset_px": round(inset, 2),
        "floor_tags_stated_mm": {str(k): [float(v) for v in size] for k, size in stated.items()},
        "floor_tag_sides_mm": {str(k): [round(v, 2) for v in vs] for k, vs in sides.items()},
        "floor_tag_detector_sides_mm": {str(k): [round(v, 2) for v in vs] for k, vs in detector_sides.items()},
        "floor_tag_scale": {str(k): [round(v, 4) for v in vs] for k, vs in floor_scale.items()},
        "sheet_tag_scale_each": {str(k): [round(v, 4) for v in vs] for k, vs in each_tag_scale.items()},
        "sheet_lines": (None if lines is None else
                        {key: (round(value, 3) if isinstance(value, float) else value) for key, value in lines.items()}),
        "single_frame_wander_mean_mm": round(wander_mean, 3),
        "single_frame_wander_max_mm": round(wander_max, 3),
        # The frame itself and where the detector saw the floor tags: what the calibration is held against.
        "px_to_mm": [[float(v) for v in row] for row in px_to_mm],
        "floor_tag_corners_px": {str(k): [[round(float(x), 3), round(float(y), 3)] for x, y in outline_px[k]]
                                 for k in FLOOR_TAG_IDS},
    }

    fit = (float(misfit.mean()), float(misfit.max()))
    worst_left_out = float(np.max(left_out))
    flat = ("tape the sheet flat along all four edges, see that no tag is in glare or partly covered, and register "
            "again")
    if lines is not None:
        # The printed lines can be read: they measure the frame between the tags, and they judge.
        said = (f"the sheet's printed lines lie {lines['rms_mm']:.2f} mm (rms) from the frame and "
                f"{lines['worst_mm']:.2f} mm at worst, on {lines['worst_where']}")
        if lines["rms_mm"] > LINES_LIMIT_MM[0] or lines["worst_mm"] > LINES_LIMIT_MM[1]:
            problems.append(f"{said} (limits {LINES_LIMIT_MM[0]} and {LINES_LIMIT_MM[1]}): the sheet is not lying "
                            f"flat there. To do: {flat}")
        elif lines["rms_mm"] > LINES_NOTE_MM[0] or lines["worst_mm"] > LINES_NOTE_MM[1]:
            notes.append(f"{said} (noted beyond {LINES_NOTE_MM[0]} and {LINES_NOTE_MM[1]}): the sheet is probably "
                         f"not quite flat there. To do: {flat}")
        if (fit[0] > GROSS_FIT_LIMIT_MM[0] or fit[1] > GROSS_FIT_LIMIT_MM[1]
                or worst_left_out > GROSS_LEAVE_ONE_OUT_LIMIT_MM):
            problems.append(
                f"the sheet's tags fit one flat plane only to {fit[0]:.2f} mm on average and {fit[1]:.2f} mm at "
                f"worst, and a tag left out is predicted only to {worst_left_out:.2f} mm (limits "
                f"{GROSS_FIT_LIMIT_MM[0]}, {GROSS_FIT_LIMIT_MM[1]} and {GROSS_LEAVE_ONE_OUT_LIMIT_MM}): a tag is "
                "partly covered, in glare, or on a lifted corner of the sheet"
            )
    else:
        if picture is not None:
            notes.append("the sheet's printed lines could not be read in this picture (too bright, too dark or "
                         "blurred?), so the sheet was judged by its tags alone")
        if fit[0] > SHEET_FIT_LIMIT_MM[0] or fit[1] > SHEET_FIT_LIMIT_MM[1]:
            problems.append(
                f"the sheet's tags fit one flat plane only to {fit[0]:.2f} mm on average and {fit[1]:.2f} mm at "
                f"worst (limits {SHEET_FIT_LIMIT_MM[0]} and {SHEET_FIT_LIMIT_MM[1]}): the sheet is not lying flat, "
                f"a tag is in glare, or the picture is over-exposed or blurred. To do: {flat}"
            )
        elif fit[0] > SHEET_FIT_NOTE_MM[0] or fit[1] > SHEET_FIT_NOTE_MM[1]:
            notes.append(
                f"the sheet's tags fit one flat plane to {fit[0]:.2f} mm on average and {fit[1]:.2f} mm at worst; a "
                f"flat sheet gives about 0.2 and 0.45 at this view (noted beyond {SHEET_FIT_NOTE_MM[0]} and "
                f"{SHEET_FIT_NOTE_MM[1]}). The sheet is probably not quite flat: on 2026-10-05 a sheet that read "
                f"0.35 and 0.79 had its printed lines up to 1.2 mm from the frame. To do: {flat}"
            )
        if worst_left_out > LEAVE_ONE_OUT_LIMIT_MM:
            problems.append(
                f"a sheet tag left out of the fit is predicted by the others only to {worst_left_out:.2f} mm (limit "
                f"{LEAVE_ONE_OUT_LIMIT_MM}): the frame does not hold between the sheet's tags. Is a corner of the "
                "sheet lifted, or a tag in glare?"
            )
        elif worst_left_out > LEAVE_ONE_OUT_NOTE_MM:
            notes.append(
                f"a sheet tag left out of the fit is predicted by the others to {worst_left_out:.2f} mm at worst "
                f"(noted beyond {LEAVE_ONE_OUT_NOTE_MM}): the frame is less sure toward the grid's corners than a "
                "flat sheet makes it"
            )
    if max(abs(value - 1.0) for value in sheet_tag_scale) > SHEET_TAG_SCALE_LIMIT:
        problems.append(
            f"the sheet's tags read {100 * (sheet_tag_scale[0] - 1.0):+.1f} % across and "
            f"{100 * (sheet_tag_scale[1] - 1.0):+.1f} % up against their printed size on average (limit "
            f"{100 * SHEET_TAG_SCALE_LIMIT:g} %): their outlines are not being measured rightly. Is the picture "
            "over-exposed or out of focus, or is this another sheet?"
        )
    else:
        sloping = [f"tag {tag_id} {100 * (scale[axis] - 1.0):+.1f} % {('across', 'up')[axis]}"
                   for tag_id, scale in each_tag_scale.items() for axis in (0, 1)
                   if abs(scale[axis] - 1.0) > SHEET_TAG_SCALE_NOTE]
        if sloping:
            notes.append(
                f"single sheet tags read off their printed size: {'; '.join(sloping)} (noted beyond "
                f"{100 * SHEET_TAG_SCALE_NOTE:g} %). A tag that reads short one way lies on a slope: the sheet is "
                "not flat there"
            )
    if max(abs(value - 1.0) for value in sheet_scale) > SHEET_SCALE_NOTE:
        notes.append(
            f"the sheet was taken as printed {100 * (sheet_scale[0] - 1.0):+.2f} % across and "
            f"{100 * (sheet_scale[1] - 1.0):+.2f} % up, from the outline typed in: more than printers usually do. "
            "Check the two numbers; nothing on the sheet can tell a mistyped size from a true one"
        )
    if wander_mean > WANDER_MEAN_LIMIT_MM:
        problems.append(
            f"solved from single frames' floor tags, the grid lies {wander_mean:.2f} mm from the frame on average "
            f"and {wander_max:.2f} mm at worst (limit {WANDER_MEAN_LIMIT_MM} on the average): a floor tag is too "
            "small in the picture, in glare, or flickering"
        )
    geometry["registration"]["notes"] = list(notes)
    return Registration(
        geometry=geometry,
        px_to_mm=px_to_mm,
        sheet_ids=sheet_ids,
        frames=len(detections),
        sheet_fit_mean_mm=float(misfit.mean()),
        sheet_fit_max_mm=float(misfit.max()),
        leave_one_out_mean_mm=float(np.mean(left_out)),
        leave_one_out_max_mm=float(np.max(left_out)),
        sheet_tag_scale=sheet_tag_scale,
        detector_inset_px=inset,
        floor_sides_mm=sides,
        floor_detector_sides_mm=detector_sides,
        floor_tag_scale=floor_scale,
        wander_mean_mm=wander_mean,
        wander_max_mm=wander_max,
        lines=lines,
        problems=tuple(problems),
        notes=tuple(notes),
    )
