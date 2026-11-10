"""Where the basket stands in the front camera's picture, held against a reference frame.

The basket stands on double-sided tape, clear of both pipes of the light box: it has no stop to go back to. On
2026-10-08 the wrist camera's mount caught its rim during an attempt and carried it four centimetres. Its place is
therefore what the camera saw of it: the outer edge of its left wall and of its near rim, read from the picture.
The two floor tags say how far the camera itself has moved since the reference, and that is taken off.

*Measured* on the kept frames of 2026-10-07 and 2026-10-08 (seven start frames of sessions and the first frames of
seven trials): the basket read the same place to within 0.2 px in every one, and 38 px and 11 px away, turned, in
the frame after it had been carried.

Pure functions on a frame and on the tags found in it: no camera here. `camera_tool`'s `basket` mode shows them live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The basket is the large dark grey thing in the upper right of the picture (640 x 480): darker than this value and
#: less coloured than this saturation, of 255. The floor reads about 200, the arm is orange, a hand is skin.
DARK_VALUE, DARK_SATURATION = 125, 100
#: Rows across the basket's left wall, searched from this column to the right for the wall's outer edge; and columns
#: across its near rim, searched between these rows for the rim's lowest dark pixel.
WALL_ROWS, WALL_FROM, WALL_TO = range(70, 200, 2), 380, 600
RIM_COLUMNS, RIM_FROM, RIM_TO = range(480, 630, 2), 150, 330
#: Fewer edge points than this on either edge: the basket is not where the search looks, or something hides it.
MIN_POINTS = 20
#: Within this of its place, both ways, and turned by less than this, the basket is in place. A placed cube is held
#: to 2 mm; the basket is a target 170 mm deep and a shape in the policy's picture, and needs no more.
IN_PLACE_MM = 2.0
IN_PLACE_LEAN = 2.0          # px of the near rim's rise per 100 px along it (about one degree)
#: Millimetres on the floor per pixel where no floor tag says it: the two tags are 526.9 mm and 486 px apart.
MM_PER_PX = 1.08
#: The part of the picture the reference frame is laid over: this far around the basket's two edges.
GHOST_MARGIN_PX = 45
#: What the live picture can show, in the order the key steps through them.
VIEWS = ("live", "live with the old basket laid over it", "the old basket alone")


@dataclass(frozen=True)
class BasketReading:
    """The basket as one frame shows it. Pixels; x to the right, y down."""

    wall_px: float                     # the left wall's outer edge: the median column over the searched rows
    rim_px: float                      # the near rim's outer edge: the median row over the searched columns
    wall_fit: tuple[float, float]      # the wall's edge as x = a * y + b
    rim_fit: tuple[float, float]       # the rim's edge as y = a * x + b
    wall_span: tuple[int, int]         # the rows the wall's edge was found in, first and last
    rim_span: tuple[int, int]          # the columns the rim's edge was found in, first and last
    tags: dict[int, tuple[float, float]]   # the centres of the tags in view

    @property
    def rim_lean(self) -> float:
        """How much the near rim rises or falls: pixels down per 100 pixels to the right."""
        return 100.0 * self.rim_fit[0]


@dataclass(frozen=True)
class BasketOffset:
    """How far the basket stands from its reference place, the camera's own movement taken off."""

    across_mm: float                   # positive: too far to the right of the picture
    down_mm: float                     # positive: too far down in the picture, which is towards the camera
    lean: float                        # positive: the rim's right end lower in the picture than it was
    camera_px: tuple[float, float] | None   # how far the tags have moved since the reference; None: no tag to say

    @property
    def in_place(self) -> bool:
        return (abs(self.across_mm) <= IN_PLACE_MM and abs(self.down_mm) <= IN_PLACE_MM
                and abs(self.lean) <= IN_PLACE_LEAN)


def read_basket(frame_bgr: np.ndarray, found: dict | None = None) -> BasketReading | None:
    """The basket's two edges in a frame of the front camera, or None when they cannot be found.

    `found` is what the tag detector gave for the same frame (tag id to its four corners); its tags' centres are
    kept so that two readings can be held against each other with the camera's own movement taken off.
    """
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < DARK_VALUE) & (hsv[..., 1] < DARK_SATURATION)
    height, width = dark.shape
    wall = []
    for y in WALL_ROWS:
        if y >= height:
            break
        columns = np.flatnonzero(dark[y, WALL_FROM:min(WALL_TO, width)])
        if columns.size:
            wall.append((y, WALL_FROM + int(columns[0])))
    rim = []
    for x in RIM_COLUMNS:
        if x >= width:
            break
        rows = np.flatnonzero(dark[RIM_FROM:min(RIM_TO, height), x])
        if rows.size:
            rim.append((x, RIM_FROM + int(rows[-1])))
    if len(wall) < MIN_POINTS or len(rim) < MIN_POINTS:
        return None
    wall_y, wall_x = np.array(wall, dtype=float).T
    rim_x, rim_y = np.array(rim, dtype=float).T
    wall_fit, rim_fit = np.polyfit(wall_y, wall_x, 1), np.polyfit(rim_x, rim_y, 1)
    centres = {int(tag): np.asarray(quad, dtype=float).mean(axis=0) for tag, quad in (found or {}).items()}
    tags = {tag: (float(centre[0]), float(centre[1])) for tag, centre in centres.items()}
    return BasketReading(
        wall_px=float(np.median(wall_x)), rim_px=float(np.median(rim_y)),
        wall_fit=(float(wall_fit[0]), float(wall_fit[1])), rim_fit=(float(rim_fit[0]), float(rim_fit[1])),
        wall_span=(int(wall_y[0]), int(wall_y[-1])), rim_span=(int(rim_x[0]), int(rim_x[-1])), tags=tags)


def camera_moved(reference: BasketReading, now: BasketReading) -> tuple[float, float] | None:
    """How far the tags both readings share stand from where the reference has them, on average: the camera's own
    movement, in pixels. None when they share no tag."""
    shared = sorted(set(reference.tags) & set(now.tags))
    if not shared:
        return None
    moved = np.mean([np.subtract(now.tags[tag], reference.tags[tag]) for tag in shared], axis=0)
    return float(moved[0]), float(moved[1])


def offset(reference: BasketReading, now: BasketReading, mm_per_px: float = MM_PER_PX) -> BasketOffset:
    """How far `now` stands from `reference`, in millimetres on the floor, the camera's own movement taken off."""
    camera = camera_moved(reference, now)
    moved_x, moved_y = camera or (0.0, 0.0)
    return BasketOffset(
        across_mm=(now.wall_px - reference.wall_px - moved_x) * mm_per_px,
        down_mm=(now.rim_px - reference.rim_px - moved_y) * mm_per_px,
        lean=now.rim_lean - reference.rim_lean, camera_px=camera)


def advice(off: BasketOffset) -> list[str]:
    """What to do with the basket, one line for each thing that is out; empty when it is in place. The directions
    are those of the camera's picture, which is what the operator watches while moving it."""
    lines = []
    if abs(off.across_mm) > IN_PLACE_MM:
        lines.append(f"move it {abs(off.across_mm):.0f} mm to the {'LEFT' if off.across_mm > 0 else 'RIGHT'}")
    if abs(off.down_mm) > IN_PLACE_MM:
        way = "UP (away from the camera)" if off.down_mm > 0 else "DOWN (towards the camera)"
        lines.append(f"move it {abs(off.down_mm):.0f} mm {way}")
    if abs(off.lean) > IN_PLACE_LEAN:
        lines.append(f"turn it: its right end {'UP' if off.lean > 0 else 'DOWN'} in the picture")
    return lines


def mm_per_px(found: dict, geometry: dict | None) -> float:
    """Millimetres on the floor per pixel, from the two floor tags as this frame shows them and as the table's
    geometry has them; the rig's usual figure when either is missing."""
    try:
        centres = {int(tag): np.asarray(quad, dtype=float).mean(axis=0) for tag, quad in found.items()}
        known = {int(tag): np.asarray(entry["centre"], dtype=float) for tag, entry in (geometry or {})["tags"].items()}
        first, second = sorted(set(centres) & set(known))[:2]
        apart_px = float(np.linalg.norm(centres[first] - centres[second]))
        apart_mm = float(np.linalg.norm(known[first] - known[second]))
        return apart_mm / apart_px if apart_px > 50 else MM_PER_PX
    except Exception:  # noqa: BLE001 -- a scale that cannot be worked out is the usual one
        return MM_PER_PX


def draw(frame_bgr: np.ndarray, reference: BasketReading, now: BasketReading | None, off: BasketOffset | None,
         lines: list[str]) -> np.ndarray:
    """The frame with the basket's place drawn in: green where its two edges belong, orange where they are (green
    too once it is in place), and `lines` at the bottom."""
    import cv2

    picture = frame_bgr.copy()
    moved_x, moved_y = (off.camera_px if off and off.camera_px else (0.0, 0.0))

    def edges(reading: BasketReading, dx: float, dy: float, colour: tuple[int, int, int], thick: int) -> None:
        (a, b), (y0, y1) = reading.wall_fit, reading.wall_span
        cv2.line(picture, (round(a * y0 + b + dx), round(y0 + dy)), (round(a * y1 + b + dx), round(y1 + dy)),
                 colour, thick)
        (a, b), (x0, x1) = reading.rim_fit, reading.rim_span
        cv2.line(picture, (round(x0 + dx), round(a * x0 + b + dy)), (round(x1 + dx), round(a * x1 + b + dy)),
                 colour, thick)

    edges(reference, moved_x, moved_y, (80, 255, 80), 2)
    if now is not None:
        edges(now, 0.0, 0.0, (80, 255, 80) if off and off.in_place else (0, 160, 255), 1)
    height = picture.shape[0]
    good = bool(off and off.in_place)
    for row, line in enumerate(lines):
        where = (10, height - 12 - 20 * (len(lines) - 1 - row))
        cv2.putText(picture, line, where, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(picture, line, where, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80) if good else (0, 190, 255), 1)
    return picture


def ghost_region(reference: BasketReading, shape: tuple[int, ...]) -> tuple[slice, slice]:
    """The rows and columns of the picture that hold the basket where it belongs, with a margin."""
    height, width = shape[:2]
    return (slice(0, min(height, round(reference.rim_px) + GHOST_MARGIN_PX)),
            slice(max(0, round(reference.wall_px) - GHOST_MARGIN_PX), width))


def ghosted(frame_bgr: np.ndarray, reference_bgr: np.ndarray, reference: BasketReading,
            camera_px: tuple[float, float] | None, share: float = 0.5) -> np.ndarray:
    """The frame with the reference frame's picture of the basket laid over it: the basket as it stood, itself and
    not two lines, to hold the basket as it stands against. `share` is how much of the old picture shows (0.5: both,
    a basket out of place is seen twice; 1: the old one alone). The old picture is moved by the camera's own
    movement first, so that floor and tags lie on themselves. A thin frame marks the part that is laid over."""
    import cv2

    height, width = frame_bgr.shape[:2]
    dx, dy = camera_px or (0.0, 0.0)
    moved = cv2.warpAffine(reference_bgr, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height),
                           borderMode=cv2.BORDER_REPLICATE)
    rows, columns = ghost_region(reference, frame_bgr.shape)
    picture = frame_bgr.copy()
    picture[rows, columns] = cv2.addWeighted(frame_bgr[rows, columns], 1.0 - share, moved[rows, columns], share, 0.0)
    cv2.rectangle(picture, (columns.start, rows.start), (columns.stop - 1, rows.stop - 1), (160, 160, 160), 1)
    return picture


def said(reference: BasketReading, now: BasketReading | None, off: BasketOffset | None) -> list[str]:
    """The lines for the live picture and the terminal: where the basket stands and what to do."""
    if now is None or off is None:
        return ["the basket cannot be seen where it belongs (upper right): is something in front of it?"]
    where = (f"basket: {off.across_mm:+.0f} mm across, {off.down_mm:+.0f} mm down, rim {off.lean:+.1f}"
             + ("" if off.camera_px else "  (no floor tag in view: the camera's own movement is not taken off)"))
    todo = advice(off)
    return [where, *(todo or ["IN PLACE"])]
