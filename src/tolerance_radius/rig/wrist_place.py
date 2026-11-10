"""Where the wrist camera stands in its mount, read from its own picture.

The gripper's fixed jaw is rigid with the wrist camera: whatever the arm does, that jaw stands still in the wrist
camera's picture, and only the other jaw moves. So the fixed jaw's place in the picture is the camera's place in its
mount. On 2026-10-08 the mount caught the basket's rim during an attempt, and the jaw stood 3 px further left
afterwards: the camera had moved.

*Measured* (2026-10-08): in both training sets the fixed jaw's inner edge has its median at 265.0 and 264.7 px and
the lower of its two holes at (193.4, 455.0) and (193.2, 455.1); in the snapshot the first session of that day
started on, 265.0 and (193.0, 455.5); after the first catch 262.0 and (189.4, 457.3), the edge's lean unchanged.

Pure functions on a frame: no camera here. `camera_tool`'s `wristplace` mode shows them live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The jaws are orange: this hue range, and at least this saturation and value (OpenCV's HSV, hue of 180).
HUE, MIN_SATURATION, MIN_VALUE = (3, 25), 120, 120
#: The fixed jaw is the left one in the picture (640 x 480). Its inner edge is read at these rows, left of this
#: column; the moving jaw never comes left of it at these rows.
EDGE_ROWS, EDGE_LEFT_OF = (420, 440, 460), 300
#: The lower hole of the fixed jaw is looked for in this part of the picture, as a small patch that is not orange
#: and lies wholly inside the jaw, within this many pixels of where the reference has it.
HOLE_ROWS, HOLE_COLUMNS, HOLE_AREA, HOLE_WITHIN_PX = (400, 480), (140, 260), (8, 250), 15.0
#: Within this of its place, both ways, the camera is in place. Its picture stood within 0.5 px of the training
#: sets' on every kept snapshot before the first catch.
IN_PLACE_PX = 1.0
#: The part of the picture the reference is laid over: the fixed jaw and some floor around it, and not the moving
#: jaw, which stands wherever the gripper was left.
OVER_ROWS, OVER_COLUMNS = (290, 480), (90, 300)
VIEWS = ("live", "live with the old picture of the jaw laid over it", "the old picture alone")


@dataclass(frozen=True)
class JawReading:
    """The fixed jaw as one frame of the wrist camera shows it. Pixels; x to the right, y down."""

    edge_px: float                          # its inner edge: the mean column over the rows read
    lean_px: float                          # that edge's column at the last row less at the first
    hole: tuple[float, float] | None        # the centre of its lower hole, where one was found


@dataclass(frozen=True)
class JawOffset:
    """How far the jaw stands from its reference place in the picture, which is how far the camera has moved."""

    across_px: float                        # positive: the jaw too far to the right of the picture
    down_px: float | None                   # positive: too far down; None when a hole could not be read
    lean_px: float                          # the edge's lean, less the reference's: the camera rolled

    @property
    def in_place(self) -> bool:
        return (abs(self.across_px) <= IN_PLACE_PX and abs(self.lean_px) <= IN_PLACE_PX
                and (self.down_px is None or abs(self.down_px) <= IN_PLACE_PX))


def orange(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] > HUE[0]) & (hsv[..., 0] < HUE[1]) & (hsv[..., 1] > MIN_SATURATION) & (hsv[..., 2] > MIN_VALUE)


def read_jaw(frame_bgr: np.ndarray, hole_near: tuple[float, float] | None = None) -> JawReading | None:
    """The fixed jaw in a frame of the wrist camera, or None when it cannot be found at the rows it is read at.
    `hole_near`: where the lower hole is expected (the reference's); without it the hole nearest the middle of the
    part searched is taken."""
    import cv2

    mask = orange(frame_bgr)
    if mask.shape[0] <= max(EDGE_ROWS):
        return None
    edges = []
    for row in EDGE_ROWS:
        columns = np.flatnonzero(mask[row, :EDGE_LEFT_OF])
        if columns.size < 20:
            return None
        edges.append(float(columns.max()))
    (top, bottom), (left, right) = HOLE_ROWS, HOLE_COLUMNS
    part = (~mask[top:bottom, left:right]).astype(np.uint8)
    count, _labels, stats, centres = cv2.connectedComponentsWithStats(part, connectivity=8)
    near = hole_near or ((left + right) / 2.0, (top + bottom) / 2.0)
    best: tuple[float, tuple[float, float]] | None = None
    for k in range(1, count):
        x, y, wide, high, area = (int(v) for v in stats[k])
        if not HOLE_AREA[0] <= area <= HOLE_AREA[1] or x == 0 or y == 0 or x + wide >= part.shape[1] \
                or y + high >= part.shape[0]:
            continue
        centre = (float(centres[k][0]) + left, float(centres[k][1]) + top)
        apart = float(np.hypot(centre[0] - near[0], centre[1] - near[1]))
        if (hole_near is None or apart <= HOLE_WITHIN_PX) and (best is None or apart < best[0]):
            best = (apart, centre)
    return JawReading(edge_px=float(np.mean(edges)), lean_px=edges[-1] - edges[0],
                      hole=None if best is None else best[1])


def offset(reference: JawReading, now: JawReading) -> JawOffset:
    """How far the jaw stands from its reference place. By its hole, a point, where both readings have it: the
    inner edge leans (four tenths of a pixel across for every row), so read at fixed rows it mixes a movement down
    into the one across. Without a hole the edge alone says how far across, and nothing says how far down."""
    lean = now.lean_px - reference.lean_px
    if reference.hole is None or now.hole is None:
        return JawOffset(across_px=now.edge_px - reference.edge_px, down_px=None, lean_px=lean)
    return JawOffset(across_px=now.hole[0] - reference.hole[0], down_px=now.hole[1] - reference.hole[1], lean_px=lean)


def said(now: JawReading | None, off: JawOffset | None) -> list[str]:
    """The lines for the live picture and the terminal."""
    if now is None or off is None:
        return ["the fixed jaw cannot be seen at the bottom left of the wrist camera's picture"]
    down = "?" if off.down_px is None else f"{off.down_px:+.1f}"
    where = f"wrist camera: the jaw stands {off.across_px:+.1f} px across, {down} px down, lean {off.lean_px:+.1f}"
    if off.in_place:
        return [where, "IN PLACE (within 1 px of where it stood in training)"]
    todo = []
    if abs(off.across_px) > IN_PLACE_PX:
        todo.append(f"the jaw must come {abs(off.across_px):.1f} px to the {'LEFT' if off.across_px > 0 else 'RIGHT'}")
    if off.down_px is not None and abs(off.down_px) > IN_PLACE_PX:
        todo.append(f"the jaw must come {abs(off.down_px):.1f} px {'UP' if off.down_px > 0 else 'DOWN'}")
    if abs(off.lean_px) > IN_PLACE_PX:
        todo.append("the camera is turned about its own line of sight")
    return [where, *todo]


def laid_over(frame_bgr: np.ndarray, reference_bgr: np.ndarray, share: float = 0.5) -> np.ndarray:
    """The frame with the reference picture of the fixed jaw laid over that part of it: at 0.5 a jaw out of place is
    seen twice, at 1 the old picture alone. A thin frame marks the part that is laid over."""
    import cv2

    (top, bottom), (left, right) = OVER_ROWS, OVER_COLUMNS
    picture = frame_bgr.copy()
    picture[top:bottom, left:right] = cv2.addWeighted(frame_bgr[top:bottom, left:right], 1.0 - share,
                                                      reference_bgr[top:bottom, left:right], share, 0.0)
    cv2.rectangle(picture, (left, top), (right - 1, bottom - 1), (160, 160, 160), 1)
    return picture


def draw(frame_bgr: np.ndarray, reference: JawReading, now: JawReading | None, off: JawOffset | None,
         lines: list[str]) -> np.ndarray:
    """The frame with the jaw's place drawn in: a green line where its inner edge belongs and a green cross where
    its hole belongs, orange ones where they are (green too once in place), and `lines` at the top."""
    import cv2

    picture = frame_bgr.copy()
    good = bool(off and off.in_place)

    def marks(reading: JawReading, colour: tuple[int, int, int], thick: int) -> None:
        first, last = EDGE_ROWS[0], EDGE_ROWS[-1]
        start = reading.edge_px - reading.lean_px / 2.0
        cv2.line(picture, (round(start), first), (round(start + reading.lean_px), last), colour, thick)
        if reading.hole is not None:
            cv2.drawMarker(picture, (round(reading.hole[0]), round(reading.hole[1])), colour, cv2.MARKER_CROSS, 14,
                           thick)

    marks(reference, (80, 255, 80), 2)
    if now is not None:
        marks(now, (80, 255, 80) if good else (0, 160, 255), 1)
    for row, line in enumerate(lines):
        where = (10, 24 + 22 * row)
        cv2.putText(picture, line, where, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(picture, line, where, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80) if good else (0, 190, 255), 1)
    return picture


def steady(readings: list[JawReading]) -> JawReading | None:
    """The median of the last readings: a live edge steps by a third of a pixel from frame to frame."""
    if not readings:
        return None
    holes = [reading.hole for reading in readings if reading.hole is not None]
    hole = None if not holes else (float(np.median([h[0] for h in holes])), float(np.median([h[1] for h in holes])))
    return JawReading(edge_px=float(np.median([reading.edge_px for reading in readings])),
                      lean_px=float(np.median([reading.lean_px for reading in readings])), hole=hole)
