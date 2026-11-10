"""Putting the cube on a target given in table millimetres, and recording where it ended up.

Before every demonstration and every trial the cube is set on a planned target. This module is the loop that helps a
person do it: it shows the front camera's picture with the target marked, measures the cube in every frame, and says
how far off it is and which way to move it. The placement is accepted only when the measured position has stayed
within tolerance for a run of frames and a person confirms: with the space bar in the camera tool, or with the
session's own "go" key in an arm session. A hand next to the cube does not stop the detector from seeing the cube, so
steadiness alone is never taken as "hands are clear": what follows a placement is an arm that moves, and a person says
when. What it returns is the measured position, which is what the analysis uses; the target is only what was asked
for.

It commands nothing. It runs between control loops, while the arm holds its position, and only reads camera frames.
The frame source, the display and the keyboard are passed in, so the logic is tested without a camera or a window.

Keys are space (accept), `v` (skip this placement) and `x` (abort the session's placement loop). The arm session's own
keyboard listener is global and binds n, r, q and the arrow keys, so those are avoided here on purpose.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tolerance_radius.rig.vision import CubeMeasurement, TableView, median_position
from tolerance_radius.schema import Point

KEY_ACCEPT = ord(" ")
KEY_SKIP = ord("v")
KEY_ABORT = ord("x")


def consume_go(events: MutableMapping[str, Any]) -> bool:
    """True once when the operator pressed the session's go key and nothing else; reading it clears it.

    `events` holds the flags LeRobot's key listener sets. Its go key (n, the right arrow) sets `exit_early`; so do the
    re-record key (with `rerecord_episode`) and the stop key (with `stop_recording`), because all three end a phase.
    During a placement only the first is a go: what follows a go is an arm that moves. A re-record press means nothing
    here and is forgotten; a stop is left set for the session to act on.
    """
    if not events.get("exit_early"):
        return False
    events["exit_early"] = False
    if events.get("stop_recording"):
        return False
    if events.get("rerecord_episode"):
        events["rerecord_episode"] = False
        return False
    return True

GREEN, RED, AMBER, WHITE = (80, 255, 80), (60, 60, 255), (0, 190, 255), (255, 255, 255)
#: The grid's lines and names in the placement picture: there to be found, not to be looked at.
GRID_LINE = (150, 170, 150)
#: The magnified picture beside the camera's: this many times larger, of this many pixels either side of the target
#: (96 px across is about 115 mm on the floor at the grid, so a cube 30 mm wide is a quarter of it).
ZOOM, ZOOM_HALF_PX = 5, 48


@dataclass(frozen=True, slots=True)
class Assessment:
    """How one frame's measurement compares with the target."""

    #: "ok" (within tolerance), "off" (a clean cube, outside tolerance) or "none" (no single clean cube in view).
    status: str
    error_mm: float | None = None
    #: Measured minus target, table axes.
    dx_mm: float | None = None
    dy_mm: float | None = None
    measurement: CubeMeasurement | None = None


@dataclass(frozen=True, slots=True)
class PlacementResult:
    """The outcome of one placement."""

    #: "accepted", "skipped", "aborted", "stopped" (the session asked to stop) or "timeout".
    status: str
    target: Point
    #: The median of the measurements over the stable run that was accepted. None unless accepted.
    measured: Point | None = None
    error_mm: float | None = None
    frames_seen: int = 0

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def assess(measurement: CubeMeasurement | None, target: Point, tolerance_mm: float) -> Assessment:
    """Compare one measurement with the target."""
    if measurement is None:
        return Assessment(status="none")
    dx = measurement.position.x_mm - target.x_mm
    dy = measurement.position.y_mm - target.y_mm
    error = float(np.hypot(dx, dy))
    status = "ok" if error <= tolerance_mm else "off"
    return Assessment(status=status, error_mm=error, dx_mm=dx, dy_mm=dy, measurement=measurement)


class PlacementPhase:
    """The interactive placement loop for one session.

    `grab` returns the front camera's current frame, BGR, upright. `show(title, image)` displays a picture and
    `wait_key(ms)` returns a key code or -1; by default these are OpenCV's window functions. `clock` is a monotonic
    clock in seconds.

    The cube counts as steady when every frame for at least `hold_s` seconds, and at least `stable_frames` frames, has
    been within tolerance. A steady cube is accepted when the space bar is pressed in the window, or when `confirm`
    returns True. `confirm` is asked on every frame and must consume its own signal (an arm session passes a function
    that reads and clears its "go" key), so a press made before the cube was steady is forgotten, not remembered.
    `auto_accept` accepts a steady cube with no confirmation at all: for tests and unattended rehearsal only, never
    where an arm moves next. `should_stop` is asked every frame and ends the placement when it returns True.
    `forget`, when given, is called in place of `confirm` on every frame in which the cube is not steady: for a
    `confirm` that does not answer at once (the trial runner's holds a go for half a second to see whether it was
    typed), so that a go pressed before the cube was steady is dropped then and cannot count a moment later.
    `on_end` is called once when a placement is over, however it ended (a display uses it to take its picture away);
    a failure in it is logged and does not cost the placement.
    """

    def __init__(
        self,
        view: TableView,
        grab: Callable[[], np.ndarray],
        *,
        tolerance_mm: float,
        stable_frames: int = 8,
        hold_s: float = 0.0,
        auto_accept: bool = False,
        confirm: Callable[[], bool] | None = None,
        accept_hint: str = "press space to accept",
        should_stop: Callable[[], bool] | None = None,
        on_end: Callable[[], None] | None = None,
        timeout_s: float = 600.0,
        forget: Callable[[], Any] | None = None,
        show: Callable[[str, np.ndarray], None] | None = None,
        wait_key: Callable[[int], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        window: str = "placement  [space] accept  [v] skip  [x] abort",
        grid: dict[str, Any] | None = None,
    ) -> None:
        if not view.has_background:
            raise ValueError("the view needs an empty-scene background before placements can be measured")
        if tolerance_mm <= 0 or stable_frames < 1 or hold_s < 0:
            raise ValueError("tolerance and stable_frames are positive, and hold_s is not negative")
        self._view, self._grab = view, grab
        self._tolerance_mm, self._stable_frames, self._timeout_s = tolerance_mm, stable_frames, timeout_s
        self._hold_s, self._auto_accept, self._should_stop = hold_s, auto_accept, should_stop
        self._confirm, self._accept_hint, self._on_end = confirm, accept_hint, on_end
        self._forget = forget
        self._clock, self._window = clock, window
        #: The grid's `cols`, `rows` and `cell_mm`, drawn into the picture when given: it is not drawn on the floor.
        self._grid = grid
        if show is None or wait_key is None:
            import cv2  # only needed for the real window

            if show is None and on_end is None:
                # The session's own window: taken away when a placement is over, so that it does not stand there
                # with a still picture while the arm moves.
                def close_window() -> None:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)

                self._on_end = close_window
            show = show or cv2.imshow
            wait_key = wait_key or cv2.waitKey
        self._show, self._wait_key = show, wait_key

    def run(self, target: Point, label: str = "") -> PlacementResult:
        """Guide one placement. Returns when it is accepted, skipped, aborted, stopped or timed out."""
        try:
            return self._run(target, label)
        finally:
            if self._on_end is not None:
                try:
                    self._on_end()
                except Exception as error:  # noqa: BLE001 -- tidying a display must never cost a placement
                    logging.warning("the end of the placement could not be announced (%s: %s)",
                                    type(error).__name__, error)

    def _run(self, target: Point, label: str) -> PlacementResult:
        run_of_ok: deque[Assessment] = deque(maxlen=max(self._stable_frames, 30))
        ok_since: float | None = None
        deadline = self._clock() + self._timeout_s
        frames_seen = 0
        while self._clock() < deadline:
            if self._should_stop is not None and self._should_stop():
                return PlacementResult("stopped", target, frames_seen=frames_seen)
            frame = self._grab()
            frames_seen += 1
            now = self._clock()
            current = assess(self._view.measure(frame), target, self._tolerance_mm)
            if current.status == "ok":
                ok_since = now if ok_since is None else ok_since
                run_of_ok.append(current)
            else:
                ok_since = None
                run_of_ok.clear()
            stable = (
                ok_since is not None and len(run_of_ok) >= self._stable_frames and now - ok_since >= self._hold_s
            )
            self._show(
                self._window,
                render(frame, self._view, target, current, self._tolerance_mm, stable, label, self._accept_hint,
                       grid=self._grid),
            )
            key = self._wait_key(30) & 0xFF
            # Asked on every frame, steady or not, so that an early press is consumed and cannot fire later. Where
            # the answer to a press takes time, the session gives `forget` for the frames that are not steady.
            if stable or self._forget is None:
                confirmed = self._confirm is not None and self._confirm()
            else:
                self._forget()
                confirmed = False
            if key == KEY_ABORT:
                return PlacementResult("aborted", target, frames_seen=frames_seen)
            if key == KEY_SKIP:
                return PlacementResult("skipped", target, frames_seen=frames_seen)
            if stable and (self._auto_accept or key == KEY_ACCEPT or confirmed):
                # A stop asked for during this frame wins over anything that looks like a confirmation. In an arm
                # session the stop key sets the same flag as the go key, so without this a stop pressed while the
                # cube is steady would be taken as the go, and what follows a go is an arm that moves.
                if self._should_stop is not None and self._should_stop():
                    return PlacementResult("stopped", target, frames_seen=frames_seen)
                measured = median_position([a.measurement for a in run_of_ok if a.measurement is not None])
                return PlacementResult("accepted", target, measured=measured,
                                       error_mm=measured.distance_to(target), frames_seen=frames_seen)
        return PlacementResult("timeout", target, frames_seen=frames_seen)


def grid_lines_mm(grid: dict[str, Any]) -> list[tuple[Point, Point]]:
    """The grid's lines as pairs of table points: its outline and the lines between its cells."""
    cols, rows, cell = int(grid["cols"]), int(grid["rows"]), float(grid["cell_mm"])
    lines = [(Point(i * cell, 0.0), Point(i * cell, rows * cell)) for i in range(cols + 1)]
    lines += [(Point(0.0, j * cell), Point(cols * cell, j * cell)) for j in range(rows + 1)]
    return lines


def cell_names_mm(grid: dict[str, Any]) -> list[tuple[str, Point]]:
    """Every cell's name and centre: columns A, B, ... along x, rows 1, 2, ... along y."""
    cols, rows, cell = int(grid["cols"]), int(grid["rows"]), float(grid["cell_mm"])
    return [(f"{chr(ord('A') + col)}{row + 1}", Point((col + 0.5) * cell, (row + 0.5) * cell))
            for row in range(rows) for col in range(cols)]


def render(
    frame_bgr: np.ndarray,
    view: TableView,
    target: Point,
    current: Assessment,
    tolerance_mm: float,
    stable: bool,
    label: str = "",
    accept_hint: str = "press space to accept",
    *,
    grid: dict[str, Any] | None = None,
) -> np.ndarray:
    """The picture shown to the person placing the cube: the camera's frame with the target marked, and beside it
    the surroundings of the target magnified.

    In both: the ring the cube's measured centre has to come into, and **the cube's own outline moved to where it
    belongs** (the outline the detector sees now, shifted so that its centre lies on the target). Sliding the cube
    into that outline is the placement; nothing is modelled, so the outline and the ring cannot disagree. The
    magnified picture also has two lines through the target along the grid's two directions: the cube's sides are
    kept along them. With `grid`, the grid's lines and the cells' names are drawn, which are not on the floor.
    """
    import cv2

    colour = GREEN if stable else AMBER if current.status == "ok" else RED
    target_px = np.asarray(view.to_px(target), dtype=float)
    ring_px = max(2.0, tolerance_mm * view.px_per_mm_at(target))
    measurement = current.measurement
    cube_px = None if measurement is None else np.asarray(measurement.centroid_px, dtype=float)
    outline = None if measurement is None or measurement.outline_px is None else np.asarray(measurement.outline_px)
    if outline is not None and len(outline) >= 3:
        # The detector's patch follows every finger and shadow that cuts into the cube. Its hull is the cube's own
        # shape with those dents closed: steadier to look at, and the same thing once the hand is away.
        outline = cv2.convexHull(outline.astype(np.float32)).reshape(-1, 2).astype(float)
    ghost = None if outline is None or cube_px is None else outline + (target_px - cube_px)
    lines_px = [] if grid is None else [(np.asarray(view.to_px(a)), np.asarray(view.to_px(b)))
                                        for a, b in grid_lines_mm(grid)]
    axes_px = [(np.asarray(view.to_px(Point(target.x_mm - 70.0, target.y_mm))),
                np.asarray(view.to_px(Point(target.x_mm + 70.0, target.y_mm)))),
               (np.asarray(view.to_px(Point(target.x_mm, target.y_mm - 70.0))),
                np.asarray(view.to_px(Point(target.x_mm, target.y_mm + 70.0))))]

    def whole(point) -> tuple[int, int]:
        return int(round(float(point[0]))), int(round(float(point[1])))

    # ---- the camera's picture
    picture = frame_bgr.copy()
    for a, b in lines_px:
        cv2.line(picture, whole(a), whole(b), GRID_LINE, 1, cv2.LINE_AA)
    if grid is not None:
        for name, centre in cell_names_mm(grid):
            x, y = view.to_px(centre)
            cv2.putText(picture, name, (int(x) - 9, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRID_LINE, 1,
                        cv2.LINE_AA)
    if ghost is not None:
        cv2.polylines(picture, [np.round(ghost).astype(np.int32)], True, colour, 1, cv2.LINE_AA)
    cv2.circle(picture, whole(target_px), int(round(ring_px)), colour, 1)
    cv2.drawMarker(picture, whole(target_px), colour, cv2.MARKER_CROSS, 16, 1)
    if cube_px is not None:
        cv2.circle(picture, whole(cube_px), 3, WHITE, -1)
        cv2.arrowedLine(picture, whole(cube_px), whole(target_px), colour, 1, tipLength=0.25)
    for row, text in enumerate(_lines(current, stable, label, accept_hint)):
        cv2.putText(picture, text, (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(picture, text, (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # ---- the target's surroundings, magnified
    side = 2 * ZOOM_HALF_PX * ZOOM
    corner = target_px - ZOOM_HALF_PX
    enlarge = np.array([[ZOOM, 0.0, -corner[0] * ZOOM], [0.0, ZOOM, -corner[1] * ZOOM]])
    close = cv2.warpAffine(frame_bgr, enlarge, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(40, 40, 40))

    def near(point) -> tuple[int, int]:
        return whole((np.asarray(point, dtype=float) - corner) * ZOOM)

    for a, b in lines_px:
        cv2.line(close, near(a), near(b), GRID_LINE, 1, cv2.LINE_AA)
    for a, b in axes_px:
        cv2.line(close, near(a), near(b), WHITE, 1, cv2.LINE_AA)
    cv2.circle(close, near(target_px), int(round(ring_px * ZOOM)), colour, 2, cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(close, [np.array([near(point) for point in outline], dtype=np.int32)], True, WHITE, 1,
                      cv2.LINE_AA)
    if ghost is not None:
        cv2.polylines(close, [np.array([near(point) for point in ghost], dtype=np.int32)], True, colour, 2,
                      cv2.LINE_AA)
    if cube_px is not None:
        cv2.arrowedLine(close, near(cube_px), near(target_px), colour, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(close, near(cube_px), 5, WHITE, -1, cv2.LINE_AA)
    if current.status == "none":
        words = ["no cube seen here"]
    else:
        words = [f"dx {current.dx_mm:+.1f}   dy {current.dy_mm:+.1f} mm   ({current.error_mm:.1f} off)",
                 "steady: hands out, then the go key" if stable else
                 "inside the ring: hold still" if current.status == "ok" else
                 "slide it into the outline, sides along the white lines"]
    for row, text in enumerate(words):
        scale = 0.6 if row == 0 else 0.45
        cv2.putText(close, text, (10, side - 34 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(close, text, (10, side - 34 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1 + (row == 0))
    cv2.rectangle(close, (0, 0), (side - 1, side - 1), colour, 3)

    height = max(picture.shape[0], side)
    both = np.full((height, picture.shape[1] + side, 3), 40, np.uint8)
    both[: picture.shape[0], : picture.shape[1]] = picture
    both[:side, picture.shape[1]:] = close
    return both


def _lines(current: Assessment, stable: bool, label: str, accept_hint: str) -> list[str]:
    head = [label] if label else []
    if current.status == "none":
        return [*head, "no single clean cube in view (hands out of the picture)"]
    reading = f"off {current.error_mm:.1f} mm   dx {current.dx_mm:+.1f}  dy {current.dy_mm:+.1f}"
    if stable:
        return [*head, reading, f"steady, hands clear: {accept_hint}"]
    if current.status == "ok":
        return [*head, reading, "within tolerance: hold still"]
    return [*head, reading, "follow the arrow"]
