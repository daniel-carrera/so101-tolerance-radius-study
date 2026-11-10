"""The measuring instrument: where the cube is on the table, from the front camera.

One instrument measures every start position in the study: the demonstrations' (in the audit), the targets a cube is
placed on, and the placements of the trials. Because it is the same detector and the same table frame each time, a
constant offset in what it calls "the cube's position" cancels in every distance between two measurements.

It is built from two pieces of the earlier project, used as they are: the AprilTag detection that ties pixels to table
millimetres through the two tags, and the colour-and-change detector that finds the orange cube.

The tie between pixels and the table is solved by every session when it starts, from the two floor tags' outlines
averaged over its first frames (`camera_tool.checked_stored_view`). The table frame is the tags': they are taped to the
floor, while the camera of this rig hangs on the light box's walls and moved by 2 mm within an hour of its calibration
(2026-10-06). The calibration's own solve is stored (`TableView.save`) as the reference a session is held against: how
far the camera's picture has moved, and the picture's levels.

It was the other way round until that afternoon: one stored conversion for every measurement, and a session that only
checked a fresh solve against it. That was written when a tag's corners were the detector's own and one frame's solve
wandered by up to 3 mm; measured by their outlines, single frames lie within 0.06 mm of the frame of forty. What a
fresh solve still costs: when the picture shifts by part of a pixel, a solve from two tags moves by up to about 0.5 mm
somewhere on the grid (synthetic pictures of this view), and that is now in every distance between two sessions.

Known property: the detector reports the centre of the cube's visible outline (its top and the sides the camera sees),
mapped through the table plane. For a 30 mm cube seen at an angle that point is displaced from the footprint's centre,
by an amount that changes slowly across the table. Distances over a few centimetres are therefore accurate to about a
millimetre, not exact: the instrument check in `protocol/protocol.md` (block V0) measures this against a physical scale.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius.config import REPO_ROOT
from tolerance_radius.rig import lerobot101, outline
from tolerance_radius.schema import Point

#: This rig's table geometry: where the two floor tags are, in the grid's millimetres. It is written by the table
#: registration (`scripts/rig/measure.py register-table`, `docs/runbook/00-table-layout.md`). The earlier project's
#: `eval/table_geometry.json` describes the table of August, which was taken down: it is not read here.
TABLE_GEOMETRY = REPO_ROOT / "config" / "table_geometry.json"

#: How many known tags a frame must show before pixels can be tied to the table. One tag fits any view.
TAGS_NEEDED = 2


class VisionError(RuntimeError):
    """The camera view cannot be tied to the table, or the detector cannot run."""


@dataclass(frozen=True, slots=True)
class CubeMeasurement:
    """One reading of the cube's position."""

    position: Point
    #: Centre of the detected outline in the image, in pixels.
    centroid_px: tuple[float, float]
    area_px: int
    #: The outline itself, as the detector saw it: its corner points in pixels, N x 2, or None. For showing a person
    #: where the cube is taken to be; nothing is measured from it, and two readings are not compared by it.
    outline_px: np.ndarray | None = field(default=None, compare=False, repr=False)


#: The cube is looked for on the grid and this far around it, and nowhere else (`TableView.with_cube_region`). A cube
#: whose centre is 5 mm inside the grid's far edge reaches 10 mm beyond it, and its top face, seen from a camera 64 cm
#: up and half a metre away, another 25 mm: 35 mm in all at the worst place, so 55 leaves room. On this rig the basket
#: is then 43 px clear of the region (*measured* on a recorded frame of 2026-10-06). The arm is not always: at rest
#: turned aside, as in the recorder's third run, it was 44 px clear; folded and facing the grid, as since the fourth,
#: its gripper's tip reaches about 9 px into the region, with 120 to 160 cube-coloured pixels there (*measured* on
#: sitting 1's frames). That is under the 246 pixels the detector asks of a cube, and the cube read the same to
#: 0.01 mm with the arm resting 5.8 degrees apart in pan and 10.7 in roll (nine pairs of recorded frames). An arm
#: whose lift, elbow or wrist flex is off reaches further in: what the recorder's start-pose check holds to 5 degrees.
CUBE_REGION_MARGIN_MM = 55.0


class TableView:
    """The front camera's view of the table: converts pixels to table millimetres and finds the cube.

    A session builds one when it starts, from frames in which both tags are visible, then gives it an empty-scene
    background with `with_background`. Within a session the conversion is kept: a camera that moves while a session
    runs is not followed.
    """

    def __init__(
        self,
        px_to_mm: np.ndarray,
        background_bgr: np.ndarray | None = None,
        *,
        residual_mm: float | None = None,
        frames_used: int | None = None,
        floor_levels: dict[str, float] | None = None,
        geometry_created: str | None = None,
        region: np.ndarray | None = None,
    ) -> None:
        matrix = np.asarray(px_to_mm, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
            raise VisionError("the pixel-to-table transform must be an invertible 3 x 3 matrix")
        self._px_to_mm = matrix
        self._mm_to_px = np.linalg.inv(matrix)
        self._background = None if background_bgr is None else np.asarray(background_bgr).copy()
        #: Where in the picture a cube is looked for (True), or None for everywhere: see `with_cube_region`.
        self._region = None if region is None else np.asarray(region, dtype=bool).copy()
        #: How well the tags' stored corner positions fit one flat table through this view: the root-mean-square
        #: distance, in table millimetres, between where the fit puts each tag corner and where the geometry file
        #: says it is. It grows when a tag has moved against the other since the geometry was measured.
        self.residual_mm = residual_mm
        #: How many frames the tag corners were averaged over.
        self.frames_used = frames_used
        #: How bright and how warm the floor inside the grid was when this view was stored (`floor_levels`), or None.
        self.floor_levels = None if floor_levels is None else dict(floor_levels)
        #: The `created` stamp of the table geometry this view was solved against: a view stored for one registration
        #: must not be measured through after another.
        self.geometry_created = geometry_created

    # ------------------------------------------------------------------------------------------------ construction

    @classmethod
    def from_frames(cls, frames_bgr: Iterable[np.ndarray], geometry: dict[str, Any] | None = None) -> TableView:
        """Tie the view to the table from frames that show both tags, averaging the tags' corners over the frames.

        `geometry` is the table geometry document (tag corners in table millimetres); by default this rig's own,
        `config/table_geometry.json`. A tag's corners are those of its outline measured from the picture
        (`outline.py`), not the detector's own, which sit a few tenths of a pixel off by an amount that depends on
        where the pixels fall. Frames in which a known tag is not found, or its outline cannot be measured, are
        skipped for that tag; every known tag must be measured at least once.
        """
        geometry = load_geometry() if geometry is None else geometry
        tags = {int(tag_id): corners for tag_id, corners in geometry["tags"].items()}
        mean_quads, found_in, measured_in, count = tag_outlines_over(frames_bgr, tags)
        missing = sorted(tag_id for tag_id in tags if not found_in[tag_id])
        if len(tags) < TAGS_NEEDED or missing:
            raise VisionError(
                f"need every known tag in view to place the table frame; tags {missing or sorted(tags)} were not "
                f"found in {count} frame(s). Is the arm hiding a tag, or has the camera moved?"
            )
        unmeasured = sorted(tag_id for tag_id in tags if not measured_in[tag_id])
        if unmeasured:
            raise VisionError(
                f"tags {unmeasured} are in view but their outline could not be measured in {count} frame(s). Is "
                "something lying across a tag (a cable, the arm's shadow), or is it in glare?"
            )
        matrix = lerobot101.load("placement").homography_from_tag_geometry(mean_quads, tags, geometry["tag_mm"])
        if matrix is None:
            raise VisionError("the tags in view do not determine the table frame")
        return cls(matrix, residual_mm=_fit_residual_mm(matrix, mean_quads, tags), frames_used=count,
                   geometry_created=geometry.get("created"))

    @classmethod
    def from_frame(cls, frame_bgr: np.ndarray, geometry: dict[str, Any] | None = None) -> TableView:
        """As `from_frames`, from a single frame. Good for a look; too noisy to measure with."""
        return cls.from_frames([frame_bgr], geometry)

    # -------------------------------------------------------------------------------------------------- stored view

    def save(self, path: str | Path, *, note: str = "", record: dict[str, Any] | None = None) -> None:
        """Store this view's conversion, so that every session measures through the same one.

        `record` is written beside it and read by nobody here: what the calibration saw when it was made (the frame
        it kept, where the floor tags were in the picture, how they had moved since the registration).
        """
        document = {
            "px_to_mm": self._px_to_mm.tolist(),
            "residual_mm": self.residual_mm,
            "frames_used": self.frames_used,
            "floor_levels": self.floor_levels,
            "geometry_created": self.geometry_created,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": note,
        }
        if record:
            document["record"] = record
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TableView:
        """The stored view."""
        path = Path(path)
        if not path.is_file():
            raise VisionError(f"no stored table view at {path}: run `scripts/rig/measure.py calibrate` first")
        document = json.loads(path.read_text(encoding="utf-8"))
        return cls(document["px_to_mm"], residual_mm=document.get("residual_mm"),
                   frames_used=document.get("frames_used"), floor_levels=document.get("floor_levels"),
                   geometry_created=document.get("geometry_created"))

    def disagreement_mm(self, other: TableView, probes_px: Sequence[Sequence[float]]) -> float:
        """The largest distance, in table millimetres, between where two views put the same pixels."""
        return max(self.to_mm(pixel).distance_to(other.to_mm(pixel)) for pixel in probes_px)

    def with_background(self, background_bgr: np.ndarray) -> TableView:
        """The same view with an empty-scene picture, which the cube detector needs."""
        return TableView(self._px_to_mm, background_bgr, residual_mm=self.residual_mm, frames_used=self.frames_used,
                         floor_levels=self.floor_levels, geometry_created=self.geometry_created, region=self._region)

    def with_cube_region(self, grid: dict[str, Any], margin_mm: float = CUBE_REGION_MARGIN_MM) -> TableView:
        """The same view, looking for the cube only on the grid and `margin_mm` around it.

        The detector finds the cube in what has changed since the empty-scene picture and is cube-coloured. The arm is
        cube-coloured too: resting a centimetre from where it rested in that picture it is a second such thing, and
        the detector then reports no cube at all. That made the arm's rest pose something to be hit to two degrees
        before every placement (the recorder's second run at the rig, 2026-10-06). Outside the region the picture is
        taken to be the empty scene's, so nothing there counts as changed: the arm may rest where it likes as long
        as it stays out of the region, and so may a cube left in the basket.
        """
        if self._background is None:
            raise VisionError("the cube region needs the empty-scene background first: call with_background")
        import cv2

        cols, rows, cell = int(grid["cols"]), int(grid["rows"]), float(grid["cell_mm"])
        width, depth, m = cols * cell, rows * cell, float(margin_mm)
        corners = [self.to_px(Point(x, y)) for x, y in ((-m, -m), (width + m, -m), (width + m, depth + m), (-m, depth + m))]
        mask = np.zeros(self._background.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(np.asarray(corners)).astype(np.int32), 255)
        return TableView(self._px_to_mm, self._background, residual_mm=self.residual_mm, frames_used=self.frames_used,
                         floor_levels=self.floor_levels, geometry_created=self.geometry_created, region=mask > 0)

    @property
    def has_background(self) -> bool:
        return self._background is not None

    @property
    def px_to_mm(self) -> np.ndarray:
        """The conversion itself: a 3 x 3 matrix from pixels to table millimetres."""
        return self._px_to_mm.copy()

    # -------------------------------------------------------------------------------------------------- conversion

    def to_mm(self, xy_px: Sequence[float]) -> Point:
        """A pixel as a point on the table."""
        x, y, w = self._px_to_mm @ np.array([xy_px[0], xy_px[1], 1.0])
        return Point(float(x / w), float(y / w))

    def to_px(self, point: Point) -> tuple[float, float]:
        """A point on the table as a pixel."""
        x, y, w = self._mm_to_px @ np.array([point.x_mm, point.y_mm, 1.0])
        return float(x / w), float(y / w)

    def px_per_mm_at(self, point: Point) -> float:
        """Image scale near a table point: pixels per millimetre, averaged over the two table axes."""
        origin = np.array(self.to_px(point))
        along_x = np.array(self.to_px(Point(point.x_mm + 10.0, point.y_mm)))
        along_y = np.array(self.to_px(Point(point.x_mm, point.y_mm + 10.0)))
        return float((np.linalg.norm(along_x - origin) + np.linalg.norm(along_y - origin)) / 20.0)

    # ------------------------------------------------------------------------------------------------- measurement

    def measure(self, frame_bgr: np.ndarray) -> CubeMeasurement | None:
        """Where the cube is in this frame, or None when the detector does not see exactly one clean cube.

        None is a real answer: a hand in view, no cube yet, or two orange things. Guiding a placement toward a hand
        would be worse than refusing.
        """
        if self._background is None:
            raise VisionError("the cube detector needs an empty-scene background: call with_background first")
        if frame_bgr.shape != self._background.shape:
            raise VisionError(f"frame is {frame_bgr.shape}, the background is {self._background.shape}")
        looked_at = frame_bgr
        if self._region is not None:
            # outside the region the picture is the empty scene's: nothing there has changed
            looked_at = np.where(self._region[:, :, None], frame_bgr, self._background)
        detection = lerobot101.load("placement").detect_object(looked_at, self._background)
        if detection is None or detection.get("n_candidates") != 1:
            return None
        centroid = (float(detection["centroid"][0]), float(detection["centroid"][1]))
        return CubeMeasurement(position=self.to_mm(centroid), centroid_px=centroid, area_px=int(detection["area"]),
                               outline_px=_outline_of(detection.get("mask")))


def _outline_of(mask: np.ndarray | None) -> np.ndarray | None:
    """The outer outline of the detector's patch of cube-coloured pixels, N x 2 in pixels, or None. A picture for a
    person: a failure here must never cost a measurement."""
    if mask is None:
        return None
    try:
        import cv2

        contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    except Exception:  # noqa: BLE001 -- see the docstring
        return None


def tag_outlines_over(
    frames_bgr: Iterable[np.ndarray], tag_ids: Iterable[int]
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, int], int]:
    """The given tags' outline corners averaged over a run of frames.

    Returns the mean corners of each tag that was measured at least once, in how many frames the detector found each
    tag, in how many its outline could be measured, and the number of frames.
    """
    import cv2

    detect = lerobot101.load("tag_check").detect
    wanted = [int(tag_id) for tag_id in tag_ids]
    quads: dict[int, list[np.ndarray]] = {tag_id: [] for tag_id in wanted}
    found_in = dict.fromkeys(wanted, 0)
    count = 0
    for frame in frames_bgr:
        count += 1
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = {tag_id: quad for tag_id, quad in detect(gray).items() if tag_id in quads}
        for tag_id in found:
            found_in[tag_id] += 1
        for tag_id, quad in outline.tag_outlines(gray, found).items():
            quads[tag_id].append(np.asarray(quad, dtype=float))
    mean_quads = {tag_id: np.mean(np.stack(seen), axis=0) for tag_id, seen in quads.items() if seen}
    return mean_quads, found_in, {tag_id: len(seen) for tag_id, seen in quads.items()}, count


def _fit_residual_mm(px_to_mm: np.ndarray, quads_px: dict[int, np.ndarray], tags: dict[int, Any]) -> float:
    """Root-mean-square distance between the fitted and the stored positions of the tag corners, in millimetres."""
    errors: list[float] = []
    for tag_id, quad in quads_px.items():
        stored = tags[tag_id]
        corners_mm = np.asarray(stored.get("corners") if isinstance(stored, dict) else stored, dtype=float)
        if corners_mm.shape != (4, 2):
            continue  # a tag given only by its centre carries no corner positions to compare with
        homogeneous = np.hstack([quad, np.ones((4, 1))]) @ px_to_mm.T
        fitted = homogeneous[:, :2] / homogeneous[:, 2:]
        errors.extend(np.linalg.norm(fitted - corners_mm, axis=1).tolist())
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def floor_levels(
    frame_bgr: np.ndarray,
    corners_px: Sequence[Sequence[float]],
    tag_quads_px: Sequence[Sequence[Sequence[float]]] | None = None,
) -> dict[str, float]:
    """How bright and how warm the floor inside the grid is in one frame, and how dark the floor tags' black.

    `grey` is the median grey level (0 to 255) of the pixels inside the grid's outline, `red_over_blue` the median of
    red over blue there. They are what a camera's exposure and white balance set: a camera left on automatic, or one
    whose settings were lost when it was unplugged, shows here while every tag is still where it was. A cube or the
    arm on the grid covers a small part of it and does not move a median.

    With `tag_quads_px` (the floor tags' corners in the picture) it adds `tag_black`: the grey level of the tags'
    black, the mean over the tags of the median of each tag's darker half. White against black is the picture's
    contrast, and that is what tells a picture handed over in another tone state (seen on 2026-10-06: the same
    camera picture stretched from video range to full range on some openings, whites up by 7 % and blacks halved)
    from a lamp or an exposure that changed, which moves both the same way.
    """
    import cv2

    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.asarray(corners_px, dtype=float)).astype(np.int32), 255)
    inside = frame_bgr[mask > 0].astype(float)
    if len(inside) < 100:
        raise VisionError("the grid covers too little of the picture to read the floor's brightness")
    blue, green, red = inside[:, 0], inside[:, 1], inside[:, 2]
    grey = 0.114 * blue + 0.587 * green + 0.299 * red
    levels = {"grey": round(float(np.median(grey)), 1),
              "red_over_blue": round(float(np.median(red / np.maximum(blue, 1.0))), 3)}
    blacks = []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    for quad in tag_quads_px or ():
        quad = np.asarray(quad, dtype=float).reshape(4, 2)
        centre = quad.mean(axis=0)
        tag_mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(tag_mask, np.round(centre + 0.9 * (quad - centre)).astype(np.int32), 255)
        pixels = gray[tag_mask > 0].astype(float)
        if len(pixels) >= 50:
            blacks.append(float(np.median(pixels[pixels <= np.median(pixels)])))
    if blacks:
        levels["tag_black"] = round(float(np.mean(blacks)), 1)
    return levels


#: The wrist camera has no tag in its picture and sets its own exposure. With the arm at rest in its start pose it
#: looks down at the white floor past its own two jaws, which lie in the same place of its picture whatever the arm
#: does. Its stream has two tone states (2026-10-06): the usual one, and the same picture with its tone range
#: stretched, 16 to 0 and 235 to 255, which the stream fell into on a USB link that was dropping frames. Two levels
#: of the rest picture tell them apart: the floor's grey and the green of the jaws.
#: *Measured* on 2,893 frames of the recorded takes of 2026-10-06 in which the joints were at the take's own start
#: pose (1,215 in the usual state, four sessions between 17:51 and 21:12; 1,678 in the other, two sessions). In the
#: usual state the two levels lie on one line as the camera's exposure wanders, green = -52.5 + 0.659 floor (the
#: floor from 187 to 203; no frame further than 2.1 levels of green from the line, and none further than 2.2 from
#: a line fitted without its own session). The stretch moves a point of that line 15 levels of green below it; the
#: other state's frames lie 12.7 to 17.8 below.
#: The number the sessions use is the plain one, the floor over the jaws' green, with the green first brought along
#: that line to a floor of `WRIST_FLOOR_REFERENCE`, so that the exposure does not move it: `WRIST_USUAL_READS` in
#: the usual state and `WRIST_OTHER_READS` in the other on those frames, 17 % apart at the nearest. The line
#: between them is `WRIST_OTHER_ABOVE`, 8 % from either. Those are saved frames; a session reads live ones, which
#: read about one percent higher (LeRobot's encoding lowers the white by a level and leaves the jaws' green:
#: *measured* on the two wrist frames kept without loss).
#: Two rules came before this one in one night, and neither ran at the rig. The first went by the picture's
#: brightest hundredth, taken from the takes' first frames and asked at their ends, where it reads 5 to 10 levels
#: lower: three of ten good takes thrown away. The second was the plain ratio, 2.47 to 2.65 against 2.84 to 3.09:
#: it falls by 0.009 for every level the floor is brighter, so that a floor 12 levels darker than any seen would
#: have read as the other state (the review of that night).
#: **The number says something only with the arm at rest in its start pose.** Anywhere else the jaws are lit
#: another way: in the frames from the middle of those takes that still showed the jaws over a plain floor, some of
#: each state read as the other. So the sessions ask only when the joints are at the start pose; a session is
#: refused there by the line, and after that a take only when the number has also stepped away from the session's
#: own first reading by `WRIST_STEP` (two frames of one state lay up to 8.8 % apart, a frame of each at least 16.7 %).
WRIST_SLOPE = 0.659
WRIST_FLOOR_REFERENCE = 195.0
WRIST_USUAL_READS = (2.50, 2.64)
WRIST_OTHER_READS = (3.08, 3.35)
WRIST_OTHER_ABOVE = 2.85
WRIST_STEP = 1.10
#: What makes a picture the rest picture, as far as the picture can say: the jaws take up this share of it (7.4 to
#: 7.8 % in every take at rest); at least this share of the window above them is floor, and that floor is even (its
#: grey scatters by 2.4 to 3.5 levels at rest, by 22 and more where the arm was parked looking at the box's wall or
#: its lamps, 16:46 and 17:23); and the number lies where the two states can put it, with room either side. A
#: picture that fails any of these is not judged.
WRIST_JAW_SHARE = (0.065, 0.09)
WRIST_FLOOR_SHARE = 0.5
WRIST_FLOOR_EVEN = 8.0
WRIST_PLAUSIBLE = (2.20, 3.80)


def wrist_reads() -> str:
    """What the wrist camera's number reads in each of its two states, as words for the operator."""
    return (f"{WRIST_USUAL_READS[0]:.2f} to {WRIST_USUAL_READS[1]:.2f} in the usual tone state, "
            f"{WRIST_OTHER_READS[0]:.2f} to {WRIST_OTHER_READS[1]:.2f} in the other")


def wrist_levels(frame: np.ndarray) -> dict[str, float]:
    """The wrist camera's picture by three numbers: its darkest and its brightest hundredth, and its mean."""
    grey = np.asarray(frame, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    return {"darkest": round(float(np.percentile(grey, 1)), 1), "brightest": round(float(np.percentile(grey, 99)), 1),
            "mean": round(float(grey.mean()), 1)}


def wrist_contrast(frame_bgr: np.ndarray) -> float | None:
    """The wrist camera's rest picture by one number: the floor's grey over the green of the jaws, the green
    brought to one floor level along the line the usual state keeps (`WRIST_SLOPE`), so that the camera's exposure
    does not move the number.

    None when the picture is not the rest picture as far as it can tell (the jaws are not where they are with the
    arm at rest, the window above them is not an even floor, or the number is nowhere near either state's): then
    nothing can be said from it. The jaws are the orange in the picture's lowest third; the floor is read in a
    window at the picture's middle, clear of the cube, which shows at the top when it lies near the gripper. A
    picture can pass all of this with the arm elsewhere: the caller asks only when the joints are at the start pose.
    """
    picture = np.asarray(frame_bgr, dtype=np.float32)
    if picture.ndim != 3 or picture.shape[2] < 3:
        return None
    blue, green, red = picture[..., 0], picture[..., 1], picture[..., 2]
    height, width = blue.shape
    jaws = (red - blue > 90.0) & (red > 150.0)
    jaws[: int(height * 2 / 3)] = False
    window = (slice(int(height * 0.375), int(height * 0.625)), slice(int(width * 0.3125), int(width * 0.6875)))
    floor = np.abs(red[window] - blue[window]) < 40.0
    low, high = WRIST_JAW_SHARE
    if not low <= jaws.mean() <= high or floor.mean() < WRIST_FLOOR_SHARE:
        return None
    floor_grey = ((blue + green + red) / 3.0)[window][floor]
    jaw_green = float(np.median(green[jaws]))
    if float(floor_grey.std()) > WRIST_FLOOR_EVEN or jaw_green < 1.0:
        return None
    floor_level = float(np.median(floor_grey))
    green_at_reference = jaw_green - WRIST_SLOPE * (floor_level - WRIST_FLOOR_REFERENCE)
    if green_at_reference < 1.0:
        return None
    contrast = WRIST_FLOOR_REFERENCE / green_at_reference
    return contrast if WRIST_PLAUSIBLE[0] <= contrast <= WRIST_PLAUSIBLE[1] else None


def wrist_other_state(contrast: float | None) -> str | None:
    """For a session's start: the sentence when the wrist camera's rest picture reads its other tone state.

    None when it reads the usual one, and when it could not be judged (`wrist_contrast` gave None).
    """
    if contrast is None or contrast <= WRIST_OTHER_ABOVE:
        return None
    return (f"the wrist camera's picture is not in its usual tone state: the floor over the jaws' green reads "
            f"{contrast:.2f} ({wrist_reads()})")


def wrist_changed(start: float | None, now: float | None) -> str | None:
    """After a take, before a trial: the sentence when the wrist camera's picture has changed its tone state since
    the session's own first reading. None when it has not, and when either reading could not be judged.

    Both must hold: the two readings lie either side of the line, and more than a step apart.
    """
    if start is None or now is None:
        return None
    if (start > WRIST_OTHER_ABOVE) == (now > WRIST_OTHER_ABOVE):
        return None
    if max(start, now) / max(min(start, now), 1e-6) <= WRIST_STEP:
        return None
    return (f"the wrist camera's picture has changed its tone state since this session began: the floor over the "
            f"jaws' green reads {now:.2f} now and read {start:.2f} then ({wrist_reads()})")


def grid_probes_px(view: TableView, geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """Pixels of the grid's four corners and its centre: where two views of the table are compared."""
    cols, rows, cell = geometry["grid"]["cols"], geometry["grid"]["rows"], geometry["grid"]["cell_mm"]
    width, depth = cols * cell, rows * cell
    points = [Point(0, 0), Point(width, 0), Point(width, depth), Point(0, depth), Point(width / 2, depth / 2)]
    return [view.to_px(point) for point in points]


def load_geometry(path: str | Path | None = None) -> dict[str, Any]:
    """The table geometry document: by default this rig's own, `config/table_geometry.json`."""
    if path is None and not TABLE_GEOMETRY.is_file():
        raise VisionError(
            f"this rig has no table geometry yet ({TABLE_GEOMETRY.name}): the two floor tags have to be measured "
            "against the registration sheet first (docs/runbook/00-table-layout.md)"
        )
    path = Path(path) if path is not None else TABLE_GEOMETRY
    if not path.is_file():
        raise VisionError(f"no table geometry at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def empty_scene(grab: Callable[[], np.ndarray], frames: int = 15) -> np.ndarray:
    """A picture of the scene with no cube in it: the per-pixel median of several frames.

    The caller makes sure the cube is off the table and the arm is where it will be during placements.
    """
    if frames < 1:
        raise ValueError("at least one frame")
    stack = np.stack([np.asarray(grab()) for _ in range(frames)])
    return np.median(stack, axis=0).astype(np.uint8)


def median_position(measurements: Sequence[CubeMeasurement]) -> Point:
    """The middle of several readings of a cube that is not moving, to take the jitter out of one frame."""
    if not measurements:
        raise ValueError("no measurements")
    xs = [m.position.x_mm for m in measurements]
    ys = [m.position.y_mm for m in measurements]
    return Point(float(np.median(xs)), float(np.median(ys)))
