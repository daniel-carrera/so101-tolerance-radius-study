"""Camera-only tools: check the view, calibrate it, log measurements, rehearse placements. No arm is touched.

These jobs are done with the front camera alone and no arm session running:

* `check`     : can the table frame be placed from this view? Saves an annotated picture with the grid drawn on it,
                and says whether the view still agrees with the stored calibration. Before the table is registered
                it shows which tags are in view and how near the picture's edge they are.
* `live`      : the same look, live: for positioning the camera. Shows the picture the arm sessions will see with
                every tag outlined, how large each floor tag is and how near the edge, and whether that is enough.
* `register-table` : measure the two floor tags against the printed registration sheet, and store the table
                geometry. Done once per table layout, with the sheet seated where the grid should be.
* `calibrate` : tie pixels to table millimetres from tag corners averaged over many frames, and store the result.
                Done once, when the camera mount is final; every later measurement goes through this one conversion.
* `log`       : show the cube's measured position live and write a reading to a CSV file at each press of space. This
                is the tool for the instrument check (cube moved along a physical scale), for repeatability, and for
                the reset-by-eye measurement.
* `place`     : rehearse the placement loop on given targets.
* `basket`    : the basket's place, live: the picture with the two edges where they belong and where they are, and
                how many millimetres to move it which way. The basket stands on tape and has no stop to go back to.
* `focus`     : the wrist camera's picture, live, with how sharp it is as a number: for setting a lens that is
                focused by hand.
* `wrist`     : the wrist camera watched for a minute, the arm at rest: which tone state its picture is in, and
                whether its USB link drops frames. For trying a plug, an adapter or a port before a sitting.
* `wristplace`: where the wrist camera stands in its mount, live: the gripper's fixed jaw is rigid with that camera,
                so its place in the picture is the camera's place. With the picture of the jaw as it stood in
                training laid over it: for tightening the mount, and after anything has knocked it.
* `byeye`     : measure how far off a reset lands when the cube is put back from a reference picture, with no aid.
                First every target is placed with the ring and its reference picture taken; then each is put back by
                eye beside its picture, in another order.

The camera is opened the way the arm sessions see it: 640 x 480, turned upright per the rig table. On this camera
640 x 480 is the centre crop of the wide picture, not a scaled copy, so a view that shows both tags at 1280 x 720 may
not show them here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius.config import REPO_ROOT, load_protocol
from tolerance_radius.geometry import Grid
from tolerance_radius.rig import lerobot101, outline, registration
from tolerance_radius.rig import vision as vision_module
from tolerance_radius.rig.placement_phase import KEY_ABORT, KEY_ACCEPT, PlacementPhase
from tolerance_radius.rig.vision import (
    CubeMeasurement,
    TableView,
    VisionError,
    empty_scene,
    floor_levels,
    grid_probes_px,
    load_geometry,
    median_position,
    WRIST_OTHER_ABOVE,
    tag_outlines_over,
    wrist_changed,
    wrist_contrast,
    wrist_reads,
)
from tolerance_radius.schema import Point
from tolerance_radius.seeds import derived_rng
from tolerance_radius.targets import uniform_placements

FRAME_SIZE = (640, 480)
#: The stored conversion from pixels to table millimetres that every measurement in the study goes through.
STORED_VIEW = REPO_ROOT / "data" / "instrument" / "table_view.json"
#: Where the instrument's own records go: view checks, registration pictures, logged readings.
INSTRUMENT_DIR = REPO_ROOT / "data" / "instrument"
#: Where `wrist` writes what each of its runs found, one line a run (on this computer only: `data/logs` is not
#: committed), so that a morning's tries of plugs and ports can be read afterwards.
WRIST_LOG = REPO_ROOT / "data" / "logs" / "wrist_checks.log"
#: Frames averaged for the table registration.
REGISTRATION_FRAMES = 40
#: A floor tag smaller than this in the picture is too small to place the table by: 5 px for each of its 8 squares
#: is the line below which the arm tooling calls decoding unreliable, and this is that with a margin. It was 48 at
#: first; the view chosen on 2026-10-05 has the far floor tag at 48.3 px, where a limit of 48 would come and go from
#: frame to frame, and the registration made in that view shows it is enough (solved from single frames, the grid
#: wandered by 0.08 mm). And no tag nearer the picture's edge than `EDGE_MIN_PX`.
FLOOR_TAG_MIN_PX = 46.0
EDGE_MIN_PX = 25.0
#: The grid's outline must stay this far inside the picture: a cube at the grid's edge reaches about 17 mm beyond the
#: outline as the camera sees it, and has to be whole in the picture with room to spare.
GRID_EDGE_MIN_PX = 40.0
GRID_CORNERS = {"A1": (0.0, 0.0), "E1": (250.0, 0.0), "E3": (250.0, 150.0), "A3": (0.0, 150.0)}
#: The live view places the grid from this many of the sheet's tags or more (a registration wants all six).
LIVE_MIN_SHEET_TAGS = 4
#: Between the registration and the calibration the two floor tags must still lie where they lay against each other.
#: What can be seen is where each tag's outline is in the picture now against where it was then. A camera pushed
#: sideways moves both alike. A camera that turned about its own line of sight turns the pair: a turn of 0.11 degrees
#: moved one tag a pixel against the other on 2026-10-06, while the sheet was peeled off, and the pipes, the arm and
#: the basket turned with them (`analysis/instrument/picture_motion.md`). A camera that came closer stretches the
#: pair. A floor tag moved on the floor (pressed flat, bumped, its tape gave) turns or stretches the pair as well,
#: and the two tags by themselves cannot say which of these it was. So:
#: * shifts that differ by more than `TAG_SHIFT_NOTE_PX` are said, with the pair's turn and stretch, and the
#:   calibration is stored: the frame is taken from the tags as they are seen now, which is right if it was the camera;
#: * beyond `RELATIVE_MOVE_LIMIT_MM`, the two tags' movement against each other read on the floor through the
#:   registered frame, it is refused until it is known which it was (`analysis/instrument/picture_motion.py`: the
#:   rest of the picture moves with the tags when it was the camera). If it was a tag, the frame is off by about
#:   that much somewhere on the grid and distances are stretched by about 0.17 % for each millimetre. In a trial on
#:   the first sitting's geometry a floor tag pressed flat moved the grid by 2.6 mm (an independent review,
#:   2026-10-05). The limit was 1.5 mm for two hours: this rig's camera, moved by the light box's walls, has since
#:   read 1.0 mm by itself (a turn of 0.1 degrees, a picture 0.07 % larger; the rest of the picture agreed with the
#:   tags to 0.3 px both times), so 1.5 would have stopped a session for nothing. Beyond
#:   `RELATIVE_MOVE_NOTE_MM` a session says so and the pictures are held against each other.
#: The first rule here (2026-10-06, morning) took the two shifts' difference in pixels, with three tenths of the
#: larger shift allowed for a camera that turns: a turn's effect grows with the distance between the tags, not with
#: the shift, and the rule refused the first calibration at the rig for a camera that had turned.
#: The tags are measured by their outlines: through the detector's own corners a picture shifted by a tenth of a
#: pixel put this solve up to 1.2 mm from the frame, through outlines 0.5 mm at most (synthetic pictures of the
#: rig's view, 2026-10-06).
TAG_SHIFT_NOTE_PX = 0.5
RELATIVE_MOVE_NOTE_MM = 1.5
RELATIVE_MOVE_LIMIT_MM = 3.0
#: A camera whose picture has moved by more than this since the registration is said to have moved.
CAMERA_SHIFT_NOTE_PX = 0.5
#: A floor tag found in a smaller share of the frames than this is said to be flickering.
FOUND_SHARE_NOTE = 0.9
#: Frames averaged for the stored calibration, and for the view a session solves when it starts.
CALIBRATION_FRAMES = 90
SESSION_CHECK_FRAMES = 20
#: A calibration's two halves must put the grid within a third of this of each other.
AGREEMENT_LIMIT_MM = 1.5
#: A session takes the table frame from the two floor tags as it finds them, and holds the stored calibration
#: against it (`checked_stored_view`). The camera of this rig does not keep its place to a millimetre: its pipe is
#: carried by the light box's walls, and within an hour of the calibration of 2026-10-06 its picture stood 0.04,
#: 0.28, 1.25 and 2.2 mm from where the calibration saw it, the pipes, the arm and the basket moving with the tags.
#: The tags are taped to the floor, and the frame is theirs. So the camera's movement is followed: said beyond
#: `VIEW_MOVED_NOTE_MM` (where a session used to be refused, when it measured through the stored conversion), and
#: refused beyond `VIEW_MOVED_LIMIT_MM`, where the picture is no longer the one the policy's demonstrations were
#: recorded in: about twice what the arm tooling's own baseline check allows (3 px of its 1280 x 720 picture, 2.4 mm
#: on this floor), so that check speaks first.
VIEW_MOVED_NOTE_MM = 1.5
VIEW_MOVED_LIMIT_MM = 5.0
# A session solves its conversion once, from the floor tags as its first frames show them, and goes on measuring
# through it, while the front camera goes on moving as it warms. On 2026-10-07 its picture moved by 1.3 and 1.5 mm
# in two recording sessions of 32 and 33 minutes that began with a camera that had been off, and by 0.4 mm in the
# two that began with it warm; the positions the first two measured live came to stand 1.2 and 1.7 mm from where
# each episode's own first frame has the cube (the last third of each session; r = 0.83 and 0.89 with the camera's
# movement), against 0.4 and 0.6 mm in the other two. For a demonstration that is no matter: its start is measured
# again from its own frame (the audit). A trial's start is what its session measured. So before an attempt a trial
# session asks how far the picture has crept since its conversion was solved (`crept_since`), and beyond the
# protocol's limit (`view_crept_limit_mm` in its `[execution]` section, 1.0 mm) makes no attempt: it is started
# again, which solves the conversion afresh. The limit is the protocol's and no constant of this module.

#: Above this, the tags' stored corner positions no longer fit one flat table: a tag has moved by centimetres. A tag
#: moved by millimetres does not show here (the fit absorbs it, and distances are stretched by about 0.17 % per
#: millimetre of movement): only the check against a ruler validates distances.
RESIDUAL_LIMIT_MM = 1.5
#: Readings averaged for one logged measurement, and the scatter above which the cube is taken to be moving.
LOG_FRAMES = 8
STILL_MM = 0.7


#: The picture a session opens on must be the picture the calibration saw. Two things are held against the
#: calibration's, and a session is refused on either:
#: * the floor's grey level, which may differ by this much. With every camera setting unchanged it repeated to 0.3
#:   grey levels over nine openings (2026-10-06); a hand-set exposure lost (the vendor's window closed) moved it from
#:   207 to 148.
#: * the picture's contrast, white floor over the floor tags' black, as a ratio to the calibration's. On some
#:   openings (3 of 13 that day) the same camera picture is handed over stretched from video range to full range:
#:   whites up by 7 %, blacks halved, the contrast doubled, for the whole session. A lamp or an exposure that changed
#:   moves white and black the same way and leaves the contrast where it was.
#: What a refusal costs is starting the command again; what it saves is a sitting or a block of trials recorded in a
#: picture the policy was not trained on. The colour balance only warns.
#: A floor more than `PICTURE_GREY_NOTE` off is said and let through. One of three openings a minute after the
#: calibration read 12 grey levels darker all through (the floor 190 for 202, the tags' black 25 for 32), the camera
#: and the scene unmoved; the two beside it read within 0.3 of the calibration (2026-10-06, 13:17). Whether such an
#: opening stays that way for a session is not known. Not refused: a hand over the box or the daylight could do as
#: much, and a guard that stops a sitting for that costs more than it saves.
PICTURE_GREY_LIMIT = 25.0
PICTURE_GREY_NOTE = 6.0
PICTURE_CONTRAST_RANGE = (0.7, 1.4)
FLOOR_COLOUR_NOTE = 0.06


class Camera:
    """The front camera, opened as the arm sessions see it. Use as a context manager."""

    def __init__(self, index: int | None = None, warmup_s: float = 3.0, grab_timeout_s: float = 5.0) -> None:
        self._tag_check = lerobot101.load("tag_check")
        self._index = lerobot101.load("rig").FRONT_CAMERA_INDEX if index is None else index
        self._warmup_s, self._grab_timeout_s = warmup_s, grab_timeout_s
        self._capture = None

    def __enter__(self) -> Camera:
        self._capture = self._tag_check.open_cam(self._index, *FRAME_SIZE)
        if self._capture is None:
            raise VisionError(f"could not open camera {self._index}")
        self._tag_check.warm_up(self._capture, self._warmup_s)
        return self

    def __exit__(self, *_exc) -> None:
        import cv2

        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()

    def grab(self) -> np.ndarray:
        """The current frame, BGR, upright."""
        frame, _ = self._tag_check.read_frame(self._capture, self._grab_timeout_s)
        if frame is None:
            raise VisionError("the camera stopped delivering frames")
        if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
            raise VisionError(
                f"the camera gave {frame.shape[1]} x {frame.shape[0]}, expected {FRAME_SIZE[0]} x {FRAME_SIZE[1]}: "
                "another program has changed the camera's picture while this one was using it. On 2026-10-06 it was "
                "the vendor's window (Logi Tune) coming to the front: it then starts a preview of its own. Minimise "
                "that window (do not close it, do not quit it), leave it alone while a command runs, and start "
                "this command again. What was logged up to here is kept."
            )
        return frame


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fresh_view(grab, frames: int, geometry: dict | None = None) -> TableView:
    """A view solved now, from tag corners averaged over `frames` frames."""
    return TableView.from_frames((grab() for _ in range(frames)), geometry)


def since_calibration(
    stored: TableView, now: TableView, geometry: dict, frames: list[np.ndarray],
    relative_limit_mm: float | None = None,
) -> tuple[list[str], str | None]:
    """What has moved since the calibration was stored: lines to print, and what stops a session, if anything.

    `now` is the view solved from `frames`. The second result is None when a session may go on; otherwise a sentence.

    `relative_limit_mm` replaces the limit on how far the two floor tags may read against each other since the
    registration (`RELATIVE_MOVE_LIMIT_MM`). It is for the operator who has held the kept frame against the
    calibration's and seen that the tags moved with the rest of the picture: the figure is then the camera's turn
    since the registration (1.05 mm of it were there at the calibration of 2026-10-06, 1.7 mm that evening), it is
    measured from the registration whatever is calibrated afterwards, and without this a camera that has turned by
    a third of a degree in all would stop every session with nothing to be done but to register the table again.
    """
    limit = RELATIVE_MOVE_LIMIT_MM if relative_limit_mm is None else float(relative_limit_mm)
    apart = stored.disagreement_mm(now, grid_probes_px(stored, geometry))
    lines = [f"the table frame is taken from the two floor tags as they are now ({now.frames_used} frames; they fit "
             f"the stored geometry to {now.residual_mm:.2f} mm)",
             f"the camera's picture stands {apart:.2f} mm from where the calibration saw it (said beyond "
             f"{VIEW_MOVED_NOTE_MM}, refused beyond {VIEW_MOVED_LIMIT_MM})"]
    if not now.residual_mm <= RESIDUAL_LIMIT_MM:
        return lines, ("the two floor tags do not fit the stored table geometry "
                       f"({now.residual_mm:.1f} mm, limit {RESIDUAL_LIMIT_MM}): a tag has moved by centimetres, or "
                       "something lies across one. Nothing can be measured until that is put right")
    if apart > VIEW_MOVED_LIMIT_MM:
        return lines, (
            f"the camera's picture is no longer the calibration's: it has moved {apart:.1f} mm on the floor (limit "
            f"{VIEW_MOVED_LIMIT_MM} mm). The policy would be looking through another view than the one its "
            "demonstrations were recorded in. Run the camera baseline check, and say so before anything is stored "
            "again"
        )
    if apart > VIEW_MOVED_NOTE_MM:
        scale = now.px_per_mm_at(Point(*probes_mm_centre(geometry)))
        lines.append(f"NOTE: the camera has moved since the calibration (about {apart * scale:.1f} px at the grid). "
                     "The frame follows the floor tags, so positions are measured as before; what the policy sees "
                     "has moved by that much. Write it down, and say so if it grows.")
    seen = floor_tags_then_and_now(geometry, frames)
    registered = (geometry.get("registration") or {}).get("px_to_mm")
    if seen is not None and registered is not None:
        motion = pair_motion(*seen, registered)
        either = ("A camera that turned or came closer reads like this, and so does a floor tag that was moved; the "
                  "two tags cannot say which. The frame this command kept is held against the calibration's to see "
                  "(analysis/instrument/picture_motion.py)")
        if motion["relative_mm"] > limit:
            return lines, (
                f"the two floor tags read {motion['relative_mm']:.1f} mm against each other since the table was "
                f"registered (limit {limit:g} mm; as a pair they have turned by "
                f"{motion['turn_deg']:+.2f} degrees and their distance has changed by {100 * motion['stretch']:+.2f} "
                f"%). {either}. Stop and say so: if a tag has moved, the frame has moved with it. If the pictures "
                "show that the tags moved with everything else, it is the camera, and --tags-relative-limit sets "
                "another limit for one run"
            )
        if motion["relative_mm"] > RELATIVE_MOVE_NOTE_MM:
            lines.append(f"NOTE: the two floor tags read {motion['relative_mm']:.1f} mm against each other since "
                         f"the table was registered (said beyond {RELATIVE_MOVE_NOTE_MM}, refused beyond "
                         f"{limit:g}). {either}. Say so.")
    return lines, None


def checked_stored_view(grab, *, path: Path | None = None, frames: int = SESSION_CHECK_FRAMES,
                        relative_limit_mm: float | None = None) -> TableView:
    """The view a session measures through: solved now from the two floor tags, and held against the stored
    calibration.

    The table frame is the floor tags'. The camera is only what looks at them, and on this rig it moves by a
    millimetre or two in an hour (`VIEW_MOVED_NOTE_MM`). So every session solves the conversion afresh from the tags'
    outlines, and the stored calibration is the reference it is held against: it must belong to the registration in
    place, the camera's picture must not have moved too far from it, the tags must not have moved against each
    other, and the picture's levels must be the calibration's. The frame the session started on is kept beside the
    calibration, unmarked (`session_<time>.png`).

    Raises `VisionError` when there is no stored calibration, when the tags cannot be seen, or when one of those is
    not so.
    """
    import cv2

    where = STORED_VIEW if path is None else path
    stored = TableView.load(where)
    geometry = load_geometry()
    other = other_registration(stored, geometry)
    if other:
        raise VisionError(other)
    seen = [grab() for _ in range(frames)]
    now = TableView.from_frames(seen, geometry)
    kept = where.parent / f"session_{stamp()}.png"
    cv2.imwrite(str(kept), seen[-1])
    lines, wrong = since_calibration(stored, now, geometry, seen, relative_limit_mm)
    for line in lines:
        print(f"  {line}")
    print(f"  picture: {kept}")
    if wrong:
        raise VisionError(wrong)
    lines, wrong = picture_state(stored, grab, geometry)
    for line in lines:
        print(f"  {line}")
    if wrong:
        raise VisionError(wrong)
    now.floor_levels = stored.floor_levels
    return now


def other_registration(stored: TableView, geometry: dict) -> str | None:
    """What is wrong when a stored view was solved against another table geometry than the one in place, or None."""
    if stored.geometry_created and geometry.get("created") and stored.geometry_created != geometry["created"]:
        return (f"the stored calibration belongs to an earlier registration of the table (made {stored.geometry_created}"
                f"; the table geometry in place was made {geometry['created']}). Run `scripts/rig/measure.py "
                "calibrate --replace`: until then every position would be measured through the old frame.")
    return None


def tag_quads_px(view: TableView, geometry: dict) -> list[list[tuple[float, float]]]:
    """Where the floor tags' corners are in the picture, by the stored geometry through a view."""
    return [[view.to_px(Point(float(x), float(y))) for x, y in tag["corners"]] for tag in geometry["tags"].values()]


def picture_levels(view: TableView, frame: np.ndarray, geometry: dict) -> dict[str, float]:
    """The floor's grey level and colour inside the grid and the floor tags' black, in one frame."""
    return floor_levels(frame, grid_probes_px(view, geometry)[:4], tag_quads_px(view, geometry))


def levels_off(then: dict[str, float], now: dict[str, float]) -> tuple[str, float] | None:
    """Which of the picture's levels is no longer the calibration's: ("contrast", its ratio to the calibration's),
    ("grey", the floor's difference in grey levels), or None when a session may go on (`PICTURE_CONTRAST_RANGE`,
    `PICTURE_GREY_LIMIT`). The contrast is asked first: it is what tells a stretched tone range from a lamp."""
    if "tag_black" in now and "tag_black" in then:
        # a black of nought is taken as one grey level: the ratio must not hang on a division by nothing
        contrast = (now["grey"] / max(now["tag_black"], 1.0)) / (then["grey"] / max(then["tag_black"], 1.0))
        low, high = PICTURE_CONTRAST_RANGE
        if not low <= contrast <= high:
            return "contrast", contrast
    if abs(now["grey"] - then["grey"]) > PICTURE_GREY_LIMIT:
        return "grey", now["grey"] - then["grey"]
    return None


def picture_state(stored: TableView, grab, geometry: dict) -> tuple[list[str], str | None]:
    """The picture now against the picture the calibration saw: lines to print, and what is wrong, if anything.

    The second result is None when a session may go on. It is a sentence when the floor's level or the picture's
    contrast is not the calibration's: the session then stops, and starting it again opens the camera afresh. A
    failure to read the picture is said and passed over: this check must not stop a session by its own fault.
    """
    then = stored.floor_levels
    if not then:
        return [], None
    try:
        now = picture_levels(stored, grab(), geometry)
    except Exception as error:  # noqa: BLE001 - see the docstring
        return [f"(the picture's levels could not be read: {error})"], None
    lines = [f"the floor inside the grid: grey level {now['grey']:.0f} (at the calibration {then['grey']:.0f}), "
             f"red over blue {now['red_over_blue']:.2f} ({then['red_over_blue']:.2f})"]
    if abs(now["red_over_blue"] / then["red_over_blue"] - 1.0) > FLOOR_COLOUR_NOTE:
        lines.append("WARNING: the picture's colour balance is not what it was at the calibration. Has the camera's "
                     "white balance gone back to automatic? The cube is found by its colour.")
    again = ("Start this command again: the camera is opened afresh. If it says the same three times running, look "
             "at the camera's settings in the vendor's software (its window must be open for a hand-set exposure to "
             "hold), and only if the picture was changed on purpose store the calibration again")
    if "tag_black" in now and "tag_black" in then:
        contrast = (now["grey"] / max(now["tag_black"], 1.0)) / (then["grey"] / max(then["tag_black"], 1.0))
        lines.append(f"the floor tags' black: grey level {now['tag_black']:.0f} (at the calibration "
                     f"{then['tag_black']:.0f}); the picture's contrast is {contrast:.2f} of the calibration's")
    off = levels_off(then, now)
    low, high = PICTURE_CONTRAST_RANGE
    if off and off[0] == "contrast":
        return lines, (
            f"the camera has opened in another picture state: white {now['grey']:.0f} and black "
            f"{now['tag_black']:.0f} against {then['grey']:.0f} and {then['tag_black']:.0f} at the calibration, a "
            f"contrast {off[1]:.2f} of what it was (allowed {low} to {high}). The policy would see a picture "
            f"it was not trained on. {again}"
        )
    if off:
        return lines, (
            f"the picture is {'brighter' if now['grey'] > then['grey'] else 'darker'} than at the calibration: the "
            f"floor reads grey level {now['grey']:.0f} against {then['grey']:.0f} (allowed {PICTURE_GREY_LIMIT:.0f} "
            f"either way). Has the camera's exposure gone back to automatic, or a lamp changed? {again}"
        )
    if abs(now["grey"] - then["grey"]) > PICTURE_GREY_NOTE:
        lines.append(f"NOTE: the floor reads {abs(now['grey'] - then['grey']):.0f} grey levels "
                     f"{'brighter' if now['grey'] > then['grey'] else 'darker'} than at the calibration (said beyond "
                     f"{PICTURE_GREY_NOTE:.0f}, refused beyond {PICTURE_GREY_LIMIT:.0f}). The camera has opened on a "
                     "picture like this with nothing changed. If nothing is shading the floor, starting this command "
                     "again usually gives the usual picture; write down which it was.")
    return lines, None


#: The look at the picture after a take (`changed_since`): frames to a look, and how many looks half a second apart
#: are made before the operator is told what is in the way. From then it goes on looking for `AFTER_TAKE_PATIENCE_S`
#: and takes the first clean look. A hand at the basket hid a floor tag for two seconds and a good take was thrown
#: away and the sitting stopped; that is what the patience is for (the review and the dry run of 2026-10-06, on
#: recorded frames; it did not happen in the 27 takes of that evening). A picture that has really changed stays
#: changed, so waiting loses nothing but the seconds.
AFTER_TAKE_FRAMES = 5
AFTER_TAKE_TRIES = 4
AFTER_TAKE_PATIENCE_S = 15.0


def crept_since(start: TableView, grab, *, frames: int = AFTER_TAKE_FRAMES) -> float | None:
    """How far the front camera's picture now stands from the one `start` was solved in: millimetres on the floor,
    the largest over the grid's probe points, which is how far off a position measured through `start` now is.

    None when it cannot be said (a floor tag not in sight, no frame from the camera, a solve that gives no number):
    `changed_since` is the look that speaks of those, and it is made first.

    What it reads (the review of the night of 2026-10-07, *measured* on drawn tags and read from the first pilot's
    log): a shift of 1 mm reads 0.998; the same scene a minute apart read 0.48 and 0.47 mm; `grab` gives the camera's
    latest frame without waiting, so the frames of one look are in fact one. The reading follows what stands in the
    picture by a few tenths: a shadow's edge along a tag's side reads 0.2 to 0.9 mm with nothing else amiss, and the
    five placements beside floor tag 0 read 0.13 to 0.36 mm above their neighbours in time. Something solid over a
    tag gives None. So a reading over a limit is asked again before it is acted on (`trial_session`).
    """
    geometry = load_geometry()
    try:
        now = TableView.from_frames([grab() for _ in range(frames)], geometry)
    except Exception:  # noqa: BLE001 -- a look that cannot say; what is wrong with the picture is `changed_since`'s
        return None
    apart = float(now.disagreement_mm(start, grid_probes_px(start, geometry)))
    return apart if math.isfinite(apart) else None


def changed_since(start: TableView, grab, *, frames: int = AFTER_TAKE_FRAMES, tries: int = AFTER_TAKE_TRIES,
                  wait_s: float = 0.5, limit_mm: float = VIEW_MOVED_LIMIT_MM,
                  patience_s: float = AFTER_TAKE_PATIENCE_S, told=None, clock=time.monotonic) -> str | None:
    """Is the picture still the one the session started on? None when it is; otherwise what is wrong, in a clause.

    Asked after every take, and before every attempt of a trial. On 2026-10-06 the front camera went back to its own
    zoom, exposure and focus 3.4 seconds into a take: everything 14 % smaller in the picture, the floor at grey
    level 150 for 200. The session had checked the picture at its start and the take was saved; it showed at the
    next placement, where the grid drawn from the session's first frame no longer lay on the table.

    Two things are asked: where the floor tags lie against the session's start (the camera's zoom, its mode, the
    camera itself), by the limit a session's start has for the calibration; and the picture's levels against the
    calibration's, which `start` carries (`levels_off`).

    The first look that finds both right ends it. A look that cannot see the tags, that gets no frame from the
    camera, or that finds something wrong is made again: `tries` times half a second apart, then `told` is given a
    sentence for the operator (a hand or the arm may be in the way), and then for `patience_s` more. Only what is
    still wrong then is said. The arm is not always at rest when this is asked (a stop key ends a reset wherever
    the arm is), and `grab` gives the camera's latest frame, so the frames of one look may be one frame.
    """
    geometry = load_geometry()
    probes = grid_probes_px(start, geometry)
    then = start.floor_levels
    wrong = short = ""
    looks = 0
    asked_at: float | None = None
    while True:
        if looks:
            time.sleep(wait_s)
        looks += 1
        try:
            seen = [grab() for _ in range(frames)]
            now = TableView.from_frames(seen, geometry)
        except VisionError as error:
            short = "The floor tags cannot be seen"
            wrong = (f"the floor tags could not be seen ({error}): whether the front camera's picture is still the "
                     "session's cannot be said")
        except Exception as error:  # noqa: BLE001 -- a frame the camera did not give in time must not end the session
            short = "The front camera gives no frame"
            wrong = (f"the front camera gave no frame ({error}): whether its picture is still the session's cannot "
                     "be said")
        else:
            apart = now.disagreement_mm(start, probes)
            off = None
            if apart <= limit_mm and then:
                try:
                    levels = picture_levels(start, seen[-1], geometry)
                    off = levels_off(then, levels)
                except Exception:  # noqa: BLE001 -- as in `picture_state`: not a stop by this check's own fault
                    off = None
            short = "The front camera's picture does not read as it did"
            if apart > limit_mm:
                wrong = (f"the front camera's picture stands {apart:.0f} mm from the one this session started on "
                         f"(allowed {limit_mm:g}): the camera's zoom or its mode has changed, or the camera has "
                         "been moved")
            elif off and off[0] == "contrast":
                low, high = PICTURE_CONTRAST_RANGE
                wrong = (f"the front camera's picture is in another tone state: white {levels['grey']:.0f} and black "
                         f"{levels['tag_black']:.0f} against {then['grey']:.0f} and {then['tag_black']:.0f} at the "
                         f"calibration, a contrast {off[1]:.2f} of what it was (allowed {low} to {high})")
            elif off:
                wrong = (f"the front camera's picture is {'brighter' if off[1] > 0 else 'darker'} than the "
                         f"calibration's: the floor reads grey level {levels['grey']:.0f} against "
                         f"{then['grey']:.0f} (allowed {PICTURE_GREY_LIMIT:.0f} either way)")
            else:
                return None
        if looks < max(1, tries):
            continue
        if asked_at is None:
            asked_at = clock()
            if told is not None and patience_s > 0:
                told(f"{short}. Hands and the arm out of the picture: looking again for {patience_s:g} seconds.")
        if clock() - asked_at >= patience_s:
            return wrong


def wrist_reading(grab_wrist, *, looks: int = 3, wait_s: float = 0.2) -> float | None:
    """The wrist camera's rest picture by its one number (`vision.wrist_contrast`), the middle of up to `looks` looks.

    `grab_wrist` gives that camera's latest frame, BGR. None when no look could be judged: the picture is not the
    rest picture, or the camera gave no frame. Never raises.
    """
    readings = []
    for look in range(max(1, looks)):
        if look:
            time.sleep(wait_s)
        try:
            value = wrist_contrast(grab_wrist())
        except Exception:  # noqa: BLE001 -- a frame the camera did not give in time is a look that says nothing
            value = None
        if value is not None:
            readings.append(value)
    return float(np.median(readings)) if readings else None


def wrist_changed_since(start: float | None, grab_wrist, *, wait_s: float = 1.0) -> tuple[float | None, str | None]:
    """The wrist camera's reading now, and the sentence when its picture has changed its tone state since the
    session's first reading `start` (`vision.wrist_changed`); the sentence is None when it has not, and when either
    reading could not be judged. A change is said only when a second reading, `wait_s` later, says it too.
    """
    now = wrist_reading(grab_wrist)
    if wrist_changed(start, now) is None:
        return now, None
    time.sleep(wait_s)
    again = wrist_reading(grab_wrist)
    return again, wrist_changed(start, again)


def table_view(camera: Camera, *, ask=input) -> TableView:
    """The stored view, confirmed against the live picture, with the empty-scene picture the detector needs."""
    view = checked_stored_view(camera.grab)
    ask("\n  Take the cube off the table and keep hands out of the picture, then press Enter. ")
    return view.with_background(empty_scene(camera.grab))


# ---------------------------------------------------------------------------------------------------------------- check


def describe_tags(found: dict, frame_size: tuple[int, int] = FRAME_SIZE) -> list[str]:
    """One line per tag in view: how large it is in the picture and how near the picture's edge."""
    lines = []
    for tag_id, quad in sorted(found.items()):
        quad = np.asarray(quad, dtype=float)
        side = float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))
        edge = float(min(quad[:, 0].min(), frame_size[0] - quad[:, 0].max(), quad[:, 1].min(),
                         frame_size[1] - quad[:, 1].max()))
        kind = "floor tag" if tag_id in registration.FLOOR_TAG_IDS else (
            "sheet tag" if tag_id in registration.SHEET_TAGS else "NOT one of ours")
        warn = "  <-- too near the edge: move it in" if edge < 25 else ""
        lines.append(f"tag {tag_id} ({kind}): {side:.0f} px across, {edge:.0f} px from the nearest edge{warn}")
    return lines


def exposure_lines(frame_bgr: np.ndarray, found: dict, calibrated: bool = False) -> list[str]:
    """What the white around the tags says of the camera's exposure: the line to set it by before registering.

    Once a calibration is stored (`calibrated`) the exposure is no longer anyone's to set: the picture's levels are
    held against the calibration's, and this only says the level.
    """
    import cv2

    if not found:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    whites = [outline.white_level(gray, quad) for quad in found.values()]
    if calibrated:
        return [f"the white around the tags reads grey level {min(whites):.0f} to {max(whites):.0f}"]
    line = (f"the white around the tags reads grey level {min(whites):.0f} to {max(whites):.0f}: for a registration "
            f"it should be 225 to 245, and is clipped from {outline.CLIPPED_GREY:.0f}")
    if max(whites) >= outline.CLIPPED_GREY:
        return [line, "OVER-EXPOSED: lower the camera's exposure before registering the table"]
    if max(whites) > 245.0:
        return [line, "bright: the sheet's tags read a few levels higher than the floor tags, and a registration is "
                      f"refused from {outline.CLIPPED_GREY:.0f}. Aim for about 235 to 240 on this line"]
    if max(whites) < 200.0:
        return [line, "DARK: the white floor reads grey. A camera on automatic exposure dims a scene that is mostly "
                      "white (it read 140 with the arm and the basket out of the picture, 2026-10-06). Set the "
                      "exposure by hand in the vendor's software until this line reads about 235 to 240"]
    return [line]


def mode_check(args: argparse.Namespace) -> int:
    import cv2

    grid_overlay, tag_check = lerobot101.load("grid_overlay"), lerobot101.load("tag_check")
    with Camera(args.camera) as camera:
        frames = [camera.grab() for _ in range(SESSION_CHECK_FRAMES)]
    frame = frames[-1]
    found = tag_check.detect(frame)
    picture = tag_check.annotate(frame, found)  # returns an annotated copy
    when = stamp()
    out = Path(args.out) if args.out else INSTRUMENT_DIR / f"view_check_{when}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out:
        # the frame as it came, beside the marked one: two pictures can only be held against each other unmarked
        cv2.imwrite(str(INSTRUMENT_DIR / f"view_raw_{when}.png"), frame)
    print(f"\n  frame {frame.shape[1]} x {frame.shape[0]}; tags seen: {sorted(found) or 'none'}")
    for line in describe_tags(found):
        print(f"  {line}")
    for line in exposure_lines(frame, found, calibrated=STORED_VIEW.is_file()):
        print(f"  {line}")
    if not vision_module.TABLE_GEOMETRY.is_file():
        cv2.imwrite(str(out), picture)
        missing = [tag_id for tag_id in registration.FLOOR_TAG_IDS if tag_id not in found]
        print(f"  the table is not registered yet, so no grid can be drawn. Floor tags "
              f"{list(registration.FLOOR_TAG_IDS)} in view: {'both' if not missing else f'NOT {missing}'}")
        print(f"  picture: {out}\n  NOT READY: register the table (docs/runbook/00-table-layout.md)")
        return 1
    geometry = load_geometry()
    known = sorted(int(k) for k in geometry["tags"])
    print(f"  tags needed: {known}")
    try:
        view = TableView.from_frames(frames, geometry)
    except VisionError as error:
        cv2.imwrite(str(out), picture)
        print(f"  NOT READY: {error}\n  picture: {out}")
        return 1
    outlines_px, found_in, _measured_in, _count = tag_outlines_over(frames, known)
    to_px = grid_overlay.mm_to_px(outlines_px, geometry)
    picture = grid_overlay.draw_grid(picture, to_px, geometry, allow_holdout=True)  # also returns a copy
    corners_px = grid_probes_px(view, geometry)[:4]
    margin_px = min(min(x, FRAME_SIZE[0] - x, y, FRAME_SIZE[1] - y) for x, y in corners_px)
    cols, rows, cell = geometry["grid"]["cols"], geometry["grid"]["rows"], geometry["grid"]["cell_mm"]
    scale = view.px_per_mm_at(Point(cols * cell / 2, rows * cell / 2))
    cv2.imwrite(str(out), picture)
    print(f"  table frame placed from {view.frames_used} frames. Scale at the grid's centre: {scale:.2f} px per mm "
          f"({1 / scale:.2f} mm per px)")
    print(f"  fit of the two tags to the stored table geometry: {view.residual_mm:.2f} mm (limit {RESIDUAL_LIMIT_MM}; "
          f"this catches only a tag moved by centimetres: the ruler check is what validates distances)")
    print(f"  grid corners in the picture: {[(round(x), round(y)) for x, y in corners_px]}")
    print(f"  the grid corner nearest the picture's edge is {margin_px:.0f} px from it")
    print(f"  picture: {out}")
    problems = []
    for tag_id in known:
        if found_in[tag_id] < FOUND_SHARE_NOTE * len(frames):
            problems.append(f"tag {tag_id} was found in only {found_in[tag_id]} of {len(frames)} frames: something "
                            "is moving across it, it is in glare, or it is too small in the picture")
    if margin_px < GRID_EDGE_MIN_PX:
        problems.append(f"part of the grid is outside the picture or within {GRID_EDGE_MIN_PX:.0f} px of its edge")
    if not view.residual_mm <= RESIDUAL_LIMIT_MM:
        problems.append("the two tags do not fit the stored table geometry: a tag has probably moved since it was "
                        "measured, and the geometry must be measured again before anything is recorded")
    if STORED_VIEW.is_file():
        other = other_registration(TableView.load(STORED_VIEW), geometry)
        if other:
            problems.append(other)
        moved_lines, wrong = since_calibration(TableView.load(STORED_VIEW), view, geometry, frames,
                                               getattr(args, "tags_relative_limit", None))
        for line in moved_lines:
            print(f"  {line}")
        if wrong:
            problems.append(wrong)
        state_lines, wrong = picture_state(TableView.load(STORED_VIEW), lambda: frame, geometry)
        for line in state_lines:
            print(f"  {line}")
        if wrong:
            problems.append(wrong)
    else:
        print("  no stored calibration yet: run `scripts/rig/measure.py calibrate` once the mount is final")
    for problem in problems:
        print(f"  NOT READY: {problem}")
    if not problems:
        print("  READY: both tags and the whole grid are in the 640 x 480 view")
    return 1 if problems else 0


# ----------------------------------------------------------------------------------------------------------------- live


def sheet_grid_px(found: dict) -> dict[str, tuple[float, float]] | None:
    """Where the grid's four corners are in the picture, from the registration sheet's tags; None without the sheet.

    For looking only: the sheet is taken as designed, which is good to a pixel.
    """
    import cv2

    ids = sorted(set(found) & set(registration.SHEET_TAGS))
    if len(ids) < LIVE_MIN_SHEET_TAGS:
        return None
    table = np.concatenate([registration.sheet_corners_mm(tag_id) for tag_id in ids])
    pixels = np.concatenate([np.asarray(found[tag_id], dtype=float).reshape(4, 2) for tag_id in ids])
    mm_to_px, _ = cv2.findHomography(table, pixels, 0)
    if mm_to_px is None:
        return None
    corners = {}
    for name, (x_mm, y_mm) in GRID_CORNERS.items():
        x, y, w = mm_to_px @ np.array([x_mm, y_mm, 1.0])
        corners[name] = (float(x / w), float(y / w))
    return corners


def view_problems(found: dict, frame_size: tuple[int, int] = FRAME_SIZE) -> list[str]:
    """What is not yet right about a camera position, from the tags in view. Empty when the view will do."""
    problems = []
    grid = sheet_grid_px(found)
    if grid is not None:
        for name, (x, y) in grid.items():
            edge = min(x, frame_size[0] - x, y, frame_size[1] - y)
            if edge < GRID_EDGE_MIN_PX:
                problems.append(f"the grid's corner {name} is {edge:.0f} px from the edge (at least "
                                f"{GRID_EDGE_MIN_PX:.0f}): a cube there must be whole in the picture")
    for tag_id in registration.FLOOR_TAG_IDS:
        if tag_id not in found:
            problems.append(f"floor tag {tag_id} is not in the picture")
            continue
        quad = np.asarray(found[tag_id], dtype=float)
        side = float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))
        edge = float(min(quad[:, 0].min(), frame_size[0] - quad[:, 0].max(), quad[:, 1].min(),
                         frame_size[1] - quad[:, 1].max()))
        if side < FLOOR_TAG_MIN_PX:
            problems.append(f"floor tag {tag_id} is {side:.0f} px across: bring the camera closer "
                            f"(at least {FLOOR_TAG_MIN_PX:.0f})")
        if edge < EDGE_MIN_PX:
            problems.append(f"floor tag {tag_id} is {edge:.0f} px from the edge (at least {EDGE_MIN_PX:.0f})")
    return problems


def registered_grid(picture: np.ndarray, frame: np.ndarray, found: dict, geometry: dict | None) -> np.ndarray:
    """The picture with the registered grid and its cells' names drawn where the two floor tags put them in this
    frame; the picture as it is when the table is not registered, the sheet is still on the floor, or a floor tag is
    not in view. For looking: one frame's tags place the grid to a few tenths of a millimetre."""
    import cv2

    if not geometry or set(found) & set(registration.SHEET_TAGS):
        return picture
    floor = {tag_id: found[tag_id] for tag_id in registration.FLOOR_TAG_IDS if tag_id in found}
    if len(floor) < len(registration.FLOOR_TAG_IDS):
        return picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outlines = outline.tag_outlines(gray, floor)
    if len(outlines) < len(floor):
        return picture
    grid_overlay = lerobot101.load("grid_overlay")
    to_px = grid_overlay.mm_to_px(outlines, geometry)
    return picture if to_px is None else grid_overlay.draw_grid(picture, to_px, geometry, allow_holdout=True)


def live_picture(frame: np.ndarray, found: dict, tag_check, geometry: dict | None = None) -> np.ndarray:
    """The live view: the frame with every tag outlined, the border no tag should cross, and what is still wrong.
    With the table's `geometry` and the sheet gone, the registered grid is drawn too."""
    import cv2

    picture = registered_grid(tag_check.annotate(frame, found), frame, found, geometry)
    height, width = picture.shape[:2]
    margin = int(EDGE_MIN_PX)
    cv2.rectangle(picture, (margin, margin), (width - margin, height - margin), (200, 200, 200), 1)
    grid = sheet_grid_px(found)
    if grid is not None:  # the grid's outline, as the sheet's tags place it
        outline = np.array([grid[name] for name in ("A1", "E1", "E3", "A3")]).round().astype(np.int32)
        cv2.polylines(picture, [outline], True, (255, 200, 0), 1)
    problems = view_problems(found, (width, height))
    sheet = sorted(set(found) & set(registration.SHEET_TAGS))
    lines = problems or ["the view will do: the floor tags are large enough, and tags and grid are clear of the edge"]
    lines.append(f"sheet tags in view: {len(sheet)} of {len(registration.SHEET_TAGS)}    [s] save a picture  [x] quit")
    for row, line in enumerate(lines):
        colour = (80, 255, 80) if not problems or row == len(lines) - 1 else (0, 190, 255)
        cv2.putText(picture, line, (10, height - 12 - 18 * (len(lines) - 1 - row)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3)
        cv2.putText(picture, line, (10, height - 12 - 18 * (len(lines) - 1 - row)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1)
    return picture


def mode_live(args: argparse.Namespace) -> int:
    """Show what the arm sessions will see, live, while the camera is being positioned. Saves the last picture."""
    import cv2

    tag_check = lerobot101.load("tag_check")
    window = "camera view  [s] save a picture  [x] quit"
    INSTRUMENT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  click the video window so it gets the keys. Move the camera until the text at the bottom turns green.\n"
          "  The grey border is as near the edge as a tag may come.")
    frame, found = None, {}
    geometry = load_geometry() if vision_module.TABLE_GEOMETRY.is_file() else None
    if geometry:
        print("  The table is registered: with the sheet off the floor the grid is drawn where the floor tags put it.")
    with Camera(args.camera) as camera:
        while True:
            frame = camera.grab()
            found = tag_check.detect(frame)
            cv2.imshow(window, live_picture(frame, found, tag_check, geometry))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                when = stamp()
                saved = INSTRUMENT_DIR / f"view_live_{when}.png"
                cv2.imwrite(str(saved), live_picture(frame, found, tag_check, geometry))
                cv2.imwrite(str(INSTRUMENT_DIR / f"view_raw_{when}.png"), frame)  # the frame itself, unmarked
                print(f"  saved {saved} (and the unmarked frame beside it)")
            if key in (KEY_ABORT, ord("q"), 27):
                break
    when = stamp()
    out = INSTRUMENT_DIR / f"view_check_{when}.png"
    cv2.imwrite(str(out), tag_check.annotate(frame, found))
    cv2.imwrite(str(INSTRUMENT_DIR / f"view_raw_{when}.png"), frame)
    print(f"\n  frame {frame.shape[1]} x {frame.shape[0]}; tags seen: {sorted(found) or 'none'}")
    for line in describe_tags(found):
        print(f"  {line}")
    problems = view_problems(found)
    for problem in problems:
        print(f"  NOT YET: {problem}")
    if not problems:
        print("  the view will do: the floor tags are large enough, and tags and grid are clear of the edge")
    print(f"  picture: {out}")
    return 1 if problems else 0


# --------------------------------------------------------------------------------------------------------------- basket

#: The frame that says where the basket belongs: the one the first session of 2026-10-08 started on. The basket
#: read this place to within 0.2 px in every kept frame of 2026-10-07 and of that morning.
BASKET_REFERENCE = INSTRUMENT_DIR / "basket_reference.png"


def mode_basket(args: argparse.Namespace) -> int:
    """Show where the basket stands against its reference frame, live, while it is put back. No arm, no session."""
    import cv2

    from tolerance_radius.rig import basket_place

    tag_check = lerobot101.load("tag_check")
    where = Path(args.reference) if args.reference else BASKET_REFERENCE
    kept = cv2.imread(str(where))
    if kept is None:
        raise VisionError(f"no reference frame for the basket's place at {where}")
    reference = basket_place.read_basket(kept, tag_check.detect(kept))
    if reference is None:
        raise VisionError(f"the basket cannot be found in the reference frame {where}")
    geometry = load_geometry() if vision_module.TABLE_GEOMETRY.is_file() else None
    window = "basket  [o] lay the old basket over it  [s] save a picture  [x] quit"
    print("\n  click the video window so it gets the keys. Green lines: where the basket's left wall and near rim "
          "belong.\n  Orange lines: where they are. Move the basket until the text at the bottom says IN PLACE, "
          "hands out of the picture for the last look, then x.\n  o steps through three views: the live picture, "
          "the live picture with the old basket laid over it (a basket out of place is seen twice), and the old "
          "basket alone.")
    frame, now, off, view = None, None, None, 0

    def shown() -> tuple[np.ndarray, list[str]]:
        moved = off.camera_px if off is not None else None
        under = frame if view == 0 else basket_place.ghosted(frame, kept, reference, moved, 0.5 if view == 1 else 1.0)
        lines = [*basket_place.said(reference, now, off), f"[o] view: {basket_place.VIEWS[view]}"]
        return basket_place.draw(under, reference, now, off, lines), lines

    with Camera(args.camera) as camera:
        while True:
            frame = camera.grab()
            found = tag_check.detect(frame)
            now = basket_place.read_basket(frame, found)
            off = None if now is None else basket_place.offset(reference, now, basket_place.mm_per_px(found, geometry))
            picture, _lines = shown()
            cv2.imshow(window, picture)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("o"):
                view = (view + 1) % len(basket_place.VIEWS)
            if key == ord("s"):
                when = stamp()
                saved = INSTRUMENT_DIR / f"basket_{when}.png"
                cv2.imwrite(str(saved), picture)
                cv2.imwrite(str(INSTRUMENT_DIR / f"view_raw_{when}.png"), frame)  # the frame itself, unmarked
                print(f"  saved {saved} (and the unmarked frame beside it)")
            if key in (KEY_ABORT, ord("q"), 27):
                break
    when = stamp()
    view = 0
    picture, lines = shown()
    out = INSTRUMENT_DIR / f"basket_{when}.png"
    cv2.imwrite(str(out), picture)
    cv2.imwrite(str(INSTRUMENT_DIR / f"view_raw_{when}.png"), frame)
    view = 1                                  # ... and the same look with the old basket laid over it, to keep
    laid_over = INSTRUMENT_DIR / f"basket_over_{when}.png"
    cv2.imwrite(str(laid_over), shown()[0])
    print(f"\n  the last look, against {where.name}:")
    for line in lines[:-1]:
        print(f"  {line}")
    print(f"  picture: {out}\n  with the old basket laid over it: {laid_over}")
    return 0 if off is not None and off.in_place else 1


# ----------------------------------------------------------------------------------------------------------- wristplace

#: The picture that says where the wrist camera belongs in its mount: the snapshot the first session of 2026-10-08
#: started on, before the mount first caught the basket. The fixed jaw stands in it where it stands in both training
#: sets (inner edge 265.0 px against medians of 265.0 and 264.7; lower hole within half a pixel of theirs).
WRIST_REFERENCE = INSTRUMENT_DIR / "wrist_reference.png"


def mode_wristplace(args: argparse.Namespace) -> int:
    """Show where the wrist camera stands in its mount against its reference picture, live. No arm session."""
    import cv2

    from tolerance_radius.rig import wrist_place

    index = lerobot101.load("rig").WRIST_CAMERA_INDEX if args.camera is None else args.camera
    where = Path(args.reference) if args.reference else WRIST_REFERENCE
    kept = cv2.imread(str(where))
    if kept is None:
        raise VisionError(f"no reference picture for the wrist camera's place at {where}")
    reference = wrist_place.read_jaw(kept)
    if reference is None or reference.hole is None:
        raise VisionError(f"the gripper's fixed jaw cannot be read in the reference picture {where}")
    window = "wrist camera  [o] lay the old picture over it  [s] save a picture  [x] quit"
    print(f"\n  camera {index}. Click the video window so it gets the keys. The gripper's left jaw in the picture is "
          "fixed to the camera:\n  green line and cross are where its inner edge and its hole belong, orange where "
          "they are. o steps through three views:\n  the live picture, the live picture with the old picture of "
          "the jaw laid over it (a camera out of place shows the jaw twice),\n  and the old picture alone. Bring "
          "the camera to where the text says IN PLACE, tighten it while watching that it stays, then x.")
    frame, now, off, view = None, None, None, 0
    last: deque = deque(maxlen=10)

    def shown() -> tuple[np.ndarray, list[str]]:
        under = frame if view == 0 else wrist_place.laid_over(frame, kept, 0.5 if view == 1 else 1.0)
        lines = [*wrist_place.said(now, off), f"[o] view: {wrist_place.VIEWS[view]}"]
        return wrist_place.draw(under, reference, now, off, lines), lines

    with Camera(index) as camera:
        while True:
            frame = camera.grab()
            reading = wrist_place.read_jaw(frame, reference.hole)
            if reading is None:
                last.clear()
            else:
                last.append(reading)
            now = wrist_place.steady(list(last))
            off = None if now is None else wrist_place.offset(reference, now)
            picture, _lines = shown()
            cv2.imshow(window, picture)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("o"):
                view = (view + 1) % len(wrist_place.VIEWS)
            if key == ord("s"):
                when = stamp()
                saved = INSTRUMENT_DIR / f"wrist_place_{when}.png"
                cv2.imwrite(str(saved), picture)
                cv2.imwrite(str(INSTRUMENT_DIR / f"wrist_raw_{when}.png"), frame)  # the frame itself, unmarked
                print(f"  saved {saved} (and the unmarked frame beside it)")
            if key in (KEY_ABORT, ord("q"), 27):
                break
    when = stamp()
    view = 0
    picture, lines = shown()
    out = INSTRUMENT_DIR / f"wrist_place_{when}.png"
    cv2.imwrite(str(out), picture)
    cv2.imwrite(str(INSTRUMENT_DIR / f"wrist_raw_{when}.png"), frame)
    view = 1                                  # ... and the same look with the old picture laid over it, to keep
    over = INSTRUMENT_DIR / f"wrist_place_over_{when}.png"
    cv2.imwrite(str(over), shown()[0])
    print(f"\n  the last look, against {where.name}:")
    for line in lines[:-1]:
        print(f"  {line}")
    print(f"  picture: {out}\n  with the old picture laid over it: {over}")
    return 0 if off is not None and off.in_place else 1


# ------------------------------------------------------------------------------------------------------- register-table


def parse_sizes(text: str, count: int) -> list[tuple[float, float]]:
    """`count` sizes written as WIDTHxHEIGHT and separated by commas, for example `59.7x60.0,59.78x60.0`."""
    sizes = []
    for part in text.split(","):
        width, _, height = part.strip().lower().partition("x")
        sizes.append((float(width), float(height)))
    if len(sizes) != count or not all(value > 0 for size in sizes for value in size):
        raise ValueError(f"{count} sizes are needed, each WIDTHxHEIGHT in millimetres")
    return sizes


def registration_frames(args: argparse.Namespace) -> list[np.ndarray]:
    """The pictures a registration is made from: the camera's, or saved unmarked frames (`--pictures`)."""
    import cv2

    if not args.pictures:
        with Camera(args.camera) as camera:
            return [camera.grab() for _ in range(args.frames)]
    frames = []
    for name in args.pictures:
        frame = cv2.imread(str(name))
        if frame is None:
            raise VisionError(f"cannot read the picture {name}")
        if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
            raise VisionError(f"{name} is {frame.shape[1]} x {frame.shape[0]}, expected {FRAME_SIZE}")
        frames.append(frame)
    return frames


def mode_register(args: argparse.Namespace) -> int:
    """Measure the two floor tags against the seated registration sheet, and store the table geometry."""
    import json

    import cv2

    grid_overlay, tag_check = lerobot101.load("grid_overlay"), lerobot101.load("tag_check")
    out = Path(args.out) if args.out else vision_module.TABLE_GEOMETRY
    if out.is_file() and not args.replace:
        print(f"\n  a table geometry is already stored at {out}.\n  The calibration and every position measured so far"
              " rest on it. Pass --replace only if a floor tag has moved or the table was laid out again, and nothing"
              " recorded with the old one is kept.")
        return 1
    # The sheet and the floor tags as they came out of the printer, if they were measured: before the camera is
    # opened, so that a mistyped size costs nothing.
    outline_mm, sheet_scale, floor_tags_mm = None, (1.0, 1.0), None
    if args.sheet_outline_mm:
        try:
            outline_mm = parse_sizes(args.sheet_outline_mm, 1)[0]
            sheet_scale = registration.sheet_scale_from_outline(*outline_mm)
        except (ValueError, registration.RegistrationError) as error:
            print(f"\n  NOT REGISTERED: --sheet-outline-mm takes the grid's outline as WIDTHxHEIGHT in millimetres, "
                  f"for example 250.25x149.67 ({error})")
            return 1
    if args.floor_tags_mm:
        try:
            floor_tags_mm = dict(zip(registration.FLOOR_TAG_IDS, parse_sizes(args.floor_tags_mm, 2), strict=True))
        except ValueError as error:
            print(f"\n  NOT REGISTERED: --floor-tags-mm takes the two floor tags' sizes by calipers, tag 0 then tag "
                  f"1, each WIDTHxHEIGHT as printed upright, for example 59.7x60.0,59.78x60.0 ({error})")
            return 1
    frames = registration_frames(args)
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    detections = [tag_check.detect(gray) for gray in grays]
    outlines = [outline.tag_outlines(gray, found) for gray, found in zip(grays, detections, strict=True)]
    taken = stamp()
    # A registration worked through again from saved pictures into another file leaves its picture beside that file.
    picture_dir = Path(args.out).parent if (args.pictures and args.out) else INSTRUMENT_DIR
    picture_path = picture_dir / f"table_registration_{taken}.png"
    picture_path.parent.mkdir(parents=True, exist_ok=True)
    source = "saved pictures" if args.pictures else "frames"
    print(f"\n  {len(frames)} {source}; tags seen in the last one: {sorted(detections[-1]) or 'none'}")
    for line in describe_tags(detections[-1]):
        print(f"  {line}")
    if not args.pictures:
        # Three of the frames as they came, so that this registration can be worked through again from its evidence
        # (`register-table --pictures`).
        for number, index in enumerate(sorted({0, len(frames) // 2, len(frames) - 1}), start=1):
            cv2.imwrite(str(INSTRUMENT_DIR / f"table_registration_{taken}_raw{number}.png"), frames[index])
    table = load_protocol()["table"]
    # An over-exposed picture eats into the black of every tag: said here by its name, before it shows as a sheet
    # that "does not fit".
    whites = {tag_id: outline.white_level(grays[-1], quad) for tag_id, quad in sorted(detections[-1].items())}
    if whites:
        print(f"  the white around the tags reads grey level {min(whites.values()):.0f} to {max(whites.values()):.0f} "
              f"(clipped from {outline.CLIPPED_GREY:.0f})")
    clipped = [tag_id for tag_id, level in whites.items() if level >= outline.CLIPPED_GREY]
    if clipped:
        cv2.imwrite(str(picture_path), tag_check.annotate(frames[-1], detections[-1]))
        print(f"  NOT REGISTERED: the picture is over-exposed around tag(s) {clipped}: the white is clipped, which "
              "eats into the black and makes every tag read small. Lower the camera's exposure until the floor reads "
              f"about 225 to 245, then register again.\n  picture: {picture_path}")
        return 1
    if outline_mm is None:
        print("  the sheet is taken as designed (no --sheet-outline-mm): measure its outline with calipers if you can")
    else:
        said = [f"{100 * abs(v - 1.0):.2f} % {'long' if v > 1.0 else 'short'}" for v in sheet_scale]
        print(f"  the sheet as printed: outline {outline_mm[0]:g} x {outline_mm[1]:g} mm, so {said[0]} across and "
              f"{said[1]} up; the tags on it are taken to be where that puts them")
    try:
        mean_gray = np.mean([gray.astype(np.float32) for gray in grays], axis=0)
        result = registration.register(detections, outlines, grid=table, note=args.note, sheet_scale=sheet_scale,
                                       sheet_outline_measured_mm=outline_mm, floor_tags_mm=floor_tags_mm,
                                       picture=mean_gray)
    except registration.RegistrationError as error:
        cv2.imwrite(str(picture_path), tag_check.annotate(frames[-1], detections[-1]))
        print(f"  NOT REGISTERED: {error}\n  picture: {picture_path}")
        return 1

    scale = result.sheet_tag_scale
    each = result.geometry["registration"]["sheet_tag_scale_each"]
    furthest = max(((abs(value - 1.0), tag_id, axis) for tag_id, pair in each.items() for axis, value in enumerate(pair)))
    lines = result.lines
    print(f"  the table frame, through the outlines of sheet tags {list(result.sheet_ids)}:")
    if lines is not None:
        print(f"    the sheet's printed lines, which the fit does not use, lie {lines['rms_mm']:.2f} mm (rms) from the "
              f"frame and {lines['worst_mm']:.2f} mm at worst over 10 mm of a line (noted beyond "
              f"{registration.LINES_NOTE_MM[0]} and {registration.LINES_NOTE_MM[1]}, refused beyond "
              f"{registration.LINES_LIMIT_MM[0]} and {registration.LINES_LIMIT_MM[1]}): this is what judges whether "
              "the sheet lies flat")
        fit_limits = (f"refused only beyond {registration.GROSS_FIT_LIMIT_MM[0]} and "
                      f"{registration.GROSS_FIT_LIMIT_MM[1]}: small tags' outlines are noisy")
        left_out_limits = f"refused only beyond {registration.GROSS_LEAVE_ONE_OUT_LIMIT_MM}"
    else:
        print("    the sheet's printed lines could not be read: the sheet is judged by its tags alone")
        fit_limits = (f"a flat sheet: about 0.2 and 0.45; noted beyond {registration.SHEET_FIT_NOTE_MM[0]} and "
                      f"{registration.SHEET_FIT_NOTE_MM[1]}, refused beyond {registration.SHEET_FIT_LIMIT_MM[0]} and "
                      f"{registration.SHEET_FIT_LIMIT_MM[1]}")
        left_out_limits = (f"noted beyond {registration.LEAVE_ONE_OUT_NOTE_MM}, refused beyond "
                           f"{registration.LEAVE_ONE_OUT_LIMIT_MM}")
    print(f"    the sheet's tags fit one flat plane to {result.sheet_fit_mean_mm:.2f} mm on average, "
          f"{result.sheet_fit_max_mm:.2f} mm at worst ({fit_limits})")
    print(f"    a sheet tag left out is predicted by the others to {result.leave_one_out_mean_mm:.2f} mm on average, "
          f"{result.leave_one_out_max_mm:.2f} mm at worst ({left_out_limits})")
    print(f"    the sheet's tags read {100 * (scale[0] - 1.0):+.1f} % across and {100 * (scale[1] - 1.0):+.1f} % up "
          f"against their printed size on average (refused beyond {100 * registration.SHEET_TAG_SCALE_LIMIT:g} %); "
          f"the one furthest off is tag {furthest[1]}, {100 * (each[furthest[1]][furthest[2]] - 1.0):+.1f} % "
          f"{('across', 'up')[furthest[2]]}")
    print(f"    the detector puts a tag's corners {result.detector_inset_px:.2f} px inside its outline: the frame is "
          "fitted through the sheet tags' outlines, and the floor tags are stored by theirs")
    for tag_id, sides in result.floor_sides_mm.items():
        centre = result.geometry["tags"][str(tag_id)]["centre"]
        by_outline = result.floor_tag_scale[tag_id]
        print(f"  floor tag {tag_id}: centre ({centre[0]:.1f}, {centre[1]:.1f}) mm in the grid's frame; sides "
              f"{', '.join(f'{side:.1f}' for side in sides)} mm by its outline, "
              f"{100 * (by_outline[0] - 1.0):+.1f} % across and {100 * (by_outline[1] - 1.0):+.1f} % up against its "
              f"stated size (noted beyond {100 * registration.FLOOR_TAG_SIZE_NOTE:g} %)")
    few = "" if len(frames) >= 10 else f" (from {len(frames)} frames only: this says little)"
    print(f"  one frame alone: solved from a single frame's floor tags the grid lies {result.wander_mean_mm:.2f} mm "
          f"from the frame on average, {result.wander_max_mm:.2f} mm at worst, taking the largest gap anywhere on "
          f"the grid (refused beyond {registration.WANDER_MEAN_LIMIT_MM} on the average){few}")
    notes, problems = list(result.notes), list(result.problems)
    for tag_id in registration.FLOOR_TAG_IDS:
        seen_in = sum(1 for found in detections if tag_id in found)
        if seen_in < FOUND_SHARE_NOTE * len(detections):
            notes.append(f"floor tag {tag_id} was found in only {seen_in} of {len(detections)} frames: something is "
                         "moving across it, it is in glare, or it is too small in the picture. Sessions find it the "
                         "same way, so put that right first")
    # The whole grid must be in the picture with room to spare, or the view check refuses once the sheet is gone.
    corners_px = registration._apply(np.linalg.inv(result.px_to_mm), np.array(list(GRID_CORNERS.values())))
    for name, (x, y) in zip(GRID_CORNERS, corners_px, strict=True):
        edge = min(x, FRAME_SIZE[0] - x, y, FRAME_SIZE[1] - y)
        if edge < GRID_EDGE_MIN_PX:
            problems.append(f"the grid's corner {name} is {edge:.0f} px from the picture's edge (at least "
                            f"{GRID_EDGE_MIN_PX:.0f}): a cube there must be whole in the picture. Move the sheet "
                            "inward and register again")
    result.geometry["registration"]["notes"] = notes
    for note in notes:
        print(f"  NOTE: {note}")

    # The grid as the floor tags alone place it, over the picture: it must lie on the sheet's printed grid.
    to_px = grid_overlay.mm_to_px(registration.mean_corners(outlines), result.geometry)
    drawn = grid_overlay.draw_grid(tag_check.annotate(frames[-1], detections[-1]), to_px, result.geometry,
                                   allow_holdout=True)
    cv2.imwrite(str(picture_path), drawn)
    print(f"  picture: {picture_path}")
    if problems:
        for problem in problems:
            print(f"  NOT REGISTERED: {problem}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.geometry, indent=2) + "\n", encoding="utf-8")
    print(f"  REGISTERED: table geometry stored at {out}")
    if STORED_VIEW.is_file():
        print("  A calibration stored before this registration no longer applies: run "
              "`scripts/rig/measure.py calibrate --replace` once the sheet is off the table.")
    print("  Look at the picture: the drawn grid must lie on the sheet's printed grid.")
    after = ("take the sheet away without touching the two floor tags or the camera's pipe, then go straight on with "
             "runbook 1, steps 1 to 3 (view check, camera baseline, calibration)")
    if notes:
        print(f"  REGISTERED WITH {len(notes)} NOTE(S): read them above. LEAVE THE SHEET WHERE IT IS, do what they "
              "say, and register again with --replace: it takes two minutes and nothing has been recorded yet. If a "
              f"note is still there after that, say so; only then {after}.")
    else:
        print(f"  Next: {after}.")
    return 0


# ---------------------------------------------------------------------------------------------------------------- focus


def sharpness(gray: np.ndarray) -> float:
    """How sharp a picture is, as one number: the mean squared slope of its brightness (grey levels squared per pixel
    squared) after a light blur that takes the camera's noise out. It rises as edges get crisper and has its largest
    value at best focus; it means nothing between two different scenes."""
    import cv2

    smooth = cv2.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), 1.0)
    across = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    down = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return float(np.mean(across * across + down * down))


def focus_box(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """The middle of the picture, half its width and half its height: (x0, y0, x1, y1)."""
    height, width = shape[:2]
    return width // 4, height // 4, 3 * width // 4, 3 * height // 4


def mode_focus(args: argparse.Namespace) -> int:
    """Show a camera's picture with its sharpness as a number, for a lens that is focused by hand."""
    import cv2

    index = lerobot101.load("rig").WRIST_CAMERA_INDEX if args.camera is None else args.camera
    window = "focus  [r] forget the best so far  [x] quit"
    print(f"\n  camera {index}. Put the arm and the cube as they are when the gripper is about to close, hands out."
          "\n  Turn the lens slowly: the numbers rise towards best focus and fall beyond it. Go past the peak, come"
          "\n  back to it, and fix the lens there. Click the video window first; r forgets the best so far, x quits.")
    best = {"whole": 0.0, "middle": 0.0}
    now = dict(best)
    frame = None
    with Camera(index) as camera:
        while True:
            frame = camera.grab()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x0, y0, x1, y1 = focus_box(gray.shape)
            now = {"whole": sharpness(gray), "middle": sharpness(gray[y0:y1, x0:x1])}
            best = {key: max(best[key], value) for key, value in now.items()}
            picture = frame.copy()
            cv2.rectangle(picture, (x0, y0), (x1, y1), (255, 200, 0), 1)
            at_peak = now["middle"] >= 0.97 * best["middle"]
            for row, key in enumerate(("middle", "whole")):
                text = f"{key}: {now[key]:7.1f}   best so far {best[key]:7.1f}"
                colour = (80, 255, 80) if key == "middle" and at_peak else (0, 190, 255)
                cv2.putText(picture, text, (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(picture, text, (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.imshow(window, picture)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                best = dict(now)
            if key in (KEY_ABORT, ord("q"), 27):
                break
    INSTRUMENT_DIR.mkdir(parents=True, exist_ok=True)
    kept = INSTRUMENT_DIR / f"focus_camera{index}_{stamp()}.png"
    cv2.imwrite(str(kept), frame)
    print(f"\n  left at: the middle of the picture {now['middle']:.1f} (best seen {best['middle']:.1f}), the whole "
          f"picture {now['whole']:.1f} (best seen {best['whole']:.1f})\n  picture as it was left: {kept}")
    if now["middle"] < 0.9 * best["middle"]:
        print("  The lens was left short of the best it showed: run this again and come back to the peak.")
        return 1
    print("  Fix the lens where it is (its lock ring, or a strip of tape), and do not touch it again.")
    return 0


# ---------------------------------------------------------------------------------------------------------------- wrist


def wrist_verdict(readings: Sequence[float | None], errors: int | None) -> tuple[list[str], bool]:
    """What a stretch of the wrist camera's frames says: lines for the operator, and whether it is good to record
    with. `readings` holds one `vision.wrist_contrast` per frame (None where the frame was not the rest picture);
    `errors` is the USB transfer errors on its link in that stretch, None when the system's log could not be read.
    """
    judged = [value for value in readings if value is not None]
    lines, good = [], True
    if not readings:
        return ["the camera gave no frame"], False
    if len(judged) < 0.8 * len(readings):
        lines.append(f"its picture could not be judged in {len(readings) - len(judged)} of {len(readings)} frames: "
                     "it is not the rest picture. Is the arm at rest in its start pose, the jaws over the white "
                     "floor, and are hands out?")
        good = False
    if judged:
        low, high = min(judged), max(judged)
        other = sum(value > WRIST_OTHER_ABOVE for value in judged)
        read = f"the floor over the jaws' green read {low:.2f} to {high:.2f} ({wrist_reads()})"
        if other == 0:
            lines.append(f"{read}: the usual tone state")
        elif other == len(judged):
            lines.append(f"{read}: THE OTHER TONE STATE. Run this again: the camera is opened afresh")
            good = False
        else:
            lines.append(f"{read}: IT CHANGED ITS TONE STATE while this ran ({other} of {len(judged)} frames in "
                         "the other state)")
            good = False
    if errors is None:
        lines.append("USB transfer errors on its link: not read (the system's log could not be read)")
    elif errors:
        lines.append(f"USB transfer errors on its link: {errors}. Frames are being dropped: an extension cable "
                     "(take it out), its plug, its adapter, another port, and the small connector at the camera "
                     "itself")
        good = False
    else:
        lines.append("USB transfer errors on its link: none")
    return lines, good


def mode_wrist(args: argparse.Namespace) -> int:
    """Watch the wrist camera for a while with the arm at rest: its tone state, and its USB link. Camera only."""
    from tolerance_radius import usb_errors
    from tolerance_radius.rig.camera_list import WRIST_NAME

    index = lerobot101.load("rig").WRIST_CAMERA_INDEX if args.camera is None else args.camera
    print(f"\n  camera {index}, for {args.seconds:g} seconds. The arm at rest in its start pose (the jaws over the "
          "white floor), hands out.\n  Nothing is shown and the arm is not touched; a line every five seconds.")
    began = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = time.monotonic()
    readings: list[float | None] = []
    with Camera(index) as camera:
        said = time.monotonic()
        since_said: list[float | None] = []
        while time.monotonic() - started < args.seconds:
            value = wrist_contrast(camera.grab())
            readings.append(value)
            since_said.append(value)
            if time.monotonic() - said >= 5.0:
                judged = [v for v in since_said if v is not None]
                print(f"  {time.monotonic() - started:4.0f} s: " + (f"{float(np.median(judged)):.2f}" if judged
                                                                      else "not the rest picture"))
                said, since_said = time.monotonic(), []
    try:
        minutes = max(1, int((time.monotonic() - started) // 60) + 2)
        errors = usb_errors.since(usb_errors.system_log(minutes), WRIST_NAME, began)
    except usb_errors.LogError:
        errors = None
    lines, good = wrist_verdict(readings, errors)
    print(f"\n  the wrist camera, {len(readings)} frames in {time.monotonic() - started:.0f} seconds:")
    for line in lines:
        print(f"  {line}")
    print("\n  " + ("WRIST CAMERA GOOD" if good else "WRIST CAMERA NOT GOOD"))
    try:
        WRIST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(WRIST_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{began} | camera {index} | {len(readings)} frames | {'GOOD' if good else 'NOT GOOD'} | "
                         + " | ".join(lines) + "\n")
    except OSError as error:  # the run's result is on the screen; a log that cannot be written costs nothing else
        print(f"  (not written to {WRIST_LOG.name}: {error})")
    return 0 if good else 4


# ------------------------------------------------------------------------------------------------------------ calibrate


def floor_tags_then_and_now(
    geometry: dict, frames: list[np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """The floor tags' outline corners in the picture when the table was registered, and now: two dictionaries, tag
    id to four corners in pixels. None when the table geometry does not hold where they were, or a tag is not seen."""
    stored = (geometry.get("registration") or {}).get("floor_tag_corners_px")
    if not stored:
        return None
    then = {int(tag_id): np.asarray(corners, dtype=float) for tag_id, corners in stored.items()}
    now, _found, _measured, _count = tag_outlines_over(frames, list(then))
    if sorted(now) != sorted(then) or sorted(then) != sorted(registration.FLOOR_TAG_IDS):
        return None
    return then, now


def pair_motion(then: dict[int, np.ndarray], now: dict[int, np.ndarray], registered_px_to_mm) -> dict[str, Any]:
    """How the two floor tags have moved in the picture since the table was registered, each and as a pair.

    `shift_px`: each tag's shift, the mean over its four outline corners. `differ_px`: how far the two shifts are
    apart. `turn_deg` and `stretch`: the turn of the line from the first tag's centre to the second's, and the change
    of its length as a share. `relative_mm`: the two tags' movement against each other read on the floor, each tag's
    centre put through the registered frame before and after; `tags_apart_mm`: how far apart the two tags lie. A
    camera that turned or came closer and a tag that was moved on the floor both show here, and nothing in these
    figures tells them apart.
    """
    first, second = registration.FLOOR_TAG_IDS
    matrix = np.asarray(registered_px_to_mm, dtype=float)

    def on_floor(pixel: np.ndarray) -> np.ndarray:
        x, y, w = matrix @ np.array([pixel[0], pixel[1], 1.0])
        return np.array([x / w, y / w])

    centre_then = {tag_id: then[tag_id].mean(axis=0) for tag_id in (first, second)}
    centre_now = {tag_id: now[tag_id].mean(axis=0) for tag_id in (first, second)}
    shift = {tag_id: centre_now[tag_id] - centre_then[tag_id] for tag_id in (first, second)}
    line_then, line_now = centre_then[second] - centre_then[first], centre_now[second] - centre_now[first]
    turn = np.arctan2(line_then[0] * line_now[1] - line_then[1] * line_now[0], float(line_then @ line_now))
    moved_mm = {tag_id: on_floor(centre_now[tag_id]) - on_floor(centre_then[tag_id]) for tag_id in (first, second)}
    return {
        "shift_px": {tag_id: [round(float(v), 3) for v in shift[tag_id]] for tag_id in (first, second)},
        "larger_px": float(max(np.linalg.norm(shift[first]), np.linalg.norm(shift[second]))),
        "differ_px": float(np.linalg.norm(shift[second] - shift[first])),
        "turn_deg": float(np.degrees(turn)),
        "stretch": float(np.linalg.norm(line_now) / np.linalg.norm(line_then) - 1.0),
        "relative_mm": float(np.linalg.norm(moved_mm[second] - moved_mm[first])),
        "tags_apart_mm": float(np.linalg.norm(on_floor(centre_then[second]) - on_floor(centre_then[first]))),
    }


def mode_calibrate(args: argparse.Namespace) -> int:
    import cv2

    if STORED_VIEW.is_file() and not args.replace:
        print(f"\n  a calibration is already stored at {STORED_VIEW}.\n  Every measurement so far went through it. "
              f"Pass --replace only if the camera or a tag has moved and nothing recorded with the old one is kept, "
              f"or after the table was registered again.")
        return 1
    geometry = load_geometry()
    tag_check = lerobot101.load("tag_check")
    with Camera(args.camera) as camera:
        frames = [camera.grab() for _ in range(args.frames)]
        last_frame = camera.grab()
    sheet_in_view = sorted(set(tag_check.detect(last_frame)) & set(registration.SHEET_TAGS))
    if sheet_in_view:
        print(f"\n  NOT STORED: the registration sheet is still on the floor (its tags {sheet_in_view} are in the "
              "picture). Take it away without touching the two floor tags, then run this again.")
        return 1
    half = len(frames) // 2
    halves = [TableView.from_frames(frames[:half], geometry), TableView.from_frames(frames[half:], geometry)]
    view = TableView.from_frames(frames, geometry)
    probes = grid_probes_px(view, geometry)
    repeat = halves[0].disagreement_mm(halves[1], probes)
    print(f"\n  solved from {view.frames_used} frames; fit of the two tags to the stored geometry: "
          f"{view.residual_mm:.2f} mm (limit {RESIDUAL_LIMIT_MM})")
    print(f"  two independent solves of {half} frames each agree within {repeat:.2f} mm over the grid")
    if not view.residual_mm <= RESIDUAL_LIMIT_MM:
        print("  NOT STORED: the two tags do not fit the stored table geometry. A tag has probably moved.")
        return 1
    if repeat > AGREEMENT_LIMIT_MM / 3:
        print("  NOT STORED: the solve does not repeat well enough. Check the light on the tags and the focus.")
        return 1
    # The frame as it came, kept beside the calibration: what the picture looked like when it was made, and what
    # the registration's own pictures are held against when the tags are found to have moved.
    INSTRUMENT_DIR.mkdir(parents=True, exist_ok=True)
    kept = INSTRUMENT_DIR / f"calibration_{stamp()}.png"
    cv2.imwrite(str(kept), last_frame)
    print(f"  picture: {kept}")
    # Has anything moved since the table was registered? The two tags' fit above cannot say: a tag moved by a
    # millimetre still fits. Where each tag's outline now is in the picture, against where it was, can.
    record: dict[str, Any] = {"picture": kept.name}
    seen = floor_tags_then_and_now(geometry, frames)
    registered = (geometry.get("registration") or {}).get("px_to_mm")
    if seen is not None and registered is not None:
        then, now = seen
        first, second = registration.FLOOR_TAG_IDS
        motion = pair_motion(then, now, registered)
        apart = registration.frames_apart_mm(registered, view.px_to_mm, geometry["grid"])
        record["floor_tag_corners_px"] = {str(tag_id): np.round(now[tag_id], 3).tolist() for tag_id in sorted(now)}
        record["since_registration"] = {**motion, "shift_px": {str(k): v for k, v in motion["shift_px"].items()},
                                        "frames_apart_mm": round(apart, 3)}
        (x0, y0), (x1, y1) = motion["shift_px"][first], motion["shift_px"][second]
        print(f"  since the table was registered: floor tag {first} has moved by ({x0:+.2f}, {y0:+.2f}) px in "
              f"the picture and floor tag {second} by ({x1:+.2f}, {y1:+.2f}) px; this solve lies within "
              f"{apart:.2f} mm of the registered frame everywhere on the grid")
        pair = (f"their shifts differ by {motion['differ_px']:.1f} px; seen as a pair they have turned by "
                f"{motion['turn_deg']:+.2f} degrees and the distance between them has changed by "
                f"{100 * motion['stretch']:+.2f} %. Read on the floor through the registered frame they have moved "
                f"{motion['relative_mm']:.1f} mm against each other")
        either = ("A camera that turned a little about its own line of sight, or came closer, does this; so does a "
                  "floor tag that was moved (pressed flat, bumped while the sheet was taken away, its tape gave). The "
                  "two tags by themselves cannot say which")
        if motion["relative_mm"] > RELATIVE_MOVE_LIMIT_MM:
            print(f"  NOT STORED: the two floor tags have not moved alike: {pair} (refused beyond "
                  f"{RELATIVE_MOVE_LIMIT_MM} mm). {either}, and if it was a tag the frame is off by about that much. "
                  "Do not seat the sheet again yet. Say so: the picture saved above is held against the "
                  "registration's own (analysis/instrument/picture_motion.py) to see whether the rest of the picture "
                  "moved with the tags. If it did, it was the camera and the table need not be registered again; if "
                  "it did not, seat the sheet and register the table again (runbook 0, step 4, with --replace), "
                  "then calibrate straight after.")
            return 1
        if motion["differ_px"] > TAG_SHIFT_NOTE_PX:
            print(f"  NOTE: the two floor tags have not moved alike since the table was registered: {pair} (noted "
                  f"beyond {TAG_SHIFT_NOTE_PX} px, refused beyond {RELATIVE_MOVE_LIMIT_MM} mm). {either}. The frame "
                  "is taken from the tags as they are seen now, which is right if it was the camera. If it was a "
                  f"tag, positions on the grid are off by up to about {motion['relative_mm']:.1f} mm and distances "
                  f"by about {100 * motion['relative_mm'] / motion['tags_apart_mm']:.1f} %: the check against a ruler "
                  "(runbook 1, step 4) is what judges distances. To see which it was, the picture saved above is "
                  "held against the registration's own (analysis/instrument/picture_motion.py): do that whenever "
                  f"this reads more than {RELATIVE_MOVE_NOTE_MM} mm.")
        elif motion["larger_px"] > CAMERA_SHIFT_NOTE_PX:
            print(f"  NOTE: the camera's picture has moved by about {motion['larger_px']:.1f} px since the table was "
                  "registered (both floor tags moved the same way). The tags lie where they lay, so the frame is "
                  "taken from them as they are seen now. If this happens again, look at the camera's mount.")
    try:
        view.floor_levels = picture_levels(view, last_frame, geometry)
        print(f"  the picture's levels: the floor inside the grid at grey level {view.floor_levels['grey']:.0f}, the "
              f"floor tags' black at {view.floor_levels.get('tag_black', float('nan')):.0f}, red over blue "
              f"{view.floor_levels['red_over_blue']:.2f}. Every session holds the picture it opens on against "
              "these and refuses to start on another.")
    except VisionError as error:
        print(f"  (the picture's levels were not stored: {error})")
    view.save(STORED_VIEW, note=args.note, record=record)
    print(f"  stored: {STORED_VIEW}\n  Commit this file. Every session takes the table frame from the floor tags as "
          f"it finds them and holds this against it: it says so when the camera's picture has moved more than "
          f"{VIEW_MOVED_NOTE_MM} mm from here, and refuses beyond {VIEW_MOVED_LIMIT_MM} mm.")
    return 0


def probes_mm_centre(geometry: dict) -> tuple[float, float]:
    """The centre of the grid, in table millimetres."""
    grid = geometry["grid"]
    return grid["cols"] * grid["cell_mm"] / 2.0, grid["rows"] * grid["cell_mm"] / 2.0


# ------------------------------------------------------------------------------------------------------------------ log


def mode_log(args: argparse.Namespace) -> int:
    import cv2

    out = Path(args.out) if args.out else REPO_ROOT / "data" / "instrument" / f"{args.label}_{stamp()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    window = "measure  [space] log a reading  [x] quit"
    with Camera(args.camera) as camera, open(out, "w", newline="", encoding="utf-8") as handle:
        view = table_view(camera)
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "x_mm", "y_mm", "scatter_mm", "frames", "time"])
        recent: deque[CubeMeasurement | None] = deque(maxlen=LOG_FRAMES)
        logged = 0
        # The grid is not drawn on the floor: the window shows where it is, through the stored conversion, with the
        # cells' names. Drawn on a copy: the cube is measured in the frame as it came.
        grid_overlay, geometry, grid_to_px = lerobot101.load("grid_overlay"), load_geometry(), np.linalg.inv(view.px_to_mm)
        print(f"\n  logging to {out}\n  click the video window, then: space logs a reading, x quits\n"
              "  the window shows the grid and its cells: x runs from column A (x = 0) to E (250), y from row 1 "
              "(y = 0) to row 3 (150)\n")
        while True:
            frame = camera.grab()
            recent.append(view.measure(frame))
            readings = [m for m in recent if m is not None]
            still = len(readings) == LOG_FRAMES and _scatter_mm(readings) <= STILL_MM
            picture = grid_overlay.draw_grid(frame, grid_to_px, geometry, allow_holdout=True)  # returns a copy
            if readings:
                here = readings[-1]
                text = f"x {here.position.x_mm:6.1f}  y {here.position.y_mm:6.1f} mm   logged {logged}"
                cv2.circle(picture, tuple(int(round(v)) for v in here.centroid_px), 3, (255, 255, 255), -1)
            else:
                text = f"no single clean cube in view   logged {logged}"
            colour = (80, 255, 80) if still else (0, 190, 255)
            cv2.putText(picture, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(picture, "steady: space logs" if still else "wait for steady", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.imshow(window, picture)
            key = cv2.waitKey(30) & 0xFF
            if key == KEY_ABORT:
                break
            if key == KEY_ACCEPT and still:
                middle = median_position(readings)
                writer.writerow([logged, args.label, f"{middle.x_mm:.2f}", f"{middle.y_mm:.2f}",
                                 f"{_scatter_mm(readings):.2f}", len(readings), datetime.now().isoformat(timespec="seconds")])
                handle.flush()
                print(f"  {logged:3d}  x {middle.x_mm:7.2f}  y {middle.y_mm:7.2f} mm")
                logged += 1
                recent.clear()
    print(f"\n  {logged} readings in {out}")
    return 0


def _scatter_mm(readings: list[CubeMeasurement]) -> float:
    """Largest distance of any reading from the middle of the set."""
    middle = median_position(readings)
    return max(m.position.distance_to(middle) for m in readings)


# ---------------------------------------------------------------------------------------------------------------- place


def mode_place(args: argparse.Namespace) -> int:
    tolerance = load_protocol()["placement"]["tolerance_mm"]
    targets = [Point(*(float(v) for v in pair.split(","))) for pair in args.targets]
    with Camera(args.camera) as camera:
        view = table_view(camera)
        phase = PlacementPhase(view, camera.grab, tolerance_mm=tolerance, grid=load_geometry()["grid"],
                               on_end=lambda: None)          # the window stays between two targets
        print("\n  click the video window, then: space accepts once steady, v skips, x quits\n")
        for number, target in enumerate(targets, start=1):
            started = time.monotonic()
            result = phase.run(target, f"target {number}/{len(targets)}  ({target.x_mm:.0f}, {target.y_mm:.0f}) mm")
            if result.status == "aborted":
                break
            where = f"measured ({result.measured.x_mm:.1f}, {result.measured.y_mm:.1f}), off {result.error_mm:.1f} mm" \
                if result.accepted else result.status
            print(f"  target ({target.x_mm:.1f}, {target.y_mm:.1f}): {where}  in {time.monotonic() - started:.0f} s")
    return 0


# ---------------------------------------------------------------------------------------------------------------- byeye


def side_by_side(reference_bgr: np.ndarray, live_bgr: np.ndarray, text: str, steady: bool) -> np.ndarray:
    """The picture for a reset by eye: the reference on the left, the live view on the right, and no marks on either."""
    import cv2

    picture = np.hstack([reference_bgr, live_bgr])
    colour = (80, 255, 80) if steady else (0, 190, 255)
    cv2.putText(picture, "reference", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(picture, text, (reference_bgr.shape[1] + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return picture


def byeye_order(count: int, rng: np.random.Generator) -> list[int]:
    """The order in which the references are put back by eye: shuffled, and never starting with the one taken last.

    A cube put back seconds after it sat on the ring is placed from the memory of the hand, which an evaluation that
    runs over days does not have. So all the reference pictures are taken first and the resets follow in another
    order.
    """
    order = [int(i) for i in rng.permutation(count)]
    while count > 1 and order[0] == count - 1:
        order = [int(i) for i in rng.permutation(count)]
    return order


def _wait_until_off_the_table(camera: Camera, view: TableView, window: str, reference: np.ndarray) -> bool:
    """Show the reference beside the live view until no cube has been seen for a run of frames.

    False when the operator quits.
    """
    import cv2

    clear = 0
    while clear < LOG_FRAMES:
        frame = camera.grab()
        clear = clear + 1 if view.measure(frame) is None else 0
        cv2.imshow(window, side_by_side(reference, frame, "take the cube off the table", False))
        if cv2.waitKey(30) & 0xFF == KEY_ABORT:
            return False
    return True


def mode_byeye(args: argparse.Namespace) -> int:
    import cv2

    protocol = load_protocol()
    rng = derived_rng(int(protocol["datasets"]["layout_seed"]), "reset-by-eye", args.label)
    targets = uniform_placements(Grid.from_protocol(protocol), args.count, rng, "trained_cells")
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "instrument" / f"byeye_{args.label}_{stamp()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    window = "reset by eye  [space] done  [x] quit"
    done = 0
    with Camera(args.camera) as camera, open(out, "w", newline="", encoding="utf-8") as handle:
        view = table_view(camera)
        guided = PlacementPhase(view, camera.grab, tolerance_mm=protocol["placement"]["tolerance_mm"], window=window,
                                grid=load_geometry()["grid"], on_end=lambda: None)
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "target_x_mm", "target_y_mm", "reference_x_mm", "reference_y_mm",
                         "eye_x_mm", "eye_y_mm", "error_mm", "seconds", "order", "since_reference_s"])
        print(f"\n  logging to {out}\n  click the video window. Two passes over {len(targets)} targets:\n"
              "    FIRST, for each target: put the cube on the ring, wait for steady, press space (a reference picture"
              " is taken);\n"
              "    THEN, in another order: put the cube back from the reference picture alone, hands out, and press"
              " space when it says steady; take it off the table again.\n")

        # First pass: where each reference really was, and the picture of it.
        references: list[tuple[Point, Point, np.ndarray, float]] = []
        for number, target in enumerate(targets, start=1):
            placed = guided.run(target, f"reference {number}/{len(targets)}: WITH the ring, then space")
            if placed.status == "aborted":
                print(f"\n  stopped while taking the references: nothing logged in {out}")
                return 0
            if not placed.accepted:
                continue
            references.append((target, placed.measured, camera.grab(), time.monotonic()))

        # Second pass: each one put back by eye, in a shuffled order, from an empty table.
        for position, index in enumerate(byeye_order(len(references), rng), start=1):
            target, reference_at, reference_picture, taken = references[index]
            # The cube has to leave the table before it is put back, or nothing is being tested.
            if not _wait_until_off_the_table(camera, view, window, reference_picture):
                break
            started = time.monotonic()
            recent: deque[CubeMeasurement | None] = deque(maxlen=LOG_FRAMES)
            quit_now = False
            while True:
                frame = camera.grab()
                recent.append(view.measure(frame))
                readings = [m for m in recent if m is not None]
                steady = len(readings) == LOG_FRAMES and _scatter_mm(readings) <= STILL_MM
                text = (f"{position}/{len(references)}  steady: space when it matches" if steady
                        else f"{position}/{len(references)}  put it back by eye, hands out")
                cv2.imshow(window, side_by_side(reference_picture, frame, text, steady))
                key = cv2.waitKey(30) & 0xFF
                if key == KEY_ABORT:
                    quit_now = True
                    break
                if key == KEY_ACCEPT and steady:
                    eye = median_position(readings)
                    error = eye.distance_to(reference_at)
                    now = time.monotonic()
                    writer.writerow([index, args.label, f"{target.x_mm:.2f}", f"{target.y_mm:.2f}",
                                     f"{reference_at.x_mm:.2f}", f"{reference_at.y_mm:.2f}",
                                     f"{eye.x_mm:.2f}", f"{eye.y_mm:.2f}", f"{error:.2f}", f"{now - started:.1f}",
                                     position, f"{now - taken:.0f}"])
                    handle.flush()
                    print(f"  {position:2d}  by eye: {error:5.1f} mm from where the ring put it")
                    done += 1
                    break
            if quit_now:
                break
    print(f"\n  {done} resets in {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=None, help="capture index; default: the rig's front camera")
    modes = parser.add_subparsers(dest="mode", required=True)
    check = modes.add_parser("check", help="can the table frame be placed from this view?")
    check.add_argument("--out", default=None, help="where to save the annotated picture")
    check.add_argument("--tags-relative-limit", type=float, default=None, metavar="MM",
                       help=f"how far the two floor tags may read against each other since the registration "
                            f"(default {RELATIVE_MOVE_LIMIT_MM:g}); for an operator who has seen in the pictures "
                            "that the tags moved with everything else")
    modes.add_parser("live", help="the view the arm sessions will see, live, for positioning the camera")
    register = modes.add_parser("register-table", help="measure the two floor tags against the registration sheet")
    register.add_argument("--frames", type=int, default=REGISTRATION_FRAMES)
    register.add_argument("--out", default=None, help="default: config/table_geometry.json")
    register.add_argument("--note", default="", help="a line to store with it, for example the layout's name")
    register.add_argument("--sheet-outline-mm", default=None, metavar="WIDTHxHEIGHT",
                          help="the grid's outline on the printed sheet as measured with calipers, outer edge to "
                               "outer edge, long side first: for example 250.25x149.67. Corrects for a printer that "
                               "does not print at exactly 100 %% both ways")
    register.add_argument("--floor-tags-mm", default=None, metavar="WxH,WxH",
                          help="the two floor tags' sizes by calipers, tag 0 then tag 1, each as printed upright: "
                               "for example 59.7x60.0,59.78x60.0. Without it they are taken as 60 x 60")
    register.add_argument("--pictures", nargs="+", default=None, metavar="PNG",
                          help="register from saved unmarked frames instead of the camera: to work a registration "
                               "through again from its evidence")
    register.add_argument("--replace", action="store_true", help="overwrite a stored table geometry")
    calibrate = modes.add_parser("calibrate", help="store the conversion from pixels to table millimetres")
    calibrate.add_argument("--frames", type=int, default=CALIBRATION_FRAMES)
    calibrate.add_argument("--note", default="", help="a line to store with it, for example the baseline's name")
    calibrate.add_argument("--replace", action="store_true", help="overwrite a stored calibration")
    log = modes.add_parser("log", help="log the cube's measured position at each press of space")
    log.add_argument("--label", default="readings", help="a short name for this set of readings")
    log.add_argument("--out", default=None, help="CSV file; default data/instrument/<label>_<time>.csv")
    basket = modes.add_parser("basket", help="put the basket back where it belongs: its place, live, against a "
                                             "reference frame")
    basket.add_argument("--reference", default=None, metavar="PNG",
                        help="the unmarked frame that shows the basket where it belongs; default "
                             "data/instrument/basket_reference.png")
    wristplace = modes.add_parser("wristplace", help="where the wrist camera stands in its mount, live, against "
                                                      "the picture of the gripper's fixed jaw as it stood in training")
    wristplace.add_argument("--reference", default=None, metavar="PNG",
                            help="the unmarked wrist picture that shows the fixed jaw where it belongs; default "
                                 "data/instrument/wrist_reference.png")
    place = modes.add_parser("place", help="rehearse placing the cube on targets")
    place.add_argument("targets", nargs="+", help="targets as x,y in table millimetres, for example 75,25 125,75")
    modes.add_parser("focus", help="a camera's picture with its sharpness as a number (default: the wrist camera), "
                                   "for a lens focused by hand")
    wrist = modes.add_parser("wrist", help="the wrist camera for a minute, the arm at rest: its tone state, and the "
                                           "USB transfer errors on its link")
    wrist.add_argument("--seconds", type=float, default=60.0, help="how long to watch (default 60)")
    byeye = modes.add_parser("byeye", help="measure how far off a reset by eye lands")
    byeye.add_argument("--count", type=int, default=20, help="how many resets")
    byeye.add_argument("--label", default="op1", help="who is placing, as a role code")
    byeye.add_argument("--out", default=None, help="CSV file; default data/instrument/byeye_<label>_<time>.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lerobot101.check_pin(allow_other=True)
        modes = {"check": mode_check, "live": mode_live, "register-table": mode_register, "calibrate": mode_calibrate,
                 "log": mode_log, "place": mode_place, "byeye": mode_byeye, "focus": mode_focus,
                 "wrist": mode_wrist, "basket": mode_basket, "wristplace": mode_wristplace}
        return modes[args.mode](args)
    except (VisionError, lerobot101.ToolingError) as error:
        print(f"\n  {error}", file=sys.stderr)
        return 1
