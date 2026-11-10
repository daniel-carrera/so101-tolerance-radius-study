"""The scene at the start and the end of every trial, read from the trial's own two videos. No camera, no arm.

The protocol (section 8, frozen 2026-10-08): the basket stands on tape and the arm can carry it off, and the wrist
camera can be moved in its mount by the same catch. So for every trial the first frames of its videos are read for
the basket's place and the wrist camera's place, and a trial that began with the basket more than 2 mm or the wrist
camera more than 2 px from its place is named in the report. The last frames say whether the trial itself moved
either.

What is read is what the two live views read (`basket_place`, `wrist_place`), against the same reference pictures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius.rig import basket_place, wrist_place

#: Beyond these at a trial's first frames, the trial is named in the report (protocol, section 8).
BASKET_NAMED_MM, WRIST_NAMED_PX = 2.0, 2.0
#: A basket that ends a trial this far from where it began was moved in it.
BASKET_MOVED_MM = 5.0
#: How many frames at each end of a video are read, and how far apart: the median of them is the reading.
FRAMES_READ, FRAME_STEP = 5, 3
#: The arm standing before the basket hides its left wall, and its dark parts are then read for the wall (the end of
#: T1a-012 on 2026-10-08: "53 mm" for a basket that had not moved). More of the arm's orange than this many pixels in
#: the strip the wall is searched in, from this far left of the wall's place: the frame says nothing of the basket.
ARM_BEFORE_BASKET_PX, ARM_STRIP_LEFT_PX = 150, 70


def frames_at(path: Path, *, end: bool) -> list[np.ndarray]:
    """A few frames from the start of a video, or from its end; empty when the video cannot be read."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return []
        wanted = [k * FRAME_STEP for k in range(FRAMES_READ)]
        indexes = [count - 1 - k for k in reversed(wanted)] if end else wanted
        frames = []
        for index in indexes:
            if not 0 <= index < count:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def arm_before_basket(frame: np.ndarray, reference: basket_place.BasketReading) -> bool:
    """Whether the arm stands in the strip of the picture that the basket's left wall is searched in."""
    rows = slice(basket_place.WALL_ROWS[0], basket_place.WALL_ROWS[-1] + 1)
    left = max(0, round(reference.wall_px) - ARM_STRIP_LEFT_PX)
    strip = wrist_place.orange(frame[rows, left:round(reference.wall_px) + 3])
    return int(strip.sum()) > ARM_BEFORE_BASKET_PX


def basket_in(frames: list[np.ndarray], reference: basket_place.BasketReading, detect: Any,
              geometry: dict | None) -> dict[str, float] | None:
    """Where the basket stands in these frames against its reference: the median over those it could be read in.
    A frame in which the arm stands before the basket is not read."""
    offsets = []
    for frame in frames:
        if arm_before_basket(frame, reference):
            continue
        found = detect(frame)
        now = basket_place.read_basket(frame, found)
        if now is not None:
            offsets.append(basket_place.offset(reference, now, basket_place.mm_per_px(found, geometry)))
    if not offsets:
        return None
    return {"across_mm": round(float(np.median([o.across_mm for o in offsets])), 1),
            "down_mm": round(float(np.median([o.down_mm for o in offsets])), 1),
            "lean": round(float(np.median([o.lean for o in offsets])), 1)}


def wrist_in(frames: list[np.ndarray], reference: wrist_place.JawReading) -> dict[str, float | None] | None:
    """Where the gripper's fixed jaw stands in these frames of the wrist camera against its reference."""
    readings = [reading for reading in (wrist_place.read_jaw(frame, reference.hole) for frame in frames)
                if reading is not None]
    steady = wrist_place.steady(readings)
    if steady is None:
        return None
    off = wrist_place.offset(reference, steady)
    return {"across_px": round(off.across_px, 1), "down_px": None if off.down_px is None else round(off.down_px, 1),
            "by": "hole" if off.down_px is not None else "edge"}


def far(reading: dict[str, Any] | None, keys: tuple[str, str], limit: float) -> bool | None:
    """Whether a reading stands beyond `limit` either way; None when there is no reading to say."""
    if reading is None:
        return None
    values = [reading[key] for key in keys if reading.get(key) is not None]
    return bool(values) and max(abs(float(value)) for value in values) > limit


def audit_trial(front: Path | None, wrist: Path | None, basket_reference: basket_place.BasketReading,
                jaw_reference: wrist_place.JawReading, detect: Any, geometry: dict | None) -> dict[str, Any]:
    """One trial's scene: the basket and the wrist camera at its first frames and at its last."""
    out: dict[str, Any] = {"basket_start": None, "basket_end": None, "wrist_start": None, "wrist_end": None}
    if front is not None and front.is_file():
        out["basket_start"] = basket_in(frames_at(front, end=False), basket_reference, detect, geometry)
        out["basket_end"] = basket_in(frames_at(front, end=True), basket_reference, detect, geometry)
    if wrist is not None and wrist.is_file():
        out["wrist_start"] = wrist_in(frames_at(wrist, end=False), jaw_reference)
        out["wrist_end"] = wrist_in(frames_at(wrist, end=True), jaw_reference)
    start, end = out["basket_start"], out["basket_end"]
    out["basket_moved_mm"] = None if start is None or end is None else round(float(np.hypot(
        end["across_mm"] - start["across_mm"], end["down_mm"] - start["down_mm"])), 1)
    out["basket_was_moved"] = None if out["basket_moved_mm"] is None else out["basket_moved_mm"] > BASKET_MOVED_MM
    out["named_basket"] = far(start, ("across_mm", "down_mm"), BASKET_NAMED_MM)
    out["named_wrist"] = far(out["wrist_start"], ("across_px", "down_px"), WRIST_NAMED_PX)
    return out


def audit_log(log: Path, root: Path, basket_picture: Path, wrist_picture: Path, detect: Any,
              geometry: dict | None) -> list[dict[str, Any]]:
    """Every trial of a log that left videos, in the log's order, with its scene."""
    import cv2

    basket_frame, wrist_frame = cv2.imread(str(basket_picture)), cv2.imread(str(wrist_picture))
    if basket_frame is None or wrist_frame is None:
        raise FileNotFoundError(f"a reference picture is missing: {basket_picture} or {wrist_picture}")
    basket_reference = basket_place.read_basket(basket_frame, detect(basket_frame))
    jaw_reference = wrist_place.read_jaw(wrist_frame)
    if basket_reference is None or jaw_reference is None:
        raise ValueError("the basket or the fixed jaw cannot be read in its reference picture")
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        data = entry.get("data", entry)
        media = data.get("media") or {}
        front = root / media["front"] if media.get("front") else None
        wrist = root / media["wrist"] if media.get("wrist") else None
        planned = data.get("planned") or {}
        rows.append({"trial_id": planned.get("trial_id"), "condition": planned.get("condition"),
                     "outcome": data.get("outcome"), "session": None if front is None else front.parent.name,
                     **audit_trial(front, wrist, basket_reference, jaw_reference, detect, geometry)})
    # A trial whose last frames could not say (the arm stood before the basket) is judged by where the basket stood
    # when the next trial of the same session began: nobody touches the basket inside a session.
    for row, after in zip(rows, rows[1:]):
        start, later = row["basket_start"], after["basket_start"]
        if row["basket_was_moved"] is None and start is not None and later is not None \
                and row["session"] is not None and row["session"] == after["session"]:
            row["basket_moved_mm"] = round(float(np.hypot(later["across_mm"] - start["across_mm"],
                                                          later["down_mm"] - start["down_mm"])), 1)
            row["basket_was_moved"] = row["basket_moved_mm"] > BASKET_MOVED_MM
            row["basket_end_by"] = "the next trial's first frames"
    return rows


def told(rows: list[dict[str, Any]]) -> list[str]:
    """The table for the terminal, and what it comes to."""

    def pair(reading: dict[str, Any] | None, keys: tuple[str, str]) -> str:
        if reading is None:
            return "       ?      "
        first, second = reading.get(keys[0]), reading.get(keys[1])
        return f"{first:+6.1f} {'     ?' if second is None else f'{second:+6.1f}'}"

    lines = ["trial      kind            outcome   basket at start   at end (mm)     wrist at start   at end (px)   "
             "note"]
    for row in rows:
        notes = [text for flag, text in ((row["basket_was_moved"], f"basket moved {row['basket_moved_mm']} mm"),
                                         (row["named_basket"], "began with the basket out of place"),
                                         (row["named_wrist"], "began with the wrist camera over 2 px off")) if flag]
        lines.append(f"{row['trial_id'] or '?':10} {row['condition'] or '?':15} {row['outcome'] or '?':8}  "
                     f"{pair(row['basket_start'], ('across_mm', 'down_mm'))}   "
                     f"{pair(row['basket_end'], ('across_mm', 'down_mm'))}   "
                     f"{pair(row['wrist_start'], ('across_px', 'down_px'))}   "
                     f"{pair(row['wrist_end'], ('across_px', 'down_px'))}   {'; '.join(notes)}")
    read = [row for row in rows if row["basket_start"] is not None]
    lines.append(f"{len(rows)} trials with a record; the basket was moved in "
                 f"{sum(1 for row in rows if row['basket_was_moved'])}; began with the basket more than "
                 f"{BASKET_NAMED_MM:g} mm out of place: {sum(1 for row in read if row['named_basket'])}; with the "
                 f"wrist camera more than {WRIST_NAMED_PX:g} px from its place: "
                 f"{sum(1 for row in rows if row['named_wrist'])}; videos that could not be read: "
                 f"{sum(1 for row in rows if row['basket_start'] is None or row['wrist_start'] is None)}; trials "
                 f"whose end says nothing of the basket (the arm before it, and no later trial in the session): "
                 f"{sum(1 for row in rows if row['basket_start'] is not None and row['basket_was_moved'] is None)}")
    return lines
