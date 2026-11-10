#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter (the one that has OpenCV), as scripts/second_scoring.py
# does: to `sh` they are an `exec`, to Python a string.
__doc__ = """The appendix's picture of the placement overlay, rendered afterwards from a recorded trial. No camera, no arm.

    scripts/make_overlay_example.py            # writes paper/figures/fig5-overlay-example.png

What it shows is what the person placing the cube saw: the front camera's picture with the grid, the ring and the
cube's outline at the target, and beside it the surroundings of the target magnified (`placement_phase.render`, the
session's own code). Two rows:

  top     the cube 25 mm from a target, the overlay guiding it. The target is one chosen for this picture, 20 mm and
          15 mm from where the cube lies: no trial asked for it.
  bottom  the same frame against the trial's own planned target, as it was accepted.

The frame is the first of a counted trial's front video (the cube lies where the trial began), and the camera's view
is solved from the frame its session started on, with that frame as the empty scene, as the session did. The first
trial of the log is taken unless another is named. The videos are not in git: this runs where the trials were run.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: How far from the cube the top row's target lies, in table millimetres: a 3-4-5 triangle, 25 mm in all.
SHOWN_OFFSET_MM = (20.0, 15.0)


def main(argv: list[str] | None = None) -> int:
    import cv2
    import numpy as np

    from tolerance_radius import config
    from tolerance_radius.geometry import Point
    from tolerance_radius.rig import placement_phase
    from tolerance_radius.rig.vision import TableView, load_geometry

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=ROOT / "data" / "trials" / "main.jsonl")
    parser.add_argument("--trial", default=None, help="a trial's id (default: the first record of the log)")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "figures" / "fig5-overlay-example.png")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record.get("data", record) for record in records]
    wanted = [r for r in records if args.trial in (None, r["planned"]["trial_id"])]
    if not wanted:
        raise SystemExit(f"no trial {args.trial} in {args.log}")
    record = wanted[0]
    video = ROOT / record["media"]["front"]
    session = video.parent.name
    starts = sorted((ROOT / "data" / "instrument").glob(f"session_{session[:8]}_*.png"))
    # the frame the session started on is the first one kept at or after the session's own stamp
    start = next((p for p in starts if p.stem.split("_", 1)[1] >= session), None)
    if start is None or not video.is_file():
        raise SystemExit(f"this needs {video} and the start frame of session {session} under data/instrument/")

    geometry = load_geometry()
    empty = cv2.imread(str(start))
    view = TableView.from_frames([empty], geometry).with_background(empty).with_cube_region(geometry["grid"])
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    measurement = view.measure(frame) if ok else None
    if measurement is None:
        raise SystemExit(f"no cube is found in the first frame of {video}")

    tolerance = float(config.load_protocol()["placement"]["tolerance_mm"])
    planned = Point(record["planned"]["target"]["x_mm"], record["planned"]["target"]["y_mm"])
    shown = Point(measurement.position.x_mm + SHOWN_OFFSET_MM[0], measurement.position.y_mm + SHOWN_OFFSET_MM[1])
    rows = []
    for target, stable in ((shown, False), (planned, True)):
        assessed = placement_phase.assess(measurement, target, tolerance)
        rows.append(placement_phase.render(frame, view, target, assessed, tolerance, stable, grid=geometry["grid"]))
        print(f"target ({target.x_mm:.1f}, {target.y_mm:.1f}) mm: the cube lies {assessed.error_mm:.1f} mm from it")
    picture = np.vstack([rows[0], np.full((6, rows[0].shape[1], 3), 255, np.uint8), rows[1]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), picture)
    print(f"trial {record['planned']['trial_id']}, frame 0 of {video.relative_to(ROOT)}, view from "
          f"{start.relative_to(ROOT)}; wrote {args.out.relative_to(ROOT)} ({picture.shape[1]} x {picture.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
