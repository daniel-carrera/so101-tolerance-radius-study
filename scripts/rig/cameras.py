#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Which cameras the computer offers right now, and whether they are the two the sessions take. Opens no camera.

    scripts/rig/cameras.py

The sessions take camera 0 for the front camera and camera 1 for the wrist camera. This prints the system's list in
that order, by name. Exit 0 when both cameras of the study are there in their places, 4 when one is missing or
another camera sits at its number.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius.rig import camera_list  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front-index", type=int, default=0)
    parser.add_argument("--wrist-index", type=int, default=1)
    args = parser.parse_args()
    try:
        cameras = camera_list.system_cameras()
    except camera_list.CameraListError as error:
        print(f"  {error}", file=sys.stderr)
        return 1
    for line in camera_list.lines(cameras, args.front_index, args.wrist_index):
        print(f"  {line}")
    certain, likely = camera_list.problems(cameras, args.front_index, args.wrist_index)
    for problem in certain + likely:
        print(f"  NOT RIGHT: {problem}")
    print("\n  " + ("CAMERAS IN PLACE" if not (certain or likely) else "CAMERAS NOT IN PLACE"))
    return 0 if not (certain or likely) else 4


if __name__ == "__main__":
    raise SystemExit(main())
