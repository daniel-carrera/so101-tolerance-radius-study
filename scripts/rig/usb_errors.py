#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """How many USB transfer errors each camera's link gave in the last minutes. Opens no camera.

    scripts/rig/usb_errors.py              # the last 10 minutes
    scripts/rig/usb_errors.py --minutes 3

Run it after a camera has been streaming for a few minutes (a session's warm-up will do). A good link has no error
at all. Each error is a video frame that was dropped, and now and then the stream comes out of one in another state
(2026-10-06: the camera on its own zoom and exposure in the middle of a take, the tone range stretched). Exit 0 when
neither camera had an error, 4 when one had.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius import usb_errors  # noqa: E402
from tolerance_radius.rig.camera_list import FRONT_NAME, WRIST_NAME  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=int, default=10)
    args = parser.parse_args()
    try:
        lines = usb_errors.system_log(args.minutes)
    except usb_errors.LogError as error:
        print(f"  {error}", file=sys.stderr)
        return 1
    names = {FRONT_NAME: "front camera", WRIST_NAME: "wrist camera"}
    counts = usb_errors.per_minute(lines, names)
    where = usb_errors.ports(lines, names)
    print(f"  USB transfer errors in the last {args.minutes} minutes (a camera that was not streaming has none):\n")
    bad = False
    for name, role in names.items():
        total = sum(counts[name].values())
        bad = bad or total > 0
        if not total:
            print(f"  {role} ({name}): none")
            continue
        minutes = sorted(counts[name])
        worst = max(counts[name].values())
        print(f"  {role} ({name}): {total} in {len(minutes)} of those minutes, {minutes[0]} to {minutes[-1]}, up to "
              f"{worst} in one minute (port {', '.join(sorted(where[name]))})")
    print("\n  " + ("NO ERRORS" if not bad else "ERRORS ON A CAMERA'S LINK: frames are being dropped there"))
    return 0 if not bad else 4


if __name__ == "__main__":
    raise SystemExit(main())
