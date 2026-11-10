#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Camera-only tools: check the view, log measurements, rehearse placements. See `--help`.

    scripts/rig/measure.py check
    scripts/rig/measure.py log --label ruler_x
    scripts/rig/measure.py place 75,25 125,75
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius.rig.camera_tool import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
