#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Record one sitting of a planned demonstration layout. THE ARM MOVES (teleoperation). See `--help`.

    scripts/rig/record_layout.py --sitting 1 --plan-only          # print what sitting 1 records, touch nothing
    scripts/rig/record_layout.py --sitting 1 --namespace NAME      # first sitting of a dataset
    scripts/rig/record_layout.py --sitting 3 --namespace NAME --resume
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius.rig.leave import leave  # noqa: E402
from tolerance_radius.rig.record_layout import cli  # noqa: E402

if __name__ == "__main__":
    # As the arm tooling's own launchers do: leave without running exit handlers, which can hang on the viewer.
    leave(cli())
