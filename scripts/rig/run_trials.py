#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Run planned evaluation trials. THE ARM MOVES (a policy or a replay, through the gates). See `--help`.

    scripts/rig/run_trials.py --blocks T1b --log data/trials/pilot.jsonl --replay-root DIR --dry-run --limit 2
    scripts/rig/run_trials.py --blocks T1a T1b --log data/trials/main.jsonl \
        --policy act_clustered=DIR/080000/pretrained_model --replay-root DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius.rig.leave import leave  # noqa: E402
from tolerance_radius.rig.trial_session import cli  # noqa: E402

if __name__ == "__main__":
    # As the arm tooling's own launchers do: leave without running exit handlers, which can hang on the viewer.
    leave(cli())
