#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Put a recorded set that a session left without closing back to its last sound state. Deletes nothing.

    scripts/rig/repair_set.py clustered             # says what it found and what it would do; changes nothing
    scripts/rig/repair_set.py clustered --apply     # does it

For when the recorder refuses with "the dataset at ... does not add up", and after any session that did not end by
itself (the terminal closed, the process killed, Ctrl-C). The episodes that were closed properly are kept as they
are. What the session left that belongs to no closed episode (table files with no end, its video files, the pictures
of an unsaved take) is moved into a folder beside the set, tolrad-clustered-v1-unclosed-<time>; meta/info.json and
meta/stats.json are written again for the sound episodes; the placement log is cut back to them, the cut lines kept
in that folder; and a data file that holds the rows of a take whose save was cut short is written again without
them, the file as it was kept in that folder. Then --resume goes on at the next episode.

It refuses, changing nothing, when it cannot tell what is sound and while a recorder or a trial session is running;
on a set that adds up it says so and changes nothing. Run it from the checkout the set was recorded from (the
placement log is that checkout's data/recording/<dataset>_placements.jsonl). No hardware, no camera.

Exit 0: the set adds up (already, or after --apply). 1: refused. 2: a repair is needed and was only described.
3: a repair was begun and the set still does not add up (say what it prints).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius import schema  # noqa: E402
from tolerance_radius.rig import lerobot101, record_layout  # noqa: E402
from tolerance_radius.rig.leave import leave  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=schema.DATASETS)
    parser.add_argument("--apply", action="store_true", help="do it; without this, only say what would be done")
    parser.add_argument("--namespace", default=os.environ.get("TOLRAD_NAMESPACE"),
                        help="the datasets' owner on disk (or set TOLRAD_NAMESPACE)")
    parser.add_argument("--name", default=record_layout.DEFAULT_NAME)
    parser.add_argument("--placement-log", default=None, metavar="FILE",
                        help="the set's placement log, when it is not this checkout's own (a copy of a set)")
    args = parser.parse_args()
    if not args.namespace:
        print("no dataset owner: pass --namespace or set TOLRAD_NAMESPACE", file=sys.stderr)
        return 1
    try:
        lerobot101.check_pin(allow_other=True)
        lerobot101.add_to_path()
        lerobot101.use_pinned_lerobot()
    except lerobot101.ToolingError as error:
        print(f"  REFUSING -- {error}\n  Nothing was changed.", file=sys.stderr)
        return 1
    from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402 -- the pinned LeRobot's own

    from tolerance_radius.rig import repair_set  # noqa: E402

    name = args.name.format(dataset=args.dataset)
    dataset_dir = Path(HF_LEROBOT_HOME) / args.namespace / name
    log = Path(args.placement_log) if args.placement_log else record_layout.placement_log_path(args.dataset, name)
    report = repair_set.repair(dataset_dir, log, apply=args.apply, stamp=time.strftime("%Y%m%d_%H%M%S"))
    print()
    for line in report.lines:
        print(line)
    return report.code


if __name__ == "__main__":
    # A process that has read the set's tables may not leave by itself (rig/leave.py).
    leave(main())
