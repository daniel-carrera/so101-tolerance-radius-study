#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Set a recorded set and its placement log aside, so that the set can be recorded anew. Deletes nothing.

    scripts/rig/set_aside.py clustered --as attempt2

The set's folder is renamed tolrad-clustered-v1-attempt2 and its log data/recording/
tolrad-clustered-v1-attempt2_placements.jsonl. The recorder's next run of that set's first sitting, without
--resume, then begins at episode 0. Run it with no session running. No hardware, no camera.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius import schema  # noqa: E402
from tolerance_radius.rig import lerobot101, record_layout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=schema.DATASETS)
    parser.add_argument("--as", dest="label", required=True, metavar="LABEL",
                        help="what to call it: the folder and the log get this after their name (attempt2)")
    parser.add_argument("--namespace", default=os.environ.get("TOLRAD_NAMESPACE"),
                        help="the datasets' owner on disk (or set TOLRAD_NAMESPACE)")
    parser.add_argument("--name", default=record_layout.DEFAULT_NAME)
    parser.add_argument("--log-only", action="store_true",
                        help="set aside a placement log whose set is no longer there (without this, a missing "
                             "set is taken for a mistyped owner name and nothing is renamed)")
    args = parser.parse_args()
    if not args.namespace:
        print("no dataset owner: pass --namespace or set TOLRAD_NAMESPACE", file=sys.stderr)
        return 1
    try:
        lerobot101.check_pin(allow_other=True)
        lerobot101.add_to_path()
        lerobot101.use_pinned_lerobot()
    except lerobot101.ToolingError as error:
        print(f"  NOTHING WAS RENAMED -- {error}", file=sys.stderr)
        return 1
    from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402 -- the pinned LeRobot's own

    name = args.name.format(dataset=args.dataset)
    dataset_dir = Path(HF_LEROBOT_HOME) / args.namespace / name
    log = record_layout.placement_log_path(args.dataset, name)
    held = record_layout.dataset_counts(dataset_dir)["said"] if dataset_dir.exists() else None
    lines = record_layout.placement_lines(log)
    try:
        moved = record_layout.set_aside(dataset_dir, log, args.label, log_only=args.log_only)
    except (ValueError, OSError) as error:
        print(f"  NOTHING WAS RENAMED -- {error}", file=sys.stderr)
        return 1
    print("\n  Set aside, nothing deleted:")
    for old, new in moved:
        what = (f"{held} episodes" if old == dataset_dir and held is not None
                else f"{lines} lines" if old == log else "")
        print(f"    {old}\n      -> {new.name}" + (f"   ({what})" if what else ""))
    print(f"\n  The {args.dataset} set is recorded anew from episode 0 by its first sitting, without --resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
