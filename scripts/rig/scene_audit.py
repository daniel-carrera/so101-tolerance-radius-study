#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell (see scripts/rig/run_trials.py).
__doc__ = """Read the scene at the start and end of every trial of a log from the trials' own videos. No camera, no arm.

    scripts/rig/scene_audit.py --log data/trials/main.jsonl

For each trial: where the basket stood against its reference frame at the first frames and at the last (did the arm
carry it off?), and where the wrist camera stood in its mount (the gripper's fixed jaw in its picture). Writes the
readings beside the audits (data/audit/scene_<log's name>.json) and prints a table. Protocol, section 8.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tolerance_radius.rig import lerobot101, scene_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="a trial log, for example data/trials/main.jsonl")
    parser.add_argument("--root", default=str(REPO), help="the folder the log's video paths start from")
    parser.add_argument("--out", default=None, help="where to write the readings; default data/audit/scene_<log>.json")
    parser.add_argument("--no-write", action="store_true", help="print the table only")
    args = parser.parse_args()
    log, root = Path(args.log), Path(args.root)
    geometry_file = root / "config" / "table_geometry.json"
    geometry = json.loads(geometry_file.read_text(encoding="utf-8")) if geometry_file.is_file() else None
    rows = scene_audit.audit_log(log, root, root / "data" / "instrument" / "basket_reference.png",
                                 root / "data" / "instrument" / "wrist_reference.png",
                                 lerobot101.load("tag_check").detect, geometry)
    for line in scene_audit.told(rows):
        print(line)
    if not args.no_write:
        out = Path(args.out) if args.out else root / "data" / "audit" / f"scene_{log.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"log": log.name, "basket_named_mm": scene_audit.BASKET_NAMED_MM,
                                   "wrist_named_px": scene_audit.WRIST_NAMED_PX, "trials": rows}, indent=1) + "\n",
                       encoding="utf-8")
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
