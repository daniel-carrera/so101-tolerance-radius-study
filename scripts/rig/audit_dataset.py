#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Measure where the cube started in every episode of a recorded dataset, and compare with the plan. No hardware.

    scripts/rig/audit_dataset.py clustered --namespace NAME
    scripts/rig/audit_dataset.py spread --namespace NAME

Writes data/audit/<dataset>.json (the measured starts every later distance is taken to), a contact sheet picture of
every detection for a person to look through, and data/audit/<dataset>_camera.json: for every episode, how far the
camera's picture stood from the stored calibration's and how bright the picture was; what the front camera's picture
did inside every take (one frame a second: a camera that changes its zoom or its tone range in the middle of a take
leaves the take's first frame as it should be); and the wrist camera's first frame of every episode, which has no
check of its own. Every start is measured through the floor tags as its own first frame shows them. Exit 0 when the
dataset is as planned, 4 when something needs eyes. `--out FOLDER` writes elsewhere (a set still being recorded).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius import layouts, schema  # noqa: E402
from tolerance_radius.config import REPO_ROOT, load_protocol  # noqa: E402
from tolerance_radius.geometry import Grid  # noqa: E402
from tolerance_radius.rig import lerobot101  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=schema.DATASETS)
    parser.add_argument("--namespace", default=os.environ.get("TOLRAD_NAMESPACE"),
                        help="the datasets' owner on disk (or set TOLRAD_NAMESPACE)")
    parser.add_argument("--name", default="tolrad-{dataset}-v1")
    parser.add_argument("--limit-mm", type=float, default=None, help="largest allowed distance from a target")
    parser.add_argument("--out", default=None,
                        help="where the three files are written (default: data/audit in this repository). Give "
                             "another folder to look at a set that is still being recorded")
    args = parser.parse_args()
    if not args.namespace:
        print("no dataset owner: pass --namespace or set TOLRAD_NAMESPACE", file=sys.stderr)
        return 1

    lerobot101.check_pin(allow_other=True)
    lerobot101.add_to_path()
    print(f"  {lerobot101.use_pinned_lerobot()}")
    from tolerance_radius.rig import audit  # noqa: E402 -- after the pinned LeRobot is on the path
    from tolerance_radius.rig.camera_tool import STORED_VIEW  # noqa: E402
    from tolerance_radius.rig.vision import TableView  # noqa: E402

    protocol = load_protocol()
    planned = layouts.read_layout(REPO_ROOT / "config" / "layouts" / f"{args.dataset}.json")
    view = TableView.load(STORED_VIEW)
    limit = args.limit_mm if args.limit_mm is not None else audit.DEFAULT_LIMIT_MM
    repo_id = f"{args.namespace}/{args.name.format(dataset=args.dataset)}"
    from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402 -- the pinned LeRobot's own

    result, frames = audit.audit_dataset(args.dataset, repo_id, planned, view, Grid.from_protocol(protocol),
                                         limit_mm=limit, root=Path(HF_LEROBOT_HOME) / repo_id)

    out = Path(args.out) if args.out else REPO_ROOT / "data" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    schema.write_demo_starts(out / f"{args.dataset}.json", result.starts)
    sheet = audit.contact_sheet(frames, view, out / f"{args.dataset}_contact_sheet.png")
    print(f"\n  {len(result.starts)} of {len(frames)} episodes measured -> {out / (args.dataset + '.json')}")
    print(f"  distance from target: mean {result.mean_error_mm:.2f} mm, largest {result.max_error_mm:.2f} mm "
          f"(limit {limit:g} mm)")
    for note in result.notes:
        print(f"  {note}")
    if result.camera:
        camera_file = out / f"{args.dataset}_camera.json"
        document = {"dataset": args.dataset, "calibration_created": json.loads(STORED_VIEW.read_text())["created"],
                    "tag_noise_px": result.tag_noise_px,
                    "episodes": {str(index): entry for index, entry in sorted(result.camera.items())},
                    # inside every take: frames looked at, how far the floor tags lay from the take's first frame
                    # and the range of their black as a share of the first frame's
                    "takes": {str(index): entry for index, entry in sorted((result.takes or {}).items())},
                    # the wrist camera's first frame of every episode
                    "wrist": {str(index): entry for index, entry in sorted((result.wrist or {}).items())}}
        camera_file.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        greys = [entry["grey"] for entry in result.camera.values() if "grey" in entry]
        if greys:
            print(f"  the floor inside the grid read grey level {min(greys):.0f} to {max(greys):.0f} over the episodes "
                  f"(at the calibration {(view.floor_levels or {}).get('grey', float('nan')):.0f})")
        if result.tag_noise_px is not None:
            print(f"  from one episode to the next the tags' outline corners differ by {result.tag_noise_px:.2f} px "
                  "(rms): what one recorded frame's corners are worth")
        if result.takes:
            looked = sum(int(entry["frames"]) for entry in result.takes.values())
            with_tags = sum(int(entry["frames_with_tags"]) for entry in result.takes.values())
            moved = max(entry.get("tags_moved_px", 0.0) for entry in result.takes.values())
            print(f"  inside the takes: {looked} frames looked at (one a second), {with_tags} with both floor tags "
                  f"in sight; the tags at most {moved:.1f} px from where their take's first frame has them")
        if result.wrist:
            read = [entry["contrast"] for entry in result.wrist.values() if entry.get("contrast") is not None]
            print("  the wrist camera's first frames: the floor over the jaws' green reads "
                  + (f"{min(read):.2f} to {max(read):.2f} ({audit.wrist_reads()})"
                     if read else "nothing that could be judged"))
        print(f"  the camera and the picture, episode by episode -> {camera_file}")
    print(f"  contact sheet -> {sheet}\n  LOOK AT EVERY TILE before trusting this table.")
    for problem in result.problems:
        print(f"  PROBLEM: {problem}")
    print("  CLEAN: every episode is where the plan put it" if result.clean else f"  {len(result.problems)} problem(s)")
    return 0 if result.clean else 4


if __name__ == "__main__":
    from tolerance_radius.rig.leave import leave

    # In the rig's environment a process that has read a dataset often never exits by itself; everything is written.
    leave(main())
