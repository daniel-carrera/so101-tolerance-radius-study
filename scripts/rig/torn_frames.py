#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Find torn camera frames in a recorded dataset: pictures damaged in a band of their rows. No hardware.

    scripts/rig/torn_frames.py clustered --namespace NAME
    scripts/rig/torn_frames.py spread --save-frames FOLDER

Every frame of both cameras is held against the frames around it (`tolerance_radius.rig.torn_frames` says how, and
what was measured). Every torn frame is named by episode, camera, frame within the episode and time in seconds;
`--save-frames FOLDER` writes each with the frame before and after it as one picture, to be looked at before
anything is done about it. Exit 0 with the line NO TORN FRAME when there is none, 4 when there are some.

It also counts, as a measurement and no verdict, the frames whose picture came twice while the arm was moving: what a
frame dropped on the camera's link looks like in the recording. `--no-repeats` leaves that out.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius import schema  # noqa: E402
from tolerance_radius.rig import torn_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=schema.DATASETS)
    parser.add_argument("--namespace", default=os.environ.get("TOLRAD_NAMESPACE"),
                        help="the datasets' owner on disk (or set TOLRAD_NAMESPACE)")
    parser.add_argument("--name", default="tolrad-{dataset}-v1")
    parser.add_argument("--cache", default="~/.cache/huggingface/lerobot", help="where the recorder puts datasets")
    parser.add_argument("--save-frames", default=None, metavar="FOLDER",
                        help="write every torn frame, with the frame before and after it, as a picture into FOLDER")
    parser.add_argument("--no-repeats", action="store_true", help="do not count the frames that came twice")
    args = parser.parse_args()
    if not args.namespace:
        print("no dataset owner: pass --namespace or set TOLRAD_NAMESPACE", file=sys.stderr)
        return 1
    root = Path(args.cache).expanduser() / args.namespace / args.name.format(dataset=args.dataset)
    if not (root / "meta" / "info.json").is_file():
        print(f"no dataset at {root}", file=sys.stderr)
        return 1

    report = torn_frames.scan_dataset(root, measure_repeats=not args.no_repeats)
    looked = ", ".join(f"{camera} camera {count} frames" for camera, count in sorted(report.frames.items()))
    print(f"  the {args.dataset} set: {report.episodes} episodes at {report.fps:g} frames a second; {looked} looked at")
    for finding in report.findings:
        print(f"  TORN: episode {finding.episode}, {finding.camera} camera, frame {finding.frame} "
              f"({finding.time_s:.3f} s): {finding.tear.words()}")
    for camera, episode, frame in report.not_judged:
        print(f"  not judged: episode {episode}, {camera} camera, frame {frame} (an end of a video file, with "
              "something moving in the frames next to it): look at it")
    if args.save_frames and report.findings:
        written = torn_frames.save_pictures(root, report.findings, Path(args.save_frames))
        print(f"  {len(written)} picture(s) -> {args.save_frames}: LOOK AT EACH before acting on this "
              "(the frame before, the frame, the frame after; a red bar at both edges over the damaged rows)")

    if not args.no_repeats:
        steps = torn_frames.joint_steps(root)
        print("\n  frames whose picture is the one before again although the arm was moving (a measurement, no verdict;"
              "\n  only frames the video codes by themselves, with the picture changing two frames before and after):")
        stands = {}
        for camera in sorted(report.moved):
            judged, twice, stands[camera] = 0, [], []
            for episode, moved in sorted(report.moved[camera].items()):
                stepped = steps.get(episode)
                if stepped is None or len(stepped) != len(moved):
                    continue
                told = torn_frames.repeats(moved, stepped, report.keyframes[camera][episode])
                judged += told["judged"]
                twice += [(episode, frame) for frame in told["repeats"]]
                stands[camera] += [(episode, first, count)
                                   for first, count in torn_frames.stood(report.same[camera][episode], stepped)]
            share = f"{100.0 * len(twice) / judged:.2f} %" if judged else "none judged"
            where = "; ".join(f"episode {episode} frame {frame}" for episode, frame in twice[:8])
            print(f"  {camera} camera: {len(twice)} of {judged} ({share})" + (f": {where}" if where else "")
                  + (" ..." if len(twice) > 8 else ""))
        print("  pictures that stood for three frames or more while the arm moved (the camera gave one picture again "
              "and again):")
        for camera, found in stands.items():
            where = "; ".join(f"episode {episode} frames {first} to {first + count - 1}"
                              for episode, first, count in found)
            print(f"  {camera} camera: {len(found)}" + (f": {where}" if where else ""))

    if not report.findings:
        left = f" ({len(report.not_judged)} frame(s) could not be judged, named above)" if report.not_judged else ""
        print(f"\n  NO TORN FRAME{left}")
        return 0
    episodes = sorted({finding.episode for finding in report.findings})
    print(f"\n  {len(report.findings)} TORN FRAME(S) in {len(episodes)} episode(s): {episodes}")
    return 4


if __name__ == "__main__":
    from tolerance_radius.rig.leave import leave

    # In the rig's environment a process that has read a dataset often never exits by itself; everything is written.
    leave(main())
