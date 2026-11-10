#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """The rate the recording loop held in every saved episode of a dataset, from the sessions' logs. No hardware.

    scripts/rig/recording_rate.py clustered
    scripts/rig/recording_rate.py spread --no-write

An episode's frames carry nominal times, so a loop that ran slow leaves no trace in the dataset (in August 16 of 55
episodes were recorded at about 28 Hz and stamped 30), and since the operator ends a take with a key the frame count
shows nothing either. The session's log has one line per control cycle with the clock's time: this counts them, take
by take, for the take that became each saved episode (named by data/recording/<dataset>_placements.jsonl).

Writes data/audit/<dataset>_rate.json. With the dataset on disk (TOLRAD_NAMESPACE or --namespace) each take's cycles
are also held against its episode's frames. Exit 0 when every saved take held the rate, 4 when something needs eyes:
a take more than 2 % under the rate asked for, a take the logs do not hold, or cycles that are not the episode's frames.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tolerance_radius import recording_rate, schema  # noqa: E402
from tolerance_radius.config import REPO_ROOT  # noqa: E402


def episode_lengths(root: Path) -> tuple[dict[int, int], float] | None:
    """Every episode's frame count and the rate the dataset says it was recorded at; None when it cannot be read."""
    try:
        import pandas as pd

        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        # use_threads=False as in the replay's reader: with the reader's thread pool a process in the rig's environment
        # that has read parquet files failed to exit about half the time (measured 2026-10-05).
        tables = [pd.read_parquet(path, columns=["episode_index", "length"], use_threads=False)
                  for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))]
        episodes = pd.concat(tables, ignore_index=True)
        return ({int(i): int(n) for i, n in zip(episodes["episode_index"], episodes["length"], strict=True)},
                float(info["fps"]))
    except (ImportError, OSError, KeyError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=schema.DATASETS)
    parser.add_argument("--namespace", default=os.environ.get("TOLRAD_NAMESPACE"),
                        help="the datasets' owner on disk (or set TOLRAD_NAMESPACE); without it the frame counts "
                             "are not compared")
    parser.add_argument("--name", default="tolrad-{dataset}-v1")
    parser.add_argument("--cache", default="~/.cache/huggingface/lerobot", help="where the recorder puts datasets")
    parser.add_argument("--logs", default=str(REPO_ROOT / "data" / "logs"), help="the sessions' logs")
    parser.add_argument("--fps", type=float, default=30.0, help="the rate asked for, when the dataset is not read")
    parser.add_argument("--no-write", action="store_true", help="print only (a dataset still being recorded)")
    args = parser.parse_args()

    placements = recording_rate.read_placements(REPO_ROOT / "data" / "recording" / f"{args.dataset}_placements.jsonl")
    if not placements:
        print(f"no placement log for the {args.dataset} set: nothing recorded yet?", file=sys.stderr)
        return 1
    rows = recording_rate.saved_takes(placements, Path(args.logs))

    lengths, fps = None, args.fps
    if args.namespace:
        read = episode_lengths(Path(args.cache).expanduser() / args.namespace / args.name.format(dataset=args.dataset))
        if read is None:
            print("  (the dataset could not be read: frame counts not compared)")
        else:
            lengths, fps = read
    told = recording_rate.summary(rows, fps, lengths)

    print(f"  the {args.dataset} set: {told['episodes']} saved episodes, the loop asked for {fps:g} Hz\n")
    for row in rows:
        frames = "" if lengths is None else f"   frames in the dataset {lengths.get(row['episode_index'], '?')}"
        if row["hz"] is None:
            print(f"  episode {row['episode_index']:2d}  session {row['session']}  NOT READ from the log{frames}")
        else:
            again = f"   ({row['takes_in_session']} takes in that session)" if row["takes_in_session"] > 1 else ""
            print(f"  episode {row['episode_index']:2d}  session {row['session']}  {row['cycles']:5d} cycles in "
                  f"{row['seconds']:7.3f} s  {row['hz']:6.2f} Hz{frames}{again}")
    if told["median_hz"] is not None:
        print(f"\n  median {told['median_hz']:.2f} Hz, lowest {told['lowest_hz']:.2f} Hz "
              f"(short of the rate: under {told['short_of']:.1f})")

    problems = []
    if told["fell_short"]:
        problems.append(f"episodes {told['fell_short']} were recorded more than 2 % under {fps:g} Hz")
    if told["not_read"]:
        problems.append(f"episodes {told['not_read']} have no take in their session's log (is the log there?)")
    if told.get("frames_differ"):
        problems.append(f"episodes {told['frames_differ']}: the log's cycles are not the dataset's frames")
    for problem in problems:
        print(f"  NEEDS EYES: {problem}")
    if not args.no_write:
        out = REPO_ROOT / "data" / "audit" / f"{args.dataset}_rate.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dataset": args.dataset, **told, "episodes_read": rows}, indent=2) + "\n",
                       encoding="utf-8")
        print(f"  -> {out}")
    print("\n  " + ("RATE HELD" if not problems else "NEEDS EYES"))
    return 0 if not problems else 4


if __name__ == "__main__":
    raise SystemExit(main())
