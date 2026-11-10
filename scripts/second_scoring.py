#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON. The rig's interpreter is the one that has OpenCV, which reads and shows the videos.
__doc__ = """The second scoring from video: draw the items, score them, compare the two scores.

    scripts/second_scoring.py draw --log data/trials/main.jsonl --seed 20261006 --out data/second_scoring/main
    scripts/second_scoring.py score --items data/second_scoring/main/items.csv --scorer sc2
    scripts/second_scoring.py compare --dir data/second_scoring/main --log data/trials/main.jsonl

`draw`: after the last counted trial. Takes one scored trial in five of each condition of each block, by the seed
and the log's own hash, checks that each one's videos can be read to the end, and writes a folder: `items.csv` and
`videos/` for the scorer (item codes and neutral file names, in a shuffled order), and `key/`, which says which trial
each item is and is not opened until every item is scored. It opens no camera: it reads video files.

`score`: shows each item that has no score yet, the views side by side. The space bar starts an item's video; once it
has ended, s success, f failure or c cannot tell, then Enter (the Return key) to record it. Each score is appended to
`scores.csv` at once; x stops, and the same command carries on later. The scorer is named by a role code (`sc2`,
`op1`), never by a name.

`compare`: once every item is scored. Prints in how many items the two scores agree, lists the trials in which they
differ, and writes `agreement.json`.

`docs/runbook/05-second-scoring.md` has the procedure; `tolerance_radius.second_scoring` is the logic.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tolerance_radius import config, second_scoring  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    root_help = "the repository the logs' video names are relative to (default: this one)"

    draw = commands.add_parser("draw", help="draw the trials that are scored again and write the scorer's folder")
    draw.add_argument("--log", action="append", required=True, type=Path, metavar="FILE",
                      help="a counted trial log; give the option once for each log of the experiment")
    draw.add_argument("--seed", type=int, required=True, help="the seed fixed before the first counted trial")
    draw.add_argument("--out", type=Path, required=True, metavar="DIR",
                      help="the folder to write, for example data/second_scoring/main; it must not hold anything")
    draw.add_argument("--views", nargs="+", default=list(second_scoring.VIEWS), metavar="CAMERA",
                      help="the camera views shown side by side, left to right (default: front wrist)")
    draw.add_argument("--rehearsal", action="store_true",
                      help="take pilot trials (and no counted one), to try the tool before the counted trials exist; "
                           "the draw's record and the comparison are marked as a rehearsal")
    draw.add_argument("--root", type=Path, default=config.REPO_ROOT, help=root_help)

    score = commands.add_parser("score", help="show each unscored item and take its score")
    score.add_argument("--items", type=Path, required=True, metavar="FILE", help="the folder's items.csv")
    score.add_argument("--scorer", required=True, metavar="CODE", help="who scores, as a role code such as sc2")

    compare = commands.add_parser("compare", help="compare the two scores, once every item is scored")
    compare.add_argument("--dir", type=Path, required=True, metavar="DIR", help="the folder the draw wrote")
    compare.add_argument("--log", action="append", required=True, type=Path, metavar="FILE",
                         help="the trial logs the draw was made from")
    compare.add_argument("--root", type=Path, default=config.REPO_ROOT, help=root_help)
    return parser


def shown(path: Path, root: Path) -> str:
    """A folder or file as the printed lines name it: relative to the repository, or as it was typed when it lies
    elsewhere (the bare name would not be found again)."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None, *, player=None, screen=None) -> int:
    """Run one sub-command. `player` and `screen` stand in for the video reader and the window in tests."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "draw":
            root = args.root.resolve()
            print("reading the logs, and each drawn trial's videos to their end ...", flush=True)
            record = second_scoring.make_draw(args.log, args.seed, args.out, root=root, views=args.views,
                                              rehearsal=args.rehearsal)
            for line in second_scoring.draw_lines(record, shown(args.out, root)):
                print(line)
        elif args.command == "score":
            second_scoring.check_role_code(args.scorer)
            second_scoring.score_items(args.items, args.scorer,
                                       player=player if player is not None else second_scoring.VideoPlayer(),
                                       screen=screen if screen is not None else second_scoring.Window())
        else:
            root = args.root.resolve()
            agreement = second_scoring.compare(args.dir, args.log, root=root)
            written = second_scoring.write_agreement(args.dir, agreement)
            for line in second_scoring.agreement_lines(agreement):
                print(line)
            print(f"wrote {shown(written, root)}")
    except second_scoring.SecondScoringError as refusal:
        print(f"second_scoring: refused: {refusal}.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        after = {
            "draw": "the draw was cut short: if it left a folder, remove it by hand and draw again.",
            "score": "scores recorded so far are kept; the same command carries on.",
            "compare": "nothing was compared.",
        }[args.command]
        print(f"\nstopped: {after}", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
