#!/usr/bin/env python3
"""Draw the paper's figures from the data files.

    .venv/bin/python scripts/make_figures.py                  every figure whose input is there
    .venv/bin/python scripts/make_figures.py --only 1 2       some of them
    .venv/bin/python scripts/make_figures.py --only 2 --planned
    .venv/bin/python scripts/make_figures.py --only 3 4 --results analysis/report-draft/results.json

Figure 1 is drawn from `analysis/pilot_august/trials_with_distance.csv`, Figure 2 from the start-state audits
(`data/audit/`), Figures 3 and 4 from the analysis results. They are written to `paper/figures/` as PDF, one column
wide, with no metadata.

A figure whose input is not there yet is skipped, with a line that names the file it waits for; that is not an error.
A results file that says it is a draft is never drawn into `paper/figures/`: its figures carry the word DRAFT and go
to `analysis/report-draft/figures/`. `--planned` also draws Figure 2 from the planned start positions, with the word
PLANNED across it, into `analysis/layouts/`.

The command ends with 1 when a figure was refused: its input is there and cannot be used, or the file would hold a
string the paper must not hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tolerance_radius import config  # noqa: E402
from tolerance_radius.figures import make  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the paper's figures from the data files.")
    add = parser.add_argument
    add("--only", type=int, nargs="+", choices=make.NUMBERS, default=list(make.NUMBERS), metavar="N",
        help="the figures to draw, by number (default: all four)")  # fmt: skip
    add("--results", type=Path, default=None,
        help=f"the analysis results, for Figures 3 and 4 (default: {make.RESULTS.as_posix()})")  # fmt: skip
    add("--out", type=Path, default=None,
        help=f"the folder for the paper's figures (default: {make.PAPER_FIGURES.as_posix()})")  # fmt: skip
    add("--draft-out", type=Path, default=None,
        help=f"the folder for figures of draft results (default: {make.DRAFT_FIGURES.as_posix()})")  # fmt: skip
    add("--planned", action="store_true",
        help=f"also draw Figure 2 from the planned positions, into {make.PLANNED_PREVIEW.parent.as_posix()}")
    add("--png", type=Path, default=None, metavar="DIR",
        help="also save each figure written as a picture at 300 dots per inch, for looking at")  # fmt: skip
    add("--forbidden", type=Path, action="append", default=[], metavar="PATH",
        help="one more list of strings no figure may hold (as for paper/check_paper.py)")  # fmt: skip
    add("--root", type=Path, default=config.REPO_ROOT, help="the folder holding analysis/, config/, data/ and paper/")
    args = parser.parse_args(argv)

    made = make.make_figures(
        args.only,
        root=args.root,
        results_path=args.results,
        out_dir=args.out,
        draft_dir=args.draft_out,
        planned=args.planned,
        png_dir=args.png,
        extra_forbidden=args.forbidden,
    )
    for figure in made:
        refused = figure.status == make.REFUSED
        print(("error: " if refused else "") + figure.line, file=sys.stderr if refused else sys.stdout)
        for note in figure.notes:
            print(f"  note: {note}")
    return 1 if any(figure.status == make.REFUSED for figure in made) else 0


if __name__ == "__main__":
    sys.exit(main())
