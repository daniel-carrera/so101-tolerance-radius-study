#!/usr/bin/env python3
"""Write the analysis of the counted trials: the results file, the report, and the paper's number macros.

    python scripts/make_report.py            # the counted report; needs a frozen protocol
    python scripts/make_report.py --draft    # whatever can be read, into analysis/report-draft/; nothing under paper/

It reads `config/protocol.toml`, every `data/trials/*.jsonl` that is not a pilot log, `data/audit/` (the start-state
audits and the scene audit of each log, `scene_<log>.json`), `data/instrument/`, `data/recording/`,
`docs/decisions.md`, `analysis/simulation/` and the agreement of the second scoring from video
(`data/second_scoring/main/agreement.json`), and writes

    analysis/report/results.json     everything the paper's numbers and figures are made from
    analysis/report/report.md        the same for a person, each number with where it came from
    paper/generated/numbers.tex      the paper's number macros (and MACROS.md, protocol.tex), through
                                     paper/make_numbers_stub.py

A result the data do not support is `null` in the results and stays a visible `??` in the paper. An input that is
refused (a pilot log, a trial that was not run under the frozen protocol, a distance that does not match the audit,
trials of one condition run with different settings, a scene audit of another log, an agreement of the second
scoring that does not add up) ends the command with one sentence and nothing written.

With `--draft` it also accepts trials that were not run under a frozen protocol, marks every output as a draft and
writes nothing under `paper/`. The same inputs always give the same bytes.

The analysis itself is `tolerance_radius.report`; `protocol/protocol.md`, section 10, fixes it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tolerance_radius import config, report

#: The script that owns the list of the paper's number macros and writes `paper/generated/`.
STUB = Path(__file__).resolve().parents[1] / "paper" / "make_numbers_stub.py"


def load_stub(path: Path = STUB) -> ModuleType:
    """`paper/make_numbers_stub.py` as a module: it is a script, not part of the package."""
    spec = importlib.util.spec_from_file_location("make_numbers_stub", path)
    if spec is None or spec.loader is None:
        raise report.ReportError(f"{path.name} cannot be loaded, and it writes the paper's number macros")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their own module up by name
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the analysis: results.json, report.md and the paper's number macros."
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="accept trials that were not run under a frozen protocol, write to analysis/report-draft/ "
        "and write nothing under paper/",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=config.REPO_ROOT,
        help="the folder holding config/, data/, docs/, analysis/ and paper/ (default: this repository)",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        nargs="+",
        default=None,
        help="trial logs to read (default: every data/trials/*.jsonl that is not a pilot log)",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="the start-state audits and the logs' scene audits, scene_<log>.json (default: data/audit)",
    )
    parser.add_argument("--instrument-dir", type=Path, default=None, help="default: data/instrument")
    parser.add_argument("--recording-dir", type=Path, default=None, help="default: data/recording")
    parser.add_argument(
        "--second-scoring-dir",
        type=Path,
        default=None,
        help="the folder of the second scoring from video, which holds agreement.json "
        "(default: data/second_scoring/main)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="with --draft: another folder than analysis/report-draft"
    )
    parser.add_argument(
        "--note", default=None, help="with --draft: a line that says what this output is, put at the top of every file"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    def shown(path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.name

    try:
        if not args.draft and (args.out is not None or args.note is not None):
            raise report.ReportError("--out and --note belong to a draft: the counted report has one place and no note")
        out = (
            root / "analysis" / ("report-draft" if args.draft else "report") if args.out is None else args.out.resolve()
        )
        if args.draft and (out == root / "analysis" / "report" or root / "paper" in (out, *out.parents)):
            raise report.ReportError(
                f"a draft is not written to {shown(out)}: that place is kept for the counted report"
            )
        inputs = report.load_inputs(
            root,
            draft=args.draft,
            trials=args.trials,
            audit_dir=args.audit_dir,
            instrument_dir=args.instrument_dir,
            recording_dir=args.recording_dir,
            second_scoring_dir=args.second_scoring_dir,
            note=args.note,
        )
        made = report.make_report(inputs)
        stub = load_stub()
        try:
            stub.check_lists()
            stub.checked_values(made.values)
        except ValueError as problem:
            raise report.ReportError(
                f"paper/make_numbers_stub.py does not take the values of this analysis: {problem}"
            ) from problem
    except report.ReportError as refusal:
        print(f"make_report: refused: {refusal}.", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    written = {out / "results.json": made.results_json, out / "report.md": made.report_md}
    for path, text in written.items():
        path.write_text(text, encoding="utf-8")
    paths = list(written)
    if not args.draft:
        paths += stub.write_generated(
            made.values,
            out_dir=root / "paper" / "generated",
            protocol_toml=root / "config" / "protocol.toml",
            by_hand_tex=root / "paper" / "numbers_by_hand.tex",
        )
    for path in paths:
        print(f"wrote {shown(path)}")
    total = len(stub.result_names())
    source = made.results["provenance"]
    print(f"trials: {source['n_scored']} scored and {source['n_void']} void, from {len(source['logs'])} log(s)")
    print(
        f"number macros: {len(made.values)} of {total} have a value; the others stay ??"
        + (" (a draft: nothing was written under paper/)" if args.draft else "")
    )
    missing = len(made.results["not_computed"])
    print(f"could not be computed: {missing} thing(s), listed at the end of {shown(out / 'report.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
