"""The analysis: from the trial logs to one results file, a readable report and the paper's number macros.

`protocol/protocol.md`, section 10, fixes the analysis in advance; this package is that section as code.
`scripts/make_report.py` is the command.

* `inputs`: the files that are read, and everything that is refused (pilot logs, records that were not run under a
  frozen protocol, distances that do not match the audits, trials of a condition that were not all run the same way
  or not as the protocol says, policy trials of more than one training step, a scene audit of another log, an
  agreement of the second scoring that does not add up);
* `estimates`: the statistics, as pure functions of numbers;
* `bookkeeping`: the instrument check, the datasets, how the trials were run, the scene at each trial's start, the
  second scoring from video, what is quoted of the simulations;
* `build`: the results, in the shape `protocol/data-contracts.md` fixes for `analysis/report/results.json`;
* `macros`: which result each number macro of the paper prints, and how it is written;
* `markdown`: the report for a person.

One run reads the inputs once and writes everything from the same results, so a number in a figure, in the report
and in the paper cannot come from different analyses. The same inputs give the same bytes: nothing here reads a
clock or draws a random number.

Standard library and NumPy only. Nothing here imports `tolerance_radius.rig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tolerance_radius.report.build import build_results, dump_json
from tolerance_radius.report.inputs import Inputs, ReportError, load_inputs
from tolerance_radius.report.macros import macro_values
from tolerance_radius.report.markdown import render_report

__all__ = [
    "Inputs",
    "Report",
    "ReportError",
    "build_results",
    "dump_json",
    "load_inputs",
    "macro_values",
    "make_report",
    "render_report",
]


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one run of the analysis writes."""

    #: The results as plain data.
    results: dict[str, Any]
    #: The text of `results.json` and of `report.md`.
    results_json: str
    report_md: str
    #: The values of the paper's number macros, as `paper/make_numbers_stub.py` takes them.
    values: dict[str, dict[str, str]]


def make_report(inputs: Inputs) -> Report:
    """Analyse what was read and render every output from the one set of results."""
    results = build_results(inputs)
    values = macro_values(results)
    return Report(
        results=results, results_json=dump_json(results), report_md=render_report(results, values), values=values
    )
