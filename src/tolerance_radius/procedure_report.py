"""The report of the procedure simulation: tables, a plain reading of them, and a recommendation.

`simulate` produces rows of numbers: how each block design estimates the tolerance radius for each assumed truth, what
a floor in the model does, and how large a difference the paired test can show. This module turns those rows into
`procedure.md`, the report the protocol is checked against before it is frozen.

The reading and the recommendation are not written by hand. They follow from the numbers by fixed rules, each stated
in the report where it is used, so that a run with another seed, another replication count or a changed protocol
gives a report that still agrees with its own tables. The rules:

* a design is better or worse than the design as written when its RMSE, pooled over the truths, differs by more than
  `MEANINGFUL_CHANGE`; to count as better it must also not widen the interval, open it or lose the truth more often;
* a floor in the model is recommended when ignoring a true floor is worse by that margin and allowing for an absent
  one widens the interval by at most `ACCEPTABLE_FLOOR_COST`;
* an interval is said to fall short of its level only where it does so by more than three standard errors of the
  simulation itself.

Every number in a report is simulated or computed under stated assumptions, and every table says which.

Standard library and NumPy only.
"""

from __future__ import annotations

import math
import statistics
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tolerance_radius.psychometric import EDGE_RATIO, grids_from_protocol
from tolerance_radius.simulate import (
    FINE_STEP_FINAL_MM,
    FIT_FLOOR,
    FIT_NO_FLOOR,
    FLOOR_CHECK_FAR_ANCHORS,
    FLOOR_CHECK_FLOOR_VALUES,
    POWER_TARGET,
    STAIRCASE_TRIALS_DELTA,
    WIDER_RANGE,
    BlockDesign,
    DesignPerformance,
    FloorCheckRow,
    PowerRow,
    SweepRow,
    Truth,
    blocks_from_protocol,
    rate_needed,
    steps_from_protocol,
)

#: A choice among the truths of a run: a function that says whether a truth is wanted.
_Keep = Callable[[Truth], bool]
#: A row of one design next to the row of the design as written for the same block and truth.
_Pair = tuple[DesignPerformance, DesignPerformance]

#: One design counts as better or worse than another when its error, pooled over the truths, differs by more than
#: this share. A smaller difference is not a reason to change a protocol.
MEANINGFUL_CHANGE = 0.05
#: A floor in the model counts as cheap when, for truths without a floor, it widens the interval by at most this share.
ACCEPTABLE_FLOOR_COST = 0.10
#: A curve's fall counts as inside a range when the range reaches two widths either side of r50: from 88 % of the way
#: up to 12 %.
FALL_HALF_SPAN_WIDTHS = 2.0

_DESIGN_NAMES = {
    "protocol": "as written",
    "constant": "fixed distances",
    "fine_step": "final step 2.5 mm",
    "longer": "staircases 4 longer",
    "shorter": "staircases 4 shorter",
    WIDER_RANGE: "larger range",
}
_VARIANT_NAMES = {
    "as_written": "as written",
    "far_anchors": f"with {FLOOR_CHECK_FAR_ANCHORS} far anchors added",
    "longer_staircases": f"with {FLOOR_CHECK_FAR_ANCHORS} staircase trials added",
}
_FIT_NAMES = {FIT_NO_FLOOR: "without a floor", FIT_FLOOR: "with floor values"}
_FALL_NAMES = {
    "inside": "inside the tested range",
    "early": "already under way at 0 mm (`r50` under two widths)",
    "late": "running past the largest distance (`r50` within two widths of it)",
    "both": "wider than the tested range",
}


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What a report says about the run that produced it."""

    seed: int
    replications: int
    placement_sd_mm: float
    #: The protocol file's name, not its path: the report is committed and names no place on any machine.
    protocol_file: str
    #: The first digits of the SHA-256 of the protocol file.
    protocol_digest: str


def render_report(
    protocol: dict[str, Any],
    rows: Sequence[SweepRow],
    floor_rows: Sequence[FloorCheckRow],
    power_rows: Sequence[PowerRow],
    info: RunInfo,
) -> str:
    """The report of a run, as Markdown: the tables, a plain reading of them, and a recommendation.

    The reading and the recommendation are written from the numbers by fixed rules, each stated where it is used, so
    that another seed or another protocol gives a report that still agrees with its own tables.
    """
    results = _Results(protocol, rows, floor_rows, power_rows)
    sections = [
        _head(results, info),
        _what_was_simulated(results, info),
        _terms(),
        _as_written(results),
        _where_it_fails(results),
        _other_placements(results),
        _what_trials_buy(results),
        _larger_range(results),
        _floor_section(results),
        _paired_section(results),
        _reading(results, info),
        _recommendation(results),
        _limits(results, info),
    ]
    return "\n".join(line for section in sections for line in section).rstrip("\n") + "\n"


class _Results:
    """The rows of a run, laid out for the report."""

    def __init__(
        self,
        protocol: dict[str, Any],
        rows: Sequence[SweepRow],
        floor_rows: Sequence[FloorCheckRow],
        power_rows: Sequence[PowerRow],
    ) -> None:
        self.protocol = protocol
        self.blocks = {block.name: block for block in blocks_from_protocol(protocol)}
        self.steps = steps_from_protocol(protocol)
        self.cells = {(row.design, row.performance.block, row.performance.truth): row.performance for row in rows}
        self.truths = list(dict.fromkeys(row.performance.truth for row in rows))
        self.radii = sorted({truth.r50_mm for truth in self.truths})
        self.widths = sorted({truth.width_mm for truth in self.truths})
        self.designs = list(dict.fromkeys(row.design for row in rows))
        self.floor = {(row.variant, row.performance.block, row.performance.truth, row.fit): row for row in floor_rows}
        self.floor_blocks = list(dict.fromkeys((row.variant, row.performance.block) for row in floor_rows))
        self.floor_truths = list(dict.fromkeys(row.performance.truth for row in floor_rows))
        self.power = {(row.pairs, row.p_first, row.p_second): row.power for row in power_rows}
        self.pairs_options = sorted({row.pairs for row in power_rows})
        self.power_rates = sorted({row.p_first for row in power_rows})
        self.pairs = next((int(block["pairs"]) for block in protocol["blocks"].values() if "pairs" in block), None)

    def group(self, design: str, block: str, keep: _Keep | None = None) -> list[DesignPerformance]:
        """The rows of one design and block, for the truths that `keep` accepts."""
        return [
            self.cells[(design, block, truth)]
            for truth in self.truths
            if (design, block, truth) in self.cells and (keep is None or keep(truth))
        ]

    def against(self, design: str, blocks: Sequence[str], keep: _Keep | None = None) -> list[_Pair]:
        """Each row of a design next to the row of the design as written for the same block and truth."""
        return [
            (self.cells[(design, block, truth)], self.cells[("protocol", block, truth)])
            for block in blocks
            for truth in self.truths
            if (design, block, truth) in self.cells
            and ("protocol", block, truth) in self.cells
            and (keep is None or keep(truth))
        ]

    def floor_group(self, variant: str, block: str, fit: str, keep: _Keep | None = None) -> list[FloorCheckRow]:
        """The floor-check rows of one block variant and fit, for the truths that `keep` accepts."""
        return [
            self.floor[(variant, block, truth, fit)]
            for truth in self.floor_truths
            if (variant, block, truth, fit) in self.floor and (keep is None or keep(truth))
        ]


def _where(**wanted: float) -> _Keep:
    """Accepts the truths that have the given values, as in `_where(width_mm=3.0)`."""
    return lambda truth: all(getattr(truth, name) == value for name, value in wanted.items())


def _falls(block: BlockDesign, *positions: str) -> _Keep:
    """Accepts the truths whose fall lies in one of the given positions relative to a block's range."""
    return lambda truth: _fall_position(truth, block) in positions


def _fall_position(truth: Truth, block: BlockDesign) -> str:
    """Where the fall of a true curve lies relative to the distances a block can test."""
    half_span = FALL_HALF_SPAN_WIDTHS * truth.width_mm
    early, late = truth.r50_mm < half_span, truth.r50_mm > block.max_level_mm - half_span
    return "both" if early and late else "early" if early else "late" if late else "inside"


def _relative(pairs: Sequence[_Pair], attribute: str) -> float | None:
    """How many times larger a quantity is in the first row of each pair than in the second: the geometric mean of
    the ratios, so that no single pair dominates. None when no pair has both values positive."""
    logs = [
        math.log(getattr(first, attribute) / getattr(second, attribute))
        for first, second in pairs
        if getattr(first, attribute) > 0.0 and getattr(second, attribute) > 0.0
    ]
    return math.exp(statistics.fmean(logs)) if logs else None


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ---- small pieces of text


def _mm(value: float) -> str:
    return f"{value:.1f}"


def _signed(value: float) -> str:
    """A difference with its sign, and a plain 0.0 where it rounds to nothing."""
    return "0.0" if abs(value) < 0.05 else f"{value:+.1f}"


def _pct(share: float) -> str:
    return f"{100.0 * share:.0f}"


def _times(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{ratio:.2f}"


def _change(ratio: float | None, more: str = "larger", less: str = "smaller") -> str:
    """A ratio in words: '12 % larger', '8 % smaller', 'the same'."""
    if ratio is None:
        return "not available"
    percent = round((ratio - 1.0) * 100.0)
    return "the same" if percent == 0 else f"{abs(percent)} % {more if percent > 0 else less}"


def _list(values: Sequence[float], unit: str = "") -> str:
    """'5, 10 and 15 mm'."""
    words = [f"{value:g}" for value in values]
    joined = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    return f"{joined} {unit}".strip()


def _wrap(text: str, bullet: str = "") -> list[str]:
    return textwrap.wrap(
        text,
        width=118,
        initial_indent=bullet,
        subsequent_indent=" " * len(bullet),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _paragraph(text: str) -> list[str]:
    return [*_wrap(text), ""]


def _bullets(items: Sequence[str]) -> list[str]:
    return [line for item in items for line in _wrap(item, "- ")] + [""]


def _numbered(items: Sequence[str]) -> list[str]:
    return [line for number, item in enumerate(items, start=1) for line in _wrap(item, f"{number}. ")] + [""]


def _table(caption: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A table under its caption. The caption says where the numbers come from: simulated, or computed."""
    lines = [*_paragraph(caption), "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


def _staircases_text(block: BlockDesign) -> str:
    return "; ".join(f"{plan.id} from {plan.start_level_mm:g} mm ({plan.n_trials} trials)" for plan in block.staircases)


# ---- the sections


def _head(results: _Results, info: RunInfo) -> list[str]:
    stated = results.protocol["protocol"]
    return [
        "# Procedure simulation: how well a block of trials estimates the tolerance radius",
        "",
        *_paragraph(
            "**Every number in this report is simulated, or computed exactly from stated assumptions. Nothing here "
            "is a measurement.**"
        ),
        *_paragraph(
            f"Written by `python -m tolerance_radius.simulate`. Seed {info.seed}; {info.replications} simulated "
            f"blocks for every row. Protocol file `{info.protocol_file}`: version {stated.get('version')}, status "
            f"{stated.get('status', 'not stated')}, SHA-256 `{info.protocol_digest}`. The rows behind the tables are "
            "in `procedure.csv` (sections 1 to 5), `floor_check.csv` (section 6) and `paired_power.csv` (section 7)."
        ),
        "## The question",
        "",
        *_paragraph(
            "Before any trial is run: with the trial budget of the protocol, how well is the tolerance radius `r50` "
            "estimated, and are the trials placed the right way? Where the budget cannot pin the radius down, this "
            "report says so with the numbers, so that the protocol can be changed before it is frozen."
        ),
    ]


def _what_was_simulated(results: _Results, info: RunInfo) -> list[str]:
    analysis = results.protocol["analysis"]
    grids = grids_from_protocol(analysis)
    r50, width = analysis["r50_grid_mm"], analysis["width_grid_mm"]
    level = _pct(float(analysis["interval_level"]))
    if np.array_equal(grids.floor_values, [0.0]):
        floor_text = "no floor (the protocol file has no `floor_values`; section 6 is about adding them)"
    else:
        floor_text = f"floor values {_list(grids.floor_values.tolist())}, equally weighted"
    largest = max(block.max_level_mm for block in results.blocks.values())
    designs = [
        "**as written**: the block and the staircase steps of the protocol file "
        f"({results.steps.step_initial_mm:g} mm until a staircase first reverses, then "
        f"{results.steps.step_final_mm:g} mm);",
        "**fixed distances**: the same number of trials at fixed distances, equally spaced from 0 to the block's "
        "largest distance, one trial at each (the method of constant stimuli);",
        f"**final step {FINE_STEP_FINAL_MM:g} mm**: as written, with a final step of {FINE_STEP_FINAL_MM:g} mm;",
        f"**staircases {STAIRCASE_TRIALS_DELTA} longer** and **{STAIRCASE_TRIALS_DELTA} shorter**: as written, with "
        f"every staircase {STAIRCASE_TRIALS_DELTA} trials longer or shorter, so "
        f"{2 * STAIRCASE_TRIALS_DELTA} trials more or fewer in a block of two staircases;",
    ]
    if WIDER_RANGE in results.designs:
        designs.append(
            f"**larger range**, only for a block whose largest distance is below {largest:g} mm: as written, with "
            f"that limit raised to {largest:g} mm."
        )
    block_rows = [
        [
            block.name,
            str(block.n_trials),
            str(block.anchors_zero),
            _staircases_text(block),
            f"{block.anchors_far} at {block.far_level_mm:g} mm" if block.anchors_far else "none",
            f"{block.max_level_mm:g}",
        ]
        for block in results.blocks.values()
    ]
    share_error = 100.0 * math.sqrt(0.95 * 0.05 / info.replications)
    half_error = 100.0 * math.sqrt(0.25 / info.replications)
    rmse_error = 100.0 / math.sqrt(2.0 * info.replications)
    return [
        "## What was simulated",
        "",
        *_paragraph(
            "A block of trials is played against a success curve that is assumed true. Each requested distance "
            "comes from the protocol's own staircase rule (`staircase.next_level`). The cube lands at the requested "
            f"distance plus a placement error (normal, standard deviation {info.placement_sd_mm:g} mm, and not "
            "below 0). The trial succeeds with the probability the assumed curve gives at the distance achieved. The "
            "block is then fitted as the analysis will fit the real trials (`psychometric.fit_threshold`, on the "
            "achieved distances), and the fit is compared with the truth."
        ),
        *_bullets(
            [
                f"**The fit.** A posterior on the protocol's grid: `r50` from {r50['min']:g} to {r50['max']:g} mm in "
                f"steps of {r50['step']:g} mm; `width` at {width['count']} values from {width['min']:g} to "
                f"{width['max']:g} mm, evenly spaced in its logarithm; lapse values "
                f"{_list(grids.lapse_values.tolist())}, equally weighted; {floor_text}. Reported: the posterior "
                f"median of `r50` and its {level} % interval.",
                f"**The truths.** Every combination of `r50` = {_list(results.radii, 'mm')}, width = "
                f"{_list(results.widths, 'mm')} and lapse = {_list(sorted({t.lapse for t in results.truths}))}: "
                f"{len(results.truths)} curves. A width of 1 mm is a cliff: success goes from 88 % to 12 % of its "
                "plateau within 4 mm. A width of 10 mm is a slope: the same fall takes 40 mm.",
            ]
        ),
        *_table(
            "**The blocks**, as the protocol file states them (read from the file):",
            ["Block", "Trials", "At 0 mm", "Staircases", "Far anchors", "Largest distance (mm)"],
            block_rows,
        ),
        *_paragraph("**The designs** compared for each block:"),
        *_bullets(designs),
        *_paragraph(
            f"**How exact the simulation is.** With {info.replications} simulated blocks in a row, a share near "
            f"95 % is known to about {share_error:.1f} points (one standard error), a share near 50 % to about "
            f"{half_error:.1f} points, and an RMSE to about {rmse_error:.0f} % of its value. Numbers that pool "
            "several truths are steadier. A difference smaller than that between two single rows is noise."
        ),
    ]


def _terms() -> list[str]:
    terms = [
        ("Tolerance radius (`r50`)", "the distance to the nearest demonstration at which success is halfway between "
         "its plateau and its floor (half the plateau when there is no floor)."),
        ("Width", "the scale of the curve's fall. Success goes from 88 % to 12 % of the way down over four widths."),
        ("Lapse", "the share of failures that have nothing to do with distance. It lowers the plateau."),
        ("Floor", "the chance of a lucky success far from every demonstration. It is what is left of the curve far "
         "out."),
        ("Posterior median, 95 % interval", "the estimate and its range: the values of `r50` that leave half, and "
         "2.5 % on each side, of the probability the trials give to each value."),
        ("Bias", "the mean of (estimate - true radius) over the simulated blocks."),
        ("RMSE", "root-mean-square error: the square root of the mean of (estimate - true radius) squared. It is "
         "the typical distance of the estimate from the truth, in millimetres."),
        ("Contains the truth", "the share of simulated blocks whose 95 % interval holds the true radius. Statistics "
         "calls this the coverage of the interval; it has nothing to do with coverage of the workspace."),
        ("Open on a side", "the interval reaches an end of the `r50` grid (0 mm below, the grid's largest value "
         "above): the trials do not bound the radius on that side. The rule: the posterior at the grid's end value "
         "is at least half of what it is at the interval's own bound on that side."),
        ("Anchor", "a trial at a fixed distance, not chosen by a staircase: at 0 mm, or far out."),
        ("Reversal", "a staircase trial whose outcome differs from the one before, so that the staircase turns."),
        ("Method of constant stimuli", "testing at distances fixed in advance, whatever the outcomes."),
        ("Replication", "one simulated block. A seed makes the random numbers, and so the whole report, "
         "repeatable."),
        ("Discordant pair", "a placement at which one policy succeeded and the other failed."),
        ("Exact McNemar test", "the test of a paired comparison: are the discordant pairs split more unevenly than "
         "a fair coin would split them?"),
        ("Power", "the probability that a test detects a difference that is really there."),
        ("False-alarm rate", "the probability that a test reports a difference that is not there."),
    ]
    return ["## Terms", "", *_bullets([f"**{name}**: {meaning}" for name, meaning in terms])]


def _as_written(results: _Results) -> list[str]:
    lines = [
        "## 1. The design as written",
        "",
        *_paragraph(
            "Table 1 pools, for each block and each width of the true curve, the truths with that width (every "
            "radius and lapse rate). In each cell the first number is the middle one of those truths and the number "
            "in brackets is the worst."
        ),
    ]
    headline = []
    for block in results.blocks.values():
        for width in results.widths:
            cells = results.group("protocol", block.name, _where(width_mm=width))
            if not cells:
                continue
            rmse = [cell.rmse_mm for cell in cells]
            span = [cell.mean_interval_width_mm for cell in cells]
            inside = [cell.interval_coverage for cell in cells]
            open_side = [cell.any_edge_rate for cell in cells]
            headline.append(
                [
                    block.name,
                    str(block.n_trials),
                    f"{width:g}",
                    f"{_mm(statistics.median(rmse))} ({_mm(max(rmse))})",
                    f"{_mm(statistics.median(span))} ({_mm(max(span))})",
                    f"{_pct(statistics.median(inside))} ({_pct(min(inside))})",
                    f"{_pct(statistics.median(open_side))} ({_pct(max(open_side))})",
                ]
            )
    lines += [
        *_table(
            "**Table 1.** The design as written, by block and by width of the true curve. Simulated.",
            [
                "Block",
                "Trials",
                "Width (mm)",
                "RMSE (mm)",
                "Interval width (mm)",
                "Contains the truth (%)",
                "Open on a side (%)",
            ],
            headline,
        ),
        *_paragraph(
            "Tables 2 to 5 give the same quantities, and the bias, for each true radius, averaged over the lapse "
            "rates."
        ),
    ]
    details = (
        (
            "Table 2",
            "Bias / RMSE of the estimate, in mm",
            lambda cells: f"{_signed(_mean([cell.bias_mm for cell in cells]))} / "
            f"{_mm(_mean([cell.rmse_mm for cell in cells]))}",
        ),
        (
            "Table 3",
            "Mean width of the 95 % interval, in mm",
            lambda cells: _mm(_mean([cell.mean_interval_width_mm for cell in cells])),
        ),
        (
            "Table 4",
            "Share of blocks whose interval is open below / above, in %",
            lambda cells: f"{_pct(_mean([cell.lower_edge_rate for cell in cells]))} / "
            f"{_pct(_mean([cell.upper_edge_rate for cell in cells]))}",
        ),
        (
            "Table 5",
            "Share of blocks whose interval contains the true radius, in %",
            lambda cells: _pct(_mean([cell.interval_coverage for cell in cells])),
        ),
    )
    for number, title, cell_text in details:
        table_rows = []
        for block in results.blocks.values():
            for radius in results.radii:
                row = [block.name, f"{radius:g}"]
                for width in results.widths:
                    cells = results.group("protocol", block.name, _where(r50_mm=radius, width_mm=width))
                    row.append(cell_text(cells) if cells else "n/a")
                table_rows.append(row)
        lines += _table(
            f"**{number}.** {title}, by true radius (rows) and width of the true curve (columns). Simulated.",
            ["Block", "True `r50` (mm)", *[f"width {width:g}" for width in results.widths]],
            table_rows,
        )
    return lines


def _where_it_fails(results: _Results) -> list[str]:
    lines = [
        "## 2. Where the trials cannot pin the radius down",
        "",
        *_paragraph(
            "A radius is hard to bound on a side when the fall of the curve does not lie inside the distances that "
            "are tested. Table 6 sorts the truths by where their fall lies. Under way at 0 mm: `r50` is less than "
            "two widths, so success at 0 mm is already below 88 % of its plateau and the trials cannot show where "
            "the plateau ends. Running past the largest distance: `r50` is within two widths of the largest "
            "distance the block tests, so success there is still above 12 % of the plateau."
        ),
    ]
    table_rows = []
    for block in results.blocks.values():
        for position in ("inside", "early", "late", "both"):
            cells = results.group("protocol", block.name, _falls(block, position))
            if not cells:
                continue
            table_rows.append(
                [
                    block.name,
                    _FALL_NAMES[position],
                    str(len(cells)),
                    _mm(_mean([cell.rmse_mm for cell in cells])),
                    _mm(_mean([cell.mean_interval_width_mm for cell in cells])),
                    _pct(_mean([cell.lower_edge_rate for cell in cells])),
                    _pct(_mean([cell.upper_edge_rate for cell in cells])),
                    _pct(_mean([cell.interval_coverage for cell in cells])),
                ]
            )
    return [
        *lines,
        *_table(
            "**Table 6.** The design as written, by where the fall of the true curve lies. Means over the truths "
            "of each kind. Simulated.",
            [
                "Block",
                "The fall of the true curve is",
                "Truths",
                "RMSE (mm)",
                "Interval width (mm)",
                "Open below (%)",
                "Open above (%)",
                "Contains the truth (%)",
            ],
            table_rows,
        ),
    ]


def _design_row(results: _Results, design: str, block: BlockDesign) -> list[str] | None:
    pairs = results.against(design, [block.name])
    if not pairs:
        return None
    row = [block.name, _DESIGN_NAMES[design], str(pairs[0][0].n_trials), _times(_relative(pairs, "rmse_mm"))]
    for width in results.widths:
        of_width = results.against(design, [block.name], _where(width_mm=width))
        row.append(_times(_relative(of_width, "rmse_mm")))
    return [
        *row,
        _times(_relative(pairs, "mean_interval_width_mm")),
        f"{_pct(_mean([first.any_edge_rate for first, _ in pairs]))} : "
        f"{_pct(_mean([second.any_edge_rate for _, second in pairs]))}",
        f"{_pct(_mean([first.interval_coverage for first, _ in pairs]))} : "
        f"{_pct(_mean([second.interval_coverage for _, second in pairs]))}",
    ]


def _design_table(results: _Results, caption: str, designs: Sequence[str]) -> list[str]:
    table_rows = [
        row
        for block in results.blocks.values()
        for design in designs
        if (row := _design_row(results, design, block)) is not None
    ]
    return _table(
        caption,
        [
            "Block",
            "Design",
            "Trials",
            "RMSE, all truths",
            *[f"RMSE, width {width:g}" for width in results.widths],
            "Interval width, all truths",
            "Open on a side (%), design : as written",
            "Contains the truth (%), design : as written",
        ],
        table_rows,
    )


def _other_placements(results: _Results) -> list[str]:
    return [
        "## 3. Other ways to place the same trials",
        "",
        *_paragraph(
            "Two designs with the same number of trials as the design as written. In Table 7 the RMSE and the "
            "interval width are given relative to the design as written: 1.00 is equal, 1.20 is 20 % larger. The "
            "ratios are averaged over the truths on a logarithmic scale, so that no single truth dominates."
        ),
        *_design_table(
            results,
            "**Table 7.** Fixed distances and a finer final step, relative to the design as written. Simulated.",
            ("constant", "fine_step"),
        ),
    ]


def _what_trials_buy(results: _Results) -> list[str]:
    return [
        "## 4. What trials buy",
        "",
        *_paragraph(
            f"Every staircase made {STAIRCASE_TRIALS_DELTA} trials shorter, and {STAIRCASE_TRIALS_DELTA} trials "
            "longer. Table 8 reads like Table 7."
        ),
        *_design_table(
            results,
            "**Table 8.** Shorter and longer staircases, relative to the design as written. Simulated.",
            ("shorter", "longer"),
        ),
    ]


def _larger_range(results: _Results) -> list[str]:
    narrow = [block for block in results.blocks.values() if results.group(WIDER_RANGE, block.name)]
    if not narrow:
        return [
            "## 5. A larger range for a block that stops short",
            "",
            *_paragraph("Every block of the protocol goes out to the same largest distance: nothing to compare."),
        ]
    largest = max(block.max_level_mm for block in results.blocks.values())
    table_rows = []
    for block in narrow:
        for radius in results.radii:
            pairs = results.against(WIDER_RANGE, [block.name], _where(r50_mm=radius))
            if not pairs:
                continue
            row = [block.name, f"{radius:g}"]
            for attribute, text in (("rmse_mm", _mm), ("mean_interval_width_mm", _mm), ("upper_edge_rate", _pct)):
                before = _mean([getattr(old, attribute) for _, old in pairs])
                after = _mean([getattr(new, attribute) for new, _ in pairs])
                row.append(f"{text(before)} → {text(after)}")
            table_rows.append(row)
    names = " and ".join(f"{block.name} (largest distance {block.max_level_mm:g} mm)" for block in narrow)
    return [
        "## 5. A larger range for a block that stops short",
        "",
        *_paragraph(
            f"{names} cannot test beyond that distance, while another block goes to {largest:g} mm. Table 9 gives "
            f"the same block as written and with its largest distance raised to {largest:g} mm, for each true "
            "radius (means over the widths and lapse rates)."
        ),
        *_table(
            f"**Table 9.** As written → with the largest distance raised to {largest:g} mm. Simulated.",
            ["Block", "True `r50` (mm)", "RMSE (mm)", "Interval width (mm)", "Open above (%)"],
            table_rows,
        ),
    ]


def _floor_mean(rows: Sequence[FloorCheckRow], attribute: str) -> float:
    return _mean([getattr(row.performance, attribute) for row in rows])


def _floor_section(results: _Results) -> list[str]:
    if not results.floor:
        return []
    true_floors = sorted({truth.floor for truth in results.floor_truths}, reverse=True)
    radii = sorted({truth.r50_mm for truth in results.floor_truths})
    widths = sorted({truth.width_mm for truth in results.floor_truths})
    lapses = sorted({truth.lapse for truth in results.floor_truths})
    pooled_rows = []
    for variant, block in results.floor_blocks:
        for floor in true_floors:
            for fit in (FIT_NO_FLOOR, FIT_FLOOR):
                rows = results.floor_group(variant, block, fit, _where(floor=floor))
                if not rows:
                    continue
                pooled_rows.append(
                    [
                        f"{block}, {_VARIANT_NAMES[variant]}",
                        str(rows[0].performance.n_trials),
                        f"{floor:g}",
                        _FIT_NAMES[fit],
                        _signed(_floor_mean(rows, "bias_mm")),
                        _mm(_floor_mean(rows, "rmse_mm")),
                        _mm(_floor_mean(rows, "mean_interval_width_mm")),
                        _pct(_floor_mean(rows, "interval_coverage")),
                        f"{_mean([row.mean_floor_estimate for row in rows]):.3f}" if fit == FIT_FLOOR else "-",
                    ]
                )
    with_floor = [floor for floor in true_floors if floor > 0.0]
    by_width_rows = []
    for variant, block in results.floor_blocks:
        if variant != "as_written":
            continue
        for width in widths:
            row = [block, f"{width:g}"]
            for floor in true_floors:
                both = [
                    results.floor_group(variant, block, fit, _where(floor=floor, width_mm=width))
                    for fit in (FIT_NO_FLOOR, FIT_FLOOR)
                ]
                for attribute, text in (("rmse_mm", _mm), ("bias_mm", _signed), ("mean_interval_width_mm", _mm)):
                    row.append(" → ".join(text(_floor_mean(rows, attribute)) for rows in both))
            by_width_rows.append(row)
    by_width_header = ["Block", "Width (mm)"]
    for floor in true_floors:
        by_width_header += [f"True floor {floor:g}: {name}" for name in ("RMSE (mm)", "bias (mm)", "interval (mm)")]
    blocks_text = "; ".join(f"{block} {_VARIANT_NAMES[variant]}" for variant, block in results.floor_blocks)
    checked = {block for _, block in results.floor_blocks}
    left_out = [name for name, block in results.blocks.items() if block.anchors_far == 0 and name not in checked]
    has, was = ("has", "was") if len(left_out) == 1 else ("have", "were")
    left_out_text = (
        f" {' and '.join(left_out)} {has} no far anchors either and {was} not run in this check." if left_out else ""
    )
    return [
        "## 6. A floor in the model",
        "",
        *_paragraph(
            "The curves of sections 1 to 5 end at 0 far from the demonstrations. If a policy now and then succeeds "
            "far out by luck, a model without a floor can explain that success only by a wider or a farther curve, "
            "and the radius moves. This check plays truths with a floor and the same truths without one, and fits "
            "every simulated block twice: without a floor in the model, and with floor values "
            f"{_list(FLOOR_CHECK_FLOOR_VALUES)} (equally weighted). The two fits see the same trials."
        ),
        *_bullets(
            [
                f"**Truths:** floor = {_list(true_floors)}; `r50` = {_list(radii, 'mm')}; width = "
                f"{_list(widths, 'mm')}; lapse = {_list(lapses)}.",
                f"**Blocks:** {blocks_text}. The far anchors added to a block that has none stand at its largest "
                "distance; the staircase trials added instead give the far anchors a fair comparison at an equal "
                f"number of trials.{left_out_text}",
            ]
        ),
        *_table(
            "**Table 10.** The two fits of the same simulated blocks, by block and by true floor. Means over the "
            f"{len(radii) * len(widths) * len(lapses)} truths of each floor. The last column is the posterior mean "
            "of the floor, averaged over the blocks (its prior mean is "
            f"{_mean(FLOOR_CHECK_FLOOR_VALUES):.3f}). Simulated.",
            [
                "Block",
                "Trials",
                "True floor",
                "Fit",
                "Bias (mm)",
                "RMSE (mm)",
                "Interval width (mm)",
                "Contains the truth (%)",
                "Floor estimate",
            ],
            pooled_rows,
        ),
        *_table(
            "**Table 11.** The blocks as written, by width of the true curve: fit without a floor → fit with "
            "floor values. Simulated.",
            by_width_header,
            by_width_rows,
        ),
    ]


def _paired_section(results: _Results) -> list[str]:
    if not results.power:
        return ["## 7. The paired comparison", "", *_paragraph("The protocol has no block of paired trials.")]
    options = results.pairs_options
    pairs = results.pairs if results.pairs in options else options[len(options) // 2]
    rates = results.power_rates
    matrix = [
        [f"{first:g}", *[_pct(results.power[(pairs, first, second)]) for second in rates]] for first in rates
    ]
    needed = [
        [f"{low:.2f}", *[_needed_text(option, low) for option in results.pairs_options]]
        for low in rates
        if low <= 0.7
    ]
    return [
        "## 7. The paired comparison",
        "",
        *_paragraph(
            f"The workspace test places the cube at {pairs} positions, runs both policies at each, and compares "
            "them with the exact McNemar test. Table 12 gives the probability that the test reaches p < 0.05 for "
            "assumed success rates of the two policies, with outcomes independent of each other. These are exact "
            "sums over every possible table of outcomes, not simulations. On the diagonal the two policies are "
            "equal: the entry is the test's false-alarm rate."
        ),
        *_table(
            f"**Table 12.** Probability (%) that the exact McNemar test gives p < 0.05 with {pairs} pairs. Rows: "
            "the success rate of one policy. Columns: that of the other. Computed.",
            ["", *[f"{rate:g}" for rate in rates]],
            matrix,
        ),
        *_table(
            "**Table 13.** The smallest success rate (in steps of 0.05) that the test tells from the lower one at "
            f"least {_pct(POWER_TARGET)} % of the time, by number of pairs. Computed.",
            ["Lower rate", *[f"{option} pairs" for option in results.pairs_options]],
            needed,
        ),
    ]


def _needed_text(pairs: int, lower_rate: float) -> str:
    needed = rate_needed(pairs, lower_rate)
    return "not reached" if needed is None else f"{needed:.2f}"


def _tells_apart(pairs: int, lower_rate: float) -> str:
    """'0.40 from 0.90': the lower rate and the smallest rate the paired test tells from it."""
    needed = rate_needed(pairs, lower_rate)
    return f"{lower_rate:.2f} from no higher rate" if needed is None else f"{lower_rate:.2f} from {needed:.2f}"


# ---- what the reading and the recommendation are drawn from


@dataclass(frozen=True, slots=True)
class _Comparison:
    """A design against the design as written, over every block and truth. Ratios: 1 is equal."""

    rmse: float | None
    interval: float | None
    rmse_by_width: dict[float, float | None]
    #: The share of intervals open on a side, and the share that contain the truth: this design, then as written.
    open_share: float
    open_share_written: float
    inside: float
    inside_written: float

    @property
    def verdict(self) -> str:
        """`"better"`, `"worse"` or `"the same"`, by the report's rule: the RMSE differs by more than
        `MEANINGFUL_CHANGE`; and to count as better a design must not widen the interval by more than that, nor
        leave it open on a side or missing the truth more often by more than 2 points."""
        if self.rmse is None or self.interval is None:
            return "the same"
        if self.rmse >= 1.0 + MEANINGFUL_CHANGE:
            return "worse"
        harmless = (
            self.interval <= 1.0 + MEANINGFUL_CHANGE
            and self.open_share <= self.open_share_written + 0.02
            and self.inside >= self.inside_written - 0.02
        )
        return "better" if self.rmse <= 1.0 - MEANINGFUL_CHANGE and harmless else "the same"

    @property
    def numbers(self) -> str:
        return (
            f"the RMSE is {_change(self.rmse)} and the interval {_change(self.interval, 'wider', 'narrower')}, and "
            f"the interval is open on a side in {_pct(self.open_share)} % of blocks against "
            f"{_pct(self.open_share_written)} %"
        )

    @property
    def width_range(self) -> str:
        """' (from 16 % larger at width 10 mm to 71 % larger at width 1 mm)', or nothing."""
        known = {width: ratio for width, ratio in self.rmse_by_width.items() if ratio is not None}
        if not known:
            return ""
        low, high = min(known, key=known.get), max(known, key=known.get)
        return f" (from {_change(known[low])} at width {low:g} mm to {_change(known[high])} at width {high:g} mm)"


def _compare(results: _Results, design: str) -> _Comparison | None:
    blocks = list(results.blocks)
    pairs = results.against(design, blocks)
    if not pairs:
        return None
    return _Comparison(
        rmse=_relative(pairs, "rmse_mm"),
        interval=_relative(pairs, "mean_interval_width_mm"),
        rmse_by_width={
            width: _relative(results.against(design, blocks, _where(width_mm=width)), "rmse_mm")
            for width in results.widths
        },
        open_share=_mean([first.any_edge_rate for first, _ in pairs]),
        open_share_written=_mean([second.any_edge_rate for _, second in pairs]),
        inside=_mean([first.interval_coverage for first, _ in pairs]),
        inside_written=_mean([second.interval_coverage for _, second in pairs]),
    )


@dataclass(frozen=True, slots=True)
class _RangeChange:
    """A block as written against the same block with a larger range."""

    block: BlockDesign
    largest_mm: float
    #: For the truths whose fall runs past the block's largest distance: as written, then with the larger range.
    open_above: tuple[float, float]
    rmse_mm: tuple[float, float]
    rmse_ratio: float | None
    #: For the other truths: RMSE with the larger range relative to as written.
    elsewhere: float | None

    @property
    def worthwhile(self) -> bool:
        """The report's rule: the larger range helps where the fall runs past the block's reach (the share of
        intervals open above falls by 10 points or the RMSE by more than `MEANINGFUL_CHANGE`) and costs nothing
        elsewhere."""
        helps = self.open_above[0] - self.open_above[1] >= 0.10 or (self.rmse_ratio or 1.0) <= 1.0 - MEANINGFUL_CHANGE
        return helps and (self.elsewhere is None or self.elsewhere <= 1.0 + MEANINGFUL_CHANGE)


def _range_changes(results: _Results) -> list[_RangeChange]:
    largest = max(block.max_level_mm for block in results.blocks.values())
    changes = []
    for block in results.blocks.values():
        top = results.against(WIDER_RANGE, [block.name], _falls(block, "late", "both"))
        rest = results.against(WIDER_RANGE, [block.name], _falls(block, "inside", "early"))
        if top:
            changes.append(
                _RangeChange(
                    block=block,
                    largest_mm=largest,
                    open_above=(
                        _mean([old.upper_edge_rate for _, old in top]),
                        _mean([new.upper_edge_rate for new, _ in top]),
                    ),
                    rmse_mm=(_mean([old.rmse_mm for _, old in top]), _mean([new.rmse_mm for new, _ in top])),
                    rmse_ratio=_relative(top, "rmse_mm"),
                    elsewhere=_relative(rest, "rmse_mm"),
                )
            )
    return changes


@dataclass(frozen=True, slots=True)
class _TrialChange:
    """What longer and shorter staircases do to one block. Ratios to the block as written."""

    block: BlockDesign
    longer_rmse: float | None
    longer_interval: float | None
    shorter_rmse: float | None
    shorter_interval: float | None


def _trial_changes(results: _Results) -> list[_TrialChange]:
    changes = []
    for block in results.blocks.values():
        longer, shorter = results.against("longer", [block.name]), results.against("shorter", [block.name])
        if longer and shorter:
            changes.append(
                _TrialChange(
                    block=block,
                    longer_rmse=_relative(longer, "rmse_mm"),
                    longer_interval=_relative(longer, "mean_interval_width_mm"),
                    shorter_rmse=_relative(shorter, "rmse_mm"),
                    shorter_interval=_relative(shorter, "mean_interval_width_mm"),
                )
            )
    return changes


@dataclass(frozen=True, slots=True)
class _FloorFindings:
    """What the floor check found. Unless said otherwise, pooled over the blocks as written."""

    true_floor: float
    # A floor in the truth: the fit without one (`harm`) against the fit with one (`kept`).
    harm_rmse: float | None
    harm_bias_mm: float
    kept_bias_mm: float
    harm_inside: float
    kept_inside: float
    # No floor in the truth: the fit with one (`cost`) against the fit without (`plain`).
    cost_interval: float | None
    cost_rmse: float | None
    cost_bias_mm: float
    plain_bias_mm: float
    cost_inside: float
    plain_inside: float
    # The posterior mean of the floor with a floor in the truth and without, in the blocks that have far anchors.
    anchored: tuple[str, ...]
    anchors: int
    estimates: tuple[float, float]
    # The block that was given far anchors: against the same number of staircase trials (fit with floor values),
    # against itself without them (fit without, floor in the truth), and its floor estimates with and without them.
    far_block: str | None
    far_rmse: float | None
    far_interval: float | None
    far_unprotected: float | None
    far_estimates: tuple[float, float]
    bare_estimates: tuple[float, float]

    @property
    def worth_having(self) -> bool:
        """The report's rule, first half: ignoring a true floor makes the RMSE worse by more than
        `MEANINGFUL_CHANGE`."""
        return self.harm_rmse is not None and self.harm_rmse >= 1.0 + MEANINGFUL_CHANGE

    @property
    def cheap(self) -> bool:
        """The rule's second half: where there is no floor, allowing for one widens the interval by at most
        `ACCEPTABLE_FLOOR_COST` and does not lose the truth more often by more than 2 points."""
        return (
            self.cost_interval is not None
            and self.cost_interval <= 1.0 + ACCEPTABLE_FLOOR_COST
            and self.cost_inside >= self.plain_inside - 0.02
        )


def _floor_is_readable(estimates: tuple[float, float], true_floor: float) -> bool:
    """The report's rule: the floor can be read from the trials when its posterior mean moves by at least half the
    true floor between a truth with that floor and a truth without one."""
    return estimates[0] - estimates[1] >= 0.5 * true_floor


def _floor_findings(results: _Results) -> _FloorFindings | None:
    if not results.floor:
        return None
    as_written = [(variant, block) for variant, block in results.floor_blocks if variant == "as_written"]
    anchored = [(variant, block) for variant, block in as_written if results.blocks[block].anchors_far > 0]
    far = [(variant, block) for variant, block in results.floor_blocks if variant == "far_anchors"]
    longer = [(variant, block) for variant, block in results.floor_blocks if variant == "longer_staircases"]
    bare = [("as_written", block) for _, block in far]

    def rows(fit: str, floor_in_truth: bool, blocks: Sequence[tuple[str, str]] = as_written) -> list[FloorCheckRow]:
        return [
            row
            for variant, block in blocks
            for row in results.floor_group(variant, block, fit, lambda truth: (truth.floor > 0.0) == floor_in_truth)
        ]

    def relative(first: list[FloorCheckRow], second: list[FloorCheckRow], attribute: str) -> float | None:
        if not first or len(first) != len(second):
            return None
        return _relative([(a.performance, b.performance) for a, b in zip(first, second, strict=True)], attribute)

    def estimates(blocks: Sequence[tuple[str, str]]) -> tuple[float, float]:
        with_floor, without = (
            _mean([row.mean_floor_estimate for row in rows(FIT_FLOOR, floor_in_truth, blocks)])
            for floor_in_truth in (True, False)
        )
        return with_floor, without

    far_fitted = rows(FIT_FLOOR, True, far) + rows(FIT_FLOOR, False, far)
    longer_fitted = rows(FIT_FLOOR, True, longer) + rows(FIT_FLOOR, False, longer)
    return _FloorFindings(
        true_floor=max(truth.floor for truth in results.floor_truths),
        harm_rmse=relative(rows(FIT_NO_FLOOR, True), rows(FIT_FLOOR, True), "rmse_mm"),
        harm_bias_mm=_floor_mean(rows(FIT_NO_FLOOR, True), "bias_mm"),
        kept_bias_mm=_floor_mean(rows(FIT_FLOOR, True), "bias_mm"),
        harm_inside=_floor_mean(rows(FIT_NO_FLOOR, True), "interval_coverage"),
        kept_inside=_floor_mean(rows(FIT_FLOOR, True), "interval_coverage"),
        cost_interval=relative(rows(FIT_FLOOR, False), rows(FIT_NO_FLOOR, False), "mean_interval_width_mm"),
        cost_rmse=relative(rows(FIT_FLOOR, False), rows(FIT_NO_FLOOR, False), "rmse_mm"),
        cost_bias_mm=_floor_mean(rows(FIT_FLOOR, False), "bias_mm"),
        plain_bias_mm=_floor_mean(rows(FIT_NO_FLOOR, False), "bias_mm"),
        cost_inside=_floor_mean(rows(FIT_FLOOR, False), "interval_coverage"),
        plain_inside=_floor_mean(rows(FIT_NO_FLOOR, False), "interval_coverage"),
        anchored=tuple(block for _, block in anchored),
        anchors=max((results.blocks[block].anchors_far for _, block in anchored), default=0),
        estimates=estimates(anchored),
        far_block=far[0][1] if far else None,
        far_rmse=relative(far_fitted, longer_fitted, "rmse_mm"),
        far_interval=relative(far_fitted, longer_fitted, "mean_interval_width_mm"),
        far_unprotected=relative(rows(FIT_NO_FLOOR, True, far), rows(FIT_NO_FLOOR, True, bare), "rmse_mm"),
        far_estimates=estimates(far),
        bare_estimates=estimates(bare),
    )


def _coverage_of(cell: DesignPerformance) -> float:
    return cell.interval_coverage


def _row_name(cell: DesignPerformance) -> str:
    truth = cell.truth
    return f"{cell.block}, `r50` {truth.r50_mm:g} mm, width {truth.width_mm:g} mm, lapse {truth.lapse:g}"


# ---- the reading


def _reading(results: _Results, info: RunInfo) -> list[str]:
    items = [
        *_read_blocks(results),
        *_read_bias(results),
        *_read_interval(results, info),
        *_read_bounds(results),
        *_read_fixed_distances(results),
        *_read_final_step(results),
        *_read_trials(results),
        *_read_range(results),
        *_read_floor(results),
        *_read_pairs(results),
    ]
    return ["## Reading", "", *_bullets(items)]


def _read_blocks(results: _Results) -> list[str]:
    narrowest, widest = results.widths[0], results.widths[-1]
    items = []
    for block in results.blocks.values():
        sharp = results.group("protocol", block.name, _where(width_mm=narrowest))
        gentle = results.group("protocol", block.name, _where(width_mm=widest))
        if sharp and gentle:
            items.append(
                f"**{block.name} as written ({block.n_trials} trials).** A cliff (width {narrowest:g} mm) is "
                f"typically located with an RMSE of {_mm(statistics.median(cell.rmse_mm for cell in sharp))} mm and a "
                f"95 % interval {_mm(statistics.median(cell.mean_interval_width_mm for cell in sharp))} mm wide. A "
                f"slope of width {widest:g} mm is located with an RMSE of "
                f"{_mm(statistics.median(cell.rmse_mm for cell in gentle))} mm and an interval "
                f"{_mm(statistics.median(cell.mean_interval_width_mm for cell in gentle))} mm wide."
            )
    return items


def _read_bias(results: _Results) -> list[str]:
    written = [cell for block in results.blocks for cell in results.group("protocol", block)]
    if not written:
        return []
    narrowest, widest = results.widths[0], results.widths[-1]
    by_width = [
        _mean([cell.bias_mm for cell in written if cell.truth.width_mm == width]) for width in (narrowest, widest)
    ]
    largest = max(written, key=lambda cell: abs(cell.bias_mm))
    return [
        "**Which way the estimate leans.** The mean error of the estimate (its bias) is "
        f"{_signed(by_width[0])} mm at width {narrowest:g} mm and {_signed(by_width[1])} mm at width {widest:g} mm, "
        f"averaged over the blocks, radii and lapse rates. The largest in any row is {_signed(largest.bias_mm)} mm "
        f"({_row_name(largest)}). A positive bias is an estimate beyond the true radius."
    ]


def _read_interval(results: _Results, info: RunInfo) -> list[str]:
    """Whether the interval holds the truth as often as its level says, judged against the simulation's own noise."""
    written = [cell for block in results.blocks for cell in results.group("protocol", block)]
    if not written:
        return []
    level = float(results.protocol["analysis"]["interval_level"])
    noise = 3.0 * math.sqrt(level * (1.0 - level) / info.replications)  # three standard errors of one row
    short = sorted((cell for cell in written if cell.interval_coverage < level - noise), key=_coverage_of)
    lowest = min(written, key=_coverage_of)
    numbers = (
        f"Over the {len(written)} rows of the design as written, the {_pct(level)} % interval contained the true "
        f"radius in {_pct(_mean([cell.interval_coverage for cell in written]))} % of the simulated blocks on average "
        f"and in {_pct(lowest.interval_coverage)} % in the lowest row ({_row_name(lowest)})."
    )
    allowance = f"the simulation's own noise ({100.0 * noise:.1f} points, three standard errors of one row)"
    if not short:
        return [
            f"**The interval holds what it promises.** {numbers} No row is below {_pct(level)} % by more than "
            f"{allowance}."
        ]
    names = "; ".join(f"{_row_name(cell)}: {_pct(cell.interval_coverage)} %" for cell in short[:3])
    return [
        f"**The interval falls short of {_pct(level)} % for some truths.** {numbers} {len(short)} rows are below "
        f"{_pct(level)} % by more than {allowance}. The lowest: {names}."
    ]


def _read_bounds(results: _Results) -> list[str]:
    positions: dict[str, list[DesignPerformance]] = {}
    for block in results.blocks.values():
        for truth in results.truths:
            cell = results.cells.get(("protocol", block.name, truth))
            if cell is not None:
                positions.setdefault(_fall_position(truth, block), []).append(cell)
    if not positions.get("inside"):
        return []
    text = (
        "**What the trials cannot bound.** Where the fall of the true curve lies inside the tested range, the "
        f"interval is open on a side in {_pct(_mean([cell.any_edge_rate for cell in positions['inside']]))} % of the "
        "simulated blocks."
    )
    if positions.get("early"):
        early = positions["early"]
        text += (
            " Where the curve is already falling at 0 mm (`r50` under two widths), it is open below in "
            f"{_pct(_mean([cell.lower_edge_rate for cell in early]))} %: the estimate then reads as \"at most\" and "
            "not as a radius, although its interval still holds the truth "
            f"({_pct(_mean([cell.interval_coverage for cell in early]))} %)."
        )
    if positions.get("late"):
        late = positions["late"]
        text += (
            " Where the fall runs past the largest distance tested, it is open above in "
            f"{_pct(_mean([cell.upper_edge_rate for cell in late]))} % and "
            f"{_mm(_mean([cell.mean_interval_width_mm for cell in late]))} mm wide on average."
        )
    return [text + " Table 6 has this block by block."]


def _read_fixed_distances(results: _Results) -> list[str]:
    found = _compare(results, "constant")
    if found is None:
        return []
    return [
        "**Staircases against fixed distances, same number of trials.** With fixed distances the RMSE is "
        f"{_change(found.rmse)}{found.width_range} and the interval is "
        f"{_change(found.interval, 'wider', 'narrower')}; the interval is open on a side in "
        f"{_pct(found.open_share)} % of blocks against {_pct(found.open_share_written)} %."
    ]


def _read_final_step(results: _Results) -> list[str]:
    found = _compare(results, "fine_step")
    if found is None:
        return []
    return [
        f"**A final step of {FINE_STEP_FINAL_MM:g} mm in place of {results.steps.step_final_mm:g} mm.** The RMSE is "
        f"{_change(found.rmse)}{found.width_range} and the interval is "
        f"{_change(found.interval, 'wider', 'narrower')}."
    ]


def _read_trials(results: _Results) -> list[str]:
    trials = 2 * STAIRCASE_TRIALS_DELTA
    sentences = [
        f"{change.block.name}: {trials} trials fewer make the RMSE {_change(change.shorter_rmse)} and the interval "
        f"{_change(change.shorter_interval, 'wider', 'narrower')}; {trials} more make the RMSE "
        f"{_change(change.longer_rmse)} and the interval {_change(change.longer_interval, 'wider', 'narrower')}"
        for change in _trial_changes(results)
    ]
    return ["**What trials buy.** " + ". ".join(sentences) + "."] if sentences else []


def _read_range(results: _Results) -> list[str]:
    return [
        f"**{change.block.name} stops at {change.block.max_level_mm:g} mm.** For a true radius within two widths of "
        f"that distance, its interval is open above in {_pct(change.open_above[0])} % of blocks and its RMSE is "
        f"{_mm(change.rmse_mm[0])} mm. Allowed to go to {change.largest_mm:g} mm, the same block gives "
        f"{_pct(change.open_above[1])} % and {_mm(change.rmse_mm[1])} mm, and for the other radii its RMSE is "
        f"{_change(change.elsewhere)}."
        for change in _range_changes(results)
    ]


def _read_floor(results: _Results) -> list[str]:
    floor = _floor_findings(results)
    if floor is None:
        return []
    items = [
        f"**A floor that the model ignores.** With a true floor of {floor.true_floor:g}, the fit without a floor is "
        f"off by {_signed(floor.harm_bias_mm)} mm on average, against {_signed(floor.kept_bias_mm)} mm for the fit "
        f"with floor values, and its RMSE is {_change(floor.harm_rmse)}. Both intervals hold the truth about as "
        f"often ({_pct(floor.harm_inside)} % and {_pct(floor.kept_inside)} %): the fit without a floor pays with a "
        "radius pushed outward and a wider interval.",
        "**A floor in the model when there is none in the truth.** The interval is "
        f"{_change(floor.cost_interval, 'wider', 'narrower')}, the RMSE is {_change(floor.cost_rmse)}, the estimate "
        f"is off by {_signed(floor.cost_bias_mm)} mm on average (against {_signed(floor.plain_bias_mm)} mm without a "
        f"floor in the model), and the interval holds the truth in {_pct(floor.cost_inside)} % of blocks (against "
        f"{_pct(floor.plain_inside)} %).",
    ]
    prior = _mean(FLOOR_CHECK_FLOOR_VALUES)
    if floor.anchored:
        numbers = (
            f"In {' and '.join(floor.anchored)} the posterior mean of the floor is {floor.estimates[0]:.3f} when the "
            f"true floor is {floor.true_floor:g} and {floor.estimates[1]:.3f} when it is 0; its prior mean is "
            f"{prior:.3f}."
        )
        if _floor_is_readable(floor.estimates, floor.true_floor):
            items.append(f"**{floor.anchors} far anchors show the size of the floor.** {numbers}")
        else:
            weights = " and ".join(
                f"{(1.0 - value) ** floor.anchors:.2f} for a floor of {value:g}"
                for value in FLOOR_CHECK_FLOOR_VALUES
                if value > 0.0
            )
            items.append(
                f"**The size of the floor cannot be read from {floor.anchors} far anchors.** {numbers} The two "
                "differ by less than half the true floor. That is arithmetic more than chance: "
                f"{floor.anchors} far anchors that all fail multiply the weight of each floor value by {weights}, "
                "against 1 for no floor. The floor values make the radius robust to a lucky success; they do not "
                "measure how often luck strikes."
            )
    if floor.far_block is not None and floor.far_rmse is not None:
        readable = _floor_is_readable(floor.far_estimates, floor.true_floor)
        items.append(
            f"**Far anchors for a block that has none ({floor.far_block}).** With floor values in the model, "
            f"{FLOOR_CHECK_FAR_ANCHORS} far anchors give an RMSE {_change(floor.far_rmse)} and an interval "
            f"{_change(floor.far_interval, 'wider', 'narrower')} than the same {FLOOR_CHECK_FAR_ANCHORS} trials spent "
            "on the staircases. With the far anchors the posterior mean of the floor is "
            f"{floor.far_estimates[0]:.3f} when the true floor is {floor.true_floor:g} and "
            f"{floor.far_estimates[1]:.3f} when it is 0 (without them: {floor.bare_estimates[0]:.3f} and "
            f"{floor.bare_estimates[1]:.3f}): "
            + ("they make the floor readable." if readable else "they do not make the floor readable either.")
            + " Without floor values in the model and with a floor in the truth, the far anchors make the RMSE "
            f"{_change(floor.far_unprotected)} than in the same block without them: a far anchor is where a lucky "
            "success does most harm to a model that cannot explain it."
        )
    return items


def _read_pairs(results: _Results) -> list[str]:
    if not results.power or results.pairs is None:
        return []
    lows = [rate for rate in (0.3, 0.4, 0.5) if rate in results.power_rates]
    needed = "; ".join(f"from {low:.2f}, {_needed_text(results.pairs, low)}" for low in lows)
    alarm = max(results.power[(results.pairs, rate, rate)] for rate in results.power_rates)
    others = "; ".join(
        f"with {option} pairs: " + ", ".join(_needed_text(option, low) for low in lows)
        for option in results.pairs_options
        if option != results.pairs
    )
    return [
        f"**The paired test with {results.pairs} pairs detects large differences only.** The success rate the other "
        f"policy needs for the test to reach p < 0.05 at least {_pct(POWER_TARGET)} % of the time: {needed}. For the "
        f"same lower rates, {others}. The false-alarm rate is at most {100.0 * alarm:.1f} %."
    ]


# ---- the recommendation


@dataclass(frozen=True, slots=True)
class _Advice:
    """One item of the recommendation."""

    text: str
    #: The change to the protocol file this item asks for, in a few words; None when it asks for none.
    change: str | None = None
    #: True when the change replaces the design itself and not one of its parameters.
    replaces_design: bool = False


def _recommendation(results: _Results) -> list[str]:
    items = [
        *_advise_staircases(results),
        *_advise_final_step(results),
        *_advise_range(results),
        *_advise_trials(results),
        *_advise_floor(results),
        *_advise_pairs(results),
        _advise_edge_rule(),
    ]
    changes = [item.change for item in items if item.change is not None]
    count = f"{len(changes)} change{'s' if len(changes) != 1 else ''} to the protocol file"
    if not changes:
        short = (
            "**In short: the simulation supports the design as written; no parameter of the protocol file needs to "
            "change.** The reasons follow."
        )
    elif any(item.replaces_design for item in items):
        short = f"**In short: the simulation asks for {count}:** {'; '.join(changes)}. The reasons follow."
    else:
        short = (
            f"**In short: the simulation supports the design as written, with {count}:** {'; '.join(changes)}. The "
            "reasons follow."
        )
    return [
        "## Recommendation",
        "",
        *_paragraph(short),
        *_paragraph(
            "The items below are written from the numbers above by fixed rules. A design counts as better or worse "
            "than the design as written when its RMSE, pooled over the truths, differs by more than "
            f"{_pct(MEANINGFUL_CHANGE)} %. A floor in the model counts as cheap when it widens the interval by at "
            f"most {_pct(ACCEPTABLE_FLOOR_COST)} % where the truth has no floor."
        ),
        *_numbered([item.text for item in items]),
    ]


def _advise_staircases(results: _Results) -> list[_Advice]:
    found = _compare(results, "constant")
    if found is None:
        return []
    if found.verdict == "worse":
        return [
            _Advice(
                "**Keep the staircases** (`staircase.rule = \"1up1down\"`). With the same trials at fixed, equally "
                f"spaced distances {found.numbers}."
            )
        ]
    if found.verdict == "better":
        return [
            _Advice(
                "**Replace the staircases by fixed, equally spaced distances**, which needs a new description of "
                f"the blocks in `config/protocol.toml`. With fixed distances {found.numbers}.",
                change="fixed, equally spaced distances in place of the staircases",
                replaces_design=True,
            )
        ]
    return [
        _Advice(
            "**Keep the staircases** (`staircase.rule = \"1up1down\"`): fixed distances do no better. With them "
            f"{found.numbers}."
        )
    ]


def _advise_final_step(results: _Results) -> list[_Advice]:
    found = _compare(results, "fine_step")
    if found is None:
        return []
    numbers = f"the RMSE is {_change(found.rmse)} and the interval {_change(found.interval, 'wider', 'narrower')}"
    current = results.steps.step_final_mm
    if found.verdict == "better":
        change = f"`staircase.step_final_mm` from {current:g} to {FINE_STEP_FINAL_MM:g}"
        return [_Advice(f"**Change {change}.** With the finer step {numbers}.", change=change)]
    return [
        _Advice(
            f"**Keep `staircase.step_final_mm = {current:g}`.** With a final step of {FINE_STEP_FINAL_MM:g} mm "
            f"{numbers}: nothing to gain. `staircase.step_initial_mm` and the starting levels were not varied."
        )
    ]


def _advise_range(results: _Results) -> list[_Advice]:
    items = []
    for change in _range_changes(results):
        block = change.block
        numbers = (
            f"For a true radius within two widths of {block.max_level_mm:g} mm the share of intervals open above "
            f"goes from {_pct(change.open_above[0])} % to {_pct(change.open_above[1])} % and the RMSE from "
            f"{_mm(change.rmse_mm[0])} to {_mm(change.rmse_mm[1])} mm; for the other radii the RMSE is "
            f"{_change(change.elsewhere)}."
        )
        if change.worthwhile:
            asked = f"`blocks.{block.name}.max_level_mm` from {block.max_level_mm:g} to {change.largest_mm:g}"
            items.append(
                _Advice(
                    f"**Raise {asked}**, if targets that far from the demonstration fit on the table. {numbers} The "
                    "larger limit costs nothing when the radius is small, because the staircases then never reach "
                    "it.",
                    change=asked,
                )
            )
        else:
            items.append(_Advice(f"**Keep `blocks.{block.name}.max_level_mm = {block.max_level_mm:g}`.** {numbers}"))
    return items


def _advise_trials(results: _Results) -> list[_Advice]:
    known = [
        change
        for change in _trial_changes(results)
        if change.longer_interval is not None and change.shorter_interval is not None
    ]
    if not known:
        return []
    best = min(change.longer_interval for change in known)
    neediest = [change.block for change in known if change.longer_interval <= best + 0.03]  # within three points
    changes = "; ".join(
        f"`blocks.{block.name}.staircases` from "
        + " and ".join(str(plan.n_trials) for plan in block.staircases)
        + " trials to "
        + " and ".join(str(plan.n_trials + STAIRCASE_TRIALS_DELTA) for plan in block.staircases)
        for block in neediest
    )
    adds = "; ".join(f"{item.block.name}: {_change(item.longer_interval, 'wider', 'narrower')}" for item in known)
    cuts = "; ".join(f"{item.block.name}: {_change(item.shorter_interval, 'wider', 'narrower')}" for item in known)
    return [
        _Advice(
            "**Do not shorten a staircase, and if trials can be added, add them to "
            f"{' or '.join(block.name for block in neediest)} first** ({changes}). {STAIRCASE_TRIALS_DELTA} trials "
            f"more per staircase change the interval's width by: {adds}. {STAIRCASE_TRIALS_DELTA} fewer: {cuts}. The "
            "number of trials is the one parameter here that moves the result; whether the session has room for "
            "more is not something this simulation can weigh."
        )
    ]


def _advise_floor(results: _Results) -> list[_Advice]:
    floor = _floor_findings(results)
    if floor is None or floor.harm_rmse is None or floor.cost_interval is None:
        return []
    setting = f"`analysis.floor_values = {_values_text(FLOOR_CHECK_FLOOR_VALUES)}`"
    stated = grids_from_protocol(results.protocol["analysis"]).floor_values.tolist()
    has_floor, has_these = any(value > 0.0 for value in stated), stated == list(FLOOR_CHECK_FLOOR_VALUES)
    numbers = (
        f"With a true floor of {floor.true_floor:g}, leaving the floor out of the model makes the RMSE "
        f"{_change(floor.harm_rmse)} and shifts the estimate by {_signed(floor.harm_bias_mm - floor.kept_bias_mm)} "
        "mm. With no floor in the truth, having it in the model makes the interval "
        f"{_change(floor.cost_interval, 'wider', 'narrower')} and the RMSE {_change(floor.cost_rmse)}, and shifts "
        f"the estimate by {_signed(floor.cost_bias_mm - floor.plain_bias_mm)} mm."
    )
    if floor.worth_having and floor.cheap:
        insurance = (
            "The floor is cheap insurance against one lucky success; it is not an estimate of how often luck "
            "strikes (see the reading). For a condition in which a lucky success far out cannot happen, it costs "
            "that little and buys nothing."
        )
        if has_these:
            items = [_Advice(f"**Keep {setting}.** {numbers} {insurance}")]
        else:
            replaced = f" in place of {_values_text(stated)}" if has_floor else ""
            items = [_Advice(f"**Set {setting}{replaced}.** {numbers} {insurance}", change=setting)]
    elif not floor.worth_having and has_floor:
        items = [
            _Advice(
                f"**Take `floor_values` out of the protocol.** {numbers} The floor buys nothing measurable here.",
                change="`analysis.floor_values` taken out",
            )
        ]
    elif not floor.worth_having:
        items = [
            _Advice(f"**Leave `floor_values` out of the protocol.** {numbers} The floor buys nothing measurable here.")
        ]
    else:
        items = [_Advice(f"**{setting} is a trade to decide, not a clear gain.** {numbers}")]
    bare = [block.name for block in results.blocks.values() if block.anchors_far == 0]
    if floor.far_rmse is None or not bare:
        return items
    names = " and ".join(bare)
    numbers = (
        f"In {floor.far_block}, {FLOOR_CHECK_FAR_ANCHORS} far anchors give an RMSE {_change(floor.far_rmse)} and an "
        f"interval {_change(floor.far_interval, 'wider', 'narrower')} than {FLOOR_CHECK_FAR_ANCHORS} more staircase "
        "trials, and they "
        + ("make" if _floor_is_readable(floor.far_estimates, floor.true_floor) else "do not make")
        + " the floor readable."
    )
    if floor.far_rmse <= 1.0 - MEANINGFUL_CHANGE:
        items.append(_Advice(f"**Give {names} far anchors.** {numbers}", change=f"far anchors for {names}"))
    elif floor.far_rmse >= 1.0 + MEANINGFUL_CHANGE:
        items.append(
            _Advice(
                f"**Do not add far anchors to {names} for the estimate's sake.** {numbers} Spend spare trials on "
                "the staircases. Far anchors remain what they were meant to be: a direct count of successes far out."
            )
        )
    else:
        items.append(
            _Advice(
                f"**Far anchors for {names} are optional.** {numbers} They neither help nor hurt the estimate; add "
                "them only for the direct count of successes far out."
            )
        )
    if floor.far_unprotected is not None and floor.far_unprotected >= 1.0 + MEANINGFUL_CHANGE:
        items.append(
            _Advice(
                "**Do not run far anchors with a model that has no floor.** With a floor in the truth and none in "
                f"the model, far anchors make the RMSE {_change(floor.far_unprotected)} in {floor.far_block}. If "
                "`floor_values` is not adopted, the far anchors of the blocks that have them should be reported as "
                "counts only and left out of the fit."
            )
        )
    return items


def _values_text(values: Sequence[float]) -> str:
    """'[0.00, 0.05, 0.10]': a list of rates as it would stand in the protocol file."""
    return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"


def _advise_pairs(results: _Results) -> list[_Advice]:
    if not results.power or results.pairs is None:
        return []
    lows = [rate for rate in (0.4, 0.5) if rate in results.power_rates]
    pairs, more = results.pairs, max(results.pairs_options)
    now = f"{' and '.join(_tells_apart(pairs, low) for low in lows)} at least {_pct(POWER_TARGET)} % of the time"
    then = " and ".join(_tells_apart(more, low) for low in lows)
    report_anyway = (
        "If the two policies turn out closer than that, the paper reports the counts and their intervals and says "
        "that the test could not have decided."
    )
    if more > pairs and _more_pairs_matter(pairs, more, lows):
        return [
            _Advice(
                f"**With {pairs} pairs the paired test can show very little; {more} pairs would show more.** With "
                f"{pairs} pairs it tells {now}; with {more} pairs it tells {then}. Whether {more - pairs} more pairs "
                f"fit in the session is not something this report can weigh. {report_anyway}"
            )
        ]
    return [
        _Advice(
            f"**Keep the paired test at {pairs} pairs and write down now what it can show.** It tells {now}, and "
            f"smaller differences mostly not. With {more} pairs it tells {then}: at most one step of 0.05 better. "
            f"{report_anyway}"
        )
    ]


def _more_pairs_matter(pairs: int, more: int, lower_rates: Sequence[float]) -> bool:
    """The report's rule: more pairs matter when, from some lower rate, they bring the rate that can be told apart
    down by more than one step of 0.05, or make one reachable at all."""
    for low in lower_rates:
        now, then = rate_needed(pairs, low), rate_needed(more, low)
        if then is not None and (now is None or now - then > 0.05 + 1e-9):
            return True
    return False


def _advise_edge_rule() -> _Advice:
    return _Advice(
        "**Write the rule for an open interval into `protocol/protocol.md` before the freeze.** An equal-tailed interval "
        "never ends exactly on the last value of the grid, so \"the interval reaches the edge\" needs a rule. The "
        f"one used here and in the analysis code (`psychometric.EDGE_RATIO = {EDGE_RATIO:g}`): the interval is open "
        "on a side when the posterior at that end of the grid is at least half of what it is at the interval's own "
        "bound on that side."
    )


def _limits(results: _Results, info: RunInfo) -> list[str]:
    return [
        "## What this simulation does not test",
        "",
        *_bullets(
            [
                "**The shape of the curve.** Every truth is a curve of the model's own family. A real curve that is "
                "lopsided, or whose plateau wears away with distance, is not covered.",
                "**Independence.** Each simulated trial depends on its distance only. Nothing drifts during a "
                "block, no trial affects the next, and neither the direction from the demonstration nor the place "
                "on the table matters. Under these assumptions the order of the trials within a block is "
                "irrelevant, so the seeded interleaving of the protocol was not simulated.",
                f"**The placement error.** It is normal with a standard deviation of {info.placement_sd_mm:g} mm "
                "along the distance and is cut off at 0. A real placement at 0 mm misses by a small distance in "
                "some direction and never lands exactly on 0.",
                "**Trials from other blocks.** Each block is fitted alone here. An analysis that pools a "
                "condition's trials from several blocks (the paired test also yields trials at measured distances) "
                "will have narrower intervals than these.",
                f"**Truths outside the sweep.** Radii from {results.radii[0]:g} to {results.radii[-1]:g} mm, widths "
                f"from {results.widths[0]:g} to {results.widths[-1]:g} mm, lapse rates up to "
                f"{max(truth.lapse for truth in results.truths):g}. The staircases' starting levels and the "
                "initial step were not varied.",
                f"**Other floors.** Only a model without a floor and one with the floor values "
                f"{_list(FLOOR_CHECK_FLOOR_VALUES)} were compared, against truths with a floor of "
                f"{_list(sorted({truth.floor for truth in results.floor_truths}, reverse=True))}, at "
                f"{len(results.floor_truths)} truths in all."
                if results.floor_truths
                else "**Other floors.** The floor check was not run.",
                "**The paired test.** Its table assumes that the two policies' outcomes at a placement are "
                "independent. Placements that are hard for both policies would change the numbers.",
                "**Trials that do not count.** Void trials, scoring mistakes and placements outside tolerance are "
                "not simulated.",
            ]
        ),
    ]
