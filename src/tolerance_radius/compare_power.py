"""The comparison rule, simulated: what two blocks of trials can tell apart, before any trial is run.

Prediction 2 of the protocol says that the clustered policy's tolerance radius is larger than the replay's, and gives
the rule by which the trials decide (`protocol/protocol.md`, section 9): from the two block fits, taken as independent,
"larger" means a posterior probability of at least 0.95 that the policy's radius exceeds the replay's, "smaller" at
most 0.05, and anything between is "not told apart".

This module plays the two blocks as the protocol file states them (the clustered policy's threshold block and the
replay's) against two success curves that are assumed true, fits each block as the analysis fits it, applies the rule
with the analysis's own function (`report.estimates.radius_difference`), and counts how often each verdict comes
out: when the two radii are equal, and when the policy's is larger by a stated amount.

    python -m tolerance_radius.compare_power --out analysis/simulation

writes `comparison_rule.csv` (one row for every pair of assumed curves) and the report `comparison_rule.md`.

**Nothing here is a measurement.** Every block is played against a curve assumed true, as in `simulate`: the next
distance of a staircase comes from the protocol's own rule, the cube lands at the requested distance plus a
placement error, and a trial succeeds with the probability the assumed curve gives where the cube landed.

Standard library and NumPy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius import config, simulate
from tolerance_radius.psychometric import AnalysisGrids, grids_from_protocol
from tolerance_radius.report import estimates
from tolerance_radius.seeds import derived_rng

#: The replay's radius in every simulated case, and how far beyond it the policy's radius lies.
REPLAY_R50_MM = 15.0
DIFFERENCES_MM = (0.0, 3.0, 5.0, 8.0, 10.0, 15.0)
#: The width of both curves, and their lapse.
WIDTHS_MM = (1.0, 3.0, 6.0)
LAPSE = 0.0
#: The command's defaults: simulated pairs of blocks for every case, and the seed.
DEFAULT_PAIRS = 300
DEFAULT_SEED = simulate.DEFAULT_SEED
#: The conditions whose threshold blocks the rule compares: the first minus the second.
POLICY, REPLAY = "act_clustered", "replay"
#: A verdict counts as reliable in the report's reading when it comes out at least this often.
RELIABLE_SHARE = 0.8
VERDICTS = (estimates.VERDICT_LARGER, estimates.VERDICT_NOT_TOLD_APART, estimates.VERDICT_SMALLER)

CSV_COLUMNS = (
    "policy_block",
    "replay_block",
    "replay_r50_mm",
    "policy_r50_mm",
    "difference_true_mm",
    "width_mm",
    "lapse",
    "pairs",
    "n_larger",
    "n_not_told_apart",
    "n_smaller",
    "share_larger",
    "share_not_told_apart",
    "share_smaller",
    "mean_difference_mm",
    "mean_interval_width_mm",
    "interval_coverage",
)


@dataclass(frozen=True, slots=True)
class RuleCase:
    """Two success curves assumed true: the replay's and the policy's, of one width and one lapse."""

    replay_r50_mm: float
    policy_r50_mm: float
    width_mm: float
    lapse: float = LAPSE

    @property
    def difference_mm(self) -> float:
        """The policy's radius minus the replay's."""
        return self.policy_r50_mm - self.replay_r50_mm

    @property
    def policy_truth(self) -> simulate.Truth:
        return simulate.Truth(self.policy_r50_mm, self.width_mm, self.lapse)

    @property
    def replay_truth(self) -> simulate.Truth:
        return simulate.Truth(self.replay_r50_mm, self.width_mm, self.lapse)


@dataclass(frozen=True, slots=True)
class RuleRow:
    """What the rule said over many simulated pairs of blocks for one case."""

    case: RuleCase
    policy_block: str
    replay_block: str
    pairs: int
    #: How many pairs gave each verdict.
    n_larger: int
    n_not_told_apart: int
    n_smaller: int
    #: The mean, over the pairs, of the estimated difference (the posterior median of policy minus replay).
    mean_difference_mm: float
    #: The mean width of the difference's interval, and the share of the intervals that hold the true difference.
    mean_interval_width_mm: float
    interval_coverage: float

    def share(self, verdict: str) -> float:
        """The share of the pairs that gave a verdict."""
        counts = dict(zip(VERDICTS, (self.n_larger, self.n_not_told_apart, self.n_smaller), strict=True))
        return counts[verdict] / self.pairs


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What a run was made with, for the head of its report."""

    seed: int
    pairs: int
    placement_sd_mm: float
    #: The protocol file by its bare name, its version and status, and the first 12 hex digits of its SHA-256.
    protocol_file: str
    protocol_version: str
    protocol_status: str
    protocol_digest: str


def cases() -> list[RuleCase]:
    """The cases of the report: every width with every difference, the replay's radius fixed."""
    return [
        RuleCase(REPLAY_R50_MM, REPLAY_R50_MM + difference, width)
        for width in WIDTHS_MM
        for difference in DIFFERENCES_MM
    ]


def compared_blocks(protocol: Mapping[str, Any]) -> tuple[simulate.BlockDesign, simulate.BlockDesign]:
    """The two blocks the rule compares, as the protocol file states them: the clustered policy's threshold block
    and the replay's."""
    designs = {design.name: design for design in simulate.blocks_from_protocol(dict(protocol))}
    found = []
    for condition in (POLICY, REPLAY):
        names = [name for name, entry in protocol["blocks"].items() if entry.get("condition") == condition]
        if len(names) != 1 or names[0] not in designs:
            raise ValueError(f"the protocol needs one threshold block of {condition}, and has {names}")
        found.append(designs[names[0]])
    return found[0], found[1]


def simulate_pair(
    case: RuleCase,
    policy_block: simulate.BlockDesign,
    replay_block: simulate.BlockDesign,
    steps: simulate.StaircaseSteps,
    rng: np.random.Generator,
    *,
    grids: AnalysisGrids,
    level: float = 0.95,
    placement_sd_mm: float = simulate.DEFAULT_PLACEMENT_SD_MM,
) -> dict[str, Any]:
    """One simulated pair of blocks, compared as the analysis compares them.

    Each block is played against its own curve (`simulate.simulate_block`) with nothing shared between the two, and
    fitted by itself on the protocol's grids. The result is what `report.estimates.radius_difference` says of the two
    fits: the difference's median and interval, the probability that the policy's radius is the larger, the verdict.
    """
    posteriors = []
    for truth, block in ((case.policy_truth, policy_block), (case.replay_truth, replay_block)):
        played = simulate.simulate_block(
            truth, block, steps, rng, grids=grids, placement_sd_mm=placement_sd_mm, level=level
        )
        posteriors.append(estimates.fit(played.distances_mm, played.successes, grids))
    return estimates.radius_difference(posteriors[0], posteriors[1], level)


def evaluate_case(
    case: RuleCase,
    policy_block: simulate.BlockDesign,
    replay_block: simulate.BlockDesign,
    steps: simulate.StaircaseSteps,
    rng: np.random.Generator,
    *,
    grids: AnalysisGrids,
    pairs: int,
    level: float = 0.95,
    placement_sd_mm: float = simulate.DEFAULT_PLACEMENT_SD_MM,
) -> RuleRow:
    """Simulate `pairs` pairs of blocks for one case and count the verdicts."""
    if pairs < 1:
        raise ValueError("at least one pair of blocks is needed")
    compared = [
        simulate_pair(
            case, policy_block, replay_block, steps, rng, grids=grids, level=level, placement_sd_mm=placement_sd_mm
        )
        for _ in range(pairs)
    ]
    verdicts = [found["verdict"] for found in compared]
    inside = [found["lo_mm"] <= case.difference_mm <= found["hi_mm"] for found in compared]
    return RuleRow(
        case=case,
        policy_block=policy_block.name,
        replay_block=replay_block.name,
        pairs=pairs,
        n_larger=verdicts.count(estimates.VERDICT_LARGER),
        n_not_told_apart=verdicts.count(estimates.VERDICT_NOT_TOLD_APART),
        n_smaller=verdicts.count(estimates.VERDICT_SMALLER),
        mean_difference_mm=math.fsum(found["median_mm"] for found in compared) / pairs,
        mean_interval_width_mm=math.fsum(found["hi_mm"] - found["lo_mm"] for found in compared) / pairs,
        interval_coverage=sum(inside) / pairs,
    )


def run(
    protocol: Mapping[str, Any],
    *,
    pairs: int = DEFAULT_PAIRS,
    seed: int = DEFAULT_SEED,
    rule_cases: Sequence[RuleCase] | None = None,
    placement_sd_mm: float = simulate.DEFAULT_PLACEMENT_SD_MM,
) -> list[RuleRow]:
    """One row for every case: how often the rule gave each verdict over `pairs` simulated pairs of blocks.

    Each case has its own stream of random numbers, derived from the seed and from what the case is, so a case's
    numbers do not depend on which other cases are run.
    """
    grids = grids_from_protocol(protocol["analysis"])
    level = float(protocol["analysis"]["interval_level"])
    steps = simulate.steps_from_protocol(dict(protocol))
    policy_block, replay_block = compared_blocks(protocol)
    rows = []
    for case in cases() if rule_cases is None else rule_cases:
        rng = derived_rng(
            seed,
            "comparison-rule",
            policy_block.name,
            replay_block.name,
            f"{case.replay_r50_mm:g}",
            f"{case.policy_r50_mm:g}",
            f"{case.width_mm:g}",
            f"{case.lapse:g}",
        )
        rows.append(
            evaluate_case(
                case,
                policy_block,
                replay_block,
                steps,
                rng,
                grids=grids,
                pairs=pairs,
                level=level,
                placement_sd_mm=placement_sd_mm,
            )
        )
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------------


def write_csv(path: str | Path, rows: Sequence[RuleRow]) -> None:
    """Every row, one line each (`CSV_COLUMNS`)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            case = row.case
            writer.writerow(
                [
                    row.policy_block,
                    row.replay_block,
                    f"{case.replay_r50_mm:g}",
                    f"{case.policy_r50_mm:g}",
                    f"{case.difference_mm:g}",
                    f"{case.width_mm:g}",
                    f"{case.lapse:g}",
                    row.pairs,
                    row.n_larger,
                    row.n_not_told_apart,
                    row.n_smaller,
                    *(f"{row.share(verdict):.4f}" for verdict in VERDICTS),
                    f"{row.mean_difference_mm:.4f}",
                    f"{row.mean_interval_width_mm:.4f}",
                    f"{row.interval_coverage:.4f}",
                ]
            )


def _pct(share: float) -> str:
    return f"{100.0 * share:.0f}"


def _signed(value: float) -> str:
    """A difference in millimetres with its sign, and no sign on one that rounds to zero."""
    text = f"{value:+.1f}"
    return "0.0" if text in ("+0.0", "-0.0") else text


def _listed(values: Sequence[float]) -> str:
    """`3, 5 and 8`."""
    words = [f"{value:g}" for value in values]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


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


def _table(caption: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A table under its caption. The caption says where the numbers come from."""
    lines = [*_paragraph(caption), "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


def _block_text(block: simulate.BlockDesign) -> str:
    staircases = " and ".join(f"from {plan.start_level_mm:g} mm ({plan.n_trials} trials)" for plan in block.staircases)
    far = f", {block.anchors_far} trials at {block.far_level_mm:g} mm" if block.anchors_far else ""
    return (
        f"{block.name}, {block.n_trials} trials ({block.anchors_zero} at 0 mm, staircases {staircases} up to "
        f"{block.max_level_mm:g} mm{far})"
    )


def _standard_error(share: float, pairs: int) -> float:
    """One standard error of a simulated share, in percentage points."""
    return 100.0 * math.sqrt(share * (1.0 - share) / pairs)


def smallest_reliable_difference(rows: Sequence[RuleRow], width_mm: float) -> float | None:
    """The smallest simulated difference from which the rule says "larger" reliably (`RELIABLE_SHARE`) for a width:
    at that difference and at every larger one that was simulated. None when the largest one is not reliable."""
    found = None
    own = [row for row in rows if row.case.width_mm == width_mm and row.case.difference_mm > 0.0]
    for row in sorted(own, key=lambda row: -row.case.difference_mm):
        if row.share(estimates.VERDICT_LARGER) < RELIABLE_SHARE:
            break
        found = row.case.difference_mm
    return found


def _by_difference(rows: Sequence[RuleRow], width_mm: float) -> list[RuleRow]:
    return sorted((row for row in rows if row.case.width_mm == width_mm), key=lambda row: row.case.difference_mm)


def render_report(protocol: Mapping[str, Any], rows: Sequence[RuleRow], info: RunInfo) -> str:
    """The text of `comparison_rule.md`, written from the rows by fixed rules."""
    policy_block, replay_block = compared_blocks(protocol)
    level = f"{100.0 * float(protocol['analysis']['interval_level']):g}"
    widths = sorted({row.case.width_mm for row in rows})
    differences = sorted({row.case.difference_mm for row in rows if row.case.difference_mm > 0.0})
    replay_radii = sorted({row.case.replay_r50_mm for row in rows})
    lapses = sorted({row.case.lapse for row in rows})
    larger, smaller = estimates.VERDICT_PROBABILITY, 1.0 - estimates.VERDICT_PROBABILITY
    said_larger, said_smaller = estimates.VERDICT_LARGER, estimates.VERDICT_SMALLER
    lines = [
        "# The comparison rule: what two blocks of trials can tell apart",
        "",
        "**Every number in this report is simulated. Nothing here is a measurement.**",
        "",
        *_paragraph(
            f"Written by `python -m tolerance_radius.compare_power`. Seed {info.seed}; {info.pairs} simulated pairs of "
            f"blocks for every row. Protocol file `{info.protocol_file}`: version {info.protocol_version}, status "
            f"{info.protocol_status}, SHA-256 `{info.protocol_digest}`. The rows behind the tables are in "
            "`comparison_rule.csv`."
        ),
        "## The question",
        "",
        *_paragraph(
            "Prediction 2 of the protocol is that the clustered policy's tolerance radius is larger than the "
            "replay's. The rule that decides it (`protocol/protocol.md`, section 9): from the two block fits, taken as "
            f'independent, "larger" means a posterior probability of at least {larger:g} that the policy\'s radius '
            f'exceeds the replay\'s, "smaller" means at most {smaller:.2f}, and anything between is "not told '
            'apart". Before any trial is run: with the two blocks as the protocol plans them, how large must a true '
            'difference be for the rule to say "larger", and how often does it give a verdict when the two radii '
            "are equal?"
        ),
        "## What was simulated, and what was assumed",
        "",
        *_bullets(
            [
                "**The two blocks**, as the protocol file states them (read from the file). The policy's: "
                f"{_block_text(policy_block)}. The replay's: {_block_text(replay_block)}.",
                f"**The curves assumed true.** The replay's radius is {_listed(replay_radii)} mm in every case. The "
                f"policy's radius is equal to it, or larger by {_listed(differences)} mm. Both curves have the same "
                f"width ({_listed(widths)} mm), a lapse of {_listed(lapses)} and no floor. A width of 1 mm is a "
                "cliff: success goes from 88 % to 12 % of its plateau within 4 mm.",
                "**A trial**, as in the procedure simulation (`procedure.md`). The next distance of a staircase comes "
                "from the protocol's own rule. The cube lands at the requested distance plus a placement error "
                f"(normal, standard deviation {info.placement_sd_mm:g} mm, and not below 0). The trial succeeds with "
                "the probability its curve gives at the distance achieved.",
                "**The two blocks are independent.** Each is played with its own random numbers. Nothing is shared "
                "between them, and nothing drifts. The rule assumes the same.",
                "**The fit and the rule are the analysis's own.** Each block is fitted by itself on the protocol's "
                "grid (`psychometric.fit_threshold`). The two fits are compared by "
                "`report.estimates.radius_difference`: every pair of grid values is weighted by the product of the "
                f"two posteriors, a tie on the grid counts half for each side, and the difference gets its {level} % "
                "interval and the verdict.",
            ]
        ),
        *_paragraph(
            f"**How exact the simulation is.** With {info.pairs} pairs in a row, a share near 50 % is known to about "
            f"{_standard_error(0.5, info.pairs):.1f} points (one standard error), and a share near 5 % or 95 % to "
            f"about {_standard_error(0.05, info.pairs):.1f} points. A smaller difference between two rows is noise."
        ),
        "## 1. How often each verdict comes out",
        "",
    ]
    verdict_rows = [
        [
            f"{width:g}",
            f"{row.case.difference_mm:g}",
            _pct(row.share(said_larger)),
            _pct(row.share(estimates.VERDICT_NOT_TOLD_APART)),
            _pct(row.share(said_smaller)),
        ]
        for width in widths
        for row in _by_difference(rows, width)
    ]
    lines += _table(
        "**Table 1.** Share of the simulated pairs of blocks with each verdict, in %, by the width of the two curves "
        "and by the true difference (the policy's radius minus the replay's). Simulated.",
        ["Width (mm)", "True difference (mm)", "larger", "not told apart", "smaller"],
        verdict_rows,
    )
    lines += ["## 2. The difference that is reported beside the verdict", ""]
    estimate_rows = [
        [
            f"{width:g}",
            f"{row.case.difference_mm:g}",
            _signed(row.mean_difference_mm),
            f"{row.mean_interval_width_mm:.1f}",
            _pct(row.interval_coverage),
        ]
        for width in widths
        for row in _by_difference(rows, width)
    ]
    lines += _table(
        "**Table 2.** The estimated difference (the posterior median of the policy's radius minus the replay's), the "
        f"width of its {level} % interval, and how often that interval holds the true difference. Means over the "
        "pairs. Simulated.",
        [
            "Width (mm)",
            "True difference (mm)",
            "Estimated difference (mm)",
            "Interval width (mm)",
            "Contains the truth (%)",
        ],
        estimate_rows,
    )

    reading = []
    equal = sorted((row for row in rows if row.case.difference_mm == 0.0), key=lambda row: row.case.width_mm)
    if equal:
        told = "; ".join(
            f'at width {row.case.width_mm:g} mm, "larger" in {_pct(row.share(said_larger))} % of pairs and "smaller" '
            f"in {_pct(row.share(said_smaller))} %"
            for row in equal
        )
        worst = max(row.share(said_larger) + row.share(said_smaller) for row in equal)
        reading.append(
            '**When the two radii are equal**, the right answer is "not told apart". The rule gave a verdict all the '
            f"same: {told}. At most {_pct(worst)} % of pairs got a verdict for a difference that is not there."
        )
    for width in widths:
        own = [row for row in _by_difference(rows, width) if row.case.difference_mm > 0.0]
        if not own:
            continue
        shares = ", ".join(f"{_pct(row.share(said_larger))} % at {row.case.difference_mm:g} mm" for row in own[1:])
        first = f"{_pct(own[0].share(said_larger))} % of pairs at a true difference of {own[0].case.difference_mm:g} mm"
        reliable = smallest_reliable_difference(rows, width)
        if reliable is None:
            reached = (
                f"No simulated difference up to {own[-1].case.difference_mm:g} mm gives that verdict in "
                f"{_pct(RELIABLE_SHARE)} % of pairs."
            )
        else:
            reached = (
                f"The smallest simulated difference from which it does so in at least {_pct(RELIABLE_SHARE)} % of "
                f"pairs is {reliable:g} mm."
            )
        told = ", ".join(part for part in (first, shares) if part)
        reading.append(f'**Both curves of width {width:g} mm.** The rule says "larger" in {told}. {reached}')
    beyond = [row for row in rows if row.case.difference_mm > 0.0]
    if beyond:
        worst = max(row.share(said_smaller) for row in beyond)
        reading.append(
            f'**The wrong way round.** With the policy\'s radius truly the larger, the rule said "smaller" in at most '
            f"{_pct(worst)} % of pairs."
        )
    if rows:
        covered = [row.interval_coverage for row in rows]
        reading.append(
            f"**The interval of the difference** held the true difference in {_pct(min(covered))} % of pairs in the "
            f"lowest row and in {_pct(max(covered))} % in the highest."
        )
    lines += ["## Reading", "", *_bullets(reading)]
    lines += [
        "## What this simulation does not show",
        "",
        *_bullets(
            [
                f"**Other radii.** The replay's radius is {_listed(replay_radii)} mm in every case. A radius near 0 "
                "mm, or near the largest distance a block tests, is located less well (`procedure.md`, section 2), "
                "and the rule then tells less apart than here.",
                "**Curves that differ in more than their radius.** Both curves have one width, no lapse and no "
                "floor. A replay that is a cliff against a policy that is a slope was not simulated, nor was a "
                "plateau below 1 (`lapse_values.md` has what a low plateau does to one radius).",
                "**The shape of the curve.** Every truth is a curve of the model's own family.",
                "**What the two blocks share.** They are run interleaved in one session. Whatever the two share "
                "there (the hour, the servos' temperature, the state of the cameras) is not in this simulation.",
                "**Trials that do not count.** Void trials and scoring mistakes are not simulated.",
                "**The other comparison.** The spread policy's block against the clustered policy's is another pair "
                "of blocks, run on another day, and was not simulated.",
            ]
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tolerance_radius.compare_power --out <dir> [--protocol <path>] [--pairs N] [--seed S]`."""
    parser = argparse.ArgumentParser(
        prog="python -m tolerance_radius.compare_power",
        description="Simulate the rule that compares the clustered policy's radius with the replay's, and write "
        "comparison_rule.csv and the report comparison_rule.md into the output directory.",
    )
    parser.add_argument("--out", required=True, type=Path, help="the directory to write into; created if missing")
    parser.add_argument(
        "--protocol", type=Path, default=config.DEFAULT_PROTOCOL_PATH, help="the protocol file (config/protocol.toml)"
    )
    parser.add_argument(
        "--pairs",
        type=_at_least(1),
        default=DEFAULT_PAIRS,
        help=f"simulated pairs of blocks for every case (default {DEFAULT_PAIRS})",
    )
    parser.add_argument(
        "--seed", type=_at_least(0), default=DEFAULT_SEED, help=f"seed of the random numbers (default {DEFAULT_SEED})"
    )
    arguments = parser.parse_args(argv)
    protocol = config.load_protocol(arguments.protocol)
    info = RunInfo(
        seed=arguments.seed,
        pairs=arguments.pairs,
        placement_sd_mm=simulate.DEFAULT_PLACEMENT_SD_MM,
        protocol_file=arguments.protocol.name,
        protocol_version=str(protocol["protocol"]["version"]),
        protocol_status=str(protocol["protocol"].get("status", "")),
        protocol_digest=hashlib.sha256(arguments.protocol.read_bytes()).hexdigest()[:12],
    )
    rows = run(protocol, pairs=info.pairs, seed=info.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.out / "comparison_rule.csv", rows)
    (arguments.out / "comparison_rule.md").write_text(render_report(protocol, rows, info), encoding="utf-8")
    print(f"wrote comparison_rule.md and comparison_rule.csv: {len(rows)} cases, {info.pairs} pairs of blocks each")
    return 0


def _at_least(smallest: int) -> Any:
    """An argument type for argparse: a whole number that is at least `smallest`."""

    def parse(text: str) -> int:
        value = int(text)
        if value < smallest:
            raise argparse.ArgumentTypeError(f"must be at least {smallest}, got {value}")
        return value

    return parse


if __name__ == "__main__":
    raise SystemExit(main())
