"""The procedure simulation: how well a block of trials estimates the tolerance radius, before any trial is run.

A block of the protocol is a few anchors at distance 0, two interleaved staircases, and for one condition a few anchors
far out. This module plays such a block against a success curve that is assumed true, fits the result exactly as the
analysis will (`psychometric.fit_threshold`), and repeats that many times, to say for each assumed truth how far the
estimate lands from it, how wide its interval is, and how often the interval contains the truth.

What is simulated, and what is not
----------------------------------
* The next distance of a staircase comes from `staircase.next_level`, the rule the planner uses at the rig.
* The cube lands at the requested distance plus a placement error (normal, 1 mm standard deviation by default), and
  not below 0. The fit uses the achieved distance, as the analysis does.
* A trial succeeds with the probability the assumed curve gives at the achieved distance, independently of every other
  trial. Nothing drifts, nothing depends on direction or on where on the table the cube is.
* The order in which a block's trials are run does not matter under these assumptions, so anchors are simulated first
  and the staircases then take turns.

Three questions are put to it. `python -m tolerance_radius.simulate` runs them, writes every row to CSV files, and
has `procedure_report` write the report:

* the sweep (`run_sweep`): every block of the protocol as written and four other designs, over a range of truths;
* the floor check (`run_floor_check`): what a chance of lucky successes far out does to a fit that ignores it, and
  what allowing for it costs when there is none;
* the paired comparison (`paired_power`): how large a difference between two policies the exact McNemar test can
  show with the protocol's number of pairs. This one is an exact sum, not a simulation.

Every number this module produces is simulated or computed under stated assumptions. None is a measurement.

Standard library and NumPy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius.config import load_protocol
from tolerance_radius.intervals import mcnemar_exact
from tolerance_radius.psychometric import (
    AnalysisGrids,
    ThresholdPosterior,
    ThresholdSummary,
    distance_at_probability,
    fit_threshold,
    grids_from_protocol,
)
from tolerance_radius.staircase import StaircaseParams, next_level

#: The truths of the sweep: every combination of these.
SWEEP_R50_MM = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
SWEEP_WIDTH_MM = (1.0, 3.0, 6.0, 10.0)
SWEEP_LAPSE = (0.0, 0.05)

#: The designs compared for every block, in the order they are reported (see `design_variants`).
DESIGNS = ("protocol", "constant", "fine_step", "longer", "shorter")
#: A further design, for a block whose largest level is below the largest of the protocol: the same block with that
#: larger limit. It answers what a threshold near the block's own limit costs.
WIDER_RANGE = "wider_range"
#: The final step of the "fine_step" design.
FINE_STEP_FINAL_MM = 2.5
#: How many trials each staircase gains in the "longer" design and loses in the "shorter" one.
STAIRCASE_TRIALS_DELTA = 4

#: Standard deviation of the placement error along the distance, in millimetres.
DEFAULT_PLACEMENT_SD_MM = 1.0

#: The success rates of the paired test's table, and the chance of telling two rates apart that counts as enough.
POWER_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
POWER_TARGET = 0.8

#: The floor check (`run_floor_check`): truths with a floor and without, each fitted without a floor in the model and
#: with these floor values.
FLOOR_CHECK_TRUE_FLOORS = (0.08, 0.0)
FLOOR_CHECK_FLOOR_VALUES = (0.0, 0.05, 0.10)
FLOOR_CHECK_R50_MM = (15.0, 30.0)
FLOOR_CHECK_WIDTH_MM = (1.0, 3.0, 6.0)
FLOOR_CHECK_LAPSE = 0.05
#: How many far anchors a block without any is given in the check.
FLOOR_CHECK_FAR_ANCHORS = 4
#: The names of the two fits of the check.
FIT_NO_FLOOR, FIT_FLOOR = "no_floor", "floor"

#: The command's defaults: the seed, and the number of simulated blocks for every row.
DEFAULT_SEED = 20261005
DEFAULT_REPLICATIONS = 300
#: The paired test's table covers the protocol's number of pairs and this many more and fewer.
PAIRS_DELTA = 4

#: What an anchor or a fixed-level trial is called in a simulated block; a staircase trial carries its staircase's id.
KIND_ZERO, KIND_FAR, KIND_FIXED = "zero", "far", "fixed"


@dataclass(frozen=True, slots=True)
class Truth:
    """A success curve assumed true in a simulation (the model of `psychometric`)."""

    r50_mm: float
    width_mm: float
    lapse: float = 0.0
    #: The chance of a lucky success far from every demonstration.
    floor: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.r50_mm):
            raise ValueError("r50_mm must be a finite number")
        if not self.width_mm > 0.0:
            raise ValueError("width_mm must be positive")
        if not (self.lapse >= 0.0 and self.floor >= 0.0 and self.lapse + self.floor < 1.0):
            raise ValueError("lapse and floor must be at least 0 and add up to less than 1")


@dataclass(frozen=True, slots=True)
class StaircasePlan:
    """One staircase of a block: its name, where it starts and how many trials it runs."""

    id: str
    start_level_mm: float
    n_trials: int


@dataclass(frozen=True, slots=True)
class StaircaseSteps:
    """The staircases' steps: large until the first reversal, small after it."""

    step_initial_mm: float
    step_final_mm: float

    def __post_init__(self) -> None:
        if not (self.step_initial_mm > 0.0 and self.step_final_mm > 0.0):
            raise ValueError("steps are positive")


@dataclass(frozen=True, slots=True)
class BlockDesign:
    """Where the trials of one block are requested."""

    name: str
    #: Trials at distance 0: an exact demonstration start.
    anchors_zero: int
    staircases: tuple[StaircasePlan, ...]
    #: No distance outside these limits is requested by a staircase.
    min_level_mm: float
    max_level_mm: float
    #: Trials at `far_level_mm`, to confirm that success has gone.
    anchors_far: int = 0
    far_level_mm: float | None = None
    #: Distances requested whatever the outcomes (the method of constant stimuli).
    fixed_levels_mm: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "staircases", tuple(self.staircases))
        object.__setattr__(self, "fixed_levels_mm", tuple(float(level) for level in self.fixed_levels_mm))
        if self.anchors_zero < 0 or self.anchors_far < 0:
            raise ValueError("a block cannot have a negative number of anchors")
        if not 0.0 <= self.min_level_mm <= self.max_level_mm:
            raise ValueError("the level limits must satisfy 0 <= min <= max")
        if self.anchors_far > 0 and (self.far_level_mm is None or self.far_level_mm < 0.0):
            raise ValueError("far anchors need the distance they are placed at")
        if any(not (math.isfinite(level) and level >= 0.0) for level in self.fixed_levels_mm):
            raise ValueError("a fixed level is a finite distance, zero or more")
        names = [plan.id for plan in self.staircases]
        if len(set(names)) != len(names) or set(names) & {KIND_ZERO, KIND_FAR, KIND_FIXED}:
            raise ValueError(f"staircases need distinct names other than {KIND_ZERO!r}, {KIND_FAR!r}, {KIND_FIXED!r}")
        for plan in self.staircases:
            if plan.n_trials < 1 or not self.min_level_mm <= plan.start_level_mm <= self.max_level_mm:
                raise ValueError(f"staircase {plan.id!r} needs at least one trial and a start within the limits")

    @property
    def n_trials(self) -> int:
        """The number of trials the block runs."""
        in_staircases = sum(plan.n_trials for plan in self.staircases)
        return self.anchors_zero + self.anchors_far + len(self.fixed_levels_mm) + in_staircases


@dataclass(frozen=True, slots=True, eq=False)
class SimulatedBlock:
    """One simulated block: the truth it was played against, its trials in the order simulated, and the estimate."""

    truth: Truth
    #: The fit of the block's trials, as the analysis reports it.
    summary: ThresholdSummary
    #: For each trial `"zero"`, `"far"`, `"fixed"` or the id of its staircase.
    kinds: tuple[str, ...]
    #: Requested distances.
    levels_mm: np.ndarray
    #: Achieved distances: the request plus the placement error, not below 0.
    distances_mm: np.ndarray
    successes: np.ndarray


@dataclass(frozen=True, slots=True)
class DesignPerformance:
    """How a block design estimates the tolerance radius for one assumed truth, over many simulated blocks."""

    truth: Truth
    block: str
    n_trials: int
    replications: int
    #: Mean of (posterior median - true r50).
    bias_mm: float
    #: Root-mean-square of (posterior median - true r50).
    rmse_mm: float
    mean_interval_width_mm: float
    #: Share of the intervals that contain the true r50.
    interval_coverage: float
    #: Share of the intervals flagged as reaching the lower end of the r50 grid, the upper end, and either.
    lower_edge_rate: float
    upper_edge_rate: float
    any_edge_rate: float


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One row of the sweep: a design, and how it did for one block and one truth."""

    design: str
    performance: DesignPerformance


@dataclass(frozen=True, slots=True)
class FloorCheckRow:
    """One row of the floor check: a block, a truth, and one of the two fits of the same simulated blocks."""

    #: `"as_written"`, `"far_anchors"` (far anchors added) or `"longer_staircases"` (as many trials added to the
    #: staircases instead).
    variant: str
    #: `"no_floor"` or `"floor"`.
    fit: str
    performance: DesignPerformance
    #: The posterior mean of the floor, averaged over the simulated blocks; 0 for the fit without a floor.
    mean_floor_estimate: float


@dataclass(frozen=True, slots=True)
class PowerRow:
    """One entry of the paired test's table: the chance that it tells two success rates apart."""

    pairs: int
    p_first: float
    p_second: float
    power: float


# --------------------------------------------------------------------------------------------------------------------
# One block, and many
# --------------------------------------------------------------------------------------------------------------------


def simulate_block(
    truth: Truth,
    block: BlockDesign,
    steps: StaircaseSteps,
    rng: np.random.Generator,
    *,
    grids: AnalysisGrids,
    placement_sd_mm: float = DEFAULT_PLACEMENT_SD_MM,
    level: float = 0.95,
) -> SimulatedBlock:
    """Play one block against an assumed truth and fit it as the analysis will, on the given grids.

    Everything random about a trial is drawn before the block starts: how far the placement lands from the request,
    and the trial's own limit, the distance short of which it succeeds. That limit is the distance at which the true
    curve equals a uniform random number, so a trial at any distance succeeds with exactly the curve's probability
    there (`psychometric.distance_at_probability`).
    """
    kinds, levels, distances, successes = _simulate_trials(truth, block, steps, rng, placement_sd_mm)
    return SimulatedBlock(
        truth=truth,
        summary=_fit(distances, successes, grids).summary(level),
        kinds=tuple(kinds),
        levels_mm=np.array(levels, dtype=float),
        distances_mm=np.array(distances, dtype=float),
        successes=np.array(successes, dtype=bool),
    )


def evaluate_design(
    truth: Truth,
    block: BlockDesign,
    steps: StaircaseSteps,
    rng: np.random.Generator,
    *,
    grids: AnalysisGrids,
    replications: int,
    placement_sd_mm: float = DEFAULT_PLACEMENT_SD_MM,
    level: float = 0.95,
) -> DesignPerformance:
    """Simulate a block `replications` times against one truth and say how the estimates turned out."""
    if replications < 1:
        raise ValueError("at least one replication is needed")
    summaries = [
        simulate_block(truth, block, steps, rng, grids=grids, placement_sd_mm=placement_sd_mm, level=level).summary
        for _ in range(replications)
    ]
    return _performance(truth, block, summaries)


def evaluate_fits(
    truth: Truth,
    block: BlockDesign,
    steps: StaircaseSteps,
    rng: np.random.Generator,
    *,
    fits: dict[str, AnalysisGrids],
    replications: int,
    placement_sd_mm: float = DEFAULT_PLACEMENT_SD_MM,
    level: float = 0.95,
) -> dict[str, tuple[DesignPerformance, float]]:
    """Simulate a block `replications` times and analyse every simulated block in several ways.

    `fits` names the grids of each analysis. All of them see the same trials, so a difference between two of them is
    a difference between the analyses and not between two draws of blocks. For each name the result holds how the
    estimates turned out and the posterior mean of the floor, averaged over the blocks.
    """
    if not fits:
        raise ValueError("at least one fit is needed")
    if replications < 1:
        raise ValueError("at least one replication is needed")
    summaries: dict[str, list[ThresholdSummary]] = {name: [] for name in fits}
    for _ in range(replications):
        _, _, distances, successes = _simulate_trials(truth, block, steps, rng, placement_sd_mm)
        for name, grids in fits.items():
            summaries[name].append(_fit(distances, successes, grids).summary(level))
    return {
        name: (_performance(truth, block, items), float(np.mean([summary.floor_mean for summary in items])))
        for name, items in summaries.items()
    }


def _simulate_trials(
    truth: Truth, block: BlockDesign, steps: StaircaseSteps, rng: np.random.Generator, placement_sd_mm: float
) -> tuple[list[str], list[float], list[float], list[bool]]:
    """The trials of one simulated block: for each its kind, requested distance, achieved distance and outcome."""
    if placement_sd_mm < 0.0:
        raise ValueError("placement_sd_mm cannot be negative")
    n_trials = block.n_trials
    placement_errors = rng.normal(0.0, placement_sd_mm, size=n_trials).tolist()
    limits = distance_at_probability(
        rng.random(n_trials), truth.r50_mm, truth.width_mm, truth.lapse, truth.floor
    ).tolist()
    kinds: list[str] = []
    levels: list[float] = []
    distances: list[float] = []
    successes: list[bool] = []

    def run_trial(kind: str, level_mm: float) -> bool:
        index = len(levels)
        distance = max(level_mm + placement_errors[index], 0.0)
        success = distance < limits[index]
        kinds.append(kind)
        levels.append(level_mm)
        distances.append(distance)
        successes.append(success)
        return success

    for _ in range(block.anchors_zero):
        run_trial(KIND_ZERO, 0.0)
    for _ in range(block.anchors_far):
        run_trial(KIND_FAR, float(block.far_level_mm))
    for level_mm in block.fixed_levels_mm:
        run_trial(KIND_FIXED, level_mm)

    rules = [
        StaircaseParams(
            start_level_mm=plan.start_level_mm,
            n_trials=plan.n_trials,
            step_initial_mm=steps.step_initial_mm,
            step_final_mm=steps.step_final_mm,
            min_level_mm=block.min_level_mm,
            max_level_mm=block.max_level_mm,
        )
        for plan in block.staircases
    ]
    histories: list[list[tuple[float, bool]]] = [[] for _ in rules]
    running = bool(rules)
    while running:  # the staircases take turns until each has run its trials
        running = False
        for plan, rule, history in zip(block.staircases, rules, histories, strict=True):
            level_mm = next_level(rule, history)
            if level_mm is None:
                continue
            history.append((level_mm, run_trial(plan.id, level_mm)))
            running = True
    return kinds, levels, distances, successes


def _fit(distances: list[float], successes: list[bool], grids: AnalysisGrids) -> ThresholdPosterior:
    return fit_threshold(
        distances,
        successes,
        r50_grid_mm=grids.r50_grid_mm,
        width_grid_mm=grids.width_grid_mm,
        lapse_values=grids.lapse_values,
        floor_values=grids.floor_values,
    )


def _performance(truth: Truth, block: BlockDesign, summaries: list[ThresholdSummary]) -> DesignPerformance:
    """How the estimates of many simulated blocks turned out against the truth they were played against."""
    errors = np.array([summary.r50_median_mm for summary in summaries]) - truth.r50_mm
    low = np.array([summary.r50_low_mm for summary in summaries])
    high = np.array([summary.r50_high_mm for summary in summaries])
    at_lower = np.array([summary.reaches_lower_edge for summary in summaries])
    at_upper = np.array([summary.reaches_upper_edge for summary in summaries])
    return DesignPerformance(
        truth=truth,
        block=block.name,
        n_trials=block.n_trials,
        replications=len(summaries),
        bias_mm=float(errors.mean()),
        rmse_mm=float(np.sqrt(np.mean(errors**2))),
        mean_interval_width_mm=float(np.mean(high - low)),
        interval_coverage=float(np.mean((low <= truth.r50_mm) & (truth.r50_mm <= high))),
        lower_edge_rate=float(at_lower.mean()),
        upper_edge_rate=float(at_upper.mean()),
        any_edge_rate=float((at_lower | at_upper).mean()),
    )


# --------------------------------------------------------------------------------------------------------------------
# The designs
# --------------------------------------------------------------------------------------------------------------------


def constant_stimuli(block: BlockDesign) -> BlockDesign:
    """The same number of trials at fixed distances, equally spaced from the block's smallest level to its largest.

    This is the method of constant stimuli: the distances are chosen before the block and no outcome changes them.
    Each distance is tested once.
    """
    levels = np.linspace(block.min_level_mm, block.max_level_mm, block.n_trials).tolist()
    return replace(
        block, anchors_zero=0, anchors_far=0, far_level_mm=None, staircases=(), fixed_levels_mm=tuple(levels)
    )


def with_staircase_trials(block: BlockDesign, extra_trials: int) -> BlockDesign:
    """The block with every staircase `extra_trials` longer (or shorter, if negative)."""
    plans = tuple(replace(plan, n_trials=plan.n_trials + extra_trials) for plan in block.staircases)
    if any(plan.n_trials < 1 for plan in plans):
        raise ValueError("a staircase runs at least one trial")
    return replace(block, staircases=plans)


def design_variants(
    block: BlockDesign, steps: StaircaseSteps, widest_level_mm: float | None = None
) -> dict[str, tuple[BlockDesign, StaircaseSteps]]:
    """The designs compared for one block, by name, in the order of `DESIGNS`.

    * `protocol`: the block and the steps as given;
    * `constant`: the same number of trials at fixed, equally spaced distances (`constant_stimuli`);
    * `fine_step`: as given, with a final step of `FINE_STEP_FINAL_MM`;
    * `longer`, `shorter`: as given, with every staircase `STAIRCASE_TRIALS_DELTA` trials longer or shorter;
    * `wider_range`, only when `widest_level_mm` is given and exceeds the block's largest level: as given, with the
      largest level raised to it.
    """
    variants = {
        "protocol": (block, steps),
        "constant": (constant_stimuli(block), steps),
        "fine_step": (block, replace(steps, step_final_mm=FINE_STEP_FINAL_MM)),
        "longer": (with_staircase_trials(block, STAIRCASE_TRIALS_DELTA), steps),
        "shorter": (with_staircase_trials(block, -STAIRCASE_TRIALS_DELTA), steps),
    }
    if widest_level_mm is not None and widest_level_mm > block.max_level_mm:
        variants[WIDER_RANGE] = (replace(block, max_level_mm=widest_level_mm), steps)
    return variants


def blocks_from_protocol(protocol: dict[str, Any]) -> list[BlockDesign]:
    """Every block of the protocol file that runs staircases, in the file's order."""
    min_level_mm = float(protocol["staircase"]["min_level_mm"])
    blocks = []
    for name, stated in protocol["blocks"].items():
        if "staircases" not in stated:
            continue
        plans = tuple(
            StaircasePlan(id=item["id"], start_level_mm=float(item["start_level_mm"]), n_trials=int(item["trials"]))
            for item in stated["staircases"]
        )
        blocks.append(
            BlockDesign(
                name=name,
                anchors_zero=int(stated.get("anchors_zero", 0)),
                staircases=plans,
                min_level_mm=min_level_mm,
                max_level_mm=float(stated["max_level_mm"]),
                anchors_far=int(stated.get("anchors_far", 0)),
                far_level_mm=stated.get("far_level_mm"),
            )
        )
    return blocks


def steps_from_protocol(protocol: dict[str, Any]) -> StaircaseSteps:
    """The staircase steps of the protocol file. Only the one-up-one-down rule of `staircase.next_level` is known."""
    stated = protocol["staircase"]
    if stated.get("rule") != "1up1down":
        raise ValueError(f"the simulation knows the rule '1up1down' only, got {stated.get('rule')!r}")
    return StaircaseSteps(
        step_initial_mm=float(stated["step_initial_mm"]), step_final_mm=float(stated["step_final_mm"])
    )


# --------------------------------------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------------------------------------


def sweep_truths() -> list[Truth]:
    """The truths of the sweep: every combination of `SWEEP_R50_MM`, `SWEEP_WIDTH_MM` and `SWEEP_LAPSE`."""
    return [Truth(r50, width, lapse) for r50 in SWEEP_R50_MM for width in SWEEP_WIDTH_MM for lapse in SWEEP_LAPSE]


def run_sweep(
    protocol: dict[str, Any],
    *,
    replications: int,
    seed: int,
    truths: Sequence[Truth] | None = None,
    designs: Sequence[str] = (*DESIGNS, WIDER_RANGE),
    placement_sd_mm: float = DEFAULT_PLACEMENT_SD_MM,
) -> list[SweepRow]:
    """Evaluate every design for every staircase block of the protocol and every truth.

    Rows come block by block (in the file's order), then design by design, then truth by truth. The `wider_range`
    design is run only for a block whose largest level is below the largest among the blocks. Each row has its own
    stream of random numbers, derived from the seed and from what the row is, so a row's numbers do not depend on
    which other rows are run.
    """
    unknown = [design for design in designs if design not in (*DESIGNS, WIDER_RANGE)]
    if unknown:
        raise ValueError(f"unknown designs {unknown}; the designs are {(*DESIGNS, WIDER_RANGE)}")
    truths = sweep_truths() if truths is None else list(truths)
    grids = grids_from_protocol(protocol["analysis"])
    level = float(protocol["analysis"]["interval_level"])
    steps = steps_from_protocol(protocol)
    blocks = blocks_from_protocol(protocol)
    widest_level_mm = max((block.max_level_mm for block in blocks), default=None)
    rows = []
    for block in blocks:
        variants = design_variants(block, steps, widest_level_mm)
        for design in designs:
            if design not in variants:
                continue
            variant, variant_steps = variants[design]
            for truth in truths:
                performance = evaluate_design(
                    truth,
                    variant,
                    variant_steps,
                    _row_generator(seed, design, block.name, truth),
                    grids=grids,
                    replications=replications,
                    placement_sd_mm=placement_sd_mm,
                    level=level,
                )
                rows.append(SweepRow(design=design, performance=performance))
    return rows


def _row_generator(seed: int, design: str, block_name: str, truth: Truth) -> np.random.Generator:
    """The random stream of one row of the sweep: a function of the seed and of what the row is."""
    label = f"{design}|{block_name}|{truth.r50_mm!r}|{truth.width_mm!r}|{truth.lapse!r}|{truth.floor!r}"
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    key = tuple(int.from_bytes(digest[start : start + 4], "big") for start in range(0, 16, 4))
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=key))


# --------------------------------------------------------------------------------------------------------------------
# The floor check
# --------------------------------------------------------------------------------------------------------------------


def floor_check_truths() -> list[Truth]:
    """The truths of the floor check: the same curves with a floor (`FLOOR_CHECK_TRUE_FLOORS`) and without."""
    return [
        Truth(r50, width, FLOOR_CHECK_LAPSE, floor)
        for floor in FLOOR_CHECK_TRUE_FLOORS
        for r50 in FLOOR_CHECK_R50_MM
        for width in FLOOR_CHECK_WIDTH_MM
    ]


def floor_check_designs(protocol: dict[str, Any]) -> list[tuple[str, BlockDesign]]:
    """The blocks of the floor check, each with the name of its variant.

    Every block of the protocol that has far anchors, as written. Then the largest block without far anchors three
    times: as written; with `FLOOR_CHECK_FAR_ANCHORS` far anchors added at its largest level; and, for a fair
    comparison, with as many trials added to its staircases instead.
    """
    blocks = blocks_from_protocol(protocol)
    designs = [("as_written", block) for block in blocks if block.anchors_far > 0]
    without = [block for block in blocks if block.anchors_far == 0]
    if without:
        largest = max(without, key=lambda block: block.n_trials)
        share, left_over = divmod(FLOOR_CHECK_FAR_ANCHORS, len(largest.staircases))
        longer = tuple(
            replace(plan, n_trials=plan.n_trials + share + (index < left_over))
            for index, plan in enumerate(largest.staircases)
        )
        designs += [
            ("as_written", largest),
            ("far_anchors", replace(largest, anchors_far=FLOOR_CHECK_FAR_ANCHORS, far_level_mm=largest.max_level_mm)),
            ("longer_staircases", replace(largest, staircases=longer)),
        ]
    return designs


def run_floor_check(
    protocol: dict[str, Any],
    *,
    replications: int,
    seed: int,
    placement_sd_mm: float = DEFAULT_PLACEMENT_SD_MM,
) -> list[FloorCheckRow]:
    """What a floor in the truth does to a fit without one, and what a floor in the fit costs where there is none.

    Every block of `floor_check_designs` is simulated against every truth of `floor_check_truths`, and each simulated
    block is fitted twice: without a floor in the model, and with the floor values `FLOOR_CHECK_FLOOR_VALUES`. The
    other grids are the protocol's; the protocol's own floor values play no part.
    """
    grids = grids_from_protocol(protocol["analysis"])
    fits = {
        FIT_NO_FLOOR: grids._replace(floor_values=np.array([0.0])),
        FIT_FLOOR: grids._replace(floor_values=np.array(FLOOR_CHECK_FLOOR_VALUES)),
    }
    level = float(protocol["analysis"]["interval_level"])
    steps = steps_from_protocol(protocol)
    rows = []
    for variant, block in floor_check_designs(protocol):
        for truth in floor_check_truths():
            results = evaluate_fits(
                truth,
                block,
                steps,
                _row_generator(seed, f"floor_check:{variant}", block.name, truth),
                fits=fits,
                replications=replications,
                placement_sd_mm=placement_sd_mm,
                level=level,
            )
            for fit, (performance, mean_floor_estimate) in results.items():
                rows.append(FloorCheckRow(variant, fit, performance, mean_floor_estimate))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# The paired comparison
# --------------------------------------------------------------------------------------------------------------------


def paired_power(pairs: int, p_first: float, p_second: float, alpha: float = 0.05) -> float:
    """The probability that the exact McNemar test gives a p-value below `alpha`.

    Each of `pairs` placements is tested with both policies, which succeed with probabilities `p_first` and `p_second`
    independently of each other. The result is exact: every possible table of outcomes is listed, and the
    probabilities of those the test rejects are added. With `p_first == p_second` it is the test's false-alarm rate.
    """
    if pairs < 1:
        raise ValueError("at least one pair is needed")
    if not (0.0 <= p_first <= 1.0 and 0.0 <= p_second <= 1.0):
        raise ValueError("success rates lie between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    only_first = p_first * (1.0 - p_second)
    only_second = (1.0 - p_first) * p_second
    same = p_first * p_second + (1.0 - p_first) * (1.0 - p_second)
    total = math.fsum(
        ways * only_first**first * only_second**second * same ** (pairs - first - second)
        for first, second, ways in _rejected_tables(pairs, alpha)
    )
    return min(1.0, total)


def paired_power_table(
    pairs_options: Sequence[int], rates: Sequence[float] = POWER_RATES, alpha: float = 0.05
) -> list[PowerRow]:
    """`paired_power` for every number of pairs and every pair of success rates."""
    return [
        PowerRow(pairs=pairs, p_first=first, p_second=second, power=paired_power(pairs, first, second, alpha))
        for pairs in pairs_options
        for first in rates
        for second in rates
    ]


def rate_needed(pairs: int, lower_rate: float, power: float = POWER_TARGET, alpha: float = 0.05) -> float | None:
    """The smallest success rate above `lower_rate`, in steps of 0.05, that the paired test tells from it with at
    least the given probability; None when no rate up to 1 is told apart that often."""
    for twentieths in range(1, 21):
        rate = twentieths / 20
        if rate > lower_rate + 1e-9 and paired_power(pairs, lower_rate, rate, alpha) >= power:
            return rate
    return None


@lru_cache(maxsize=None)
def _rejected_tables(pairs: int, alpha: float) -> tuple[tuple[int, int, int], ...]:
    """Every split of `pairs` into (only the first succeeded, only the second, same outcome) that the test rejects,
    with the number of orders in which it can arise."""
    tables = []
    for first in range(pairs + 1):
        for second in range(pairs + 1 - first):
            if mcnemar_exact(first, second) < alpha:
                tables.append((first, second, math.comb(pairs, first) * math.comb(pairs - first, second)))
    return tuple(tables)


# --------------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------------


def write_sweep_csv(path: str | Path, rows: Sequence[SweepRow]) -> None:
    """Every row of the sweep, one line each."""
    header = ["design", "block", "n_trials", "r50_true_mm", "width_true_mm", "lapse_true", *_PERFORMANCE_COLUMNS]
    lines = [
        [row.design, row.performance.block, row.performance.n_trials, *_truth_cells(row.performance.truth)[:3]]
        + _performance_cells(row.performance)
        for row in rows
    ]
    _write_csv(path, header, lines)


def write_floor_check_csv(path: str | Path, rows: Sequence[FloorCheckRow]) -> None:
    """Every row of the floor check, one line each."""
    header = ["block", "variant", "n_trials", "fit", "r50_true_mm", "width_true_mm", "lapse_true", "floor_true"]
    lines = [
        [row.performance.block, row.variant, row.performance.n_trials, row.fit, *_truth_cells(row.performance.truth)]
        + _performance_cells(row.performance)
        + [f"{row.mean_floor_estimate:.4f}"]
        for row in rows
    ]
    _write_csv(path, [*header, *_PERFORMANCE_COLUMNS, "mean_floor_estimate"], lines)


def write_power_csv(path: str | Path, rows: Sequence[PowerRow]) -> None:
    """The paired test's table, one line for each number of pairs and pair of success rates."""
    lines = [[row.pairs, f"{row.p_first:g}", f"{row.p_second:g}", f"{row.power:.6f}"] for row in rows]
    _write_csv(path, ["pairs", "p_first", "p_second", "power"], lines)


_PERFORMANCE_COLUMNS = (
    "replications",
    "bias_mm",
    "rmse_mm",
    "mean_interval_width_mm",
    "interval_coverage",
    "lower_edge_rate",
    "upper_edge_rate",
    "any_edge_rate",
)


def _truth_cells(truth: Truth) -> list[str]:
    return [f"{truth.r50_mm:g}", f"{truth.width_mm:g}", f"{truth.lapse:g}", f"{truth.floor:g}"]


def _performance_cells(performance: DesignPerformance) -> list[str]:
    return [
        str(performance.replications),
        f"{performance.bias_mm:.4f}",
        f"{performance.rmse_mm:.4f}",
        f"{performance.mean_interval_width_mm:.4f}",
        f"{performance.interval_coverage:.4f}",
        f"{performance.lower_edge_rate:.4f}",
        f"{performance.upper_edge_rate:.4f}",
        f"{performance.any_edge_rate:.4f}",
    ]


def _write_csv(path: str | Path, header: Sequence[str], lines: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(lines)


# --------------------------------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tolerance_radius.simulate --protocol <path> --out <dir> [--replications N] [--seed S]`."""
    parser = argparse.ArgumentParser(
        prog="python -m tolerance_radius.simulate",
        description="Simulate the protocol's blocks of trials and the fit, and write procedure.csv, floor_check.csv, "
        "paired_power.csv and the report procedure.md into the output directory.",
    )
    parser.add_argument("--protocol", required=True, type=Path, help="the protocol file (config/protocol.toml)")
    parser.add_argument("--out", required=True, type=Path, help="the directory to write into; created if missing")
    parser.add_argument(
        "--replications",
        type=_whole_number(1),
        default=DEFAULT_REPLICATIONS,
        help=f"simulated blocks for every row (default {DEFAULT_REPLICATIONS})",
    )
    parser.add_argument(
        "--seed",
        type=_whole_number(0),
        default=DEFAULT_SEED,
        help=f"seed of the random numbers (default {DEFAULT_SEED})",
    )
    arguments = parser.parse_args(argv)
    # The report is written from this module's rows by another module, which itself builds on this one.
    from tolerance_radius import procedure_report

    protocol = load_protocol(arguments.protocol)
    info = procedure_report.RunInfo(
        seed=arguments.seed,
        replications=arguments.replications,
        placement_sd_mm=DEFAULT_PLACEMENT_SD_MM,
        protocol_file=arguments.protocol.name,
        protocol_digest=hashlib.sha256(arguments.protocol.read_bytes()).hexdigest()[:12],
    )
    rows = run_sweep(protocol, replications=info.replications, seed=info.seed)
    floor_rows = run_floor_check(protocol, replications=info.replications, seed=info.seed)
    pairs = next((int(block["pairs"]) for block in protocol["blocks"].values() if "pairs" in block), None)
    options = [] if pairs is None else [pairs - PAIRS_DELTA, pairs, pairs + PAIRS_DELTA]
    power_rows = paired_power_table([option for option in options if option > 0])

    arguments.out.mkdir(parents=True, exist_ok=True)
    write_sweep_csv(arguments.out / "procedure.csv", rows)
    write_floor_check_csv(arguments.out / "floor_check.csv", floor_rows)
    write_power_csv(arguments.out / "paired_power.csv", power_rows)
    report = arguments.out / "procedure.md"
    report.write_text(procedure_report.render_report(protocol, rows, floor_rows, power_rows, info), encoding="utf-8")
    print(f"wrote {report} and the tables behind it: {len(rows)} sweep rows, {len(floor_rows)} floor-check rows")
    return 0


def _whole_number(smallest: int) -> Any:
    """An argument type for argparse: a whole number that is at least `smallest`."""

    def parse(text: str) -> int:
        value = int(text)
        if value < smallest:
            raise argparse.ArgumentTypeError(f"must be at least {smallest}, got {value}")
        return value

    return parse


if __name__ == "__main__":
    # Run the module under its own name, so that the report module and this command share one copy of it.
    from tolerance_radius.simulate import main as run

    raise SystemExit(run())
