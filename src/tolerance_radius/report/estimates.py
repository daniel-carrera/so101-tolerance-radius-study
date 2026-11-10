"""The statistics of the report: what `protocol/protocol.md`, section 10, computes from trials that have been read.

Every function here takes plain numbers (distances, outcomes, posteriors) and returns plain numbers. Nothing reads a
file, draws a random number or looks at a clock, so the same trials always give the same results.

Where the protocol leaves a detail open, the choice made here is stated once, in the function that makes it:

* `scale_rule`: when the scale check's runs "agree within one percentage point";
* `radius_difference`: what "the probability that one radius exceeds another" is on a grid, where two radii can be
  equal;
* `count_interval`: which whole numbers the 95 % interval of a predicted count holds;
* `distance_bins`: which bin a trial exactly on an edge belongs to;
* `plateau_check`: when the two fits of the plateau check "differ by more than" the limit;
* `layout_outcome`: what the paired test says when it decides for the clustered policy.

Standard library and NumPy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from tolerance_radius.intervals import mcnemar_exact, wilson_interval
from tolerance_radius.psychometric import AnalysisGrids, ThresholdPosterior, fit_threshold

#: Width of the bins of the model-free summary (protocol, section 10: "success counts in 10 mm bins").
BIN_WIDTH_MM = 10.0
#: The rule of prediction 2 (protocol, section 9): "larger" at this posterior probability or above, "smaller" at one
#: minus it or below, "not told apart" between.
VERDICT_PROBABILITY = 0.95
#: The scale check's runs agree when their factors lie within this many percentage points (protocol, section 7).
SCALE_AGREEMENT_POINTS = 1.0
#: Fewer runs than this cannot be said to agree, so no correction is made.
SCALE_MIN_RUNS = 2
#: Step of the distances a fitted curve is written on.
CURVE_STEP_MM = 0.5
#: Every number of the results is kept to this many significant digits (see `stable`).
SIGNIFICANT_DIGITS = 9
#: A running total counts as having reached its mark when it is within this relative margin of it (the same margin
#: `psychometric` uses), so that an exact tie does not depend on the last bit of a sum.
_QUANTILE_SLACK = 1e-12
#: Slack for comparisons of numbers that were computed, not read: far below anything that is measured.
_ROUNDING = 1e-9

#: The rule of prediction 3 (protocol, section 9): the exact paired test gives a p-value below this, with the spread
#: policy ahead.
LAYOUT_P_BELOW = 0.05

VERDICT_LARGER, VERDICT_SMALLER, VERDICT_NOT_TOLD_APART = "larger", "smaller", "not told apart"
LOOKUP_MATCHES, LOOKUP_FALLS_SHORT, LOOKUP_EXCEEDS = "matches", "falls short", "exceeds"
LAYOUT_SPREAD, LAYOUT_CLUSTERED, LAYOUT_UNDECIDED = "spread ahead", "clustered ahead", "not decided"


# --------------------------------------------------------------------------------------------------------------------
# The scale correction (protocol, section 7)
# --------------------------------------------------------------------------------------------------------------------


def scale_rule(factors: Sequence[float]) -> dict[str, Any]:
    """Whether distances are divided by the scale check's factor, and by what.

    `factors` holds one number per run of the scale check: the measured length of the run divided by its true length.
    The protocol's rule: if the runs agree within one percentage point, every distance is divided by the factor;
    otherwise distances are left as read. As applied here:

    * the runs agree when the largest factor exceeds the smallest by at most 0.01 (one point), the line itself
      included;
    * the factor is the mean of the runs;
    * with no run, or a single one, nothing can be said to agree and no correction is made.

    `divisor` is the number every distance of the analysis is divided by: the factor when the correction applies,
    1 when it does not.
    """
    values = [float(value) for value in factors]
    if any(not (math.isfinite(value) and value > 0.0) for value in values):
        raise ValueError("a scale factor is a positive number")
    if not values:
        return {
            "applied": False,
            "factor": None,
            "factor_min": None,
            "factor_max": None,
            "n_runs": 0,
            "spread_points": None,
            "divisor": 1.0,
            "why": "no scale run was found",
        }
    factor = stable(math.fsum(values) / len(values))
    spread_points = stable(100.0 * (max(values) - min(values)))
    agree = spread_points <= SCALE_AGREEMENT_POINTS + _ROUNDING
    if len(values) < SCALE_MIN_RUNS:
        applied, why = False, "one scale run cannot show agreement"
    elif agree:
        applied, why = True, f"the {len(values)} scale runs agree within {SCALE_AGREEMENT_POINTS:g} percentage point"
    else:
        applied, why = (
            False,
            f"the {len(values)} scale runs differ by more than {SCALE_AGREEMENT_POINTS:g} percentage point",
        )
    return {
        "applied": applied,
        "factor": factor,
        "factor_min": min(values),
        "factor_max": max(values),
        "n_runs": len(values),
        "spread_points": spread_points,
        "divisor": factor if applied else 1.0,
        "why": why,
    }


# --------------------------------------------------------------------------------------------------------------------
# Without a model: bins, the farthest success and the nearest failure
# --------------------------------------------------------------------------------------------------------------------


def bin_index(distance_mm: float, width_mm: float = BIN_WIDTH_MM) -> int:
    """The number of the bin a distance falls in, from 0. A bin is closed below and open above.

    A trial at exactly 10 mm is in the bin from 10 to 20 mm; one at 9.999 mm in the bin from 0 to 10 mm.
    """
    if not (math.isfinite(distance_mm) and distance_mm >= 0.0):
        raise ValueError(f"a distance is a finite number, zero or more, got {distance_mm!r}")
    if not width_mm > 0.0:
        raise ValueError("the bin width is positive")
    index = math.floor(distance_mm / width_mm)
    # The division can round across an edge; the comparison with the edge itself decides.
    while (index + 1) * width_mm <= distance_mm:
        index += 1
    while index * width_mm > distance_mm:
        index -= 1
    return index


def distance_bins(
    distances_mm: Sequence[float], successes: Sequence[bool], *, width_mm: float = BIN_WIDTH_MM, level: float = 0.95
) -> list[dict[str, Any]]:
    """Success counts in bins of distance, each with its Wilson interval, nearest bin first.

    Only bins that hold at least one trial are listed: an empty bin has no rate and no interval.
    """
    _same_length(distances_mm, successes)
    counts: dict[int, list[int]] = {}
    for distance, success in zip(distances_mm, successes, strict=True):
        entry = counts.setdefault(bin_index(float(distance), width_mm), [0, 0])
        entry[0] += 1
        entry[1] += int(bool(success))
    bins = []
    for index in sorted(counts):
        n, k = counts[index]
        low, high = wilson_interval(k, n, level)
        bins.append(
            {
                "lo_mm": index * width_mm,
                "hi_mm": (index + 1) * width_mm,
                "n": n,
                "k": k,
                "wilson_lo": low,
                "wilson_hi": high,
            }
        )
    return bins


def extremes(distances_mm: Sequence[float], successes: Sequence[bool]) -> tuple[float | None, float | None]:
    """The largest distance with a success and the smallest with a failure; None where there was none."""
    _same_length(distances_mm, successes)
    succeeded = [float(d) for d, s in zip(distances_mm, successes, strict=True) if s]
    failed = [float(d) for d, s in zip(distances_mm, successes, strict=True) if not s]
    return (max(succeeded) if succeeded else None, min(failed) if failed else None)


def proportion(successes: int, trials: int, level: float = 0.95) -> dict[str, Any]:
    """A count of successes with its Wilson interval. With no trials there is no interval."""
    low, high = wilson_interval(successes, trials, level) if trials else (None, None)
    return {"k": int(successes), "n": int(trials), "wilson_lo": low, "wilson_hi": high}


# --------------------------------------------------------------------------------------------------------------------
# The fitted curve
# --------------------------------------------------------------------------------------------------------------------


def fit(distances_mm: Sequence[float], successes: Sequence[bool], grids: AnalysisGrids) -> ThresholdPosterior:
    """The posterior of the success curve for a set of trials, on the protocol's grids."""
    return fit_threshold(
        np.asarray(distances_mm, dtype=float),
        np.asarray(successes, dtype=bool),
        r50_grid_mm=grids.r50_grid_mm,
        width_grid_mm=grids.width_grid_mm,
        lapse_values=grids.lapse_values,
        floor_values=grids.floor_values,
    )


def curve_distances(largest_tested_mm: float, grid_end_mm: float, step_mm: float = CURVE_STEP_MM) -> np.ndarray:
    """The distances a fitted curve is written on: from 0 in equal steps to the end of the `r50` grid, and beyond it
    if a trial was run farther out, so that the curve always reaches the largest distance tested."""
    reach = max(float(largest_tested_mm), float(grid_end_mm))
    steps = math.ceil(reach / step_mm - _ROUNDING)
    return step_mm * np.arange(steps + 1)


def fit_summary(
    posterior: ThresholdPosterior, blocks: Sequence[str], *, level: float, largest_tested_mm: float
) -> dict[str, Any]:
    """What is reported of a fit: the radius with its interval, whether the data bound it, and the curve.

    `open_below` and `open_above` are the rule of protocol section 10: the interval is open on a side when the
    posterior at that end of the `r50` grid is at least half of what it is at the interval's own bound.
    """
    summary = posterior.summary(level)
    distances = curve_distances(largest_tested_mm, float(posterior.r50_grid_mm[-1]))
    low, high = posterior.band(distances, level)
    return {
        "blocks": list(blocks),
        "n": summary.n_trials,
        "k": summary.n_success,
        "r50_median_mm": summary.r50_median_mm,
        "r50_lo_mm": summary.r50_low_mm,
        "r50_hi_mm": summary.r50_high_mm,
        "open_below": summary.reaches_lower_edge,
        "open_above": summary.reaches_upper_edge,
        "width_median_mm": summary.width_median_mm,
        "width_lo_mm": summary.width_low_mm,
        "width_hi_mm": summary.width_high_mm,
        "lapse_mean": summary.lapse_mean,
        "floor_mean": summary.floor_mean,
        "curve": {
            "d_mm": distances.tolist(),
            "mean": posterior.predict(distances).tolist(),
            "lo": low.tolist(),
            "hi": high.tolist(),
        },
    }


def trial_counts(distances_mm: Sequence[float], successes: Sequence[bool], at_zero: Sequence[bool]) -> dict[str, Any]:
    """What a set of trials says without a model: the largest distance with a success, the smallest with a failure,
    and the successes among the trials asked at a demonstration's own start (`at_zero` marks those)."""
    _same_length(distances_mm, at_zero)
    max_success, min_failure = extremes(distances_mm, successes)
    return {
        "max_success_d_mm": max_success,
        "min_failure_d_mm": min_failure,
        "zero_k": sum(bool(success) for success, zero in zip(successes, at_zero, strict=True) if zero),
        "zero_n": sum(bool(zero) for zero in at_zero),
    }


def plateau_check(
    main_median_mm: float, wide: ThresholdPosterior, limit_mm: float | None, level: float = 0.95
) -> dict[str, Any]:
    """The plateau check of protocol section 10: the same trials fitted with a longer list of lapse values.

    `wide` is that second fit. Reported: its radius with its interval, whether the data bound it, and how far its
    median lies from the main fit's (`median_shift_mm`, second minus first). The plateau is in doubt when the two
    medians differ by more than `limit_mm`; a difference of exactly the limit is not more than it. Without a limit
    nothing is said (`plateau_in_doubt` is None). `lapse_mean` is the posterior mean of the lapse under the longer
    list: one minus it is the plateau this fit believes in.
    """
    told = wide.summary(level)
    shift = stable(told.r50_median_mm - main_median_mm)
    return {
        "lapse_values": wide.lapse_values.tolist(),
        "r50_median_mm": told.r50_median_mm,
        "r50_lo_mm": told.r50_low_mm,
        "r50_hi_mm": told.r50_high_mm,
        "open_below": told.reaches_lower_edge,
        "open_above": told.reaches_upper_edge,
        "lapse_mean": told.lapse_mean,
        "median_shift_mm": shift,
        "limit_mm": limit_mm,
        "plateau_in_doubt": None if limit_mm is None else bool(abs(shift) > limit_mm + _ROUNDING),
    }


# --------------------------------------------------------------------------------------------------------------------
# Comparison of radii (protocol, section 10, and the rule of prediction 2)
# --------------------------------------------------------------------------------------------------------------------


def verdict(prob_greater: float) -> str:
    """The rule of prediction 2 applied to the probability that the first radius exceeds the second."""
    if prob_greater >= VERDICT_PROBABILITY:
        return VERDICT_LARGER
    if prob_greater <= 1.0 - VERDICT_PROBABILITY:
        return VERDICT_SMALLER
    return VERDICT_NOT_TOLD_APART


def difference_distribution(first: ThresholdPosterior, second: ThresholdPosterior) -> tuple[np.ndarray, np.ndarray]:
    """The posterior of `r50` of the first fit minus `r50` of the second, the two taken as independent.

    Returns the differences that can occur, in increasing order, and the probability of each. Every pair of grid
    values is weighted by the product of its two marginal probabilities, so the result is exact: no random numbers.
    """
    first_grid, first_probability = first.r50_marginal()
    second_grid, second_probability = second.r50_marginal()
    differences = (first_grid[:, None] - second_grid[None, :]).reshape(-1)
    weights = (first_probability[:, None] * second_probability[None, :]).reshape(-1)
    values, where = np.unique(differences, return_inverse=True)
    return values, np.bincount(where.reshape(-1), weights=weights, minlength=values.size)


def radius_difference(first: ThresholdPosterior, second: ThresholdPosterior, level: float = 0.95) -> dict[str, Any]:
    """The first fit's radius minus the second's: median, interval, and the probability that the first is larger.

    On a grid two radii can be exactly equal. Such a tie is counted half for each side, as it would fall if each
    grid value stood for the stretch of radii around it. `prob_equal` gives the size of the tie. Without this, two
    identical posteriors on one grid value would give a probability of 0 and the verdict "smaller".
    """
    values, probabilities = difference_distribution(first, second)
    tail = (1.0 - level) / 2.0
    low, median, high = _quantile_indices(probabilities, (tail, 0.5, 1.0 - tail))
    total = float(probabilities.sum())
    greater = float(probabilities[values > 0.0].sum()) / total
    equal = stable(float(probabilities[values == 0.0].sum()) / total)
    # Rounded before the rule is applied, so that the verdict is the rule applied to the number that is reported.
    prob_greater = stable(min(1.0, greater + 0.5 * equal))
    return {
        "fits": "block",
        "median_mm": float(values[median]),
        "lo_mm": float(values[low]),
        "hi_mm": float(values[high]),
        "prob_greater": prob_greater,
        "prob_equal": equal,
        "verdict": verdict(prob_greater),
    }


# --------------------------------------------------------------------------------------------------------------------
# The workspace test and the lookup prediction
# --------------------------------------------------------------------------------------------------------------------


def paired_table(first: Sequence[bool], second: Sequence[bool]) -> dict[str, int]:
    """The 2 x 2 table of paired outcomes: how often both succeeded, neither, only the first, only the second."""
    _same_length(first, second)
    table = {"both": 0, "neither": 0, "only_first": 0, "only_second": 0}
    for a, b in zip(first, second, strict=True):
        if a and b:
            table["both"] += 1
        elif a:
            table["only_first"] += 1
        elif b:
            table["only_second"] += 1
        else:
            table["neither"] += 1
    return table


def paired_p_value(table: dict[str, int]) -> float | None:
    """The exact two-sided McNemar p-value of a paired table; None when the table holds no pair."""
    if sum(table.values()) == 0:
        return None
    return mcnemar_exact(table["only_first"], table["only_second"])


def layout_outcome(p_value: float | None, k_spread: int, k_clustered: int) -> str:
    """What the paired test says about the two layouts, by the rule of prediction 3.

    `spread ahead` when the exact test gives p below 0.05 and the spread policy has more successes: the prediction
    holds. The protocol says of every other case that the test did not decide. One of them is told apart here,
    because it is a finding of its own: p below 0.05 with the clustered policy ahead (`clustered ahead`). Everything
    else is `not decided`.
    """
    if p_value is None or not p_value < LAYOUT_P_BELOW or k_spread == k_clustered:
        return LAYOUT_UNDECIDED
    return LAYOUT_SPREAD if k_spread > k_clustered else LAYOUT_CLUSTERED


def count_interval(probabilities: Sequence[float], level: float = 0.95) -> tuple[int, int]:
    """The equal-tailed interval of a count, given the probability of every count from 0 up.

    The lower end is the smallest count at which the running total of probability reaches 2.5 % (for a 95 %
    interval), the upper end the smallest at which it reaches 97.5 %. Less than 2.5 % lies below the interval and at
    most 2.5 % above it, so it holds at least 95 %. Both ends belong to it.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0.0) or not values.sum() > 0.0:
        raise ValueError("a count's distribution is a list of probabilities, one per count from 0")
    tail = (1.0 - level) / 2.0
    if not 0.0 < tail < 0.5:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    low, high = _quantile_indices(values, (tail, 1.0 - tail))
    return low, high


def lookup_verdict(observed: int, low: int, high: int) -> str:
    """Whether an observed count matches a predicted one: inside the interval (ends included), below it or above."""
    if observed < low:
        return LOOKUP_FALLS_SHORT
    if observed > high:
        return LOOKUP_EXCEEDS
    return LOOKUP_MATCHES


def lookup_prediction(
    posterior: ThresholdPosterior, distances_mm: Sequence[float], observed: int, level: float = 0.95
) -> dict[str, Any]:
    """The lookup prediction for a set of placements: the count a fitted curve expects at their distances.

    `expected` is the sum of the posterior-mean success probabilities. `lo` and `hi` are the interval of the
    predicted count (`count_interval`) from the count's whole distribution (`ThresholdPosterior.count_distribution`),
    which carries both the uncertainty of the fit and the chance in the trials. `pmf` is that distribution.
    """
    distances = np.asarray(distances_mm, dtype=float)
    if not 0 <= observed <= distances.size:
        raise ValueError("the observed count lies between 0 and the number of placements")
    probabilities = posterior.count_distribution(distances)
    low, high = count_interval(probabilities, level)
    return {
        "expected": float(posterior.predict(distances).sum()),
        "lo": low,
        "hi": high,
        "observed": int(observed),
        "verdict": lookup_verdict(int(observed), low, high),
        "pmf": probabilities.tolist(),
    }


# --------------------------------------------------------------------------------------------------------------------
# Reset sensitivity
# --------------------------------------------------------------------------------------------------------------------


def mean_success_at(posterior: ThresholdPosterior, errors_mm: Sequence[float]) -> float | None:
    """The success a fitted curve gives when a trial at a demonstration's own start is reset with the given errors.

    A reset that lands `e` millimetres from the demonstration's start is a trial at distance `e`, so this is the
    posterior-mean curve averaged over the errors. A calculation, not a measurement. None when there is no error to
    average over.
    """
    errors = np.asarray(errors_mm, dtype=float)
    if errors.size == 0:
        return None
    return float(posterior.predict(errors).mean())


# --------------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------------


def stable(value: float) -> float:
    """A number as the results hold it: rounded to `SIGNIFICANT_DIGITS` significant digits.

    That is far finer than anything measured or reported. It keeps the last digits of a sum, which differ from one
    build of NumPy to another, out of the results, so that the same inputs give the same file with either of the
    project's interpreters. Distances are rounded this way before they are binned and fitted, so the trials listed
    in the results are exactly the numbers the analysis used.
    """
    return float(f"{float(value):.{SIGNIFICANT_DIGITS}g}")


def median(values: Sequence[float]) -> float | None:
    """The median of some numbers, or None when there is none."""
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    middle = len(numbers) // 2
    return numbers[middle] if len(numbers) % 2 else 0.5 * (numbers[middle - 1] + numbers[middle])


def percentile(values: Sequence[float], percent: float) -> float | None:
    """A percentile by linear interpolation between the two nearest ranks (NumPy's default), or None without values.

    With `n` sorted values the percentile sits at rank `(n - 1) * percent / 100`, counted from 0.
    """
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    if not 0.0 <= percent <= 100.0:
        raise ValueError("a percentile lies between 0 and 100")
    rank = (len(numbers) - 1) * percent / 100.0
    below = math.floor(rank)
    above = min(below + 1, len(numbers) - 1)
    return numbers[below] + (numbers[above] - numbers[below]) * (rank - below)


def _same_length(first: Sequence[Any], second: Sequence[Any]) -> None:
    if len(first) != len(second):
        raise ValueError(f"{len(first)} values against {len(second)}: one outcome for every trial")


def _quantile_indices(probabilities: np.ndarray, marks: tuple[float, ...]) -> list[int]:
    """For each mark, the index of the first value at which the running total of `probabilities` reaches it."""
    running = np.cumsum(probabilities)
    targets = np.asarray(marks) * running[-1] * (1.0 - _QUANTILE_SLACK)
    return np.minimum(np.searchsorted(running, targets, side="left"), probabilities.size - 1).tolist()
