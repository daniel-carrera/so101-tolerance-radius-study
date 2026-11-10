"""The success curve and the estimate of the tolerance radius.

The model
---------
Success falls with the distance `d` from a trial's start position to the nearest demonstration:

    P(success at d) = floor + (1 - lapse - floor) / (1 + exp((d - r50) / width))

* `r50`, the tolerance radius, is the distance at which success is halfway between its plateau and its floor;
* `width` is the scale of the fall: a small width is a cliff, a large one a slope. Success goes from three quarters
  of the way up to one quarter over `2.2 * width`;
* `lapse` is the share of failures that have nothing to do with distance. It lowers the plateau to `1 - lapse`;
* `floor` is the chance of a lucky success far from every demonstration. It is what remains of the curve far out.
  Without it (`floor = 0`, the default everywhere) a single lucky success far out can only be explained by a wider
  or a farther curve, and moves the radius.

The estimate
------------
A posterior on a grid: the probability of every combination of `r50`, `width`, `lapse` and `floor` on four lists of
values, given the trials. Every combination has the same weight before the data. With an evenly spaced `r50` grid and
a logarithmically spaced `width` grid, as `grids_from_protocol` builds them, that is a flat prior on `r50`, a flat
prior on `log(width)`, and equal weights on the lapse values and on the floor values.

No optimiser is involved, so the estimate exists and is the same on every machine whatever the data: all successes,
all failures, successes and failures perfectly separated, or no trials at all (the posterior is then the prior).

What is reported (`ThresholdPosterior.summary`) is the posterior median of `r50` with an equal-tailed interval: by
default the values that leave 2.5 % of the probability below and 2.5 % above. Each reported value is a value of the
grid: the first one at which the running total of the probability reaches its mark.

When the data do not bound the radius
-------------------------------------
If every trial succeeds, nothing says where success ends, and the posterior stays level all the way to the top of the
grid. The interval's upper bound then only says where the grid stops. `summary` flags this: an end of the `r50` grid
counts as reached when the posterior probability of the grid's end value is at least `EDGE_RATIO` (one half) of that
at the interval's own bound on that side. The posterior has then not come down between the bound and the end of the
grid, and the report must say that the data do not bound the radius on that side.

Standard library and NumPy only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

#: An end of the r50 grid counts as reached when the posterior probability there is at least this share of the
#: probability at the interval's own bound on that side.
EDGE_RATIO = 0.5

#: Trials handled at a time when the likelihood is built, and distances at a time when a curve is predicted.
#: They bound the working memory (a few megabytes with the protocol's grids); they do not change any result.
_TRIALS_PER_BLOCK = 64
_DISTANCES_PER_BLOCK = 128
#: The smallest value a product of probabilities may reach before it is turned into a logarithm, as an exponent of e.
_SAFE_EXPONENT = 690.0
#: The largest exponent for which `exp(z)` is built from two factors (see `_split_exponential`): exp(300) squared is
#: still a finite number.
_SPLIT_EXPONENT_LIMIT = 300.0
#: A running total counts as having reached its mark when it is within this relative margin of it, so that an exact
#: tie (half of thirty equal weights, say) does not depend on the last bit of a sum.
_QUANTILE_SLACK = 1e-12
#: The floor values of a model without a floor.
_NO_FLOOR = np.zeros(1)
_NO_FLOOR.setflags(write=False)


def success_probability(
    distance_mm: ArrayLike, r50_mm: ArrayLike, width_mm: ArrayLike, lapse: ArrayLike = 0.0, floor: ArrayLike = 0.0
) -> np.ndarray:
    """The model: the probability that a trial succeeds at each distance.

    The arguments broadcast against each other, so one call can give a curve over many distances or one distance over a
    grid of parameters. Any distance is safe, however far from `r50` in units of `width`.
    """
    distance = np.asarray(distance_mm, dtype=float)
    r50 = np.asarray(r50_mm, dtype=float)
    width, lapse_rate, floor_level = _checked_curve(width_mm, lapse, floor)
    with np.errstate(over="ignore", under="ignore"):
        steps_beyond = (distance - r50) / width
        # 1 / (1 + exp(z)), written so that a large z gives 0 and not an overflow.
        fall = np.exp(-np.logaddexp(0.0, steps_beyond))
        return np.asarray(floor_level + (1.0 - lapse_rate - floor_level) * fall)


def distance_at_probability(
    probability: ArrayLike, r50_mm: ArrayLike, width_mm: ArrayLike, lapse: ArrayLike = 0.0, floor: ArrayLike = 0.0
) -> np.ndarray:
    """The model read the other way: the distance at which success has the given probability.

    Success falls with distance, so a trial at distance `d` succeeds with probability above `p` exactly when `d` is
    less than this distance. That lets a simulation draw one number per trial in advance and decide the outcome once
    the distance is known. Where `p` is at or above the plateau `1 - lapse`, no distance is that good and the result is
    minus infinity; where `p` is at or below the floor, every distance is and the result is plus infinity.
    """
    wanted = np.asarray(probability, dtype=float)
    r50 = np.asarray(r50_mm, dtype=float)
    width, lapse_rate, floor_level = _checked_curve(width_mm, lapse, floor)
    if np.any(wanted < 0.0) or np.any(wanted > 1.0):
        raise ValueError("probability must lie between 0 and 1")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        odds = (1.0 - lapse_rate - floor_level) / (wanted - floor_level) - 1.0  # exp(z) at that probability
        between = (wanted > floor_level) & (odds > 0.0)
        steps_beyond = np.log(np.where(between, odds, 1.0))
        outside = np.where(wanted <= floor_level, np.inf, -np.inf)
        return np.asarray(np.where(between, r50 + width * steps_beyond, outside))


@dataclass(frozen=True, slots=True)
class ThresholdSummary:
    """What a fit reports: the tolerance radius with its interval, and what else the trials say."""

    #: The share of posterior probability the intervals hold, for example 0.95.
    level: float
    n_trials: int
    n_success: int
    #: The posterior median of the tolerance radius, and the bounds of its equal-tailed interval.
    r50_median_mm: float
    r50_low_mm: float
    r50_high_mm: float
    #: True when the posterior has not come down at that end of the r50 grid (see `EDGE_RATIO`): the data do not
    #: bound the radius on that side, and the bound there only says where the grid stops.
    reaches_lower_edge: bool
    reaches_upper_edge: bool
    width_median_mm: float
    width_low_mm: float
    width_high_mm: float
    #: The posterior mean of the lapse rate.
    lapse_mean: float
    #: The posterior mean of the floor; 0 when the model has no floor.
    floor_mean: float = 0.0

    @property
    def r50_interval_width_mm(self) -> float:
        """The length of the interval of the tolerance radius."""
        return self.r50_high_mm - self.r50_low_mm


@dataclass(frozen=True, slots=True, eq=False)
class ThresholdPosterior:
    """The posterior probability of every combination of `r50`, `width`, `lapse` and `floor` on the grids.

    Built by `fit_threshold`. Its arrays are read-only.
    """

    r50_grid_mm: np.ndarray
    width_grid_mm: np.ndarray
    lapse_values: np.ndarray
    floor_values: np.ndarray
    #: Shape `(r50, width, lapse, floor)`; sums to 1.
    probabilities: np.ndarray
    n_trials: int
    n_success: int

    def r50_marginal(self) -> tuple[np.ndarray, np.ndarray]:
        """The grid of `r50` and the posterior probability of each of its values, whatever the other three."""
        return self.r50_grid_mm, self._marginals()[0]

    def width_marginal(self) -> tuple[np.ndarray, np.ndarray]:
        """The grid of `width` and the posterior probability of each of its values."""
        return self.width_grid_mm, self._marginals()[1]

    def lapse_marginal(self) -> tuple[np.ndarray, np.ndarray]:
        """The lapse values and the posterior probability of each."""
        return self.lapse_values, self._marginals()[2]

    def floor_marginal(self) -> tuple[np.ndarray, np.ndarray]:
        """The floor values and the posterior probability of each."""
        return self.floor_values, self._marginals()[3]

    def _marginals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The posterior probability of each value of `r50`, of `width`, of `lapse` and of `floor`.

        The joint posterior is first summed over the two rates and over the two shape parameters, which NumPy does
        several times faster than four sums that each skip one axis.
        """
        by_shape = self.probabilities.sum(axis=(2, 3))  # (r50, width)
        by_rates = self.probabilities.sum(axis=(0, 1))  # (lapse, floor)
        return by_shape.sum(axis=1), by_shape.sum(axis=0), by_rates.sum(axis=1), by_rates.sum(axis=0)

    def summary(self, level: float = 0.95) -> ThresholdSummary:
        """The numbers to report: medians, equal-tailed intervals at `level`, and whether the data bound `r50`."""
        tail = _tail(level)
        marks = (tail, 0.5, 1.0 - tail)
        r50_probability, width_probability, lapse_probability, floor_probability = self._marginals()
        low, median, high = _quantile_indices(r50_probability, marks)
        width_low, width_median, width_high = _quantile_indices(width_probability, marks)
        return ThresholdSummary(
            level=float(level),
            n_trials=self.n_trials,
            n_success=self.n_success,
            r50_median_mm=float(self.r50_grid_mm[median]),
            r50_low_mm=float(self.r50_grid_mm[low]),
            r50_high_mm=float(self.r50_grid_mm[high]),
            reaches_lower_edge=bool(r50_probability[0] >= EDGE_RATIO * r50_probability[low]),
            reaches_upper_edge=bool(r50_probability[-1] >= EDGE_RATIO * r50_probability[high]),
            width_median_mm=float(self.width_grid_mm[width_median]),
            width_low_mm=float(self.width_grid_mm[width_low]),
            width_high_mm=float(self.width_grid_mm[width_high]),
            lapse_mean=float(lapse_probability @ self.lapse_values),
            floor_mean=float(floor_probability @ self.floor_values),
        )

    def predict(self, distances_mm: ArrayLike) -> np.ndarray:
        """The posterior-mean probability of success at each distance: every curve of the grid, weighted by its
        posterior probability. The result has the shape of `distances_mm`.
        """
        distances = np.asarray(distances_mm, dtype=float)
        flat = distances.reshape(-1)
        # A curve is `(1 - lapse) * fall + floor * (1 - fall)`, where `fall` is the curve without lapse or floor, so
        # the lapse and floor values can be summed out first.
        on_plateau = (self.probabilities * (1.0 - self.lapse_values)[:, None]).sum(axis=(2, 3))
        on_floor = (self.probabilities * self.floor_values).sum(axis=(2, 3))
        weights = on_plateau - on_floor
        far_out = on_floor.sum()
        prediction = np.empty(flat.size)
        for start in range(0, flat.size, _DISTANCES_PER_BLOCK):
            block = flat[start : start + _DISTANCES_PER_BLOCK]
            falls = success_probability(
                block[:, None, None], self.r50_grid_mm[None, :, None], self.width_grid_mm[None, None, :]
            )
            prediction[start : start + block.size] = np.tensordot(falls, weights, axes=([1, 2], [0, 1])) + far_out
        return np.clip(prediction, 0.0, 1.0).reshape(distances.shape)

    def band(self, distances_mm: ArrayLike, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """A pointwise interval of the success curve, for plotting: `(low, high)`, each shaped like `distances_mm`.

        At each distance, taken on its own, the curve's value lies between `low` and `high` with posterior
        probability `level`, with equal shares outside on each side. It is not a band for the curve as a whole.
        """
        tail = _tail(level)
        distances = np.asarray(distances_mm, dtype=float)
        flat = distances.reshape(-1)
        weights = self.probabilities.reshape(-1)
        span = 1.0 - self.lapse_values[:, None] - self.floor_values[None, :]  # plateau minus floor
        low, high = np.empty(flat.size), np.empty(flat.size)
        for index, distance in enumerate(flat):
            falls = success_probability(distance, self.r50_grid_mm[:, None], self.width_grid_mm[None, :])
            values = (falls[:, :, None, None] * span + self.floor_values).reshape(-1)
            order = np.argsort(values, kind="stable")
            at_low, at_high = _quantile_indices(weights[order], (tail, 1.0 - tail))
            low[index], high[index] = values[order[at_low]], values[order[at_high]]
        return low.reshape(distances.shape), high.reshape(distances.shape)

    def count_distribution(self, distances_mm: ArrayLike) -> np.ndarray:
        """The predicted number of successes among trials at the given distances, as a probability for every count.

        Entry `k` of the result is the probability that exactly `k` of the trials succeed, for `k` from 0 to the
        number of distances. Under one curve of the grid the trials are independent, each with its own probability,
        so their count follows a Poisson-binomial distribution. The result is the mixture of those distributions
        over the whole grid, each weighted by its curve's posterior probability. It therefore carries both what the
        fit leaves uncertain and the chance in the trials themselves, which a count predicted from the posterior-mean
        curve alone would understate.

        Computed exactly, up to rounding: no random numbers. The order of the distances does not matter, down to the
        last bit. With no distances the count is certainly 0.
        """
        distances = np.sort(np.asarray(distances_mm, dtype=float).reshape(-1))
        if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
            raise ValueError("a distance is a finite number, zero or more")
        if distances.size == 0:
            return np.ones(1)
        span = 1.0 - self.lapse_values[:, None] - self.floor_values[None, :]  # plateau minus floor
        # counts[k] holds, for every curve of the grid, the probability of k successes among the trials seen so far.
        counts = np.zeros((distances.size + 1, *self.probabilities.shape))
        counts[0] = 1.0
        for seen, distance in enumerate(distances.tolist()):
            falls = success_probability(distance, self.r50_grid_mm[:, None], self.width_grid_mm[None, :])
            success = falls[:, :, None, None] * span + self.floor_values  # this trial, under every curve
            one_more = counts[: seen + 1] * success
            counts[: seen + 2] *= 1.0 - success
            counts[1 : seen + 2] += one_more
        return np.tensordot(counts, self.probabilities, axes=self.probabilities.ndim)


def fit_threshold(
    distances_mm: ArrayLike,
    successes: ArrayLike,
    *,
    r50_grid_mm: ArrayLike,
    width_grid_mm: ArrayLike,
    lapse_values: ArrayLike,
    floor_values: ArrayLike = (0.0,),
) -> ThresholdPosterior:
    """The posterior of the success curve given a set of trials.

    `distances_mm` holds each trial's measured distance to the nearest demonstration and `successes` its outcome
    (booleans, or 0 and 1). Every combination of the four grids has the same prior weight (see the module's
    docstring). Without `floor_values` the model has no floor. The order of the trials does not matter, down to the
    last bit. With no trials the posterior is the prior.
    """
    distances, outcomes = _checked_trials(distances_mm, successes)
    r50, width, lapse, floor = _checked_grids(r50_grid_mm, width_grid_mm, lapse_values, floor_values)
    log_likelihood = _log_likelihood(distances, outcomes, r50, width, lapse, floor)
    peak = log_likelihood.max()
    if not np.isfinite(peak):
        raise ValueError("these trials are impossible under every curve of the grid")
    with np.errstate(under="ignore"):
        probabilities = np.exp(log_likelihood - peak)
    probabilities /= probabilities.sum()
    for array in (r50, width, lapse, floor, probabilities):
        array.setflags(write=False)
    return ThresholdPosterior(
        r50_grid_mm=r50,
        width_grid_mm=width,
        lapse_values=lapse,
        floor_values=floor,
        probabilities=probabilities,
        n_trials=int(distances.size),
        n_success=int(np.count_nonzero(outcomes)),
    )


def log_likelihood_grid(
    distances_mm: ArrayLike,
    successes: ArrayLike,
    *,
    r50_grid_mm: ArrayLike,
    width_grid_mm: ArrayLike,
    lapse_values: ArrayLike,
    floor_values: ArrayLike = (0.0,),
) -> np.ndarray:
    """The logarithm of the probability of the observed outcomes under every curve of the grid.

    Shape `(r50, width, lapse, floor)`. `fit_threshold` is this, exponentiated and rescaled to sum to 1.
    """
    distances, outcomes = _checked_trials(distances_mm, successes)
    r50, width, lapse, floor = _checked_grids(r50_grid_mm, width_grid_mm, lapse_values, floor_values)
    return _log_likelihood(distances, outcomes, r50, width, lapse, floor)


class AnalysisGrids(NamedTuple):
    """The lists of values a posterior is computed on. `fit_threshold(d, s, **grids._asdict())` uses them."""

    r50_grid_mm: np.ndarray
    width_grid_mm: np.ndarray
    lapse_values: np.ndarray
    #: A single 0 for a model without a floor.
    floor_values: np.ndarray = _NO_FLOOR


def grids_from_protocol(analysis: dict[str, Any]) -> AnalysisGrids:
    """The grids described by the `[analysis]` section of the protocol file.

    `r50_grid_mm = {min, max, step}` gives evenly spaced values from `min` to `max` inclusive (a flat prior on `r50`);
    `width_grid_mm = {min, max, count}` gives `count` values from `min` to `max` with a constant ratio (a flat prior
    on `log(width)`); `lapse_values` is taken as listed, and so is `floor_values` when the section has it. A section
    without `floor_values` describes a model without a floor.
    """
    try:
        r50_min, r50_max, r50_step = (float(analysis["r50_grid_mm"][key]) for key in ("min", "max", "step"))
        width_min, width_max = (float(analysis["width_grid_mm"][key]) for key in ("min", "max"))
        width_count = analysis["width_grid_mm"]["count"]
        lapse = np.array([float(value) for value in analysis["lapse_values"]])
        stated_floors = analysis.get("floor_values", [0.0])
        if not isinstance(stated_floors, (list, tuple)):
            raise TypeError("floor_values must be a list")
        floor = np.array([float(value) for value in stated_floors])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "the analysis section needs r50_grid_mm = {min, max, step}, width_grid_mm = {min, max, count}, "
            f"lapse_values and, if the model has a floor, a list floor_values (problem: {error!r})"
        ) from error
    if r50_step <= 0.0 or r50_max <= r50_min:
        raise ValueError("r50_grid_mm needs min < max and a positive step")
    intervals = (r50_max - r50_min) / r50_step
    if abs(intervals - round(intervals)) > 1e-9:
        raise ValueError("r50_grid_mm: a whole number of steps must lead from min to max")
    r50_grid = r50_min + r50_step * np.arange(round(intervals) + 1)
    r50_grid[-1] = r50_max
    if isinstance(width_count, bool) or not isinstance(width_count, int) or width_count < 2:
        raise ValueError("width_grid_mm needs a whole count of at least 2")
    if width_min <= 0.0 or width_max <= width_min:
        raise ValueError("width_grid_mm needs 0 < min < max")
    grids = AnalysisGrids(r50_grid, np.geomspace(width_min, width_max, width_count), lapse, floor)
    _checked_grids(*grids)
    return grids


# --------------------------------------------------------------------------------------------------------------------
# The likelihood
# --------------------------------------------------------------------------------------------------------------------


def _log_likelihood(
    distances: np.ndarray,
    outcomes: np.ndarray,
    r50: np.ndarray,
    width: np.ndarray,
    lapse: np.ndarray,
    floor: np.ndarray,
) -> np.ndarray:
    """Log-likelihood on the grid, shape `(r50, width, lapse, floor)`, computed without overflow or cancellation.

    Write `z = (d - r50) / width` and `t = exp(-|z|)`, which lies between 0 and 1 whatever `z`, and let `near` be
    `t` where the trial lies short of `r50` (`z < 0`) and 1 elsewhere, and `far` be 1 where it lies short and `t`
    elsewhere. Then `near + far = 1 + t` and

        P(success) = ((1 - lapse) * far + floor * near) / (1 + t)
        P(failure) = (lapse * far + (1 - floor) * near) / (1 + t)

    Every term is positive, so nothing cancels. Taking `1 - lapse` out of a success and `1 - floor` out of a failure
    leaves, per trial,

        success:  far + (floor / (1 - lapse)) * near      which is  exp(-max(z, 0))  when floor = 0
        failure:  near + (lapse / (1 - floor)) * far      which is  exp(min(z, 0))   when lapse = 0

    The two special cases are sums of distances. In every other case each factor is at least the smallest positive
    lapse or floor value and at most 2, so a block of trials is multiplied first and its logarithm taken once.

    Inside, the arrays run over `(trials, width, r50)`: the longest axis last, which is what NumPy loops over fastest.
    """
    # A fixed order of the trials, so that the result does not depend on the order they were given in.
    order = np.lexsort((outcomes, distances))
    distances, outcomes = distances[order], outcomes[order]
    inverse_width = (1.0 / width)[:, None]  # shape (width, 1)
    block = _trials_per_block(lapse, floor)
    n_success = np.count_nonzero(outcomes)
    result = np.zeros((lapse.size, floor.size, width.size, r50.size))
    result += (n_success * np.log1p(-lapse))[:, None, None, None]
    result += ((outcomes.size - n_success) * np.log1p(-floor))[None, :, None, None]
    with np.errstate(over="ignore", under="ignore"):
        split = _split_exponential(distances, r50, inverse_width)
        for group in _blocks(distances[outcomes], block):
            offset = group[:, None] - r50  # d - r50, shape (trials, r50)
            decay = _decay(group, offset, inverse_width, split)
            shared = np.log(np.multiply.reduce(1.0 + decay, axis=0))
            beyond = np.maximum(offset, 0.0).sum(axis=0) * inverse_width  # sum of max(z, 0)
            near, far, work = _near_and_far(offset, decay) if np.any(floor > 0.0) else (None, None, None)
            for floor_index, level in enumerate(floor):
                if level == 0.0:
                    result[:, floor_index] -= beyond + shared
                    continue
                for lapse_index, rate in enumerate(lapse):
                    np.multiply(near, level / (1.0 - rate), out=work)
                    work += far
                    result[lapse_index, floor_index] += np.log(np.multiply.reduce(work, axis=0)) - shared
        for group in _blocks(distances[~outcomes], block):
            offset = group[:, None] - r50
            decay = _decay(group, offset, inverse_width, split)
            shared = np.log(np.multiply.reduce(1.0 + decay, axis=0))
            shortfall = np.minimum(offset, 0.0).sum(axis=0) * inverse_width  # sum of min(z, 0)
            near, far, work = _near_and_far(offset, decay) if np.any(lapse > 0.0) else (None, None, None)
            for lapse_index, rate in enumerate(lapse):
                for floor_index, level in enumerate(floor):
                    if rate == 0.0:
                        numerator = shortfall
                    else:
                        np.multiply(far, rate / (1.0 - level), out=work)
                        work += near
                        numerator = np.log(np.multiply.reduce(work, axis=0))
                    result[lapse_index, floor_index] += numerator - shared
    return np.ascontiguousarray(result.transpose(3, 2, 0, 1))


def _near_and_far(offset: np.ndarray, decay: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`near` and `far` of `_log_likelihood` for a block of trials, and an array of the same shape to work in."""
    short = (offset < 0.0)[:, None, :]  # the trial lies short of r50
    return np.where(short, decay, 1.0), np.where(short, 1.0, decay), np.empty_like(decay)


def _split_exponential(
    distances: np.ndarray, r50: np.ndarray, inverse_width: np.ndarray
) -> tuple[float, np.ndarray] | None:
    """What the quick route to `exp(z)` needs, or None where that route could overflow.

    `exp((d - r50) / width)` is `exp((d - c) / width) * exp((c - r50) / width)` for any `c`: one factor per trial and
    one per grid value, in place of an exponential for every trial at every grid value. With `c` in the middle of the
    distances and the grid, neither exponent exceeds half their span divided by the smallest width. The route is
    taken when that is at most `_SPLIT_EXPONENT_LIMIT`, and then nothing can overflow. Returns `c` and the grid's
    factor, shape `(width, r50)`. `distances` are sorted.
    """
    if distances.size == 0:
        return None
    low, high = min(r50[0], distances[0]), max(r50[-1], distances[-1])
    centre = 0.5 * (low + high)
    if not (high - centre) * inverse_width.max() <= _SPLIT_EXPONENT_LIMIT:
        return None
    return float(centre), np.exp((centre - r50) * inverse_width)


def _decay(
    group: np.ndarray, offset: np.ndarray, inverse_width: np.ndarray, split: tuple[float, np.ndarray] | None
) -> np.ndarray:
    """`t = exp(-|d - r50| / width)` for a block of trials, shape `(trials, width, r50)`."""
    if split is None:
        return np.exp(-np.abs(offset)[:, None, :] * inverse_width)
    centre, grid_factor = split
    odds = np.exp((group - centre)[:, None] * inverse_width[:, 0])[:, :, None] * grid_factor  # exp(z)
    return np.minimum(odds, 1.0 / odds)


def _trials_per_block(lapse: np.ndarray, floor: np.ndarray) -> int:
    """How many trials may be multiplied before a logarithm is taken, so that the product cannot underflow.

    Every factor of a product is at least the smallest positive lapse or floor value, so a block of `n` factors is
    at least that value to the power `n`. Without a positive value no product of small numbers is formed.
    """
    rates = np.concatenate((lapse, floor))
    positive = rates[rates > 0.0]
    if positive.size == 0:
        return _TRIALS_PER_BLOCK
    return int(max(1, min(_TRIALS_PER_BLOCK, _SAFE_EXPONENT // -np.log(positive.min()))))


def _blocks(values: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, values.size, size):
        yield values[start : start + size]


# --------------------------------------------------------------------------------------------------------------------
# Checks and small helpers
# --------------------------------------------------------------------------------------------------------------------


def _checked_curve(
    width_mm: ArrayLike, lapse: ArrayLike, floor: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = np.asarray(width_mm, dtype=float)
    lapse_rate = np.asarray(lapse, dtype=float)
    floor_level = np.asarray(floor, dtype=float)
    if np.any(width <= 0.0):
        raise ValueError("width_mm must be positive")
    if np.any(lapse_rate < 0.0) or np.any(floor_level < 0.0) or np.any(lapse_rate + floor_level >= 1.0):
        raise ValueError("lapse and floor must be at least 0 and add up to less than 1")
    return width, lapse_rate, floor_level


def _checked_trials(distances_mm: ArrayLike, successes: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(distances_mm, dtype=float)
    given = np.asarray(successes)
    if distances.ndim != 1:
        raise ValueError("distances_mm must be one list of distances, one per trial")
    if given.shape != distances.shape:
        raise ValueError(f"{distances.size} distances but outcomes of shape {given.shape}: one outcome per trial")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("a distance is a finite number, zero or more")
    if given.dtype != bool and given.size:
        if not np.issubdtype(given.dtype, np.number) or not np.all((given == 0) | (given == 1)):
            raise ValueError("an outcome is True or False, or 1 or 0")
    return distances, given.astype(bool)


def _checked_grids(
    r50_grid_mm: ArrayLike, width_grid_mm: ArrayLike, lapse_values: ArrayLike, floor_values: ArrayLike = (0.0,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r50 = np.array(r50_grid_mm, dtype=float)
    width = np.array(width_grid_mm, dtype=float)
    lapse = np.array(lapse_values, dtype=float)
    floor = np.array(floor_values, dtype=float)
    for name, grid in (
        ("r50_grid_mm", r50),
        ("width_grid_mm", width),
        ("lapse_values", lapse),
        ("floor_values", floor),
    ):
        if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError(f"{name} must be a list of at least one finite number")
    if np.any(np.diff(r50) <= 0.0):
        raise ValueError("r50_grid_mm must increase")
    if width[0] <= 0.0 or np.any(np.diff(width) <= 0.0):
        raise ValueError("width_grid_mm must be positive and increase")
    for name, rates in (("lapse_values", lapse), ("floor_values", floor)):
        if np.any(rates < 0.0) or np.any(rates >= 1.0) or np.unique(rates).size != rates.size:
            raise ValueError(f"{name} must be distinct, at least 0 and less than 1")
    if lapse.max() + floor.max() >= 1.0:
        raise ValueError("the largest lapse and the largest floor must add up to less than 1")
    return r50, width, lapse, floor


def _tail(level: float) -> float:
    """The share of probability an interval at `level` leaves outside on each side."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    return (1.0 - level) / 2.0


def _quantile_indices(probabilities: np.ndarray, marks: tuple[float, ...]) -> list[int]:
    """For each mark, the index of the first value at which the running total of `probabilities` reaches it."""
    running = np.cumsum(probabilities)
    targets = np.asarray(marks) * running[-1] * (1.0 - _QUANTILE_SLACK)
    return np.minimum(np.searchsorted(running, targets, side="left"), probabilities.size - 1).tolist()
