"""Intervals and exact tests for small counts of successes.

The analysis reports proportions from a handful of trials: success counts in bins of distance, each policy's success in
the paired workspace test, and the paired test itself. This module holds what those need:

* `wilson_interval`: an interval for a proportion. Unlike the textbook "share plus or minus two standard errors" it
  stays inside 0 to 1 and does not collapse to a point when every trial succeeds or every trial fails.
* `mcnemar_exact`: the exact test for a paired comparison. It looks only at the pairs in which the two conditions
  disagreed (the discordant pairs) and asks how surprising their split is if neither condition is better.
* `binomial_two_sided` and `normal_quantile`: what the two above are built from.

Standard library only, so that it runs in the rig's environment, which has no SciPy.

Methods and accuracy
--------------------
* The normal quantile is the standard library's `statistics.NormalDist.inv_cdf`, which implements Wichura's algorithm
  AS 241 (1988), accurate to about 16 significant digits.
* Binomial probabilities are summed as whole numbers over a common denominator (`p` is read as the exact fraction its
  float stands for) and divided once, at the end. Nothing overflows or cancels, whatever the counts: the result is the
  exact sum, correctly rounded.
* The Wilson bounds are a closed formula evaluated in floating point; for the counts of this study they agree with
  SciPy to 12 decimal places.

`tests/test_intervals.py` checks each against values worked by hand, against its defining property, and against SciPy
where SciPy is installed.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral
from statistics import NormalDist

#: Outcomes whose probability is within one part in this many of the observed outcome's count as equally likely.
#: SciPy's `binomtest` uses the same margin, so that a tie is not broken by the rounding of `p` to a float.
_TIE_PARTS = 10**7


def normal_quantile(probability: float) -> float:
    """The value below which a standard normal variable falls with the given probability.

    For example `normal_quantile(0.975)` is 1.96, the multiplier of a two-sided 95 % interval.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must lie strictly between 0 and 1, got {probability!r}")
    return NormalDist().inv_cdf(probability)


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """The Wilson score interval for a proportion: `(low, high)`, both between 0 and 1.

    It is the set of true proportions that a score test at this level would not reject given the observed count.
    With no trials nothing is known and the interval is `(0.0, 1.0)`.
    """
    _check_counts(successes, trials)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    if trials == 0:
        return (0.0, 1.0)
    successes, trials = int(successes), int(trials)
    z = normal_quantile(0.5 + level / 2.0)
    share = successes / trials
    shrink = 1.0 + z * z / trials
    centre = (share + z * z / (2.0 * trials)) / shrink
    half_width = z * math.sqrt(share * (1.0 - share) / trials + z * z / (4.0 * trials * trials)) / shrink
    # At 0 and at `trials` successes the formula gives exactly 0 and 1; say so without the rounding error.
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == trials else min(1.0, centre + half_width)
    return (low, high)


def binomial_two_sided(successes: int, trials: int, p: float) -> float:
    """The exact two-sided p-value of a count of successes, if each trial succeeds with probability `p`.

    It is the total probability of every outcome that is no more likely than the observed one. With no trials there is
    no evidence and the value is 1.
    """
    _check_counts(successes, trials)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie between 0 and 1, got {p!r}")
    if trials == 0:
        return 1.0
    successes, trials = int(successes), int(trials)  # plain integers: a NumPy integer would overflow in the powers
    chance = Fraction(p)
    odds_for, odds_against = chance.numerator, chance.denominator - chance.numerator
    powers_for, powers_against = [1], [1]
    for _ in range(trials):
        powers_for.append(powers_for[-1] * odds_for)
        powers_against.append(powers_against[-1] * odds_against)
    # Each outcome's probability, as a whole number of parts out of `chance.denominator ** trials`.
    parts = []
    ways = 1  # the number of ways to place k successes among the trials, built up from k = 0
    for k in range(trials + 1):
        parts.append(ways * powers_for[k] * powers_against[trials - k])
        ways = ways * (trials - k) // (k + 1)
    limit = parts[successes] * (_TIE_PARTS + 1)
    return sum(part for part in parts if part * _TIE_PARTS <= limit) / chance.denominator**trials


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """The exact two-sided McNemar p-value for a paired comparison of two conditions.

    `only_first` counts the pairs in which the first condition succeeded and the second failed, `only_second` the
    reverse. Pairs in which both succeeded or both failed carry no information about the difference and are left out.
    If neither condition is better, each discordant pair is a fair coin, so the p-value is that of `only_first` heads
    in `only_first + only_second` tosses. With no discordant pairs it is 1.
    """
    _check_counts(only_first, only_first + only_second)
    _check_counts(only_second, only_first + only_second)
    return binomial_two_sided(only_first, only_first + only_second, 0.5)


def _check_counts(successes: int, trials: int) -> None:
    if not isinstance(successes, Integral) or not isinstance(trials, Integral):
        raise ValueError(f"counts must be whole numbers, got {successes!r} of {trials!r}")
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError(f"a count must lie between 0 and the number of trials, got {successes!r} of {trials!r}")
