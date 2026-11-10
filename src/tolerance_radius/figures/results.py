"""Reading `analysis/report/results.json`, the one file Figures 3 and 4 take the experiment from.

`protocol/data-contracts.md` fixes the file's shape. This module reads the parts the figures use and nothing else, and
says plainly what is wrong when a part is missing or is not what the contract says: a figure drawn from a half-read
file would be worse than no figure.

Four fields that the contract's example does not have are used when they are there and done without when they are
not, because the analysis is expected to add them: a pair's `region`, the counts `workspace.by_region` (for each
region and policy `k`, `n`, and when given `lookup_expected` and the interval `lookup_lo`, `lookup_hi`), each
policy's own placement in a pair (`x_mm`, `y_mm` beside its `success`), and `distance_scale.divisor`. Nothing else
outside the contract is looked for.

Where the file says one thing twice (a count and the trials it counts), both are read and compared. A disagreement
between two fields of the contract is an error; one that involves a field outside it is a sentence for the person
who runs the figures.

Standard library only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tolerance_radius.intervals import wilson_interval

SCHEMA_VERSION = 1
#: The two policies of the workspace test, and every condition a trial can have.
POLICIES = ("act_clustered", "act_spread")
CONDITIONS = ("act_clustered", "replay", "act_spread")
#: The demonstration set a policy was trained on.
DATASET_OF_POLICY = {"act_clustered": "clustered", "act_spread": "spread"}
#: The two fits a condition can have.
FITS = ("block", "pooled")
#: The regions a placement of the workspace test can be drawn from, in the order they are reported.
REGIONS = ("trained_cells", "held_out_cells")
#: Two statements of one expected count may differ by this much, from rounding, before the difference is reported.
EXPECTED_TOLERANCE = 0.05


class ResultsError(ValueError):
    """The results file is not what the contract says, in a part a figure needs."""


@dataclass(frozen=True, slots=True)
class ScoredTrial:
    """One scored trial, as far as a figure needs it."""

    block: str
    #: Distance to the nearest demonstration, as the analysis used it.
    d_mm: float
    success: bool


@dataclass(frozen=True, slots=True)
class Bin:
    """The trials of one stretch of distance (closed below, open above), with the Wilson interval of their rate."""

    lo_mm: float
    hi_mm: float
    n: int
    k: int
    wilson_lo: float
    wilson_hi: float
    #: The mean distance of the bin's trials, when the bin was counted from trials; None for a bin read from the file.
    mean_d_mm: float | None = None

    @property
    def rate(self) -> float:
        return self.k / self.n


@dataclass(frozen=True, slots=True)
class Fit:
    """A fitted success curve: the tolerance radius with its interval, and the curve with its band."""

    #: The blocks whose trials were fitted.
    blocks: tuple[str, ...]
    #: The posterior median of the radius; None only where the file gives none, which it may when a side is open.
    r50_median_mm: float | None
    #: The ends of the interval; None only where the file gives none, which it may for an open side.
    r50_lo_mm: float | None
    r50_hi_mm: float | None
    #: True when the trials do not bound the radius on that side: the bound only says where the prior's grid ends.
    open_below: bool
    open_above: bool
    curve_d_mm: tuple[float, ...]
    curve_mean: tuple[float, ...]
    curve_lo: tuple[float, ...]
    curve_hi: tuple[float, ...]

    @property
    def bounded(self) -> bool:
        """True when the trials bound the radius on both sides, so that its median may be called a radius."""
        return not (self.open_below or self.open_above)

    def mean_at(self, d_mm: float) -> float:
        """The posterior-mean success at a distance, read off the curve between its two nearest points."""
        return interpolate(self.curve_d_mm, self.curve_mean, d_mm)


@dataclass(frozen=True, slots=True)
class Placement:
    """One policy's trial at one placement of the workspace test."""

    x_mm: float
    y_mm: float
    success: bool
    d_mm: float
    #: `trained_cells` or `held_out_cells`, when the file says; None when it does not.
    region: str | None
    #: The success the lookup prediction gives this trial; None when the file has no prediction.
    p_lookup: float | None = None
    #: True when the position is this policy's own placement, False when it is the pair's shared position.
    own_position: bool = False


@dataclass(frozen=True, slots=True)
class Lookup:
    """The lookup prediction of one policy's number of successes in the workspace test."""

    expected: float
    lo: int
    hi: int
    observed: int
    #: `matches`, `falls short` or `exceeds`.
    verdict: str


@dataclass(frozen=True, slots=True)
class Count:
    """Successes among placements, and what the lookup prediction expects there (None when the file has none)."""

    k: int
    n: int
    lookup_expected: float | None
    #: The interval of the predicted count for these placements alone, when the file gives one.
    lookup_lo: int | None = None
    lookup_hi: int | None = None


@dataclass(frozen=True, slots=True)
class Demonstration:
    """Where one demonstration started, as measured."""

    x_mm: float
    y_mm: float
    trained: bool


# --------------------------------------------------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------------------------------------------------


def read_results(path: str | Path) -> dict[str, Any]:
    """The results file as plain data, after checking that it is one this code can read."""
    path = Path(path)
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ResultsError(f"{path.name} is not a JSON file: {error}") from error
    check_header(results, path.name)
    return results


def check_header(results: Any, name: str = "the results file") -> None:
    """Refuse what is not a results file of the version this code reads."""
    if not isinstance(results, dict):
        raise ResultsError(f"{name} does not hold one JSON object")
    if results.get("schema_version") != SCHEMA_VERSION:
        found = results.get("schema_version")
        raise ResultsError(f"{name}: schema_version is {found!r}, this code reads {SCHEMA_VERSION}")
    if not isinstance(results.get("draft"), bool):
        raise ResultsError(f"{name}: draft must be true or false")


def is_draft(results: Mapping[str, Any]) -> bool:
    """Whether the file is a draft: made from trials that do not count, or made up. Its figures say so."""
    return bool(_need(results, "draft", bool, ""))


def interval_level(results: Mapping[str, Any]) -> float:
    """The share of probability every interval of the file holds, for example 0.95."""
    level = _number(results, "interval_level", "")
    if not 0.0 < level < 1.0:
        raise ResultsError(f"interval_level must lie between 0 and 1, got {level!r}")
    return level


def bin_width_mm(results: Mapping[str, Any]) -> float:
    """The width of the stretches of distance in which success is counted."""
    width = _number(results, "bin_width_mm", "")
    if not width > 0.0:
        raise ResultsError(f"bin_width_mm must be positive, got {width!r}")
    return width


def scale_applied(results: Mapping[str, Any]) -> bool:
    """Whether the file's distances were divided by the instrument's scale factor."""
    return bool(_need(_need(results, "distance_scale", dict, ""), "applied", bool, "distance_scale"))


def distance_divisor(results: Mapping[str, Any]) -> float:
    """What the file's distances were divided by: the scale factor when it was applied, 1 when not.

    Positions are not divided, so a distance of the file is this many times longer between two positions of the file.
    """
    scale = _need(results, "distance_scale", dict, "")
    if isinstance(scale.get("divisor"), (int, float)) and not isinstance(scale.get("divisor"), bool):
        divisor = float(scale["divisor"])
    elif scale_applied(results):
        divisor = _number(scale, "factor", "distance_scale")
    else:
        divisor = 1.0
    if not (math.isfinite(divisor) and 0.5 < divisor < 2.0):
        raise ResultsError(f"distances cannot have been divided by {divisor!r}")
    if not scale_applied(results) and divisor != 1.0:
        raise ResultsError("distance_scale says the correction was not applied and gives a divisor other than 1")
    return divisor


# --------------------------------------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------------------------------------


def trials(results: Mapping[str, Any], condition: str) -> list[ScoredTrial]:
    """The scored trials of a condition, in the file's order; none for a condition the file does not have."""
    entry = _condition(results, condition)
    if entry is None:
        return []
    where = f"conditions.{condition}.trials"
    found = []
    for index, item in enumerate(_need(entry, "trials", list, f"conditions.{condition}")):
        here = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise ResultsError(f"{here} is not an object")
        d_mm = _number(item, "d_mm", here)
        if d_mm < 0.0:
            raise ResultsError(f"{here}.d_mm is negative")
        found.append(ScoredTrial(_need(item, "block", str, here), d_mm, _need(item, "success", bool, here)))
    return found


def fit(results: Mapping[str, Any], condition: str, which: str = "block") -> Fit | None:
    """A condition's block fit or pooled fit; None where the file has none (a block that was not run)."""
    if which not in FITS:
        raise ValueError(f"a fit is one of {FITS}, got {which!r}")
    entry = _condition(results, condition)
    if entry is None:
        return None
    key = f"{which}_fit"
    where = f"conditions.{condition}.{key}"
    if key not in entry:
        raise ResultsError(f"{where} is missing")
    stated = entry[key]
    if stated is None:
        return None
    if not isinstance(stated, dict):
        raise ResultsError(f"{where} is neither an object nor null")
    blocks = _need(stated, "blocks", list, where)
    curve = _need(stated, "curve", dict, where)
    columns = {name: tuple(_numbers(curve, name, f"{where}.curve")) for name in ("d_mm", "mean", "lo", "hi")}
    lengths = {len(column) for column in columns.values()}
    if len(lengths) != 1 or lengths == {0}:
        raise ResultsError(f"{where}.curve: d_mm, mean, lo and hi must be lists of one length, not empty")
    distances = columns["d_mm"]
    if any(later <= earlier for earlier, later in zip(distances, distances[1:], strict=False)):
        raise ResultsError(f"{where}.curve.d_mm must increase")
    for name in ("mean", "lo", "hi"):
        if any(not 0.0 <= value <= 1.0 for value in columns[name]):
            raise ResultsError(f"{where}.curve.{name} holds a value that is not a probability")
    open_below, open_above = (_need(stated, name, bool, where) for name in ("open_below", "open_above"))
    # What the trials do not establish may be null in the file: the end of the interval on an open side, and the
    # radius itself when a side is open. Everything else must be a number.
    low = _number_unless(open_below, stated, "r50_lo_mm", where)
    high = _number_unless(open_above, stated, "r50_hi_mm", where)
    median = _number_unless(open_below or open_above, stated, "r50_median_mm", where)
    known = [value for value in (low, median, high) if value is not None]
    if known != sorted(known) or (known and known[0] < 0.0):
        raise ResultsError(f"{where}: the radius {median!r} is not inside its interval {low!r} to {high!r}")
    return Fit(
        blocks=tuple(str(block) for block in blocks),
        r50_median_mm=median,
        r50_lo_mm=low,
        r50_hi_mm=high,
        open_below=open_below,
        open_above=open_above,
        curve_d_mm=distances,
        curve_mean=columns["mean"],
        curve_lo=columns["lo"],
        curve_hi=columns["hi"],
    )


def file_bins(results: Mapping[str, Any], condition: str) -> list[Bin]:
    """The bins the analysis wrote for a condition; none for a condition the file does not have."""
    entry = _condition(results, condition)
    if entry is None:
        return []
    where = f"conditions.{condition}.bins"
    found = []
    for index, item in enumerate(_need(entry, "bins", list, f"conditions.{condition}")):
        here = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise ResultsError(f"{here} is not an object")
        n, k = _whole(item, "n", here), _whole(item, "k", here)
        if n == 0:
            continue
        found.append(Bin(_number(item, "lo_mm", here), _number(item, "hi_mm", here), n, k,
                         _number(item, "wilson_lo", here), _number(item, "wilson_hi", here)))  # fmt: skip
    return sorted(found, key=lambda bin_: bin_.lo_mm)


def binned(some_trials: Sequence[ScoredTrial], width_mm: float, level: float) -> list[Bin]:
    """Trials counted in stretches of `width_mm`, each closed below and open above, with Wilson intervals: the
    counting rule of the protocol (section 10), by the project's own interval function. Only stretches that hold a
    trial are listed, each with the mean distance of its trials.
    """
    counts: dict[int, list[float]] = {}
    for trial in some_trials:
        entry = counts.setdefault(int(trial.d_mm // width_mm), [0, 0, 0.0])
        entry[0] += 1
        entry[1] += trial.success
        entry[2] += trial.d_mm
    found = []
    for index in sorted(counts):
        n, k, total_mm = counts[index]
        low, high = wilson_interval(k, n, level)
        found.append(Bin(index * width_mm, (index + 1) * width_mm, n, k, low, high, total_mm / n))
    return found


def same_bins(first: Sequence[Bin], second: Sequence[Bin], tolerance: float = 1e-6) -> bool:
    """Whether two lists of bins say the same thing: the same stretches, the same counts, the same intervals."""
    if len(first) != len(second):
        return False
    for a, b in zip(first, second, strict=True):
        if (a.n, a.k) != (b.n, b.k) or abs(a.lo_mm - b.lo_mm) > tolerance or abs(a.hi_mm - b.hi_mm) > tolerance:
            return False
        if abs(a.wilson_lo - b.wilson_lo) > tolerance or abs(a.wilson_hi - b.wilson_hi) > tolerance:
            return False
    return True


# --------------------------------------------------------------------------------------------------------------------
# The workspace test
# --------------------------------------------------------------------------------------------------------------------


def has_workspace(results: Mapping[str, Any]) -> bool:
    """Whether the workspace test has anything in the file."""
    if "workspace" not in results:
        raise ResultsError("workspace is missing (it is null when the block was not run)")
    workspace = results["workspace"]
    if workspace is None:
        return False
    if not isinstance(workspace, dict):
        raise ResultsError("workspace is neither an object nor null")
    return len(_need(workspace, "pairs", list, "workspace")) > 0


def placements(results: Mapping[str, Any], policy: str) -> list[Placement]:
    """One policy's trials of the workspace test, a placement each, in the file's order.

    A placement is drawn where that policy's own trial was measured when the file says where, and at the pair's
    position otherwise.
    """
    _check_policy(policy)
    found = []
    for index, pair in enumerate(_need(_workspace(results), "pairs", list, "workspace")):
        here = f"workspace.pairs[{index}]"
        if not isinstance(pair, dict):
            raise ResultsError(f"{here} is not an object")
        own = _need(pair, policy, dict, here)
        own_position = _is_number(own.get("x_mm")) and _is_number(own.get("y_mm"))
        position = own if own_position else pair
        region, p_lookup = pair.get("region"), own.get("p_lookup")
        found.append(
            Placement(
                x_mm=_number(position, "x_mm", here),
                y_mm=_number(position, "y_mm", here),
                success=_need(own, "success", bool, f"{here}.{policy}"),
                d_mm=_number(own, "d_mm", f"{here}.{policy}"),
                region=region if isinstance(region, str) else None,
                p_lookup=float(p_lookup) if _is_number(p_lookup) else None,
                own_position=own_position,
            )
        )
    return found


def lookup(results: Mapping[str, Any], policy: str) -> Lookup | None:
    """The lookup prediction for a policy; None when the file has none (no block fit of the clustered policy)."""
    _check_policy(policy)
    workspace = _workspace(results)
    if "lookup" not in workspace:
        raise ResultsError("workspace.lookup is missing (it is null when there is no fit to predict from)")
    stated = workspace["lookup"]
    if stated is None:
        return None
    where = f"workspace.lookup.{policy}"
    own = _need(_as_dict(stated, "workspace.lookup"), policy, dict, "workspace.lookup")
    return Lookup(_number(own, "expected", where), _whole(own, "lo", where), _whole(own, "hi", where),
                  _whole(own, "observed", where), _need(own, "verdict", str, where))  # fmt: skip


def region_counts(results: Mapping[str, Any], policy: str) -> tuple[dict[str, Count] | None, list[str]]:
    """A policy's successes, and the count the lookup prediction expects, in each region of the workspace test.

    The file's own counts by region are used when it has them (`workspace.by_region`). Without them the counts are
    made here from the pairs, when the pairs name their regions: the expected count is then the sum of the pairs'
    `p_lookup`. None when the file says nothing of regions. A region without a placement is left out, and the
    protocol's two regions come first.

    The second value holds sentences for the person who runs the figures: the file's counts by region could not be
    read, or they are not the counts of its pairs.
    """
    _check_policy(policy)
    workspace = _workspace(results)
    counted, notes = _counted_by_region(results, policy)
    stated = None
    if workspace.get("by_region") is not None:
        try:
            stated = _stated_by_region(workspace["by_region"], policy)
        except ResultsError as error:
            used = "the counts by region were made from the pairs" if counted else "no count by region is shown"
            notes.append(f"workspace.by_region is not in the shape this code reads ({error}): {used}")
    if stated is None:
        return counted, notes
    for region in sorted(set(stated) | set(counted or {})) if counted is not None else ():
        ours, theirs = counted.get(region), stated.get(region)
        if ours is None or theirs is None or (ours.k, ours.n) != (theirs.k, theirs.n):
            notes.append(
                f"workspace.by_region.{region}.{policy} gives {_said(theirs)}, the pairs of that region hold "
                f"{_said(ours)}"
            )
        elif None not in (ours.lookup_expected, theirs.lookup_expected) and (
            abs(ours.lookup_expected - theirs.lookup_expected) > EXPECTED_TOLERANCE
        ):
            notes.append(
                f"workspace.by_region.{region}.{policy}.lookup_expected is {theirs.lookup_expected:.2f}, the pairs' "
                f"p_lookup of that region add up to {ours.lookup_expected:.2f}"
            )
    return _in_order(stated), notes


def _said(count: Count | None) -> str:
    return "no placement" if count is None else f"{count.k} of {count.n}"


def _in_order(counts: Mapping[str, Count]) -> dict[str, Count]:
    """Counts by region with the protocol's regions first, in their order, and any other after them."""
    names = [region for region in REGIONS if region in counts] + [region for region in counts if region not in REGIONS]
    return {region: counts[region] for region in names}


def _stated_by_region(stated: Any, policy: str) -> dict[str, Count]:
    """What `workspace.by_region` says of a policy: for each region `k`, `n` and, when given, the count the lookup
    prediction expects there (`lookup_expected`) with the interval of the predicted count (`lookup_lo`, `lookup_hi`).
    """
    found = {}
    for region, entry in _as_dict(stated, "workspace.by_region").items():
        where = f"workspace.by_region.{region}.{policy}"
        own = _need(_as_dict(entry, f"workspace.by_region.{region}"), policy, dict, f"workspace.by_region.{region}")
        n, k = _whole(own, "n", where), _whole(own, "k", where)
        if k > n:
            raise ResultsError(f"{where} counts {k} successes in {n} placements")
        if n == 0:
            continue
        expected = own.get("lookup_expected")
        low, high = own.get("lookup_lo"), own.get("lookup_hi")
        with_interval = _is_number(expected) and all(_is_number(end) and end == int(end) for end in (low, high))
        if with_interval and not 0 <= low <= high <= n:
            raise ResultsError(f"{where}: the interval of the predicted count runs from {low!r} to {high!r} of {n}")
        found[region] = Count(
            k,
            n,
            float(expected) if _is_number(expected) else None,
            int(low) if with_interval else None,
            int(high) if with_interval else None,
        )
    return found


def _counted_by_region(results: Mapping[str, Any], policy: str) -> tuple[dict[str, Count] | None, list[str]]:
    """The counts by region made from the pairs, when they name their regions, and what there is to say of them."""
    placed = placements(results, policy)
    if all(placement.region is None for placement in placed):
        return None, []
    predicted = lookup(results, policy) is not None
    found = {}
    for region in dict.fromkeys(placement.region for placement in placed if placement.region is not None):
        inside = [placement for placement in placed if placement.region == region]
        expected = None
        if predicted and all(placement.p_lookup is not None for placement in inside):
            expected = sum(placement.p_lookup for placement in inside)
        found[region] = Count(sum(placement.success for placement in inside), len(inside), expected)
    nowhere = sum(placement.region is None for placement in placed)
    notes = [f"{nowhere} of the {len(placed)} placements of {policy} name no region: they are counted under all"
             " placements only"] if nowhere else []  # fmt: skip
    return _in_order(found), notes


def pair_checks(results: Mapping[str, Any], policy: str) -> list[str]:
    """Check that what the file says of a policy's workspace test in two places is the same thing.

    The successes among the placements (`workspace.success`) and the count the prediction is compared with
    (`workspace.lookup.<policy>.observed`) must be the count of the pairs listed: these are fields of the contract,
    and a figure that drew one and wrote the other would contradict itself, so a difference is an error. That the
    pairs' `p_lookup` add up to the expected count is checked too, and a difference there is returned as a sentence.
    """
    placed = placements(results, policy)
    total = success_count(results, policy)
    k, n = sum(placement.success for placement in placed), len(placed)
    if (total.k, total.n) != (k, n):
        raise ResultsError(
            f"workspace.success.{policy} counts {total.k} of {total.n}, and the pairs listed hold {k} successes in "
            f"{n} placements"
        )
    stated = lookup(results, policy)
    if stated is None:
        return []
    if stated.observed != k:
        raise ResultsError(
            f"workspace.lookup.{policy}.observed is {stated.observed}, and the pairs listed hold {k} successes"
        )
    if not stated.lo <= stated.hi:
        raise ResultsError(f"workspace.lookup.{policy}: the interval runs from {stated.lo} to {stated.hi}")
    if all(placement.p_lookup is not None for placement in placed):
        added = sum(placement.p_lookup for placement in placed)
        if abs(added - stated.expected) > EXPECTED_TOLERANCE:
            return [f"workspace.lookup.{policy}.expected is {stated.expected:.2f}, the pairs' p_lookup add up to "
                    f"{added:.2f}"]  # fmt: skip
    return []


def success_count(results: Mapping[str, Any], policy: str) -> Count:
    """A policy's successes over all placements of the workspace test."""
    _check_policy(policy)
    where = f"workspace.success.{policy}"
    own = _need(_need(_workspace(results), "success", dict, "workspace"), policy, dict, "workspace.success")
    stated = lookup(results, policy)
    return Count(_whole(own, "k", where), _whole(own, "n", where), None if stated is None else stated.expected)


def demonstrations(results: Mapping[str, Any], dataset: str) -> list[Demonstration]:
    """The demonstrations of one set with their measured starts; none for a set without an audit."""
    where = f"demonstrations.{dataset}"
    found = []
    for index, item in enumerate(_need(_need(results, "demonstrations", dict, ""), dataset, list, "demonstrations")):
        here = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise ResultsError(f"{here} is not an object")
        found.append(Demonstration(_number(item, "x_mm", here), _number(item, "y_mm", here),
                                   _need(item, "trained", bool, here)))  # fmt: skip
    return found


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _condition(results: Mapping[str, Any], condition: str) -> Mapping[str, Any] | None:
    if condition not in CONDITIONS:
        raise ValueError(f"a condition is one of {CONDITIONS}, got {condition!r}")
    conditions = _need(results, "conditions", dict, "")
    entry = conditions.get(condition)
    if entry is None:
        return None
    return _as_dict(entry, f"conditions.{condition}")


def _workspace(results: Mapping[str, Any]) -> Mapping[str, Any]:
    if not has_workspace(results):
        raise ResultsError("the file has no workspace test")
    return results["workspace"]


def _check_policy(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"a policy is one of {POLICIES}, got {policy!r}")


def _as_dict(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ResultsError(f"{where} is not an object")
    return value


def _need(mapping: Mapping[str, Any], key: str, kind: type, where: str) -> Any:
    """A field of the file that must be there and of a kind; the error names it by its place in the file."""
    place = f"{where}.{key}" if where else key
    if key not in mapping:
        raise ResultsError(f"{place} is missing")
    value = mapping[key]
    if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
        raise ResultsError(f"{place} should be {_KIND_NAMES.get(kind, kind.__name__)}, got {value!r}")
    return value


_KIND_NAMES = {dict: "an object", list: "a list", str: "text", bool: "true or false"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(mapping: Mapping[str, Any], key: str, where: str) -> float:
    place = f"{where}.{key}" if where else key
    if key not in mapping:
        raise ResultsError(f"{place} is missing")
    if not _is_number(mapping[key]):
        raise ResultsError(f"{place} should be a number, got {mapping[key]!r}")
    return float(mapping[key])


def _number_unless(may_be_null: bool, mapping: Mapping[str, Any], key: str, where: str) -> float | None:
    """A number of the file; None for a null where a null is allowed. The field itself must be there."""
    if may_be_null and key in mapping and mapping[key] is None:
        return None
    return _number(mapping, key, where)


def _whole(mapping: Mapping[str, Any], key: str, where: str) -> int:
    value = _number(mapping, key, where)
    if value != int(value) or value < 0:
        raise ResultsError(f"{where}.{key} should be a whole number of 0 or more, got {mapping[key]!r}")
    return int(value)


def _numbers(mapping: Mapping[str, Any], key: str, where: str) -> list[float]:
    values = _need(mapping, key, list, where)
    if not all(_is_number(value) for value in values):
        raise ResultsError(f"{where}.{key} should be a list of numbers")
    return [float(value) for value in values]


def interpolate(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """`ys` at `x` on the straight lines between the points; the end values beyond the ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    low, high = 0, len(xs) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if xs[middle] <= x:
            low = middle
        else:
            high = middle
    share = (x - xs[low]) / (xs[high] - xs[low])
    return ys[low] + share * (ys[high] - ys[low])
