"""The paper's number macros: which result each one prints, and how it is written.

`paper/make_numbers_stub.py` owns the list of macro names and writes `paper/generated/numbers.tex`. This module says,
for each name, where its value is in the results (`build.build_results`) and in which form the paper prints it.
`macro_values` returns the values in the form the stub takes. A macro whose result is None gets no value and stays a
visible `??` in the paper.

How a value is written (the rules at the top of `paper/generated/MACROS.md`):

* a distance: millimetres with one decimal, no unit;
* the camera's creep before an attempt: millimetres with two decimals, because one decimal can print a reading that
  lies under the protocol's limit as the limit itself;
* a count: a whole number;
* a share: a whole percentage, without the sign;
* a probability or a p-value: two decimals, or a bound (`<0.01`, `>0.99`);
* the scale factor: three decimals;
* a negative number carries its sign as `\\ensuremath{-}`, which prints a minus sign in text and in mathematics;
* words typed by a person (the reasons of void trials) are escaped for TeX.

Numbers are rounded half up on their shortest decimal form, so that 2.25 prints as 2.3, the way a reader rounds it.

Standard library only.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from tolerance_radius.report.bookkeeping import SIMULATED_WIDTHS_MM
from tolerance_radius.report.build import CLUSTERED, REGIONS, REPLAY, SPREAD

Results = Mapping[str, Any]

MINUS = r"\ensuremath{-}"
#: Macros of the stub's list that no committed record can fill, each with the reason. They stay `??`.
NOT_PRODUCED = {
    "gridDriftMax": "the tools print the disagreement of a session's camera solve and no committed record holds it",
    "nPracticeEpisodes": "no record says which episodes were practice",
    "nTornFramesNew": "the torn-frame scan prints its findings and no committed record holds them",
    "nMispacedEpisodesNew": "the pacing check prints its findings and no committed record holds them",
}


# --------------------------------------------------------------------------------------------------------------------
# How a value is written
# --------------------------------------------------------------------------------------------------------------------


def rounded(value: float, decimals: int, times: int = 1, minus: str = MINUS) -> str:
    """A number with a fixed count of decimals, rounded half up on its shortest decimal form.

    `times` multiplies the number first, exactly (100 turns a share into a percentage). A negative number's sign is
    written as `minus` (for the paper `MINUS`, which prints in text and in mathematics), and a result that rounds to
    zero has no sign.
    """
    quantum = Decimal(1).scaleb(-decimals)
    amount = (Decimal(repr(float(value))) * times).quantize(quantum, rounding=ROUND_HALF_UP)
    if amount == 0:
        amount = abs(amount)
    return (minus if amount < 0 else "") + f"{abs(amount):.{decimals}f}"


def distance(value: float) -> str:
    """Millimetres with one decimal."""
    return rounded(value, 1)


def two_decimals(value: float) -> str:
    """Millimetres with two decimals, for a reading that is held against a limit one decimal cannot tell it from."""
    return rounded(value, 2)


def one_decimal(value: float) -> str:
    """A number that is printed with one decimal and is not a distance (minutes, an expected count)."""
    return rounded(value, 1)


def count(value: int) -> str:
    """A whole number, with a comma every three digits."""
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"a count is a whole number, got {value!r}")
    return f"{int(value):,}"


def percent(share: float) -> str:
    """A share between 0 and 1 as a whole percentage, without the sign."""
    return rounded(share, 0, times=100)


def probability(value: float) -> str:
    """A posterior probability: two decimals, or a bound where two decimals would say 0 or 1."""
    if value < 0.01:
        return "<0.01"
    if value > 0.99:
        return ">0.99"
    return rounded(value, 2)


def p_value(value: float) -> str:
    """A p-value: two decimals, `<0.01` below that, and `1.00` only when it is exactly 1."""
    if value < 0.01:
        return "<0.01"
    if value == 1.0:
        return "1.00"
    return ">0.99" if value > 0.99 else rounded(value, 2)


def factor(value: float) -> str:
    """The scale factor: three decimals."""
    return rounded(value, 3)


def plain_number(value: float) -> str:
    """A setting as the protocol's own macros print one: a whole number with commas, anything else without padding."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


_TEX = {
    "\\": r"\textbackslash{}",
    "{": r"\textbraceleft{}",
    "}": r"\textbraceright{}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    "%": r"\%",
    "#": r"\#",
    "&": r"\&",
    "_": r"\_",
    "$": r"\$",
}
#: Typed characters outside ASCII that have a plain equivalent.
_PLAIN = {
    "`": "'",
    "\u2018": "'",  # typographic single and double quotes
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",  # en and em dash
    "\u2014": "-",
    "\u2026": "...",
    "\u00d7": "x",
    "\u00b0": " deg",
    "\u00b5": "u",
}


def tex_escaped(text: str) -> str:
    """Words typed by a person, made safe to print in the paper.

    White space becomes single spaces. Characters TeX reads as commands are escaped. Accents are dropped and other
    characters outside ASCII are replaced by a plain equivalent or left out, because the paper's sources are ASCII.
    A run of question marks is cut to one, so that the text cannot hold the placeholder mark `??`.
    """
    text = "".join(_PLAIN.get(character, character) for character in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character for character in text if character.isascii() and (character.isprintable() or character.isspace())
    )
    text = " ".join(text.split())
    while "??" in text:
        text = text.replace("??", "?")
    return "".join(_TEX.get(character, character) for character in text)


def void_reasons_text(reasons: list[Mapping[str, Any]]) -> str:
    """The reasons of void trials as one text, each with its count: `bumped the table (2), camera fault (1)`.

    `none` when no trial was void. Reasons that hold a comma are separated by semicolons.
    """
    if not reasons:
        return "none"
    separator = "; " if any("," in reason["reason"] for reason in reasons) else ", "
    return separator.join(f"{tex_escaped(reason['reason'])} ({reason['n']})" for reason in reasons)


# --------------------------------------------------------------------------------------------------------------------
# Where each macro's value is
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Macro:
    """One number macro: where its value is in the results, how it is written, and where the value comes from."""

    name: str
    #: The value in the results, or None when the data do not support one.
    find: Callable[[Results], Any]
    write: Callable[[Any], str]
    #: Where the value comes from, for the comment beside the macro and for the report.
    source: str


def _at(*path: str) -> Callable[[Results], Any]:
    """The value at a path of keys in the results; None as soon as a step is None or missing."""

    def find(results: Results) -> Any:
        node: Any = results
        for key in path:
            if not isinstance(node, Mapping) or node.get(key) is None:
                return None
            node = node[key]
        return node

    return find


def _logged(find: Callable[[Results], Any]) -> Callable[[Results], Any]:
    """A count of trials: None when no trial log was read, so that an absent experiment does not print as 0."""

    def guarded(results: Results) -> Any:
        return find(results) if results.get("blocks") is not None else None

    return guarded


def _successes(*path: str) -> Callable[[Results], Any]:
    """The successes of a count `{n, k}` at a path; None when it counts no trial."""

    def find(results: Results) -> Any:
        entry = _at(*path)(results)
        return entry["k"] if entry and entry["n"] > 0 else None

    return find


def _block(condition: str, key: str) -> Callable[[Results], Any]:
    """A count of the threshold block of a condition."""

    def find(results: Results) -> Any:
        name = _at("conditions", condition, "threshold_block")(results)
        return _at("blocks", name, key)(results) if name else None

    return find


def _simulated(condition: str, *path: str) -> Callable[[Results], Any]:
    """A figure of the procedure simulation for the threshold block of a condition, as the block is written."""

    def find(results: Results) -> Any:
        name = _at("conditions", condition, "threshold_block")(results)
        return _at("simulation", "blocks", name, *path)(results) if name else None

    return find


#: The cases of the simulated comparison rule that have a macro, by the word that names each in a macro: the width
#: of both curves and the true difference between the two radii, in millimetres.
RULE_WIDTHS_MM = {"One": 1.0, "Three": 3.0, "Six": 6.0}
RULE_DIFFERENCES_MM = {"Equal": 0.0, "Three": 3.0, "Five": 5.0, "Eight": 8.0, "Ten": 10.0, "Fifteen": 15.0}


def _rule_share(width_mm: float, difference_mm: float, verdict: str) -> Callable[[Results], Any]:
    """The share of simulated pairs of blocks to which the comparison rule gave a verdict, for one case."""

    def find(results: Results) -> Any:
        for case in _at("simulation", "comparison_rule", "cases")(results) or ():
            if case["width_mm"] == width_mm and case["difference_mm"] == difference_mm:
                return case[verdict]
        return None

    return find


def _region_pairs(region: str) -> Callable[[Results], Any]:
    """Placements of the workspace test in a region: 0 when the test was run and none lies there."""

    def find(results: Results) -> Any:
        return _at("workspace", "by_region", region, "n_pairs")(results) if results.get("workspace") else None

    return find


def _paired_block(key: str) -> Callable[[Results], Any]:
    def find(results: Results) -> Any:
        blocks = results.get("blocks") or {}
        name = next((name for name, entry in blocks.items() if entry["kind"] == "paired"), None)
        return blocks[name][key] if name else None

    return find


def _pairs(results: Results) -> Any:
    """Placements tested with both policies: 0 when trials were read and the workspace test has none."""
    return _at("workspace", "n_pairs")(results) or 0


def _table(key: str) -> Callable[[Results], Any]:
    """An entry of the paired table; None when there is no pair."""

    def find(results: Results) -> Any:
        return _at("workspace", "table", key)(results) if _at("workspace", "n_pairs")(results) else None

    return find


def _discordant(results: Results) -> Any:
    table = _at("workspace", "table")(results)
    return table["only_spread"] + table["only_clustered"] if table and _at("workspace", "n_pairs")(results) else None


def _replay_episodes(results: Results) -> Any:
    episodes = _at("conditions", REPLAY, "replay_episodes")(results)
    return len(episodes) if episodes else None


def _void_reasons(results: Results) -> Any:
    return _at("conduct", "void_reasons")(results) if results.get("conduct") is not None else None


def _length(*path: str) -> Callable[[Results], Any]:
    def find(results: Results) -> Any:
        found = _at(*path)(results)
        return len(found) if found else None

    return find


def _setting(key: str, conditions: tuple[str, ...]) -> Callable[[Results], Any]:
    """An execution setting that the given conditions share, among those that were run; None when they differ."""

    def find(results: Results) -> Any:
        ran = _at("conduct", "execution", "by_condition")(results) or {}
        values = [ran[condition].get(key) for condition in conditions if condition in ran]
        if not values or any(value is None or value != values[0] for value in values):
            return None
        return values[0]

    return find


def _rerecorded(dataset: str) -> Callable[[Results], Any]:
    def find(results: Results) -> Any:
        recording = _at("datasets", dataset, "recording")(results)
        return recording["n_rerecorded_episodes"] if recording and recording["matches_audit"] else None

    return find


_POLICIES, _EVERY = (CLUSTERED, SPREAD), (CLUSTERED, SPREAD, REPLAY)
_SIM = "analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured)"
_SIM_POWER = "analysis/simulation/paired_power.csv, the case fixed before the data (computed exactly, not measured)"
_SIM_BLOCKS = (
    "analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width "
    "(simulated, not measured)"
)
_SIM_RULE = (
    "analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written "
    "(simulated, not measured)"
)
_SIM_FALLING = (
    "analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths "
    "(simulated, not measured)"
)
_RULER = "data/instrument/ruler_*.csv, as the camera reads"
_BY_EYE = "data/instrument/byeye_*.csv"
_AUDITS = "data/audit/clustered.json and data/audit/spread.json"
_SECOND = "data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare"
_SCENE = "data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials"
_TRIALS = "the counted trial logs"


def _fit_named(kind: str, condition: str) -> str:
    """A fit of a condition as a source names it: `block fit of replay`, `pooled fit of act_spread`."""
    return f"{'block' if kind == 'block_fit' else 'pooled'} fit of {condition}"


def _condition_macros(suffix: str, condition: str, block: str, radius_fit: str) -> list[Macro]:
    """The macros every condition has: counts, the radius, and for its threshold block the anchors at d = 0 and the
    two extreme distances (the paper prints those on the block's row of its table)."""
    logs = f"{_TRIALS}, {condition}"
    fit = _fit_named(radius_fit, condition)
    return [
        Macro(
            f"nTrials{suffix}", _logged(_at("conditions", condition, "n_scored")), count, f"{logs}, scored, every block"
        ),
        Macro(f"nVoid{suffix}", _at("conditions", condition, "n_void"), count, f"{logs}, void, every block"),
        Macro(
            f"successZero{suffix}",
            _successes("conditions", condition, "anchors", "zero"),
            count,
            f"{logs}, block {block}, anchors at d = 0",
        ),
        Macro(
            f"nZero{suffix}",
            _logged(_at("conditions", condition, "anchors", "zero", "n")),
            count,
            f"{logs}, block {block}, anchors at d = 0",
        ),
        Macro(
            f"radius{suffix}",
            _at("conditions", condition, radius_fit, "r50_median_mm"),
            distance,
            f"{fit}: posterior median of r50",
        ),
        Macro(
            f"radius{suffix}Low",
            _at("conditions", condition, radius_fit, "r50_lo_mm"),
            distance,
            f"{fit}: lower end of the interval",
        ),
        Macro(
            f"radius{suffix}High",
            _at("conditions", condition, radius_fit, "r50_hi_mm"),
            distance,
            f"{fit}: upper end of the interval",
        ),
        Macro(
            f"maxSuccessDist{suffix}",
            _at("conditions", condition, "block_fit", "max_success_d_mm"),
            distance,
            f"{logs}, scored, block {block} alone",
        ),
        Macro(
            f"minFailureDist{suffix}",
            _at("conditions", condition, "block_fit", "min_failure_d_mm"),
            distance,
            f"{logs}, scored, block {block} alone",
        ),
    ]


def _block_fit_macros(prefix: str, condition: str) -> list[Macro]:
    """The radius of a condition's block fit with its interval, under the names `prefix`, `prefixLow`, `prefixHigh`."""
    fit = f"block fit of {condition}"
    return [
        Macro(
            prefix,
            _at("conditions", condition, "block_fit", "r50_median_mm"),
            distance,
            f"{fit}: posterior median of r50",
        ),
        Macro(
            f"{prefix}Low",
            _at("conditions", condition, "block_fit", "r50_lo_mm"),
            distance,
            f"{fit}: lower end of the interval",
        ),
        Macro(
            f"{prefix}High",
            _at("conditions", condition, "block_fit", "r50_hi_mm"),
            distance,
            f"{fit}: upper end of the interval",
        ),
    ]


def _wide_macros(prefix: str, condition: str, kind: str = "block_fit") -> list[Macro]:
    """The plateau check of a condition's fit, its block fit unless `kind` names the pooled one: the radius with its
    interval under the longer list of lapse values, as `prefixWide`, `prefixWideLow`, `prefixWideHigh`."""
    fit = f"{_fit_named(kind, condition)} made again with the protocol's longer list of lapse values"
    path = ("conditions", condition, kind, "sensitivity")
    return [
        Macro(f"{prefix}Wide", _at(*path, "r50_median_mm"), distance, f"{fit}: posterior median of r50"),
        Macro(f"{prefix}WideLow", _at(*path, "r50_lo_mm"), distance, f"{fit}: lower end of the interval"),
        Macro(f"{prefix}WideHigh", _at(*path, "r50_hi_mm"), distance, f"{fit}: upper end of the interval"),
    ]


def _shift_macro(name: str, condition: str, kind: str = "block_fit") -> Macro:
    """The shift of the plateau check of a condition's fit: the median of the fit made again with the longer list of
    lapse values, minus the median that is reported. Negative when the longer list pulls the radius inward."""
    return Macro(
        name,
        _at("conditions", condition, kind, "sensitivity", "median_shift_mm"),
        distance,
        f"{_fit_named(kind, condition)}: its posterior median with the protocol's longer list of lapse values, "
        "minus its posterior median as reported",
    )


def _simulation_macros() -> list[Macro]:
    """The procedure simulation by threshold block and by width of the true curve, and how often an interval is
    open below when the curve is already falling at d = 0."""
    macros = [
        Macro(
            f"simWidth{name.capitalize()}",
            _simulated(CLUSTERED, "by_width", name, "width_true_mm"),
            distance,
            _SIM_BLOCKS,
        )
        for name in SIMULATED_WIDTHS_MM
    ]
    for suffix, condition in (("Replay", REPLAY), ("Clustered", CLUSTERED), ("Spread", SPREAD)):
        for name in SIMULATED_WIDTHS_MM:
            curve = name.capitalize()
            macros += [
                Macro(
                    f"simRmse{suffix}{curve}",
                    _simulated(condition, "by_width", name, "rmse_mm"),
                    distance,
                    _SIM_BLOCKS,
                ),
                Macro(
                    f"simIntervalWidth{suffix}{curve}",
                    _simulated(condition, "by_width", name, "interval_width_mm"),
                    distance,
                    _SIM_BLOCKS,
                ),
            ]
        macros.append(
            Macro(f"simOpenBelow{suffix}", _simulated(condition, "open_below_falling_at_zero"), percent, _SIM_FALLING)
        )
    macros.append(Macro("simOpenBelow", _at("simulation", "open_below_falling_at_zero"), percent, _SIM_FALLING))
    return macros


def _rule_macros() -> list[Macro]:
    """The simulation of the comparison rule: how often it says "larger" for every width and difference, and how
    often "smaller" when the two radii are equal."""
    macros = [
        Macro("simRuleReplayRadius", _at("simulation", "comparison_rule", "replay_r50_mm"), distance, _SIM_RULE),
        Macro("simRulePairs", _at("simulation", "comparison_rule", "pairs"), count, _SIM_RULE),
    ]
    for width_word, width in RULE_WIDTHS_MM.items():
        for difference_word, difference in RULE_DIFFERENCES_MM.items():
            name = f"simRuleLargerWidth{width_word}Diff{difference_word}"
            macros.append(Macro(name, _rule_share(width, difference, "larger"), percent, _SIM_RULE))
        macros.append(
            Macro(f"simRuleSmallerWidth{width_word}DiffEqual", _rule_share(width, 0.0, "smaller"), percent, _SIM_RULE)
        )
    return macros


def _workspace_macros() -> list[Macro]:
    macros = [
        Macro(
            "nPairs", _logged(_pairs), count, f"{_TRIALS}, block T2: placements with a scored trial of both policies"
        ),
        Macro("nTrialsWorkspace", _paired_block("n_scored"), count, f"{_TRIALS}, block T2, scored"),
        Macro("nVoidWorkspace", _paired_block("n_void"), count, f"{_TRIALS}, block T2, void"),
        Macro("pairsBoth", _table("both"), count, "the paired table of block T2"),
        Macro("pairsNeither", _table("neither"), count, "the paired table of block T2"),
        Macro("pairsOnlySpread", _table("only_spread"), count, "the paired table of block T2"),
        Macro("pairsOnlyClustered", _table("only_clustered"), count, "the paired table of block T2"),
        Macro("pairsDiscordant", _discordant, count, "the paired table of block T2"),
        Macro(
            "mcnemarP",
            _at("workspace", "mcnemar_p"),
            p_value,
            "exact McNemar test on the discordant pairs of block T2, all pairs together",
        ),
    ]
    for suffix, policy in (("Clustered", CLUSTERED), ("Spread", SPREAD)):
        pairs = f"the pairs of block T2, {policy}"
        macros += [
            Macro(f"workspaceSuccess{suffix}", _successes("workspace", "success", policy), count, pairs),
            Macro(
                f"workspace{suffix}Low",
                _at("workspace", "success", policy, "wilson_lo"),
                percent,
                f"{pairs}: Wilson interval",
            ),
            Macro(
                f"workspace{suffix}High",
                _at("workspace", "success", policy, "wilson_hi"),
                percent,
                f"{pairs}: Wilson interval",
            ),
            Macro(
                f"workspaceDist{suffix}Median",
                _at("workspace", "distances", policy, "median_mm"),
                distance,
                f"{pairs}: measured d",
            ),
            Macro(
                f"workspaceDist{suffix}Max",
                _at("workspace", "distances", policy, "max_mm"),
                distance,
                f"{pairs}: measured d",
            ),
            Macro(
                f"lookupPredicted{suffix}",
                _at("workspace", "lookup", policy, "expected"),
                one_decimal,
                f"block fit of {CLUSTERED} at the distances of {pairs}",
            ),
            Macro(
                f"lookup{suffix}Low",
                _at("workspace", "lookup", policy, "lo"),
                count,
                f"predicted count's interval, block fit of {CLUSTERED}, {pairs}",
            ),
            Macro(
                f"lookup{suffix}High",
                _at("workspace", "lookup", policy, "hi"),
                count,
                f"predicted count's interval, block fit of {CLUSTERED}, {pairs}",
            ),
        ]
    for region, where, named in zip(REGIONS, ("Trained", "HeldOut"), ("trained cells", "held-out cells"), strict=True):
        cells = f"the pairs of block T2 whose measured position lies in {named}"
        macros.append(Macro(f"nPairs{where}", _region_pairs(region), count, cells))
        for suffix, policy in (("Clustered", CLUSTERED), ("Spread", SPREAD)):
            macros += [
                Macro(
                    f"workspaceSuccess{suffix}{where}",
                    _successes("workspace", "by_region", region, policy),
                    count,
                    f"{cells}, {policy}",
                ),
                Macro(
                    f"lookupPredicted{suffix}{where}",
                    _at("workspace", "by_region", region, policy, "lookup_expected"),
                    one_decimal,
                    f"block fit of {CLUSTERED} at the distances of {cells}, {policy}",
                ),
            ]
    return macros


def _comparison_macros(suffix: str, key: str, what: str) -> list[Macro]:
    source = f"block fits, {what}, taken as independent"
    return [
        Macro(
            f"radiusDiff{suffix}", _at("comparisons", key, "median_mm"), distance, f"{source}: median of the difference"
        ),
        Macro(
            f"radiusDiff{suffix}Low", _at("comparisons", key, "lo_mm"), distance, f"{source}: lower end of the interval"
        ),
        Macro(
            f"radiusDiff{suffix}High",
            _at("comparisons", key, "hi_mm"),
            distance,
            f"{source}: upper end of the interval",
        ),
    ]


MACROS: tuple[Macro, ...] = (
    # sim
    Macro("simTruthRadius", _at("simulation", "case", "r50_true_mm"), distance, _SIM),
    Macro("simTruthWidth", _at("simulation", "case", "width_true_mm"), distance, _SIM),
    Macro("simReplications", _at("simulation", "case", "replications"), count, _SIM),
    Macro("simBias", _at("simulation", "case", "bias_mm"), distance, _SIM),
    Macro("simRmse", _at("simulation", "case", "rmse_mm"), distance, _SIM),
    Macro("simIntervalWidth", _at("simulation", "case", "interval_width_mm"), distance, _SIM),
    Macro("simCoverage", _at("simulation", "case", "coverage"), percent, _SIM),
    Macro("simRmseConstant", _at("simulation", "case", "rmse_fixed_mm"), distance, _SIM),
    Macro("simIntervalWidthConstant", _at("simulation", "case", "interval_width_fixed_mm"), distance, _SIM),
    Macro("simPairedRateLow", _at("simulation", "case", "paired", "rate_low"), percent, _SIM_POWER),
    Macro("simPairedRateHigh", _at("simulation", "case", "paired", "rate_high"), percent, _SIM_POWER),
    Macro("simPairedPower", _at("simulation", "case", "paired", "power"), percent, _SIM_POWER),
    *_simulation_macros(),
    *_rule_macros(),
    # V0
    Macro("nScaleRuns", _at("instrument", "scale_check", "n_runs"), count, _RULER),
    Macro("nScaleSteps", _at("instrument", "scale_check", "n_steps"), count, _RULER),
    Macro("instrumentMeanAbsError", _at("instrument", "scale_check", "mean_abs_step_error_mm"), distance, _RULER),
    Macro("instrumentMaxStepError", _at("instrument", "scale_check", "max_abs_step_error_mm"), distance, _RULER),
    Macro("scaleFactor", _at("distance_scale", "factor"), factor, f"{_RULER}: mean of the runs"),
    Macro("scaleFactorMin", _at("distance_scale", "factor_min"), factor, f"{_RULER}: smallest of the runs"),
    Macro("scaleFactorMax", _at("distance_scale", "factor_max"), factor, f"{_RULER}: largest of the runs"),
    Macro("nRepeatability", _at("instrument", "repeatability", "n"), count, "data/instrument/repeat_*.csv"),
    Macro(
        "repeatabilityScatter",
        _at("instrument", "repeatability", "scatter_max_mm"),
        distance,
        "data/instrument/repeat_*.csv, as the camera reads",
    ),
    # V1
    Macro("nByEye", _at("instrument", "by_eye", "n"), count, _BY_EYE),
    Macro("nOperatorsByEye", _at("instrument", "by_eye", "n_operators"), count, f"{_BY_EYE}: the labels"),
    Macro("byEyeErrorMedian", _at("instrument", "by_eye", "error_median_mm"), distance, _BY_EYE),
    Macro(
        "byEyeErrorNinetyFive",
        _at("instrument", "by_eye", "error_p95_mm"),
        distance,
        f"{_BY_EYE}: 95th percentile, interpolated between ranks",
    ),
    Macro("byEyeErrorMax", _at("instrument", "by_eye", "error_max_mm"), distance, _BY_EYE),
    # data
    Macro("nDemosTrainedClustered", _at("datasets", "clustered", "n_trained"), count, "data/audit/clustered.json"),
    Macro("nDemosTrainedSpread", _at("datasets", "spread", "n_trained"), count, "data/audit/spread.json"),
    Macro(
        "nRerecordedClustered",
        _rerecorded("clustered"),
        count,
        "data/recording/clustered_placements.jsonl: kept episodes with at least one earlier take",
    ),
    Macro(
        "nRerecordedSpread",
        _rerecorded("spread"),
        count,
        "data/recording/spread_placements.jsonl: kept episodes with at least one earlier take",
    ),
    Macro(
        "demoPlacementErrorMedian",
        _at("datasets", "both", "placement_error_median_mm"),
        distance,
        f"{_AUDITS}: measured start to planned target",
    ),
    Macro(
        "demoPlacementErrorMax",
        _at("datasets", "both", "placement_error_max_mm"),
        distance,
        f"{_AUDITS}: measured start to planned target",
    ),
    Macro(
        "clusteredScatterMax",
        _at("datasets", "clustered", "scatter_max_mm"),
        distance,
        "data/audit/clustered.json: trained starts about the mean of their point",
    ),
    Macro(
        "spreadSpacingMin",
        _at("datasets", "spread", "spacing_min_mm"),
        distance,
        "data/audit/spread.json: trained starts",
    ),
    Macro(
        "nRecordingDays",
        _length("datasets", "both", "recording_days"),
        count,
        "data/recording/*_placements.jsonl: dates of the kept episodes",
    ),
    # T1b, T1a, T3
    *_condition_macros("Replay", REPLAY, "T1b", "block_fit"),
    *_wide_macros("radiusReplay", REPLAY),
    _shift_macro("plateauShiftReplay", REPLAY),
    Macro("nReplayEpisodes", _replay_episodes, count, f"{_TRIALS}, {REPLAY}, scored: distinct replayed episodes"),
    *_condition_macros("Clustered", CLUSTERED, "T1a", "pooled_fit"),
    *_wide_macros("radiusClustered", CLUSTERED, "pooled_fit"),
    _shift_macro("plateauShiftClustered", CLUSTERED, "pooled_fit"),
    Macro("nTrialsClusteredThreshold", _block(CLUSTERED, "n_scored"), count, f"{_TRIALS}, block T1a, scored"),
    Macro("nVoidClusteredThreshold", _block(CLUSTERED, "n_void"), count, f"{_TRIALS}, block T1a, void"),
    Macro(
        "successFarClustered",
        _successes("conditions", CLUSTERED, "anchors", "far"),
        count,
        f"{_TRIALS}, block T1a, far anchors",
    ),
    Macro(
        "nFarClustered",
        _logged(_at("conditions", CLUSTERED, "anchors", "far", "n")),
        count,
        f"{_TRIALS}, block T1a, far anchors",
    ),
    *_block_fit_macros("radiusClusteredThreshold", CLUSTERED),
    *_wide_macros("radiusClusteredThreshold", CLUSTERED),
    _shift_macro("plateauShiftClusteredThreshold", CLUSTERED),
    *_condition_macros("Spread", SPREAD, "T3", "pooled_fit"),
    *_wide_macros("radiusSpread", SPREAD, "pooled_fit"),
    _shift_macro("plateauShiftSpread", SPREAD, "pooled_fit"),
    Macro("nTrialsSpreadThreshold", _block(SPREAD, "n_scored"), count, f"{_TRIALS}, block T3, scored"),
    Macro("nVoidSpreadThreshold", _block(SPREAD, "n_void"), count, f"{_TRIALS}, block T3, void"),
    *_block_fit_macros("radiusSpreadThreshold", SPREAD),
    *_wide_macros("radiusSpreadThreshold", SPREAD),
    # T2
    *_workspace_macros(),
    # compare
    *_comparison_macros("ClusteredReplay", "clustered_minus_replay", f"{CLUSTERED} minus {REPLAY}"),
    Macro(
        "probClusteredExceedsReplay",
        _at("comparisons", "clustered_minus_replay", "prob_greater"),
        probability,
        f"block fits of {CLUSTERED} and {REPLAY}, taken as independent; a tie on the grid counts half",
    ),
    *_comparison_macros("SpreadClustered", "spread_minus_clustered", f"{SPREAD} minus {CLUSTERED}"),
    Macro(
        "probSpreadExceedsClustered",
        _at("comparisons", "spread_minus_clustered", "prob_greater"),
        probability,
        f"block fits of {SPREAD} and {CLUSTERED}, taken as independent; a tie on the grid counts half",
    ),
    # calc
    Macro(
        "resetSensitivityPlaced",
        _at("reset_sensitivity", "placed"),
        percent,
        f"block fit of {CLUSTERED} at the placement errors of the scored trials (a calculation)",
    ),
    Macro(
        "resetSensitivityByEye",
        _at("reset_sensitivity", "by_eye"),
        percent,
        f"block fit of {CLUSTERED} at the by-eye errors of {_BY_EYE} (a calculation)",
    ),
    # scene
    Macro(
        "nTrialsBasketNamed",
        _at("scene", "n_named_basket"),
        count,
        f"{_SCENE} whose first frames show the basket beyond the audit's limit from its place",
    ),
    Macro(
        "nTrialsWristNamed",
        _at("scene", "n_named_wrist"),
        count,
        f"{_SCENE} whose first frames show the wrist camera beyond the audit's limit from its place",
    ),
    Macro(
        "nTrialsBasketMoved",
        _at("scene", "n_basket_moved"),
        count,
        f"{_SCENE} that the audit marks as having moved the basket",
    ),
    Macro(
        "basketStartMax",
        _at("scene", "basket_start_max_mm"),
        distance,
        f"{_SCENE}: the basket at the first frames against its reference, the larger of across and down",
    ),
    Macro(
        "wristStartMax",
        _at("scene", "wrist_start_max_px"),
        one_decimal,
        f"{_SCENE}: the fixed jaw in the wrist camera's first frames against its reference, the larger of across "
        "and down",
    ),
    # second
    Macro("nSecondScored", _at("second_scoring", "all", "n_items"), count, f"{_SECOND}: the trials scored again"),
    Macro(
        "nSecondScoredAgree",
        _at("second_scoring", "all", "n_agree"),
        count,
        f"{_SECOND}: the trials whose second score is the first score",
    ),
    Macro(
        "nSecondScoredCannotTell",
        _at("second_scoring", "all", "n_cannot_tell"),
        count,
        f"{_SECOND}: the trials the second scorer could not tell",
    ),
    # all
    Macro("nTrialsTotal", _logged(_at("provenance", "n_scored")), count, f"{_TRIALS}, scored"),
    Macro("nVoidTotal", _logged(_at("provenance", "n_void")), count, f"{_TRIALS}, void"),
    Macro("voidReasons", _void_reasons, void_reasons_text, f"{_TRIALS}: void_reason, as typed"),
    Macro(
        "resetErrorMedian", _at("conduct", "reset_error_median_mm"), distance, f"{_TRIALS}, scored: placed to target"
    ),
    Macro("resetErrorMax", _at("conduct", "reset_error_max_mm"), distance, f"{_TRIALS}, scored: placed to target"),
    Macro(
        "viewCreptMedian",
        _at("conduct", "view_crept", "median_mm"),
        two_decimals,
        f"{_TRIALS}, scored: execution.view_crept_mm, as the session read it before the attempt",
    ),
    Macro(
        "viewCreptMax",
        _at("conduct", "view_crept", "max_mm"),
        two_decimals,
        f"{_TRIALS}, scored: execution.view_crept_mm, as the session read it before the attempt",
    ),
    Macro(
        "trialMinutesMedian",
        _at("conduct", "trial_minutes_median"),
        one_decimal,
        f"{_TRIALS}: one record's ts_start to the next one's in the same log",
    ),
    Macro("nTrialDays", _length("conduct", "trial_days"), count, f"{_TRIALS}: dates of ts_start"),
    Macro("nOperators", _length("conduct", "people"), count, f"{_TRIALS}: operator and scorer codes"),
    Macro("nDeviations", _at("deviations", "n"), count, "docs/decisions.md: rows marked DEVIATION"),
    Macro(
        "evalCheckpoint",
        _at("conduct", "checkpoints", "evaluation_step"),
        count,
        f"{_TRIALS}: checkpoint of the policy trials",
    ),
    Macro("execClamp", _setting("clamp_deg", _EVERY), plain_number, f"{_TRIALS}: execution.clamp_deg"),
    Macro(
        "execEnsembleCoeff",
        _setting("temporal_ensemble_coeff", _POLICIES),
        plain_number,
        f"{_TRIALS}: execution.temporal_ensemble_coeff of the policy trials",
    ),
    Macro("execControlRate", _setting("control_hz", _EVERY), plain_number, f"{_TRIALS}: execution.control_hz"),
    Macro(
        "execControlRateAchieved",
        _at("conduct", "execution", "achieved_hz", "policy", "median"),
        one_decimal,
        f"{_TRIALS}, scored policy trials: the steps sent after the first over execution.loop_s, the time from "
        "the first step sent to the last; median over the trials",
    ),
    Macro("execTimeLimit", _setting("time_limit_s", _EVERY), plain_number, f"{_TRIALS}: execution.time_limit_s"),
    Macro("lerobotCommit", _at("conduct", "code", "lerobot_commit"), str, f"{_TRIALS}: versions.lerobot"),
)


def macro_values(results: Results) -> dict[str, dict[str, str]]:
    """The values the results give the paper's macros, as `paper/make_numbers_stub.py` takes them:
    `{name: {"value": text, "source": where from}}`. A macro whose result is None is left out and stays `??`."""
    values: dict[str, dict[str, str]] = {}
    for macro in MACROS:
        found = macro.find(results)
        if found is not None:
            values[macro.name] = {"value": macro.write(found), "source": macro.source}
    return values
