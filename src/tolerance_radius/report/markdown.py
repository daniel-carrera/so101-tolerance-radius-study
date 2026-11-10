"""The report for a person: `analysis/report/report.md`, written from the results.

Every number is given with where it came from. The five predictions of `protocol/protocol.md`, section 9, are each set
beside what was found. The void trials are listed with their reasons, the second scoring from video is set beside the
first, the trials that began with the basket or the wrist camera out of place are named, and the last section but one
lists what the analysis could not compute and why. The last section gives every number macro of the paper with its
value or `??`.

The text is a function of the results alone: no clock, no machine name, so two runs on the same inputs give the same
bytes.

Standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tolerance_radius.report import estimates
from tolerance_radius.report.bookkeeping import ACHIEVED_RATE_MEANING, SCENE_MM_PER_PX_SOURCE, SIMULATED_WIDTHS_MM
from tolerance_radius.report.build import CLUSTERED, PREDICTION_CURVE, REGIONS, REPLAY, SPREAD
from tolerance_radius.report.macros import MACROS, NOT_PRODUCED, rounded

Results = Mapping[str, Any]

_NAMES = {CLUSTERED: "the clustered policy", SPREAD: "the spread policy", REPLAY: "the replay"}
_TITLES = {CLUSTERED: "Clustered policy", SPREAD: "Spread policy", REPLAY: "Replay"}
_REGION_NAMES = {"trained_cells": "trained cells", "held_out_cells": "held-out cells"}
_POLICIES = (CLUSTERED, SPREAD)


def render_report(results: Results, values: Mapping[str, Mapping[str, str]]) -> str:
    """The text of `report.md`. `values` are the macro values of the same results (`macros.macro_values`)."""
    lines: list[str] = []
    if results["banner"]:
        lines += [f"**{results['banner']}**", ""]
    lines += ["# Analysis report", ""]
    lines += _head(results)
    lines += _answer(results)
    lines += _distances(results)
    lines += _instrument(results)
    lines += _datasets(results)
    lines += _trials(results)
    for condition in (CLUSTERED, REPLAY, SPREAD):
        lines += _condition(results, condition)
    lines += _comparisons(results)
    lines += _workspace(results)
    lines += _sensitivity(results)
    lines += _predictions(results)
    lines += _descriptive(results)
    lines += _second_scoring(results)
    lines += _scene(results)
    lines += _conduct(results)
    lines += _simulation(results)
    lines += _not_computed(results)
    lines += _macros(values)
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------------------------------------------------
# Small formatters
# --------------------------------------------------------------------------------------------------------------------


# Numbers are rounded as the paper's macros round them (half up on the decimal form), so that a number in this
# report and the same number in the paper never differ in their last digit.


def _mm(value: float | None, digits: int = 1) -> str:
    return "none" if value is None else f"{rounded(value, digits, minus='-')} mm"


def _span(low: float | None, high: float | None) -> str:
    return f"{_mm(low)} to {_mm(high)}"


def _num(value: float | None, digits: int = 2) -> str:
    return "none" if value is None else rounded(value, digits, minus="-")


def _share(value: float | None) -> str:
    return "none" if value is None else f"{rounded(value, 0, times=100, minus='-')} %"


def _shift(value: float) -> str:
    """A difference with its sign: `+1.5 mm`, `-0.5 mm`, and `0.0 mm` for none."""
    text = rounded(value, 1, minus="-")
    return f"{'+' if value > 0 and float(text) != 0.0 else ''}{text} mm"


def _of(entry: Mapping[str, Any] | None) -> str:
    return "none" if not entry else f"{entry['k']} of {entry['n']}"


def _wilson(entry: Mapping[str, Any]) -> str:
    if entry["wilson_lo"] is None:
        return "no interval"
    return f"{_share(entry['wilson_lo'])} to {_share(entry['wilson_hi'])}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return [*lines, ""]


def _files(names: Sequence[str | None]) -> str:
    found = [name for name in names if name]
    return ", ".join(f"`{name}`" for name in found) if found else "none"


def _section(title: str) -> list[str]:
    return [f"## {title}", ""]


def _level(results: Results) -> str:
    return f"{100.0 * results['interval_level']:.0f} %"


def _caveat(sentences: Sequence[str]) -> str:
    """The caveats of a result, to be printed beside it: what it rests on that does not establish a radius.

    Empty when the result has none. It starts with a space, so that it can follow a sentence.
    """
    if not sentences:
        return ""
    return " **Caveat:** " + " ".join(f"{text[0].upper()}{text[1:]}." for text in sentences)


def _open(fit: Mapping[str, Any]) -> str:
    if fit["open_below"] and fit["open_above"]:
        return "no: open on both sides"
    if fit["open_below"]:
        return 'open below: report "at most"'
    if fit["open_above"]:
        return 'open above: report "at least"'
    return "yes, on both sides"


# --------------------------------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------------------------------


def _head(results: Results) -> list[str]:
    source = results["provenance"]
    kind = "a draft" if results["draft"] else "the counted analysis"
    lines = [
        f"This is {kind}, written by `scripts/make_report.py` together with `results.json`. Every number below is "
        "computed from the files named in this section, by the rules of `protocol/protocol.md`, section 10. "
        "A number is followed by where it comes from.",
        "",
        f"- **Protocol:** `{source['protocol_file']}`, version {source['protocol_version']}, status "
        f"{source['protocol_status']}, SHA-256 `{source['protocol_sha256'][:12]}`.",
        f"- **Trial logs read:** {_files(source['logs'])}: {source['n_records']} records, {source['n_scored']} scored "
        f"and {source['n_void']} void.",
    ]
    if source["pilot_logs_not_read"]:
        lines.append(
            f"- **Pilot logs left alone:** {_files(source['pilot_logs_not_read'])}. Pilot trials are never analysed."
        )
    if source["project_commits"]:
        commits = ", ".join(f"`{commit}`" for commit in source["project_commits"])
        lines.append(f"- **The trials ran on** commit {commits} of this repository (from the records).")
    lines += [
        f"- **Audits:** {_files(list(source['audits'].values()))}. **Scene audits:** "
        f"{_files(source['scene_audits'])}.",
        f"- **Instrument files:** scale runs {_files(source['scale_runs'])}; repeated placements "
        f"{_files(source['repeat_sets'])}; resets by eye {_files(source['by_eye'])}.",
        f"- **Placement logs of the recorder:** {_files(list(source['placement_logs'].values()))}.",
        f"- **Decision log:** {_files([source['decisions']])}.",
        f"- **Second scoring from video:** {_files([source['second_scoring']])}.",
        "",
    ]
    return lines


def _fit_name(kind: str, fit: Mapping[str, Any]) -> str:
    return f"{'block' if kind == 'block_fit' else 'pooled'} ({', '.join(fit['blocks'])})"


def _fits(results: Results) -> list[tuple[str, str, Mapping[str, Any]]]:
    """Every fit of the results, as `(condition, kind, fit)`, in the order the report gives them."""
    return [
        (condition, kind, results["conditions"][condition][kind])
        for condition in (CLUSTERED, REPLAY, SPREAD)
        for kind in ("block_fit", "pooled_fit")
        if results["conditions"][condition][kind] is not None
    ]


def _answer(results: Results) -> list[str]:
    lines = _section("The radii at a glance")
    fits = _fits(results)
    if not fits:
        return [*lines, "No condition has a scored trial, so there is no radius.", ""]
    rows = []
    for condition, kind, fit in fits:
        check = fit["sensitivity"]
        rows.append(
            [
                _NAMES[condition],
                _fit_name(kind, fit),
                f"{fit['k']} of {fit['n']}",
                f"{fit['zero_k']} of {fit['zero_n']}" if fit["zero_n"] else "none asked",
                _mm(fit["r50_median_mm"]),
                _span(fit["r50_lo_mm"], fit["r50_hi_mm"]),
                _open(fit),
                "not made" if check is None else _mm(check["r50_median_mm"]),
                "" if check is None else _span(check["r50_lo_mm"], check["r50_hi_mm"]),
                "" if check is None else _shift(check["median_shift_mm"]),
            ]
        )
    lines += _table(
        [
            "Condition",
            "Fit",
            "Successes",
            "At d = 0",
            "Radius (median)",
            f"{_level(results)} interval",
            "Do the data bound it?",
            "Radius, longer lapse list",
            "Its interval",
            "Shift",
        ],
        rows,
    )
    lines += [
        "The radius is the posterior median of `r50`. Comparisons and predictions use block fits only; a pooled fit "
        "is every scored trial of a policy and is given as a check.",
        "",
    ]
    lines += _plateau(results, fits)
    if results["caveats"]:
        lines += [
            "**Computed from a fit that does not establish a radius** (by the rules fixed in advance, all the same; "
            "each is said again beside the result it concerns):",
            "",
        ]
        lines += [f"- {text[0].upper()}{text[1:]}." for text in results["caveats"]]
        lines += [""]
    return lines


def _plateau(results: Results, fits: Sequence[tuple[str, str, Mapping[str, Any]]]) -> list[str]:
    """The plateau check of protocol section 10, in words."""
    checks = [(condition, kind, fit) for condition, kind, fit in fits if fit["sensitivity"] is not None]
    if not checks:
        return [
            "**The plateau check was not made:** the protocol file states no longer list of lapse values "
            "(`analysis.lapse_values_sensitivity`), so no fit was made a second time.",
            "",
        ]
    first = checks[0][2]["sensitivity"]
    longer = ", ".join(f"{value:g}" for value in first["lapse_values"])
    lines = [
        f"**The plateau check** (protocol section 10): every fit is made a second time with the lapse values "
        f"{longer}, everything else the same. The first fit is the radius that is reported; the shift is the "
        "second fit's median minus the first's. The counts at d = 0 are the trials asked at a demonstration's own "
        "start among the trials of that fit.",
        "",
    ]
    limit = first["limit_mm"]
    if limit is None:
        return [
            *lines,
            "The protocol file states no limit for the shift (`analysis.sensitivity_limit_mm`), so no plateau is "
            "called in doubt here.",
            "",
        ]
    doubted = [(condition, kind, fit) for condition, kind, fit in checks if fit["sensitivity"]["plateau_in_doubt"]]
    if not doubted:
        return [*lines, f"No plateau is in doubt: no fit's two medians differ by more than {limit:g} mm.", ""]
    for condition, kind, fit in doubted:
        check = fit["sensitivity"]
        lines += [
            f"- **The plateau of {_NAMES[condition]} is in doubt** ({_fit_name(kind, fit)} fit): with the longer "
            f"list the radius is {_mm(check['r50_median_mm'])} ({_span(check['r50_lo_mm'], check['r50_hi_mm'])}) "
            f"against {_mm(fit['r50_median_mm'])}, a shift of {_shift(check['median_shift_mm'])}, more than the "
            f"protocol's {limit:g} mm. At d = 0 it succeeded in {fit['zero_k']} of {fit['zero_n']} trials. By "
            "protocol section 10 the paper says that this condition's radius is not established.",
        ]
    return [*lines, ""]


def _distances(results: Results) -> list[str]:
    scale = results["distance_scale"]
    lines = _section("Distances and the scale correction")
    if scale["n_runs"] == 0:
        lines += [
            "**No scale run was found, so no correction was made: every distance is as the camera reads it.** "
            "The protocol expects distances to read a few percent long (section 7).",
            "",
        ]
        return lines
    measured = (
        f"The {scale['n_runs']} scale run(s) read a true length as longer by factors from {scale['factor_min']:.4f} "
        f"to {scale['factor_max']:.4f} (mean {scale['factor']:.4f}); they differ by "
        f"{scale['spread_points']:.2f} percentage points."
    )
    if scale["applied"]:
        lines += [
            f"**Every distance in this report and in `results.json` is divided by {scale['factor']:.4f}.** {measured} "
            "That is within the one percentage point of protocol section 7, so the correction applies.",
            "",
            "Divided: every distance between two measured positions (a trial's distance to its demonstration, a "
            "placement error, a by-eye error, the spacing and scatter of the demonstrations). Not divided: positions "
            "(`x_mm`, `y_mm`), the distances the planner asked for (`level_mm`), and the scale check itself, which "
            "is given as read.",
            "",
        ]
    else:
        lines += [
            f"**Distances are left as the camera reads them.** {measured} Reason: {scale['why']}. "
            "The spread of the factor is the instrument's accuracy.",
            "",
        ]
    return lines


def _instrument(results: Results) -> list[str]:
    instrument = results["instrument"]
    lines = _section("The instrument (blocks V0 and V1)")
    check = instrument["scale_check"]
    if check is None:
        lines += ["No scale run (`data/instrument/ruler_*.csv`) was found.", ""]
    else:
        line = check["pass_line"]
        verdict = "passes" if check["passes"] else "**does not pass**"
        lines += [
            f"**Scale check:** over {check['n_steps']} steps of {check['true_step_mm']:g} mm in {check['n_runs']} "
            f"run(s), a measured step was off by {_mm(check['mean_abs_step_error_mm'], 2)} on average and by "
            f"{_mm(check['max_abs_step_error_mm'], 2)} at worst, as read. It {verdict} the protocol's line (mean at "
            f"most {line['mean_abs_mm']:g} mm, no step off by more than {line['max_abs_mm']:g} mm).",
            "",
        ]
        rows = [
            [
                f"`{run['file']}`",
                run["n_steps"],
                f"{run['measured_span_mm']:.2f} of {run['true_span_mm']:g} mm",
                f"{run['factor']:.4f}",
                _mm(run["mean_abs_step_error_mm"], 2),
                _mm(run["max_abs_step_error_mm"], 2),
                _mm(run["max_off_line_mm"], 2),
            ]
            for run in check["runs"]
        ]
        header = ["Run", "Steps", "Measured span", "Factor", "Mean step error", "Worst step", "Off the line"]
        lines += _table(header, rows)
    repeat = instrument["repeatability"]
    if repeat is None:
        lines += ["No file of repeated placements (`data/instrument/repeat_*.csv`) was found.", ""]
    else:
        lines += [
            f"**Repeatability:** {repeat['n']} placements in {repeat['n_sets']} set(s) ({_files(repeat['files'])}) "
            f"lay at most {_mm(repeat['scatter_max_mm'], 2)} from the mean of their set, as read.",
            "",
        ]
    by_eye = instrument["by_eye"]
    if by_eye is None:
        lines += ["No file of resets by eye (`data/instrument/byeye_*.csv`) was found.", ""]
    else:
        lines += [
            f"**Resets by eye:** {by_eye['n']} resets by {by_eye['n_operators']} person(s) "
            f"({', '.join(by_eye['operators'])}) landed a median of {_mm(by_eye['error_median_mm'], 2)} from where "
            f"the cube was for the reference picture; 95th percentile {_mm(by_eye['error_p95_mm'], 2)} (interpolated "
            f"between ranks), largest {_mm(by_eye['error_max_mm'], 2)}. From "
            f"{_files(results['provenance']['by_eye'])}.",
            "",
        ]
    lines += [
        "Not in any committed record: the disagreement of each session's camera solve with the stored one (the tools "
        "print it).",
        "",
    ]
    return lines


def _datasets(results: Results) -> list[str]:
    lines = _section("The two datasets")
    rows = []
    for name in ("clustered", "spread"):
        summary = results["datasets"][name]
        if summary is None:
            lines += [f"The {name} set has no audit (`data/audit/{name}.json`).", ""]
            continue
        recording = summary["recording"]
        if recording is None:
            recorded = "no placement log"
        elif not recording["matches_audit"]:
            recorded = "its placement log does not hold the audited episodes"
        else:
            recorded = (
                f"{recording['n_rerecorded_episodes']} episodes recorded again ({recording['n_rerecords']} extra "
                f"takes); {recording['n_superseded']} earlier lines recorded over; days {', '.join(recording['days'])}"
            )
        if name == "clustered":
            special = f"scatter about a point at most {_mm(summary['scatter_max_mm'], 2)}"
        else:
            special = f"closest two starts {_mm(summary['spacing_min_mm'], 2)} apart"
        rows.append(
            [
                name,
                f"{summary['n_trained']} of {summary['n_episodes']}",
                _mm(summary["placement_error_median_mm"], 2),
                _mm(summary["placement_error_max_mm"], 2),
                special,
                recorded,
            ]
        )
    if rows:
        header = ["Set", "Trained on", "Start to target, median", "Largest", "Layout as measured", "Recording"]
        lines += _table(header, rows)
        both = results["datasets"]["both"]
        if both["placement_error_median_mm"] is not None:
            lines += [
                f"Both sets together: a start lay a median of {_mm(both['placement_error_median_mm'], 2)} from its "
                f"planned target, at most {_mm(both['placement_error_max_mm'], 2)} (from the audits).",
                "",
            ]
        lines += [
            '"Recorded again" counts kept episodes that had at least one earlier take in the same sitting. "Recorded '
            'over" counts lines of the placement log whose episode number was saved again later: a smoke test or a '
            "restart. No record says which of those were practice episodes.",
            "",
        ]
    lines += [
        "Not in any committed record: torn frames and frame pacing of the new datasets (the checks print them).",
        "",
    ]
    return lines


def _trials(results: Results) -> list[str]:
    lines = _section("Trials")
    blocks = results["blocks"]
    if blocks is None:
        return [*lines, "No counted trial log was found.", ""]
    rows = [
        [name, " and ".join(entry["conditions"]), entry["planned"], entry["n_scored"], entry["n_void"]]
        for name, entry in blocks.items()
    ]
    source = results["provenance"]
    rows.append(["all", "", sum(entry["planned"] for entry in blocks.values()), source["n_scored"], source["n_void"]])
    lines += _table(["Block", "Condition", "Planned", "Scored", "Void"], rows)
    conduct = results["conduct"]
    if not conduct["void_trials"]:
        lines += ["No trial was void.", ""]
    else:
        lines += ["**Void trials**, in the order of the logs. A void trial is not analysed and was asked again.", ""]
        lines += _table(
            ["Trial", "Block", "Condition", "Reason, as typed"],
            [
                [trial["trial_id"], trial["block"], trial["condition"], trial["reason"]]
                for trial in conduct["void_trials"]
            ],
        )
        counted = "; ".join(f"{reason['reason']} ({reason['n']})" for reason in conduct["void_reasons"])
        lines += [f"By reason: {counted}.", ""]
    if results["notes"]:
        lines += ["**Notes on the trials:**", ""]
        lines += [f"- {note[0].upper()}{note[1:].rstrip('.')}." for note in results["notes"]]
        lines += [""]
    return lines


def _condition(results: Results, condition: str) -> list[str]:
    section = results["conditions"][condition]
    lines = _section(f"{_TITLES[condition]} (`{condition}`)")
    if not section["trials"]:
        return [*lines, "No scored trial.", ""]
    block = section["threshold_block"]
    anchors = section["anchors"]
    far = f", {_of(anchors['far'])} at the far anchors" if anchors["far"]["n"] else ""
    lines += [
        f"{section['n_scored']} scored trials and {section['n_void']} void, from the counted logs. "
        f"In block {block}: {_of(anchors['zero'])} succeeded at d = 0{far}. "
        f"Over all its scored trials, the largest distance with a success is {_mm(section['max_success_d_mm'])} "
        f"and the smallest with a failure {_mm(section['min_failure_d_mm'])}.",
        "",
    ]
    rows = [
        [f"{entry['lo_mm']:g} to {entry['hi_mm']:g} mm", f"{entry['k']} of {entry['n']}", _wilson(entry)]
        for entry in section["bins"]
    ]
    header = ["Distance (closed below, open above)", "Successes", f"Wilson {_level(results)} interval"]
    lines += _table(header, rows)
    for kind, title in (("block_fit", "Block fit"), ("pooled_fit", "Pooled fit")):
        fit = section[kind]
        if fit is None:
            if kind == "block_fit":
                lines += [f"**Block fit:** none, block {block} has no scored trial.", ""]
            continue
        check = fit["sensitivity"]
        checked = ""
        if check is not None:
            doubt = " **The plateau is in doubt.**" if check["plateau_in_doubt"] else ""
            checked = (
                f" With lapse values up to {max(check['lapse_values']):g}: radius {_mm(check['r50_median_mm'])} "
                f"({_span(check['r50_lo_mm'], check['r50_hi_mm'])}; {_open(check)}), a shift of "
                f"{_shift(check['median_shift_mm'])}; lapse {_num(check['lapse_mean'], 3)} (posterior mean).{doubt}"
            )
        lines += [
            f"**{title}** ({', '.join(fit['blocks'])}; {fit['k']} successes in {fit['n']} trials): radius "
            f"{_mm(fit['r50_median_mm'])}, {_level(results)} interval {_span(fit['r50_lo_mm'], fit['r50_hi_mm'])}. "
            f"Bounded by the data: {_open(fit)}. Width {_mm(fit['width_median_mm'], 2)} (median), lapse "
            f"{_num(fit['lapse_mean'], 3)} and floor {_num(fit['floor_mean'], 3)} (posterior means). "
            f"Its trials: {fit['zero_k']} of {fit['zero_n']} at d = 0; farthest success "
            f"{_mm(fit['max_success_d_mm'])}, nearest failure {_mm(fit['min_failure_d_mm'])}.{checked}",
            "",
        ]
    if condition == REPLAY:
        lines += [f"Demonstrations replayed: episodes {', '.join(str(e) for e in section['replay_episodes'])}.", ""]
    return lines


def _comparisons(results: Results) -> list[str]:
    lines = _section("Comparison of radii")
    names = {
        "clustered_minus_replay": "The clustered policy's radius minus the replay's",
        "spread_minus_clustered": "The spread policy's radius minus the clustered policy's",
    }
    for key, title in names.items():
        difference = results["comparisons"][key]
        if difference is None:
            lines += [f"**{title}:** not computed, it needs both block fits.", ""]
            continue
        lines += [
            f"**{title}:** {_mm(difference['median_mm'])} ({_level(results)} interval "
            f"{_span(difference['lo_mm'], difference['hi_mm'])}). Posterior probability that the first is larger: "
            f"{difference['prob_greater']:.4f}. Verdict by the rule of prediction 2: **{difference['verdict']}**."
            f"{_caveat(difference['caveats'])}",
            "",
        ]
    larger, smaller = estimates.VERDICT_PROBABILITY, 1.0 - estimates.VERDICT_PROBABILITY
    lines += [
        "From the two block fits, taken as independent, with every pair of grid values weighted exactly. "
        f'"Larger" is a probability of at least {larger:g}, "smaller" at most {smaller:.2f}. '
        "On the grid two radii can be equal; such a tie counts half for each side.",
        "",
    ]
    return lines


def _workspace(results: Results) -> list[str]:
    workspace = results["workspace"]
    lines = _section("The workspace test")
    if workspace is None:
        return [*lines, "Not run: its block has no scored trial.", ""]
    table = workspace["table"]
    lines += [
        f"{workspace['n_pairs']} placements were tested with both policies (block {workspace['block']}). "
        f"Both succeeded at {table['both']}, neither at {table['neither']}, only the spread policy at "
        f"{table['only_spread']}, only the clustered policy at {table['only_clustered']}. "
        f"Exact McNemar test over all pairs together: p = {_num(workspace['mcnemar_p'], 4)}.",
        "",
    ]
    rows = []
    for policy in _POLICIES:
        success, distances = workspace["success"][policy], workspace["distances"][policy]
        lookup = workspace["lookup"][policy] if workspace["lookup"] else None
        predicted = "none"
        if lookup is not None:
            predicted = f"{lookup['expected']:.2f} ({lookup['lo']} to {lookup['hi']}): {lookup['verdict']}"
        rows.append(
            [
                _NAMES[policy],
                _of(success),
                _wilson(success),
                _mm(distances["median_mm"]),
                _mm(distances["max_mm"]),
                predicted,
            ]
        )
    header = ["Policy", "Successes", f"Wilson {_level(results)} interval", "Median d", "Largest d"]
    lines += _table([*header, f"Lookup prediction (expected, {_level(results)} interval of the count)"], rows)
    lines += [
        "The lookup prediction is the clustered policy's block fit, which has not seen these trials, at each trial's "
        "measured distance to its own policy's nearest demonstration. The interval is that of the predicted count: it "
        "carries the uncertainty of the fit and the chance in the trials. An observed count inside it, ends included, "
        f"matches.{_caveat(workspace['lookup']['caveats'] if workspace['lookup'] else ())}",
        "",
    ]
    rows = []
    for region in REGIONS:
        entry = workspace["by_region"][region]
        for policy in _POLICIES:
            found = entry[policy]
            successes = _of(found) if found["n"] else "none"
            rows.append(
                [_REGION_NAMES[region], entry["n_pairs"], _NAMES[policy], successes, _wilson(found), _expected(found)]
            )
    lines += [
        "By region. A pair's region is the grid cell of its measured position (the mean of its two placements). The "
        "lookup prediction's verdict is made over all pairs; the expected count and the interval of the predicted "
        "count are given for each group besides.",
        "",
    ]
    header = ["Region", "Pairs", "Policy", "Successes", f"Wilson {_level(results)} interval", "Lookup expects"]
    lines += _table(header, rows)

    def run_name(entry: Mapping[str, Any], policy: str) -> str:
        return f"`{entry['runs'][policy]}`" if entry["runs"][policy] else "a run the records do not name"

    by_run = workspace["by_run"]
    if len(by_run) == 1:
        lines += [
            f"By training run: every pair was run with one pair of runs, {run_name(by_run[0], CLUSTERED)} for the "
            f"clustered policy and {run_name(by_run[0], SPREAD)} for the spread policy (from the records' "
            "`checkpoint`).",
            "",
        ]
    elif by_run:
        rows = []
        for entry in by_run:
            for policy in _POLICIES:
                found = entry[policy]
                rows.append(
                    [
                        run_name(entry, policy),
                        entry["n_pairs"],
                        _NAMES[policy],
                        _of(found),
                        _wilson(found),
                        _expected(found),
                    ]
                )
        lines += [
            f"By training run. The pairs were run with {len(by_run)} pairs of runs (from the records' `checkpoint`); "
            "the paired test above is made over all of them together.",
            "",
        ]
        header = ["Run", "Pairs", "Policy", "Successes", f"Wilson {_level(results)} interval", "Lookup expects"]
        lines += _table(header, rows)
    return lines


def _sensitivity(results: Results) -> list[str]:
    sensitivity = results["reset_sensitivity"]
    lines = _section("Reset sensitivity (a calculation, not a measurement)")
    if sensitivity is None:
        return [*lines, "Not computed: it needs the block fit of the clustered policy.", ""]
    if sensitivity["by_eye"] is None:
        by_eye = "With by-eye errors: not computed, there is no by-eye file."
    else:
        by_eye = (
            f"With each of the {sensitivity['n_by_eye']} by-eye errors of block V1 instead: "
            f"{_share(sensitivity['by_eye'])}."
        )
    lines += [
        "The clustered policy's block fit, at a demonstration's own start, with a reset that is off by each of the "
        f"{sensitivity['n_placed']} placement errors of this study's scored trials: {_share(sensitivity['placed'])} "
        f"success. {by_eye}{_caveat(sensitivity['caveats'])}",
        "",
    ]
    return lines


def _expected(found: Mapping[str, Any]) -> str:
    """What the lookup prediction expects of a group of pairs: the count and the interval of the predicted count."""
    if found["lookup_expected"] is None:
        return "none"
    return f"{_num(found['lookup_expected'])} ({found['lookup_lo']} to {found['lookup_hi']})"


def _held(entry: Mapping[str, Any]) -> str:
    if entry["held"] is None:
        return f"**not judged** ({entry['why']})"
    return "**held**" if entry["held"] else "**did not hold**"


def _predictions(results: Results) -> list[str]:
    lines = _section("The predictions of protocol section 9")
    by_id = {entry["id"]: entry for entry in results["predictions"]}
    rule = PREDICTION_CURVE
    level = _level(results)

    curve = by_id[1]["numbers"]
    lines += [
        f"1. **The curve.** Predicted: at least {rule['zero_min_successes']} successes in {rule['zero_trials']} at "
        f"d = 0, at most {rule['far_max_successes']} in {rule['far_trials']} at the far anchors, and a radius between "
        f"{rule['r50_min_mm']:g} and {rule['r50_max_mm']:g} mm. Found: {_of(curve['zero'])} at d = 0, "
        f"{_of(curve['far'])} at the far anchors, radius {_mm(curve['r50_median_mm'])} (block fit, posterior median). "
        f"{_held(by_id[1])}.{_caveat(by_id[1]['caveats'])}",
    ]

    hardware = by_id[2]
    difference = hardware["numbers"]["difference"]
    found = "no comparison"
    if difference is not None:
        found = (
            f"the difference is {_mm(difference['median_mm'])} ({_span(difference['lo_mm'], difference['hi_mm'])}), "
            f"and the probability that the policy's radius is larger is {difference['prob_greater']:.4f}: "
            f"{difference['verdict']}"
        )
    lines += [
        "2. **Hardware.** Predicted: the clustered policy's radius is larger than the replay's, by a posterior "
        f"probability of at least {estimates.VERDICT_PROBABILITY:g}. Found: {found}. {_held(hardware)}."
        f"{_caveat(hardware['caveats'])}",
    ]

    layout = by_id[3]
    found, said = "no pair", ""
    if layout["held"] is not None:
        counted = layout["numbers"]
        regions = "; ".join(
            f"in {_REGION_NAMES[region]} {entry['k_spread']} against {entry['k_clustered']} of {entry['n_pairs']}"
            for region, entry in counted["by_region"].items()
        )
        found = (
            f"the spread policy succeeded at {counted['k_spread']} of {counted['n_pairs']} placements and the "
            f"clustered policy at {counted['k_clustered']} ({regions}); exact McNemar p = "
            f"{_num(counted['mcnemar_p'], 4)}"
        )
        said = {
            estimates.LAYOUT_SPREAD: "",
            estimates.LAYOUT_CLUSTERED: " The prediction failed: the test decided, and for the clustered policy.",
            estimates.LAYOUT_UNDECIDED: " The test did not decide: the counts are what there is to report.",
        }[counted["outcome"]]
    lines += [
        "3. **Layout.** Predicted: at the placements of the workspace test the spread policy succeeds more often than "
        f"the clustered one, by the exact paired test over all pairs: p below {estimates.LAYOUT_P_BELOW:g} with the "
        f"spread policy ahead. Found: {found}. {_held(layout)}.{said}",
    ]

    lookup = by_id[4]
    found = "no prediction"
    if lookup["held"] is not None:
        parts = []
        for policy in _POLICIES:
            entry = results["workspace"]["lookup"][policy]
            parts.append(
                f"{_NAMES[policy]} {entry['observed']} against {entry['expected']:.2f} "
                f"({entry['lo']} to {entry['hi']}): {entry['verdict']}"
            )
        found = "; ".join(parts)
    lines += [
        f"4. **Lookup.** Predicted: each policy's count in the workspace test lies inside the {level} interval of its "
        f"lookup prediction. Found: {found}. {_held(lookup)}.{_caveat(lookup['caveats'])}",
    ]

    difference = by_id[5]["numbers"]["difference"]
    found = "the threshold block of the spread policy or of the clustered policy was not run"
    if difference is not None:
        found = (
            f"the spread policy's radius minus the clustered policy's is {_mm(difference['median_mm'])} "
            f"({_span(difference['lo_mm'], difference['hi_mm'])}); probability that it is larger "
            f"{difference['prob_greater']:.4f}"
        )
    lines += [
        f"5. **Layout and radius.** No prediction was made; reported as an estimate. Found: {found}."
        f"{_caveat(by_id[5]['caveats'])}",
        "",
    ]
    return lines


def _descriptive(results: Results) -> list[str]:
    lines = _section("Descriptive only")
    rows = []
    for condition in (CLUSTERED, REPLAY, SPREAD):
        section = results["conditions"][condition]
        if not section["trials"]:
            continue
        described = section["descriptive"]
        hull = described["hull"]
        inside = "does not apply"
        if hull is not None:
            inside = f"{_of(hull['inside'])} inside, {_of(hull['outside'])} outside"
        failures = "; ".join(
            f"{entry['label']} {entry['n']} ({_span(entry['d_min_mm'], entry['d_max_mm'])})"
            for entry in described["failures"]
        )
        rows.append([_NAMES[condition], inside, failures or "none", f"{described['duration_median_s']:.0f} s"])
    if not rows:
        return [*lines, "No scored trial.", ""]
    header = ["Condition", "Successes inside and outside the hull of the demonstrations"]
    lines += _table([*header, "Failure labels (count, range of d)", "Median duration"], rows)
    lines += [
        "A duration runs from a record's `ts_start` to its `ts_end`: from just before the handoff gate to after the "
        "videos were closed, in whole seconds. It is longer than the attempt. The control loop's own duration is "
        "`loop_s`, and the rate made from it is under \"How the trials were run\".",
        "",
        "**Success by the cell of the placement, at each distance.** A trial counts as in a trained cell or in a "
        "held-out cell by the grid cell its measured position falls in. Beyond the largest gap of a layout every "
        "placement is in a held-out cell, so on the far side of a curve distance and a cell without demonstrations "
        "cannot be told apart (protocol section 10).",
        "",
    ]
    for condition in (CLUSTERED, REPLAY, SPREAD):
        section = results["conditions"][condition]
        if not section["trials"]:
            continue
        split = section["descriptive"]["by_region"]
        bins = {
            region: {(entry["lo_mm"], entry["hi_mm"]): entry for entry in split[region]["bins"]} for region in REGIONS
        }
        rows = []
        for low, high in sorted({edges for found in bins.values() for edges in found}):
            row = [f"{low:g} to {high:g} mm"]
            for region in REGIONS:
                entry = bins[region].get((low, high))
                row.append("no trial" if entry is None else f"{_of(entry)} ({_wilson(entry)})")
            rows.append(row)
        totals = (_of(split[region]) if split[region]["n"] else "no trial" for region in REGIONS)
        rows.append(["all distances", *totals])
        off_grid = section["descriptive"]["n_off_grid"]
        aside = f" {off_grid} trial(s) were measured off the grid and are in neither column." if off_grid else ""
        lines += [
            f"{_TITLES[condition]} (`{condition}`), successes of trials with a Wilson {_level(results)} interval."
            f"{aside}",
            "",
        ]
        lines += _table(["Distance", "In trained cells", "In held-out cells"], rows)
    return lines


def _second_scoring(results: Results) -> list[str]:
    """The second scoring from video: how many trials were scored again, and in how many the two scores agree."""
    second = results["second_scoring"]
    lines = _section("The second scoring from video")
    if second is None:
        if results["blocks"] is None:
            return [*lines, "No counted trial log was found.", ""]
        return [
            *lines,
            "No agreement file (`data/second_scoring/main/agreement.json`) was found: no trial has a second score "
            "in this report.",
            "",
        ]
    if second["why_not"] is not None:
        return [*lines, f"Not reported: {second['why_not']}.", ""]
    whole, scorers = second["all"], second["scorers"]
    level = f"{100.0 * second['interval_level']:.0f} %"
    lines += [
        f"{whole['n_items']} of the {second['n_scored']} scored trials were scored again from their videos alone, "
        f"with neither the condition nor the distance shown to the scorer: one trial in {second['one_in']} of each "
        f"condition of each block, rounded up, drawn with seed {second['seed']} from {_files(second['logs'])} after "
        f"its last trial (protocol section 8). From `{second['file']}`, written by `scripts/second_scoring.py "
        "compare`. The first score, given at the rig, is the one the analysis uses whatever the second says.",
        "",
    ]
    if whole["n_compared"]:
        agreement = (
            f"the two scores agree in {whole['n_agree']} of the {whole['n_compared']} trials the second scorer could "
            f"tell: {_share(whole['agreement'])} (Wilson {level} interval {_wilson(whole)})"
        )
    else:
        agreement = "the second scorer could tell none of the trials, so there is no share that agree"
    lines += [
        f"- **Agreement:** {agreement}. They differ in {whole['n_disagree']}, and in {whole['n_cannot_tell']} the "
        'second scorer answered "cannot tell".',
        f"- **Who scored:** the first score was given at the rig by {_people(scorers['first'])}, the second from "
        f"the video by {_people(scorers['second'])} (role codes, from the trial records and from the scores).",
    ]
    if second["same_person"]:
        lines.append(
            f"- **Both scores of all {whole['n_items']} trials are by the same person**, the second given from the "
            "video with the trials' identities hidden: no second person scored (protocol section 8 provides for "
            "this)."
        )
    elif scorers["n_same_role"]:
        lines.append(
            f"- **In {scorers['n_same_role']} of the {whole['n_items']} trials both scores are by the same person**, "
            "the second given from the video with the trials' identities hidden."
        )
    lines += ["", "First score against second score, in trials:", ""]
    table, untold = whole["table"], whole["cannot_tell_first"]
    header = ["First score (at the rig)", "Second: success", "Second: failure", "Second: cannot tell"]
    rows = [
        [first, table[f"{first}_success"], table[f"{first}_failure"], untold[first]] for first in ("success", "failure")
    ]
    lines += _table(header, rows)
    rows = []
    for kind, tally in second["by_kind"].items():
        agree = f"{tally['n_agree']} of {tally['n_compared']}" if tally["n_compared"] else "none compared"
        interval = _wilson(tally) if tally["n_compared"] else "no interval"
        rows.append([kind, tally["n_items"], agree, tally["n_disagree"], tally["n_cannot_tell"], interval])
    lines += ["By kind of trial:", ""]
    header = ["Kind of trial", "Scored again", "Agree", "Differ", "Cannot tell"]
    lines += _table([*header, f"Wilson {level} interval of the share that agree"], rows)
    for entry in second["disagreements"]:
        lines.append(f"- {entry['trial_id']}: {entry['first']} at the rig, {entry['second']} from the video.")
    for entry in second["cannot_tell"]:
        lines.append(f"- {entry['trial_id']}: {entry['first']} at the rig; the second scorer could not tell.")
    if second["disagreements"] or second["cannot_tell"]:
        lines.append("")
    return lines


def _scene(results: Results) -> list[str]:
    """Where the basket and the wrist camera stood when the scored trials began, and the trials named for either."""
    scene = results["scene"]
    lines = _section("The scene at each trial's start")
    if scene is None:
        if results["blocks"] is None:
            return [*lines, "No counted trial log was found.", ""]
        return [
            *lines,
            "No scene audit (`data/audit/scene_<log>.json`) holds a scored trial of this report: where the basket "
            "and the wrist camera stood when a trial began is not reported.",
            "",
        ]
    basket_limit, wrist_limit = scene["basket_named_mm"], scene["wrist_named_px"]
    lines += [
        "The basket stands on tape and the arm can carry it off; the same catch can move the wrist camera in its "
        "mount. So the first and the last frames of every trial's two videos were read for the basket's place and "
        "for the wrist camera's, each against its reference picture, and a trial that began with the basket more "
        f"than {basket_limit:g} mm or the wrist camera more than {wrist_limit:g} px from its place is named here "
        f"(protocol section 8). From {_files(scene['files'])}, written by `scripts/rig/scene_audit.py`; a trial's "
        "kind and outcome are those of its record. An offset is the larger of across and down, as the audit reads "
        "it: millimetres on the floor for the basket, pixels of the wrist camera's picture for that camera.",
        "",
    ]

    def listed(names: Sequence[str]) -> str:
        return ", ".join(names)

    read = (
        f"- **Trials read:** {scene['n_audited']} of the {scene['n_scored']} scored trials are in the audit. The "
        f"basket's place at the start was read in {scene['n_basket_read']} of them and the wrist camera's in "
        f"{scene['n_wrist_read']}."
    )
    if scene["not_audited"]:
        read += f" **Not in the audit:** {listed(scene['not_audited'])}."
    if scene["basket_unread"] or scene["wrist_unread"]:
        parts = [
            f"the {camera} camera's of {listed(scene[key])}"
            for camera, key in (("front", "basket_unread"), ("wrist", "wrist_unread"))
            if scene[key]
        ]
        read += f" **Videos whose first frames did not say:** {'; '.join(parts)}."
    else:
        read += " The first frames of every video in the audit could be read."
    lines.append(read)

    moved_rule = (
        f"it stood more than {scene['basket_moved_limit_mm']:g} mm from where the trial began, at the trial's last "
        "frames or, where the arm hid it then, when the next trial of the session began: the audit's rule"
    )
    if scene["basket_moved"]:
        trials = "; ".join(
            f"{entry['trial_id']} ({_NAMES[entry['condition']]}, {entry['outcome']}, {_mm(entry['moved_mm'])})"
            for entry in scene["basket_moved"]
        )
        moved = f"- **The basket was moved in {scene['n_basket_moved']} trial(s)** ({moved_rule}): {trials}."
    else:
        moved = f"- **The basket was moved in no trial** ({moved_rule})."
    moved += " A trial in which the arm moves the basket stands as scored (protocol section 8)."
    if scene["moved_unknown"]:
        moved += (
            f" The end of {len(scene['moved_unknown'])} trial(s) says nothing of the basket (its last frames did "
            "not show the basket, and the audit found no later trial of the session to judge it by): "
            f"{listed(scene['moved_unknown'])}."
        )
    lines.append(moved)

    for what, key, limit in (
        ("basket", "named_basket", f"{basket_limit:g} mm"),
        ("wrist camera", "named_wrist", f"{wrist_limit:g} px"),
    ):
        count = scene[f"n_{key}"]
        if count:
            lines.append(
                f"- **Named for the {what}:** {count} trial(s) began with it more than {limit} from its place: "
                f"{listed(scene[key])}."
            )
        else:
            lines.append(f"- **Named for the {what}:** no trial began with it more than {limit} from its place.")
    wrist_max = "none read" if scene["wrist_start_max_px"] is None else f"{_num(scene['wrist_start_max_px'], 1)} px"
    lines.append(
        f"- **Largest offsets at a start:** the basket {_mm(scene['basket_start_max_mm'])}, the wrist camera "
        f"{wrist_max}."
    )
    success = scene["success"]

    def succeeded(group: str, which: str, none: str) -> str:
        found = success[group]
        return f"{found['k']} of the {found['n']} trials {which} succeeded" if found["n"] else none

    counted = (
        "- **Successes, as plain counts and with no test:** "
        f"{succeeded('named', 'named for the basket or for the wrist camera', 'no trial was named')}, and "
        f"{succeeded('unnamed', 'named for neither', 'no trial was read at its start and named for neither')}."
    )
    if success["start_not_read"]["n"]:
        counted += f" Besides, {succeeded('start_not_read', 'whose start could not be read', '')}."
    lines += [counted, ""]

    def bin_name(entry: Mapping[str, Any]) -> str:
        if entry["up_to_mm"] is None:
            return f"over {entry['over_mm']:g} mm"
        if entry["over_mm"] == 0.0:
            return f"up to {entry['up_to_mm']:g} mm"
        return f"over {entry['over_mm']:g} mm, up to {entry['up_to_mm']:g} mm"

    rows = [[bin_name(entry), entry["n"]] for entry in scene["basket_start_bins"]]
    rows.append(["all trials whose start was read", scene["n_basket_read"]])
    lines += _table(["The basket at a trial's start, from its place", "Trials"], rows)
    per_px = scene["mm_per_px"]
    # The fewest whole pixels that lie beyond the line; a line that is itself a whole number of pixels is not beyond.
    whole = math.floor(basket_limit / per_px + 1e-9) + 1
    pixels = (
        f"One pixel of the front camera is about {per_px:g} mm at the basket ({SCENE_MM_PER_PX_SOURCE}), so the "
        f"{basket_limit:g} mm line lies {_num(basket_limit / per_px, 1)} pixels from the basket's place, and an "
        f"offset of {whole} whole pixels, about {_mm(whole * per_px)}, is the first pixel step past it."
    )
    if scene["n_named_basket"]:
        past = (scene["basket_start_max_mm"] - basket_limit) / per_px
        pixels += (
            f" The trials named for the basket began {_mm(scene['named_basket_min_mm'])} to "
            f"{_mm(scene['basket_start_max_mm'])} from it: at most {_num(past, 1)} pixels past the line."
        )
    return [*lines, pixels, ""]


def _people(codes: Sequence[str]) -> str:
    """Role codes in a sentence: `op1`, `op1 and sc2`, or `nobody named`."""
    return " and ".join(codes) or "nobody named"


def _conduct(results: Results) -> list[str]:
    conduct = results["conduct"]
    lines = _section("How the trials were run")
    if conduct is None:
        return [*lines, "No counted trial log was found.", ""]
    lines += [
        f"- **Resets:** over the {len(conduct['reset_errors_mm'])} scored trials the accepted placement lay a median "
        f"of {_mm(conduct['reset_error_median_mm'], 2)} from its target, at most "
        f"{_mm(conduct['reset_error_max_mm'], 2)} (placed to target, from the records).",
        f"- **Pace:** a median of {_num(conduct['trial_minutes_median'], 2)} minutes from one record's start to the "
        f"next one's in the same log ({conduct['n_trial_intervals']} intervals; void trials included, no pause left "
        "out).",
        f"- **Days:** {', '.join(conduct['trial_days']) or 'none'}. **People** (operator and scorer codes): "
        f"{', '.join(conduct['people']) or 'none'}.",
    ]
    for condition, settings in conduct["execution"]["by_condition"].items():
        shown = ", ".join(f"`{key}` = {value}" for key, value in sorted(settings.items()))
        own = conduct["execution"]["per_trial"].get(condition, {})
        seen = "; ".join(
            f"`{key}` {', '.join(str(value) for value in values)}" for key, values in sorted(own.items()) if values
        )
        lines.append(
            f"- **Execution settings of {_NAMES[condition]}** (the same in every scored trial, and every setting of "
            f"the protocol's `[execution]` section with the protocol's value): {shown}. "
            f"Belonging to the single trial and not held to anything: {seen or 'nothing recorded'}."
        )
    lines += _control_rate(conduct["execution"]["achieved_hz"])
    lines.append(_view_crept(conduct["view_crept"], len(conduct["reset_errors_mm"])))
    poses = conduct["start_pose_deg"]
    if poses is not None:
        shown = "; ".join(
            f"`{joint}` {entry['min']:.1f} to {entry['max']:.1f} (median {entry['median']:.1f})"
            for joint, entry in poses["joints"].items()
        )
        lines.append(
            f"- **Start pose of the arm** when the attempt began, over the {poses['n']} scored trials that record "
            f"it, in the records' own units: {shown}."
        )
    checkpoints = conduct["checkpoints"]
    if checkpoints["by_condition"]:
        named = "; ".join(
            f"{_NAMES[condition]} {', '.join(f'`{name}`' for name in names)}"
            for condition, names in sorted(checkpoints["by_condition"].items())
        )
        step = checkpoints["evaluation_step"]
        lines.append(
            f"- **Checkpoints** (from the records): {named}. Training step: "
            f"{'not named as a number' if step is None else f'{step:,}'}; every scored policy trial names the same one."
        )
    for key, found in conduct["code"]["versions"].items():
        lines.append(f"- **`versions.{key}`:** {', '.join(f'`{value}`' for value in found)}.")
    deviations = results["deviations"]
    if deviations is None:
        lines.append("- **Deviations:** no decision log was found.")
    elif deviations["n"] == 0:
        lines.append("- **Deviations:** the decision log marks none.")
    else:
        lines.append(f"- **Deviations:** the decision log marks {deviations['n']}:")
        lines += [f"  - {row['date']}: {row['decision']} Why: {row['why']}" for row in deviations["rows"]]
    return [*lines, ""]


def _view_crept(crept: Mapping[str, Any] | None, n_scored: int) -> str:
    """How far the front camera's picture had crept before the scored attempts, against the protocol's limit."""
    head = "- **The camera's creep before an attempt:**"
    if crept is None:
        return (
            f"{head} none of the {n_scored} scored trials records how far the front camera's picture stood from "
            "where its session solved the conversion (`execution.view_crept_mm`)."
        )
    limit = crept["limit_mm"]
    if limit is None:
        held = "The protocol file states no limit for it (`execution.view_crept_limit_mm`)."
    elif crept["n_over_limit"]:
        held = (
            f"**{crept['n_over_limit']} of them lie beyond the protocol's limit of {limit:g} mm** "
            "(`execution.view_crept_limit_mm`), beyond which a session makes no attempt."
        )
    else:
        held = (
            f"None lies beyond the protocol's limit of {limit:g} mm (`execution.view_crept_limit_mm`), beyond which "
            "a session makes no attempt."
        )
    return (
        f"{head} {crept['n']} of the {n_scored} scored trials record how far the front camera's picture stood from "
        "where its session solved the conversion, read just before the attempt (`execution.view_crept_mm`, from the "
        f"records): a median of {_mm(crept['median_mm'], 2)}, at most {_mm(crept['max_mm'], 2)}. These are as the "
        f"sessions read them, not divided by the scale factor. {held}"
    )


def _control_rate(rates: Mapping[str, Any]) -> list[str]:
    """The control rate the scored trials were asked to run at and the rate they achieved, kind by kind, and in
    plain words where the trials of a kind ran short of what was asked."""
    kinds = {"policy": "policy trials", "replay": "replay trials", "all": "all scored trials"}
    limit = _share(rates["shortfall_limit"])
    scored = {kind: rates[kind]["n"] + rates[kind]["n_without"] for kind in kinds}
    lines, warnings = [], []
    for kind, named in kinds.items():
        found = rates[kind]
        if scored[kind] == 0 or (kind == "all" and 0 in (scored["policy"], scored["replay"])):
            continue  # nothing of that kind was scored, or "all" would say one kind's line again
        asked = "no one rate" if found["requested"] is None else f"{found['requested']:g} Hz"
        without = ""
        if found["n_without"]:
            without = (
                f"; {found['n_without']} scored trial(s) have no rate (fewer than two steps were sent, or the record "
                "carries no `loop_s`)"
            )
        if found["n"] == 0:
            lines.append(f"  - {named}: asked {asked}; achieved: no rate{without}.")
            continue
        lines.append(
            f"  - {named}: asked {asked}; achieved a median of {_num(found['median'], 1)} Hz "
            f"({_num(found['min'], 1)} to {_num(found['max'], 1)} Hz over {found['n']} trials){without}."
        )
        if found["short_of_requested"] and kind != "all":
            below = (found["requested"] - found["median"]) / found["requested"]
            warnings.append(
                f"  - **The {named} ran more than {limit} below the rate they were asked for: a median of "
                f"{_num(found['median'], 1)} Hz against {found['requested']:g} Hz, {_num(100.0 * below, 1)} % below.**"
            )
    if not lines:
        return []
    head = (
        "- **Control rate**, asked and achieved, from the records (`control_hz`, `steps_sent`, `loop_s`). Achieved is "
        f"{ACHIEVED_RATE_MEANING}."
    )
    return [head, *lines, *warnings]


def _simulation(results: Results) -> list[str]:
    simulation = results["simulation"]
    lines = _section("What the paper quotes of the simulations of the design (simulated, not measured)")
    if simulation is None:
        return [*lines, "The simulation's tables (`analysis/simulation/`) were not found.", ""]
    if simulation["stale"]:
        lines += [
            f"No case: {simulation['why_no_case']}. **The simulation is stale: run it again on the present protocol "
            "file.** The `sim` macros stay empty.",
            "",
        ]
    else:
        lines += _simulated_blocks(results)
        lines += _simulated_case(results)
    return [*lines, *_simulated_rule(results)]


def _simulated_case(results: Results) -> list[str]:
    """The one case of the procedure simulation that was fixed before the data."""
    simulation = results["simulation"]
    if simulation["case"] is None:
        return [f"No quoted case: {simulation['why_no_case']}. Its `sim` macros stay empty.", ""]
    case = simulation["case"]
    paired = case["paired"]
    return [
        f"Fixed before any data: block {case['block']} as written, against a curve with radius "
        f"{case['r50_true_mm']:g} mm, width {case['width_true_mm']:g} mm and lapse {case['lapse_true']:g}; "
        f"{case['replications']} simulated blocks ({_files(simulation['files'])}, made from protocol file "
        f"`{simulation['protocol_digest_simulated']}`).",
        "",
        f"- As written: bias {_shift(case['bias_mm'])}, root-mean-square error {_mm(case['rmse_mm'], 2)}, mean "
        f"interval width {_mm(case['interval_width_mm'], 2)}, the interval held the truth in "
        f"{_share(case['coverage'])} of blocks.",
        f"- The same trials at fixed, equally spaced distances: error {_mm(case['rmse_fixed_mm'], 2)}, interval width "
        f"{_mm(case['interval_width_fixed_mm'], 2)}.",
        f"- Paired test, {paired['pairs']} pairs, success rates {paired['rate_low']:g} against "
        f"{paired['rate_high']:g}: the exact McNemar test tells them apart {_share(paired['power'])} of the time "
        "(computed exactly).",
        "",
    ]


def _simulated_rule(results: Results) -> list[str]:
    """The simulation of the comparison rule: how often it gave each verdict."""
    rule = results["simulation"]["comparison_rule"]
    if rule is None:
        return ["The simulation of the comparison rule (`analysis/simulation/comparison_rule.csv`) was not found.", ""]
    if rule["cases"] is None:
        said = f"`{rule['protocol_digest_simulated']}`" if rule["protocol_digest_simulated"] else "no protocol file"
        return [
            f"The simulation of the comparison rule names {said}, and the protocol file is now "
            f"`{results['simulation']['protocol_digest_now']}`. **It is stale: run it again on the present protocol "
            "file.** Its `simRule` macros stay empty.",
            "",
        ]
    rows = [
        [
            f"{case['width_mm']:g} mm",
            f"{case['difference_mm']:g} mm",
            _share(case["larger"]),
            _share(case["not_told_apart"]),
            _share(case["smaller"]),
        ]
        for case in rule["cases"]
    ]
    pairs = "the" if rule["pairs"] is None else f"{rule['pairs']}"
    radius = "" if rule["replay_r50_mm"] is None else f", the replay's radius {rule['replay_r50_mm']:g} mm"
    lines = [
        "The comparison rule of prediction 2, simulated on the clustered policy's block and the replay's as written "
        f"(`analysis/simulation/comparison_rule.csv`, made from protocol file `{rule['protocol_digest_simulated']}`): "
        f"the share of {pairs} simulated pairs of blocks with each verdict, both curves of one width{radius}.",
        "",
    ]
    header = ["Width of both curves", "True difference (policy minus replay)", "larger", "not told apart", "smaller"]
    return [*lines, *_table(header, rows)]


def _simulated_blocks(results: Results) -> list[str]:
    """The simulation's figures for each threshold block as written, by width of the true curve."""
    simulation = results["simulation"]
    names = {section["threshold_block"]: _NAMES[condition] for condition, section in results["conditions"].items()}
    rows = []
    for block, entry in (simulation["blocks"] or {}).items():
        trials = "none" if entry["n_trials"] is None else entry["n_trials"]
        row = [f"{block} ({names.get(block, 'no condition')})", trials]
        for name in SIMULATED_WIDTHS_MM:
            found = entry["by_width"][name]
            row.append("no row" if found is None else f"{found['rmse_mm']:.2f} / {found['interval_width_mm']:.2f}")
        falling = entry["open_below_falling_at_zero"]
        row.append("no row" if falling is None else _share(falling))
        rows.append(row)
    if not rows:
        return []
    widths = [f"{name.capitalize()} (width {width:g} mm)" for name, width in SIMULATED_WIDTHS_MM.items()]
    lines = [
        "Each threshold block as written. A cell is the error of the estimated radius (RMSE) / the mean width of its "
        f"{_level(results)} interval, in millimetres, for the middle one of the simulated truths with that width "
        "(the simulation report's Table 1). The last column is the share of simulated blocks whose interval is open "
        "below, over the truths whose curve is already falling at 0 mm (radius under two widths).",
        "",
    ]
    lines += _table(["Block", "Trials", *widths, "Open below when falling at 0 mm"], rows)
    if simulation["open_below_falling_at_zero"] is not None:
        lines += [
            f"Over the blocks together, the interval is open below in "
            f"{_share(simulation['open_below_falling_at_zero'])} of such simulated blocks.",
            "",
        ]
    return lines


def _not_computed(results: Results) -> list[str]:
    lines = _section("What the analysis could not compute")
    for entry in results["not_computed"]:
        what, why = entry["what"], entry["why"].rstrip(".")
        lines.append(f"- **{what[0].upper()}{what[1:]}:** {why}.")
    return [*lines, ""]


def _macros(values: Mapping[str, Mapping[str, str]]) -> list[str]:
    lines = _section("The paper's number macros")
    filled = sum(macro.name in values for macro in MACROS)
    total = len(MACROS) + len(NOT_PRODUCED)
    lines += [
        f"{filled} of {total} have a value; the others print `??` in the paper. A value is written as the paper "
        "prints it; the source is the comment beside the macro in `paper/generated/numbers.tex`.",
        "",
    ]
    rows = [
        [f"`\\{macro.name}`", f"`{values[macro.name]['value']}`" if macro.name in values else "??", macro.source]
        for macro in MACROS
    ]
    rows += [[f"`\\{name}`", "??", f"cannot be produced: {why}"] for name, why in NOT_PRODUCED.items()]
    return [*lines, *_table(["Macro", "Value", "From"], rows)]
