"""From the inputs to the results: the one structure that `results.json`, the report and the paper's macros share.

`build_results` applies `protocol/protocol.md`, section 10, to what `inputs.load_inputs` read, and returns plain data in
the shape `protocol/data-contracts.md` fixes for `analysis/report/results.json` (plus the additions that page lists).
The report's text and the paper's number macros are both written from this structure and from nothing else, so a
number in a figure, in the report and in the paper cannot come from three analyses.

Whatever the data do not support is None: a block that was not run, a fit with no trials, a file that is not there.
`not_computed` lists each such thing with the reason, for the report's last section.

Standard library and NumPy only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tolerance_radius import geometry, schema
from tolerance_radius.geometry import Grid
from tolerance_radius.psychometric import AnalysisGrids, ThresholdPosterior, grids_from_protocol
from tolerance_radius.report import bookkeeping, estimates
from tolerance_radius.report.inputs import (
    CountedTrial,
    Inputs,
    ReportError,
    block_conditions,
    is_paired,
    parse_time,
    split_checkpoint,
)
from tolerance_radius.schema import DemoStart, Point

SCHEMA_VERSION = 1
#: The two policies of the workspace test, in the order the contract names them.
CLUSTERED, SPREAD, REPLAY = "act_clustered", "act_spread", "replay"
REGIONS = ("trained_cells", "held_out_cells")
#: The fit every comparison and every prediction uses (protocol, section 10).
LOOKUP_FIT = f"{CLUSTERED} block_fit"

#: Prediction 1 of protocol section 9, in numbers: at least 5 successes in 6 trials at d = 0, at most 1 in 4 at the
#: far anchors, and a tolerance radius between 5 and 40 mm.
PREDICTION_CURVE = {
    "zero_trials": 6,
    "zero_min_successes": 5,
    "far_trials": 4,
    "far_max_successes": 1,
    "r50_min_mm": 5.0,
    "r50_max_mm": 40.0,
}

DRAFT_BANNER = (
    "DRAFT: written with --draft. It may hold trials that were not run under the frozen protocol. "
    "Nothing in it goes into the paper."
)


@dataclass(frozen=True, slots=True)
class Fitting:
    """How a set of trials is fitted: the protocol's grids, and the plateau check that goes with every fit."""

    grids: AnalysisGrids
    #: The same grids with the protocol's longer list of lapse values; None when the protocol states none.
    wide: AnalysisGrids | None
    #: How far the two fits' medians may lie apart before the plateau is in doubt; None when the protocol states none.
    limit_mm: float | None
    level: float


def fitting_from_protocol(protocol: Mapping[str, Any]) -> Fitting:
    """The fits the protocol's `[analysis]` section fixes: its grids, and for the plateau check (section 10) the same
    grids with `lapse_values_sensitivity` and the limit `sensitivity_limit_mm`, where the section states them."""
    analysis = protocol["analysis"]
    try:
        grids = grids_from_protocol(analysis)
        wide, limit_mm = None, None
        if analysis.get("lapse_values_sensitivity") is not None:
            wide = grids._replace(lapse_values=np.array([float(v) for v in analysis["lapse_values_sensitivity"]]))
            estimates.fit([], [], wide)  # refuses a list the model cannot hold, before any trial is fitted
            if analysis.get("sensitivity_limit_mm") is not None:
                limit_mm = float(analysis["sensitivity_limit_mm"])
                if not limit_mm >= 0.0:
                    raise ValueError("sensitivity_limit_mm is a distance, zero or more")
    except (TypeError, ValueError) as error:
        raise ReportError(f"the [analysis] section of the protocol file cannot be used: {error}") from error
    return Fitting(grids=grids, wide=wide, limit_mm=limit_mm, level=float(analysis["interval_level"]))


def build_results(inputs: Inputs) -> dict[str, Any]:
    """Everything the report computes, as plain data (see the module's notes)."""
    protocol = inputs.protocol
    fitting = fitting_from_protocol(protocol)
    level = fitting.level
    grid = Grid.from_protocol(protocol)
    have_logs = bool(inputs.logs)
    not_computed: list[dict[str, str]] = []
    notes: list[str] = list(inputs.execution_notes)

    scale_check = bookkeeping.scale_check(inputs.scale_runs)
    scale = estimates.scale_rule([run["factor"] for run in scale_check["runs"]] if scale_check else [])
    divisor = scale["divisor"]

    rows = trial_rows(inputs, grid, divisor)
    threshold_block, paired_block = _blocks_by_role(protocol)
    conditions: dict[str, Any] = {}
    block_fits: dict[str, ThresholdPosterior | None] = {}
    for condition in schema.CONDITIONS:
        own = [row for row in rows if row["condition"] == condition]
        n_void = sum(t.condition == condition and not t.record.scored for t in inputs.trials) if have_logs else None
        conditions[condition], block_fits[condition] = condition_section(
            condition, own, n_void, threshold_block.get(condition), fitting, list(protocol["blocks"])
        )

    comparisons = {
        "clustered_minus_replay": _difference(block_fits[CLUSTERED], block_fits[REPLAY], level),
        "spread_minus_clustered": _difference(block_fits[SPREAD], block_fits[CLUSTERED], level),
    }
    workspace = workspace_section(rows, paired_block, block_fits[CLUSTERED], grid, level, notes)
    by_eye = bookkeeping.by_eye_resets(inputs.by_eye, divisor)
    crept_limit = protocol.get("execution", {}).get("view_crept_limit_mm")
    conduct = (
        bookkeeping.conduct(inputs.trials, divisor, None if crept_limit is None else float(crept_limit))
        if have_logs
        else None
    )
    sensitivity = reset_sensitivity(block_fits[CLUSTERED], conduct, by_eye)
    notes += training_step_notes(conduct, protocol)
    # Whatever rests on a block fit that does not establish a radius carries the caveat beside it.
    by_result = caveats_by_result(conditions, comparisons, workspace, sensitivity)
    for name, difference in comparisons.items():
        if difference is not None:
            difference["caveats"] = by_result[f"the comparison {name}"]
    if workspace is not None and workspace["lookup"] is not None:
        workspace["lookup"]["caveats"] = by_result["the lookup prediction"]
    if sensitivity is not None:
        sensitivity["caveats"] = by_result["the reset sensitivity"]

    datasets: dict[str, Any] = {
        name: bookkeeping.dataset_summary(name, inputs.audits[name], inputs.placements[name], divisor)
        for name in schema.DATASETS
    }
    both = [error for summary in datasets.values() if summary for error in summary["placement_errors_mm"]]
    datasets["both"] = {
        "placement_error_median_mm": estimates.median(both)
        if all(datasets[name] for name in schema.DATASETS)
        else None,
        "placement_error_max_mm": max(both) if both and all(datasets[name] for name in schema.DATASETS) else None,
        "recording_days": bookkeeping.recording_days({name: datasets[name] for name in schema.DATASETS}),
    }

    pairs_planned = int(protocol["blocks"][paired_block]["pairs"]) if paired_block else None
    results: dict[str, Any] = {
        "banner": " ".join(part for part in (inputs.note, DRAFT_BANNER if inputs.draft else None) if part) or None,
        "schema_version": SCHEMA_VERSION,
        "draft": inputs.draft,
        "provenance": provenance(inputs),
        "distance_scale": scale,
        "bin_width_mm": estimates.BIN_WIDTH_MM,
        "interval_level": level,
        "conditions": conditions,
        "comparisons": comparisons,
        "workspace": workspace,
        "reset_sensitivity": sensitivity,
        "demonstrations": {
            name: [
                {
                    "episode_index": start.episode_index,
                    "cell": start.cell,
                    "x_mm": start.measured.x_mm,
                    "y_mm": start.measured.y_mm,
                    "trained": start.trained,
                }
                for start in inputs.audits[name] or ()
            ]
            for name in schema.DATASETS
        },
        "blocks": block_counts(inputs, protocol) if have_logs else None,
        "instrument": {
            "scale_check": scale_check,
            "repeatability": bookkeeping.repeatability(inputs.repeat_sets),
            "by_eye": by_eye,
        },
        "datasets": datasets,
        "conduct": conduct,
        "scene": bookkeeping.scene_at_start(inputs.scored, inputs.scenes) if have_logs else None,
        "second_scoring": (
            bookkeeping.second_scoring(inputs.second_scoring, inputs.logs, inputs.logs_digest) if have_logs else None
        ),
        "deviations": bookkeeping.deviations(inputs.deviations),
        "caveats": [sentence for sentences in by_result.values() for sentence in sentences],
        "simulation": bookkeeping.simulated_case(
            inputs.simulation,
            inputs.protocol_sha256,
            pairs_planned,
            {name: float(protocol["blocks"][name]["max_level_mm"]) for name in threshold_block.values()},
        ),
        "notes": notes,
    }
    results["predictions"] = predictions(results)
    results["not_computed"] = not_computed + missing_things(results, inputs)
    return _plain(results)


def dump_json(results: Mapping[str, Any]) -> str:
    """The text of `results.json`: sorted keys, one entry per line, and the banner on the first line.

    The banner says what the file is when it is not a counted report (a draft, a synthetic example) and is `null`
    when it is one. It comes first so that nobody opens such a file without seeing it.
    """
    rest = {key: value for key, value in results.items() if key != "banner"}
    head = json.dumps({"banner": results.get("banner")}, ensure_ascii=False)[:-1]
    body = json.dumps(rest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return head + "," + body[1:] + "\n"


# --------------------------------------------------------------------------------------------------------------------
# Provenance and counts
# --------------------------------------------------------------------------------------------------------------------


def provenance(inputs: Inputs) -> dict[str, Any]:
    """Where the results come from: the protocol file, the logs, the other files, and the commit the trials ran on."""
    commits = sorted({t.record.versions.get("project_commit", "") for t in inputs.trials} - {""})
    return {
        "protocol_version": str(inputs.protocol["protocol"]["version"]),
        "protocol_status": str(inputs.protocol["protocol"].get("status", "")),
        "protocol_sha256": inputs.protocol_sha256,
        "protocol_file": inputs.protocol_file,
        "project_commit": commits[0] if len(commits) == 1 else None,
        "project_commits": commits,
        "logs": list(inputs.logs),
        "pilot_logs_not_read": list(inputs.pilot_logs),
        "n_records": len(inputs.trials),
        "n_scored": len(inputs.scored),
        "n_void": len(inputs.trials) - len(inputs.scored),
        "audits": dict(inputs.audit_files),
        "scale_runs": [run.file for run in inputs.scale_runs],
        "repeat_sets": [log.file for log in inputs.repeat_sets],
        "by_eye": list(inputs.by_eye_files),
        "placement_logs": dict(inputs.placement_files),
        "decisions": inputs.decisions_file,
        "scene_audits": [audit.file for audit in inputs.scenes.values() if audit is not None],
        "second_scoring": inputs.second_scoring.file if inputs.second_scoring is not None else None,
        "note": inputs.note,
    }


def block_counts(inputs: Inputs, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """For every block of the protocol: what kind it is, how many trials it plans, and how many were scored and void."""
    counts: dict[str, Any] = {}
    for name, entry in protocol["blocks"].items():
        own = [trial for trial in inputs.trials if trial.block == name]
        by_condition = {
            condition: {
                "n_scored": sum(t.condition == condition and t.record.scored for t in own),
                "n_void": sum(t.condition == condition and not t.record.scored for t in own),
            }
            for condition in block_conditions(entry)
        }
        counts[name] = {
            "kind": "paired" if is_paired(entry) else "threshold",
            "conditions": block_conditions(entry),
            "planned": planned_trials(entry),
            "n_scored": sum(t.record.scored for t in own),
            "n_void": sum(not t.record.scored for t in own),
            "by_condition": by_condition,
        }
    return counts


def planned_trials(entry: Mapping[str, Any]) -> int:
    """How many scored trials a block of the protocol asks for."""
    if is_paired(entry):
        return int(entry["pairs"]) * len(block_conditions(entry))
    in_staircases = sum(int(staircase["trials"]) for staircase in entry.get("staircases", []))
    return int(entry.get("anchors_zero", 0)) + int(entry.get("anchors_far", 0)) + in_staircases


def _blocks_by_role(protocol: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    """The threshold block of each condition, and the paired block (None if the protocol has none)."""
    threshold: dict[str, str] = {}
    paired: list[str] = []
    for name, entry in protocol["blocks"].items():
        if is_paired(entry):
            paired.append(name)
            if sorted(block_conditions(entry)) != sorted((CLUSTERED, SPREAD)):
                raise ReportError(
                    f"block {name} pairs {' and '.join(block_conditions(entry))}, and the report knows the workspace "
                    f"test between {CLUSTERED} and {SPREAD} only"
                )
        else:
            if entry["condition"] in threshold:
                raise ReportError(
                    f"blocks {threshold[entry['condition']]} and {name} are both threshold blocks of one condition"
                )
            threshold[entry["condition"]] = name
    if len(paired) > 1:
        raise ReportError(
            f"the protocol has more than one paired block ({', '.join(paired)}), and the report knows one"
        )
    return threshold, (paired[0] if paired else None)


# --------------------------------------------------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------------------------------------------------


def part_of(trial: CountedTrial) -> str:
    """Which part of its block a trial belongs to: `zero`, `far`, a staircase's id, or `pair`."""
    planned = trial.record.planned
    if planned.pair_id is not None:
        return "pair"
    if planned.staircase_id is not None:
        return planned.staircase_id
    return "zero" if planned.level_mm == 0 else "far"


def trial_rows(inputs: Inputs, grid: Grid, divisor: float) -> list[dict[str, Any]]:
    """One row for every scored trial, in the order of the logs: the contract's fields and a few more.

    `d_mm` and `placement_error_mm` are distances and are divided by `divisor`. Positions and the requested level
    are left as they were measured and asked for.
    """
    hulls: dict[str, list[DemoStart]] = {}
    rows = []
    for trial in inputs.scored:
        record, planned = trial.record, trial.record.planned
        episode, distance = inputs.nearest[trial.trial_id]
        in_hull = None
        if planned.condition != REPLAY:
            if planned.dataset not in hulls:
                hulls[planned.dataset] = geometry.trained_demos(inputs.audits[planned.dataset], planned.dataset)
            in_hull = geometry.in_convex_hull(record.placed, hulls[planned.dataset])
        duration = (parse_time(record.ts_end) - parse_time(record.ts_start)).total_seconds()
        rows.append(
            {
                "trial_id": planned.trial_id,
                "block": planned.block,
                "condition": planned.condition,
                "part": part_of(trial),
                "d_mm": estimates.stable(distance / divisor),
                "success": record.success,
                "subgoal": record.subgoal,
                "failure": record.failure,
                "x_mm": record.placed.x_mm,
                "y_mm": record.placed.y_mm,
                "level_mm": planned.level_mm,
                "pair_id": planned.pair_id,
                "nearest_demo_episode": episode,
                "replay_episode": planned.replay_episode,
                "cell": grid.cell_of(record.placed),
                "region": region_of(record.placed, grid),
                "in_hull": in_hull,
                "checkpoint": record.checkpoint,
                "placement_error_mm": estimates.stable(record.placement_error_mm / divisor),
                "duration_s": duration,
                "steps_sent": record.execution.get("steps_sent"),
                "loop_s": bookkeeping.loop_seconds(record.execution),
                "achieved_hz": bookkeeping.trial_rate_hz(trial),
            }
        )
    return rows


def fitted(
    rows: Sequence[Mapping[str, Any]], blocks: Sequence[str], fitting: Fitting
) -> tuple[ThresholdPosterior, dict[str, Any]]:
    """The fit of a set of trials, and what is reported of it.

    Beside the radius and the curve (`estimates.fit_summary`): what these trials say without a model
    (`estimates.trial_counts`), and the plateau check, the same trials fitted with the longer list of lapse values
    (`estimates.plateau_check`; None when the protocol states no such list).
    """
    distances = [row["d_mm"] for row in rows]
    outcomes = [row["success"] for row in rows]
    posterior = estimates.fit(distances, outcomes, fitting.grids)
    summary = estimates.fit_summary(posterior, blocks, level=fitting.level, largest_tested_mm=max(distances))
    summary.update(estimates.trial_counts(distances, outcomes, [row["part"] == "zero" for row in rows]))
    summary["sensitivity"] = None
    if fitting.wide is not None:
        wide = estimates.fit(distances, outcomes, fitting.wide)
        summary["sensitivity"] = estimates.plateau_check(
            summary["r50_median_mm"], wide, fitting.limit_mm, fitting.level
        )
    return posterior, summary


def condition_section(
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    n_void: int | None,
    threshold_block: str | None,
    fitting: Fitting,
    block_order: Sequence[str],
) -> tuple[dict[str, Any], ThresholdPosterior | None]:
    """One condition's part of the results, and the posterior of its block fit (None when the block was not run).

    The block fit uses the trials of the condition's own threshold block. The pooled fit uses every scored trial of
    a policy, whatever its block; a replay has none. A fit without a trial is None.
    """
    distances = [row["d_mm"] for row in rows]
    outcomes = [row["success"] for row in rows]
    max_success, min_failure = estimates.extremes(distances, outcomes)
    in_block = [row for row in rows if row["block"] == threshold_block]

    block_posterior = block_fit = None
    if in_block:
        block_posterior, block_fit = fitted(in_block, [threshold_block], fitting)
    pooled_fit = None
    if condition != REPLAY and rows:
        blocks = sorted({row["block"] for row in rows}, key=list(block_order).index)
        if block_fit is not None and len(in_block) == len(rows):
            pooled_fit = {**block_fit, "blocks": blocks}
        else:
            pooled_fit = fitted(rows, blocks, fitting)[1]

    anchors = {}
    for part in ("zero", "far"):
        found = [row for row in in_block if row["part"] == part]
        anchors[part] = {"n": len(found), "k": sum(row["success"] for row in found)}
    section = {
        "trials": [{key: value for key, value in row.items() if key != "condition"} for row in rows],
        "n_scored": len(rows),
        "n_void": n_void,
        "threshold_block": threshold_block,
        "anchors": anchors,
        "bins": estimates.distance_bins(distances, outcomes, level=fitting.level),
        "max_success_d_mm": max_success,
        "min_failure_d_mm": min_failure,
        "block_fit": block_fit,
        "pooled_fit": pooled_fit,
        "replay_episodes": sorted({row["replay_episode"] for row in rows}) if condition == REPLAY else None,
        "descriptive": descriptive(condition, rows, fitting.level),
    }
    return section, block_posterior


def descriptive(condition: str, rows: Sequence[Mapping[str, Any]], level: float = 0.95) -> dict[str, Any]:
    """What protocol section 10 calls descriptive only: success inside and outside the convex hull of the
    demonstrations, success by whether the placement lies in a trained cell or a held-out cell at each distance,
    failure labels by distance, and durations."""
    hull = None
    if condition != REPLAY:
        hull = {}
        for name, inside in (("inside", True), ("outside", False)):
            found = [row for row in rows if row["in_hull"] is inside]
            hull[name] = {"n": len(found), "k": sum(row["success"] for row in found)}
    by_region = {}
    for region in REGIONS:
        found = [row for row in rows if row["region"] == region]
        by_region[region] = {
            "n": len(found),
            "k": sum(row["success"] for row in found),
            "bins": estimates.distance_bins(
                [row["d_mm"] for row in found], [row["success"] for row in found], level=level
            ),
        }
    labels: dict[str, list[float]] = {}
    for row in rows:
        if not row["success"]:
            labels.setdefault(row["failure"], []).append(row["d_mm"])
    steps = [row["steps_sent"] for row in rows if isinstance(row["steps_sent"], (int, float))]
    return {
        "hull": hull,
        "by_region": by_region,
        "n_off_grid": sum(row["region"] is None for row in rows),
        "failures": [
            {
                "label": label,
                "n": len(found),
                "d_min_mm": min(found),
                "d_median_mm": estimates.median(found),
                "d_max_mm": max(found),
            }
            for label, found in sorted(labels.items())
        ],
        "duration_median_s": estimates.median([row["duration_s"] for row in rows]),
        "steps_sent_median": estimates.median(steps),
    }


def _difference(
    first: ThresholdPosterior | None, second: ThresholdPosterior | None, level: float
) -> dict[str, Any] | None:
    if first is None or second is None:
        return None
    return estimates.radius_difference(first, second, level)


# --------------------------------------------------------------------------------------------------------------------
# The workspace test
# --------------------------------------------------------------------------------------------------------------------


def region_of(point: Point, grid: Grid) -> str | None:
    """`trained_cells` or `held_out_cells`, by the grid cell a measured position falls in; None off the grid."""
    cell = grid.cell_of(point)
    if cell is None:
        return None
    return REGIONS[0] if grid.is_trained(cell) else REGIONS[1]


def workspace_section(
    rows: Sequence[Mapping[str, Any]],
    paired_block: str | None,
    lookup_fit: ThresholdPosterior | None,
    grid: Grid,
    level: float,
    notes: list[str],
) -> dict[str, Any] | None:
    """The workspace test: the paired table, the exact McNemar test, each policy's success, the lookup prediction.

    A pair is a placement at which both policies have a scored trial. Its position is the mean of the two measured
    placements, and its region is the grid cell that position falls in: where the cube was, not where the plan
    asked for it. The paired test is made over all pairs together. The lookup prediction evaluates the clustered
    policy's block fit, which has not seen these trials, at each trial's own distance to its own policy's nearest
    demonstration. The counts are also given by region, and by the pair of training runs the two trials of a pair
    were run with. None when the block has no scored trial.
    """
    own = [row for row in rows if row["block"] == paired_block]
    if not own:
        return None
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in own:
        by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
    complete = {pair_id: found for pair_id, found in by_pair.items() if len(found) == 2}
    unpaired = sorted(row["trial_id"] for found in by_pair.values() if len(found) == 1 for row in found.values())
    if unpaired:
        notes.append(
            f"{len(unpaired)} scored trial(s) of block {paired_block} have no scored partner at the same placement "
            f"({', '.join(unpaired)}): they are left out of the paired table and of the lookup prediction, and stay "
            f"in the counts of trials and in the pooled fits"
        )

    pairs = []
    for pair_id in sorted(complete):
        found = complete[pair_id]
        x = (found[CLUSTERED]["x_mm"] + found[SPREAD]["x_mm"]) / 2.0
        y = (found[CLUSTERED]["y_mm"] + found[SPREAD]["y_mm"]) / 2.0
        entry: dict[str, Any] = {"pair_id": pair_id, "x_mm": x, "y_mm": y, "region": region_of(Point(x, y), grid)}
        for policy in (CLUSTERED, SPREAD):
            row = found[policy]
            if region_of(Point(row["x_mm"], row["y_mm"]), grid) != entry["region"]:
                notes.append(
                    f"at placement {pair_id} the {policy} trial ({row['trial_id']}) was placed in another kind of "
                    f"cell than the pair's own position: the pair counts as {entry['region']}"
                )
            entry[policy] = {
                "trial_id": row["trial_id"],
                "success": row["success"],
                "d_mm": row["d_mm"],
                "p_lookup": float(lookup_fit.predict(row["d_mm"])) if lookup_fit is not None else None,
                "x_mm": row["x_mm"],
                "y_mm": row["y_mm"],
                "checkpoint": row["checkpoint"],
            }
        pairs.append(entry)

    table = estimates.paired_table([p[SPREAD]["success"] for p in pairs], [p[CLUSTERED]["success"] for p in pairs])
    lookup = None
    if lookup_fit is not None and pairs:
        lookup = {"fit": LOOKUP_FIT}
        for policy in (CLUSTERED, SPREAD):
            observed = sum(pair[policy]["success"] for pair in pairs)
            lookup[policy] = estimates.lookup_prediction(
                lookup_fit, [pair[policy]["d_mm"] for pair in pairs], observed, level
            )

    def counts(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """A group of pairs: how many, and each policy's successes with what the lookup prediction expects there:
        the expected count and the interval of the predicted count, as for all pairs together."""
        entry: dict[str, Any] = {"n_pairs": len(group)}
        for policy in (CLUSTERED, SPREAD):
            observed = sum(pair[policy]["success"] for pair in group)
            found = estimates.proportion(observed, len(group), level)
            expected = low = high = None
            if lookup is not None and group:
                predicted = estimates.lookup_prediction(
                    lookup_fit, [pair[policy]["d_mm"] for pair in group], observed, level
                )
                expected, low, high = predicted["expected"], predicted["lo"], predicted["hi"]
            entry[policy] = {**found, "lookup_expected": expected, "lookup_lo": low, "lookup_hi": high}
        return entry

    by_region = {region: counts([pair for pair in pairs if pair["region"] == region]) for region in REGIONS}
    runs: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for pair in pairs:
        named = tuple(_run_of(pair[policy]["checkpoint"]) or "" for policy in (CLUSTERED, SPREAD))
        runs.setdefault(named, []).append(pair)
    by_run = [
        {"runs": {CLUSTERED: named[0] or None, SPREAD: named[1] or None}, **counts(group)}
        for named, group in sorted(runs.items())
    ]
    return {
        "block": paired_block,
        "n_pairs": len(pairs),
        "n_unpaired": len(unpaired),
        "pairs": pairs,
        "table": {
            "both": table["both"],
            "neither": table["neither"],
            "only_spread": table["only_first"],
            "only_clustered": table["only_second"],
        },
        "mcnemar_p": estimates.paired_p_value(table),
        "success": {
            policy: estimates.proportion(sum(pair[policy]["success"] for pair in pairs), len(pairs), level)
            for policy in (CLUSTERED, SPREAD)
        },
        "distances": {
            policy: {
                "median_mm": estimates.median([pair[policy]["d_mm"] for pair in pairs]),
                "max_mm": max((pair[policy]["d_mm"] for pair in pairs), default=None),
            }
            for policy in (CLUSTERED, SPREAD)
        },
        "lookup": lookup,
        "by_region": by_region,
        "by_run": by_run,
    }


def _run_of(checkpoint: str | None) -> str | None:
    """The training run a trial's checkpoint names; None when the record names a step alone."""
    return None if checkpoint is None else split_checkpoint(checkpoint)[0]


def training_step_notes(conduct: Mapping[str, Any] | None, protocol: Mapping[str, Any]) -> list[str]:
    """A sentence for the report when the policies were evaluated at another training step than the protocol plans.

    The protocol declares a fallback in advance (section 5), so another step is not refused: it is said.
    """
    training = protocol.get("training", {})
    planned = training.get("evaluation_checkpoint")
    step = conduct["checkpoints"]["evaluation_step"] if conduct else None
    if step is None or planned is None or step == int(planned):
        return []
    least = training.get("fallback_min_checkpoint")
    fallback = "" if least is None else f"; its fallback is the largest common saved step, not below {int(least):,}"
    return [
        f"the policies were evaluated at training step {step:,}, and the protocol plans step {int(planned):,}{fallback}"
    ]


def reset_sensitivity(
    fit: ThresholdPosterior | None, conduct: Mapping[str, Any] | None, by_eye: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The success the clustered policy's block fit gives at a demonstration's own start when the reset carries this
    study's placement errors, and when it carries the by-eye errors of block V1. A calculation, not a measurement.
    None when the clustered policy has no block fit."""
    if fit is None:
        return None
    placed = conduct["reset_errors_mm"] if conduct else []
    by_eye_errors = by_eye["errors_mm"] if by_eye else []
    return {
        "fit": LOOKUP_FIT,
        "placed": estimates.mean_success_at(fit, placed),
        "by_eye": estimates.mean_success_at(fit, by_eye_errors),
        "n_placed": len(placed),
        "n_by_eye": len(by_eye_errors),
    }


#: The results that rest on a block fit, by the name a caveat gives each, and the conditions whose block fits it uses
#: (protocol section 10: a comparison, the lookup prediction, the reset sensitivity, prediction 1).
RESTS_ON = {
    "the comparison clustered_minus_replay": (CLUSTERED, REPLAY),
    "the comparison spread_minus_clustered": (SPREAD, CLUSTERED),
    "the lookup prediction": (CLUSTERED,),
    "the reset sensitivity": (CLUSTERED,),
    "prediction 1": (CLUSTERED,),
}


def unsettled(fit: Mapping[str, Any] | None) -> list[str]:
    """Why a fit does not establish a radius, in words; an empty list when it does, or when there is no fit.

    It does not when its interval is open on a side, or when its plateau is in doubt (protocol section 10).
    """
    if fit is None:
        return []
    found = []
    if fit["open_below"]:
        found.append("its interval is open below")
    if fit["open_above"]:
        found.append("its interval is open above, so its upper end only says where the grid of radii stops")
    if fit["sensitivity"] is not None and fit["sensitivity"]["plateau_in_doubt"]:
        found.append("its plateau is in doubt")
    return found


def caveats_by_result(
    conditions: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    workspace: Mapping[str, Any] | None,
    sensitivity: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    """For each result of `RESTS_ON`: the sentences that say it rests on a block fit that does not establish a radius.

    Such a result is computed by its rule all the same, and is reported with these sentences beside it, never
    without (protocol section 10). A result that was not computed has none, and neither has one whose block fits
    all establish their radius. Prediction 1 counts as computed whenever the clustered policy has a block fit.
    """
    computed = {
        "the comparison clustered_minus_replay": comparisons["clustered_minus_replay"] is not None,
        "the comparison spread_minus_clustered": comparisons["spread_minus_clustered"] is not None,
        "the lookup prediction": workspace is not None and workspace["lookup"] is not None,
        "the reset sensitivity": sensitivity is not None,
        "prediction 1": conditions[CLUSTERED]["block_fit"] is not None,
    }
    found: dict[str, list[str]] = {}
    for what, fits in RESTS_ON.items():
        found[what] = []
        for condition in fits if computed[what] else ():
            reasons = unsettled(conditions[condition]["block_fit"])
            if reasons:
                found[what].append(f"{what} uses the block fit of {condition}, and {' and '.join(reasons)}")
    return found


def caveats(
    conditions: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    workspace: Mapping[str, Any] | None,
    sensitivity: Mapping[str, Any] | None,
) -> list[str]:
    """Every sentence of `caveats_by_result`, in the order of `RESTS_ON`: what rests on a block fit that does not
    establish a radius, so that nobody reads such a result without knowing."""
    by_result = caveats_by_result(conditions, comparisons, workspace, sensitivity)
    return [sentence for sentences in by_result.values() for sentence in sentences]


# --------------------------------------------------------------------------------------------------------------------
# The five predictions of protocol section 9
# --------------------------------------------------------------------------------------------------------------------


def predictions(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What became of each prediction of protocol section 9: `held` is True, False, or None where it is not judged.

    Each entry has the same six fields. `numbers` holds what the judgement rests on; the report words them. `why`
    says why a prediction was not judged, or by which rule it was. `caveats` holds the sentences of
    `caveats_by_result` for what the prediction rests on: the block fit of the clustered policy (1), the comparison
    with the replay (2), the lookup prediction (4), the comparison of the two policies (5).
    """

    def entry(number: int, name: str, **numbers: Any) -> dict[str, Any]:
        return {"id": number, "name": name, "held": None, "why": None, "caveats": [], "numbers": numbers}

    def carried(result: Mapping[str, Any] | None) -> list[str]:
        """The caveats a computed result carries; none for a result that was not computed."""
        return list(result["caveats"]) if result is not None else []

    by_result = caveats_by_result(
        results["conditions"], results["comparisons"], results["workspace"], results["reset_sensitivity"]
    )
    clustered = results["conditions"][CLUSTERED]
    fit = clustered["block_fit"]
    rule = PREDICTION_CURVE
    zero, far = clustered["anchors"]["zero"], clustered["anchors"]["far"]
    radius = fit["r50_median_mm"] if fit else None
    curve = entry(1, "the curve", zero=zero, far=far, r50_median_mm=radius, rule=dict(rule))
    curve["caveats"] = by_result["prediction 1"]
    if fit is None:
        curve["why"] = "the clustered policy's threshold block has no scored trial"
    elif zero["n"] != rule["zero_trials"] or far["n"] != rule["far_trials"]:
        curve["why"] = (
            f"the prediction is stated for {rule['zero_trials']} trials at d = 0 and {rule['far_trials']} far anchors, "
            f"and the block has {zero['n']} and {far['n']}"
        )
    else:
        curve["held"] = bool(
            zero["k"] >= rule["zero_min_successes"]
            and far["k"] <= rule["far_max_successes"]
            and rule["r50_min_mm"] <= radius <= rule["r50_max_mm"]
        )
        curve["why"] = "the radius is the posterior median of the block fit"

    difference = results["comparisons"]["clustered_minus_replay"]
    hardware = entry(2, "hardware", difference=difference)
    hardware["caveats"] = carried(difference)
    if difference is None:
        hardware["why"] = "it needs the block fits of the clustered policy and of the replay"
    else:
        hardware["held"] = difference["verdict"] == estimates.VERDICT_LARGER

    workspace = results["workspace"]
    layout, lookup = entry(3, "layout"), entry(4, "lookup")
    if workspace is None or workspace["n_pairs"] == 0:
        layout["why"] = lookup["why"] = "the workspace test has no placement tested with both policies"
    else:
        k_spread, k_clustered = workspace["success"][SPREAD]["k"], workspace["success"][CLUSTERED]["k"]
        outcome = estimates.layout_outcome(workspace["mcnemar_p"], k_spread, k_clustered)
        layout["held"] = outcome == estimates.LAYOUT_SPREAD
        layout["why"] = (
            f"the rule of prediction 3: the exact paired test over all pairs gives p below "
            f"{estimates.LAYOUT_P_BELOW:g} with the spread policy ahead"
        )
        layout["numbers"] = {
            "k_spread": k_spread,
            "k_clustered": k_clustered,
            "n_pairs": workspace["n_pairs"],
            "mcnemar_p": workspace["mcnemar_p"],
            "p_below": estimates.LAYOUT_P_BELOW,
            "outcome": outcome,
            "by_region": {
                region: {
                    "k_spread": workspace["by_region"][region][SPREAD]["k"],
                    "k_clustered": workspace["by_region"][region][CLUSTERED]["k"],
                    "n_pairs": workspace["by_region"][region]["n_pairs"],
                }
                for region in REGIONS
            },
        }
        if workspace["lookup"] is None:
            lookup["why"] = "it needs the block fit of the clustered policy"
        else:
            verdicts = {policy: workspace["lookup"][policy]["verdict"] for policy in (CLUSTERED, SPREAD)}
            lookup["held"] = all(value == estimates.LOOKUP_MATCHES for value in verdicts.values())
            lookup["numbers"] = {"verdicts": verdicts}
            lookup["caveats"] = carried(workspace["lookup"])

    estimate = entry(5, "layout and radius", difference=results["comparisons"]["spread_minus_clustered"])
    estimate["caveats"] = carried(results["comparisons"]["spread_minus_clustered"])
    estimate["why"] = "no prediction was made"
    return [curve, hardware, layout, lookup, estimate]


# --------------------------------------------------------------------------------------------------------------------
# What could not be computed
# --------------------------------------------------------------------------------------------------------------------


def missing_things(results: Mapping[str, Any], inputs: Inputs) -> list[dict[str, str]]:
    """Everything the analysis could not compute from what it was given, each with the reason."""
    missing: list[dict[str, str]] = []

    def add(what: str, why: str) -> None:
        missing.append({"what": what, "why": why})

    if not inputs.logs:
        add("everything that is counted from trials", "no counted trial log was found")
    for condition, section in results["conditions"].items():
        if inputs.logs and section["block_fit"] is None:
            add(f"the block fit of {condition}", f"block {section['threshold_block']} has no scored trial")
    analysis = inputs.protocol["analysis"]
    if analysis.get("lapse_values_sensitivity") is None:
        add(
            "the plateau check (every fit made a second time with a longer list of lapse values)",
            f"{inputs.protocol_file} states no analysis.lapse_values_sensitivity",
        )
    elif analysis.get("sensitivity_limit_mm") is None:
        add(
            "whether a plateau is in doubt",
            f"{inputs.protocol_file} states no analysis.sensitivity_limit_mm to hold the two fits' medians against",
        )
    for name, difference in results["comparisons"].items():
        if inputs.logs and difference is None:
            add(f"the comparison {name}", "it needs both block fits")
    workspace = results["workspace"]
    if inputs.logs and workspace is None:
        add("the workspace test", "its block has no scored trial")
    elif workspace is not None and workspace["lookup"] is None:
        add("the lookup prediction", "it needs the block fit of the clustered policy and at least one pair")
    sensitivity = results["reset_sensitivity"]
    if inputs.logs and sensitivity is None:
        add("the reset sensitivity", "it needs the block fit of the clustered policy")
    elif sensitivity is not None and sensitivity["by_eye"] is None:
        add("the reset sensitivity with by-eye errors", "no by-eye file (data/instrument/byeye_*.csv) was found")
    instrument = results["instrument"]
    if instrument["scale_check"] is None:
        add("the scale check and the scale correction", "no scale run (data/instrument/ruler_*.csv) was found")
    if instrument["repeatability"] is None:
        add("the repeatability of a placement", "no repeatability file (data/instrument/repeat_*.csv) was found")
    if instrument["by_eye"] is None:
        add("the by-eye reset errors", "no by-eye file (data/instrument/byeye_*.csv) was found")
    for name in schema.DATASETS:
        summary = results["datasets"][name]
        if summary is None:
            add(f"the {name} set's audit figures", f"there is no audit (data/audit/{name}.json)")
        elif summary["recording"] is None:
            add(
                f"the {name} set's re-recorded episodes and recording days",
                f"there is no placement log of the {name} set",
            )
        elif not summary["recording"]["matches_audit"]:
            add(
                f"the {name} set's re-recorded episodes and recording days",
                "the placement log's episodes are not the audited ones",
            )
    without_scene = [log for log in inputs.logs if inputs.scenes.get(log) is None]
    start = "the scene at each trial's start (the basket's place and the wrist camera's)"
    if results["scene"] is None and inputs.logs:
        if len(without_scene) == len(inputs.logs):
            why = (
                f"there is no scene audit of {' or '.join(inputs.logs)} (data/audit/scene_<log>.json): "
                "scripts/rig/scene_audit.py writes it from the trials' own videos"
            )
        else:
            why = "the scene audit holds none of the scored trials"
        add(start, why)
    elif without_scene:
        add(
            f"the scene at the start of the trials of {' and '.join(without_scene)}",
            "there is no scene audit of that log (data/audit/scene_<log>.json)",
        )
    second = results["second_scoring"]
    if inputs.logs and second is None:
        add(
            "the second scoring from video",
            "no agreement file (data/second_scoring/main/agreement.json) was found: scripts/second_scoring.py "
            "compare writes it once every drawn trial has its second score",
        )
    elif second is not None and second["why_not"] is not None:
        add("the second scoring from video", second["why_not"])
    if results["deviations"] is None:
        add("the count of deviations", "there is no decision log (docs/decisions.md)")
    simulation = results["simulation"]
    if simulation is None:
        add("the simulated case", "the simulation's tables (analysis/simulation/) were not found")
    elif simulation["case"] is None:
        stale = " The simulation is stale: run it again on the present protocol file." if simulation["stale"] else ""
        add("the simulated case", simulation["why_no_case"] + "." + stale)
    if simulation is not None:
        rule = simulation["comparison_rule"]
        if rule is None:
            add(
                "the simulation of the comparison rule",
                "its table (analysis/simulation/comparison_rule.csv) was not found",
            )
        elif rule["cases"] is None:
            said = f"`{rule['protocol_digest_simulated']}`" if rule["protocol_digest_simulated"] else "no protocol file"
            add(
                "the simulation of the comparison rule",
                f"its report names {said}, and the protocol file is now `{simulation['protocol_digest_now']}`: it is "
                "stale, run it again on the present protocol file",
            )
    conduct = results["conduct"]
    if conduct is not None:
        if conduct["checkpoints"]["by_condition"] and conduct["checkpoints"]["evaluation_step"] is None:
            named = "; ".join(
                f"{condition}: {', '.join(names)}"
                for condition, names in sorted(conduct["checkpoints"]["by_condition"].items())
            )
            add(
                "the training step of the evaluated checkpoints",
                f"the records do not name it as a number ({named})",
            )
        if conduct["code"]["lerobot_commit"] is None:
            add("the LeRobot commit", "the records do not all name one commit under versions.lerobot")
        if conduct["view_crept"] is None:
            add(
                "how far the front camera's picture had crept before an attempt",
                f"none of the {len(inputs.scored)} scored trials records execution.view_crept_mm, the reading a "
                "session takes just before an attempt",
            )
    for kind in ("policy", "replay") if conduct is not None else ():
        if conduct["execution"]["achieved_hz"][kind]["n"] > 0:
            continue
        own = [t for t in inputs.scored if t.condition in bookkeeping.RATE_GROUPS[kind]]
        unrecorded = sum(bookkeeping.loop_seconds(t.record.execution) is None for t in own)
        if not own:
            if kind != "policy":  # the paper's macro is the policy trials': say why it is empty
                continue
            why = "there is no scored policy trial"
        elif unrecorded == len(own):
            why = (
                f"none of the {len(own)} scored {kind} trials records loop_s, the time from the first step sent "
                "to the last: their records were written before the trial runner kept it"
            )
        else:
            why = (
                f"none of the {len(own)} scored {kind} trials has a rate: {unrecorded} record no loop_s, and in "
                f"the other {len(own) - unrecorded} fewer than two steps were sent or loop_s is not above zero"
            )
        add(f"the achieved control rate of the {kind} trials", why)
    add(
        "the largest disagreement of a session's camera solve with the stored one",
        "the tools print it and no committed record holds it",
    )
    add(
        "torn frames and mispaced episodes of the new datasets",
        "the dataset checks print their findings and no committed record holds them",
    )
    add(
        "the number of practice episodes",
        "no record says which episodes were practice; the placement logs only show lines that were later recorded over",
    )
    return missing


def _plain(value: Any) -> Any:
    """The same data as JSON can hold it: NumPy numbers turned into Python's, and every number that is not whole
    kept to the results' significant digits (`estimates.stable`)."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return estimates.stable(value) if isinstance(value, float) else value
