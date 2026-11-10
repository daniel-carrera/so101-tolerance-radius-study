"""The report's bookkeeping: the instrument check, the two datasets, how the trials were run, the scene at each
trial's start, the second scoring, the simulated case.

These are the numbers around the experiment's results: what the scale check measured (block V0), how far resets by
eye landed (block V1), where the demonstrations really started and how often an episode was recorded again, why
trials were void, on how many days and by how many people the trials were run, with which settings and which code,
where the basket and the wrist camera stood when each trial began, in how many of the trials scored again from video
the two scores agree, and the one case of the procedure simulation that the paper quotes.

Each function returns plain data for `results.json`. What a record does not hold is returned as None, never as a
guess: the caller lists it among the things the analysis could not compute.

Standard library and NumPy only.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from tolerance_radius.instrument import check_against_scale, scatter_mm
from tolerance_radius.report.estimates import median, percentile, stable
from tolerance_radius.report.inputs import (
    HANDOFF_USED_KEY,
    ByEyeReset,
    CountedTrial,
    Deviation,
    PositionLog,
    RecordedPlacement,
    SceneAudit,
    SecondScoring,
    SimulationTables,
    execution_settings,
    parse_time,
    split_checkpoint,
)
from tolerance_radius.schema import DemoStart, Point

#: The true distance between two readings of a scale run (protocol, section 7: "in 10 mm steps over 50 mm").
SCALE_STEP_MM = 10.0
#: The pass line of the scale check (protocol, section 7): mean absolute step error, and the worst single step.
SCALE_MEAN_LIMIT_MM, SCALE_MAX_LIMIT_MM = 1.5, 3.0
#: The percentile of the by-eye errors that the paper quotes beside the median.
BY_EYE_PERCENTILE = 95.0

#: The case of the procedure simulation that the paper quotes, fixed on 2026-10-05, before any data
#: (`docs/decisions.md`): the clustered policy's block as written, against a curve with radius 20 mm, width 3 mm and
#: no lapse; the same truth with the trials at fixed distances; and for the paired test 0.40 against 0.90.
SIMULATED_CASE = {"block": "T1a", "r50_true_mm": 20.0, "width_true_mm": 3.0, "lapse_true": 0.0}
SIMULATED_DESIGN, SIMULATED_FIXED_DESIGN = "protocol", "constant"
SIMULATED_RATES = (0.4, 0.9)
#: The widths of the true curve for which each threshold block's simulated error is quoted (protocol, section 8): a
#: cliff, a middling width and a slope.
SIMULATED_WIDTHS_MM = {"cliff": 1.0, "middling": 3.0, "slope": 10.0}
#: A true curve is already falling at 0 mm when its radius is less than this many widths, and runs past a block's
#: largest distance when its radius is within as many widths of it (`procedure_report.FALL_HALF_SPAN_WIDTHS`).
FALL_HALF_SPAN_WIDTHS = 2.0

_LEROBOT_COMMIT = re.compile(r"\bat ([0-9a-f]{7,40})\b")
#: What the trial runner writes as the reason of a trial that a key ended at once (`trial_session.main`). The time
#: differs from one such trial to the next; the limit is the protocol's `execution.min_attempt_s`.
_KEY_SLIP = re.compile(r"^ended by a key \d+(?:\.\d+)? s after it began \(under (\d+(?:\.\d+)?) s\)$")
#: The entries of a record's `execution` that belong to one trial and take few values: listed for each condition.
_LISTED_PER_TRIAL = ("device", HANDOFF_USED_KEY)
#: The entry that holds the arm's joint positions when the attempt began.
_START_POSE_KEY = "start_pose_deg"
#: The entries the achieved control rate is made from: how many steps were sent, and the seconds from the first one
#: sent to the last.
_STEPS_KEY, _LOOP_KEY = "steps_sent", "loop_s"
#: The setting that holds the control rate a session asked for.
_REQUESTED_RATE_KEY = "control_hz"
#: The entry that holds how far the front camera's picture stood from where its session solved the conversion, read
#: just before the attempt, in millimetres as the session read them (the trial runner, since 2026-10-07).
_VIEW_CREPT_KEY = "view_crept_mm"
#: The report says so when the trials of a kind ran, by their median, more than this share below the requested rate.
#: The line is the analysis's own, not the protocol's. An earlier dataset of this rig was captured at 28 Hz, 7 %
#: below the 30 Hz it was stamped with, and nobody noticed.
RATE_SHORTFALL_LIMIT = 0.02
#: What the achieved control rate is, and what it leaves out. The first interval is not paced: read in the arm
#: tooling's `run_attempt` and measured once on the session tests' fake rig (11 steps at a requested 20 Hz: 0.0 ms,
#: then 51 to 55 ms).
ACHIEVED_RATE_MEANING = (
    "the rate between the first step sent and the last: the steps sent after the first, over the time from the "
    "first step to the last (the record's `loop_s`, taken on the trial runner's monotonic clock and kept to a "
    "millisecond). The handoff gate and the policy's first inference come before the first step and are outside "
    "it, and so is everything after the last step. The loop sends its second step as soon as it is predicted, "
    "without waiting out a period, so a trial's figure can read high by up to one interval of its attempt: a third "
    "of a percent for 300 steps"
)
#: The kinds of trial the achieved rate is given for, and the conditions of each.
RATE_GROUPS = {
    "policy": ("act_clustered", "act_spread"),
    "replay": ("replay",),
    "all": ("act_clustered", "act_spread", "replay"),
}
#: Stands for a setting that a condition's records do not carry.
_MISSING = object()
#: Millimetres on the floor for one pixel of the front camera at the basket: `MM_PER_PX` in
#: `src/tolerance_radius/rig/basket_place.py` (the two floor tags are 526.9 mm and 486 px apart). The scene audit
#: reads the basket's place with it where a frame's own tags give no scale. Written again here, because the analysis
#: may not import the rig's modules; `tests/test_report.py` holds the two together.
SCENE_MM_PER_PX = 1.08
SCENE_MM_PER_PX_SOURCE = (
    "`MM_PER_PX` in `src/tolerance_radius/rig/basket_place.py`: the two floor tags are 526.9 mm and 486 px apart"
)
#: A basket that ended a trial more than this far from where it began was moved in it: `BASKET_MOVED_MM` in
#: `src/tolerance_radius/rig/scene_audit.py`, the rule behind the audit's `basket_was_moved`. Written again here for
#: the report's text, and held to the audit's by the same test.
SCENE_BASKET_MOVED_MM = 5.0
#: The upper ends of the bins in which the report counts how far the basket stood from its place at a trial's start
#: (the larger of across and down); a last bin holds everything beyond. A reading on an end belongs to the bin it
#: ends, as a reading on the audit's limit is not beyond it.
SCENE_BIN_ENDS_MM = (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------------------------------------------------
# Block V0: the instrument against a scale
# --------------------------------------------------------------------------------------------------------------------


def scale_check(runs: Sequence[PositionLog], true_step_mm: float = SCALE_STEP_MM) -> dict[str, Any] | None:
    """The scale check: for each run its factor and step errors, and the errors of all steps together.

    A run's factor is the straight distance from its first reading to its last, divided by the true distance (the
    number of steps times `true_step_mm`). Everything here is as the camera reads it: this is the check that decides
    whether distances are corrected, so nothing in it is corrected. None when there is no run.
    """
    if not runs:
        return None
    listed, errors = [], []
    for run in runs:
        check = check_against_scale(list(run.readings), true_step_mm)
        errors.extend(abs(error) for error in check.step_errors_mm)
        listed.append(
            {
                "file": run.file,
                "label": run.label,
                "n_steps": len(check.step_errors_mm),
                "true_span_mm": check.true_span_mm,
                "measured_span_mm": check.measured_span_mm,
                "factor": stable(check.measured_span_mm / check.true_span_mm),
                "mean_abs_step_error_mm": check.mean_abs_step_error_mm,
                "max_abs_step_error_mm": check.max_abs_step_error_mm,
                "max_off_line_mm": check.max_off_line_mm,
            }
        )
    mean_error, worst = math.fsum(errors) / len(errors), max(errors)
    return {
        "true_step_mm": true_step_mm,
        "n_runs": len(listed),
        "n_steps": len(errors),
        "mean_abs_step_error_mm": mean_error,
        "max_abs_step_error_mm": worst,
        "pass_line": {"mean_abs_mm": SCALE_MEAN_LIMIT_MM, "max_abs_mm": SCALE_MAX_LIMIT_MM},
        "passes": mean_error <= SCALE_MEAN_LIMIT_MM and worst <= SCALE_MAX_LIMIT_MM,
        "runs": listed,
    }


def repeatability(sets: Sequence[PositionLog]) -> dict[str, Any] | None:
    """Repeated placements on one target: how far the readings of a set lie from the set's own mean, as read.

    With several sets, each is compared with its own mean and the largest distance of any of them is given. None
    when there is no set.
    """
    if not sets:
        return None
    scatters = [scatter_mm(list(log.readings)) for log in sets]
    return {
        "n": sum(len(log.readings) for log in sets),
        "n_sets": len(sets),
        "scatter_max_mm": max(largest for _, largest in scatters),
        "scatter_rms_mm": max(rms for rms, _ in scatters),
        "files": [log.file for log in sets],
    }


# --------------------------------------------------------------------------------------------------------------------
# Block V1: resets by eye
# --------------------------------------------------------------------------------------------------------------------


def by_eye_resets(resets: Sequence[ByEyeReset], divisor: float) -> dict[str, Any] | None:
    """How far resets by eye landed from where the cube was for the reference picture. None without a reset.

    Distances are divided by `divisor`, the scale correction of the analysis.
    """
    if not resets:
        return None
    errors = [stable(reset.error_mm / divisor) for reset in resets]
    operators = sorted({reset.label for reset in resets})
    return {
        "n": len(errors),
        "n_operators": len(operators),
        "operators": operators,
        "error_median_mm": median(errors),
        "error_p95_mm": percentile(errors, BY_EYE_PERCENTILE),
        "error_max_mm": max(errors),
        "errors_mm": errors,
    }


# --------------------------------------------------------------------------------------------------------------------
# The two datasets
# --------------------------------------------------------------------------------------------------------------------


def dataset_summary(
    dataset: str, starts: Sequence[DemoStart] | None, placements: Sequence[RecordedPlacement] | None, divisor: float
) -> dict[str, Any] | None:
    """One demonstration set: its audit, and what the recorder's placement log says about how it was recorded.

    From the audit: how many episodes were trained on, how far each measured start lies from its planned target,
    for the clustered set how far the starts of one point scatter about their mean, and for the spread set the
    smallest distance between two starts. None when the set has no audit.
    """
    if starts is None:
        return None
    trained = [start for start in starts if start.trained]
    errors = [
        stable(start.measured.distance_to(start.planned) / divisor) for start in starts if start.planned is not None
    ]
    summary: dict[str, Any] = {
        "n_episodes": len(starts),
        "n_trained": len(trained),
        "placement_errors_mm": errors,
        "placement_error_median_mm": median(errors),
        "placement_error_max_mm": max(errors) if errors else None,
        "scatter_max_mm": None,
        "spacing_min_mm": None,
        "recording": recording_summary(placements, [start.episode_index for start in starts]),
    }
    if dataset == "clustered" and trained:
        summary["scatter_max_mm"] = _scatter_about_points(trained) / divisor
    if dataset == "spread" and len(trained) > 1:
        summary["spacing_min_mm"] = _smallest_spacing(trained) / divisor
    return summary


def _scatter_about_points(starts: Sequence[DemoStart]) -> float:
    """The largest distance of a start from the mean start of its point. A point is a planned target; episodes
    without one are grouped by their cell."""
    points: dict[Any, list[Point]] = {}
    for start in starts:
        key = (start.planned.x_mm, start.planned.y_mm) if start.planned is not None else start.cell
        points.setdefault(key, []).append(start.measured)
    return max(scatter_mm(readings)[1] for readings in points.values())


def _smallest_spacing(starts: Sequence[DemoStart]) -> float:
    return min(
        first.measured.distance_to(second.measured)
        for index, first in enumerate(starts)
        for second in starts[index + 1 :]
    )


def recording_summary(
    placements: Sequence[RecordedPlacement] | None, audited_episodes: Sequence[int]
) -> dict[str, Any] | None:
    """What the recorder's placement log says: episodes recorded again, lines that were recorded over, days.

    The log holds one line for every episode the recorder saved, a throwaway smoke test included, so an episode
    index can occur more than once. The last line of an index is the episode the dataset holds; earlier lines of the
    same index were recorded over (`n_superseded`). `n_rerecorded_episodes` counts the kept episodes that were
    recorded again at least once before they were saved, `n_rerecords` how many extra takes that was. `matches_audit`
    says whether the kept lines are exactly the audited episodes; when they are not, the counts describe something
    other than the audited dataset and the caller should not use them. None when there is no placement log.
    """
    if placements is None:
        return None
    kept: dict[int, RecordedPlacement] = {}
    for placement in placements:
        kept[placement.episode_index] = placement
    superseded = [p for p in placements if kept[p.episode_index] is not p]
    days = sorted({parse_time(p.saved_at).date().isoformat() for p in kept.values()})
    return {
        "n_lines": len(placements),
        "n_kept": len(kept),
        "n_superseded": len(superseded),
        "superseded_sessions": sorted({p.session for p in superseded}),
        "n_rerecorded_episodes": sum(p.rerecords > 0 for p in kept.values()),
        "n_rerecords": sum(p.rerecords for p in kept.values()),
        "days": days,
        "sessions": sorted({p.session for p in kept.values()}),
        "matches_audit": sorted(kept) == sorted(audited_episodes),
    }


def recording_days(datasets: Mapping[str, Mapping[str, Any] | None]) -> list[str] | None:
    """The days on which the kept episodes of every set were saved; None unless every set has a placement log that
    matches its audit."""
    days: set[str] = set()
    for summary in datasets.values():
        recording = summary.get("recording") if summary else None
        if not recording or not recording["matches_audit"]:
            return None
        days.update(recording["days"])
    return sorted(days) if datasets else None


# --------------------------------------------------------------------------------------------------------------------
# How the trials were run
# --------------------------------------------------------------------------------------------------------------------


def reason_shown(reason: str | None) -> str:
    """A void trial's reason as it is counted: white space as single spaces, no final full stop.

    A trial that a key ended at once carries the time in its reason (`ended by a key 0.4 s after it began (under
    1 s)`). The time is left out here, so that all such slips of the key are one reason.
    """
    shown = " ".join(str(reason).split()).rstrip(".")
    slip = _KEY_SLIP.match(shown)
    return f"ended by a key less than {slip.group(1)} s after it began" if slip else shown


def void_reasons(trials: Sequence[CountedTrial]) -> list[dict[str, Any]]:
    """Why trials were void: each reason with its count, the most frequent first.

    Reasons are typed by a person or written by the trial runner. Two that differ only in capitals, spacing or a
    final full stop are one reason, shown as it was first written; so are all slips of the key (`reason_shown`).
    """
    counts: dict[str, list[Any]] = {}
    for trial in trials:
        if trial.record.scored:
            continue
        shown = reason_shown(trial.record.void_reason)
        entry = counts.setdefault(shown.casefold(), [shown, 0])
        entry[1] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1][1], item[0]))
    return [{"reason": shown, "n": n} for _, (shown, n) in ordered]


def conduct(
    trials: Sequence[CountedTrial], divisor: float, view_crept_limit_mm: float | None = None
) -> dict[str, Any]:
    """What the trial records say about how the trials were run.

    Days and people are counted over every record, void ones too. `trial_minutes_median` is the median time from
    one record's start to the next one's in the same log, void trials included and no pause left out.
    `view_crept_limit_mm` is the protocol's limit for the camera's creep before an attempt (`view_crept`).
    """
    scored = [trial for trial in trials if trial.record.scored]
    reset_errors = [stable(trial.record.placement_error_mm / divisor) for trial in scored]
    intervals = []
    by_log: dict[str, list[CountedTrial]] = {}
    for trial in trials:
        by_log.setdefault(trial.log, []).append(trial)
    for found in by_log.values():
        starts = [parse_time(trial.record.ts_start) for trial in found]
        intervals.extend((later - earlier).total_seconds() / 60.0 for earlier, later in itertools.pairwise(starts))
    people = sorted({code for trial in trials for code in (trial.record.operator, trial.record.scorer) if code})
    return {
        "void_reasons": void_reasons(trials),
        "void_trials": [
            {"trial_id": t.trial_id, "block": t.block, "condition": t.condition, "reason": t.record.void_reason}
            for t in trials
            if not t.record.scored
        ],
        "reset_errors_mm": reset_errors,
        "reset_error_median_mm": median(reset_errors),
        "reset_error_max_mm": max(reset_errors) if reset_errors else None,
        "trial_minutes_median": median(intervals),
        "n_trial_intervals": len(intervals),
        "trial_days": sorted({parse_time(trial.record.ts_start).date().isoformat() for trial in trials}),
        "people": people,
        "execution": execution_summary(scored),
        "view_crept": view_crept(scored, view_crept_limit_mm),
        "start_pose_deg": start_poses(scored),
        "checkpoints": checkpoints(scored),
        "code": code_versions(trials),
    }


def execution_summary(scored: Sequence[CountedTrial]) -> dict[str, Any]:
    """The execution settings the scored trials ran with: for each condition, and those that all conditions share.

    The reader of the logs has already refused scored trials of one condition that differ, so one trial stands for
    its condition. `common` holds the settings whose value is the same in every condition that was run. `per_trial`
    lists, for each condition, the values that occur among its scored trials of two entries that belong to the one
    trial and are not settings: the device the policy ran on and the tolerance its handoff gate used. `achieved_hz`
    is the control rate the trials really ran at, from two more such entries (`achieved_rates`).
    """
    by_condition: dict[str, dict[str, Any]] = {}
    per_trial: dict[str, dict[str, list[Any]]] = {}
    for trial in scored:
        by_condition.setdefault(trial.condition, execution_settings(trial.record))
        seen = per_trial.setdefault(trial.condition, {key: [] for key in _LISTED_PER_TRIAL})
        for key in _LISTED_PER_TRIAL:
            value = trial.record.execution.get(key)
            if value is not None and value not in seen[key]:
                seen[key].append(value)
    for seen in per_trial.values():
        for values in seen.values():
            values.sort(key=lambda value: (isinstance(value, str), value))
    common: dict[str, Any] = {}
    if by_condition:
        first, *others = by_condition.values()
        common = {
            key: value for key, value in first.items() if all(other.get(key, _MISSING) == value for other in others)
        }
    return {
        "by_condition": by_condition,
        "common": common,
        "per_trial": per_trial,
        "achieved_hz": achieved_rates(scored),
    }


def loop_seconds(execution: Mapping[str, Any]) -> float | None:
    """How long a trial's control loop ran, from the first step sent to the last, as its record says (`loop_s`);
    None when the record does not carry it as a number."""
    value = execution.get(_LOOP_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def achieved_rate_hz(steps_sent: Any, loop_s: Any) -> float | None:
    """The control rate a trial achieved: `(steps_sent - 1) / loop_s`, the intervals between the steps that were
    sent over the time from the first step to the last (`ACHIEVED_RATE_MEANING`).

    None when there is no rate to be had: fewer than two steps were sent, the loop's duration is not above zero, or
    the record does not carry one of the two as a number.
    """
    if isinstance(steps_sent, bool) or not isinstance(steps_sent, int) or steps_sent < 2:
        return None
    if isinstance(loop_s, bool) or not isinstance(loop_s, (int, float)) or not math.isfinite(loop_s) or not loop_s > 0:
        return None
    return stable((steps_sent - 1) / loop_s)


def trial_rate_hz(trial: CountedTrial) -> float | None:
    """The control rate one recorded trial achieved (`achieved_rate_hz` of its record)."""
    execution = trial.record.execution
    return achieved_rate_hz(execution.get(_STEPS_KEY), execution.get(_LOOP_KEY))


def achieved_rates(scored: Sequence[CountedTrial]) -> dict[str, Any]:
    """The control rate the scored trials achieved: for the policy trials, for the replay trials, and for all.

    For each kind: the rate the trials were asked to run at (`requested`, their `control_hz` when they share one),
    the median, the smallest and the largest achieved rate, the number of trials these rest on (`n`) and the number
    of scored trials of that kind that have no rate (`n_without`: too short, or a record without `loop_s`).
    `short_of_requested` says whether the median lies more than `RATE_SHORTFALL_LIMIT` below the requested rate;
    None when there is no median or no requested rate to hold it against.
    """
    rates: dict[str, Any] = {"shortfall_limit": RATE_SHORTFALL_LIMIT}
    for kind, conditions in RATE_GROUPS.items():
        own = [trial for trial in scored if trial.condition in conditions]
        found = [rate for rate in (trial_rate_hz(trial) for trial in own) if rate is not None]
        asked = {trial.record.execution.get(_REQUESTED_RATE_KEY) for trial in own}
        requested = next(iter(asked)) if len(asked) == 1 else None
        if isinstance(requested, bool) or not isinstance(requested, (int, float)) or not requested > 0:
            requested = None
        middle = median(found)
        short = None
        if middle is not None and requested is not None:
            short = bool((requested - middle) / requested > RATE_SHORTFALL_LIMIT + 1e-9)
        rates[kind] = {
            "requested": requested,
            "median": middle,
            "min": min(found) if found else None,
            "max": max(found) if found else None,
            "n": len(found),
            "n_without": len(own) - len(found),
            "short_of_requested": short,
        }
    return rates


def view_crept(scored: Sequence[CountedTrial], limit_mm: float | None) -> dict[str, Any] | None:
    """How far the front camera's picture had crept before the scored attempts began.

    A session reads, just before each attempt, how far the picture stands from where it solved its conversion from
    pixels to millimetres, writes the figure into the record (`view_crept_mm`) and makes no attempt beyond the
    protocol's limit (`limit_mm`, its `[execution] view_crept_limit_mm`; None when the protocol states none). Given
    here: how many scored trials carry a reading (`n`) and how many do not (`n_without`), the median and the largest
    reading, and how many readings lie beyond the limit. The readings are left as the sessions read them, not divided
    by the scale correction: they were held against the limit as read. None when no scored record carries one.
    """
    readings = []
    for trial in scored:
        value = trial.record.execution.get(_VIEW_CREPT_KEY)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            readings.append(float(value))
    if not readings:
        return None
    return {
        "n": len(readings),
        "n_without": len(scored) - len(readings),
        "median_mm": median(readings),
        "max_mm": max(readings),
        "limit_mm": limit_mm,
        "n_over_limit": None if limit_mm is None else sum(reading > limit_mm for reading in readings),
    }


def start_poses(scored: Sequence[CountedTrial]) -> dict[str, Any] | None:
    """Where the arm started from in the scored trials: for each joint the smallest, the median and the largest
    position the records give, in the records' own units. None when no scored record carries a start pose."""
    by_joint: dict[str, list[float]] = {}
    carrying = 0
    for trial in scored:
        pose = trial.record.execution.get(_START_POSE_KEY)
        if not isinstance(pose, Mapping):
            continue
        carrying += 1
        for joint, value in pose.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_joint.setdefault(str(joint), []).append(float(value))
    if not by_joint:
        return None
    return {
        "n": carrying,
        "joints": {
            joint: {"min": min(values), "median": median(values), "max": max(values)}
            for joint, values in sorted(by_joint.items())
        },
    }


def checkpoints(scored: Sequence[CountedTrial]) -> dict[str, Any]:
    """The checkpoints the policies were evaluated at: the names, the training runs, and the training step.

    A record names its checkpoint `<run>@<step>`, or by the step alone (`inputs.split_checkpoint`). For each policy
    the names and the runs that occur among its scored trials are listed. `step` is the step as the records name
    it, when all of them name the same one; `evaluation_step` is that name read as a number, or None when it is not
    one.
    """
    names: dict[str, set[str]] = {}
    runs: dict[str, set[str]] = {}
    steps: set[str] = set()
    for trial in scored:
        name = trial.record.checkpoint
        if name is None:
            continue
        run, step = split_checkpoint(name)
        names.setdefault(trial.condition, set()).add(name)
        runs.setdefault(trial.condition, set()).update([] if run is None else [run])
        steps.add(step)
    step = next(iter(steps)) if len(steps) == 1 else None
    return {
        "by_condition": {condition: sorted(found) for condition, found in sorted(names.items())},
        "runs": {condition: sorted(found) for condition, found in sorted(runs.items())},
        "step": step,
        "evaluation_step": int(step) if step is not None and step.isdigit() else None,
    }


def code_versions(trials: Sequence[CountedTrial]) -> dict[str, Any]:
    """The code the trials ran on, from the records' `versions`: every value that occurs, and the LeRobot commit when
    all records name the same one."""
    seen: dict[str, set[str]] = {}
    for trial in trials:
        for key, value in trial.record.versions.items():
            seen.setdefault(key, set()).add(str(value))
    lines = seen.get("lerobot", set())
    commit = None
    if len(lines) == 1:
        found = _LEROBOT_COMMIT.search(next(iter(lines)))
        commit = found.group(1) if found else None
    return {"versions": {key: sorted(values) for key, values in sorted(seen.items())}, "lerobot_commit": commit}


def deviations(rows: Sequence[Deviation] | None) -> dict[str, Any] | None:
    """The deviations from the frozen protocol that the decision log marks; None when there is no decision log."""
    if rows is None:
        return None
    return {"n": len(rows), "rows": [{"date": row.date, "decision": row.decision, "why": row.why} for row in rows]}


# --------------------------------------------------------------------------------------------------------------------
# The scene at each trial's start
# --------------------------------------------------------------------------------------------------------------------


def scene_at_start(
    scored: Sequence[CountedTrial], scenes: Mapping[str, SceneAudit | None]
) -> dict[str, Any] | None:
    """Where the basket and the wrist camera stood when the scored trials began, from the scene audits of the logs.

    The basket stands on tape and the arm can carry it off, and the same catch can move the wrist camera in its
    mount. So the first and the last frames of every trial's two videos are read for both, and a trial that began
    with the basket or the wrist camera beyond the audit's limit from its place is named in the report (protocol,
    section 8). Counted here over the scored trials, each with the outcome its record gives:

    * how many are in an audit (`n_audited`) and which are not; in how many the basket's place at the start could be
      read, and the wrist camera's, and which trials' videos did not say;
    * the trials in which the basket was moved, by the audit's own rule (`SCENE_BASKET_MOVED_MM`), and those whose
      end says nothing of the basket;
    * the trials named for the basket and for the wrist camera, the largest offsets at a start, and how many trials
      began with the basket in each bin of `SCENE_BIN_ENDS_MM`;
    * the successes of the named trials (named for either), of the trials named for neither, and of the rest, whose
      start could not be read. Plain counts: no test is made of them.

    The readings are as the audit gives them, in millimetres on the floor and in pixels of the wrist camera's
    picture; nothing is divided by the scale correction, which belongs to the instrument that measures the cube.
    None when no log has an audit, or when the audits hold none of the scored trials.
    """
    audits = [audit for audit in scenes.values() if audit is not None]
    readings = {
        (log, reading.trial_id): reading
        for log, audit in scenes.items()
        if audit is not None
        for reading in audit.readings
    }
    found = [(trial, readings.get((trial.log, trial.trial_id))) for trial in scored]
    audited = [(trial, reading) for trial, reading in found if reading is not None]
    if not audited:
        return None

    def names(wanted: Any) -> list[str]:
        return [trial.trial_id for trial, reading in audited if wanted(reading)]

    def successes(group: Sequence[CountedTrial]) -> dict[str, int]:
        return {"n": len(group), "k": sum(bool(trial.record.success) for trial in group)}

    basket = [reading.basket_off_mm for _, reading in audited if reading.basket_off_mm is not None]
    wrist = [reading.wrist_off_px for _, reading in audited if reading.wrist_off_px is not None]
    edges = (0.0, *SCENE_BIN_ENDS_MM, None)
    bins = [
        {
            "over_mm": low,
            "up_to_mm": high,
            # The first bin holds a basket that stood exactly in its place as well.
            "n": sum((off > low or low == 0.0) and (high is None or off <= high) for off in basket),
        }
        for low, high in itertools.pairwise(edges)
    ]
    # Named for either; named for neither, which needs both readings; and the rest, whose start did not say.
    named = [trial for trial, reading in audited if reading.named_basket or reading.named_wrist]
    unnamed = [
        trial for trial, reading in audited if reading.named_basket is False and reading.named_wrist is False
    ]
    judged = {trial.trial_id for trial in (*named, *unnamed)}
    named_basket = [reading.basket_off_mm for _, reading in audited if reading.named_basket]
    return {
        "files": [audit.file for audit in audits],
        "basket_named_mm": audits[0].basket_named_mm,
        "wrist_named_px": audits[0].wrist_named_px,
        "mm_per_px": SCENE_MM_PER_PX,
        "basket_moved_limit_mm": SCENE_BASKET_MOVED_MM,
        "n_scored": len(scored),
        "n_audited": len(audited),
        "not_audited": [trial.trial_id for trial, reading in found if reading is None],
        "n_basket_read": len(basket),
        "n_wrist_read": len(wrist),
        "basket_unread": names(lambda reading: reading.basket_start_mm is None),
        "wrist_unread": names(lambda reading: reading.wrist_start_px is None),
        "n_basket_moved": sum(bool(reading.basket_was_moved) for _, reading in audited),
        "basket_moved": [
            {
                "trial_id": trial.trial_id,
                "condition": trial.condition,
                "outcome": trial.record.outcome,
                "moved_mm": reading.basket_moved_mm,
            }
            for trial, reading in audited
            if reading.basket_was_moved
        ],
        "moved_unknown": names(lambda reading: reading.basket_was_moved is None),
        "n_named_basket": len(named_basket),
        "named_basket": names(lambda reading: bool(reading.named_basket)),
        "named_basket_min_mm": min(named_basket) if named_basket else None,
        "n_named_wrist": sum(bool(reading.named_wrist) for _, reading in audited),
        "named_wrist": names(lambda reading: bool(reading.named_wrist)),
        "basket_start_max_mm": max(basket) if basket else None,
        "wrist_start_max_px": max(wrist) if wrist else None,
        "basket_start_bins": bins,
        "success": {
            "named": successes(named),
            "unnamed": successes(unnamed),
            "start_not_read": successes([trial for trial, _ in audited if trial.trial_id not in judged]),
        },
    }


# --------------------------------------------------------------------------------------------------------------------
# The second scoring from video
# --------------------------------------------------------------------------------------------------------------------


def second_scoring(
    found: SecondScoring | None, logs: Sequence[str], logs_digest: str | None
) -> dict[str, Any] | None:
    """What the report gives of the second scoring from video, read from its agreement file.

    `logs` are the counted logs this report reads and `logs_digest` their digest as they are now
    (`inputs.logs_digest`). The counts are given only when the agreement is the second scoring of exactly these
    trials. `why_not` says why it is not, and every count is then None: the agreement is a rehearsal, made from
    trials that are not counted; it was drawn from other logs; or a log has changed since the draw.

    `same_person` is true when every item's two scores carry the same role code: the operator scored again, with the
    trials' identities hidden (`protocol/protocol.md`, section 8). None altogether when there is no agreement file.
    """
    if found is None:
        return None
    result: dict[str, Any] = {
        "file": found.file,
        "rehearsal": found.rehearsal,
        "logs": list(found.logs),
        "seed": found.seed,
        "one_in": found.one_in,
        "n_scored": found.n_scored,
        "n_drawn": found.n_drawn,
        "interval_level": found.interval_level,
        "why_not": None,
        "all": None,
        "by_kind": None,
        "scorers": None,
        "same_person": None,
        "disagreements": None,
        "cannot_tell": None,
    }
    if found.rehearsal:
        result["why_not"] = (
            f"{found.file} is a rehearsal, drawn from trials that are not counted: nothing in it is a result"
        )
    elif sorted(found.logs) != sorted(logs):
        result["why_not"] = (
            f"{found.file} is the second scoring of {' and '.join(found.logs) or 'no log'}, and this report reads "
            f"{' and '.join(logs) or 'no log'}"
        )
    elif found.logs_digest != logs_digest:
        result["why_not"] = (
            f"{found.file} was drawn from {' and '.join(found.logs)} as it was then, and the file has changed "
            "since (its digest is no longer the one the draw recorded): a draw is made after the last trial, and "
            "trials logged since need a new draw"
        )
    if result["why_not"] is not None:
        return result
    whole = found.tallies["all"]
    result.update(
        {
            "all": dict(whole),
            "by_kind": {kind: dict(tally) for kind, tally in found.tallies.items() if kind != "all"},
            "scorers": {
                "first": list(found.first_scorers),
                "second": list(found.second_scorers),
                "n_same_role": found.n_same_role,
            },
            "same_person": whole["n_items"] > 0 and found.n_same_role == whole["n_items"],
            "disagreements": [
                {"trial_id": trial, "first": first, "second": second} for trial, first, second in found.disagreements
            ],
            "cannot_tell": [{"trial_id": trial, "first": first} for trial, first, _ in found.cannot_tell],
        }
    )
    return result


# --------------------------------------------------------------------------------------------------------------------
# The simulated case the paper quotes
# --------------------------------------------------------------------------------------------------------------------


def simulated_case(
    tables: SimulationTables | None,
    protocol_sha256: str,
    pairs: int | None,
    max_level_mm: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """What the paper quotes of the procedure simulation, read from its tables.

    `case` is the one case fixed before the data (`SIMULATED_CASE`). `blocks` gives, for each threshold block as
    written, the simulated error and interval width at three widths of the true curve, and how often the interval
    is open below when the curve is already falling at 0 mm (`simulated_blocks`); `max_level_mm` names those blocks
    with the largest distance each may request. `comparison_rule` is the simulation of the rule of prediction 2
    (`simulated_rule`), which has a report and a digest of its own.

    The simulation's report names the protocol file it read by the first 12 hex digits of its SHA-256. When that is
    not the present protocol file, the simulation describes another protocol: `stale` is true, and `case` and
    `blocks` are None until the simulation is run again. `case` is also None when a row it needs is not in the
    tables. None altogether when there are no tables.
    """
    if tables is None:
        return None
    now = protocol_sha256[:12]
    result: dict[str, Any] = {
        "files": list(tables.files),
        "protocol_digest_simulated": tables.protocol_digest,
        "protocol_digest_now": now,
        "stale": tables.protocol_digest != now,
        "case": None,
        "why_no_case": None,
        "blocks": None,
        "open_below_falling_at_zero": None,
        "comparison_rule": simulated_rule(tables, now),
    }
    if result["stale"]:
        said = f"`{tables.protocol_digest}`" if tables.protocol_digest else "no protocol file"
        result["why_no_case"] = f"the simulation names {said}, and the protocol file is now `{now}`"
        return result
    result.update(simulated_blocks(tables.sweep, max_level_mm or {}))
    as_written = _sweep_row(tables.sweep, SIMULATED_DESIGN)
    fixed = _sweep_row(tables.sweep, SIMULATED_FIXED_DESIGN)
    low, high = SIMULATED_RATES
    power = next(
        (
            row
            for row in tables.power
            if pairs is not None
            and int(row["pairs"]) == pairs
            and _close(row["p_first"], low)
            and _close(row["p_second"], high)
        ),
        None,
    )
    missing = [
        name
        for name, row in (
            ("the block as written", as_written),
            ("the fixed distances", fixed),
            ("the paired test", power),
        )
        if row is None
    ]
    if missing:
        result["why_no_case"] = f"the simulation's tables have no row for {', '.join(missing)}"
        return result
    result["case"] = {
        **SIMULATED_CASE,
        "replications": int(as_written["replications"]),
        "bias_mm": float(as_written["bias_mm"]),
        "rmse_mm": float(as_written["rmse_mm"]),
        "interval_width_mm": float(as_written["mean_interval_width_mm"]),
        "coverage": float(as_written["interval_coverage"]),
        "rmse_fixed_mm": float(fixed["rmse_mm"]),
        "interval_width_fixed_mm": float(fixed["mean_interval_width_mm"]),
        "paired": {"pairs": pairs, "rate_low": low, "rate_high": high, "power": float(power["power"])},
    }
    return result


def simulated_rule(tables: SimulationTables, digest_now: str) -> dict[str, Any] | None:
    """The simulation of the comparison rule (`comparison_rule.csv`): how often the rule gave each verdict.

    One entry for every simulated case: the width of both curves, the true difference (the policy's radius minus
    the replay's), and the share of the simulated pairs of blocks with each verdict. Its report names the protocol
    file it read; when that is not the present one, `stale` is true and `cases` is None, as for the procedure
    simulation. None when the table is not there.
    """
    if tables.rule is None:
        return None
    result: dict[str, Any] = {
        "protocol_digest_simulated": tables.rule_digest,
        "stale": tables.rule_digest != digest_now,
        "pairs": None,
        "replay_r50_mm": None,
        "cases": None,
    }
    if result["stale"] or not tables.rule:
        return result
    pairs = {int(row["pairs"]) for row in tables.rule}
    radii = {float(row["replay_r50_mm"]) for row in tables.rule}
    result.update(
        pairs=pairs.pop() if len(pairs) == 1 else None,
        replay_r50_mm=radii.pop() if len(radii) == 1 else None,
        cases=[
            {
                "width_mm": float(row["width_mm"]),
                "difference_mm": float(row["difference_true_mm"]),
                "larger": float(row["share_larger"]),
                "not_told_apart": float(row["share_not_told_apart"]),
                "smaller": float(row["share_smaller"]),
            }
            for row in tables.rule
        ],
    )
    return result


def simulated_blocks(sweep: Sequence[Mapping[str, str]], max_level_mm: Mapping[str, float]) -> dict[str, Any]:
    """Each threshold block as written, in the procedure simulation's table (`procedure.csv`).

    For each block of `max_level_mm` and each width of `SIMULATED_WIDTHS_MM`: the error (RMSE) and the mean interval
    width of the middle one of the simulated truths with that width, every radius and lapse rate. That is the first
    number of a cell of the simulation report's Table 1: the median over those truths, each quantity on its own.
    And for each block the share of simulated blocks whose interval is open below, averaged over the truths whose
    curve is already falling at 0 mm and does not also run past the block's largest distance (the report's
    Table 6); `open_below_falling_at_zero` is the same over all the blocks' truths together (the report's reading).
    An entry is None where the table has no row for it.
    """
    blocks: dict[str, Any] = {}
    falling: list[float] = []
    for block, largest in max_level_mm.items():
        rows = [row for row in sweep if row["design"] == SIMULATED_DESIGN and row["block"] == block]
        by_width: dict[str, Any] = {}
        for name, width in SIMULATED_WIDTHS_MM.items():
            found = [row for row in rows if _close(row["width_true_mm"], width)]
            by_width[name] = (
                {
                    "width_true_mm": width,
                    "n_truths": len(found),
                    "rmse_mm": median([float(row["rmse_mm"]) for row in found]),
                    "interval_width_mm": median([float(row["mean_interval_width_mm"]) for row in found]),
                }
                if found
                else None
            )
        early = [float(row["lower_edge_rate"]) for row in rows if _falls_at_zero_only(row, largest)]
        falling.extend(early)
        blocks[block] = {
            "n_trials": int(rows[0]["n_trials"]) if rows else None,
            "by_width": by_width,
            "n_truths_falling_at_zero": len(early),
            "open_below_falling_at_zero": math.fsum(early) / len(early) if early else None,
        }
    return {
        "blocks": blocks,
        "open_below_falling_at_zero": math.fsum(falling) / len(falling) if falling else None,
    }


def _falls_at_zero_only(row: Mapping[str, str], largest_mm: float) -> bool:
    """True for a truth whose curve is already falling at 0 mm and has fallen before the block's largest distance."""
    half_span = FALL_HALF_SPAN_WIDTHS * float(row["width_true_mm"])
    radius = float(row["r50_true_mm"])
    return radius < half_span and not radius > largest_mm - half_span


def _sweep_row(rows: Sequence[Mapping[str, str]], design: str) -> Mapping[str, str] | None:
    for row in rows:
        if (
            row["design"] == design
            and row["block"] == SIMULATED_CASE["block"]
            and _close(row["r50_true_mm"], SIMULATED_CASE["r50_true_mm"])
            and _close(row["width_true_mm"], SIMULATED_CASE["width_true_mm"])
            and _close(row["lapse_true"], SIMULATED_CASE["lapse_true"])
        ):
            return row
    return None


def _close(text: str, value: float) -> bool:
    return math.isclose(float(text), value, rel_tol=1e-9, abs_tol=1e-12)
