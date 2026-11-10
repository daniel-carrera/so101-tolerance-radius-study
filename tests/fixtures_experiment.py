"""A whole synthetic experiment on disk, for the tests of the analysis. NOTHING HERE WAS MEASURED.

`build_experiment` writes into a folder what a finished experiment leaves behind: the protocol file (a copy of the
repository's with its status set to frozen), the two start-state audits, the recorder's placement logs, the
instrument's CSV files, a counted trial log and a pilot log, a decision log, the tables of the procedure simulation,
the scene audit of the counted log and the agreement of a second scoring from video. The trials are planned by the
real planner and answered from success curves chosen here, so the analysis can be checked against a known truth.

The synthetic world reads distances long by a known factor, as the real instrument is expected to: every position is
what the camera would read, and a trial succeeds with the probability its true curve gives at the reading divided by
that factor.

Everything is a function of the arguments: no clock, and random numbers only from seeded generators.
"""

from __future__ import annotations

import atexit
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from typing import Any

from tolerance_radius import config, geometry, layouts, planner, schema, targets
from tolerance_radius.geometry import Grid
from tolerance_radius.intervals import wilson_interval
from tolerance_radius.psychometric import success_probability
from tolerance_radius.schema import DemoStart, PlannedTrial, Point, TrialRecord
from tolerance_radius.seeds import derived_rng, shuffled

REPO = config.REPO_ROOT
#: What every output made from this experiment must say of itself.
SYNTHETIC_NOTE = (
    "SYNTHETIC: made by tests/fixtures_experiment.py from invented success curves. No trial was run. "
    "Nothing in this file is a result."
)
ALL_BLOCKS = ("T1a", "T1b", "T2", "T3")
#: The pacific offset the rig's clock writes.
_ZONE = timezone(timedelta(hours=-7))
#: The trial runner's own words for a handoff that was refused, and for a key that ended an attempt at once.
HANDOFF_REFUSED = "the handoff gate refused the start pose"
VOID_REASONS = (
    HANDOFF_REFUSED,
    "bumped the table",
    "ended by a key 0.4 s after it began (under {limit:g} s)",
    "camera dropped 50% of frames_in #2",
    "Bumped the table.",
    "ended by a key 0.7 s after it began (under {limit:g} s)",
)
SMOKE_SESSION = "20261006_083000"
#: The workspace test as the protocol states it since draft 0.3: 26 placements, 13 drawn over the trained cells and
#: 13 over the held-out cells. It is written out here so that the shared run keeps this design whatever the
#: repository's file states on another day.
STRATIFIED = (("trained_cells", 13), ("held_out_cells", 13))
#: The longer list of lapse values and the limit of the plateau check, as the protocol states them. They are given
#: to a protocol file that does not have them yet.
LAPSE_VALUES_SENSITIVITY = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
SENSITIVITY_LIMIT_MM = 5.0
#: The arm's joints, as a record names them, with a start pose near the one the operator sets.
START_POSE_DEG = {
    "elbow_flex.pos": 96.0,
    "gripper.pos": 2.0,
    "shoulder_lift.pos": -100.0,
    "shoulder_pan.pos": -3.0,
    "wrist_flex.pos": 75.0,
    "wrist_roll.pos": 0.0,
}
#: The entries of a record's `execution` that belong to the one trial (`trial_session.main`).
PER_TRIAL_KEYS = ("steps_sent", "loop_s", "device", "handoff_tolerance_used_deg", "start_pose_deg", "view_crept_mm")
#: The synthetic camera's picture stands between these distances (mm) from its session's conversion before a trial.
VIEW_CREPT_MM = (0.2, 0.8)
#: The synthetic control loop runs a little under the rate it is asked for: between these shares of it.
ACHIEVED_SHARE = (0.985, 0.995)
#: One scored trial is given a single step, so that it has no control rate. The trial runner as it is written would
#: have logged so short an attempt void; the analysis must not lean on that.
TOO_SHORT_TRIAL = "T1a-001"
#: The synthetic second scoring: one scored trial in this many of each condition of each block is scored again, by
#: this scorer unless another is named; of the trials drawn, the one at the first position below gets the other
#: score and the one at the second could not be told (where a draw is long enough to have them).
SECOND_ONE_IN = 5
SECOND_SCORER = "sc2"
SECOND_DIFFERS_AT, SECOND_UNTOLD_AT = 3, 7
SECOND_SEED = 20261006
#: The synthetic scene audit. Its two limits are the protocol's (section 8) and its rule for a moved basket the
#: audit's own (`scene_audit.BASKET_NAMED_MM`, `WRIST_NAMED_PX`, `BASKET_MOVED_MM`).
SCENE_BASKET_NAMED_MM, SCENE_WRIST_NAMED_PX, SCENE_BASKET_MOVED_MM = 2.0, 2.0, 5.0
#: Every so-many-th record of the log began with the basket a little beyond its limit. And, counted among the
#: scored trials from 0: the trial that began with the wrist camera beyond its limit, the one in which the arm carried
#: the basket off, the one whose last frames say nothing of the basket, the one whose front video could not be read
#: and the one whose wrist video could not be read.
SCENE_BASKET_NAMED_EVERY = 6
SCENE_WRIST_NAMED_AT, SCENE_MOVED_AT, SCENE_END_UNSEEN_AT = 4, 8, 11
SCENE_FRONT_UNREAD_AT, SCENE_WRIST_UNREAD_AT = 14, 15


@dataclass(frozen=True, slots=True)
class Truth:
    """A success curve the synthetic trials are answered from, in true millimetres."""

    r50_mm: float
    width_mm: float
    lapse: float = 0.0
    floor: float = 0.0


TRUTHS = {
    "act_clustered": Truth(22.0, 3.0, 0.03),
    "replay": Truth(12.0, 2.0, 0.0),
    "act_spread": Truth(18.0, 3.5, 0.03),
}
DEFAULT_SCALE_FACTORS = (1.030, 1.027, 1.033, 1.031)


@dataclass(frozen=True, slots=True)
class Experiment:
    """What was written, and what the analysis should find in it."""

    root: Path
    protocol: dict[str, Any]
    records: tuple[TrialRecord, ...]
    demo_starts: tuple[DemoStart, ...]
    truths: dict[str, Truth]
    #: The factor the synthetic instrument reads distances long by: the mean of the scale runs, 1 without runs.
    reads_long_by: float
    #: What the analysis should divide distances by: that factor when the runs agree within one point, else 1.
    divisor: float
    scale_factors: tuple[float, ...]
    #: Pairs of the workspace test that were built by hand in held-out cells, after the planner's own.
    held_out_pairs: int
    by_eye_errors_mm: tuple[float, ...] = field(default=())
    #: The agreement of the synthetic second scoring as it was written; None when none was.
    second_scoring: dict[str, Any] | None = None
    #: The synthetic scene audit of the counted log as it was written; None when none was.
    scene: dict[str, Any] | None = None

    @property
    def log(self) -> Path:
        return self.root / "data" / "trials" / "main.jsonl"

    def scored(self, block: str | None = None, condition: str | None = None) -> list[TrialRecord]:
        return [
            record
            for record in self.records
            if record.scored
            and (block is None or record.planned.block == block)
            and (condition is None or record.planned.condition == condition)
        ]

    def void(self, block: str | None = None, condition: str | None = None) -> list[TrialRecord]:
        return [
            record
            for record in self.records
            if not record.scored
            and (block is None or record.planned.block == block)
            and (condition is None or record.planned.condition == condition)
        ]


# --------------------------------------------------------------------------------------------------------------------
# The protocol file
# --------------------------------------------------------------------------------------------------------------------


def protocol_text(
    *,
    status: str = "frozen",
    version: str = "1.0",
    coarse: bool = False,
    workspace: tuple[tuple[str, int], ...] | None = None,
    sensitivity: bool = True,
) -> str:
    """The repository's protocol file with another status and version, and, if asked, other changes.

    * `coarse`: a coarse analysis grid, which makes a fit about fifteen times quicker. It is for tests of what is
      read and refused, where the numbers of a fit do not matter.
    * `workspace`: where the workspace test draws its placements, as `(region, how many)` for each region. One
      region is written as the block's `region`, several as its `regions` list; `pairs` is their total. Left out,
      the block is as the repository's file states it.
    * `sensitivity`: whether the analysis section states the longer list of lapse values and the limit of the
      plateau check. True gives them to a file that lacks them; False takes them out.
    """
    text = (REPO / "config" / "protocol.toml").read_text(encoding="utf-8")
    text, n_version = re.subn(r'(?m)^version = "[^"]*"', f'version = "{version}"', text, count=1)
    text, n_status = re.subn(r'(?m)^status = "[^"]*"', f'status = "{status}"', text, count=1)
    assert (n_version, n_status) == (1, 1), "the protocol file no longer states its version and status as expected"
    if coarse:
        text, n_r50 = re.subn(r"(?m)^r50_grid_mm = .*$", "r50_grid_mm = { min = 0.0, max = 80.0, step = 2.0 }", text)
        text, n_width = re.subn(
            r"(?m)^width_grid_mm = .*$", "width_grid_mm = { min = 0.5, max = 20.0, count = 8 }", text
        )
        assert (n_r50, n_width) == (1, 1), "the protocol file no longer states its grids as expected"
    if workspace is not None:
        placed = {"pairs": str(sum(count for _, count in workspace))}
        if len(workspace) == 1:
            placed["region"] = json.dumps(workspace[0][0])
        else:
            entries = ", ".join(f"{{ region = {json.dumps(region)}, pairs = {count} }}" for region, count in workspace)
            placed["regions"] = f"[{entries}]"
        text = _with_entries(text, "blocks.T2", ("pairs", "region", "regions"), placed)
    stated = tomllib.loads(text)["analysis"]
    checked = {
        "lapse_values_sensitivity": "[" + ", ".join(f"{value:g}" for value in LAPSE_VALUES_SENSITIVITY) + "]",
        "sensitivity_limit_mm": f"{SENSITIVITY_LIMIT_MM!r}",
    }
    if not sensitivity:
        text = _with_entries(text, "analysis", tuple(checked), {})
    else:
        missing = {key: value for key, value in checked.items() if key not in stated}
        text = _with_entries(text, "analysis", (), missing)
    return text


def _with_entries(text: str, table: str, without: tuple[str, ...], entries: dict[str, str]) -> str:
    """A protocol text in which one table has lost the keys `without` and gained `entries` (key: value as TOML).

    The table's other lines are left as they are. What comes out is read back and compared, so that a protocol file
    laid out in a way this does not expect fails here and not in a test that is about something else.
    """
    before = tomllib.loads(text)
    header = re.search(rf"(?m)^\[{re.escape(table)}\][^\n]*\n", text)
    assert header is not None, f"the protocol file has no [{table}]"
    following = re.search(r"(?m)^\[[A-Za-z]", text[header.end() :])
    stop = header.end() + following.start() if following else len(text)
    body = text[header.end() : stop]
    for key in (*without, *entries):
        body = _without_key(body, key)
    body = "".join(f"{key} = {value}\n" for key, value in entries.items()) + body
    changed = text[: header.end()] + body + text[stop:]

    expected, found = before, tomllib.loads(changed)
    for name in table.split("."):
        expected, found = expected[name], found[name]
    wanted = {key: value for key, value in expected.items() if key not in without and key not in entries}
    wanted.update({key: tomllib.loads(f"value = {value}")["value"] for key, value in entries.items()})
    assert found == wanted, f"[{table}] of the protocol file could not be rewritten as asked"
    return changed


def _without_key(body: str, key: str) -> str:
    """A table's body without the entry `key`, whether its value fits on a line or is a list over several."""
    entry = re.search(rf"(?m)^{re.escape(key)}[ \t]*=", body)
    if entry is None:
        return body
    at, depth = entry.end(), 0
    while at < len(body):
        line_end = body.find("\n", at)
        line_end = len(body) if line_end == -1 else line_end
        value = body[at:line_end].split("#", 1)[0]
        depth += sum(value.count(mark) for mark in "[{") - sum(value.count(mark) for mark in "]}")
        at = min(line_end + 1, len(body))
        if depth <= 0:
            break
    return body[: entry.start()] + body[at:]


# --------------------------------------------------------------------------------------------------------------------
# Audits
# --------------------------------------------------------------------------------------------------------------------


@cache
def audited_starts(dataset: str, seed: int = 5, jitter_mm: float = 1.2) -> tuple[DemoStart, ...]:
    """The committed layout of a set as a camera might have measured it: each start up to `jitter_mm` off in x and y."""
    rng = derived_rng(seed, "synthetic-audit", dataset)
    starts = []
    for episode in layouts.read_layout(REPO / "config" / "layouts" / f"{dataset}.json"):
        dx, dy = ((rng.random(2) * 2.0 - 1.0) * jitter_mm).tolist()
        measured = Point(round(episode.target.x_mm + dx, 3), round(episode.target.y_mm + dy, 3))
        starts.append(
            DemoStart(dataset, episode.episode_index, episode.cell, measured=measured, planned=episode.target)
        )
    return tuple(starts)


# --------------------------------------------------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------------------------------------------------


def planned_total(entry: dict[str, Any]) -> int:
    if "pairs" in entry:
        return int(entry["pairs"]) * len(entry["conditions"])
    return (
        int(entry.get("anchors_zero", 0))
        + int(entry.get("anchors_far", 0))
        + sum(int(s["trials"]) for s in entry["staircases"])
    )


def nearest_as_the_runner_does(
    planned: PlannedTrial, placed: Point, demo_starts: tuple[DemoStart, ...]
) -> tuple[int, float]:
    """The trial runner's own rule for a record's nearest demonstration (`trial_session.nearest_for_record`)."""
    demos = geometry.trained_demos(demo_starts, planned.dataset)
    if planned.condition == "replay":
        demo = next(d for d in demos if d.episode_index == planned.replay_episode)
        return demo.episode_index, placed.distance_to(demo.measured)
    index, distance = geometry.nearest(placed, demos)
    return demos[index].episode_index, float(distance)


class _Clock:
    """Times for the records: each trial some seconds long, some minutes after the last."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def trial(self, seconds: int, pause_s: int) -> tuple[str, str]:
        start = self.now
        end = start + timedelta(seconds=seconds)
        self.now = end + timedelta(seconds=pause_s)
        return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


def checkpoint_name(condition: str, run: str = "v1", step: str = "080000") -> str:
    """A policy's checkpoint as the trial runner names it when the folders show the run: `<run>@<step>`."""
    return f"act-tolrad-{schema.DATASET_OF_CONDITION[condition]}-{run}@{step}"


def achieved_hz(trial_id: str, requested_hz: float, seed: int = 20261005) -> float:
    """The rate the synthetic control loop ran at in one trial: a little under what was asked (`ACHIEVED_SHARE`)."""
    low, high = ACHIEVED_SHARE
    draw = float(derived_rng(seed, "synthetic-loop", trial_id).random())
    return requested_hz * (low + (high - low) * draw)


def make_record(
    planned: PlannedTrial,
    protocol: dict[str, Any],
    demo_starts: tuple[DemoStart, ...],
    *,
    truths: dict[str, Truth],
    scale: float,
    seed: int,
    clock: _Clock,
    void_reason: str | None,
) -> TrialRecord:
    """The record a trial would leave, with the fields the trial runner writes (`trial_session.build_record`).

    Its `execution` is in the runner's final form: every setting of the protocol's `[execution]` section as the
    session ran it, then what belongs to the one trial (`PER_TRIAL_KEYS`). Among those, `loop_s` is the time from
    the first step sent to the last, at the rate `achieved_hz` gives for the trial.
    """
    u = derived_rng(seed, "synthetic-trial", planned.trial_id).random(6).tolist()
    radius, angle = 1.6 * math.sqrt(u[0]), 2.0 * math.pi * u[1]
    placed = Point(
        round(planned.target.x_mm + radius * math.cos(angle), 2),
        round(planned.target.y_mm + radius * math.sin(angle), 2),
    )
    stated = protocol["execution"]
    replay = planned.condition == "replay"
    seconds = 12 + int(20 * u[4])
    ts_start, ts_end = clock.trial(seconds, 70 + int(80 * u[5]))
    steps = int(stated["control_hz"] * (seconds - 3))
    if void_reason == HANDOFF_REFUSED:
        steps = 0
    elif void_reason is not None and void_reason.startswith("ended by a key"):
        steps = round(stated["control_hz"] * float(void_reason.split()[4]))
    elif void_reason is None and planned.trial_id == TOO_SHORT_TRIAL:
        steps = 1
    loop_s = round((steps - 1) / achieved_hz(planned.trial_id, stated["control_hz"], seed), 3) if steps >= 2 else 0.0
    wobble = derived_rng(seed, "synthetic-start-pose", planned.trial_id).random(len(START_POSE_DEG)).tolist()
    crept = float(derived_rng(seed, "synthetic-view-crept", planned.trial_id).random())
    common = dict(
        planned=planned,
        ts_start=ts_start,
        ts_end=ts_end,
        placed=placed,
        operator="op1",
        scorer="op1",
        policy_repo=None if replay else planned.condition,
        checkpoint=None if replay else checkpoint_name(planned.condition),
        execution={
            **stated,
            "steps_sent": steps,
            "loop_s": loop_s,
            "device": "mps",
            "handoff_tolerance_used_deg": stated["replay_handoff_tolerance_deg" if replay else "handoff_tolerance_deg"],
            "start_pose_deg": {
                joint: round(value + 3.0 * (wobble[index] - 0.5), 2)
                for index, (joint, value) in enumerate(START_POSE_DEG.items())
            },
            "view_crept_mm": round(VIEW_CREPT_MM[0] + (VIEW_CREPT_MM[1] - VIEW_CREPT_MM[0]) * crept, 2),
        },
        versions={
            "project_commit": "0123abc456",
            "protocol_version": str(protocol["protocol"]["version"]),
            "protocol_status": str(protocol["protocol"]["status"]),
            "lerobot101_commit": "02b3ced82f",
            "lerobot": "lerobot 0.6.1 at 6adf515116",
        },
        media={"front": f"data/recordings/20261007_090000/20261007_090000_{planned.trial_id}_front.mp4"},
    )
    if void_reason is not None:
        return TrialRecord(outcome="void", void_reason=void_reason, **common)
    episode, distance = nearest_as_the_runner_does(planned, placed, demo_starts)
    truth = truths[planned.condition]
    chance = float(success_probability(distance / scale, truth.r50_mm, truth.width_mm, truth.lapse, truth.floor))
    success = u[2] < chance
    subgoal = schema.SUBGOAL_MAX if success else int(u[3] * schema.SUBGOAL_MAX)
    labels = ("perception", "grasp", "grasp", "transport", "no_recovery")
    return TrialRecord(
        outcome="success" if success else "failure",
        subgoal=subgoal,
        failure=None if success else labels[subgoal],
        notes="",
        nearest_demo_episode=episode,
        nearest_demo_distance_mm=distance,
        **common,
    )


def run_trials(
    protocol: dict[str, Any],
    demo_starts: tuple[DemoStart, ...],
    *,
    blocks: tuple[str, ...] = ALL_BLOCKS,
    truths: dict[str, Truth] | None = None,
    scale: float = 1.0,
    void_every: int = 11,
    held_out_pairs: int = 0,
    seed: int = 20261005,
) -> list[TrialRecord]:
    """Every record of a synthetic experiment, in the order run.

    The two first blocks are interleaved in one seeded order, as the session at the rig interleaves them. Then the
    workspace test as the planner plans it, and after it, if asked, `held_out_pairs` more pairs built by hand at
    placements drawn over the held-out cells (pairs that no plan foresaw). Then the spread policy's block, on the
    next day. Every `void_every`-th trial is void, and the planner asks for its place again.
    """
    truths = truths or TRUTHS
    log: list[TrialRecord] = []
    clock = _Clock(datetime(2026, 10, 7, 9, 0, 0, tzinfo=_ZONE))
    limit = float(protocol["execution"].get("min_attempt_s", 1.0))

    def run(planned: PlannedTrial) -> None:
        void = void_every > 0 and len(log) % void_every == void_every - 1
        reason = VOID_REASONS[(len(log) // void_every) % len(VOID_REASONS)].format(limit=limit) if void else None
        log.append(
            make_record(
                planned, protocol, demo_starts, truths=truths, scale=scale, seed=seed, clock=clock, void_reason=reason
            )
        )

    together = [block for block in ("T1a", "T1b") if block in blocks]
    labels = [block for block in together for _ in range(planned_total(protocol["blocks"][block]))]
    order = shuffled(labels, derived_rng(seed, "synthetic-block-order"))
    while together:
        done = sum(record.scored and record.planned.block in together for record in log)
        due = dict.fromkeys(order[done:] + order[:done])
        planned = next(
            (p for block in due if (p := planner.next_trial(block, protocol, demo_starts, log)) is not None), None
        )
        if planned is None:
            break
        run(planned)

    if "T2" in blocks:
        clock.now = datetime(2026, 10, 7, 13, 30, 0, tzinfo=_ZONE)
        while (planned := planner.next_trial("T2", protocol, demo_starts, log)) is not None:
            run(planned)
        grid = Grid.from_protocol(protocol)
        first_pair = int(protocol["blocks"]["T2"]["pairs"]) + 1
        held_out = targets.uniform_placements(
            grid, held_out_pairs, derived_rng(seed, "synthetic-held-out"), "held_out_cells"
        )
        for index, target in enumerate(held_out):
            policies = ["act_clustered", "act_spread"] if index % 2 == 0 else ["act_spread", "act_clustered"]
            for policy in policies:
                while True:  # a void trial is asked again, as the planner would
                    number = sum(record.planned.block == "T2" for record in log) + 1
                    planned = PlannedTrial(
                        trial_id=f"T2-{number:03d}",
                        block="T2",
                        condition=policy,
                        blind_code=f"X{number:03d}",
                        target=target,
                        pair_id=f"T2-p{first_pair + index:02d}",
                    )
                    run(planned)
                    if log[-1].scored:
                        break

    if "T3" in blocks:
        clock.now = datetime(2026, 10, 8, 9, 15, 0, tzinfo=_ZONE)
        while (planned := planner.next_trial("T3", protocol, demo_starts, log)) is not None:
            run(planned)
    return log


# --------------------------------------------------------------------------------------------------------------------
# The files around the trials
# --------------------------------------------------------------------------------------------------------------------


def write_position_csv(path: Path, label: str, points: list[tuple[float, float]]) -> None:
    """A CSV as `scripts/rig/measure.py log` writes one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "x_mm", "y_mm", "scatter_mm", "frames", "time"])
        for index, (x, y) in enumerate(points):
            writer.writerow([index, label, f"{x:.2f}", f"{y:.2f}", "0.21", 8, f"2026-10-06T08:{10 + index:02d}:00"])


def write_scale_runs(folder: Path, factors: tuple[float, ...], *, readings: int = 6, step_mm: float = 10.0) -> None:
    """One run along the ruler for each factor: readings `step_mm` apart that the camera reads long by the factor."""
    runs = (
        ("ruler_x", 60.0, 70.0, 1, 0),
        ("ruler_y", 125.0, 30.0, 0, 1),
        ("ruler_x_corner", 20.0, 20.0, 1, 0),
        ("ruler_y_corner", 230.0, 90.0, 0, 1),
    )
    for index, measured in enumerate(factors):
        label, x0, y0, along_x, along_y = runs[index % len(runs)]
        name = label if index < len(runs) else f"{label}_again{index}"
        points = [
            (x0 + along_x * step * step_mm * measured, y0 + along_y * step * step_mm * measured)
            for step in range(readings)
        ]
        write_position_csv(folder / f"{name}_20261006_08{10 + index:02d}00.csv", name, points)


def write_repeat_set(folder: Path, seed: int = 20261005, count: int = 10) -> None:
    rng = derived_rng(seed, "synthetic-repeat")
    points = [(125.0 + 0.5 * (a - 0.5), 75.0 + 0.5 * (b - 0.5)) for a, b in rng.random((count, 2)).tolist()]
    write_position_csv(folder / "repeat_20261006_082500.csv", "repeat", points)


def write_by_eye(folder: Path, label: str = "op1", count: int = 20, seed: int = 20261005) -> list[float]:
    """A CSV as `scripts/rig/measure.py byeye` writes one. Returns the errors as the file states them, in mm.

    The tool takes every reference picture first and has the cube put back by eye in another order: a row carries
    the index of its reference, its place in that order, and the seconds since its reference was taken.
    """
    rng = derived_rng(seed, "synthetic-by-eye", label)
    errors = []
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"byeye_{label}_20261007_160000.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "label",
                "target_x_mm",
                "target_y_mm",
                "reference_x_mm",
                "reference_y_mm",
                "eye_x_mm",
                "eye_y_mm",
                "error_mm",
                "seconds",
                "order",
                "since_reference_s",
            ]
        )
        draws = rng.random((count, 6)).tolist()
        order = shuffled(range(count), derived_rng(seed, "synthetic-by-eye-order", label))
        for position, index in enumerate(order, start=1):
            u = draws[index]
            target = (30.0 + 190.0 * u[0], 20.0 + 110.0 * u[1])
            reference = (round(target[0] + u[2] - 0.5, 2), round(target[1] + u[3] - 0.5, 2))
            off, angle = 1.5 + 12.0 * u[4] ** 2, 2.0 * math.pi * u[5]
            eye = (round(reference[0] + off * math.cos(angle), 2), round(reference[1] + off * math.sin(angle), 2))
            error = math.hypot(eye[0] - reference[0], eye[1] - reference[1])
            errors.append(error)
            writer.writerow(
                [
                    index,
                    label,
                    f"{target[0]:.2f}",
                    f"{target[1]:.2f}",
                    f"{reference[0]:.2f}",
                    f"{reference[1]:.2f}",
                    f"{eye[0]:.2f}",
                    f"{eye[1]:.2f}",
                    f"{error:.2f}",
                    "14.2",
                    position,
                    f"{40 * (count - index) + 25 * position}",
                ]
            )
    return errors


def takes_before_saving(episode_index: int) -> int:
    """How often a synthetic episode was recorded again before it was saved: twice for episode 7, once for some."""
    return 2 if episode_index == 7 else int(episode_index % 9 == 4)


def write_placement_log(folder: Path, dataset: str, starts: tuple[DemoStart, ...], *, smoke: int = 0) -> None:
    """The recorder's placement log of a set: one line per saved episode, after `smoke` lines of a throwaway test."""
    folder.mkdir(parents=True, exist_ok=True)
    planned = {
        episode.episode_index: episode
        for episode in layouts.read_layout(REPO / "config" / "layouts" / f"{dataset}.json")
    }
    moment = datetime(2026, 10, 6, 9, 0, 0, tzinfo=_ZONE)
    lines = []

    def line(start: DemoStart, session: str, rerecords: int, saved_at: datetime) -> dict[str, Any]:
        episode = planned[start.episode_index]
        return {
            "dataset": dataset,
            "episode_index": start.episode_index,
            "round": episode.round,
            "sitting": episode.sitting,
            "cell": episode.cell,
            "target": {"x_mm": episode.target.x_mm, "y_mm": episode.target.y_mm},
            "placed": {"x_mm": start.measured.x_mm, "y_mm": start.measured.y_mm},
            "placement_error_mm": start.measured.distance_to(episode.target),
            "rerecords": rerecords,
            "saved_at": saved_at.isoformat(timespec="seconds"),
            "session": session,
        }

    for start in starts[:smoke]:
        lines.append(line(start, SMOKE_SESSION, 0, moment - timedelta(minutes=30)))
    for start in starts:
        moment += timedelta(seconds=75)
        session = f"20261006_09{planned[start.episode_index].sitting:02d}00"
        lines.append(line(start, session, takes_before_saving(start.episode_index), moment))
    (folder / f"{dataset}_placements.jsonl").write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in lines), encoding="utf-8"
    )


def decisions_text(deviations: int = 0) -> str:
    """A decision log in the repository's form, with some rows marked as deviations."""
    rows = [
        "| 2026-10-05 | Far anchors at 40 mm, not 50 | At 50 mm nearly every point is in a corner cell | op1 |",
        "| 2026-10-05 | A placement is accepted only on the operator's key | An arm moves next | op1 |",
    ]
    for index in range(deviations):
        rows.insert(0, f"| 2026-10-07 | **DEVIATION** {index + 1}: block T3 was cut short | No time left | user |")
    head = (
        "# Decisions\n\nWhat was decided, when, by whom, and why. Newest first. A deviation from the frozen protocol "
        "is recorded here and marked\n**DEVIATION**.\n\n| Date | Decision | Why | By |\n|---|---|---|---|\n"
    )
    return head + "\n".join(rows) + "\n"


def scene_audit(records: tuple[TrialRecord, ...], log_name: str = "main.jsonl", seed: int = 20261005) -> dict[str, Any]:
    """The scene audit of a log, in the shape `scripts/rig/scene_audit.py` writes (`scene_audit.audit_log`): one
    line for every record, void ones too, with invented readings.

    Most trials began with the basket and the wrist camera inside their limits and left the basket where it was.
    Some did not (`SCENE_BASKET_NAMED_EVERY` and the positions named beside it). The flags are worked out from the
    readings by the audit's own rules: named beyond a limit across or down, moved beyond `SCENE_BASKET_MOVED_MM`.
    """

    def far(reading: dict[str, Any] | None, keys: tuple[str, str], limit: float) -> bool | None:
        if reading is None:
            return None
        return max(abs(reading[key]) for key in keys if reading[key] is not None) > limit

    rows = []
    scored_before = 0
    for index, record in enumerate(records):
        u = derived_rng(seed, "synthetic-scene", record.planned.trial_id).random(8).tolist()
        at = scored_before if record.scored else None
        scored_before += record.scored
        basket = {"across_mm": round(1.2 * u[0] - 0.6, 1), "down_mm": round(0.2 + 1.7 * u[1], 1), "lean": 0.1}
        if index % SCENE_BASKET_NAMED_EVERY == SCENE_BASKET_NAMED_EVERY - 1:
            basket["down_mm"] = round(2.1 + 0.3 * u[1], 1)
        wrist = {"across_px": round(2.0 * u[2] - 1.0, 1), "down_px": round(1.8 * u[3], 1), "by": "hole"}
        if at == SCENE_WRIST_NAMED_AT:
            wrist.update(across_px=-2.4)
        ended = {"across_mm": round(basket["across_mm"] + 0.1, 1), "down_mm": basket["down_mm"], "lean": 0.1}
        if at == SCENE_MOVED_AT:
            ended.update(across_mm=round(basket["across_mm"] + 30.0, 1), down_mm=round(basket["down_mm"] + 9.0, 1))
        elif at == SCENE_END_UNSEEN_AT:
            ended = None
        elif at == SCENE_FRONT_UNREAD_AT:
            basket = ended = None
        elif at == SCENE_WRIST_UNREAD_AT:
            wrist = None
        moved = None
        if basket is not None and ended is not None:
            moved = round(
                math.hypot(ended["across_mm"] - basket["across_mm"], ended["down_mm"] - basket["down_mm"]), 1
            )
        rows.append(
            {
                "trial_id": record.planned.trial_id,
                "condition": record.planned.condition,
                "outcome": record.outcome,
                "session": "20261007_090000",
                "basket_start": basket,
                "basket_end": ended,
                "wrist_start": wrist,
                "wrist_end": wrist,
                "basket_moved_mm": moved,
                "basket_was_moved": None if moved is None else moved > SCENE_BASKET_MOVED_MM,
                "named_basket": far(basket, ("across_mm", "down_mm"), SCENE_BASKET_NAMED_MM),
                "named_wrist": far(wrist, ("across_px", "down_px"), SCENE_WRIST_NAMED_PX),
            }
        )
    return {
        "log": log_name,
        "basket_named_mm": SCENE_BASKET_NAMED_MM,
        "wrist_named_px": SCENE_WRIST_NAMED_PX,
        "trials": rows,
    }


def write_scene_audit(folder: Path, audit: dict[str, Any]) -> Path:
    """Write a scene audit where the tool writes it, beside the start-state audits, in the bytes it writes."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"scene_{Path(audit['log']).stem}.json"
    path.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    return path


def second_scoring_agreement(
    records: tuple[TrialRecord, ...], log: Path, root: Path, *, scorer: str = SECOND_SCORER, rehearsal: bool = False
) -> dict[str, Any]:
    """The agreement of a synthetic second scoring of a log, in the shape `scripts/second_scoring.py compare` writes
    (`second_scoring.compare`), with its counts worked out here from the records.

    Of each condition of each block, in the order of the log, every `SECOND_ONE_IN`-th scored trial is drawn. The
    second score is the first one, except at two positions of the draw (`SECOND_DIFFERS_AT`, `SECOND_UNTOLD_AT`).
    The digest is that of the log's own bytes, as a real draw's is.
    """
    groups: dict[tuple[str, str], list[TrialRecord]] = {}
    for record in records:
        if record.scored:
            groups.setdefault((record.planned.block, record.planned.condition), []).append(record)
    drawn = [record for found in groups.values() for record in found[::SECOND_ONE_IN]]
    other = {"success": "failure", "failure": "success"}
    rows = []
    for position, record in enumerate(drawn):
        second = record.outcome
        if position == SECOND_DIFFERS_AT:
            second = other[record.outcome]
        elif position == SECOND_UNTOLD_AT:
            second = "cannot_tell"
        kind = "replay" if record.planned.condition == "replay" else "policy"
        rows.append((record, kind, second))

    def tally(found: list[tuple[TrialRecord, str, str]]) -> dict[str, Any]:
        agree = sum(record.outcome == second for record, _, second in found)
        untold = sum(second == "cannot_tell" for _, _, second in found)
        compared = len(found) - untold
        low, high = wilson_interval(agree, compared, 0.95) if compared else (None, None)
        return {
            "n_items": len(found),
            "n_agree": agree,
            "n_disagree": compared - agree,
            "n_cannot_tell": untold,
            "n_compared": compared,
            "agreement": float(f"{agree / compared:.9g}") if compared else None,
            "wilson_lo": None if low is None else float(f"{low:.9g}"),
            "wilson_hi": None if high is None else float(f"{high:.9g}"),
            "table": {
                f"{first}_{second}": sum(record.outcome == first and given == second for record, _, given in found)
                for first in other
                for second in other
            },
            "cannot_tell_first": {
                first: sum(record.outcome == first and given == "cannot_tell" for record, _, given in found)
                for first in other
            },
        }

    def listed(record: TrialRecord, kind: str, second: str, item: int) -> dict[str, Any]:
        return {
            "trial_id": record.planned.trial_id,
            "item": f"item-{item:02d}",
            "block": record.planned.block,
            "condition": record.planned.condition,
            "kind": kind,
            "first": record.outcome,
            "first_subgoal": record.subgoal,
            "first_failure": record.failure,
            "second": second,
            "watched_again": 0,
        }

    entries = sorted(
        (listed(record, kind, second, item) for item, (record, kind, second) in enumerate(rows, start=1)),
        key=lambda entry: entry["trial_id"],
    )
    name = log.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha256(hashlib.sha256(log.read_bytes()).hexdigest().encode("ascii")).hexdigest()
    return {
        "schema_version": 1,
        "rehearsal": rehearsal,
        "provenance": {
            "folder": "data/second_scoring/main",
            "logs": [name],
            "logs_digest": digest,
            "seed": SECOND_SEED,
            "one_in": SECOND_ONE_IN,
            "views": ["front"],
            "n_scored": sum(len(found) for found in groups.values()),
            "n_drawn": len(drawn),
            "groups": [
                {"block": block, "condition": condition, "n_scored": len(found), "n_drawn": len(found[::SECOND_ONE_IN])}
                for (block, condition), found in groups.items()
            ],
            "passed_over": [],
            "whole_pairs": [],
        },
        "scorers": {
            "first": sorted({record.scorer for record in drawn}),
            "second": [scorer],
            "n_same_role": sum(record.scorer == scorer for record in drawn),
        },
        "interval_level": 0.95,
        "all": tally(rows),
        "by_kind": {kind: tally([row for row in rows if row[1] == kind]) for kind in ("policy", "replay")},
        "disagreements": [e for e in entries if e["second"] not in ("cannot_tell", e["first"])],
        "cannot_tell": [e for e in entries if e["second"] == "cannot_tell"],
        "watched_again": {"n_items": 0, "total": 0},
    }


def write_second_scoring(folder: Path, agreement: dict[str, Any]) -> Path:
    """Write an agreement where the comparison writes it, in the bytes it writes (`second_scoring.write_agreement`)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "agreement.json"
    path.write_text(json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


#: The truths of the synthetic procedure simulation: every combination of these, for every threshold block.
SIMULATED_R50_MM = (5.0, 20.0, 40.0)
SIMULATED_WIDTH_MM = (1.0, 3.0, 10.0)
SIMULATED_LAPSE = (0.0, 0.05)
#: The row the paper quotes (block T1a as written, radius 20 mm, width 3 mm, lapse 0) and the same truth with fixed
#: distances: bias, RMSE, mean interval width, share of intervals that held the truth.
SIMULATED_QUOTED = {"protocol": (-0.2417, 1.6266, 7.8033, 0.9733), "constant": (0.2667, 2.3177, 9.0083, 0.9433)}


def simulated_row(design: str, block: str, n_trials: int, r50: float, width: float, lapse: float) -> list[str]:
    """One row of the synthetic procedure simulation, with figures invented by a rule a test can follow.

    The error and the interval grow with the width, with the radius and with the lapse, and differ from block to
    block; the interval is open below more often the smaller the radius is against the width.
    """
    quoted = (block, r50, width, lapse) == ("T1a", 20.0, 3.0, 0.0)
    if quoted and design in SIMULATED_QUOTED:
        bias, rmse, span, inside = SIMULATED_QUOTED[design]
    else:
        offset = {"T1a": 0.0, "T1b": 0.3, "T3": 0.2}[block] + (0.6 if design == "constant" else 0.0)
        rmse = 0.4 + 0.4 * width + 0.01 * r50 + 2.0 * lapse + offset
        span = 4.9 * rmse + 0.07 * r50
        bias, inside = -0.01 * width, 0.99 - 0.004 * width
    low = round(max(0.0, 0.3 * (2.0 * width - r50) / width + 2.0 * lapse), 4) if r50 < 2.0 * width else 0.0
    high = 0.02 * width if r50 > 30.0 else 0.0
    cells = [bias, rmse, span, inside, low, high, min(1.0, low + high)]
    return [design, block, str(n_trials), f"{r50:g}", f"{width:g}", f"{lapse:g}", "300", *(f"{c:.4f}" for c in cells)]


def simulated_rule_shares(width: float, difference: float) -> tuple[float, float, float]:
    """The shares of the three verdicts in the synthetic comparison-rule table: "larger" more often the larger the
    difference is against the width, a verdict either way in a few pairs when the radii are equal."""
    larger = round(min(1.0, 0.03 + 0.19 * difference / width), 4)
    smaller = 0.02 if difference == 0.0 else 0.0
    return larger, round(1.0 - larger - smaller, 4), smaller


def write_rule_simulation(folder: Path, protocol_file: Path, *, stale: bool = False) -> None:
    """The table and the report of a simulated comparison rule, in the form `tolerance_radius.compare_power` writes
    them, with invented figures. The report names the protocol file by its hash; with `stale`, another one."""
    folder.mkdir(parents=True, exist_ok=True)
    named = "0123456789ab" if stale else hashlib.sha256(protocol_file.read_bytes()).hexdigest()[:12]
    (folder / "comparison_rule.md").write_text(
        "# The comparison rule: what two blocks of trials can tell apart\n\n**Every number in this report is "
        "simulated. Nothing here is a measurement.**\n\nWritten by `python -m tolerance_radius.compare_power`. Seed "
        "1; 300 simulated pairs of\nblocks for every row. Protocol file `protocol.toml`: version 1.0, status frozen, "
        f"SHA-256\n`{named}`.\n",
        encoding="utf-8",
    )
    lines = [
        "policy_block,replay_block,replay_r50_mm,policy_r50_mm,difference_true_mm,width_mm,lapse,pairs,n_larger,"
        "n_not_told_apart,n_smaller,share_larger,share_not_told_apart,share_smaller,mean_difference_mm,"
        "mean_interval_width_mm,interval_coverage"
    ]
    for width in (1.0, 3.0, 6.0):
        for difference in (0.0, 3.0, 5.0, 8.0, 10.0, 15.0):
            larger, between, smaller = simulated_rule_shares(width, difference)
            counts = [round(300 * share) for share in (larger, between, smaller)]
            cells = ["T1a", "T1b", "15", f"{15 + difference:g}", f"{difference:g}", f"{width:g}", "0", "300", *counts]
            cells += [f"{larger:.4f}", f"{between:.4f}", f"{smaller:.4f}"]
            cells += [f"{difference:.4f}", f"{7 * width:.4f}", "0.9700"]
            lines.append(",".join(str(cell) for cell in cells))
    (folder / "comparison_rule.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simulation(
    folder: Path, protocol_file: Path, pairs: int, *, stale: bool = False, rule: str | None = "fresh"
) -> None:
    """The tables of a procedure simulation, in the form `tolerance_radius.simulate` writes, with invented figures,
    and beside them those of the comparison rule (`write_rule_simulation`).

    Each report names the protocol file it was made from; with `stale` the procedure simulation names another one.
    `rule` says the same of the comparison rule: "fresh", "stale", or None for no such table.
    """
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(protocol_file.read_bytes()).hexdigest()[:12]
    named = "0123456789ab" if stale else digest
    for name in ("comparison_rule.csv", "comparison_rule.md"):
        (folder / name).unlink(missing_ok=True)
    if rule is not None:
        write_rule_simulation(folder, protocol_file, stale=rule == "stale")
    (folder / "procedure.md").write_text(
        "# Procedure simulation\n\nWritten by `python -m tolerance_radius.simulate`. Seed 1; 300 simulated blocks for "
        f"every row. Protocol file\n`protocol.toml`: version 1.0, status frozen, SHA-256 `{named}`.\n",
        encoding="utf-8",
    )
    header = (
        "design,block,n_trials,r50_true_mm,width_true_mm,lapse_true,replications,bias_mm,rmse_mm,"
        "mean_interval_width_mm,interval_coverage,lower_edge_rate,upper_edge_rate,any_edge_rate"
    )
    blocks = tomllib.loads(protocol_file.read_text(encoding="utf-8"))["blocks"]
    sweep = [header]
    for design in ("protocol", "constant"):
        for block, entry in blocks.items():
            if "staircases" not in entry:
                continue
            for r50 in SIMULATED_R50_MM:
                for width in SIMULATED_WIDTH_MM:
                    for lapse in SIMULATED_LAPSE:
                        sweep.append(",".join(simulated_row(design, block, planned_total(entry), r50, width, lapse)))
    (folder / "procedure.csv").write_text("\n".join(sweep) + "\n", encoding="utf-8")
    power = [
        "pairs,p_first,p_second,power",
        f"{pairs},0.4,0.8,0.600000",
        f"{pairs},0.4,0.9,0.879086",
        f"{pairs + 4},0.4,0.9,0.930000",
    ]
    (folder / "paired_power.csv").write_text("\n".join(power) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------------
# The whole experiment
# --------------------------------------------------------------------------------------------------------------------


@cache
def _records(
    status: str,
    version: str,
    coarse: bool,
    workspace: tuple[tuple[str, int], ...] | None,
    blocks: tuple[str, ...],
    scale: float,
    held_out_pairs: int,
    void_every: int,
    seed: int,
) -> tuple[TrialRecord, ...]:
    """The trials of an experiment, worked out once for each set of arguments however many folders they go to."""
    protocol = tomllib.loads(protocol_text(status=status, version=version, coarse=coarse, workspace=workspace))
    demo_starts = (*audited_starts("clustered"), *audited_starts("spread"))
    return tuple(
        run_trials(
            protocol,
            demo_starts,
            blocks=blocks,
            scale=scale,
            void_every=void_every,
            held_out_pairs=held_out_pairs,
            seed=seed,
        )
    )


@cache
def _pilot_records(seed: int) -> tuple[TrialRecord, ...]:
    """Three trials of a pilot session: run before the freeze, under the pilot's own version and status."""
    protocol = tomllib.loads(protocol_text(status="pilot", version="pilot"))
    demo_starts = (*audited_starts("clustered"), *audited_starts("spread"))
    log: list[TrialRecord] = []
    clock = _Clock(datetime(2026, 10, 6, 15, 0, 0, tzinfo=_ZONE))
    for _ in range(3):
        planned = planner.next_trial("T1b", protocol, demo_starts, log)
        log.append(
            make_record(
                planned, protocol, demo_starts, truths=TRUTHS, scale=1.0, seed=seed, clock=clock, void_reason=None
            )
        )
    return tuple(log)


def build_experiment(
    root: str | Path,
    *,
    blocks: tuple[str, ...] = ALL_BLOCKS,
    status: str = "frozen",
    version: str = "1.0",
    coarse: bool = False,
    workspace: tuple[tuple[str, int], ...] | None = None,
    sensitivity: bool = True,
    scale_factors: tuple[float, ...] = DEFAULT_SCALE_FACTORS,
    held_out_pairs: int = 0,
    void_every: int = 11,
    by_eye: bool = True,
    repeat: bool = True,
    placement_logs: bool = True,
    simulation: str | None = "fresh",
    rule_simulation: str | None = "as the procedure's",
    deviations: int = 0,
    pilot_log: bool = True,
    second_scorer: str | None = SECOND_SCORER,
    scene: bool = True,
    seed: int = 20261005,
) -> Experiment:
    """Write a whole synthetic experiment below `root` and say what was written (see the module's notes).

    `workspace` and `sensitivity` are those of `protocol_text`; `held_out_pairs` adds pairs built by hand to the
    workspace test (`run_trials`). `simulation` is "fresh" (made from this protocol file), "stale" (made from
    another) or None (no tables). `rule_simulation` says the same of the comparison rule's table, which has a report
    of its own; left alone it is as the procedure simulation is. `second_scorer` is the role code of whoever scored
    a part of the trials again from video (`second_scoring_agreement`), or None for no second scoring. `scene` says
    whether the counted log has a scene audit (`scene_audit`).
    """
    root = Path(root)
    protocol_file = root / "config" / "protocol.toml"
    protocol_file.parent.mkdir(parents=True, exist_ok=True)
    text = protocol_text(status=status, version=version, coarse=coarse, workspace=workspace, sensitivity=sensitivity)
    protocol_file.write_text(text, encoding="utf-8")
    protocol = config.load_protocol(protocol_file)

    demo_starts = (*audited_starts("clustered"), *audited_starts("spread"))
    for dataset in schema.DATASETS:
        schema.write_demo_starts(root / "data" / "audit" / f"{dataset}.json", audited_starts(dataset))
        if placement_logs:
            write_placement_log(
                root / "data" / "recording", dataset, audited_starts(dataset), smoke=3 if dataset == "clustered" else 0
            )

    instrument = root / "data" / "instrument"
    instrument.mkdir(parents=True, exist_ok=True)
    write_scale_runs(instrument, scale_factors)
    agree = len(scale_factors) >= 2 and (max(scale_factors) - min(scale_factors)) <= 0.01 + 1e-9
    scale = sum(scale_factors) / len(scale_factors) if scale_factors else 1.0
    if repeat:
        write_repeat_set(instrument, seed)
    by_eye_errors = tuple(write_by_eye(instrument, seed=seed)) if by_eye else ()

    held_out = held_out_pairs if "T2" in blocks else 0
    records = _records(status, version, coarse, workspace, tuple(blocks), scale, held_out, void_every, seed)
    (root / "data" / "trials").mkdir(parents=True, exist_ok=True)
    for record in records:
        schema.append_trial(root / "data" / "trials" / "main.jsonl", record)
    if pilot_log:
        for record in _pilot_records(seed):
            schema.append_trial(root / "data" / "trials" / "pilot_replay.jsonl", record)

    audit = None
    if scene and records:
        audit = scene_audit(records, seed=seed)
        write_scene_audit(root / "data" / "audit", audit)
    agreement = None
    if second_scorer is not None and any(record.scored for record in records):
        log = root / "data" / "trials" / "main.jsonl"
        agreement = second_scoring_agreement(records, log, root, scorer=second_scorer)
        write_second_scoring(root / "data" / "second_scoring" / "main", agreement)

    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "decisions.md").write_text(decisions_text(deviations), encoding="utf-8")
    if simulation is not None:
        write_simulation(
            root / "analysis" / "simulation",
            protocol_file,
            int(protocol["blocks"]["T2"]["pairs"]),
            stale=simulation == "stale",
            rule=simulation if rule_simulation == "as the procedure's" else rule_simulation,
        )
    return Experiment(
        root=root,
        protocol=protocol,
        records=records,
        demo_starts=demo_starts,
        truths=dict(TRUTHS),
        reads_long_by=scale,
        divisor=scale if agree else 1.0,
        scale_factors=tuple(scale_factors),
        held_out_pairs=held_out_pairs if "T2" in blocks else 0,
        by_eye_errors_mm=by_eye_errors,
        second_scoring=agreement,
        scene=audit,
    )


@cache
def main_run() -> tuple[Experiment, Any]:
    """The whole experiment, written once and analysed once on the protocol's own grid.

    Every test that looks at the results of a full analysis shares this run, because a fit on the full grid takes
    most of the time. Its workspace test is the stratified one of the protocol (`STRATIFIED`), whatever the
    repository's file states today. It is a draft with the synthetic note, so each output says on its first line
    what it is. Returns the experiment and the `tolerance_radius.report.Report` made from it.
    """
    from tolerance_radius import report

    folder = tempfile.mkdtemp(prefix="tolerance-radius-synthetic-")
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    experiment = build_experiment(folder, workspace=STRATIFIED)
    return experiment, report.make_report(report.load_inputs(experiment.root, draft=True, note=SYNTHETIC_NOTE))


EXAMPLE_README = """\
SYNTHETIC. Nothing in this folder is a result: no trial was run to make it.

# An example of the analysis's output

`results.json` and `report.md` here are what `scripts/make_report.py` writes, made from a synthetic experiment:
audits built from the committed layouts, trials planned by the real planner and answered from invented success
curves (`tests/fixtures_experiment.py`). They show the shape of the two files with every block run, for whoever
draws a figure or writes a sentence before the counted trials exist. `protocol/data-contracts.md` explains the fields.

The workspace test of this experiment is the stratified one of the protocol (draft 0.3): 26 placements, 13 drawn
over the trained cells and 13 over the held-out cells, each tested with both policies.

The counted report will be in `analysis/report/`. Do not copy a number from here anywhere.

The numbers are those of the day the example was written: made from synthetic inputs with SHA-256
`{digest}`.
A test checks that the example has the shape the contract fixes today. When today's synthetic inputs are still the
ones named above, it also checks that the analysis gives these very bytes again; when the planner or the protocol
file has changed them, it does not. To write the example again:

    PYTHONPATH=src python tests/fixtures_experiment.py
"""


def inputs_digest(root: str | Path) -> str:
    """The SHA-256 of everything an experiment's folder holds: every file's name and bytes, in name order."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def example_made_from(folder: str | Path) -> str | None:
    """The digest of the synthetic inputs that an example folder says it was made from; None when it names none."""
    found = re.search(r"SHA-256\s+`([0-9a-f]{64})`", (Path(folder) / "README.md").read_text(encoding="utf-8"))
    return found.group(1) if found else None


def write_example(folder: str | Path | None = None) -> list[Path]:
    """Write the example of the analysis's output (`analysis/report-example/`) from the shared full run."""
    folder = Path(folder) if folder is not None else REPO / "analysis" / "report-example"
    experiment, made = main_run()
    folder.mkdir(parents=True, exist_ok=True)
    texts = {
        "README.md": EXAMPLE_README.format(digest=inputs_digest(experiment.root)),
        "results.json": made.results_json,
        "report.md": made.report_md,
    }
    for name, text in texts.items():
        (folder / name).write_text(text, encoding="utf-8")
    return [folder / name for name in texts]


def rewrite_log(path: Path, change) -> None:
    """Rewrite a trial log with `change(index, entry)` applied to each line's data. For tests of what is refused:
    a real log is never rewritten."""
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = [change(index, entry) or entry for index, entry in enumerate(entries)]
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in changed), encoding="utf-8")


if __name__ == "__main__":
    for written in write_example():
        print(f"wrote {written.relative_to(REPO)}")
