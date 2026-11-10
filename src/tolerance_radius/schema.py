"""Data contracts: the records this project writes at the rig and reads in the analysis.

Each record is defined once, here. The tools at the rig write them, the planner reads them back to decide the next
trial, and the analysis reads them to produce every number in the paper. `protocol/data-contracts.md` explains the fields in
prose; this module is the authority when the two disagree.

Conventions
-----------
* A position is where the front camera measures the cube to be, in the rig's table frame, in millimetres: the centre of
  the cube's visible outline mapped through the table plane. That point is offset from the centre of the cube's
  footprint by a few millimetres; the offset is shared by every measurement, so it cancels in distances (see
  `protocol/protocol.md`, section 7, for how well).
* A name ending in `_mm`, `_s`, `_deg` or `_hz` carries that unit.
* Times are ISO 8601 strings with a UTC offset.
* A trial log is a JSON Lines file: one `TrialRecord` per line, appended, never rewritten.

Standard library only, so that it imports in every environment the project uses.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

#: What produced the motion in a trial.
CONDITIONS = ("act_clustered", "act_spread", "replay")
#: Which demonstration set a condition's distances are measured against.
DATASET_OF_CONDITION = {"act_clustered": "clustered", "act_spread": "spread", "replay": "clustered"}
DATASETS = ("clustered", "spread")
#: Blocks of the protocol (`protocol/protocol.md`).
BLOCKS = ("T1a", "T1b", "T2", "T3")
OUTCOMES = ("success", "failure", "void")
#: The graded scale runs from 0 to 5; a trial is a success only at the top of it.
SUBGOAL_MAX = 5


class SchemaError(ValueError):
    """A record does not satisfy its contract."""


@dataclass(frozen=True, slots=True)
class Point:
    """A position on the table as the front camera measures it, table frame, millimetres."""

    x_mm: float
    y_mm: float

    def __post_init__(self) -> None:
        for name in ("x_mm", "y_mm"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise SchemaError(f"Point.{name} must be a finite number, got {value!r}")

    def distance_to(self, other: Point) -> float:
        """Straight-line distance to another point, in millimetres."""
        return math.hypot(self.x_mm - other.x_mm, self.y_mm - other.y_mm)


@dataclass(frozen=True, slots=True)
class DemoStart:
    """Where the cube started in one demonstration episode.

    `measured` comes from the episode's own first frame (the start-state audit), not from the plan: distances in the
    analysis are always taken to measured positions.
    """

    dataset: str
    episode_index: int
    cell: str
    measured: Point
    planned: Point | None = None
    #: False for an episode that was recorded but left out of training.
    trained: bool = True

    def __post_init__(self) -> None:
        _require(self.dataset in DATASETS, f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        _require(isinstance(self.episode_index, int) and self.episode_index >= 0, "episode_index must be >= 0")
        _require(isinstance(self.cell, str) and self.cell != "", "cell must be a non-empty label")


@dataclass(frozen=True, slots=True)
class PlannedTrial:
    """What the planner asks for before a trial runs. It becomes the first part of the trial's record."""

    trial_id: str
    block: str
    condition: str
    #: The code shown to whoever scores the trial, in place of the condition.
    blind_code: str
    target: Point
    #: Requested distance from the target to the nearest demonstration; None when the placement was drawn uniformly.
    level_mm: float | None = None
    staircase_id: str | None = None
    #: Position of this trial in its staircase, from 0.
    staircase_index: int | None = None
    #: Shared by the two trials that test the same placement with two policies.
    pair_id: str | None = None
    #: The demonstration whose recorded actions are replayed (replay condition only).
    replay_episode: int | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.trial_id, str) and self.trial_id != "", "trial_id must be a non-empty string")
        _require(self.block in BLOCKS, f"block must be one of {BLOCKS}, got {self.block!r}")
        _require(self.condition in CONDITIONS, f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        _require(isinstance(self.blind_code, str) and self.blind_code != "", "blind_code must be a non-empty string")
        if self.level_mm is not None:
            _require(math.isfinite(self.level_mm) and self.level_mm >= 0, "level_mm must be >= 0")
        _require(
            (self.staircase_id is None) == (self.staircase_index is None),
            "staircase_id and staircase_index are given together or not at all",
        )
        if self.condition == "replay":
            _require(self.replay_episode is not None, "a replay trial names the episode it replays")
        else:
            _require(self.replay_episode is None, "only a replay trial has replay_episode")

    @property
    def dataset(self) -> str:
        """The demonstration set this trial's distances are measured against."""
        return DATASET_OF_CONDITION[self.condition]


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One evaluation trial: what was planned, where the cube really was, what ran, and what happened."""

    planned: PlannedTrial
    #: "success", "failure" or "void". A void trial is kept in the log and repeated, never deleted.
    outcome: str
    ts_start: str
    ts_end: str
    #: Where the cube was when the placement was accepted, as measured by the camera. None only for a void trial that
    #: ended before the placement.
    placed: Point | None = None
    #: The demonstration nearest to `placed`, and its distance, in the condition's own demonstration set.
    nearest_demo_episode: int | None = None
    nearest_demo_distance_mm: float | None = None
    #: Graded progress, 0 to `SUBGOAL_MAX`. None for a void trial.
    subgoal: int | None = None
    #: A label from the failure taxonomy; None on success.
    failure: str | None = None
    void_reason: str | None = None
    notes: str = ""
    #: Role codes, not names (the review is double blind): for example "op1", "sc1".
    operator: str = ""
    scorer: str = ""
    #: The policy's repository id and checkpoint; both None for a replay.
    policy_repo: str | None = None
    checkpoint: str | None = None
    #: Settings of the execution loop for this trial (clamp, ensembling, rate, time limit).
    execution: dict[str, Any] = field(default_factory=dict)
    #: Commit ids and the protocol version in force.
    versions: dict[str, str] = field(default_factory=dict)
    #: Paths of the trial's videos, relative to the repository.
    media: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require(self.outcome in OUTCOMES, f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        _require(self.schema_version == SCHEMA_VERSION, f"unsupported schema_version {self.schema_version!r}")
        if self.outcome == "void":
            _require(bool(self.void_reason), "a void trial states its reason")
            _require(self.subgoal is None, "a void trial has no subgoal")
            return
        _require(self.void_reason is None, "only a void trial has void_reason")
        _require(self.placed is not None, "a scored trial records where the cube was placed")
        _require(
            self.nearest_demo_episode is not None and self.nearest_demo_distance_mm is not None,
            "a scored trial records its nearest demonstration and the distance to it",
        )
        _require(
            isinstance(self.subgoal, int) and not isinstance(self.subgoal, bool) and 0 <= self.subgoal <= SUBGOAL_MAX,
            f"subgoal must be an integer from 0 to {SUBGOAL_MAX}",
        )
        _require(
            (self.outcome == "success") == (self.subgoal == SUBGOAL_MAX),
            f"outcome is success exactly when subgoal is {SUBGOAL_MAX}",
        )
        _require((self.failure is None) == (self.outcome == "success"), "failure is named exactly when the trial failed")
        if self.planned.condition == "replay":
            _require(self.checkpoint is None and self.policy_repo is None, "a replay trial has no policy")
        else:
            _require(bool(self.policy_repo) and bool(self.checkpoint), "a policy trial names its repository and checkpoint")

    @property
    def scored(self) -> bool:
        """True for a trial that counts in the analysis (not void)."""
        return self.outcome != "void"

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def placement_error_mm(self) -> float | None:
        """Distance between the target and where the cube was placed."""
        return None if self.placed is None else self.placed.distance_to(self.planned.target)


# --------------------------------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------------------------------


def to_json(record: DemoStart | PlannedTrial | TrialRecord) -> dict[str, Any]:
    """A record as plain JSON-ready data. Points become `{"x_mm": ..., "y_mm": ...}`."""
    return asdict(record)


def demo_start_from_json(data: dict[str, Any]) -> DemoStart:
    data = _known_fields(DemoStart, data)
    data["measured"] = _point(data.get("measured"), "measured")
    if data.get("planned") is not None:
        data["planned"] = _point(data["planned"], "planned")
    return _build(DemoStart, data)


def planned_trial_from_json(data: dict[str, Any]) -> PlannedTrial:
    data = _known_fields(PlannedTrial, data)
    data["target"] = _point(data.get("target"), "target")
    return _build(PlannedTrial, data)


def trial_record_from_json(data: dict[str, Any]) -> TrialRecord:
    data = _known_fields(TrialRecord, data)
    if not isinstance(data.get("planned"), dict):
        raise SchemaError("a trial record carries its plan under 'planned'")
    data["planned"] = planned_trial_from_json(data["planned"])
    if data.get("placed") is not None:
        data["placed"] = _point(data["placed"], "placed")
    return _build(TrialRecord, data)


# --------------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------------


def append_trial(path: str | Path, record: TrialRecord) -> None:
    """Append one trial to a JSON Lines log and force it to disk before returning.

    The log is the experiment's primary data and the planner's memory, so a line is either wholly written or absent.
    """
    line = json.dumps(to_json(record), sort_keys=True, ensure_ascii=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_trials(path: str | Path) -> list[TrialRecord]:
    """Every trial in a log, in the order written. A malformed line is an error, with its line number."""
    records: list[TrialRecord] = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(trial_record_from_json(json.loads(line)))
            except (json.JSONDecodeError, SchemaError, TypeError) as error:
                raise SchemaError(f"{path}:{number}: {error}") from error
    return records


def write_demo_starts(path: str | Path, starts: Iterable[DemoStart]) -> None:
    """Write a start-state audit: one JSON document holding every demonstration's start position."""
    payload = {"schema_version": SCHEMA_VERSION, "demo_starts": [to_json(start) for start in starts]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_demo_starts(path: str | Path) -> list[DemoStart]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"{path}: unsupported schema_version {payload.get('schema_version')!r}")
    return [demo_start_from_json(item) for item in payload["demo_starts"]]


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _point(data: Any, name: str) -> Point:
    if not isinstance(data, dict) or set(data) != {"x_mm", "y_mm"}:
        raise SchemaError(f"{name} must be an object with exactly x_mm and y_mm, got {data!r}")
    return Point(x_mm=data["x_mm"], y_mm=data["y_mm"])


def _known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaError(f"{cls.__name__} must be a JSON object, got {type(data).__name__}")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise SchemaError(f"{cls.__name__} has unknown fields {sorted(unknown)}")
    return dict(data)


def _build(cls: type, data: dict[str, Any]) -> Any:
    try:
        return cls(**data)
    except TypeError as error:
        raise SchemaError(f"{cls.__name__}: {error}") from error
