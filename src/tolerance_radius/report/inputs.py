"""What the report is made from: the files it reads, and everything it refuses.

The analysis reads the counted trial logs, the start-state audits, the scene audits of the logs, the instrument's CSV
files, the recorder's placement logs, the decision log, the tables of the procedure simulation and the agreement of
the second scoring from video.
`load_inputs` reads them all and returns them as plain records, or stops with a `ReportError` whose message is one
plain sentence saying what was refused and why.

What is refused (`protocol/protocol.md`, sections 8 and 10; `protocol/data-contracts.md`):

* a log named `pilot*`: pilot trials are never analysed with the counted trials;
* a record that was not run under a frozen protocol, unless a draft is asked for;
* records of more than one protocol version, or of another version than the protocol file's;
* a scored record whose nearest demonstration and distance are not what the audit gives today;
* scored records of one condition that were not all run with the same execution settings, or that do not carry
  every setting of the protocol's `[execution]` section with the protocol's value;
* scored policy records that do not all name one training step;
* an agreement of the second scoring whose counts do not add up, or that names a scorer otherwise than by a role
  code;
* a scene audit of another log than the one it is read for: one that names a trial the log does not hold, or gives
  a trial another condition or outcome than its record; and one whose own flags are not what its readings and its
  limits give.

The last two are the guard against a repeat of the earlier evaluation, in which one execution setting was off for
part of the trials without anyone noticing.

Paths in the returned records and in every message are relative to the root folder, so that nothing written from them
names a machine or a person.

Standard library and NumPy only. Nothing here imports `tolerance_radius.rig`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tolerance_radius import config, geometry, schema
from tolerance_radius.schema import DemoStart, Point, TrialRecord
from tolerance_radius.second_scoring import AGREEMENT_FILE, KINDS, ROLE_CODE

#: A log whose file name starts with this is a pilot log (the trial runner's own rule).
PILOT_PREFIX = "pilot"
FROZEN = "frozen"
#: A recomputed distance may differ from the recorded one by this much: the rounding of a sum, nothing measured.
DISTANCE_TOLERANCE_MM = 1e-6
#: Entries of a record's `execution` that belong to the one trial: how many actions were sent, how long the control
#: loop ran from the first one sent to the last, the device the policy ran on, the tolerance of the handoff gate the
#: trial went through, the arm's joint positions when the attempt began, and how far the front camera's picture had
#: crept from the session's conversion before the attempt (since 2026-10-07). The trial runner writes them after the
#: session's settings (`trial_session.main`). Everything else in `execution` is a setting and must be the same in
#: every scored trial of a condition.
PER_TRIAL_KEYS = frozenset(
    {"steps_sent", "loop_s", "device", "handoff_tolerance_used_deg", "start_pose_deg", "view_crept_mm"}
)
#: The per-trial entry that holds the tolerance of the handoff gate a trial went through.
HANDOFF_USED_KEY = "handoff_tolerance_used_deg"
#: Separates the run from the training step in a record's `checkpoint` (`act-run-v1@080000`).
RUN_MARK = "@"
#: How many trial names a message lists before it says how many more there are.
_NAMES_SHOWN = 12
_DIGEST = re.compile(r"SHA-256 `([0-9a-f]{12})`")
_DEVIATION = "**DEVIATION**"
#: Stands for a setting that a record does not carry.
_ABSENT = object()
#: The two scores a trial can have at the rig, and the counts every tally of the second scoring's agreement holds
#: (`second_scoring._tally`).
_SCORES = ("success", "failure")
_TALLY_COUNTS = ("n_items", "n_agree", "n_disagree", "n_cannot_tell", "n_compared")
_TALLY_SHARES = ("agreement", "wilson_lo", "wilson_hi")


class ReportError(Exception):
    """An input is refused. The message is one plain sentence: what was refused, and why."""


@dataclass(frozen=True, slots=True)
class CountedTrial:
    """One record of a counted log, with where it was read from."""

    log: str
    #: Position of the record in its log, from 1.
    number: int
    record: TrialRecord

    @property
    def trial_id(self) -> str:
        return self.record.planned.trial_id

    @property
    def condition(self) -> str:
        return self.record.planned.condition

    @property
    def block(self) -> str:
        return self.record.planned.block


@dataclass(frozen=True, slots=True)
class PositionLog:
    """A CSV of logged cube positions: one run along the ruler, or one set of repeated placements."""

    file: str
    label: str
    readings: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class ByEyeReset:
    """One reset by eye: where the cube was for the reference picture, and where it was put back."""

    file: str
    #: Who placed, as a role code.
    label: str
    reference: Point
    eye: Point

    @property
    def error_mm(self) -> float:
        """How far the reset landed from where the cube was for the reference picture, as the camera reads it."""
        return self.eye.distance_to(self.reference)


@dataclass(frozen=True, slots=True)
class RecordedPlacement:
    """One line of the recorder's placement log: a saved episode's planned and measured start."""

    dataset: str
    episode_index: int
    sitting: int
    target: Point
    placed: Point
    #: How many times the episode was recorded again before it was saved.
    rerecords: int
    saved_at: str
    session: str


@dataclass(frozen=True, slots=True)
class Deviation:
    """A row of the decision log that is marked as a deviation from the frozen protocol."""

    date: str
    decision: str
    why: str


@dataclass(frozen=True, slots=True)
class SimulationTables:
    """The tables of the procedure simulation, as text, and the protocol file they were made from."""

    #: The first 12 hex digits of the SHA-256 of the protocol file the simulation read; None if its report does not say.
    protocol_digest: str | None
    sweep: tuple[Mapping[str, str], ...]
    power: tuple[Mapping[str, str], ...]
    files: tuple[str, ...]
    #: The table of the simulation of the comparison rule (`comparison_rule.csv`) and the digest its own report
    #: names; None where that table, or the digest, is not there.
    rule: tuple[Mapping[str, str], ...] | None = None
    rule_digest: str | None = None


@dataclass(frozen=True, slots=True)
class SceneReading:
    """One trial's line of a scene audit: where the basket and the wrist camera stood when the trial began, and
    whether the trial moved the basket. Each entry is None where the trial's video did not say."""

    trial_id: str
    condition: str | None
    outcome: str | None
    #: The basket against its reference picture at the first frames: millimetres on the floor, across and down.
    basket_start_mm: tuple[float, float] | None
    #: The gripper's fixed jaw in the wrist camera's picture against its reference, at the first frames: pixels
    #: across and down. Down is None when the jaw was read by its edge alone.
    wrist_start_px: tuple[float, float | None] | None
    #: How far the basket stood at the trial's end from where it began, and whether the audit calls that moved.
    basket_moved_mm: float | None
    basket_was_moved: bool | None
    #: Whether the trial began with the basket, or with the wrist camera, beyond the audit's limit from its place.
    named_basket: bool | None
    named_wrist: bool | None

    @property
    def basket_off_mm(self) -> float | None:
        """How far the basket stood from its place at the start: the larger of across and down, the figure the
        audit holds against its limit."""
        return None if self.basket_start_mm is None else max(abs(value) for value in self.basket_start_mm)

    @property
    def wrist_off_px(self) -> float | None:
        """How far the wrist camera's picture stood from its place at the start: the larger of across and down."""
        if self.wrist_start_px is None:
            return None
        return max(abs(value) for value in self.wrist_start_px if value is not None)


@dataclass(frozen=True, slots=True)
class SceneAudit:
    """The scene audit of one trial log, as `scripts/rig/scene_audit.py` wrote it from the trials' own videos."""

    file: str
    #: The file name of the log it was made from.
    log: str
    #: Beyond these at its first frames a trial is named in the report (protocol, section 8).
    basket_named_mm: float
    wrist_named_px: float
    readings: tuple[SceneReading, ...]


@dataclass(frozen=True, slots=True)
class SecondScoring:
    """The agreement of a second scoring from video, as `scripts/second_scoring.py compare` wrote it."""

    file: str
    #: True for a draw made from pilot trials to try the tool: nothing in it is a result.
    rehearsal: bool
    #: The logs the draw was made from, and their digest when it was made (`logs_digest`).
    logs: tuple[str, ...]
    logs_digest: str
    seed: int
    one_in: int
    #: The scored trials the draw was made from, and how many of them were drawn.
    n_scored: int
    n_drawn: int
    interval_level: float
    #: The counts over all items (`all`) and by kind of trial (`policy`, `replay`), as the file gives them.
    tallies: Mapping[str, Mapping[str, Any]]
    #: Who gave the first scores and who the second, as role codes, and in how many items both are one code.
    first_scorers: tuple[str, ...]
    second_scorers: tuple[str, ...]
    n_same_role: int
    #: The trials whose two scores differ, and those the second scorer could not tell: `(trial, first, second)`.
    disagreements: tuple[tuple[str, str, str], ...]
    cannot_tell: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True, slots=True)
class Inputs:
    """Everything one run of the report reads."""

    draft: bool
    protocol: Mapping[str, Any]
    protocol_file: str
    protocol_sha256: str
    #: The counted logs that were read, and the pilot logs that were found beside them and left alone.
    logs: tuple[str, ...]
    pilot_logs: tuple[str, ...]
    trials: tuple[CountedTrial, ...]
    #: For every scored trial: the demonstration its distance is measured to and that distance as read, recomputed
    #: from the audits and found equal to what the record says.
    nearest: Mapping[str, tuple[int, float]]
    #: The audited demonstration starts of each set; None where there is no audit file.
    audits: Mapping[str, tuple[DemoStart, ...] | None]
    audit_files: Mapping[str, str | None]
    scale_runs: tuple[PositionLog, ...]
    repeat_sets: tuple[PositionLog, ...]
    by_eye: tuple[ByEyeReset, ...]
    by_eye_files: tuple[str, ...]
    #: The recorder's placement log of each set, in the order written; None where there is no file.
    placements: Mapping[str, tuple[RecordedPlacement, ...] | None]
    placement_files: Mapping[str, str | None]
    #: Rows of the decision log marked as deviations; None when there is no decision log.
    deviations: tuple[Deviation, ...] | None
    decisions_file: str | None
    simulation: SimulationTables | None
    #: Void trials that ran with other execution settings than the scored trials of their condition, in words.
    execution_notes: tuple[str, ...]
    #: A line that says what the output is when it is not a counted report (a draft, a synthetic example).
    note: str | None = None
    #: The agreement of the second scoring from video; None where there is no file.
    second_scoring: SecondScoring | None = None
    #: One SHA-256 of the counted logs as they are now, as the second scoring's draw takes it; None without a log.
    logs_digest: str | None = None
    #: The scene audit of each counted log, by the log's name; None where a log has no audit file.
    scenes: Mapping[str, SceneAudit | None] = field(default_factory=dict)

    @property
    def scored(self) -> tuple[CountedTrial, ...]:
        return tuple(trial for trial in self.trials if trial.record.scored)


# --------------------------------------------------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------------------------------------------------


def load_inputs(
    root: str | Path,
    *,
    draft: bool = False,
    trials: Sequence[str | Path] | None = None,
    protocol: str | Path | None = None,
    audit_dir: str | Path | None = None,
    instrument_dir: str | Path | None = None,
    recording_dir: str | Path | None = None,
    decisions: str | Path | None = None,
    simulation_dir: str | Path | None = None,
    second_scoring_dir: str | Path | None = None,
    note: str | None = None,
) -> Inputs:
    """Read everything the report is made from, below `root` unless a place is given, and check the counted trials.

    Defaults: `config/protocol.toml`, every `data/trials/*.jsonl` that is not a pilot log, `data/audit/`,
    `data/instrument/`, `data/recording/`, `docs/decisions.md`, `analysis/simulation/` and
    `data/second_scoring/main/`. The scene audit of a log is `scene_<the log's name>.json` in the audit folder. A
    file or folder that is not there is not an error: what depends on it is left empty and the report says so. A
    counted report (not a draft) needs a frozen protocol.
    """
    root = Path(root)
    protocol_path = Path(protocol) if protocol is not None else root / "config" / "protocol.toml"
    protocol_data, digest = read_protocol(protocol_path, root)
    status = str(protocol_data["protocol"].get("status", ""))
    if not draft and status != FROZEN:
        raise ReportError(
            f"{_name(protocol_path, root)} has status {status!r}, and a counted report is written only under a "
            f"frozen protocol: pass --draft to look at what there is"
        )

    if trials is None:
        log_paths, pilot_paths = find_logs(root / "data" / "trials")
    else:
        log_paths, pilot_paths = [Path(path) for path in trials], []
    records = read_counted_trials(log_paths, root)
    check_records(records, protocol_data, _name(protocol_path, root), draft=draft)

    audit_folder = Path(audit_dir) if audit_dir is not None else root / "data" / "audit"
    audits: dict[str, tuple[DemoStart, ...] | None] = {}
    audit_files: dict[str, str | None] = {}
    for dataset in schema.DATASETS:
        path = audit_folder / f"{dataset}.json"
        audits[dataset] = read_audit(path, dataset, root) if path.is_file() else None
        audit_files[dataset] = _name(path, root) if path.is_file() else None
    nearest = check_distances(records, audits)
    execution_notes = check_execution(records, protocol_data) + check_checkpoints(records)
    scenes: dict[str, SceneAudit | None] = {}
    for log_path in log_paths:
        path = audit_folder / f"scene_{log_path.stem}.json"
        scenes[_name(log_path, root)] = read_scene_audit(path, log_path.name, root) if path.is_file() else None
    check_scenes(records, scenes)

    instrument = Path(instrument_dir) if instrument_dir is not None else root / "data" / "instrument"
    by_eye_paths = sorted(instrument.glob("byeye_*.csv"))
    recording = Path(recording_dir) if recording_dir is not None else root / "data" / "recording"
    placements: dict[str, tuple[RecordedPlacement, ...] | None] = {}
    placement_files: dict[str, str | None] = {}
    for dataset in schema.DATASETS:
        path = recording / f"{dataset}_placements.jsonl"
        placements[dataset] = read_placements(path, dataset, root) if path.is_file() else None
        placement_files[dataset] = _name(path, root) if path.is_file() else None

    decisions_path = Path(decisions) if decisions is not None else root / "docs" / "decisions.md"
    simulation_path = Path(simulation_dir) if simulation_dir is not None else root / "analysis" / "simulation"
    second_folder = (
        Path(second_scoring_dir) if second_scoring_dir is not None else root / "data" / "second_scoring" / "main"
    )
    agreement_path = second_folder / AGREEMENT_FILE
    return Inputs(
        draft=draft,
        protocol=protocol_data,
        protocol_file=_name(protocol_path, root),
        protocol_sha256=digest,
        logs=tuple(_name(path, root) for path in log_paths),
        pilot_logs=tuple(_name(path, root) for path in pilot_paths),
        trials=tuple(records),
        nearest=nearest,
        audits=audits,
        audit_files=audit_files,
        scale_runs=tuple(read_position_log(path, root, least=2) for path in sorted(instrument.glob("ruler_*.csv"))),
        repeat_sets=tuple(read_position_log(path, root, least=2) for path in sorted(instrument.glob("repeat_*.csv"))),
        by_eye=tuple(reset for path in by_eye_paths for reset in read_by_eye(path, root)),
        by_eye_files=tuple(_name(path, root) for path in by_eye_paths),
        placements=placements,
        placement_files=placement_files,
        deviations=read_deviations(decisions_path) if decisions_path.is_file() else None,
        decisions_file=_name(decisions_path, root) if decisions_path.is_file() else None,
        simulation=read_simulation(simulation_path, root),
        execution_notes=tuple(execution_notes),
        note=note,
        second_scoring=read_second_scoring(agreement_path, root) if agreement_path.is_file() else None,
        logs_digest=logs_digest(log_paths, root),
        scenes=scenes,
    )


# --------------------------------------------------------------------------------------------------------------------
# The protocol and the counted trials
# --------------------------------------------------------------------------------------------------------------------


def read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    """The protocol file's parameters and the SHA-256 of the file."""
    try:
        return config.load_protocol(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, config.ConfigError, tomllib.TOMLDecodeError) as error:
        raise ReportError(
            f"the protocol file {_name(path, root)} cannot be read: {_plain(error, path, root)}"
        ) from error


def is_pilot_log(path: str | Path) -> bool:
    """True for a log that holds pilot trials: its file name starts with `pilot`."""
    return Path(path).name.startswith(PILOT_PREFIX)


def find_logs(trials_dir: Path) -> tuple[list[Path], list[Path]]:
    """The trial logs of a folder: `(counted logs, pilot logs)`, each in name order."""
    logs = sorted(trials_dir.glob("*.jsonl")) if trials_dir.is_dir() else []
    return [path for path in logs if not is_pilot_log(path)], [path for path in logs if is_pilot_log(path)]


def read_counted_trials(paths: Iterable[Path], root: Path) -> list[CountedTrial]:
    """Every record of the given logs, in name order of the logs and in the order written. A pilot log is refused."""
    trials: list[CountedTrial] = []
    for path in sorted(Path(item) for item in paths):
        name = _name(path, root)
        if is_pilot_log(path):
            raise ReportError(f"{name} is a pilot log, and pilot trials are never analysed with the counted trials")
        if not path.is_file():
            raise ReportError(f"the trial log {name} does not exist")
        try:
            records = schema.read_trials(path)
        except schema.SchemaError as error:
            raise ReportError(f"the trial log {name} cannot be read: {_plain(error, path, root)}") from error
        trials.extend(CountedTrial(name, number, record) for number, record in enumerate(records, start=1))
    return trials


def check_records(
    trials: Sequence[CountedTrial], protocol: Mapping[str, Any], protocol_file: str, *, draft: bool
) -> None:
    """Refuse records that cannot be analysed together under this protocol (see the module's notes)."""
    if not draft:
        unfrozen = [t for t in trials if t.record.versions.get("protocol_status") != FROZEN]
        if unfrozen:
            status = unfrozen[0].record.versions.get("protocol_status", "none")
            logs = ", ".join(sorted({trial.log for trial in unfrozen}))
            raise ReportError(
                f"not run under a frozen protocol (status {status!r}), and so not counted: {_listed(unfrozen)} "
                f"({logs}); pass --draft to look at such trials anyway"
            )
    by_version: dict[str, list[CountedTrial]] = {}
    for trial in trials:
        by_version.setdefault(str(trial.record.versions.get("protocol_version", "none")), []).append(trial)
    if len(by_version) > 1:
        parts = "; ".join(
            f"{version!r} in {_listed(found)} ({', '.join(sorted({t.log for t in found}))})"
            for version, found in sorted(by_version.items())
        )
        raise ReportError(
            f"the trials were run under more than one protocol version, and one report holds one: {parts}"
        )
    version = str(protocol["protocol"]["version"])
    if by_version and version not in by_version:
        ran = next(iter(by_version))
        raise ReportError(
            f"the trials were run under protocol version {ran!r} and {protocol_file} is version {version!r}: "
            f"the analysis would use other parameters than the trials did"
        )

    seen: dict[str, CountedTrial] = {}
    slots: dict[tuple[str, str, str], CountedTrial] = {}
    blocks = protocol.get("blocks", {})
    for trial in trials:
        planned = trial.record.planned
        if planned.trial_id in seen:
            raise ReportError(
                f"the trial name {planned.trial_id} appears twice ({seen[planned.trial_id].log}, record "
                f"{seen[planned.trial_id].number}, and {trial.log}, record {trial.number}): a trial is logged once"
            )
        seen[planned.trial_id] = trial
        if planned.block not in blocks:
            raise ReportError(
                f"{planned.trial_id} belongs to block {planned.block}, which {protocol_file} does not have"
            )
        allowed = block_conditions(blocks[planned.block])
        if planned.condition not in allowed:
            raise ReportError(
                f"{planned.trial_id} is a trial of {planned.condition} in block {planned.block}, "
                f"which runs {' and '.join(allowed)}"
            )
        for stamp in (trial.record.ts_start, trial.record.ts_end):
            if parse_time(stamp) is None:
                raise ReportError(
                    f"{planned.trial_id} has the time {stamp!r}, which is not an ISO time with a UTC offset"
                )
        if is_paired(blocks[planned.block]) and trial.record.scored:
            if planned.pair_id is None:
                raise ReportError(f"{planned.trial_id} is a scored trial of block {planned.block} without a pair")
            slot = (planned.block, planned.pair_id, planned.condition)
            if slot in slots:
                raise ReportError(
                    f"{slots[slot].trial_id} and {planned.trial_id} are both scored trials of {planned.condition} "
                    f"at placement {planned.pair_id}: a placement is tested once with each policy"
                )
            slots[slot] = trial


def block_conditions(entry: Mapping[str, Any]) -> list[str]:
    """The conditions a block of the protocol tests."""
    return list(entry["conditions"]) if "conditions" in entry else [entry["condition"]]


def is_paired(entry: Mapping[str, Any]) -> bool:
    """True for a block that tests each placement with two policies."""
    return "pairs" in entry


def parse_time(stamp: str) -> datetime | None:
    """An ISO time with a UTC offset, or None when the text is not one."""
    try:
        moment = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else None


# --------------------------------------------------------------------------------------------------------------------
# Distances against the audits
# --------------------------------------------------------------------------------------------------------------------


def read_audit(path: Path, dataset: str, root: Path) -> tuple[DemoStart, ...]:
    """The audited starts of one demonstration set, in episode order."""
    name = _name(path, root)
    try:
        starts = schema.read_demo_starts(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ReportError(f"the audit {name} cannot be read: {_plain(error, path, root)}") from error
    other = sorted({start.dataset for start in starts} - {dataset})
    if other:
        raise ReportError(f"the audit {name} holds episodes of the {other[0]} set, not only of the {dataset} set")
    indices = [start.episode_index for start in starts]
    if len(set(indices)) != len(indices):
        raise ReportError(f"the audit {name} lists an episode twice")
    return tuple(sorted(starts, key=lambda start: start.episode_index))


def nearest_demonstration(record: TrialRecord, demos: Sequence[DemoStart]) -> tuple[int, float]:
    """The demonstration a trial's distance is measured to, and that distance as the camera reads it.

    For a policy: the nearest trained demonstration of the policy's own set. For a replay: the replayed demonstration
    itself, the only one a replay knows about. The trial runner's rule (`trial_session.nearest_for_record`), written
    again here because the analysis may not import the rig's modules. `demos` are the trained episodes of the set.
    """
    planned, placed = record.planned, record.placed
    if placed is None:
        raise ValueError("a trial without a placement has no distance")
    if planned.condition == "replay":
        for demo in demos:
            if demo.episode_index == planned.replay_episode:
                return demo.episode_index, placed.distance_to(demo.measured)
        raise ReportError(
            f"{planned.trial_id} replays episode {planned.replay_episode}, which is not a trained demonstration of "
            f"the {planned.dataset} set in the audit"
        )
    index, distance = geometry.nearest(placed, demos)
    return demos[index].episode_index, float(distance)


def check_distances(
    trials: Sequence[CountedTrial], audits: Mapping[str, Sequence[DemoStart] | None]
) -> dict[str, tuple[int, float]]:
    """Recompute every scored trial's nearest demonstration and distance, and refuse a record that says otherwise.

    Returns, by trial name, the demonstration and the distance as read (before any scale correction).
    """
    trained: dict[str, list[DemoStart]] = {}
    nearest: dict[str, tuple[int, float]] = {}
    for trial in trials:
        record = trial.record
        if not record.scored:
            continue
        dataset = record.planned.dataset
        if audits.get(dataset) is None:
            raise ReportError(
                f"{trial.trial_id} measures its distance to the {dataset} demonstrations, and there is no audit of "
                f"that set (data/audit/{dataset}.json): its distance cannot be checked"
            )
        if dataset not in trained:
            trained[dataset] = geometry.trained_demos(audits[dataset], dataset)
            if not trained[dataset]:
                raise ReportError(f"the audit of the {dataset} set holds no trained demonstration")
        demos = trained[dataset]
        episode, distance = nearest_demonstration(record, demos)
        said_episode, said_distance = record.nearest_demo_episode, float(record.nearest_demo_distance_mm)
        to_said = next((record.placed.distance_to(d.measured) for d in demos if d.episode_index == said_episode), None)
        same_episode = to_said is not None and abs(to_said - distance) <= DISTANCE_TOLERANCE_MM
        if record.planned.condition == "replay":
            same_episode = said_episode == episode
        if not same_episode or abs(said_distance - distance) > DISTANCE_TOLERANCE_MM:
            raise ReportError(
                f"{trial.trial_id} ({trial.log}) says its nearest demonstration is episode {said_episode} at "
                f"{said_distance:.3f} mm, and the audit of the {dataset} set gives episode {episode} at "
                f"{distance:.3f} mm: the audit changed after the trial was run, or the record was altered"
            )
        nearest[trial.trial_id] = (episode, distance)
    return nearest


# --------------------------------------------------------------------------------------------------------------------
# Execution settings
# --------------------------------------------------------------------------------------------------------------------


def execution_settings(record: TrialRecord) -> dict[str, Any]:
    """The settings a trial was run with: its `execution` entries less what belongs to the one trial."""
    return {key: value for key, value in record.execution.items() if key not in PER_TRIAL_KEYS}


def handoff_key(condition: str) -> str:
    """The entry of the protocol's `[execution]` section that states the handoff tolerance of a condition's trials."""
    return "replay_handoff_tolerance_deg" if condition == "replay" else "handoff_tolerance_deg"


def check_execution(trials: Sequence[CountedTrial], protocol: Mapping[str, Any]) -> list[str]:
    """Refuse scored trials of a condition that were not all run the same way, or not as the protocol says.

    A record's `execution` holds every setting of the protocol's `[execution]` section with the value the session
    ran with, and then what belongs to the one trial (`PER_TRIAL_KEYS`). For the scored trials of each condition:

    * every setting they carry has one value in all of them;
    * every setting the protocol states is there, with the protocol's value. A record that does not carry one is
      refused too: it cannot show that the setting ran.

    What belongs to the one trial is not a setting and stops nothing. A void trial does not enter the analysis, so
    one that ran with other settings is not refused either. Both come back as sentences for the report: a void
    trial with other settings, and a scored trial whose handoff gate used another tolerance than the protocol
    states for its kind of trial.
    """
    stated = protocol.get("execution", {})
    notes: list[str] = []
    for condition in schema.CONDITIONS:
        scored = [t for t in trials if t.condition == condition and t.record.scored]
        if not scored:
            continue
        settings = {t.trial_id: execution_settings(t.record) for t in scored}
        for key in sorted({key for values in settings.values() for key in values}):
            groups: dict[str, list[CountedTrial]] = {}
            for trial in scored:
                groups.setdefault(_shown(settings[trial.trial_id].get(key, _ABSENT)), []).append(trial)
            if len(groups) > 1:
                ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
                parts = "; ".join(f"{value} in {_listed(found)}" for value, found in ordered)
                raise ReportError(
                    f"the execution setting {key} is not the same in every scored {condition} trial ({parts}): the "
                    f"trials of a condition are analysed together only if they were all run the same way"
                )
        ran = settings[scored[0].trial_id]
        for key, value in stated.items():
            if key not in ran:
                raise ReportError(
                    f"the scored {condition} trials do not record {key}, and the protocol's [execution] section "
                    f"says {key} = {_shown(value)}: a record that does not carry a setting cannot show that it ran "
                    f"({_listed(scored)})"
                )
            if not _same_value(ran[key], value):
                raise ReportError(
                    f"the scored {condition} trials ran with {key} = {_shown(ran[key])}, and the protocol's "
                    f"[execution] section says {key} = {_shown(value)} ({_listed(scored)})"
                )
        for trial in trials:
            if trial.condition != condition or trial.record.scored:
                continue
            other = execution_settings(trial.record)
            differing = sorted(
                key for key in set(other) | set(ran) if _shown(other.get(key, _ABSENT)) != _shown(ran.get(key, _ABSENT))
            )
            if differing:
                shown = ", ".join(f"{key} = {_shown(other.get(key, _ABSENT))}" for key in differing)
                notes.append(f"the void trial {trial.trial_id} ran with {shown}, unlike the scored {condition} trials")
        gate = handoff_key(condition)
        used: dict[str, list[CountedTrial]] = {}
        for trial in scored:
            value = trial.record.execution.get(HANDOFF_USED_KEY, _ABSENT)
            if value is not _ABSENT and gate in stated and not _same_value(value, stated[gate]):
                used.setdefault(_shown(value), []).append(trial)
        for value, found in sorted(used.items()):
            notes.append(
                f"the handoff gate of {_listed(found)} used a tolerance of {value} degrees, and the protocol states "
                f"{gate} = {_shown(stated[gate])} for a scored {condition} trial"
            )
    return notes


def split_checkpoint(name: str) -> tuple[str | None, str]:
    """The training run and the training step that a record's `checkpoint` names.

    The trial runner writes `<run>@<step>` when the policy's folders show the run (`act-run-v1@080000`) and the step
    alone otherwise (`080000`). The run is None when the name does not give one.
    """
    run, mark, step = name.rpartition(RUN_MARK)
    return (run if mark else None), step


def check_checkpoints(trials: Sequence[CountedTrial]) -> list[str]:
    """Refuse scored policy trials that were not all run at one training step.

    The protocol evaluates every policy at one step (section 5), so the step is held across both policies and all
    their blocks. The run is not: a second training run of a layout may be evaluated beside the first, and the
    records keep each trial's run. Returns sentences for the report: a void trial of another step, and a threshold
    block whose scored trials name more than one run.
    """
    notes: list[str] = []
    policies = [t for t in trials if t.record.checkpoint is not None]
    by_step: dict[str, list[CountedTrial]] = {}
    for trial in policies:
        if trial.record.scored:
            by_step.setdefault(split_checkpoint(trial.record.checkpoint)[1], []).append(trial)
    if len(by_step) > 1:
        ordered = sorted(by_step.items(), key=lambda item: (-len(item[1]), item[0]))
        parts = "; ".join(f"step {step} in {_listed(found)}" for step, found in ordered)
        raise ReportError(
            f"the scored policy trials were not all run at one training step ({parts}): the protocol evaluates "
            f"every policy at the same step"
        )
    step = next(iter(by_step), None)
    for trial in policies:
        other = split_checkpoint(trial.record.checkpoint)[1]
        if not trial.record.scored and step is not None and other != step:
            notes.append(f"the void trial {trial.trial_id} ran at training step {other}, unlike the scored trials")
    by_block: dict[tuple[str, str], set[str]] = {}
    for trial in policies:
        if trial.record.scored and trial.record.planned.pair_id is None:
            run = split_checkpoint(trial.record.checkpoint)[0]
            by_block.setdefault((trial.block, trial.condition), set()).add(run or "a run the record does not name")
    for (block, condition), runs in sorted(by_block.items()):
        if len(runs) > 1:
            notes.append(
                f"the scored {condition} trials of block {block} were run with {len(runs)} training runs "
                f"({', '.join(sorted(runs))}), and the block is fitted as one policy"
            )
    return notes


def _same_value(first: Any, second: Any) -> bool:
    numbers = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (first, second))
    if numbers:
        return math.isclose(float(first), float(second), rel_tol=1e-9, abs_tol=1e-12)
    return first == second


def _shown(value: Any) -> str:
    """A setting's value as a message prints it; equal settings print alike."""
    if value is _ABSENT:
        return "not recorded"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):g}"
    return json.dumps(value, sort_keys=True) if not isinstance(value, str) else value


# --------------------------------------------------------------------------------------------------------------------
# The instrument's files
# --------------------------------------------------------------------------------------------------------------------


def read_position_log(path: Path, root: Path, *, least: int = 1) -> PositionLog:
    """A CSV written by `scripts/rig/measure.py log`: one logged cube position per row (`x_mm`, `y_mm`)."""
    name = _name(path, root)
    rows = _csv_rows(path, name, ("x_mm", "y_mm"))
    readings = tuple(Point(_number(row, "x_mm", name, at), _number(row, "y_mm", name, at)) for at, row in rows)
    if len(readings) < least:
        raise ReportError(f"{name} holds {len(readings)} reading(s), and {least} are needed to make anything of it")
    labels = {row.get("label", "") for _, row in rows}
    return PositionLog(name, labels.pop() if len(labels) == 1 else path.stem, readings)


def read_by_eye(path: Path, root: Path) -> list[ByEyeReset]:
    """A CSV written by `scripts/rig/measure.py byeye`: one reset by eye per row.

    Each row's own `error_mm` must be the distance between its two positions (both are written with two decimals).
    """
    name = _name(path, root)
    needed = ("label", "reference_x_mm", "reference_y_mm", "eye_x_mm", "eye_y_mm", "error_mm")
    resets = []
    for at, row in _csv_rows(path, name, needed):
        reset = ByEyeReset(
            file=name,
            label=row["label"],
            reference=Point(_number(row, "reference_x_mm", name, at), _number(row, "reference_y_mm", name, at)),
            eye=Point(_number(row, "eye_x_mm", name, at), _number(row, "eye_y_mm", name, at)),
        )
        if abs(reset.error_mm - _number(row, "error_mm", name, at)) > 0.05:
            raise ReportError(
                f"{name}, row {at}: the error it states ({row['error_mm']} mm) is not the distance between its two "
                f"positions ({reset.error_mm:.2f} mm)"
            )
        resets.append(reset)
    return resets


def _csv_rows(path: Path, name: str, needed: Sequence[str]) -> list[tuple[int, dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in needed if column not in (reader.fieldnames or [])]
            if missing:
                raise ReportError(f"{name} has no column {missing[0]}")
            return list(enumerate(reader, start=1))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ReportError(f"{name} cannot be read: {type(error).__name__}") from error


def _number(row: Mapping[str, str], column: str, name: str, at: int) -> float:
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        raise ReportError(f"{name}, row {at}: {column} is {row[column]!r}, not a number")
    return value


# --------------------------------------------------------------------------------------------------------------------
# The recorder's placement logs, the decision log, the simulation
# --------------------------------------------------------------------------------------------------------------------


def read_placements(path: Path, dataset: str, root: Path) -> tuple[RecordedPlacement, ...]:
    """The recorder's placement log of one set (`data/recording/<dataset>_placements.jsonl`), in the order written."""
    name = _name(path, root)
    placements = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReportError(f"{name} cannot be read: {type(error).__name__}") from error
    for at, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            placement = RecordedPlacement(
                dataset=str(entry["dataset"]),
                episode_index=int(entry["episode_index"]),
                sitting=int(entry["sitting"]),
                target=Point(entry["target"]["x_mm"], entry["target"]["y_mm"]),
                placed=Point(entry["placed"]["x_mm"], entry["placed"]["y_mm"]),
                rerecords=int(entry["rerecords"]),
                saved_at=str(entry["saved_at"]),
                session=str(entry["session"]),
            )
        except (ValueError, KeyError, TypeError) as error:
            raise ReportError(
                f"{name}, line {at}: not a placement the recorder writes ({type(error).__name__})"
            ) from error
        if placement.dataset != dataset:
            raise ReportError(
                f"{name}, line {at}: a placement of the {placement.dataset} set in the {dataset} set's log"
            )
        if parse_time(placement.saved_at) is None:
            raise ReportError(
                f"{name}, line {at}: saved_at is {placement.saved_at!r}, not an ISO time with a UTC offset"
            )
        placements.append(placement)
    return tuple(placements)


def read_deviations(path: Path) -> tuple[Deviation, ...]:
    """The rows of the decision log's table that are marked **DEVIATION**, in the order of the file."""
    deviations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|") or _DEVIATION not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        date, decision, why = [*cells, "", "", ""][:3]
        deviations.append(Deviation(date, decision, why))
    return tuple(deviations)


def read_simulation(folder: Path, root: Path) -> SimulationTables | None:
    """The simulations of the design: the procedure simulation's tables (`procedure.csv`, `paired_power.csv`) with
    the digest its report names, and the table of the comparison rule (`comparison_rule.csv`) with its report's.

    None when a table of the procedure simulation is missing. A digest is None when its report is missing or names
    none. The comparison rule's table is None when it is not there.
    """
    sweep, power, report = folder / "procedure.csv", folder / "paired_power.csv", folder / "procedure.md"
    rule, rule_report = folder / "comparison_rule.csv", folder / "comparison_rule.md"
    if not (sweep.is_file() and power.is_file()):
        return None

    def digest(path: Path) -> str | None:
        found = _DIGEST.search(" ".join(path.read_text(encoding="utf-8").split())) if path.is_file() else None
        return found.group(1) if found else None

    def table(path: Path) -> tuple[Mapping[str, str], ...]:
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                return tuple(csv.DictReader(handle))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise ReportError(f"{_name(path, root)} cannot be read: {type(error).__name__}") from error

    present = (sweep, power, report, *((rule, rule_report) if rule.is_file() else ()))
    return SimulationTables(
        protocol_digest=digest(report),
        sweep=table(sweep),
        power=table(power),
        files=tuple(_name(path, root) for path in present if path.is_file()),
        rule=table(rule) if rule.is_file() else None,
        rule_digest=digest(rule_report) if rule.is_file() else None,
    )


# --------------------------------------------------------------------------------------------------------------------
# The scene audits
# --------------------------------------------------------------------------------------------------------------------


def read_scene_audit(path: Path, log_name: str, root: Path) -> SceneAudit:
    """The scene audit of the log `log_name` (`scene_<log>.json`, written by `scripts/rig/scene_audit.py`).

    Refused: a file that is not such an audit, one made from a log of another name, one that lists a trial twice,
    and one in which a trial's flag is not what the trial's own reading and the file's limit give. The flag is the
    audit's rule (`scene_audit.far`): a trial is named when its reading stands beyond the limit across or down.
    """
    name = _name(path, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        audit = SceneAudit(
            file=name,
            log=str(data["log"]),
            basket_named_mm=_measure(data["basket_named_mm"]),
            wrist_named_px=_measure(data["wrist_named_px"]),
            readings=tuple(_scene_reading(entry) for entry in data["trials"]),
        )
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ReportError(
            f"{name} is not a scene audit as scripts/rig/scene_audit.py writes one: {_misshapen(error)}"
        ) from error
    if audit.log != log_name:
        raise ReportError(f"{name} is the scene audit of {audit.log}, and it was read for {log_name}")
    seen: set[str] = set()
    for reading in audit.readings:
        if reading.trial_id in seen:
            raise ReportError(f"the scene audit {name} lists {reading.trial_id} twice")
        seen.add(reading.trial_id)
        flags = (
            ("named_basket", reading.named_basket, reading.basket_off_mm, audit.basket_named_mm, "mm"),
            ("named_wrist", reading.named_wrist, reading.wrist_off_px, audit.wrist_named_px, "px"),
        )
        for flag, said, off, limit, unit in flags:
            if said != (None if off is None else off > limit):
                raise ReportError(
                    f"the scene audit {name} gives {reading.trial_id} {flag} = {json.dumps(said)}, and its own "
                    f"reading at the start is {'none' if off is None else f'{off:g} {unit}'} against a limit of "
                    f"{limit:g} {unit}"
                )
    return audit


def _scene_reading(entry: Mapping[str, Any]) -> SceneReading:
    """One trial's line of a scene audit (`scene_audit.audit_trial` and what `audit_log` puts before it)."""
    basket, wrist = entry["basket_start"], entry["wrist_start"]
    down = None if wrist is None or wrist["down_px"] is None else _measure(wrist["down_px"], signed=True)
    return SceneReading(
        trial_id=_text(entry["trial_id"]),
        condition=None if entry["condition"] is None else _text(entry["condition"]),
        outcome=None if entry["outcome"] is None else _text(entry["outcome"]),
        basket_start_mm=(
            None
            if basket is None
            else (_measure(basket["across_mm"], signed=True), _measure(basket["down_mm"], signed=True))
        ),
        wrist_start_px=None if wrist is None else (_measure(wrist["across_px"], signed=True), down),
        basket_moved_mm=None if entry["basket_moved_mm"] is None else _measure(entry["basket_moved_mm"]),
        basket_was_moved=None if entry["basket_was_moved"] is None else _truth(entry["basket_was_moved"]),
        named_basket=None if entry["named_basket"] is None else _truth(entry["named_basket"]),
        named_wrist=None if entry["named_wrist"] is None else _truth(entry["named_wrist"]),
    )


def check_scenes(trials: Sequence[CountedTrial], scenes: Mapping[str, SceneAudit | None]) -> None:
    """Refuse a scene audit that was not made from the log it is read for, and audits with different limits.

    An audit holds a line for every record its log had when it was made. A line of a trial the log does not hold, or
    one that gives a trial another condition or outcome than its record, comes from another log. A log that has
    gained trials since its audit is not refused: the report counts the trials the audit does not hold.
    """
    for log, audit in scenes.items():
        if audit is None:
            continue
        records = {trial.trial_id: trial.record for trial in trials if trial.log == log}
        for reading in audit.readings:
            record = records.get(reading.trial_id)
            if record is None:
                raise ReportError(
                    f"the scene audit {audit.file} holds {reading.trial_id}, which {log} does not: the audit was "
                    f"made from another log"
                )
            said = (reading.condition, reading.outcome)
            if said != (record.planned.condition, record.outcome):
                raise ReportError(
                    f"the scene audit {audit.file} gives {reading.trial_id} as {said[1]} of {said[0]}, and {log} "
                    f"as {record.outcome} of {record.planned.condition}: the audit was made from another log, or "
                    f"the record was altered"
                )
    limits = {(audit.basket_named_mm, audit.wrist_named_px) for audit in scenes.values() if audit is not None}
    if len(limits) > 1:
        files = ", ".join(audit.file for audit in scenes.values() if audit is not None)
        raise ReportError(
            f"the scene audits do not all name a trial beyond the same limits ({files}): the protocol states one "
            f"limit for the basket and one for the wrist camera"
        )


def _measure(value: Any, *, signed: bool = False) -> float:
    """A measured number as a file holds one; a distance is zero or more unless it is `signed`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{value!r} is not a number")
    if not signed and value < 0:
        raise ValueError(f"{value!r} is a distance below zero")
    return float(value)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{value!r} is not a name")
    return value


# --------------------------------------------------------------------------------------------------------------------
# The second scoring from video
# --------------------------------------------------------------------------------------------------------------------


def logs_digest(paths: Iterable[Path], root: Path) -> str | None:
    """One SHA-256 for the trial logs as they are now: of their own SHA-256s, in name order. None without a log.

    The second scoring's draw names the logs it was made from by this digest (`second_scoring.Population.digest`,
    written again here), and it changes with every line a log gains. So an agreement that carries the digest of the
    logs the report reads is the second scoring of exactly these trials.
    """
    named = sorted((_name(Path(path), root), str(path)) for path in paths)
    if not named:
        return None
    hashes = "".join(hashlib.sha256(Path(path).read_bytes()).hexdigest() for _, path in named)
    return hashlib.sha256(hashes.encode("ascii")).hexdigest()


def read_second_scoring(path: Path, root: Path) -> SecondScoring:
    """The agreement of a second scoring from video (`agreement.json` of its folder).

    Refused: a file that is not what `scripts/second_scoring.py compare` writes; counts that do not add up (the
    items of a tally are those that agree, those that differ and those the second scorer could not tell; its table
    holds the compared ones; the two kinds of trial together are all the items; all the items are the trials drawn);
    and a scorer who is not named by a role code, because the report prints who scored.
    """
    name = _name(path, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        source, scorers = data["provenance"], data["scorers"]
        found = SecondScoring(
            file=name,
            rehearsal=_truth(data["rehearsal"]),
            logs=tuple(str(log) for log in source["logs"]),
            logs_digest=str(source["logs_digest"]),
            seed=_whole(source["seed"]),
            one_in=_whole(source["one_in"]),
            n_scored=_whole(source["n_scored"]),
            n_drawn=_whole(source["n_drawn"]),
            interval_level=float(data["interval_level"]),
            tallies={"all": _tally(data["all"]), **{kind: _tally(data["by_kind"][kind]) for kind in KINDS}},
            first_scorers=tuple(str(code) for code in scorers["first"]),
            second_scorers=tuple(str(code) for code in scorers["second"]),
            n_same_role=_whole(scorers["n_same_role"]),
            disagreements=tuple(_scored_twice(entry) for entry in data["disagreements"]),
            cannot_tell=tuple(_scored_twice(entry) for entry in data["cannot_tell"]),
        )
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ReportError(
            f"{name} is not an agreement as scripts/second_scoring.py compare writes one: {_misshapen(error)}"
        ) from error
    problem = _agreement_problem(found)
    if problem is not None:
        raise ReportError(f"the second scoring's agreement {name} does not add up: {problem}")
    unnamed = [code for code in (*found.first_scorers, *found.second_scorers) if not ROLE_CODE.fullmatch(code)]
    if unnamed:
        raise ReportError(
            f"{name} names {len(unnamed)} scorer(s) otherwise than by a role code such as op1 or sc2, and the "
            f"report prints who scored: a person is never named in a record"
        )
    return found


def _tally(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One tally of an agreement: the counts, the share that agree with its interval, and the two tables."""
    tally: dict[str, Any] = {key: _whole(entry[key]) for key in _TALLY_COUNTS}
    tally.update({key: None if entry[key] is None else float(entry[key]) for key in _TALLY_SHARES})
    tally["table"] = {
        f"{first}_{second}": _whole(entry["table"][f"{first}_{second}"]) for first in _SCORES for second in _SCORES
    }
    tally["cannot_tell_first"] = {first: _whole(entry["cannot_tell_first"][first]) for first in _SCORES}
    return tally


def _scored_twice(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    """A listed trial of an agreement: its name, its first score and its second."""
    return str(entry["trial_id"]), str(entry["first"]), str(entry["second"])


def _agreement_problem(found: SecondScoring) -> str | None:
    """What does not add up in an agreement, in words; None when everything does."""
    for name, tally in found.tallies.items():
        of = "all items" if name == "all" else f"the {name} trials"
        table = tally["table"]
        if tally["n_agree"] + tally["n_disagree"] != tally["n_compared"]:
            return f"of {of}, those that agree and those that differ are not the {tally['n_compared']} compared"
        if tally["n_compared"] + tally["n_cannot_tell"] != tally["n_items"]:
            return f"of {of}, the compared and those that could not be told are not the {tally['n_items']} items"
        if sum(table.values()) != tally["n_compared"]:
            return f"the table of {of} holds {sum(table.values())} items, and {tally['n_compared']} were compared"
        if sum(table[f"{score}_{score}"] for score in _SCORES) != tally["n_agree"]:
            return f"the table of {of} does not hold its {tally['n_agree']} agreeing items on its diagonal"
        if sum(tally["cannot_tell_first"].values()) != tally["n_cannot_tell"]:
            return f"of {of}, the items that could not be told are not counted once by their first score"
        if any((tally[key] is None) != (tally["n_compared"] == 0) for key in _TALLY_SHARES):
            return f"of {of}, there is a share that agree exactly when an item was compared"
    whole = found.tallies["all"]
    for key in _TALLY_COUNTS:
        if sum(found.tallies[kind][key] for kind in KINDS) != whole[key]:
            return f"{key} of the {' and the '.join(KINDS)} trials together is not that of all items"
    if whole["n_items"] != found.n_drawn:
        return f"it tallies {whole['n_items']} items, and {found.n_drawn} trials were drawn"
    if found.n_drawn > found.n_scored:
        return f"{found.n_drawn} trials were drawn from {found.n_scored}"
    if found.n_same_role > whole["n_items"]:
        return f"both scores are by one person in {found.n_same_role} of {whole['n_items']} items"
    if (len(found.disagreements), len(found.cannot_tell)) != (whole["n_disagree"], whole["n_cannot_tell"]):
        return "the trials it lists as differing or as not told are not as many as it counts"
    if not 0.0 < found.interval_level < 1.0:
        return f"its interval level is {found.interval_level:g}"
    return None


def _whole(value: Any) -> int:
    """A count as a file holds one: a whole number, zero or more. Anything else is a `ValueError`."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{value!r} is not a count")
    return value


def _truth(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{value!r} is neither true nor false")
    return value


def _misshapen(error: Exception) -> str:
    """Why a file is not of the shape its reader expects, in a few words that name no place on this machine."""
    if isinstance(error, KeyError):
        return f"it has no {error.args[0]}"
    if isinstance(error, ValueError) and not isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)):
        return str(error)
    return type(error).__name__


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _name(path: Path, root: Path) -> str:
    """A file as the outputs name it: relative to the root folder, or by its bare name when it lies elsewhere."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _plain(error: Exception, path: Path, root: Path) -> str:
    """An error's text with the file's place on this machine replaced by its name."""
    return str(error).replace(str(path), _name(path, root)).replace(str(path.resolve()), _name(path, root))


def _listed(trials: Sequence[CountedTrial]) -> str:
    """Trials by name for a message: all of them when they are few, the first ones and a count otherwise."""
    names = [trial.trial_id for trial in trials]
    if len(names) <= _NAMES_SHOWN:
        return ", ".join(names)
    return f"{', '.join(names[:_NAMES_SHOWN])} and {len(names) - _NAMES_SHOWN} more"
