"""Run evaluation trials: plan, place, run through the gates, score, log. One session, many trials.

This is the earlier project's rollout session (`rollout_session.py`) with the loop over attempts replaced by a loop
over planned trials. Everything that touches the arm is that project's own gated code, in that project's order:

    connect once
      preflight                             refuse to start if the arm is unsafe to torque
      policies loaded, inputs checked       before the robot is built
      camera view confirmed                 against the stored calibration, arm at rest
      WARM-UP      teleop                   bring the arm to the start pose, press n
      empty scene                           cube off the table, one picture for the detector
      for each trial the planner names:
        placement  arm holding              cube onto its target; measured; accepted when steady and n is pressed
        attempt    policy or replay         only through `attempt_cycle`: handoff gate, clamp, stall rule
        score      arm holding              0 to 5, a failure label, notes; `void <reason>` if it does not count
        log                                 one line, forced to disk; the planner reads it for the next trial
        reset      teleop                   cube out of the basket, arm back to the start pose, press n
      COOL-DOWN    teleop                   park it, press n
      seed the goal to the present position, disconnect leaving torque on

What differs from August, on purpose: a teleoperation phase ends only on a key press, never on a timer, so no attempt
starts by itself, and a placement is accepted only on the operator's key, with hands clear, because the next thing
that happens is an arm that moves; a start refused by the handoff gate is logged as void, not scored; two policies can be loaded and
the planner chooses per trial; a replay of recorded actions can stand in for a policy; and the person scoring sees the
trial's code, not its condition.

If you change this file, compare it line by line with `rollout_session.main` in the arm tooling. Never call
`run_attempt`, `record_loop` or `send_action` from here: the gates live in `attempt_cycle` and `teleop_phase`.
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from tolerance_radius import geometry, layouts, planner, schema, session_rules, usb_errors
from tolerance_radius.config import PILOT_OVERLAY_PATH, REPO_ROOT, load_protocol
from tolerance_radius.rig import camera_list, lerobot101, typing_guard
from tolerance_radius.schema import DemoStart, PlannedTrial, Point, TrialRecord
from tolerance_radius.seeds import derived_rng

TASK = "pick up the orange cube and place it in the basket"
AUDIT_DIR = REPO_ROOT / "data" / "audit"
#: The sets as planned: how many episodes each audit must hold.
LAYOUT_DIR = REPO_ROOT / "config" / "layouts"
#: The wrist camera's tone state is asked before an attempt only with the arm this near the pose of the session's
#: first reading: degrees for the joints the folded arm holds, and for the two it leaves free (the recorder's own
#: limits for a reset, `record_layout`).
WRIST_ASKED_WITHIN_DEG = (5.0, 12.0)
class LiveLabel(str):
    """A phase's name for the arm tooling that reads as something else while its phase runs.

    The tooling draws a phase's name into a countdown line that it rewrites in place five times a second, from a
    thread that only prints (`PhaseClock`). It is the one line the operator has before them while they move the arm,
    and the tooling is not this project's to change. Formatted into that line while `live` is set, the name gives
    `reading()` instead: for the correction after a refused start, where the joint is and where it is wanted.
    Everywhere else (the log, the speech, a comparison) it is the plain name. `reading` is called from the
    tooling's thread, so it must only read what the control loop has left for it: it never asks the arm.
    """

    reading: Callable[[], str]
    live: bool

    def __new__(cls, name: str, reading: Callable[[], str]) -> "LiveLabel":
        label = super().__new__(cls, name)
        label.reading = reading
        label.live = False
        return label

    def __format__(self, spec: str) -> str:
        # Only in the countdown's own thread: the tooling's alignment gate, which runs first and in the session's
        # thread, would otherwise print the readout as the phase's name ("Alignment gate before 'wrist roll now
        # ...'": the review of the night of 2026-10-07).
        if self.live and threading.current_thread() is not threading.main_thread():
            try:
                return format(self.reading(), spec)
            except Exception:  # noqa: BLE001 -- a readout that fails must not take the countdown with it
                pass
        return format(str(self), spec)


#: The longest a readout gets ("shoulder lift now -102.9, wanted -100.0 ok"): with it the tooling's countdown line
#: is 78 characters, the width it is made for, for as long as the correction is under ten minutes.
READOUT_WIDTH = 42


def handoff_readout(wanted: dict[str, float], seen: dict[str, float], tolerance: float) -> str:
    """What the correction after a refused start shows: of the joints in `wanted` (every joint the gate judges, not
    only those it refused for), the one now furthest from where the first action wants it, where it is and where it
    is wanted, and `ok` once it is within the gate's tolerance (then every one of them is). `wanted` and `seen` go
    by the robot's own names (`wrist_roll.pos`); a joint the control loop has not read yet is shown with what is
    wanted alone.

    On 2026-10-07 the gate refused a replay three times for the wrist's turn, by 7.8, 8.4 and 9.8 units: the line
    then said by how much and not which way, and each correction went the wrong way.
    """
    if not wanted:
        return ""

    def gap(name: str) -> float:
        return abs(seen[name] - wanted[name]) if name in seen else float("inf")

    name = max(wanted, key=gap)
    short = name.removesuffix(".pos").replace("_", " ")
    if name not in seen:
        return f"{short} wanted {wanted[name]:+.1f}"
    return f"{short} now {seen[name]:+.1f}, wanted {wanted[name]:+.1f}{' ok' if gap(name) <= tolerance else ''}"


#: The names of a session's own phases, as the tooling draws them into its countdown line
#: ("  Reset: 0:19 elapsed -- press 'n' to end"), which it rewrites in place and pads to 78 columns.
WARM_UP, RESET, COOL_DOWN = "Warm up", "Reset", "Cool down"
#: The longest the countdown line of a phase without a timer gets, beside its name: ": 59:59 elapsed -- press 'n' to
#: end" and the two spaces before the name.
COUNTDOWN_BESIDE_NAME = 2 + len(": 59:59 elapsed -- press 'n' to end")

#: Before an attempt the front camera's picture is looked at once, at that moment. A look that is not clean sends
#: the trial back to its placement; this many in a row for one trial end the session: the picture has changed.
PICTURE_LOOKS_BEFORE_STOP = 3
#: A failure whose kind the scorer did not give. Kept visible instead of guessed.
UNLABELLED = "unlabelled"


# --------------------------------------------------------------------------------------------------------------------
# Pure helpers (tested without hardware)
# --------------------------------------------------------------------------------------------------------------------


def block_conditions(block: dict[str, Any]) -> list[str]:
    """The conditions a block of the protocol tests."""
    return list(block["conditions"]) if "conditions" in block else [block["condition"]]


def planned_total(block: dict[str, Any]) -> int:
    """How many scored trials a block of the protocol asks for."""
    if "pairs" in block:
        return int(block["pairs"]) * len(block_conditions(block))
    staircases = sum(int(staircase["trials"]) for staircase in block.get("staircases", []))
    return int(block.get("anchors_zero", 0)) + staircases + int(block.get("anchors_far", 0))


def block_order(blocks: Sequence[str], protocol: dict[str, Any]) -> list[str]:
    """The order in which several blocks run when they share a session: one label per planned trial, shuffled.

    Interleaving blocks puts their trials under the same conditions of the day, so a slow drift cannot pass for a
    difference between them. The order depends only on the protocol's seed and the blocks' names.
    """
    labels = [block for block in blocks for _ in range(planned_total(protocol["blocks"][block]))]
    rng = derived_rng(int(protocol["datasets"]["layout_seed"]), "block-order", *blocks)
    return [labels[i] for i in rng.permutation(len(labels))]


def next_planned(
    blocks: Sequence[str],
    protocol: dict[str, Any],
    demo_starts: Sequence[DemoStart],
    log: Sequence[TrialRecord],
    *,
    plan: Callable[..., PlannedTrial | None] = planner.next_trial,
) -> PlannedTrial | None:
    """The next trial of the session: from the one block, or from whichever interleaved block is due.

    A pure function of its inputs. Void trials do not advance the order: the block whose trial was void is asked again.
    """
    if len(blocks) == 1:
        return plan(blocks[0], protocol, demo_starts, log)
    order = block_order(blocks, protocol)
    scored = sum(1 for record in log if record.scored and record.planned.block in blocks)
    for label in order[scored:] + order[:scored]:
        trial = plan(label, protocol, demo_starts, log)
        if trial is not None:
            return trial
    return None


def checkpoint_label(path: str | Path) -> str:
    """A short name for a checkpoint directory that carries no account or machine name.

    `.../act-tolrad-clustered-v1/checkpoints/080000/pretrained_model`, the layout the training guide downloads into,
    becomes `act-tolrad-clustered-v1@080000`: the run and the step, so that two runs of one layout can be told apart
    in the log. Without a `checkpoints` folder above it, `.../080000/pretrained_model` becomes `080000`.
    """
    path = Path(path)
    step_dir = path.parent if path.name == "pretrained_model" else path
    if step_dir.parent.name == "checkpoints" and step_dir.parent.parent.name:
        return f"{step_dir.parent.parent.name}@{step_dir.name}"
    return step_dir.name


def nearest_for_record(planned_trial: PlannedTrial, placed: Point, demo_starts: Sequence[DemoStart]) -> tuple[int, float]:
    """The demonstration a trial's distance is measured to, and that distance.

    For a policy it is the nearest trained demonstration of the policy's own set. For a replay it is the replayed
    demonstration itself: that is the only one the replay knows about.
    """
    demos = geometry.trained_demos(demo_starts, planned_trial.dataset)
    if planned_trial.condition == "replay":
        for demo in demos:
            if demo.episode_index == planned_trial.replay_episode:
                return demo.episode_index, placed.distance_to(demo.measured)
        raise ValueError(f"replayed episode {planned_trial.replay_episode} is not among the trained demonstrations")
    index, distance = geometry.nearest(placed, demos)
    return demos[index].episode_index, float(distance)


class ScoreSheet:
    """Wraps the prompt used for scoring so that a trial can be declared void in words.

    Typing `void` followed by a reason at the score prompt marks the trial void. The remaining prompts are then
    answered with nothing, at once, so the scoring routine gives up quickly and the arm is not kept waiting.
    """

    def __init__(self, ask: Callable[[str], str]) -> None:
        self._ask = ask
        self.void_reason: str | None = None

    def __call__(self, prompt: str) -> str:
        if self.void_reason is not None:
            return ""
        answer = self._ask(prompt)
        if answer.strip().lower().startswith("void"):
            self.void_reason = answer.strip()[4:].strip(" :-") or "declared void by the scorer"
            return ""
        return answer


def build_record(
    planned_trial: PlannedTrial,
    *,
    placed: Point | None,
    demo_starts: Sequence[DemoStart],
    score: dict[str, Any] | None,
    void_reason: str | None,
    ts_start: str,
    ts_end: str,
    operator: str,
    scorer: str,
    policy_label: str | None,
    checkpoint: str | None,
    execution: dict[str, Any],
    versions: dict[str, str],
    media: dict[str, str],
) -> TrialRecord:
    """Assemble one trial's record from what was planned, measured and scored. Void when it could not count."""
    common = dict(planned=planned_trial, ts_start=ts_start, ts_end=ts_end, placed=placed, operator=operator,
                  scorer=scorer, policy_repo=policy_label, checkpoint=checkpoint, execution=execution,
                  versions=versions, media=media)
    if void_reason is None and score is None:
        void_reason = "left unscored"
    if void_reason is None and placed is None:
        void_reason = "no placement was measured"
    if void_reason is not None:
        return TrialRecord(outcome="void", void_reason=void_reason, notes=(score or {}).get("notes", ""), **common)
    episode, distance = nearest_for_record(planned_trial, placed, demo_starts)
    subgoal = int(score["subgoal"])
    success = subgoal == schema.SUBGOAL_MAX
    return TrialRecord(
        outcome="success" if success else "failure",
        subgoal=subgoal,
        failure=None if success else (score.get("failure") or UNLABELLED),
        notes=score.get("notes", ""),
        nearest_demo_episode=episode,
        nearest_demo_distance_mm=distance,
        **common,
    )


def log_problem(protocol: dict[str, Any], log_path: Path, log: Sequence[TrialRecord], *, pilot: bool) -> str | None:
    """Why this session may not write to this log, or None when it may.

    Counted trials are run only under a frozen protocol and never share a file with pilot trials, so that the analysis
    cannot pick up a trial that was run before the rules were fixed.
    """
    status = str(protocol["protocol"].get("status", ""))
    version = str(protocol["protocol"]["version"])
    named_pilot = log_path.name.startswith("pilot")
    if pilot and not named_pilot:
        return f"a pilot session logs to a file whose name starts with 'pilot', not to {log_path.name}"
    if not pilot and named_pilot:
        return f"{log_path.name} is a pilot log: start the session with --pilot, or give another file"
    if not pilot and status != "frozen":
        return (f"the protocol's status is {status!r}: counted trials are run only after it is frozen "
                f"(config/protocol.toml). Use --pilot for trials that will not be counted")
    others = sorted({record.versions.get("protocol_version", "?") for record in log} - {version})
    if others:
        return f"{log_path.name} already holds trials run under protocol version(s) {others}, not {version!r}"
    return None


def read_audits(datasets: Sequence[str], audit_dir: Path = AUDIT_DIR) -> list[DemoStart]:
    """The audited demonstration starts of the datasets a session needs."""
    starts: list[DemoStart] = []
    for dataset in datasets:
        path = audit_dir / f"{dataset}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no start-state audit at {path}: run scripts/rig/audit_dataset.py for {dataset}")
        starts.extend(schema.read_demo_starts(path))
    return starts


def audit_gaps(demo_starts: Sequence[DemoStart], planned: dict[str, int]) -> list[str]:
    """What the audits lack of their sets, as sentences: `planned` gives each dataset's number of episodes.

    The planner draws its replay demonstrations and measures every distance from the audited starts. An audit of a
    part of a set (33 episodes after one sitting) starts a session without a word, with other demonstrations to
    replay and other distances than the finished set will give.
    """
    told = []
    for dataset, total in sorted(planned.items()):
        have = sorted({start.episode_index for start in demo_starts if start.dataset == dataset})
        if have != list(range(total)):
            told.append(f"the {dataset} audit holds {len(have)} of the set's {total} episodes")
    return told


def replay_set_problem(replay_root: Path, demo_starts: Sequence[DemoStart]) -> str | None:
    """Why the dataset at `replay_root` cannot be the one the clustered audit measured; None when it can.

    A replay sends a demonstration's recorded actions by its episode number: taken from another recording than the
    audit's, the number names another demonstration, with another start.
    """
    import json

    # The audit's last episode, not its count: a start the audit could not measure is a gap in the audit
    # (`audit_gaps` says so), not another recording.
    audited = max((start.episode_index for start in demo_starts if start.dataset == "clustered"), default=-1) + 1
    try:
        held = int(json.loads((replay_root / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"])
    except (OSError, ValueError, KeyError) as error:
        return f"the replay set at {replay_root.name} cannot be read ({error})"
    if held != audited:
        return (f"the replay set at {replay_root.name} holds {held} episodes and the clustered audit ends at "
                f"episode {audited - 1}: they are not the same recording. Audit the set again, or point "
                "--replay-root at the audited one")
    return None


def discard_typed_ahead() -> None:
    """Throw away what was typed into the terminal before a prompt is shown.

    The keys that end a phase are heard by the session's listener, and the terminal, when it is in front, takes the
    same letters as text: an n pressed to end an attempt came back in front of the score ("n5", asked again), and a
    void typed after it was not taken for one (the dry run of 2026-10-06, on a pseudo-terminal).
    """
    try:
        import termios

        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 -- no terminal, or not one that can be flushed: nothing to throw away
        pass


def project_commit() -> str:
    """This repository's commit, with a mark when tracked files differ from it."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=False, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("+changes" if dirty else "")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blocks", nargs="+", required=True, help="protocol blocks to run; several are interleaved")
    parser.add_argument("--log", required=True, help="the trial log (JSON Lines), for example data/trials/main.jsonl")
    parser.add_argument("--policy", action="append", default=[], metavar="CONDITION=DIR",
                        help="a policy's local pretrained_model directory, for example act_clustered=/path/080000/"
                             "pretrained_model; give one for each policy condition the blocks need")
    parser.add_argument("--replay-root", default=None, help="the clustered dataset's directory, for replay trials")
    parser.add_argument("--operator", default="op1", help="a role code, not a name")
    parser.add_argument("--scorer", default="op1", help="a role code, not a name")
    parser.add_argument("--pilot", action="store_true",
                        help="a pilot session: block sizes from config/pilot.toml, a log named pilot*.jsonl, trials "
                             "that are never counted. The only kind of session allowed before the protocol is frozen")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many trials")
    parser.add_argument("--task", default=TASK)
    # How a policy or a replay is run. Left out, each takes the protocol's value ([execution] in the protocol file).
    # Given by hand, a value that differs from the protocol's is allowed in a pilot or a dry run and refused in a
    # counted session.
    execution = parser.add_argument_group("execution settings (default: the protocol's; a counted session takes no "
                                          "other value)")
    execution.add_argument("--attempt-seconds", type=float, default=None)
    execution.add_argument("--fps", type=int, default=None)
    execution.add_argument("--clamp", type=float, default=None, help="largest move per step, degrees, against present")
    execution.add_argument("--handoff-tolerance", type=float, default=None)
    execution.add_argument("--replay-handoff-tolerance", type=float, default=None,
                           help="a replay starts only this close to the recorded first pose, degrees per joint")
    execution.add_argument("--max-handoff-retries", type=int, default=None)
    execution.add_argument("--min-attempt-seconds", type=float, default=None,
                           help="an attempt a key ends sooner than this is logged void: a slip of the key")
    execution.add_argument("--stall-steps", type=int, default=None)
    execution.add_argument("--stall-progress", type=float, default=None)
    execution.add_argument("--stall-fraction", type=float, default=None)
    execution.add_argument("--temporal-ensemble", type=float, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds the cube must stay on target to be accepted")
    parser.add_argument("--typing-guard", type=float, nargs=2, metavar=("BEFORE", "AFTER"),
                        default=[typing_guard.QUIET_BEFORE_S, typing_guard.QUIET_AFTER_S],
                        help="a go key starts a trial only when no other key was heard this many seconds before it "
                             "and after it: the keys are heard from every program, and an n typed in a sentence "
                             "elsewhere must not hand the motors to a policy. The trial starts AFTER seconds after "
                             "the key")
    parser.add_argument("--no-typing-guard", action="store_true",
                        help="take every go key for a go, also one typed in a sentence, as before 2026-10-06")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan, place and pass the gates, but never give the motors to a policy or a replay; "
                             "nothing is logged")
    parser.add_argument("--align-tolerance", type=float, default=5.0)
    parser.add_argument("--align-timeout", type=float, default=180.0)
    parser.add_argument("--preflight-tolerance", type=int, default=50)
    parser.add_argument("--no-display", dest="display", action="store_false", default=True)
    parser.add_argument("--tags-relative-limit", type=float, default=None, metavar="MM",
                        help="how far the two floor tags may read against each other since the table was registered "
                             "before the session refuses (default 3). Only after the kept frame was held against "
                             "the calibration's (analysis/instrument/picture_motion.py) and the tags moved with the "
                             "rest of the picture: then it is the camera that turned")
    parser.add_argument("--view-crept-limit", type=float, default=None, metavar="MM",
                        help="make no attempt once the front camera's picture stands more than this from where the "
                             "session solved its conversion, in millimetres on the floor (the camera moves as it "
                             "warms, and a trial's start is what the session measures); the session is then "
                             "started again. Default: the protocol's, which a counted session must run with. "
                             "0 switches the stop off")
    parser.add_argument("--any-wrist-state", action="store_true",
                        help="run although the wrist camera's picture is not in its usual tone state (the floor "
                             "over the jaws' green, with the arm at rest, above 2.85), or changes it. It is looked "
                             "at when the session starts and before every attempt")
    parser.add_argument("--any-cameras", action="store_true",
                        help="run although the system's list of cameras does not show the study's two at the "
                             "numbers the session takes (scripts/rig/cameras.py prints the list)")
    parser.add_argument("--no-record-video", dest="record_video", action="store_false", default=True)
    parser.add_argument("--no-sounds", action="store_true")
    parser.add_argument("--log-dir", default=None, help="session logs; default data/logs (not committed)")
    parser.add_argument("--allow-other-tooling", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="skip the camera confirmation")
    return parser


#: The protocol's execution settings and the command-line options that can set them.
EXECUTION_ARGS = {
    "control_hz": "fps",
    "time_limit_s": "attempt_seconds",
    "clamp_deg": "clamp",
    "temporal_ensemble_coeff": "temporal_ensemble",
    "handoff_tolerance_deg": "handoff_tolerance",
    "replay_handoff_tolerance_deg": "replay_handoff_tolerance",
    "max_handoff_retries": "max_handoff_retries",
    "min_attempt_s": "min_attempt_seconds",
    "stall_steps": "stall_steps",
    "stall_progress_deg": "stall_progress",
    "stall_fraction": "stall_fraction",
    "view_crept_limit_mm": "view_crept_limit",
}
_WHOLE = ("control_hz", "max_handoff_retries", "stall_steps")


def settle_execution(args: argparse.Namespace, protocol: dict[str, Any], *, counted: bool) -> dict[str, Any]:
    """Fill the session's execution settings from the protocol, and hold a counted session to them.

    Returns the settings the session runs with, keyed as in the protocol. Raises `ValueError` when the protocol states
    none, or when a counted session was given a value that differs from the protocol's.
    """
    if not protocol.get("execution"):
        raise ValueError("the protocol has no [execution] section: a session has no settings to run with")
    given = {key: getattr(args, name) for key, name in EXECUTION_ARGS.items()}
    try:
        settings, changed = session_rules.resolved_execution(protocol, given)
    except KeyError as error:
        raise ValueError(f"the protocol's [execution] section lacks a setting this program needs: {error}") from error
    if changed:
        problems = session_rules.execution_problems(protocol, settings)
        if counted:
            raise ValueError("a counted session runs with the protocol's execution settings and no others: "
                             + "; ".join(problems))
        print("  NOTE -- this session's execution settings differ from the protocol's (allowed here: not counted):")
        for problem in problems:
            print(f"    {problem}")
    for key, name in EXECUTION_ARGS.items():
        setattr(args, name, int(settings[key]) if key in _WHOLE else settings[key])
    return settings


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0911, PLR0912, PLR0915 -- mirrors rollout_session.main
    args = build_parser().parse_args(argv)
    protocol = load_protocol(overlay=PILOT_OVERLAY_PATH if args.pilot else None)
    # A pilot's trials, and a dry run's, are shown under codes of their own: the counted trials' codes are fixed by
    # block and number, and a session that showed them beforehand would tell which of them is a replay.
    plan_next = partial(planner.next_trial, pilot=bool(args.pilot or args.dry_run))

    # Everything that can be refused on paper is refused here, before anything imports lerobot or opens the arm.
    unknown = [block for block in args.blocks if block not in protocol["blocks"]]
    if unknown:
        print(f"REFUSING -- unknown blocks {unknown}; the protocol has {sorted(protocol['blocks'])}", file=sys.stderr)
        return 1
    conditions = sorted({c for block in args.blocks for c in block_conditions(protocol["blocks"][block])})
    policy_dirs: dict[str, Path] = {}
    for item in args.policy:
        condition, _, directory = item.partition("=")
        policy_dirs[condition] = Path(directory).expanduser()
    for condition in conditions:
        if condition == "replay":
            if not args.replay_root or not (Path(args.replay_root).expanduser() / "meta" / "info.json").is_file():
                print("REFUSING -- replay trials need --replay-root pointing at the clustered dataset", file=sys.stderr)
                return 1
        elif condition not in policy_dirs or not (policy_dirs[condition] / "config.json").is_file():
            print(f"REFUSING -- no usable --policy {condition}=DIR (a pretrained_model directory with config.json)",
                  file=sys.stderr)
            return 1
    try:
        needed = sorted({schema.DATASET_OF_CONDITION[c] for c in conditions})
        demo_starts = read_audits(needed)
        gaps = audit_gaps(demo_starts, {dataset: len(layouts.read_layout(LAYOUT_DIR / f"{dataset}.json"))
                                        for dataset in needed})
        if gaps and not (args.pilot or args.dry_run):
            raise ValueError("; ".join(gaps) + ": a counted session needs the finished sets' audits")
        for gap in gaps:
            print(f"NOTE -- {gap}: the demonstrations replayed and every distance are those of this part, not of "
                  "the finished set")
        if "replay" in conditions:
            mismatch = replay_set_problem(Path(args.replay_root).expanduser(), demo_starts)
            if mismatch:
                raise ValueError(mismatch)
        log_path = Path(args.log)
        existing = schema.read_trials(log_path)
        problem = None if args.dry_run else log_problem(protocol, log_path, existing, pilot=args.pilot)
        if problem:
            raise ValueError(problem)
        # After the log's own rules, so that an unfrozen protocol is named as the reason before any setting is.
        execution_settings = settle_execution(args, protocol, counted=not (args.pilot or args.dry_run))
        # A staircase that would ask for a distance the table cannot give stops its block midway: found out now.
        stoppers = session_rules.block_problems(args.blocks, protocol, demo_starts)
        if stoppers:
            raise ValueError("a block of this session could not be run to its end: " + "; ".join(stoppers))
        first = next_planned(args.blocks, protocol, demo_starts, existing, plan=plan_next)
    except (FileNotFoundError, schema.SchemaError, ValueError) as error:
        print(f"REFUSING -- {error}", file=sys.stderr)
        return 1
    if first is None:
        print(f"Nothing to do: blocks {args.blocks} are complete in {log_path}.")
        return 0
    try:
        tooling_commit = lerobot101.check_pin(allow_other=args.allow_other_tooling)
        lerobot101.add_to_path()
        lerobot_line = lerobot101.use_pinned_lerobot(allow_other=args.allow_other_tooling)
    except lerobot101.ToolingError as error:
        print(f"REFUSING -- {error}", file=sys.stderr)
        return 1

    import cv2
    import numpy as np
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.utils import init_logging, log_say

    from tolerance_radius.rig.camera_tool import (
        changed_since,
        checked_stored_view,
        crept_since,
        wrist_changed_since,
        wrist_reading,
    )
    from tolerance_radius.rig.placement_phase import PlacementPhase
    from tolerance_radius.rig.record_layout import off_start_pose
    from tolerance_radius.rig.replay import ReplaySource, load_episode_actions
    from tolerance_radius.rig.vision import VisionError, empty_scene, load_geometry, wrist_other_state

    rollout = lerobot101.load("rollout_session")
    session = lerobot101.load("so101_session")
    viz = lerobot101.load("viz")
    joints = lerobot101.load("joints").JOINTS

    init_logging()

    session_started = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_log = session.start_session_log(log_dir, stamp)
    logging.info("trial_session %s | blocks %s | log %s | clamp %.1f | dry_run=%s | arm tooling %s | %s",
                 stamp, args.blocks, log_path, args.clamp, args.dry_run, tooling_commit[:10], lerobot_line)
    versions = {"project_commit": project_commit(), "protocol_version": str(protocol["protocol"]["version"]),
                "protocol_status": str(protocol["protocol"].get("status", "")),
                "lerobot101_commit": tooling_commit[:10], "lerobot": lerobot_line.split(" from ")[0]}

    # The cameras are taken by number, and a number belongs to whatever camera is plugged in (2026-10-06: the wrist
    # camera had not started, and number 1 was the laptop's own). Asked of the system's list, before anything is
    # opened; the operator's look at the two snapshots further down stays.
    numbers = lerobot101.load("rig")
    told, wrong = camera_list.check_for_session(numbers.FRONT_CAMERA_INDEX, numbers.WRIST_CAMERA_INDEX)
    for line in told:
        logging.info("%s", line)
    if wrong:
        if not args.any_cameras:
            logging.error("the cameras are not the study's two in their places: %s", wrong)
            print(f"\nREFUSING TO RUN -- {wrong}.\n  scripts/rig/cameras.py prints the list; --any-cameras runs "
                  "all the same.")
            return 3
        logging.warning("running although the cameras are not in their places (--any-cameras): %s", wrong)

    if not session.run_preflight(args.preflight_tolerance):
        print("\nREFUSING TO RUN -- the arm is not in a safe state to be torqued.")
        return 1

    play_sounds = not args.no_sounds
    gate = session.AlignmentGate(args.align_tolerance, args.align_timeout, play_sounds=play_sounds)
    class TellingGate(rollout.PolicyHandoffGate):
        """The tooling's gate, which also keeps what its last check saw.

        After a refusal: `wanted`, where the first action wants every joint it could compare, and `refused`, those
        of them that are further from the arm than the tolerance. Both empty after a check that passed, and after a
        refusal that could not compare at all. After a check that passed: `passed_from`, the arm's pose at that
        check, which is the pose the attempt began from (None after a refusal). And `unreal`: the names in the first
        action whose value was no number, when that is why this gate refused (the tooling's had passed it)."""

        refused: dict[str, float] = {}
        wanted: dict[str, float] = {}
        passed_from: dict[str, float] | None = None
        unreal: tuple[str, ...] = ()

        def check(self, action: Any, present: Any, label: str = "") -> bool:
            ok = super().check(action, present, label)
            self.unreal = ()
            if ok:
                # The tooling's gate asks whether the largest move is over the tolerance, and a value that is no
                # number is over nothing: it passes ("handoff ok -- largest move shoulder_pan.pos nan units", the
                # second review of the night of 2026-10-07). Nothing that is not a number is handed to the motors:
                # every name in the action is asked, also one the tooling had nothing to compare with.
                try:
                    unreal = sorted(str(name) for name in action if not math.isfinite(float(action[name])))
                except Exception:  # noqa: BLE001 -- an action that cannot be read as numbers is none to start from
                    unreal = ["an action that could not be read"]
                if unreal:
                    logging.error("%s: handoff REFUSED -- the first action is not a number (%s); the line above, "
                                  "which says ok, is the earlier project's gate", label, ", ".join(unreal))
                    self.unreal = tuple(unreal)
                    ok = False
            try:
                self.passed_from = ({name: round(float(value), 2) for name, value in sorted(present.items())}
                                    if ok else None)
                self.wanted = {} if ok else {name: float(action[name]) for name in action
                                             if name in present and math.isfinite(float(action[name]))}
                self.refused = {name: value for name, value in self.wanted.items()
                                if abs(value - float(present[name])) > self.tolerance}
            except Exception:  # noqa: BLE001 -- what the gate saw is a help and a record, never a reason to stop
                self.refused, self.wanted, self.passed_from = {}, {}, None
            return ok

    handoff = {condition: TellingGate(
        args.replay_handoff_tolerance if condition == "replay" else args.handoff_tolerance) for condition in conditions}
    #: The gate of the trial in hand, for the correction after a refusal to say what it refused; and the countdown
    #: of the attempt in hand, which that correction stops and starts again.
    refusing: list[Any] = [None]
    attempt_clock: list[Any] = [None]
    # A joint pinned this far behind its goal is not catching up, it is blocked.
    stall_lag = args.clamp * args.stall_fraction if args.stall_steps else None

    # Policies are loaded and checked before the robot is built, let alone connected (as in rollout_session).
    loaded: dict[str, tuple[Any, Any, Any]] = {}
    for condition, directory in policy_dirs.items():
        if condition not in conditions:
            continue
        policy, pre, post = rollout.load_policy(None, None, str(directory), args.device)
        rollout.apply_inference_overrides(policy, temporal_ensemble_coeff=args.temporal_ensemble)
        if not rollout.report_input_match(getattr(policy.config, "input_features", {}) or {}, session.CAMERAS,
                                          len(joints)):
            print(f"REFUSING -- the policy for {condition} does not match this rig's cameras or joints.",
                  file=sys.stderr)
            return 1
        loaded[condition] = (policy, pre, post)

    robot = SO101Follower(
        SO101FollowerConfig(
            port=session.FOLLOWER_PORT,
            id="my_follower",
            cameras=session.CAMERAS,
            # Never leave the arm limp, and never let one step be a lunge.
            disable_torque_on_disconnect=False,
            max_relative_target=args.clamp,
        )
    )
    teleop = SO101Leader(SO101LeaderConfig(port=session.LEADER_PORT, id="my_leader"))

    kb = None
    #: What the session ends with: 0, or 3 when it ran no trial because a camera's picture was not right.
    ended = 0

    guard = None  # set with the keyboard; the `finally` below reads it whatever happened before
    arm_state = "untouched"
    done = 0
    try:
        teleop.connect()
        robot.connect()

        streaming, detail = session.verify_camera_stream(robot, args.fps)
        if not streaming:
            logging.error("camera not streaming: %s", detail)
            print(f"\nREFUSING TO RUN -- a camera connected but will not stream.\n  {detail}")
            return 3

        # The conversion of this session is solved right after this answer, from the floor tags as the next frames
        # show them: a cube standing beside a tag moved that reading by a few tenths of a millimetre in the first
        # pilot (the review of the night of 2026-10-07).
        print("\nThe cube must be OFF the table now: this session measures the table from its next frames."
              "\nCamera snapshots -- confirm each shows what its name says:")
        if not rollout.confirm_cameras(robot, log_dir, stamp, args.yes):
            print("aborted -- camera mapping not confirmed")
            return 1

        def grab() -> np.ndarray:
            """The front camera's current frame, BGR, upright: the robot's own stream, so no second camera client."""
            return cv2.cvtColor(np.asarray(robot.get_observation()["front"]), cv2.COLOR_RGB2BGR)

        # The arm is at rest and clear of both tags: the picture must still match the stored calibration.
        try:
            view = checked_stored_view(grab, relative_limit_mm=args.tags_relative_limit)
        except VisionError as error:
            logging.error("the camera view cannot be used: %s", error)
            print(f"\nREFUSING TO RUN -- {error}")
            return 3
        # The conversion solved just now is the one every placement of this session is measured through.
        # The limit is the protocol's (`view_crept_limit_mm`), settled with the other execution settings above.
        crept_limit = float(args.view_crept_limit)
        if crept_limit > 0:
            logging.info("no attempt once the front camera's picture has crept more than %g mm from where this "
                         "session solved its conversion", crept_limit)
        else:
            logging.warning("--view-crept-limit 0: the front camera's creep stops nothing in this session")
            print("\n  NOTE: --view-crept-limit 0. How far the camera's picture has crept is written into this "
                  "session's log for every trial and stops nothing.")

        def grab_wrist() -> np.ndarray:
            """The wrist camera's current frame, BGR, as `grab` gives the front camera's."""
            return cv2.cvtColor(np.asarray(robot.get_observation()["wrist"]), cv2.COLOR_RGB2BGR)

        def said_of_wrist(reading: float | None) -> str:
            return ("not judged: its picture is not the rest picture (the jaws over the floor)" if reading is None
                    else f"the floor over the jaws' green reads {reading:.2f}")

        # The wrist camera's tone state, by its one number with the arm at rest in its start pose
        # (`vision.wrist_contrast`): the policy must be run in the picture it was trained in. A first look here,
        # which only says (the arm rests where the last session left it); the look that refuses is at the start pose.
        wrist_first = wrist_reading(grab_wrist)
        logging.info("wrist camera, before the warm-up: %s", said_of_wrist(wrist_first))
        wrist_wrong = wrist_other_state(wrist_first)
        if args.any_wrist_state:
            # Said in every session that takes the option, a counted one too: a trial's record does not carry it.
            logging.warning("--any-wrist-state: the wrist camera's tone state stops nothing in this session")
            print("\n  NOTE: --any-wrist-state. The wrist camera's tone state is written into this session's log "
                  "(not into the trials' records) and stops nothing. Write in the journal that it was used.")
        if wrist_wrong and not args.any_wrist_state:
            logging.warning("before the warm-up: %s", wrist_wrong)
            print(f"\n  NOTE: {wrist_wrong}.\n  That is its other tone state if the arm is resting in its start "
                  "pose. It is asked again at the start pose, and no trial is run in that state. To open the camera "
                  "afresh now: press q in the warm-up, and start the command again.")

        def joints_now() -> dict[str, float]:
            """Where the arm's joints are, by the names the recorder's start-pose rule goes by. Commands nothing."""
            return {name.split(".")[0]: float(value)
                    for name, value in rollout.present_joints(robot.get_observation()).items()}

        def hands_out(text: str) -> None:
            log_say("Hands out of the picture", play_sounds)
            print(f"\n  {text}")

        procs = make_default_processors()
        # A go that is a letter of somebody's sentence must not start a trial: every key's time is noted on its way to
        # the session's flags, and a placement's go is taken only when it came by itself (`typing_guard`).
        guard = None if args.no_typing_guard else typing_guard.TypingGuard(*args.typing_guard)
        # One listener for the whole session, started here, before any window is shown: a listener started after
        # the placement's window ends the program on this computer (`typing_guard.Keyboard`). The arm tooling's
        # PausableKeyboard starts a new one after every score.
        kb = typing_guard.Keyboard(guard)
        events = kb.events
        if guard is None:
            logging.info("typing guard: off")
        else:
            logging.info("typing guard: a go starts a trial when no other key came %g s before it and %g s after it",
                         *args.typing_guard)
        if args.display:
            viz.fresh_rerun_view(f"trials_{stamp}")
        logging.info("keys: n / Right = end phase, q = stop")

        dataset_features = rollout.build_dataset_features(robot, procs)
        recorder = rollout.AttemptRecorder(REPO_ROOT / "data" / "recordings" / stamp, stamp, fps=args.fps,
                                           display=args.display)
        record_video = args.record_video and not args.dry_run

        executors: dict[str, tuple[Any, Callable[[], dict]]] = {}
        for condition, (policy, pre, post) in loaded.items():
            executors[condition] = (policy, rollout.make_predict(policy, robot, pre, post, args.task, dataset_features,
                                                                 device=args.device, on_observation=recorder.write))
        replays: dict[int, ReplaySource] = {}

        def replay_for(episode: int) -> ReplaySource:
            if episode not in replays:
                actions, names = load_episode_actions(Path(args.replay_root).expanduser(), episode)
                replays[episode] = ReplaySource(actions, names, observe=robot.get_observation,
                                                on_observation=recorder.write, events=events)
            return replays[episode]

        def reenter_teleop(label: str, told: str | None = None) -> bool:
            # After a refused handoff the tooling asks for the arm to be brought closer ("... -- reposition") and
            # arms the executor again as soon as this returns: the n that ends that phase starts the policy. It is
            # held to the placement's rule, pressed by itself, and the phase is run again when it was typed.
            starts_the_policy = label.endswith("-- reposition")
            # The phase's name is short and what to do is said once, on a line of its own. The tooling draws the
            # name into a countdown line it rewrites in place, made for 78 columns: with the instruction in the name
            # the line was 95 characters, wrapped in a terminal 87 wide, and every rewrite left a row behind
            # (the first run at the rig, 2026-10-07: the placements' lines scrolled away under them).
            if told:
                print(f"\n  {told}")
            # The correction after a refused start says which joint, where it is and where it is wanted, and goes
            # on saying it while the arm is moved: the tooling's countdown line shows it in place of the phase's
            # name. Where the joint is comes from the control loop's own reads of the arm, kept as they pass (the
            # line is drawn from another thread, which must not ask the arm anything).
            shown: Any = label
            watched = False
            # Every joint the gate will judge is watched, not only those it refused for: while the wrist was turned
            # the base drifted out, unseen, and the next check refused for it (the first pilot, 19:36).
            refused = dict(getattr(refusing[0], "refused", None) or {}) if starts_the_policy else {}
            wanted = dict(getattr(refusing[0], "wanted", None) or {}) if refused else {}
            unreal = tuple(getattr(refusing[0], "unreal", None) or ()) if starts_the_policy else ()
            if unreal:
                # No pose of the arm puts this right, and the tooling's own line asks for one all the same.
                print(f"\n  The first action was not a number ({', '.join(unreal)}): moving the arm cannot help."
                      "\n  Press q to end the session, and say what the terminal shows. (n asks for a first action "
                      "once more.)")
            if wanted:
                within = float(refusing[0].tolerance)
                seen: dict[str, float] = {}
                read_arm = robot.get_observation

                def observing() -> Any:
                    observation = read_arm()
                    try:
                        for name in wanted:
                            seen[name] = float(observation[name])
                    except Exception:  # noqa: BLE001 -- the readout is a help; the loop's read is the loop's
                        pass
                    return observation

                try:
                    at = rollout.present_joints(read_arm())
                    for name in sorted(refused, key=lambda name: -abs(float(at[name]) - refused[name])):
                        print(f"\n  {name.removesuffix('.pos').replace('_', ' ')}: it is at {float(at[name]):+.1f} "
                              f"and is wanted at {refused[name]:+.1f} (within {within:g}). Move it there with the "
                              "leader: the line below shows the joint that is furthest from where it is wanted. "
                              "Then press n by itself.")
                except Exception:  # noqa: BLE001 -- as above
                    pass
                shown = LiveLabel(label, lambda: handoff_readout(wanted, seen, within))
                robot.get_observation = observing
                watched = True

            def correcting(label: str, shown: Any, starts_the_policy: bool) -> bool:
                while True:
                    # A stale exit_early from any source would make this phase end in the second it began,
                    # stranding the arm where it was: teleop_phase clears the flag on exit but not on entry.
                    events["exit_early"] = False
                    # No timer: the phase ends on the operator's key, so nothing downstream starts by itself.
                    # display_data stays False as in rollout_session: the viewer must never sit inside a control
                    # loop.
                    if isinstance(shown, LiveLabel):
                        shown.live = True
                    try:
                        ran = session.teleop_phase(shown, robot, teleop, procs, args.fps, float("inf"), events,
                                                   play_sounds, gate, display_data=False)
                    finally:
                        if isinstance(shown, LiveLabel):
                            shown.live = False
                    # The stop key ends this phase like any key. Report that as "did not run": `attempt_cycle`
                    # arms the executor again after a reposition phase and would send its first action before
                    # looking at the stop.
                    if not ran or events.get("stop_recording"):
                        return False
                    if not starts_the_policy:
                        return True
                    # Only the go ends a reposition into the policy. The re-record key ends the phase like any key and
                    # is no go key, so the guard has nothing to say of it: r, or the left arrow, handed the motors to
                    # the policy without a word (the review of the night of 2026-10-06).
                    if events.get("rerecord_episode"):
                        events["rerecord_episode"] = False
                        print("\n  That key is not the go. Turn the joint it named if it still needs it, then press n "
                              "by itself.")
                        continue
                    if typing_guard.phase_go(guard, told=lambda text: print(f"\n  {text}")):
                        # A second press while the go was being looked at must not end the attempt that begins now.
                        events["exit_early"] = False
                        return True

            # The attempt's own countdown steps aside for as long as the correction lasts.
            stood_aside = attempt_clock[0] if starts_the_policy else None
            if stood_aside is not None:
                stood_aside.__exit__(None, None, None)
                attempt_clock[0] = None
            goes_on = False
            try:
                goes_on = correcting(label, shown, starts_the_policy)
                return goes_on
            finally:
                if watched:
                    shown.live = False
                    vars(robot).pop("get_observation", None)
                if stood_aside is not None and goes_on:
                    attempt_clock[0] = session.PhaseClock(stood_aside.label, stood_aside.seconds,
                                                          stood_aside.play_sounds)
                    attempt_clock[0].__enter__()

        if not reenter_teleop(WARM_UP, "Bring the arm to the start pose, then press n."):
            events["stop_recording"] = True

        phase = None
        if not events.get("stop_recording"):
            log_say("Cube off the table", play_sounds)
            kb.pause()  # typing must not reach the session's keys
            try:
                discard_typed_ahead()
                input("\n  Take the cube OFF the table, hands out of the picture, then press Enter. ")
            finally:
                kb.resume()
            view = view.with_background(empty_scene(grab)).with_cube_region(load_geometry()["grid"])
            # The wrist camera's reading this session's attempts are held against, with the arm at its start pose,
            # and that pose: the reading says something only there.
            rest_pose = joints_now()
            wrist_start = wrist_reading(grab_wrist)
            logging.info("wrist camera, arm at its start pose: %s", said_of_wrist(wrist_start))
            wrist_wrong = wrist_other_state(wrist_start)
            if wrist_start is None:
                print("\n  NOTE: the wrist camera's picture could not be judged (it is not the rest picture: the "
                      "jaws over the white floor). Its tone state is not watched in this session.")
            elif wrist_wrong and not args.any_wrist_state:
                logging.error("%s", wrist_wrong)
                log_say("Wrist camera in its other state. No trial is run.", play_sounds)
                print(f"\n  Before any trial, {wrist_wrong}.\n  Start the command again: the camera is opened "
                      "afresh. If it comes back the same, watch it by itself (scripts/rig/measure.py wrist) while "
                      "you try its plug and its port, or run all the same with --any-wrist-state.")
                events["stop_recording"] = True
                ended = 3
            elif wrist_wrong:
                logging.warning("%s (--any-wrist-state)", wrist_wrong)
            # The placement picture goes into a window of its own, as in the recorder: in the session's viewer the
            # live cameras stood still after the first placement (2026-10-06). Without a display, no picture.
            # The placement's window takes no key in a trial (its own space, v and x are for the camera tools): the
            # one go is the session's key, which the typing guard looks at. The window is still kept alive.
            def window_alive(milliseconds: int) -> int:
                cv2.waitKey(milliseconds)
                return -1

            unseen = ({"wait_key": window_alive} if args.display
                      else {"show": lambda *_: None, "wait_key": lambda _ms: -1})
            said_early = [float("-inf")]

            def not_steady() -> None:
                # A go pressed while the cube is not steady is dropped at once, and said (at most every few
                # seconds: a typed sentence holds several): the same key a moment later would have been taken.
                if typing_guard.forget_go(events, guard) and time.monotonic() - said_early[0] > 5.0:
                    said_early[0] = time.monotonic()
                    print("\n  That n came while the cube was not steady in its ring: it is not the go. Press n "
                          "when the picture says steady.")

            phase = PlacementPhase(view, grab, tolerance_mm=protocol["placement"]["tolerance_mm"], hold_s=args.hold,
                                   confirm=lambda: typing_guard.guarded_go(
                                       events, guard, told=lambda text: print(f"\n  {text}")),
                                   accept_hint="press n to start the trial", forget=not_steady,
                                   should_stop=lambda: bool(events.get("stop_recording")),
                                   grid=load_geometry()["grid"], **unseen)

        while not events.get("stop_recording"):
            if args.limit is not None and done >= args.limit:
                break
            trial = next_planned(args.blocks, protocol, demo_starts, schema.read_trials(log_path), plan=plan_next)
            if trial is None:
                print("\nAll planned trials of these blocks are done.")
                break
            # What the people at the rig see is a running number and the blind code. The trial's own name carries its
            # block, and a block can be a condition (a policy against the replay), so it goes into the log only.
            label = f"Trial {done + 1} [{trial.blind_code}]"
            # In a pilot the distance is said: its rules are about the trials at a demonstration's own start, and
            # those do not come first. In a counted session nothing but the target is shown.
            distance = (f"   (pilot: {trial.level_mm:g} mm from the nearest demonstration)"
                        if (args.pilot or args.dry_run) and trial.level_mm is not None else "")
            print(f"\n{label}: target ({trial.target.x_mm:.1f}, {trial.target.y_mm:.1f}) mm{distance}")
            log_say("Place the cube", play_sounds)
            # The attempt is run in the picture the camera gives now, and that must still be the one the session
            # started on and the policy was trained in: the front camera's stream changed its zoom, its exposure
            # and its tone range in the middle of sessions on 2026-10-06. It is looked at once, at this moment, and
            # nothing is waited out: what follows a clean look is an arm that moves, and a look that became clean
            # because a hand was lifted off a tag must not start it (the review of the night of 2026-10-06). A look
            # that is not clean sends the trial back to its placement, where the go is asked for again.
            looks_refused = 0
            crept_too_far = False
            crept: float | None = None
            crept_before: float | None = None      # a reading over the limit at an earlier go of this trial
            while True:
                # A press of n, or of the re-record key, left over from the reset must not reach this placement: the
                # one would confirm it, the other would swallow the go that is meant.
                events["exit_early"] = False
                events["rerecord_episode"] = False
                placed = phase.run(trial.target, label)
                if not placed.accepted:
                    break
                another_picture = changed_since(view, grab, tries=1, patience_s=0.0)
                if not another_picture:
                    # ... and how far it has crept since this session solved its conversion. The camera moves as it
                    # warms, by more than a millimetre in half an hour from cold (2026-10-07), and the position just
                    # measured, which is this trial's start in every later distance, is off by about that much. The
                    # reading of the look that lets the attempt go is written into the trial's record.
                    crept = crept_since(view, grab)
                    logging.info("%s: the front camera's picture stands %s from where this session solved its "
                                 "conversion", label, "an unknown distance" if crept is None else f"{crept:.2f} mm")
                    if crept_limit <= 0:
                        break                      # the stop is switched off: whatever it read, the attempt goes on
                    if crept is None:
                        # A look that cannot say is a look that is not clean: no attempt from a start whose
                        # measurement nobody can vouch for (until the night of 2026-10-07 the attempt went on).
                        another_picture = ("how far the front camera's picture has crept since this session solved "
                                           "its conversion could not be read (a floor tag hidden for a moment, or no "
                                           "frame)")
                    elif crept <= crept_limit:
                        break
                    elif crept_before is not None:
                        # Over the limit at a second go of this trial, and at no go under it in between (a reading
                        # under the limit lets the attempt go): the camera has crept, and the session ends.
                        crept_too_far = True
                        another_picture = (f"the front camera's picture has crept {crept:.2f} mm since this session "
                                           f"solved its conversion (allowed {crept_limit:g}), after "
                                           f"{crept_before:.2f} mm at an earlier go of this trial: the camera moves as it warms, "
                                           "and the position measured for this trial would be off by about that "
                                           "much")
                        break
                    else:
                        # Over the limit once: asked again at a new go, as for any look that is not clean. The
                        # reading follows a shadow at a tag's edge by a few tenths, and a reading that changes
                        # between two looks is something standing in the picture: nothing moves on its account, and
                        # nothing moves without a new go (the second review of the night of 2026-10-07).
                        crept_before = crept
                        logging.warning("%s: before the attempt, the front camera's picture reads %.2f mm from this "
                                        "session's conversion (allowed %g): asked again at a new go", label, crept,
                                        crept_limit)
                        log_say("Hands out of the picture. Then press n again.", play_sounds)
                        print(f"\n  Before the attempt, the front camera's picture reads {crept:.2f} mm from where "
                              f"this session solved its conversion (allowed {crept_limit:g}).\n  Nothing has moved. "
                              "Hands and the arm out of the picture, then press n again: a second reading over the "
                              "limit ends the session, which is then started again.")
                        continue
                looks_refused += 1
                logging.warning("%s: before the attempt, %s (look %d of %d)", label, another_picture, looks_refused,
                                PICTURE_LOOKS_BEFORE_STOP)
                if looks_refused >= PICTURE_LOOKS_BEFORE_STOP:
                    break
                log_say("Hands out of the picture. Then press n again.", play_sounds)
                print(f"\n  Before the attempt, {another_picture}.\n  Nothing has moved. Hands and the arm out of "
                      "the picture, then press n again.")
            if not placed.accepted:
                logging.warning("%s: placement ended as %s", label, placed.status)
                break
            logging.info("%s: placed at (%.1f, %.1f) mm, %.1f mm from target", label, placed.measured.x_mm,
                         placed.measured.y_mm, placed.error_mm)
            if not another_picture:
                # ... and the wrist camera's, by its one number against this session's first reading: only with
                # the arm about where it rested for that reading (the recorder's limits), since anywhere else the
                # number says nothing; and a reading that cannot be judged says nothing either.
                at_rest = wrist_start is not None and not off_start_pose(rest_pose, joints_now(),
                                                                         *WRIST_ASKED_WITHIN_DEG)
                wrist_now, wrist_wrong = wrist_changed_since(wrist_start, grab_wrist) if at_rest else (None, None)
                logging.info("%s: wrist camera before the attempt: %s", label,
                             said_of_wrist(wrist_now) if at_rest else "not asked (the arm is not at the pose of the "
                             "session's first reading, or that reading could not be judged)")
                if wrist_wrong and args.any_wrist_state:
                    logging.warning("%s: %s (--any-wrist-state)", label, wrist_wrong)
                elif wrist_wrong:
                    another_picture = wrist_wrong
            if another_picture and crept_too_far:
                logging.error("%s: %s; no attempt is made", label, another_picture)
                log_say("The camera has crept. No attempt. Start the session again.", play_sounds)
                print(f"\n  Before the attempt, {another_picture}.\n  No attempt is made and nothing is logged; "
                      "the session stops here.\n  Take the cube off the table, then run the same command again: a "
                      "new session solves the conversion afresh from the floor tags, as its first frames show them, "
                      "and goes on with this trial. Nothing is wrong with the rig.")
            elif another_picture:
                logging.error("%s: %s; no attempt is made", label, another_picture)
                log_say("Camera picture changed. No attempt.", play_sounds)
                print(f"\n  Before the attempt, {another_picture}.\n  No attempt is made and nothing is logged; "
                      "the session stops here.\n  Run scripts/rig/measure.py check, say what it prints, and run "
                      "the same command again.")
            if another_picture:
                # The session ends now, with the arm at rest where the reset left it: no cool-down to wait in.
                events["stop_recording"] = True
                ended = 3
                break

            if trial.condition == "replay":
                source = replay_for(trial.replay_episode)
                executor, predict = source, source.predict
            else:
                executor, predict = executors[trial.condition]
            steps = [0]
            # When the first step was sent and when the latest one was, on the monotonic clock: what the control
            # loop's own duration is taken from. The record's two times also span the handoff gate and the videos.
            sent_at = [0.0, 0.0]

            def on_step(index: int, _action: dict, steps: list[int] = steps, sent_at: list[float] = sent_at) -> None:
                steps[0] = index + 1
                sent_at[1] = time.monotonic()
                if index == 0:
                    sent_at[0] = sent_at[1]

            # Where the arm starts from, for the record: the policy takes its joints as input, so the start pose is
            # part of the reset. Reading the arm commands nothing.
            start_pose = {name: round(float(value), 2)
                          for name, value in sorted(rollout.present_joints(robot.get_observation()).items())}

            log_say("Trial", play_sounds)
            events["exit_early"] = False
            if events.get("stop_recording"):
                # Asked to stop since the placement was accepted (a replay's actions may just have been read from
                # disk). `attempt_cycle` sends the first action before it looks at this flag, so it is not entered.
                logging.warning("%s: stopped before the attempt began", label)
                break
            if record_video:
                recorder.start(trial.trial_id)
            ts_start = now()
            # The attempt's countdown is held where a correction can reach it: while the arm is moved after a refused
            # start that line gives way to the correction's own (two countdowns rewrote one line in turn until
            # 2026-10-07), and it begins anew when the correction is over.
            refusing[0] = handoff[trial.condition]
            refusing[0].passed_from = None
            attempt_clock[0] = session.PhaseClock(label, args.attempt_seconds, play_sounds)
            attempt_clock[0].__enter__()
            try:
                ran = rollout.attempt_cycle(
                    robot=robot, policy=executor, predict=predict, gate=handoff[trial.condition], fps=args.fps,
                    seconds=args.attempt_seconds, label=label, events=events, max_retries=args.max_handoff_retries,
                    reenter_teleop=reenter_teleop, dry_run=args.dry_run, stall_lag=stall_lag,
                    stall_steps=args.stall_steps, stall_progress=args.stall_progress, on_step=on_step,
                )
            finally:
                if attempt_clock[0] is not None:
                    attempt_clock[0].__exit__(None, None, None)
                    attempt_clock[0] = None
                refusing[0] = None
            # The pose the attempt began from is the one the gate passed at. After a correction that is another pose
            # than the one read above, before the gate's first look: until the night of 2026-10-07 a corrected trial's
            # record carried the pose that had been refused (8 of the first pilot's 21 scored trials).
            start_pose = getattr(handoff[trial.condition], "passed_from", None) or start_pose
            # Read here: a stop pressed while the videos are being finished came after the attempt, and the attempt
            # is scored.
            stopped_in_the_attempt = bool(events.get("stop_recording"))
            videos = recorder.finish() if record_video else []
            ts_end = now()
            done += 1

            if not args.dry_run:
                # Scored here, as in rollout_session: the attempt is over, the arm holds its last goal, and the gated
                # re-entry has not run yet. Nothing below may raise or block.
                sheet = ScoreSheet(input)
                score = None
                # Not handed over: the gate refused to the end, or the session was stopped while the start pose was
                # being corrected (until the night of 2026-10-07 that too was logged as the gate's refusal).
                void_reason = None if ran else (
                    "the session was stopped while the start pose was being corrected after a refusal; no attempt "
                    "was made" if stopped_in_the_attempt
                    else "the first action was not a number; no attempt was made"
                    if getattr(handoff[trial.condition], "unreal", None)
                    else "the handoff gate refused the start pose")
                # Nothing ends an attempt by itself this soon (the stall rule needs more steps, the time limit is far
                # off): it was a key. Too short to have tested anything, so it is void and asked again.
                slip = ran and steps[0] < args.fps * args.min_attempt_seconds
                # The stop key ends an attempt like any key. What was cut short by it was not tried to its end:
                # scored, it counted as whatever the scorer typed (a replay stopped after 11 of 40 steps was
                # logged a success in the dry run of 2026-10-06). Void, asked again by the plan, and not scored.
                stopped = ran and stopped_in_the_attempt
                if stopped:
                    void_reason = f"the session was stopped during the attempt, after {steps[0] / args.fps:.1f} s"
                    print(f"\n  {label}: {void_reason}. Logged void; it will be asked again.")
                elif slip:
                    void_reason = (f"ended by a key {steps[0] / args.fps:.1f} s after it began "
                                   f"(under {args.min_attempt_seconds:g} s)")
                    print(f"\n  {label}: {void_reason}. Logged void; it will be asked again.")
                elif ran:
                    discard_typed_ahead()
                    score = rollout.score_quietly(kb, sheet)
                    void_reason = sheet.void_reason
                record = build_record(
                    trial, placed=placed.measured, demo_starts=demo_starts, score=score, void_reason=void_reason,
                    ts_start=ts_start, ts_end=ts_end, operator=args.operator, scorer=args.scorer,
                    policy_label=None if trial.condition == "replay" else trial.condition,
                    checkpoint=None if trial.condition == "replay" else checkpoint_label(policy_dirs[trial.condition]),
                    # Every setting of the protocol's [execution] section as this session ran it, then what belongs
                    # to this trial alone.
                    execution={**execution_settings, "steps_sent": steps[0],
                               # From the first step sent to the last: the loop alone, for the achieved rate.
                               "loop_s": round(sent_at[1] - sent_at[0], 3) if steps[0] >= 2 else 0.0,
                               "device": args.device,
                               "handoff_tolerance_used_deg": handoff[trial.condition].tolerance,
                               "start_pose_deg": start_pose,
                               # How far the camera's picture stood from the session's conversion before this
                               # attempt (None: it could not be read): what the measured start may be off by.
                               "view_crept_mm": None if crept is None else round(crept, 2)},
                    versions=versions,
                    media={Path(v).stem.rsplit("_", 1)[-1]: str(Path(v).relative_to(REPO_ROOT)) for v in videos},
                )
                schema.append_trial(log_path, record)
                logging.info("%s logged as %s", label, record.outcome)

            if events.get("stop_recording"):
                break
            if not reenter_teleop(RESET, "Cube out of the basket, arm back to the start pose, then press n."):
                break

        if not events.get("stop_recording"):
            reenter_teleop(COOL_DOWN, "Park the arm at its final rest pose, then press n.")

    except KeyboardInterrupt:
        logging.info("interrupted by operator")
        return 130
    finally:
        log_say("Stop", play_sounds, blocking=True)
        arm_state = session.settle_arm(robot)
        if teleop.is_connected:
            teleop.disconnect()
        if kb is not None:
            kb.stop()
        if guard is not None and (guard.refused or guard.unheard):
            logging.info("typing guard: %d go key(s) refused as typing, %d let through unheard",
                         guard.refused, guard.unheard)
        logging.info("cleanup complete -- arm %s, %d trial(s) run", arm_state, done)
        # Each error is a frame the system dropped, and now and then the stream comes out of one in another state
        # (2026-10-06). The arm is settled; the figure goes into this session's log.
        dropped = usb_errors.over_session(session_started, time.time(), {camera_list.FRONT_NAME: "front camera",
                                                                         camera_list.WRIST_NAME: "wrist camera"})
        logging.info("%s", dropped)
        print(f"\n{dropped}")
        print(f"\nsession log: {session_log}")
        print(f"trial log:   {log_path}")
        print("\n" + session.arm_state_line(arm_state))

    return ended


def cli(argv: list[str] | None = None) -> int:
    """Turn a crash into an exit code with the traceback in the session log, as the recorder's cli does.

    `main`'s own `finally` has run by then, so the arm is settled. The launcher ends with `leave(cli())`: an
    exception that escaped here would skip that way out of the process.
    """
    try:
        return main(argv) or 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.exception("SESSION FAILED")
        print("\nSESSION FAILED -- traceback above, and in the session log.")
        return 2
