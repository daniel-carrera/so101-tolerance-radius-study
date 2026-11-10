"""A rig made of fakes, for running the two arm sessions' `main()` from start to finish with no hardware.

`FakeRig` stands in for everything a session touches at the bench. A test builds one, scripts the person at the rig
with `rig.operator`, runs `rig.record_layout([...])` or `rig.run_trials([...])`, and gets back a `Run`: the exit code,
an ordered trace of what happened, everything a person at the rig could have seen or heard, and a list of safety
problems (`Run.problems`) that is empty when the session kept its promises.

What is REAL in a run (the code that will drive the arm, executed as written):

* `record_layout.main` and `trial_session.main`, and this project's `PlacementPhase`, `TableView.measure` with the
  proven cube detector, `empty_scene`, `ReplaySource`, `load_episode_actions`, the planner, the schema, `ScoreSheet`,
  `build_record`, `append_placement`;
* the arm tooling, loaded through `lerobot101.load`: `teleop_phase`, `AlignmentGate` and `wait_for_alignment`,
  `PhaseClock`, `run_preflight` and `preflight_problems`, `settle_arm` and `seed_goal_to_present`,
  `verify_camera_stream` (for 0.05 s instead of 2 s), `record_episode`, `save_episode_announced`, `attempt_cycle`,
  `first_action_for_handoff`, `run_attempt`, `PolicyHandoffGate`, `PausableKeyboard`, `score_quietly`,
  `confirm_cameras`, `report_input_match`, `build_dataset_features`;
* the pinned LeRobot's `record_loop` with its cycle timer, the default processors, the dataset feature builders,
  `build_dataset_frame`, `validate_frame`, `VideoEncodingManager`, `init_keyboard_listener` with its key mapping, and
  `ensure_safe_goal_position` (the follower's per-step clamp).

What is FAKED (the boundary of the hardware and of the heavy libraries):

* the follower, the leader and their servo buses (`FakeFollower`, the leader class, `Arm`): goals, present positions
  and torque are plain numbers, and the arm travels toward its goal a bounded distance per step;
* the cameras: the front picture is drawn from where the cube is on the table, in RGB as LeRobot delivers it;
* `LeRobotDataset`: in memory, but called through the real signatures and with the real frame validation;
* policy loading and inference (`load_policy`, `apply_inference_overrides`, `make_predict`), and by default the
  attempt recorder;
* the operating system's key hook (`create_key_listener`), the terminal (`input`, whether stdin is a terminal),
  speech (`log_say`, `say`), the viewer (`fresh_rerun_view`, `RerunDisplay`, OpenCV windows, the in-loop display);
* the stored-view check, the two version pins (checked for real once, in `prepare`), and where files go.

Nothing here can reach hardware: the session never gets a real robot class, and `cv2.VideoCapture` and
`serial.Serial` are replaced by tripwires for the length of a run.

Flows run at 200 Hz instead of 30 Hz so that a whole session takes a fraction of a second. The pacing code is
LeRobot's own either way.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import importlib.util
import inspect
import io
import json
import logging
import os
import sys
import tempfile
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius import layouts, schema
from tolerance_radius.config import PILOT_OVERLAY_PATH, REPO_ROOT, load_protocol
from tolerance_radius.rig import lerobot101
from tolerance_radius.schema import Point

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
#: Close to the start pose of August's demonstrations (degrees; the gripper is 0 to 100).
START_POSE = {"shoulder_pan": -3.0, "shoulder_lift": -100.0, "elbow_flex": 96.0, "wrist_flex": 75.0,
              "wrist_roll": -90.0, "gripper": 10.0}
TASK = "pick up the orange cube and place it in the basket"

#: The synthetic camera: two pixels per millimetre, table origin at pixel (100, 60), as in `test_rig_vision`.
PX_TO_MM = np.array([[0.5, 0.0, -50.0], [0.0, 0.5, -30.0], [0.0, 0.0, 1.0]])
FRAME_SHAPE = (480, 640, 3)
TABLE_BGR = (150, 150, 150)
ORANGE_BGR = (0, 128, 255)
CUBE_PX = 60

FPS = 200


# --------------------------------------------------------------------------------------------------------------------
# Where the arm tooling is, also from inside a worktree
# --------------------------------------------------------------------------------------------------------------------


def _roots() -> tuple[Path | None, Path | None]:
    """The arm tooling and the LeRobot checkout: from the environment, beside this checkout, or, when this checkout
    is a worktree under `.worktrees/`, beside the repository it belongs to."""
    tooling = [Path(os.environ[name]) for name in ("LEROBOT101_ROOT",) if os.environ.get(name)]
    checkout = [Path(os.environ[name]) for name in ("LEROBOT_CHECKOUT",) if os.environ.get(name)]
    bases = [REPO_ROOT]
    if REPO_ROOT.parent.name == ".worktrees":
        bases.append(REPO_ROOT.parents[1])
    for base in bases:
        tooling.append(base.parent / "lerobot-101")
        checkout.append(base.parent.parent / "so-101" / "lerobot")
    found_tooling = next((p.resolve() for p in tooling if (p / "scripts" / "data" / "placement.py").is_file()), None)
    found_checkout = next((p.resolve() for p in checkout if (p / "src" / "lerobot" / "__init__.py").is_file()), None)
    return found_tooling, found_checkout


TOOLING_ROOT, LEROBOT_ROOT = _roots()
_HAVE_LIBRARIES = all(importlib.util.find_spec(name) is not None for name in ("cv2", "pandas", "pyarrow"))
READY = _HAVE_LIBRARIES and TOOLING_ROOT is not None and LEROBOT_ROOT is not None
WHY_NOT = "needs OpenCV, pandas, the arm tooling and the LeRobot checkout (run with the rig's interpreter)"


@contextlib.contextmanager
def located_tooling():
    """Point `lerobot101` at the arm tooling and the LeRobot checkout for the length of the block."""
    wanted = {"LEROBOT101_ROOT": str(TOOLING_ROOT), "LEROBOT_CHECKOUT": str(LEROBOT_ROOT)}
    before = {name: os.environ.get(name) for name in wanted}
    os.environ.update(wanted)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _without(name: str):
    """Run the block with one environment variable unset, whatever the developer's shell holds."""
    before = os.environ.pop(name, None)
    try:
        yield
    finally:
        if before is not None:
            os.environ[name] = before


_REAL: dict[str, Any] = {}


def prepare() -> dict[str, Any]:
    """Import the pinned LeRobot and the arm tooling, once, and keep the real functions the fakes wrap.

    Raises `lerobot101.ToolingError` when either is not the tested version, or when another LeRobot was imported
    first: the flows would then exercise code that will not drive the arm.
    """
    if _REAL:
        return _REAL
    with located_tooling():
        commit = lerobot101.check_pin()
        lerobot101.add_to_path()
        line = lerobot101.use_pinned_lerobot()
        real: dict[str, Any] = {"commit": commit, "lerobot_line": line}
        for name in ("so101_session", "record_session", "rollout_session", "viz", "joints", "home_follower"):
            real[name] = lerobot101.load(name)
    import lerobot.datasets
    import lerobot.robots.so_follower
    import lerobot.scripts.lerobot_record
    import lerobot.teleoperators.so_leader
    import lerobot.utils.constants
    import lerobot.utils.keyboard_input
    import lerobot.utils.utils
    from lerobot.datasets.feature_utils import validate_frame
    from lerobot.robots.utils import ensure_safe_goal_position
    from lerobot.teleoperators import Teleoperator
    from lerobot.utils.constants import DEFAULT_FEATURES

    from tolerance_radius.rig import camera_tool, display, placement_phase, record_layout, replay, trial_session, vision

    real.update(
        validate_frame=validate_frame, ensure_safe_goal_position=ensure_safe_goal_position,
        Teleoperator=Teleoperator, DEFAULT_FEATURES=DEFAULT_FEATURES,
        LeRobotDataset=lerobot.datasets.LeRobotDataset,
        FollowerConfig=lerobot.robots.so_follower.SO101FollowerConfig,
        LeaderConfig=lerobot.teleoperators.so_leader.SO101LeaderConfig,
        record_layout=record_layout, trial_session=trial_session, placement_phase=placement_phase,
        camera_tool=camera_tool, display=display, replay=replay, vision=vision,
    )
    _REAL.update(real)
    return _REAL


# --------------------------------------------------------------------------------------------------------------------
# Small pieces
# --------------------------------------------------------------------------------------------------------------------


def ticks_of(degrees: float) -> int:
    """A joint angle as the raw register value the servo holds (4096 ticks to a turn, centre 2048)."""
    return 2048 + int(round(degrees / 360.0 * 4096))


def degrees_of(ticks: int) -> float:
    return (ticks - 2048) * 360.0 / 4096


def scene_rgb(*cubes_mm: Point | None) -> np.ndarray:
    """A 640 x 480 picture of the table with a cube on each given point, in RGB as LeRobot's cameras deliver it."""
    picture = np.empty(FRAME_SHAPE, np.uint8)
    picture[:] = TABLE_BGR[::-1]
    for cube in cubes_mm:
        if cube is None:
            continue
        left = int(round(cube.x_mm * 2 + 100 - CUBE_PX / 2))
        top = int(round(cube.y_mm * 2 + 60 - CUBE_PX / 2))
        picture[max(top, 0):top + CUBE_PX, max(left, 0):left + CUBE_PX] = ORANGE_BGR[::-1]
    return picture


class Patches:
    """Replace attributes for the length of a run, and put every one back."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, bool, Any]] = []

    def set(self, target: Any, name: str, value: Any) -> None:
        had = name in vars(target)
        self._undo.append((target, name, had, vars(target).get(name)))
        setattr(target, name, value)

    def restore(self) -> None:
        for target, name, had, old in reversed(self._undo):
            if had:
                setattr(target, name, old)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(target, name)
        self._undo.clear()


class _Stream(io.TextIOBase):
    """Stands in for the terminal's output: keeps what is printed, in order with everything else perceived."""

    def __init__(self, rig: FakeRig, channel: str) -> None:
        super().__init__()
        self._rig, self._channel = rig, channel
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        self._rig.perceived.append((self._channel, text))
        return len(text)

    def isatty(self) -> bool:
        return False

    def text(self) -> str:
        return "".join(self.parts)


class _Terminal:
    """Stands in for stdin: an interactive terminal, so that the sessions ask their questions."""

    def isatty(self) -> bool:
        return True


class _LogCapture(logging.Handler):
    def __init__(self, rig: FakeRig) -> None:
        super().__init__(level=logging.DEBUG)
        self._rig = rig

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 -- a badly formatted line must not hide the rest
            message = str(record.msg)
        self._rig.logged.append((record.levelname, message))
        if record.levelno >= logging.INFO:
            self._rig.perceived.append(("log", message))


# --------------------------------------------------------------------------------------------------------------------
# The person at the rig
# --------------------------------------------------------------------------------------------------------------------


class Phase:
    """One stretch of a session that the operator can act in: a teleoperation phase, a placement, a gate..."""

    def __init__(self, kind: str, label: str, serial: int, **info: Any) -> None:
        self.kind, self.label, self.serial, self.info = kind, label, serial, info
        self.ticks = 0

    def __repr__(self) -> str:
        return f"<{self.kind} {self.label!r} tick {self.ticks}>"


class _When:
    """What the operator does at one moment: `operator.at("placement", nth=1, tick=4).press("q")`."""

    def __init__(self, operator: Operator, kind: str, label: str | None, nth: int | None, tick: int | None) -> None:
        self._operator, self._where = operator, (kind, label, nth, tick)

    def do(self, action: Callable[[FakeRig, Phase], None]) -> Operator:
        self._operator._rules.append((*self._where, action))
        return self._operator

    def press(self, key: str) -> Operator:
        return self.do(lambda rig, _phase: rig.press(key))

    def cube(self, where: Point | None | str, *, offset_mm: tuple[float, float] = (0.0, 0.0)) -> Operator:
        """Put the cube somewhere: a point, `None` (off the table) or "target" (on the placement's target)."""
        def move(rig: FakeRig, phase: Phase) -> None:
            point = phase.info["target"] if where == "target" else where
            rig.cube = None if point is None else Point(point.x_mm + offset_mm[0], point.y_mm + offset_mm[1])
        return self.do(move)

    def second_cube(self, where: Point | None) -> Operator:
        """Something else orange in the picture (a second cube, a sleeve): the detector then refuses to measure."""
        return self.do(lambda rig, _phase: setattr(rig, "second_cube", where))

    def leader(self, **moved_by_deg: float) -> Operator:
        """Move the leader arm by hand, by these amounts."""
        def move(rig: FakeRig, _phase: Phase) -> None:
            for joint, amount in moved_by_deg.items():
                rig.leader_pose[joint] += amount
        return self.do(move)

    def fail_read(self, error: Exception) -> Operator:
        """The follower's next observation fails with `error`, as a dropped servo packet or a dead camera would."""
        return self.do(lambda rig, _phase: setattr(rig, "read_fault", error))

    def block_arm(self, blocked: bool = True) -> Operator:
        """The follower stops moving, as when it pushes against something (`False` frees it again)."""
        return self.do(lambda rig, _phase: setattr(rig.arm, "blocked", blocked))


class Operator:
    """The person at the rig, as a script.

    With no instructions it behaves like a careful operator: it ends each teleoperation phase with `n` on its third
    tick (a reset only once the leader is back at the start pose, to which it is moved a few degrees per cycle) and
    each recorded episode with `n` on its fourth, puts the cube on a placement's target at once and presses
    `n` on the eleventh frame (after the cube has been steady), lets a policy run for seven steps before ending the
    attempt with `n`, lets a replay run to its end, brings the leader back to the follower when a gate holds, and
    answers the prompts sensibly (`5` for a score).

    `at(kind, label, nth=..., tick=...)` adds something to do at one moment. `kind` is "teleop", "episode",
    "placement", "run" (an executor has the motors), "gate", "save" or "score"; `label` is a part of the phase's
    label; `nth` counts the phases that match `kind` and `label`, from 0; `tick` counts control cycles, camera
    frames, steps or polls within the phase, from 0 (None for every one). `quiet` switches the careful behaviour off
    for some phases: what="end" for the key that ends them, what="setup" for the cube or the leader. One more kind,
    "load", is the moment a replay's recorded actions are read from disk, after its placement was accepted.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[str, str | None, int | None, int | None, Callable[[FakeRig, Phase], None]]] = []
        self._quiet: list[tuple[str, str | None, int | None, str]] = []
        self._answers: list[tuple[str, list[Any]]] = []
        #: Where a careful operator's placement ends up, measured from the target: inside the tolerance, not on it.
        self.placement_offset_mm = (0.6, -0.4)

    def at(self, kind: str, label: str | None = None, *, nth: int | None = None, tick: int | None = 0) -> _When:
        return _When(self, kind, label, nth, tick)

    def quiet(self, kind: str, label: str | None = None, *, nth: int | None = None, what: str = "all") -> Operator:
        self._quiet.append((kind, label, nth, what))
        return self

    def answer(self, prompt_part: str, *answers: Any) -> Operator:
        """Answers to give, in order, to prompts containing `prompt_part`. An exception instance is raised instead."""
        self._answers.append((prompt_part, list(answers)))
        return self

    # ------------------------------------------------------------------------------------------------ internals

    @staticmethod
    def _matches(rig: FakeRig, phase: Phase, kind: str, label: str | None, nth: int | None) -> bool:
        if phase.kind != kind or (label is not None and label not in phase.label):
            return False
        if nth is None:
            return True
        earlier = sum(1 for other in rig.history if other.serial < phase.serial and other.kind == kind
                      and (label is None or label in other.label))
        return earlier == nth

    def _careful(self, rig: FakeRig, phase: Phase, what: str) -> bool:
        return not any(self._matches(rig, phase, kind, label, nth) and quiet in ("all", what)
                       for kind, label, nth, quiet in self._quiet)

    def fire(self, rig: FakeRig, phase: Phase) -> None:
        tick = phase.ticks
        for kind, label, nth, when, action in list(self._rules):
            if when in (None, tick) and self._matches(rig, phase, kind, label, nth):
                action(rig, phase)
        if phase.kind == "teleop":
            # In a reset, and when a start was refused, the arm is brought back to the start pose: by hand, so a few
            # degrees per cycle and not in one jump.
            homing = any(word in phase.label for word in ("Reset", "reposition"))
            if homing and self._careful(rig, phase, "setup"):
                for joint, home in START_POSE.items():
                    rig.leader_pose[joint] += max(-4.0, min(4.0, home - rig.leader_pose[joint]))
            home = not homing or all(abs(START_POSE[joint] - rig.leader_pose[joint]) < 1.0 for joint in JOINTS)
            if tick >= 2 and home and self._careful(rig, phase, "end"):
                rig.press("n")
        elif phase.kind == "episode":
            if tick == 3 and self._careful(rig, phase, "end"):
                rig.press("n")
        elif phase.kind == "placement":
            if tick == 0 and self._careful(rig, phase, "setup"):
                target, (dx, dy) = phase.info["target"], self.placement_offset_mm
                rig.cube = Point(target.x_mm + dx, target.y_mm + dy)
            if tick == 10 and self._careful(rig, phase, "end"):
                rig.press("n")
        elif phase.kind == "run":
            if tick == 6 and phase.info["executor"].startswith("policy") and self._careful(rig, phase, "end"):
                rig.press("n")
        elif phase.kind == "gate":
            if tick >= 1 and self._careful(rig, phase, "setup"):
                rig.leader_pose = dict(rig.arm.goal)

    def reply(self, rig: FakeRig, prompt: str) -> Any:
        for part, answers in self._answers:
            if part in prompt and answers:
                return answers.pop(0)
        if "OFF the table" in prompt:
            rig.cube = None
            return ""
        if "match their names" in prompt:
            return "y"
        if "score 0-5" in prompt:
            return "5"
        if "failure" in prompt:
            return "grasp"
        return ""


# --------------------------------------------------------------------------------------------------------------------
# The arm
# --------------------------------------------------------------------------------------------------------------------


class Arm:
    """The physical follower: where it is, where it was told to go, and whether its motors hold."""

    def __init__(self) -> None:
        self.present = dict(START_POSE)
        self.goal = dict(START_POSE)
        self.torque = True
        #: Largest travel of a joint per step, degrees.
        self.speed = 12.0
        self.blocked = False
        self.bus_open_by: str | None = None

    def step(self) -> None:
        if self.blocked or not self.torque:
            return
        for joint, goal in self.goal.items():
            gap = goal - self.present[joint]
            self.present[joint] += max(-self.speed, min(self.speed, gap))


class FollowerBus:
    """The follower's servo bus. `owner` says which object opened it: the preflight's, or the session's robot."""

    def __init__(self, rig: FakeRig, owner: str) -> None:
        self._rig, self._owner = rig, owner
        self.is_connected = False

    def connect(self) -> None:
        arm = self._rig.arm
        if arm.bus_open_by is not None:
            self._rig.violations.append(f"the servo bus was opened by {self._owner} while {arm.bus_open_by} held it")
        arm.bus_open_by = self._owner
        self.is_connected = True
        self._rig.log("bus:open", self._owner)

    def disconnect(self, disable_torque: bool = True) -> None:
        arm = self._rig.arm
        if disable_torque:
            arm.torque = False
        arm.bus_open_by = None
        self.is_connected = False
        self._rig.log("bus:close", self._owner, "torque off" if disable_torque else "torque kept")

    def _raw(self, pose: dict[str, float]) -> dict[str, int]:
        return {joint: ticks_of(value) for joint, value in pose.items()}

    def sync_read(self, register: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0) -> dict:
        arm = self._rig.arm
        if not self.is_connected:
            raise ConnectionError("the follower's bus is not open")
        if register == "Torque_Enable":
            return dict.fromkeys(JOINTS, int(arm.torque))
        if register == "Present_Position":
            arm.step()
            return dict(arm.present) if normalize else self._raw(arm.present)
        if register == "Goal_Position":
            return dict(arm.goal) if normalize else self._raw(arm.goal)
        raise AssertionError(f"unexpected read of {register}")

    def sync_write(self, register: str, values: dict, *, normalize: bool = True, num_retry: int = 0) -> None:
        arm = self._rig.arm
        if register != "Goal_Position":
            raise AssertionError(f"unexpected write of {register}")
        if not normalize:
            self._rig.log("bus:seed", {joint: int(value) for joint, value in values.items()})
            if self._rig.seed_fails:
                # One servo takes a corrupted value: the read-back will not match what was written.
                values = {**values, "gripper": int(values["gripper"]) + 7}
            values = {joint: degrees_of(int(value)) for joint, value in values.items()}
        arm.goal.update({joint: float(value) for joint, value in values.items()})
        arm.step()

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self._rig.arm.torque = False
        self._rig.log("bus:torque-off", self._owner)

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self._rig.arm.torque = True


class _Camera:
    def __init__(self, config: Any) -> None:
        self.height, self.width = int(getattr(config, "height", 480)), int(getattr(config, "width", 640))
        self.is_connected = False


class FakeFollower:
    """The follower as the sessions use it. `send_action` clamps the way the real class does, with the real helper."""

    name = "so_follower"

    def __init__(self, rig: FakeRig, config: types.SimpleNamespace) -> None:
        self._rig, self.config = rig, config
        self.robot_type = self.name
        owner = "the session's robot" if config.cameras else "the preflight"
        self.bus = FollowerBus(rig, owner)
        self.cameras = {name: _Camera(camera) for name, camera in (config.cameras or {}).items()}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINTS}

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.action_features, **{name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()}}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(camera.is_connected for camera in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        rig = self._rig
        if self.is_connected:
            raise RuntimeError("the follower is already connected")
        self.bus.connect()
        for name, camera in self.cameras.items():
            if rig.camera_that_cannot_open == name:
                raise RuntimeError(f"OpenCVCamera({name}) failed to open")
            camera.is_connected = True
        # configure(): torque is dropped for a moment and comes back.
        rig.arm.torque = True
        rig.torque_touched = True
        rig.connects += 1
        rig.log("follower:connect")

    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for camera in self.cameras.values():
            camera.is_connected = False
        self._rig.log("follower:disconnect", "torque on" if self._rig.arm.torque else "TORQUE OFF")

    def get_observation(self) -> dict[str, Any]:
        rig = self._rig
        if not self.is_connected:
            raise ConnectionError("the follower is not connected")
        rig.tick("placement")
        if rig.read_fault is not None:
            fault, rig.read_fault = rig.read_fault, None
            raise fault
        present = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        observation: dict[str, Any] = {f"{joint}.pos": value for joint, value in present.items()}
        for name in self.cameras:
            if rig.dead_camera == name:
                raise RuntimeError(f"OpenCVCamera({name}) read thread is not running.")
            observation[name] = rig.front_frame() if name == "front" else rig.wrist_frame
        rig.observations += 1
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        rig = self._rig
        if not self.is_connected:
            raise ConnectionError("the follower is not connected")
        goal = {key.removesuffix(".pos"): value for key, value in action.items() if key.endswith(".pos")}
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
            goal = _REAL["ensure_safe_goal_position"]({joint: (value, present[joint]) for joint, value in goal.items()},
                                                      self.config.max_relative_target)
        rig.command(dict(action), dict(goal))
        self.bus.sync_write("Goal_Position", goal)
        rig.tick("run")
        return {f"{joint}.pos": value for joint, value in goal.items()}


def _leader_class(rig: FakeRig) -> type:
    """The leader arm. A subclass of LeRobot's `Teleoperator`, because `record_loop` asks for one by type."""

    class LeaderBus:
        def sync_read(self, register: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0) -> dict:
            if register != "Present_Position":
                raise AssertionError(f"unexpected read of the leader's {register}")
            rig.tick("gate")
            return dict(rig.leader_pose)

    class FakeLeader(_REAL["Teleoperator"]):
        name = "so_leader"
        config_class = None

        def __init__(self, config: Any) -> None:  # no super().__init__: it would touch the calibration folder
            self.config, self.bus, self._connected = config, LeaderBus(), False

        action_features = property(lambda self: {f"{joint}.pos": float for joint in JOINTS})
        feedback_features = property(lambda self: {})
        is_connected = property(lambda self: self._connected)
        is_calibrated = property(lambda self: True)

        def connect(self, calibrate: bool = True) -> None:
            self._connected = True
            rig.log("leader:connect")

        def calibrate(self) -> None:
            raise AssertionError("a session must never calibrate")

        def configure(self) -> None:
            pass

        def send_feedback(self, feedback: dict[str, Any]) -> None:
            pass

        def disconnect(self) -> None:
            self._connected = False
            rig.log("leader:disconnect")

        def get_action(self) -> dict[str, float]:
            rig.tick("teleop", "episode")
            rig.leader_pose["shoulder_pan"] += 0.5  # the hand on the leader is never perfectly still
            return {f"{joint}.pos": value for joint, value in rig.leader_pose.items()}

    return FakeLeader


class _Listener:
    """What `create_key_listener` returns: the key hook, which hears every key typed anywhere until it is stopped."""

    def __init__(self, rig: FakeRig, dispatch: Callable[[str], None]) -> None:
        self._rig, self.dispatch, self.alive = rig, dispatch, True
        rig.log("listener:start")

    def stop(self) -> None:
        if self.alive:
            self.alive = False
            self._rig.log("listener:stop")


# --------------------------------------------------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------------------------------------------------


class DatasetOnDisk:
    """What a dataset directory holds: its settings and its saved episodes."""

    def __init__(self, root: Path, fps: int, features: dict, robot_type: str | None) -> None:
        self.root, self.fps, self.features, self.robot_type = root, fps, features, robot_type
        #: One entry per saved episode: {"frames", "task", "first_front_rgb", "first_action"}.
        self.episodes: list[dict[str, Any]] = []
        self.finalized = True


def _dataset_class(rig: FakeRig) -> type:
    real = _REAL["LeRobotDataset"]
    create_signature, resume_signature = inspect.signature(real.create), inspect.signature(real.resume)

    class Meta:
        def __init__(self, store: DatasetOnDisk) -> None:
            self._store = store

        fps = property(lambda self: self._store.fps)
        features = property(lambda self: self._store.features)
        total_episodes = property(lambda self: len(self._store.episodes))
        video_keys = property(lambda self: [key for key, ft in self._store.features.items() if ft["dtype"] == "video"])

        @property
        def tasks(self) -> Any:
            import pandas as pd

            names = list(dict.fromkeys(episode["task"] for episode in self._store.episodes))
            if not names:
                return None
            return pd.DataFrame({"task_index": range(len(names))}, index=pd.Index(names, name="task"))

    class FakeLeRobotDataset:
        #: None, so that the real `VideoEncodingManager` only has `finalize` to call on the way out.
        writer = None

        def __init__(self, store: DatasetOnDisk, repo_id: str) -> None:
            self._store, self.repo_id, self.root = store, repo_id, store.root
            self.meta = Meta(store)
            self._buffer: list[dict[str, Any]] = []
            self._finalized = False
            store.finalized = False

        # The two ways in take exactly what the installed class takes: a wrong keyword fails here as it would there.
        @classmethod
        def create(cls, *args: Any, **kwargs: Any) -> FakeLeRobotDataset:
            bound = create_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            given = bound.arguments
            root = Path(given["root"]) if given["root"] is not None else rig.hf_home / given["repo_id"]
            root.mkdir(parents=True, exist_ok=False)  # as the real one: an existing directory is refused
            features = {**given["features"], **_REAL["DEFAULT_FEATURES"]}
            rig.disk[root] = DatasetOnDisk(root, given["fps"], features, given["robot_type"])
            rig.log("dataset:create", given["repo_id"], given["fps"])
            return cls(rig.disk[root], given["repo_id"])

        @classmethod
        def resume(cls, *args: Any, **kwargs: Any) -> FakeLeRobotDataset:
            bound = resume_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            given = bound.arguments
            if not given["root"]:
                raise ValueError("resume() requires an explicit 'root' directory because it creates a DatasetWriter.")
            store = rig.disk.get(Path(given["root"]))
            if store is None:
                raise FileNotFoundError(f"no dataset metadata under {given['root']}")
            rig.log("dataset:resume", given["repo_id"], len(store.episodes))
            return cls(store, given["repo_id"])

        fps = property(lambda self: self._store.fps)
        features = property(lambda self: self._store.features)
        num_episodes = property(lambda self: len(self._store.episodes))

        def has_pending_frames(self) -> bool:
            return bool(self._buffer)

        def add_frame(self, frame: dict) -> None:
            _REAL["validate_frame"](frame, self._store.features)
            self._buffer.append(frame)

        def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
            rig.tick("save")
            if rig.save_fault is not None:
                fault, rig.save_fault = rig.save_fault, None
                raise fault
            if not self._buffer:
                raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")
            index = len(self._store.episodes)
            first = self._buffer[0]
            self._store.episodes.append({
                "frames": len(self._buffer), "task": first["task"],
                "first_front_rgb": np.array(first["observation.images.front"]),
                "first_action": np.array(first["action"]),
            })
            rig.log("dataset:save", index, len(self._buffer))
            self._buffer = []

        def clear_episode_buffer(self, delete_images: bool = True) -> None:
            rig.log("dataset:clear", len(self._buffer))
            self._buffer = []

        def finalize(self) -> None:
            if self._finalized:
                return
            self._finalized = True
            self._store.finalized = True
            rig.log("dataset:finalize", len(self._buffer))

    return FakeLeRobotDataset


# --------------------------------------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------------------------------------


class FakePolicy:
    """Stands in for a loaded policy: it walks one joint of its own away from where the arm was at its reset.

    Which joint it moves is its signature, so that the commands the motors received say which policy sent them.
    """

    MOVES = {"clustered": "shoulder_pan", "spread": "elbow_flex"}

    def __init__(self, rig: FakeRig, name: str) -> None:
        self._rig, self.name = rig, name
        self.joint = next((joint for part, joint in self.MOVES.items() if part in name), "wrist_flex")
        features = {
            "observation.state": {"type": "STATE", "shape": [len(JOINTS)]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
        }
        features.update(rig.policy_features.get(name, {}))
        self.config = types.SimpleNamespace(input_features=features, chunk_size=100, n_action_steps=100,
                                            temporal_ensemble_coeff=None)
        self.resets = 0
        self._base: dict[str, float] | None = None
        self._step = 0

    def reset(self) -> None:
        self.resets += 1
        self._base, self._step = None, 0

    def select_action(self, observation: dict[str, Any]) -> dict[str, float]:
        if self._base is None:
            self._base = {joint: float(observation[f"{joint}.pos"]) for joint in JOINTS}
            wild = self._rig.wild_first_actions.get(self.name)
            if wild:
                # An opening move far from where the arm is: what the handoff gate exists to refuse.
                self._base["wrist_roll"] += wild.pop(0)
        self._step += 1
        action = dict(self._base)
        action[self.joint] += 1.5 * self._step * self._rig.policy_rate
        return {f"{joint}.pos": value for joint, value in action.items()}


# --------------------------------------------------------------------------------------------------------------------
# The rig
# --------------------------------------------------------------------------------------------------------------------


class Run:
    """What one session did."""

    def __init__(self, rig: FakeRig, kind: str, argv: list[str], start: int) -> None:
        self.rig, self.kind, self.argv, self._start = rig, kind, argv, start
        self.rc: int | None = None
        self.error: BaseException | None = None
        self.out = self.err = ""
        self.trace: list[tuple] = []
        self.perceived: list[tuple[str, str]] = []
        self.problems: list[str] = []

    def events(self, *kinds: str) -> list[tuple]:
        return [event for event in self.trace if event[0] in kinds]

    def first(self, kind: str) -> int:
        return next(i for i, event in enumerate(self.trace) if event[0] == kind)

    def story(self, *kinds: str) -> list[str]:
        """The trace as short lines; with `kinds`, only those events."""
        chosen = kinds or STORY_KINDS
        return [brief(event) for event in self.trace if event[0] in chosen]

    def said(self) -> list[str]:
        return [event[1] for event in self.trace if event[0] == "say"]

    def text(self) -> str:
        """Everything printed, logged, spoken and asked, as one string."""
        return "\n".join(text for _channel, text in self.perceived)


#: The events a story shows by default: the session's milestones, without the control loops' inner detail.
STORY_KINDS = (
    "pin", "preflight", "policy:load", "follower:built", "dataset:create", "dataset:resume", "leader:connect",
    "follower:connect", "stream", "cameras", "view", "listener:start", "listener:stop", "keys:pause", "keys:resume",
    "viewer", "teleop",
    "background", "placement", "episode", "dataset:clear", "dataset:save", "sidecar", "dataset:finalize", "plan",
    "handoff", "run", "score", "record", "settle", "follower:disconnect", "leader:disconnect",
)


def brief(event: tuple) -> str:
    kind, rest = event[0], event[1:]
    if kind == "preflight":
        return f"preflight {'ok' if rest[0] else 'REFUSED'}"
    if kind == "follower:built":
        return "follower built: " + ", ".join(f"{key}={value}" for key, value in sorted(rest[0].items()))
    if kind == "stream":
        return f"stream {'ok' if rest[0] else 'DEAD'}"
    if kind == "teleop":
        return f"teleop [{rest[0]}] {'ran' if rest[1] else 'NOT RUN'}"
    if kind == "placement":
        label, target, status = rest[0], rest[1], rest[2]
        return f"placement [{label}] at ({target[0]:.1f}, {target[1]:.1f}): {status}"
    if kind == "episode":
        return f"episode {rest[0]} {'recorded' if rest[1] else 'NOT RECORDED'}"
    if kind == "plan":
        return "plan: nothing left" if rest[0] is None else f"plan {rest[0]} {rest[1]}"
    if kind == "handoff":
        return f"handoff {'ok' if rest[1] else 'REFUSED'}"
    if kind == "run":
        return f"run {rest[1]}: {rest[2]} steps"
    if kind == "score":
        return "score: none" if rest[0] is None else f"score {rest[0]['subgoal']}"
    if kind == "record":
        return f"record {rest[0]} {rest[1]}" + (f" ({rest[2]})" if rest[2] else "")
    return " ".join([kind, *(str(item) for item in rest)])


class FakeRig:
    """The bench: the arm, the leader, the cameras, the table, the keyboard, the disk, and the person at them."""

    def __init__(self, *, real_recorder: bool = False) -> None:
        prepare()
        self._tmp = tempfile.TemporaryDirectory(prefix="tolrad_rig_")
        self.tmp = Path(self._tmp.name).resolve()
        self.hf_home = self.tmp / "hf"
        self.log_dir = self.tmp / "logs"
        self.placement_dir = self.tmp / "recording"
        self.repo_root = self.tmp / "repo"
        self.audit_dir = self.tmp / "audit"
        self.replay_root = self.tmp / "replay"
        self.real_recorder = real_recorder
        self.namespace = "tester"

        self.operator = Operator()
        self.arm = Arm()
        self.leader_pose = dict(START_POSE)
        #: Where the cube is on the table (None: not on it), and something else orange in view.
        self.cube: Point | None = None
        self.second_cube: Point | None = None
        self.wrist_frame = np.zeros(FRAME_SHAPE, np.uint8)
        self.disk: dict[Path, DatasetOnDisk] = {}
        self.listeners: list[_Listener] = []
        self.policies: dict[str, FakePolicy] = {}

        # Things a test can break.
        self.view_agrees = True
        self.seed_fails = False
        self.dead_camera: str | None = None
        self.camera_that_cannot_open: str | None = None
        self.read_fault: Exception | None = None
        self.save_fault: Exception | None = None
        self.pin_error: str | None = None
        self.window_keys: list[int] = []
        self.placement_timeout_s: float | None = None
        self.wild_first_actions: dict[str, list[float]] = {}
        #: The follower objects the sessions built, in order: for what a session left on one.
        self.followers: list[Any] = []
        self.policy_features: dict[str, dict] = {}
        self.policy_rate = 1.0
        #: The cameras the system offers, as (name, unique identifier), in any order; an exception when its list
        #: cannot be read. The study's two on their hubs and the laptop's own, unless a flow is about that.
        self.cameras_offered: Any = [("MacBook Pro Camera", "6C707041-0000"), ("HD Pro Webcam C920", "0x1100000aaaa"),
                                     ("USB2.0_CAM1", "0x1140000bbbb")]
        #: The takes (counted from 0 over a session) after which the front camera's picture is no longer the
        #: session's, each with the sentence the check gives.
        self.picture_changed_after: dict[int, str] = {}
        self.pictures_checked = 0
        #: How each look was asked: the options a session gave `changed_since` (its patience, its announcer).
        self.picture_looks: list[dict[str, Any]] = []
        #: How far the camera's picture has crept since the session solved its conversion, by the look that asks
        #: (from 0; a look not named reads 0.0 mm, None is a look that cannot say), and how many looks asked.
        self.crept_after: dict[int, float | None] = {}
        self.creeps_asked = 0
        #: Every look of a handoff gate: whether it passed, the first action, and the arm's pose it was held against.
        self.gate_looks: list[tuple[bool, dict[str, float], dict[str, float]]] = []
        #: Every teleoperation phase as the tooling was handed it: its name, whether that name was a live one, whether
        #: the arm's reads were being watched, and what a live name read at the phase's end.
        self.phase_looks: list[dict[str, Any]] = []
        #: The tooling's countdowns as they started ("on") and stopped ("off") drawing, with their names.
        self.clock_events: list[tuple[str, str]] = []
        #: The options a session gave the check of its view against the stored calibration, once a session.
        self.view_asked: list[dict[str, Any]] = []
        #: Called when an attempt's videos are being finished: for a key pressed after the attempt was over.
        self.on_video_finish: Callable[[], None] | None = None
        #: What the wrist camera's rest picture reads (`vision.wrist_contrast`) at the readings a flow names, counted
        #: from 0 over a session: 0 before the warm-up, 1 when the empty table is pictured, then one after every take
        #: (before every attempt), and one more after a reading that says the state changed. None is a picture that
        #: could not be judged. Every other reading is the usual state's.
        self.wrist_readings: dict[int, float | None] = {}
        self.wrist_read = 0
        #: True for the flows that give the stand-in a drawn wrist picture (`wrist_frame`, RGB as the robot hands
        #: it over) and let the real reading be made from it.
        self.wrist_real = False
        #: What a dataset's files say of it when a sitting is about to append to it, for the flows about a set that
        #: a session left without closing: the store's own count unless a flow puts other counts here.
        self.counts_on_disk: dict[str, Any] | None = None
        #: What the repair's own look says of a set whose counts agree: nothing, unless a flow puts a clause here.
        self.deeper_look_says: str | None = None
        #: What the system's log holds of USB transfer errors when a session asks at its end.
        self.usb_log_lines: list[str] = []
        #: The takes (counted from 0 over a session) that nobody demonstrated in: the go key reached the session with
        #: no one at the leader. In every other take the arm is taken to have travelled as in a demonstration.
        self.still_takes: set[int] = set()
        self.takes_asked = 0
        #: The largest first move a gate may let through: the handoff tolerance, and above the alignment tolerance.
        self.largest_first_move_deg = 8.0
        #: Replaces the protocol a trial session reads (given the real one, returns the one to use). None: the real
        #: one as a draft (`draft`), so that the flows mean the same before and after the repository's own freeze.
        self.protocol_changes: Callable[[dict], dict] | None = None

        # What happened.
        self.trace: list[tuple] = []
        self.perceived: list[tuple[str, str]] = []
        self.logged: list[tuple[str, str]] = []
        self.violations: list[str] = []
        self.commands: list[dict[str, Any]] = []
        self.history: list[Phase] = []
        self._stack: list[Phase] = []
        self._loop: dict[str, Any] | None = None
        self._aligned = False
        self._handoff_ok = False
        self._last_replay_episode: int | None = None
        self.observations = 0
        self.windows_shown = 0
        self.connects = 0
        self.torque_touched = False
        self._torque_before = True
        self._frames: dict[tuple, np.ndarray] = {}

    def __enter__(self) -> FakeRig:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------------------------------------- the trace

    def log(self, kind: str, *details: Any) -> None:
        self.trace.append((kind, *details))

    def mark(self, name: str, *details: Any) -> None:
        self.perceived.append(("mark", " ".join([name, *(str(item) for item in details)])))

    # ------------------------------------------------------------------------------------------- phases

    def push(self, kind: str, label: str, **info: Any) -> Phase:
        phase = Phase(kind, label, len(self.history), **info)
        self.history.append(phase)
        self._stack.append(phase)
        return phase

    def pop(self, phase: Phase) -> None:
        while self._stack and self._stack.pop() is not phase:
            pass

    @property
    def phase(self) -> Phase | None:
        return self._stack[-1] if self._stack else None

    def tick(self, *kinds: str) -> None:
        """A control cycle, a camera frame, a step or a poll of the current phase: the operator may act."""
        phase = self.phase
        if phase is not None and phase.kind in kinds:
            self.operator.fire(self, phase)
            phase.ticks += 1

    # ------------------------------------------------------------------------------------------- the keyboard

    def press(self, key: str) -> None:
        """A key typed anywhere on the machine. It reaches the session only through a live listener."""
        heard = [listener for listener in self.listeners if listener.alive]
        self.log("key" if heard else "key:unheard", key)
        for listener in heard:
            listener.dispatch(key)

    def _input(self, prompt: str = "") -> str:
        self.perceived.append(("ask", prompt))
        if self._loop is not None:
            self.violations.append(f"a prompt was shown while a control loop was running: {prompt.strip()!r}")
        answer = self.operator.reply(self, prompt)
        self.log("ask", " ".join(prompt.split())[:60], answer if isinstance(answer, str) else type(answer).__name__)
        if isinstance(answer, BaseException):
            raise answer
        # Typing goes through the same keyboard: a live listener hears every character of it.
        heard = [listener for listener in self.listeners if listener.alive]
        # A listener whose owner has paused the keys still hears them and passes none on (`typing_guard.Keyboard`,
        # which never starts a second listener): typing is overheard only by one that is listening.
        listening = [listener for listener in heard
                     if not getattr(getattr(listener.dispatch, "__self__", None), "paused", False)]
        overheard = [char for char in answer if listening and char.lower() in "nrq"]
        for listener in heard:
            for char in answer:
                listener.dispatch(char)
            listener.dispatch("enter")
        if overheard:
            self.log("typing-overheard", "".join(overheard))
        return answer

    # ------------------------------------------------------------------------------------------- the cameras

    def front_frame(self) -> np.ndarray:
        key = (self.cube, self.second_cube)
        if key not in self._frames:
            self._frames[key] = scene_rgb(self.cube, self.second_cube)
        return self._frames[key]

    # ------------------------------------------------------------------------------------------- motor commands

    def open_loop(self, kind: str, gated: bool, **details: Any) -> None:
        if self._loop is not None:
            self.violations.append(f"a {kind} loop started inside a {self._loop['kind']} loop")
        if not gated:
            self.violations.append(f"a {kind} loop started without passing its gate ({self.phase})")
        self._loop = {"kind": kind, "commands": 0, **details}
        self.log("loop", kind, *details.values())

    def close_loop(self) -> None:
        loop, self._loop = self._loop, None
        self.log("loop:end", loop["kind"], loop["commands"])

    def command(self, asked: dict[str, float], goal: dict[str, float]) -> None:
        phase = self.phase
        if self._loop is None:
            self.violations.append(f"a motor command was sent with no control loop running ({phase})")
        else:
            if self._loop["commands"] == 0:
                # What the gates exist for: the first cycle of a loop must not ask for a jump. A leader is judged
                # against the goal the follower holds, an executor against where the follower is.
                held = self.arm.present if self._loop["kind"] == "run" else self.arm.goal
                jump = max(abs(asked[f"{joint}.pos"] - held[joint]) for joint in JOINTS)
                if jump > self.largest_first_move_deg:
                    self.violations.append(f"the first command of a {self._loop['kind']} loop asked for a jump of "
                                           f"{jump:.1f} deg ({phase})")
            self._loop["commands"] += 1
        self.commands.append({"loop": None if self._loop is None else self._loop["kind"],
                              "phase": None if phase is None else phase.serial, "asked": asked, "goal": goal})

    # ------------------------------------------------------------------------------------------- set-up helpers

    def repo_id(self, dataset: str, name: str = "tolrad-{dataset}-v1") -> str:
        return f"{self.namespace}/{name.format(dataset=dataset)}"

    def existing_dataset(self, dataset: str, episodes: int, *, task: str = TASK, fps: int = FPS,
                         name: str = "tolrad-{dataset}-v1", logged: int | None = None) -> DatasetOnDisk:
        """A dataset left on disk by an earlier sitting, holding `episodes` episodes, and its placement log with a
        line for each (`logged` lines instead, for the flows about a log that does not match its set)."""
        root = self.hf_home / self.repo_id(dataset, name)
        root.mkdir(parents=True)
        store = DatasetOnDisk(root, fps, self._features(), FakeFollower.name)
        store.episodes = [{"frames": 5, "task": task, "first_front_rgb": None, "first_action": None}
                          for _ in range(episodes)]
        self.disk[root] = store
        from tolerance_radius.rig import record_layout

        log = record_layout.placement_log_path(dataset, name.format(dataset=dataset), self.placement_dir)
        lines = episodes if logged is None else logged
        if lines:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("".join(json.dumps({"dataset": dataset, "episode_index": index}) + "\n"
                                   for index in range(lines)), encoding="utf-8")
        return store

    def _features(self) -> dict:
        """The dataset features the recorder builds for this robot, built the same way."""
        from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.utils.feature_utils import combine_feature_dicts

        procs = make_default_processors()
        robot = FakeFollower(self, self._follower_config(cameras=_REAL["so101_session"].CAMERAS))
        return {
            **combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=procs[0], initial_features=create_initial_features(action=robot.action_features),
                    use_videos=True),
                aggregate_pipeline_dataset_features(
                    pipeline=procs[2],
                    initial_features=create_initial_features(observation=robot.observation_features),
                    use_videos=True),
            ),
            **_REAL["DEFAULT_FEATURES"],
        }

    def write_audits(self, *datasets: str, episodes: int | None = None) -> None:
        """Start-state audits in which every demonstration started exactly where it was planned; with `episodes`,
        of the first so many episodes only (a set that is still being recorded)."""
        for dataset in datasets or schema.DATASETS:
            planned = layouts.read_layout(REPO_ROOT / "config" / "layouts" / f"{dataset}.json")
            starts = layouts.planned_demo_starts(planned)
            if episodes is not None:
                starts = [start for start in starts if start.episode_index < episodes]
            schema.write_demo_starts(self.audit_dir / f"{dataset}.json", starts)

    def write_replay_dataset(self, episodes: int = 55, steps: int = 8) -> Path:
        """A recorded dataset's tables on disk: every episode starts at the start pose and turns its wrist."""
        import pandas as pd

        (self.replay_root / "meta").mkdir(parents=True, exist_ok=True)
        info = {"fps": 30, "total_episodes": episodes,
                "features": {"action": {"names": [f"{joint}.pos" for joint in JOINTS]}}}
        (self.replay_root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        rows = []
        for episode in range(episodes):
            for frame, action in enumerate(self.replay_actions(episode, steps)):
                rows.append({"action": action.tolist(), "episode_index": episode, "frame_index": frame})
        folder = self.replay_root / "data" / "chunk-000"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(folder / "file-000.parquet")
        return self.replay_root

    @staticmethod
    def replay_actions(episode: int, steps: int = 8) -> np.ndarray:
        """What `write_replay_dataset` records for one episode: its gripper value says which episode it is."""
        actions = np.tile(np.array([START_POSE[joint] for joint in JOINTS]), (steps, 1))
        actions[:, JOINTS.index("wrist_roll")] += 1.5 * np.arange(steps)
        actions[:, JOINTS.index("gripper")] += 0.05 * episode
        return actions

    def policy_dir(self, name: str, checkpoint: str = "080000") -> Path:
        """A local `pretrained_model` directory, as `--policy CONDITION=DIR` wants one."""
        folder = self.tmp / "policies" / name / checkpoint / "pretrained_model"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        return folder

    def trial_log(self, name: str = "pilot_test.jsonl") -> Path:
        return self.tmp / "trials" / name

    def _follower_config(self, **given: Any) -> types.SimpleNamespace:
        """What `SO101FollowerConfig(...)` holds: the real class's fields and defaults, without opening anything."""
        import dataclasses

        fields = {field.name: field for field in dataclasses.fields(_REAL["FollowerConfig"])}
        unknown = sorted(set(given) - set(fields))
        if unknown:
            raise TypeError(f"SO101FollowerConfig got unexpected keyword arguments {unknown}")
        values = {}
        for name, field in fields.items():
            if name in given:
                values[name] = given[name]
            elif field.default is not dataclasses.MISSING:
                values[name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                values[name] = field.default_factory()
        return types.SimpleNamespace(**values)

    # ------------------------------------------------------------------------------------------- running a session

    def record_layout(self, argv: Sequence[str], *, through_cli: bool = False, namespace: bool = True) -> Run:
        """Run the recorder's `main` (or its `cli`) with these arguments."""
        module = _REAL["record_layout"]
        argv = [*argv]
        argv += ["--namespace", self.namespace] if namespace and "--namespace" not in argv else []
        argv += ["--fps", str(FPS)] if "--fps" not in argv else []
        argv += ["--hold", "0"] if "--hold" not in argv else []
        # The flows' takes last a few cycles: no shortest take, unless a flow is about that rule.
        argv += ["--min-episode-seconds", "0"] if "--min-episode-seconds" not in argv else []
        argv += ["--log-dir", str(self.log_dir)]
        return self._run("record_layout", module.cli if through_cli else module.main, argv)

    def run_trials(self, argv: Sequence[str], *, through_cli: bool = False) -> Run:
        """Run the trial runner's `main` (or its `cli`) with these arguments."""
        module = _REAL["trial_session"]
        argv = [*argv]
        argv += ["--fps", str(FPS)] if "--fps" not in argv else []
        argv += ["--hold", "0"] if "--hold" not in argv else []
        # The flows' attempts last a few cycles: no shortest attempt, unless a flow is about that rule.
        argv += ["--min-attempt-seconds", "0"] if "--min-attempt-seconds" not in argv else []
        # The flows' keys come milliseconds apart: no quiet time around a go, unless a flow is about that rule.
        if "--typing-guard" not in argv and "--no-typing-guard" not in argv:
            argv += ["--typing-guard", "0", "0"]
        argv += ["--device", "cpu"] if "--device" not in argv else []
        argv += ["--log-dir", str(self.log_dir)]
        return self._run("trial_session", module.cli if through_cli else module.main, argv)

    def _run(self, kind: str, entry: Callable[[list[str]], int], argv: list[str]) -> Run:
        run = Run(self, kind, argv, len(self.trace))
        perceived_from = len(self.perceived)
        self.torque_touched = False
        self.connects = 0
        self._torque_before = self.arm.torque
        patches = Patches()
        out, err = _Stream(self, "print"), _Stream(self, "print")
        root_logger = logging.getLogger()
        handlers_before, level_before = list(root_logger.handlers), root_logger.level
        # Only the capture listens during a run: an import may have left a console handler on the root logger.
        root_logger.handlers = [_LogCapture(self)]
        root_logger.setLevel(logging.INFO)
        try:
            with located_tooling(), _without("TOLRAD_NAMESPACE"), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                self._install(patches)
                try:
                    run.rc = entry(argv)
                except SystemExit as error:  # argparse refusing its arguments
                    run.rc = error.code if isinstance(error.code, int) else 2
                    run.error = error
                except BaseException as error:  # noqa: BLE001 -- the test decides what an exception means
                    run.error = error
        finally:
            patches.restore()
            for handler in root_logger.handlers:
                handler.close()  # the capture, and the session's own log file
            root_logger.handlers = handlers_before
            root_logger.setLevel(level_before)
        self.log("exit", run.rc if run.error is None else type(run.error).__name__)
        run.out, run.err = out.text(), err.text()
        run.trace = self.trace[run._start:]
        run.perceived = self.perceived[perceived_from:]
        run.problems = self._problems(run)
        self._stack.clear()
        self._loop = None
        return run

    # ------------------------------------------------------------------------------------------- the fakes, installed

    def _install(self, patches: Patches) -> None:  # noqa: C901, PLR0915 -- one long list of replacements
        import cv2
        import lerobot.datasets
        import lerobot.robots.so_follower
        import lerobot.scripts.lerobot_record
        import lerobot.teleoperators.so_leader
        import lerobot.utils.constants
        import lerobot.utils.keyboard_input
        import lerobot.utils.utils

        rig = self
        session, record_session = _REAL["so101_session"], _REAL["record_session"]
        rollout, viz = _REAL["rollout_session"], _REAL["viz"]
        record_layout, trial_session = _REAL["record_layout"], _REAL["trial_session"]

        # ---- which versions drive the arm (checked for real once, in prepare) --------------------------------
        def check_pin(*, allow_other: bool = False) -> str:
            rig.log("pin", "arm tooling")
            if rig.pin_error and not allow_other:
                raise lerobot101.ToolingError(rig.pin_error)
            return _REAL["commit"]

        def use_pinned_lerobot(*, allow_other: bool = False) -> str:
            rig.log("pin", "lerobot")
            return _REAL["lerobot_line"]

        patches.set(lerobot101, "check_pin", check_pin)
        patches.set(lerobot101, "use_pinned_lerobot", use_pinned_lerobot)

        # ---- the robot and the leader ------------------------------------------------------------------------
        def follower_config(**given: Any) -> types.SimpleNamespace:
            config = rig._follower_config(**given)
            if config.cameras:
                rig.log("follower:built", {"clamp": config.max_relative_target, "id": config.id,
                                           "torque_off_on_disconnect": config.disable_torque_on_disconnect})
            return config

        def follower(config: types.SimpleNamespace) -> FakeFollower:
            rig.followers.append(FakeFollower(rig, config))
            return rig.followers[-1]

        leader = _leader_class(rig)
        for module in (lerobot.robots.so_follower, session, record_session):
            patches.set(module, "SO101Follower", follower)
            patches.set(module, "SO101FollowerConfig", follower_config)
        for module in (lerobot.teleoperators.so_leader, record_session):
            patches.set(module, "SO101Leader", leader)
            patches.set(module, "SO101LeaderConfig", lambda **given: types.SimpleNamespace(**given))

        # ---- no hardware, whatever happens -------------------------------------------------------------------
        def tripwire(what: str) -> Callable[..., Any]:
            def refuse(*_args: Any, **_kwargs: Any) -> Any:
                rig.violations.append(f"a session tried to open {what}")
                raise AssertionError(f"a session tried to open {what}")
            return refuse

        patches.set(cv2, "VideoCapture", tripwire("a camera"))
        with contextlib.suppress(ImportError):
            import serial

            patches.set(serial, "Serial", tripwire("a serial port"))

        # ---- speech, logging set-up, the viewer, windows -----------------------------------------------------
        def log_say(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
            logging.info(text)
            if play_sounds:
                rig.log("say", text)
                rig.perceived.append(("say", text))

        for module in (lerobot.utils.utils, session, record_session):
            patches.set(module, "log_say", log_say)
        patches.set(lerobot.utils.utils, "say", tripwire("the loudspeaker"))
        patches.set(lerobot.utils.utils, "init_logging", lambda *a, **k: None)
        for module in (viz, record_session):
            patches.set(module, "fresh_rerun_view", lambda name: rig.log("viewer", name.split("_")[0]))
        patches.set(lerobot.scripts.lerobot_record, "log_visualization_data", lambda *a, **k: None)

        class Display:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.pictures = 0

            def show(self, _title: str, picture: np.ndarray) -> None:
                self.pictures += 1

            def clear(self) -> None:
                rig.log("placement:cleared", self.pictures)

            def wait_key(self, _ms: int) -> int:
                return -1

        patches.set(_REAL["display"], "RerunDisplay", Display)
        patches.set(record_layout, "RerunDisplay", Display)
        def imshow(title: str, _picture: np.ndarray) -> None:
            rig.windows_shown += 1

        patches.set(cv2, "imshow", imshow)
        patches.set(cv2, "waitKey", lambda _ms=0: rig.window_keys.pop(0) if rig.window_keys else -1)
        patches.set(cv2, "destroyAllWindows", lambda: None)

        # ---- the keyboard and the terminal -------------------------------------------------------------------
        def create_key_listener(dispatch: Callable[[str], None], *, controls_help: str = "") -> _Listener:
            listener = _Listener(rig, dispatch)
            rig.listeners.append(listener)
            return listener

        patches.set(lerobot.utils.keyboard_input, "create_key_listener", create_key_listener)

        # The trial runner's keyboard pauses its keys without stopping its listener: shown in the trace.
        from tolerance_radius.rig import typing_guard as _typing_guard

        real_pause, real_resume = _typing_guard.Keyboard.pause, _typing_guard.Keyboard.resume

        def pause(keyboard: Any) -> None:
            rig.log("keys:pause")
            real_pause(keyboard)

        def resume(keyboard: Any) -> None:
            real_resume(keyboard)
            rig.log("keys:resume")

        patches.set(_typing_guard.Keyboard, "pause", pause)
        patches.set(_typing_guard.Keyboard, "resume", resume)
        patches.set(builtins, "input", rig._input)
        patches.set(sys, "stdin", _Terminal())

        # ---- files ---------------------------------------------------------------------------------------------
        patches.set(lerobot.utils.constants, "HF_LEROBOT_HOME", rig.hf_home)
        patches.set(lerobot.datasets, "LeRobotDataset", _dataset_class(rig))
        patches.set(record_layout, "PLACEMENT_LOG", rig.placement_dir)

        def system_cameras(*_args: Any, **_kwargs: Any) -> list:
            from tolerance_radius.rig import camera_list

            rig.log("cameras:listed")
            if isinstance(rig.cameras_offered, BaseException):
                raise rig.cameras_offered
            return sorted((camera_list.SystemCamera(name, unique) for name, unique in rig.cameras_offered),
                          key=lambda camera: camera.unique_id)

        from tolerance_radius.rig import camera_list as _camera_list

        patches.set(_camera_list, "system_cameras", system_cameras)

        # The system's log is not asked in a test: the lines a flow gives are what it holds.
        from tolerance_radius import usb_errors as _usb_errors

        def system_log(minutes: int, timeout_s: float = 0.0) -> list[str]:
            rig.log("usb:asked", minutes)
            return list(rig.usb_log_lines)

        patches.set(_usb_errors, "system_log", system_log)
        # The fake dataset has no writer (see FakeLeRobotDataset): it says itself how many frames a take holds.
        patches.set(record_layout, "pending_frames", lambda dataset: len(dataset._buffer))
        real_travel = record_layout.joint_travel

        def joint_travel(dataset: Any) -> float | None:
            # The recorder's own reading of the take's frames, handed over in the form the real writer keeps them.
            # The test rig's operator does not act a demonstration out (an episode here is four cycles, and the
            # leader drifts by a few degrees): unless the take is one nobody was at the leader for, the arm is taken
            # to have travelled as in the recorded demonstrations.
            states = [frame["observation.state"] for frame in dataset._buffer]
            read = real_travel(types.SimpleNamespace(writer=types.SimpleNamespace(
                episode_buffer={"size": len(states), "observation.state": states})))
            take, rig.takes_asked = rig.takes_asked, rig.takes_asked + 1
            return read if take in rig.still_takes or read is None else max(read, 120.0)

        patches.set(record_layout, "joint_travel", joint_travel)
        patches.set(trial_session, "REPO_ROOT", rig.repo_root)
        patches.set(trial_session, "project_commit", lambda: "0000000000")
        real_read_audits = trial_session.read_audits
        patches.set(trial_session, "read_audits", lambda datasets: real_read_audits(datasets, rig.audit_dir))

        def protocol(path: Any = None, *, overlay: Any = None) -> dict:
            loaded = load_protocol(path, overlay=overlay)
            return (rig.protocol_changes or draft)(loaded)

        patches.set(trial_session, "load_protocol", protocol)

        # ---- the arm tooling's own functions: real, with a line in the trace -----------------------------------
        real_preflight = session.run_preflight

        def run_preflight(tolerance_ticks: int) -> bool:
            ok = real_preflight(tolerance_ticks)
            rig.log("preflight", ok)
            return ok

        patches.set(session, "run_preflight", run_preflight)

        real_stream = session.verify_camera_stream

        def verify_camera_stream(robot: Any, fps: int, seconds: float = 2.0) -> tuple[bool, str]:
            ok, detail = real_stream(robot, fps, seconds=0.05)
            rig.log("stream", ok, detail)
            return ok, detail

        patches.set(session, "verify_camera_stream", verify_camera_stream)

        real_alignment = session.wait_for_alignment

        def wait_for_alignment(robot: Any, teleop: Any, tolerance_deg: float, timeout_s: float, label: str,
                               *args: Any, **kwargs: Any) -> bool:
            phase = rig.push("gate", label)
            try:
                ok = real_alignment(robot, teleop, tolerance_deg, timeout_s, label, *args, **kwargs)
            finally:
                rig.pop(phase)
            rig._aligned = ok
            rig.log("gate", label, ok, phase.ticks)
            return ok

        patches.set(session, "wait_for_alignment", wait_for_alignment)

        real_teleop = session.teleop_phase

        def teleop_phase(label: str, *args: Any, **kwargs: Any) -> bool:
            phase = rig.push("teleop", label)
            look = {"label": str(label), "live": getattr(label, "live", None),
                    "watched": "get_observation" in vars(args[0]), "readout": None}
            rig.phase_looks.append(look)
            try:
                ok = real_teleop(label, *args, **kwargs)
                if hasattr(label, "reading"):
                    look["readout"] = label.reading()      # what the countdown's thread would be drawing now
            finally:
                rig.pop(phase)
            rig.log("teleop", label, ok, phase.ticks)
            return ok

        patches.set(session, "teleop_phase", teleop_phase)

        class RecordedClock(session.PhaseClock):
            """The tooling's countdown, which also says when it started and stopped drawing."""

            def __enter__(self) -> Any:
                rig.clock_events.append(("on", str(self.label)))
                return super().__enter__()

            def __exit__(self, *exc: Any) -> Any:
                rig.clock_events.append(("off", str(self.label)))
                return super().__exit__(*exc)

        patches.set(session, "PhaseClock", RecordedClock)

        def traced_loop(real_loop: Callable[..., Any]) -> Callable[..., Any]:
            def record_loop(*args: Any, **kwargs: Any) -> Any:
                kind = "teleop" if kwargs.get("dataset") is None else "episode"
                aligned, rig._aligned = rig._aligned, False
                rig.open_loop(kind, aligned, seconds=kwargs.get("control_time_s"), display=kwargs.get("display_data"))
                try:
                    return real_loop(*args, **kwargs)
                finally:
                    rig.close_loop()
            return record_loop

        patches.set(session, "record_loop", traced_loop(session.record_loop))
        patches.set(record_session, "record_loop", traced_loop(record_session.record_loop))

        real_episode = record_session.record_episode

        def record_episode(robot: Any, teleop: Any, procs: Any, fps: int, seconds: float, task: str, dataset: Any,
                           *args: Any, **kwargs: Any) -> bool:
            index = dataset.num_episodes
            phase = rig.push("episode", f"Episode {index}")
            try:
                ok = real_episode(robot, teleop, procs, fps, seconds, task, dataset, *args, **kwargs)
            finally:
                rig.pop(phase)
            rig.log("episode", index, ok, len(dataset._buffer))
            return ok

        patches.set(record_session, "record_episode", record_episode)

        real_save = record_session.save_episode_announced

        def save_episode_announced(dataset: Any, play_sounds: bool) -> None:
            phase = rig.push("save", f"episode {dataset.num_episodes}")
            try:
                real_save(dataset, play_sounds)
            finally:
                rig.pop(phase)

        patches.set(record_session, "save_episode_announced", save_episode_announced)

        real_settle = session.settle_arm

        def settle_arm(robot: Any) -> str:
            if rig._loop is not None:
                rig.close_loop()  # an exception left a loop open: the arm is no longer being driven
            state = real_settle(robot)
            rig.log("settle", state)
            return state

        patches.set(session, "settle_arm", settle_arm)

        # ---- the recorder's own pieces -------------------------------------------------------------------------
        real_append = record_layout.append_placement

        def append_placement(path: Path, entry: dict[str, Any]) -> None:
            real_append(path, entry)
            rig.log("sidecar", entry["episode_index"], path.name)

        patches.set(record_layout, "append_placement", append_placement)

        def checked_stored_view(grab: Callable[[], np.ndarray], **kwargs: Any) -> Any:
            rig.view_asked.append(dict(kwargs))
            frame = grab()
            if frame.shape != FRAME_SHAPE:
                raise AssertionError(f"the session's frame source gave {frame.shape}")
            if not rig.view_agrees:
                rig.log("view", "REFUSED")
                raise _REAL["vision"].VisionError(
                    "the camera's picture is no longer the calibration's: it has moved 7.0 mm on the floor "
                    "(limit 5.0 mm). Run the camera baseline check, and say so before anything is stored again")
            rig.log("view", "confirmed")
            return _REAL["vision"].TableView(PX_TO_MM)

        patches.set(_REAL["camera_tool"], "checked_stored_view", checked_stored_view)
        patches.set(record_layout, "checked_stored_view", checked_stored_view)

        def changed_since(start: Any, grab: Callable[[], np.ndarray], **kwargs: Any) -> str | None:
            # The stand-in's pictures carry no tags to solve a frame from: the check itself is tested on drawn
            # tags (test_rig_vision), and here it says what the flow tells it to.
            grab()
            rig.log("picture:checked")
            rig.picture_looks.append({name: ("announcer" if callable(value) else value)
                                      for name, value in sorted(kwargs.items())})
            take, rig.pictures_checked = rig.pictures_checked, rig.pictures_checked + 1
            return rig.picture_changed_after.get(take)

        patches.set(_REAL["camera_tool"], "changed_since", changed_since)
        patches.set(record_layout, "changed_since", changed_since)

        def crept_since(start: Any, grab: Callable[[], np.ndarray], **_kwargs: Any) -> float | None:
            # As above: no tags to solve from here. The reading itself is tested on drawn tags (test_rig_vision).
            grab()
            look, rig.creeps_asked = rig.creeps_asked, rig.creeps_asked + 1
            return rig.crept_after.get(look, 0.0)

        patches.set(_REAL["camera_tool"], "crept_since", crept_since)

        real_wrist_reading = _REAL["camera_tool"].wrist_reading

        def wrist_reading(grab_wrist: Callable[[], np.ndarray], **_kwargs: Any) -> float | None:
            # The stand-in's wrist picture is black: the number is tested on drawn pictures (test_rig_sessions,
            # test_rig_vision). Here a reading is what the flow says, and the usual state's otherwise; the rules
            # that judge it (`vision.wrist_other_state`, `vision.wrist_changed`) are the real ones. A flow that
            # gives the stand-in a drawn picture gets the real reading of it, through the session's own frame source.
            if rig.wrist_real:
                rig.log("wrist:read")
                rig.wrist_read += 1
                return real_wrist_reading(grab_wrist, wait_s=0.0)
            frame = grab_wrist()
            if frame.shape != FRAME_SHAPE:
                raise AssertionError(f"the session's wrist frame source gave {frame.shape}")
            rig.log("wrist:read")
            look, rig.wrist_read = rig.wrist_read, rig.wrist_read + 1
            return rig.wrist_readings.get(look, 2.55)

        patches.set(_REAL["camera_tool"], "wrist_reading", wrist_reading)
        patches.set(record_layout, "wrist_reading", wrist_reading)
        real_wrist_changed_since = _REAL["camera_tool"].wrist_changed_since

        def wrist_changed_since(start: Any, grab_wrist: Callable[[], np.ndarray], **_kwargs: Any) -> Any:
            return real_wrist_changed_since(start, grab_wrist, wait_s=0.0)   # the second reading without the wait

        patches.set(_REAL["camera_tool"], "wrist_changed_since", wrist_changed_since)
        patches.set(record_layout, "wrist_changed_since", wrist_changed_since)

        def dataset_counts(dataset_dir: Path) -> dict[str, Any]:
            # The stand-in's sets are not files: a set says of itself what its store holds, as a closed set does.
            if rig.counts_on_disk is not None and Path(dataset_dir) in rig.disk:
                return dict(rig.counts_on_disk)
            store = rig.disk.get(Path(dataset_dir))
            held = len(store.episodes) if store is not None else None
            return {"said": held, "rows": held or 0, "unreadable": []}

        patches.set(record_layout, "dataset_counts", dataset_counts)

        def deeper_look(dataset_dir: Path, placement_log: Path) -> str | None:
            # The stand-in's sets are not files either way: the look itself is tested on folders written by
            # LeRobot's own writer (test_rig_repair_set).
            rig.log("set:looked-through", Path(dataset_dir).name)
            return rig.deeper_look_says

        patches.set(record_layout, "deeper_look", deeper_look)

        real_empty_scene = _REAL["vision"].empty_scene

        def empty_scene(grab: Callable[[], np.ndarray], frames: int = 15) -> np.ndarray:
            rig.log("background", "cube on the table" if rig.cube is not None else "table empty")
            return real_empty_scene(grab, frames)

        patches.set(_REAL["vision"], "empty_scene", empty_scene)
        patches.set(record_layout, "empty_scene", empty_scene)

        class TracedPlacementPhase(_REAL["placement_phase"].PlacementPhase):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                rig.log("placement:built", "window" if self._show is cv2.imshow else "viewer or none")
                if rig.placement_timeout_s is not None:
                    self._timeout_s = rig.placement_timeout_s  # ten minutes is the session's own; a test cannot wait

            def run(self, target: Point, label: str = "") -> Any:
                phase = rig.push("placement", label, target=target)
                rig.perceived.append(("picture", label))
                commands_before = len(rig.commands)
                try:
                    result = super().run(target, label)
                finally:
                    rig.pop(phase)
                measured = None if result.measured is None else (result.measured.x_mm, result.measured.y_mm)
                rig.log("placement", label, (target.x_mm, target.y_mm), result.status, measured, result.frames_seen)
                if len(rig.commands) != commands_before:
                    rig.violations.append(f"the arm was commanded during the placement {label!r}")
                return result

        patches.set(_REAL["placement_phase"], "PlacementPhase", TracedPlacementPhase)
        patches.set(record_layout, "PlacementPhase", TracedPlacementPhase)

        # ---- the trial runner's pieces -------------------------------------------------------------------------
        def load_policy(repo_id: Any, checkpoint: Any, checkpoint_path: str, device: str) -> tuple[Any, str, str]:
            folder = Path(checkpoint_path)
            name = "policy:" + (folder.parent.parent.name if folder.name == "pretrained_model" else folder.name)
            rig.policies[name] = FakePolicy(rig, name)
            rig.log("policy:load", name, device)
            return rig.policies[name], f"pre:{name}", f"post:{name}"

        def apply_inference_overrides(policy: Any, n_action_steps: Any = None,
                                      temporal_ensemble_coeff: Any = None) -> Any:
            rig.log("policy:overrides", policy.name, temporal_ensemble_coeff)
            policy.reset()
            return policy

        def make_predict(policy: Any, robot: Any, preprocessor: Any, postprocessor: Any, task: str,
                         dataset_features: dict, device: str = "mps", on_observation: Any = None) -> Callable:
            if (preprocessor, postprocessor) != (f"pre:{policy.name}", f"post:{policy.name}"):
                rig.violations.append(f"{policy.name} was given another policy's processors")
            rig.log("predict:built", policy.name, task, device)

            def predict() -> dict[str, float]:
                observation = robot.get_observation()
                if on_observation is not None:
                    on_observation(observation)
                return policy.select_action(observation)

            return predict

        patches.set(rollout, "load_policy", load_policy)
        patches.set(rollout, "apply_inference_overrides", apply_inference_overrides)
        patches.set(rollout, "make_predict", make_predict)

        class Recorder:
            def __init__(self, out_dir: Any, stamp: str, fps: int = 30, cameras: tuple = ("front", "wrist"),
                         display: bool = False, **_kwargs: Any) -> None:
                self.out_dir, self.stamp, self.cameras = Path(out_dir), stamp, cameras
                self._label: str | None = None
                self._frames = 0

            def start(self, label: str) -> None:
                self._label, self._frames = label, 0
                rig.log("video:start", label)

            def write(self, observation: dict) -> None:
                if self._label is not None and isinstance(observation.get("front"), np.ndarray):
                    self._frames += 1

            def finish(self) -> list[str]:
                label, frames, self._label = self._label, self._frames, None
                rig.log("video:finish", label, frames)
                if rig.on_video_finish is not None:
                    rig.on_video_finish()
                if not frames:
                    return []
                return [str(self.out_dir / f"{self.stamp}_{label}_{camera}.mp4") for camera in self.cameras]

        if not rig.real_recorder:
            patches.set(rollout, "AttemptRecorder", Recorder)

        def executor_name(executor: Any) -> str:
            return getattr(executor, "name", None) or f"replay:{getattr(executor, 'episode', '?')}"

        real_cycle = rollout.attempt_cycle

        def attempt_cycle(**kwargs: Any) -> bool:
            name = executor_name(kwargs["policy"])
            phase = rig.push("attempt", kwargs["label"], executor=name)
            rig.log("attempt", kwargs["label"], name, "dry run" if kwargs.get("dry_run") else "live")
            try:
                ran = real_cycle(**kwargs)
            finally:
                rig.pop(phase)
            rig.log("attempt:end", kwargs["label"], ran)
            return ran

        patches.set(rollout, "attempt_cycle", attempt_cycle)

        real_attempt = rollout.run_attempt

        def run_attempt(**kwargs: Any) -> int:
            name = executor_name(kwargs["policy"])
            phase = rig.push("run", kwargs["label"], executor=name)
            handoff_ok, rig._handoff_ok = rig._handoff_ok, False
            rig.open_loop("run", handoff_ok, executor=name)
            try:
                sent = real_attempt(**kwargs)
            finally:
                rig.close_loop()
                rig.pop(phase)
            rig.log("run", kwargs["label"], name, sent)
            return sent

        patches.set(rollout, "run_attempt", run_attempt)

        real_check = rollout.PolicyHandoffGate.check

        def check(gate: Any, action: dict, present: dict, label: str = "") -> bool:
            ok = real_check(gate, action, present, label)
            rig._handoff_ok = ok
            rig.gate_looks.append((ok, dict(action), dict(present)))
            rig.log("handoff", label, ok, gate.tolerance)
            return ok

        patches.set(rollout.PolicyHandoffGate, "check", check)

        real_score = rollout.score_quietly

        def score_quietly(keyboard: Any, ask: Callable[[str], str]) -> Any:
            phase = rig.push("score", "")
            commands_before = len(rig.commands)
            try:
                score = real_score(keyboard, ask)
            finally:
                rig.pop(phase)
                rig.mark("score entered")
            rig.log("score", score)
            if len(rig.commands) != commands_before:
                rig.violations.append("the arm was commanded while the trial was being scored")
            return score

        patches.set(rollout, "score_quietly", score_quietly)

        real_confirm = rollout.confirm_cameras

        def confirm_cameras(*args: Any, **kwargs: Any) -> bool:
            ok = real_confirm(*args, **kwargs)
            rig.log("cameras", "confirmed" if ok else "NOT CONFIRMED")
            return ok

        patches.set(rollout, "confirm_cameras", confirm_cameras)

        real_next = trial_session.next_planned

        def next_planned(*args: Any, **kwargs: Any) -> Any:
            trial = real_next(*args, **kwargs)
            if trial is None:
                rig.log("plan", None)
            else:
                rig.log("plan", trial.trial_id, trial.condition, trial.blind_code,
                        (trial.target.x_mm, trial.target.y_mm), trial.replay_episode)
                rig.mark("plan", trial.trial_id)
            return trial

        patches.set(trial_session, "next_planned", next_planned)

        real_append_trial = schema.append_trial

        def append_trial(path: Any, record: Any) -> None:
            real_append_trial(path, record)
            rig.log("record", record.planned.trial_id, record.outcome, record.void_reason)

        patches.set(schema, "append_trial", append_trial)

        real_load_actions = _REAL["replay"].load_episode_actions

        def load_episode_actions(root: Any, episode: int) -> Any:
            rig._last_replay_episode = episode
            rig.log("replay:load", episode)
            phase = rig.push("load", f"episode {episode}")  # reading the tables takes a moment at the rig
            try:
                rig.tick("load")
                return real_load_actions(root, episode)
            finally:
                rig.pop(phase)

        class TracedReplaySource(_REAL["replay"].ReplaySource):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.episode = rig._last_replay_episode

        patches.set(_REAL["replay"], "load_episode_actions", load_episode_actions)
        patches.set(_REAL["replay"], "ReplaySource", TracedReplaySource)

    def events_of(self, kind: str) -> list[tuple]:
        return [event for event in self.trace if event[0] == kind]

    # ------------------------------------------------------------------------------------------- after a run

    def preflight_problems(self) -> list[str]:
        """What the next session's preflight would say about the arm as it was left (the real check)."""
        bus = FollowerBus(self, "a later preflight")
        bus.is_connected = True
        return _REAL["home_follower"].preflight_problems(bus, 50)

    def _problems(self, run: Run) -> list[str]:  # noqa: C901 -- one list of promises
        """The promises a session must keep however it ends. An empty list means it kept them."""
        problems = list(self.violations)
        self.violations = []
        kinds = [event[0] for event in run.trace]
        settles = [event[1] for event in run.trace if event[0] == "settle"]

        if self.torque_touched:
            if not settles:
                problems.append("the arm was connected and never settled")
            elif settles[-1] == "seeded":
                last_loop = max((i for i, kind in enumerate(kinds) if kind == "loop:end"), default=-1)
                if kinds.index("settle") < last_loop:
                    problems.append("a control loop ran after the arm was settled")
                if "follower:disconnect" not in kinds:
                    problems.append("the follower was never disconnected")
                if not self.arm.torque:
                    problems.append("the arm was left limp")
                elif self.preflight_problems():
                    problems.append(f"the next preflight would refuse: {self.preflight_problems()}")
        elif self._torque_before and not self.arm.torque:
            problems.append("the arm was left limp without ever being connected")
        if self.connects > 1:
            problems.append(f"the follower was connected {self.connects} times in one session")
        if self.arm.bus_open_by is not None and (settles and settles[-1] != "untouched"):
            problems.append(f"the servo bus was left open by {self.arm.bus_open_by}")
        if any(listener.alive for listener in self.listeners):
            problems.append("a key listener was left running")
        if kinds.count("leader:connect") != kinds.count("leader:disconnect"):
            problems.append("the leader was left connected")

        # The recorder: every saved episode has its line in the sidecar, written after the save, and nothing else has.
        saves = [event[1] for event in run.trace if event[0] == "dataset:save"]
        lines = [event[1] for event in run.trace if event[0] == "sidecar"]
        if saves != lines:
            problems.append(f"episodes saved {saves} but placements written for {lines}")
        for index in lines:
            position = next(i for i, event in enumerate(run.trace) if event[:2] == ("sidecar", index))
            if ("dataset:save", index) not in [event[:2] for event in run.trace[:position]]:
                problems.append(f"episode {index}'s placement was written before the episode was saved")
        if "dataset:create" in kinds or "dataset:resume" in kinds:
            if self.torque_touched and "dataset:finalize" not in kinds:
                problems.append("the dataset was not finalized")
        return problems


# --------------------------------------------------------------------------------------------------------------------
# Reading back what a session wrote
# --------------------------------------------------------------------------------------------------------------------


def read_sidecar(rig: FakeRig, dataset: str) -> list[dict[str, Any]]:
    path = rig.placement_dir / f"{dataset}_placements.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measured_in(frame_rgb: np.ndarray) -> Point | None:
    """Where the cube is in a recorded frame, by the study's own detector: what the start-state audit will find."""
    import cv2

    from tolerance_radius.rig.vision import TableView

    view = TableView(PX_TO_MM, cv2.cvtColor(scene_rgb(), cv2.COLOR_RGB2BGR))
    with located_tooling():
        reading = view.measure(cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR))
    return None if reading is None else reading.position


def frozen(protocol: dict) -> dict:
    """The protocol as it will be once frozen: counted sessions are allowed.

    The flows run their loops at 200 Hz, so the protocol's own rate (its `[execution]` section, where it has one) is
    set to that as well: a session that checks its settings against the protocol then finds them equal.
    """
    changed = {**protocol, "protocol": {**protocol["protocol"], "status": "frozen"}}
    if "execution" in protocol:
        # ... and the shortest attempt to none, as the flows' attempts last a few cycles.
        changed["execution"] = {**protocol["execution"], "control_hz": FPS, "min_attempt_s": 0.0}
    return changed


def draft(protocol: dict) -> dict:
    """The protocol as it is before the freeze, whatever the repository's own file says today: a counted session is
    refused. A flow that needs counted sessions asks for `frozen`; a pilot's protocol (status `pilot`) is left alone.

    Until the freeze was rehearsed (the night of 2026-10-07) the flows read the repository's file as it stood, and
    the one that holds "counted trials are refused until the protocol is frozen" would have failed on the day the
    file is frozen.
    """
    if protocol["protocol"].get("status") != "frozen":
        return protocol
    return {**protocol, "protocol": {**protocol["protocol"], "status": "draft"}}


def pilot_protocol() -> dict:
    return load_protocol(overlay=PILOT_OVERLAY_PATH)
