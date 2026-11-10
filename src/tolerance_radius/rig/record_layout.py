"""Record one sitting of a planned demonstration layout, with every start position placed and measured.

This is the earlier project's recording session (`record_session.py`) with one thing added: before each episode the
cube is put on that episode's planned target with the placement loop, and the measured position is written down.
Everything that touches the arm is that project's own gated code, called in that project's order:

    connect once
      preflight                          refuse to start if the arm is unsafe to torque
      WARM-UP     teleop, not recording  settle into the start pose, press n
      empty scene                        cube off the table, one picture for the detector
      for each planned episode:
        placement  arm holding           cube onto its target; measured; accepted when steady and n is pressed
        episode    recording             behind the alignment gate, as always
        reset      teleop, not recording put the arm back, press n (r = record this episode again)
        save                             the freeze, after the keypress
      COOL-DOWN   teleop, not recording  park it, press n
      seed the goal to the present position, disconnect leaving torque on

The placement runs while the control loop is stopped and the arm holds its position: the same kind of pause as the
save, and guarded the same way, because the next loop entry is `record_episode`, whose alignment gate refuses a gap
between the leader and the follower. Nothing here commands a motor directly.

If you change this file, compare it line by line with `record_session.main` in the arm tooling: the order of the
phases and the handling of the session's flags are load bearing there, and are mirrored here on purpose.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tolerance_radius import layouts, usb_errors
from tolerance_radius.config import REPO_ROOT, load_protocol
from tolerance_radius.rig import camera_list, lerobot101
from tolerance_radius.rig.display import RerunDisplay
from tolerance_radius.rig.placement_phase import PlacementPhase, consume_go
from tolerance_radius.rig.camera_tool import changed_since, checked_stored_view, wrist_changed_since, wrist_reading
from tolerance_radius.rig.vision import VisionError, empty_scene, load_geometry, wrist_other_state
from tolerance_radius.schema import DATASETS, Point

TASK = "pick up the orange cube and place it in the basket"
#: How many more resets are asked for when the arm is not back at its start pose, before the sitting is stopped.
START_POSE_TRIES = 3


#: The two joints the folded arm does not hold by itself. At rest its shoulder lift, elbow and wrist flex lie against
#: their stops and come back to the degree; the base can still swivel and the wrist turn, and those two come back to
#: where the hand left them. *Measured* in sitting 1 (2026-10-06, three episodes, read from the takes' last frames):
#: lift, elbow and wrist flex within 0.4 degrees of the start pose every time, the pan up to 5.8 and the roll up to
#: 6.6 degrees from it, which at 5 degrees for every joint asked for the reset again after two of three episodes.
FREE_JOINTS = ("shoulder_pan", "wrist_roll")


def off_start_pose(reference: dict[str, float], now: dict[str, float], tolerance_deg: float,
                   free_tolerance_deg: float | None = None) -> list[str]:
    """The joints that rest further from where they rested in `reference` than they may, each with how far and how
    far it may, as words for the operator.

    `tolerance_deg` is for the joints the folded arm holds by itself, `free_tolerance_deg` for the two it leaves
    free (`FREE_JOINTS`); None gives those two the same as the rest. Empty when the arm is back, or when the check
    is switched off (a `tolerance_deg` of 0). A `free_tolerance_deg` of 0 leaves the two free joints unchecked.
    """
    if tolerance_deg <= 0:
        return []
    away = []
    for joint in reference:
        if joint not in now:
            continue
        allowed = free_tolerance_deg if free_tolerance_deg is not None and joint in FREE_JOINTS else tolerance_deg
        if allowed > 0 and abs(now[joint] - reference[joint]) > allowed:
            away.append(f"{joint} {now[joint] - reference[joint]:+.1f} deg (allowed {allowed:g})")
    return away
#: Where each saved episode's planned and measured start position is written, one JSON line per episode.
PLACEMENT_LOG = REPO_ROOT / "data" / "recording"
#: The study's own datasets: `tolrad-clustered-v1`, `tolrad-spread-v1`.
DEFAULT_NAME = "tolrad-{dataset}-v1"
#: A take shorter than this is a slip of the key, not a demonstration (a real one takes well over ten seconds).
MIN_EPISODE_S = 3.0
#: A take in which no joint moved this far is not a demonstration either: the go key reached the session with nobody
#: at the leader (the keys are read from every program, and a message typed elsewhere holds the letter n), and the
#: recorder would otherwise save up to 35 seconds of an arm at rest as one of the 55. In 60 recorded demonstrations
#: (sitting 1's three, the smoke runs' four, 53 of August) the shoulder lift alone travelled 105 to 176 degrees
#: (*measured* 2026-10-06); an arm at rest with a hand on the leader moves a few.
MIN_TRAVEL_DEG = 30.0
#: ... and the travel is asked of these three, the joints the folded arm holds against their stops. A hand resting on
#: the leader can swivel the base, turn the wrist or squeeze the gripper by more than that with nothing else moving
#: (the review of 2026-10-06); the lift, the elbow and the wrist's flex move only when the arm is led out.
HELD_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
#: After a stop key has ended a reset, the look at the picture goes on for this long once it has asked hands out
#: (15 seconds after an ordinary reset): the arm cannot be led any more, a hand can still be withdrawn.
AFTER_STOP_PATIENCE_S = 4.0


@dataclass(frozen=True, slots=True)
class Sitting:
    """One sitting of the recording plan: which dataset, and which of its episodes."""

    number: int
    dataset: str
    rounds: tuple[int, ...]
    #: The dataset's planned episodes for these rounds, in recording order.
    episodes: tuple[Any, ...]

    @property
    def first_index(self) -> int:
        return self.episodes[0].episode_index

    @property
    def end_index(self) -> int:
        return self.episodes[-1].episode_index + 1


def sitting_from_protocol(protocol: dict[str, Any], number: int, layout_dir: Path) -> Sitting:
    """The episodes of sitting `number` (counted from 1 in the protocol's order)."""
    sittings = protocol["datasets"]["sittings"]
    if not 1 <= number <= len(sittings):
        raise ValueError(f"sitting must be 1 to {len(sittings)}")
    entry = sittings[number - 1]
    planned = layouts.read_layout(layout_dir / f"{entry['dataset']}.json")
    episodes = tuple(sorted((e for e in planned if e.round in entry["rounds"]), key=lambda e: e.episode_index))
    indices = [e.episode_index for e in episodes]
    if not episodes or indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"sitting {number}: its episodes are not one unbroken run of indices: {indices}")
    return Sitting(number=number, dataset=entry["dataset"], rounds=tuple(entry["rounds"]), episodes=episodes)


def episodes_to_record(sitting: Sitting, already_recorded: int) -> tuple[Any, ...]:
    """The episodes of the sitting still to record, given how many episodes the dataset already holds.

    Refuses when the count does not fit the plan: fewer than the sitting's first index means an earlier sitting is
    incomplete; more than its end means this sitting was already recorded.
    """
    if already_recorded < sitting.first_index:
        raise ValueError(
            f"the dataset holds {already_recorded} episodes, but sitting {sitting.number} starts at episode "
            f"{sitting.first_index}: an earlier sitting of this dataset is incomplete"
        )
    if already_recorded >= sitting.end_index:
        raise ValueError(
            f"the dataset already holds {already_recorded} episodes; sitting {sitting.number} ends at episode "
            f"{sitting.end_index - 1}: nothing left to record"
        )
    return tuple(e for e in sitting.episodes if e.episode_index >= already_recorded)


def dataset_counts(dataset_dir: Path) -> dict[str, Any]:
    """What a dataset on disk says of itself, read from its files without opening it for writing: `said`, the
    episodes its `meta/info.json` counts (None when that cannot be read); `rows`, the rows of its episode table;
    `unreadable`, its table files that cannot be read (a parquet file whose writer was never closed has no end);
    `pictures`, the files left under its `images` folder, where a take's frames lie until the take is saved."""
    try:
        said = int(json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"])
    except (OSError, ValueError, KeyError, TypeError):
        said = None
    rows, unreadable = 0, []
    pictures = sum(1 for path in (dataset_dir / "images").rglob("*") if path.is_file())
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return {"said": said, "rows": None, "unreadable": [], "pictures": pictures}
    for folder, counted in (("meta/episodes", True), ("data", False)):
        for path in sorted((dataset_dir / folder).rglob("*.parquet")):
            try:
                found = parquet.read_metadata(path).num_rows
            except Exception:  # noqa: BLE001 -- whatever is wrong with the file, it is a file that cannot be read
                unreadable.append(str(path.relative_to(dataset_dir)))
                continue
            rows += found if counted else 0
    return {"said": said, "rows": rows, "unreadable": unreadable, "pictures": pictures}


def placement_lines(path: Path) -> int:
    """How many episodes a placement log holds: its lines that are not empty. 0 when there is no such file."""
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except OSError:
        return 0


def resume_problem(counts: dict[str, Any], log_lines: int, log_name: str) -> str | None:
    """What does not add up in a dataset a sitting is about to append to; None when it adds up.

    A session that ends without closing the set (the terminal closed, the process killed: Ctrl-C and a crash do
    close it) leaves `info.json` counting the episodes it saved, their table rows unwritten and a table file
    without its end. The next session started from that count, wrote over the unfinished file, and the set no
    longer loaded: seen only at the end of the day (the dry run of 2026-10-06, with LeRobot's own writer on a
    copy). So the three counts are held against each other before anything is opened for writing.

    A session cut off in the middle of a take, with nothing saved since the set was last closed, leaves the three
    counts in agreement and the take's pictures under `images`: the next session's first save then failed on a
    half-written picture, or saved its own take with the dead one's frames behind it (the review of the same night,
    with the same writer). Pictures left there are refused as well.
    """
    said, rows, unreadable = counts.get("said"), counts.get("rows"), counts.get("unreadable") or []
    if said is None:
        return "its meta/info.json cannot be read"
    if unreadable:
        return (f"{len(unreadable)} of its table files cannot be read ({', '.join(unreadable[:3])}): a session ended "
                "without closing the set")
    if rows is not None and rows != said:
        return (f"it says it holds {said} episodes and its episode table has {rows}: a session ended without "
                "closing the set")
    if counts.get("pictures"):
        return (f"{counts['pictures']} pictures of a take that was never saved are left in its images folder: a "
                "session ended in the middle of a take")
    if log_lines != said:
        return f"it holds {said} episodes and the placement log {log_name} has {log_lines} lines"
    return None


def deeper_look(dataset_dir: Path, placement_log: Path) -> str | None:
    """What the repair's own look finds in a set whose counts agree (`repair_set.judge`), as a clause; None when it
    finds the set sound, and when the look cannot be made (it must not stop a sitting by its own fault).

    The counts do not see everything. A save cut short while it encoded its videos, in a session that then closed
    the set as Ctrl-C does, leaves the take's rows in the session's data file: every count agrees, the earlier
    project's dataset check says OK, and the next session's episode is then read from the cut take's rows beside
    its own pictures (*measured* in the night of 2026-10-06 with LeRobot's own writer on copies). The repair's look
    reads every table and every video's length, about half a second on a set of 27 episodes.
    """
    try:
        from tolerance_radius.rig import repair_set  # here: that module reads this one's two checks

        verdict = repair_set.judge(repair_set.look(dataset_dir, placement_log))
    except Exception as error:  # noqa: BLE001 -- see the docstring
        logging.warning("the set's files could not be looked through (%s: %s): going by its counts alone",
                        type(error).__name__, error)
        return None
    if verdict.refusal:
        return verdict.refusal
    if verdict.sound_already:
        return None
    left = [f"{len(verdict.move_out)} files or folders of a session that did not end properly"] * bool(verdict.move_out)
    left += [f"rows of a take that was never saved in {name}" for name, _episodes in verdict.rewrite_data]
    left += [f"{verdict.log_cut} lines too many in the placement log"] * bool(verdict.log_cut)
    return ("its files hold " + "; ".join(left)) if left else "its counts or its statistics are a save ahead of its files"


def out_of_turn(protocol: dict[str, Any], sitting: Sitting, held: dict[str, int], layout_dir: Path) -> str | None:
    """A sentence when a sitting of the other set that comes before this one is not fully recorded; None when all
    are, or when it cannot be said. `held` is how many episodes each set holds.

    The four sittings alternate the two sets so that a drift over the day falls on both. Each sitting is held to
    its own set's count already (`episodes_to_record`); nothing held the two sets to each other, and sitting 2 could
    be started with sitting 1 six episodes short (the dry run of 2026-10-06). Said, not refused: the operator may
    have a reason.
    """
    short = []
    for earlier in range(1, sitting.number):
        try:
            before = sitting_from_protocol(protocol, earlier, layout_dir)
        except (ValueError, OSError, KeyError):
            return None
        have = held.get(before.dataset, 0)
        if before.dataset != sitting.dataset and have < before.end_index:
            short.append(f"sitting {earlier} ends at episode {before.end_index - 1} of the {before.dataset} set, "
                         f"which holds {have} episodes")
    if not short:
        return None
    return f"sitting {sitting.number} is meant to come after the sittings before it: {'; '.join(short)}"


def set_aside(dataset_dir: Path, placement_log: Path, label: str, *, log_only: bool = False
              ) -> list[tuple[Path, Path]]:
    """Move a recorded set and its placement log out of the recorder's way under one new name. Nothing is deleted.

    The set's folder gets `-<label>` after its name, and its log becomes `<that name>_placements.jsonl` where it
    was: the name a set recorded under another name has for its log (`placement_log_path`), so the two still
    belong together. Returns what was renamed, as (from, to). Raises ValueError, with nothing renamed, when
    neither is there, when a new name is taken, when the label is not a plain word, and when the log is there and
    the set is not: a mistyped owner name looks just like that, and the log would be parted from its set
    (`log_only` says that the set really is gone).

    Done by hand on 2026-10-06 for the first four episodes of the clustered set; the recorder refuses to begin a set
    under a log that holds another's lines, and this is the way out it names.
    """
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError(f"the label {label!r} is not a plain word (letters, digits, - _ .)")
    new_dir = dataset_dir.with_name(f"{dataset_dir.name}-{label}")
    new_log = placement_log.with_name(f"{new_dir.name}_placements.jsonl")
    moves = [(old, new) for old, new in ((dataset_dir, new_dir), (placement_log, new_log)) if old.exists()]
    if not moves:
        raise ValueError(f"there is nothing to set aside: no {dataset_dir} and no {placement_log}")
    if not dataset_dir.exists() and not log_only:
        raise ValueError(f"there is no dataset at {dataset_dir}, and its placement log {placement_log.name} is "
                         "there: is the owner's name right? A log really left without its set is set aside with "
                         "--log-only")
    taken = [new for _old, new in ((dataset_dir, new_dir), (placement_log, new_log)) if new.exists()]
    if taken:
        raise ValueError(f"the name is taken: {', '.join(str(path) for path in taken)} is already there; choose "
                         "another label")
    for old, new in moves:
        old.rename(new)
    return moves


def append_placement(path: Path, entry: dict[str, Any]) -> None:
    """Write one episode's planned and measured start position, forced to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def placement_log_path(dataset: str, name: str, directory: Path | None = None) -> Path:
    """The file a sitting's placements are appended to.

    The study's own dataset of a layout gets `<dataset>_placements.jsonl`, which is what the analysis reads. A dataset
    recorded under any other name (a smoke test, the practice block) gets a file named after itself, so that a
    rehearsal cannot leave lines that pass for the real set's episodes.
    """
    directory = PLACEMENT_LOG if directory is None else directory
    own = name == DEFAULT_NAME.format(dataset=dataset)
    return directory / (f"{dataset}_placements.jsonl" if own else f"{name}_placements.jsonl")


def pending_frames(dataset: Any) -> int | None:
    """How many frames the take being recorded holds so far; None when the dataset cannot say.

    Read where LeRobot's own `has_pending_frames` reads it: the writer's episode buffer.
    """
    buffer = getattr(getattr(dataset, "writer", None), "episode_buffer", None)
    if isinstance(buffer, dict) and "size" in buffer:
        return int(buffer["size"])
    return None


def too_short(frames: int | None, fps: int, min_seconds: float) -> bool:
    """True for a take that ended before `min_seconds`: a second press of the go key, not a demonstration."""
    return frames is not None and frames < fps * min_seconds


def joint_travel(dataset: Any) -> float | None:
    """How far the arm was led in the take being recorded: the largest distance one of the three joints the folded
    arm holds (`HELD_JOINTS`) covered between its lowest and its highest reading, in degrees. When the dataset does
    not name its joints, the largest over all of them, as it was. None when the dataset cannot say.

    Read where `pending_frames` reads: the writer's episode buffer, which holds the take's joint readings frame by
    frame until the episode is saved.
    """
    buffer = getattr(getattr(dataset, "writer", None), "episode_buffer", None)
    states = buffer.get("observation.state") if isinstance(buffer, dict) else None
    if not states:
        return None
    try:
        import numpy as np

        table = np.asarray(states, dtype=float)
    except (TypeError, ValueError):
        return None
    if table.ndim != 2 or table.size == 0:
        return None
    travel = table.max(axis=0) - table.min(axis=0)
    try:
        names = [str(name).split(".")[0] for name in dataset.features["observation.state"]["names"]]
    except (AttributeError, KeyError, TypeError):
        names = []
    held = [index for index, name in enumerate(names) if name in HELD_JOINTS]
    if len(names) == table.shape[1] and len(held) == len(HELD_JOINTS):
        travel = travel[held]
    return float(travel.max())


def stood_still(travel: float | None, min_travel: float) -> bool:
    """True for a take in which no joint moved by `min_travel`: nobody was at the leader, so not a demonstration.
    False when the travel is not known, or when the check is switched off (a `min_travel` of 0)."""
    return travel is not None and min_travel > 0 and travel < min_travel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sitting", type=int, required=True, help="which sitting of the protocol's plan, from 1")
    parser.add_argument("--namespace", default=os.environ.get("TOLRAD_NAMESPACE"),
                        help="the datasets' owner on disk and on the hub (or set TOLRAD_NAMESPACE)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="dataset name; {dataset} is filled in")
    parser.add_argument("--resume", action="store_true",
                        help="append to the dataset a previous sitting created (sittings 3 and 4, or after a stop)")
    parser.add_argument("--limit", type=int, default=None, help="record at most this many episodes (a smoke test)")
    parser.add_argument("-t", "--episode-time", type=float, default=35.0,
                        help="how long an episode is recorded, in seconds. It was 25 until the recorder's first runs "
                             "at the rig, where the operator found it too short for the task and the way back")
    parser.add_argument("--min-episode-seconds", type=float, default=MIN_EPISODE_S,
                        help="a take shorter than this is not saved and is recorded again: the go key pressed twice")
    parser.add_argument("--min-travel", type=float, default=MIN_TRAVEL_DEG,
                        help="a take in which no joint moved this many degrees is not saved and is recorded again: "
                             "the go key pressed with nobody at the leader. 0 switches the check off")
    parser.add_argument("-r", "--reset-time", type=float, default=300.0,
                        help="the longest a reset may last, in seconds; it ends when n is pressed. It was 15 until "
                             "the recorder's first run at the rig, where it ended by itself with the arm away from "
                             "its start pose and the cube still in the basket")
    parser.add_argument("--start-pose-tolerance", type=float, default=5.0,
                        help="degrees a joint may rest from where it rested when the empty table was pictured; "
                             "beyond it the reset is asked for again, so that every episode starts from about the "
                             "same pose and the arm stays clear of where the cube is looked for. 0 switches the "
                             "check off. It was 2 while the detector looked at the whole picture")
    parser.add_argument("--free-joint-tolerance", type=float, default=12.0,
                        help="the same for the two joints the folded arm does not hold by itself, the shoulder pan "
                             "and the wrist roll. It was 5 like the others until sitting 1, where those two came "
                             "back up to 6.6 degrees away and the reset was asked for again after two of three "
                             "episodes. 0 leaves the two unchecked")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--hold", type=float, default=1.5, help="seconds the cube must stay on target to be accepted")
    parser.add_argument("--warmup-max", type=float, default=300.0)
    parser.add_argument("--cooldown-max", type=float, default=300.0)
    parser.add_argument("--align-tolerance", type=float, default=5.0)
    parser.add_argument("--align-timeout", type=float, default=180.0)
    parser.add_argument("--preflight-tolerance", type=int, default=50)
    parser.add_argument("--writer-processes", type=int, default=0)
    parser.add_argument("--no-sounds", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false", default=True)
    parser.add_argument("--placement-view", choices=("rerun", "window"), default="window",
                        help="where the placement picture is shown: a window of its own (the camera's picture and "
                             "the target's surroundings magnified), or the session's Rerun viewer. In the viewer "
                             "the live cameras stood still after the first placement (2026-10-06)")
    parser.add_argument("--layouts", default=str(REPO_ROOT / "config" / "layouts"))
    parser.add_argument("--log-dir", default=None, help="default: data/logs in this repository (not committed)")
    parser.add_argument("--allow-other-tooling", action="store_true",
                        help="run even if the arm tooling is not at the commit this project was tested against")
    parser.add_argument("--tags-relative-limit", type=float, default=None, metavar="MM",
                        help="how far the two floor tags may read against each other since the table was registered "
                             "before the session refuses (default 3). Only after the kept frame was held against "
                             "the calibration's (analysis/instrument/picture_motion.py) and the tags moved with the "
                             "rest of the picture: then it is the camera that turned")
    parser.add_argument("--any-wrist-state", action="store_true",
                        help="record although the wrist camera's picture is not in its usual tone state (the "
                             "floor over the jaws' green, with the arm at rest, above 2.85), or changes it. It is "
                             "looked at before the warm-up, when the empty table is pictured and after every take")
    parser.add_argument("--any-cameras", action="store_true",
                        help="record although the system's list of cameras does not show the study's two at the "
                             "numbers the session takes (scripts/rig/cameras.py prints the list)")
    parser.add_argument("--plan-only", action="store_true", help="print what this sitting would record, and stop")
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915 -- mirrors record_session.main
    args = build_parser().parse_args(argv)
    protocol = load_protocol()
    try:
        sitting = sitting_from_protocol(protocol, args.sitting, Path(args.layouts))
    except (ValueError, OSError) as error:
        print(f"REFUSING -- {error}", file=sys.stderr)
        return 1
    if sitting.dataset not in DATASETS:
        print(f"REFUSING -- unknown dataset {sitting.dataset!r}", file=sys.stderr)
        return 1
    tolerance_mm = protocol["placement"]["tolerance_mm"]

    if args.plan_only:
        print(f"sitting {sitting.number}: dataset {sitting.dataset}, rounds {list(sitting.rounds)}, "
              f"episodes {sitting.first_index} to {sitting.end_index - 1}")
        for episode in sitting.episodes:
            print(f"  episode {episode.episode_index:2d}  round {episode.round}  cell {episode.cell}  "
                  f"target ({episode.target.x_mm:6.1f}, {episode.target.y_mm:6.1f}) mm")
        return 0

    if not args.namespace:
        print("REFUSING -- no dataset owner: pass --namespace or set TOLRAD_NAMESPACE", file=sys.stderr)
        return 1
    repo_id = f"{args.namespace}/{args.name.format(dataset=sitting.dataset)}"

    try:
        tooling_commit = lerobot101.check_pin(allow_other=args.allow_other_tooling)
    except lerobot101.ToolingError as error:
        print(f"REFUSING -- {error}", file=sys.stderr)
        return 1
    lerobot101.add_to_path()
    try:
        # Before anything imports lerobot: the arm runs on the checkout that drove it in August.
        lerobot_line = lerobot101.use_pinned_lerobot(allow_other=args.allow_other_tooling)
    except lerobot101.ToolingError as error:
        print(f"REFUSING -- {error}", file=sys.stderr)
        return 1

    import cv2
    import numpy as np
    from lerobot.datasets import (
        LeRobotDataset,
        VideoEncodingManager,
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.feature_utils import combine_feature_dicts
    from lerobot.utils.keyboard_input import init_keyboard_listener
    from lerobot.utils.utils import init_logging, log_say

    record_session = lerobot101.load("record_session")
    session = lerobot101.load("so101_session")
    viz = lerobot101.load("viz")

    init_logging()
    session_started = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = session.start_session_log(log_dir, stamp)
    logging.info("record_layout %s | sitting %d | %s rounds %s | task=%r | arm tooling %s | %s",
                 stamp, sitting.number, sitting.dataset, list(sitting.rounds), args.task, tooling_commit[:10],
                 lerobot_line)

    # The cameras are taken by number, and a number belongs to whatever camera is plugged in: on 2026-10-06 the wrist
    # camera had not started and number 1 was the laptop's own. Asked of the system's list, before anything is opened.
    numbers = lerobot101.load("rig")
    told, wrong = camera_list.check_for_session(numbers.FRONT_CAMERA_INDEX, numbers.WRIST_CAMERA_INDEX)
    for line in told:
        logging.info("%s", line)
    if wrong:
        if not args.any_cameras:
            logging.error("the cameras are not the study's two in their places: %s", wrong)
            print(f"\nREFUSING TO RECORD -- {wrong}.\n  scripts/rig/cameras.py prints the list; --any-cameras "
                  "records all the same.")
            return 3
        logging.warning("recording although the cameras are not in their places (--any-cameras): %s", wrong)

    if not session.run_preflight(args.preflight_tolerance):
        print("\nREFUSING TO RECORD -- the arm is not in a safe state to be torqued.")
        return 1

    play_sounds = not args.no_sounds
    gate = session.AlignmentGate(args.align_tolerance, args.align_timeout, play_sounds=play_sounds)

    robot = SO101Follower(
        SO101FollowerConfig(
            port=session.FOLLOWER_PORT,
            id="my_follower",
            cameras=session.CAMERAS,
            # As in record_session: never let the arm go limp when we disconnect.
            disable_torque_on_disconnect=False,
        )
    )
    teleop = SO101Leader(SO101LeaderConfig(port=session.LEADER_PORT, id="my_leader"))

    procs = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=procs[0],
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=procs[2],
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    dataset_dir = Path(HF_LEROBOT_HOME) / repo_id
    placement_log = placement_log_path(sitting.dataset, args.name.format(dataset=sitting.dataset))
    if args.resume and not dataset_dir.exists():
        print(f"REFUSING -- --resume, but there is no dataset at {dataset_dir}", file=sys.stderr)
        return 1
    if not args.resume and dataset_dir.exists():
        there = dataset_counts(dataset_dir)
        if there["said"] == 0 and not (resume_problem(there, placement_lines(placement_log), placement_log.name)
                                       or deeper_look(dataset_dir, placement_log)):
            # A start that was refused after the set was created (the picture, the wrist camera, a stop in the
            # warm-up) leaves it there, empty. "Start the command again" must then work as it is written.
            logging.info("an empty set of this name is there, from a start that recorded nothing: going on in it")
            print(f"  (an empty set is there at {dataset_dir}, from a start that recorded nothing: going on in it)")
            args.resume = True
        else:
            holds = there["said"]
            print(f"REFUSING -- a dataset already exists at {dataset_dir}"
                  f"{'' if holds is None else f' (it holds {holds} episodes)'}. Pass --resume to append to it, or "
                  "set it aside first (scripts/rig/set_aside.py) to record the set anew.", file=sys.stderr)
            return 1
    if not args.resume and placement_lines(placement_log):
        print(f"REFUSING -- there is no dataset at {dataset_dir}, and the placement log {placement_log.name} holds "
              f"{placement_lines(placement_log)} lines of an earlier one. A new set's lines must not follow "
              f"another's: set that log aside (scripts/rig/set_aside.py {sitting.dataset} --as NAME --log-only, "
              "with a name of your choosing; it deletes nothing).", file=sys.stderr)
        return 1
    if args.resume:
        # Before anything is opened for writing: a set that a session left without closing must not be written to.
        unsound = (resume_problem(dataset_counts(dataset_dir), placement_lines(placement_log), placement_log.name)
                   or deeper_look(dataset_dir, placement_log))
        if unsound:
            logging.error("the dataset does not add up: %s", unsound)
            print(f"REFUSING -- the dataset at {dataset_dir} does not add up: {unsound}.\n  Nothing was changed. "
                  "Recording into it as it is would spoil what it holds: delete nothing. With no session running, "
                  f"scripts/rig/repair_set.py {sitting.dataset} says what it found and what it would do (runbook 2, "
                  "'A set that does not add up').", file=sys.stderr)
            return 1
        # LeRobot refuses to resume without the dataset's directory (the arm tooling's own --resume omits it).
        dataset = LeRobotDataset.resume(
            repo_id,
            root=dataset_dir,
            image_writer_processes=args.writer_processes,
            image_writer_threads=4 * len(robot.cameras),
        )
        already = int(getattr(dataset.meta, "total_episodes", 0) or 0)
        # The task string is a model input: a near-duplicate silently splits the dataset into two tasks.
        existing = getattr(dataset.meta, "tasks", None)
        names = list(existing.index) if hasattr(existing, "index") else list(existing or [])
        if names and args.task not in names:
            print(f"REFUSING -- task {args.task!r} is not the task this dataset was recorded with: {names}",
                  file=sys.stderr)
            return 1
    else:
        try:
            # Before anything is created on disk: a sitting refused for its plan must not leave an empty dataset
            # in the way of the sitting that should have been asked for.
            episodes_to_record(sitting, 0)
        except ValueError as error:
            print(f"REFUSING -- {error}", file=sys.stderr)
            return 1
        dataset = LeRobotDataset.create(
            repo_id,
            args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_processes=args.writer_processes,
            image_writer_threads=4 * len(robot.cameras),
        )
        already = 0
    try:
        todo = episodes_to_record(sitting, already)
    except ValueError as error:
        print(f"REFUSING -- {error}", file=sys.stderr)
        return 1
    if args.limit is not None:
        todo = todo[: args.limit]
    logging.info("%s %s at episode %d; recording episodes %d to %d",
                 "resuming" if args.resume else "creating", repo_id, already,
                 todo[0].episode_index, todo[-1].episode_index)
    print(f"\nSitting {sitting.number}: {len(todo)} episodes of the {sitting.dataset} set into {repo_id} "
          f"({'resumed' if args.resume else 'new'}), push to the hub: OFF\n")
    if args.name == DEFAULT_NAME:
        # The study's own two sets: said when a sitting of the other set that comes first is not all recorded.
        held = {name: dataset_counts(Path(HF_LEROBOT_HOME) / args.namespace / DEFAULT_NAME.format(dataset=name))["said"]
                or 0 for name in DATASETS}
        early = out_of_turn(protocol, sitting, {**held, sitting.dataset: already}, Path(args.layouts))
        if early:
            logging.warning("%s", early)
            print(f"  NOTE: {early}. The sittings alternate the two sets on purpose; go on only if this is meant.\n")

    listener = None
    arm_state = "untouched"
    #: What the session ends with: 0, or 3 when it recorded nothing because a camera's picture was not right.
    ended = 0
    try:
        teleop.connect()
        robot.connect()

        streaming, detail = session.verify_camera_stream(robot, args.fps)
        if not streaming:
            logging.error("camera not streaming before warm-up: %s", detail)
            print(f"\nREFUSING TO RECORD -- a camera connected but will not stream.\n  {detail}")
            return 3

        def grab() -> np.ndarray:
            """The front camera's current frame, BGR, upright: the robot's own stream, so no second camera client."""
            return cv2.cvtColor(np.asarray(robot.get_observation()["front"]), cv2.COLOR_RGB2BGR)

        # The arm is at rest and clear of both tags: confirm that the picture still matches the stored calibration
        # before anything is recorded through it. Read-only; a refusal here costs the sitting, not the arm.
        try:
            view = checked_stored_view(grab, relative_limit_mm=args.tags_relative_limit)
        except VisionError as error:
            logging.error("the camera view cannot be used: %s", error)
            print(f"\nREFUSING TO RECORD -- {error}")
            return 3

        def grab_wrist() -> np.ndarray:
            """The wrist camera's current frame, BGR, as `grab` gives the front camera's."""
            return cv2.cvtColor(np.asarray(robot.get_observation()["wrist"]), cv2.COLOR_RGB2BGR)

        def said_of_wrist(reading: float | None) -> str:
            return ("not judged: its picture is not the rest picture (the jaws over the floor)" if reading is None
                    else f"the floor over the jaws' green reads {reading:.2f}")

        # The wrist camera has no tag and no calibration: one number says which of its two tone states its picture
        # is in, with the arm at rest in its start pose (`vision.wrist_contrast`). 21 of the first 27 episodes of
        # the clustered set were recorded in the other state, and nothing said so (2026-10-06). A first look here,
        # where stopping costs seconds; it only says, because the arm rests where the last session left it, and
        # away from the start pose the number is not to be trusted. The look that refuses is at the start pose.
        wrist_first = wrist_reading(grab_wrist)
        logging.info("wrist camera, before the warm-up: %s", said_of_wrist(wrist_first))
        wrist_wrong = wrist_other_state(wrist_first)
        if args.any_wrist_state:
            logging.warning("--any-wrist-state: the wrist camera's tone state stops nothing in this session")
            print("\n  NOTE: --any-wrist-state. The wrist camera's tone state is written into this session's log "
                  "and stops nothing. Write in the journal that it was used.")
        if wrist_wrong and not args.any_wrist_state:
            logging.warning("before the warm-up: %s", wrist_wrong)
            print(f"\n  NOTE: {wrist_wrong}.\n  That is its other tone state if the arm is resting in its start "
                  "pose. It is asked again at the start pose, and nothing is recorded in that state. To open the "
                  "camera afresh now: press q in the warm-up, and start the command again.")

        def hands_out(text: str) -> None:
            log_say("Hands out of the picture", play_sounds)
            print(f"\n  {text}")

        listener, events = init_keyboard_listener()
        if args.display:
            viz.fresh_rerun_view(f"record_{stamp}")
        logging.info("keys: n / Right = end current phase, r = re-record this episode, q = stop")

        if not session.teleop_phase("Warm up", robot, teleop, procs, args.fps, args.warmup_max, events,
                                    play_sounds, gate, display_data=args.display):
            events["stop_recording"] = True

        phase = None
        if not events["stop_recording"]:
            # The arm is holding at the start pose. Take the empty-scene picture the detector needs, with the arm
            # where it will be at every placement.
            log_say("Cube off the table", play_sounds)
            input("\n  Take the cube OFF the table, hands out of the picture, then press Enter. ")
            # Typing in the terminal reaches the session's key listener too: clear what it may have set.
            events["exit_early"] = False
            events["rerecord_episode"] = False
            # The cube is looked for on the grid and a little around it: the arm, which is cube-coloured, may then
            # rest where it likes outside that region.
            view = view.with_background(empty_scene(grab)).with_cube_region(load_geometry()["grid"])
            # Where the arm rests in that picture: the pose every episode starts from, within the tolerance below.
            start_pose = robot.bus.sync_read("Present_Position", num_retry=3)
            logging.info("start pose: %s", ", ".join(f"{joint} {value:.1f}" for joint, value in start_pose.items()))
            # The wrist camera's reading this session's takes are held against, with the arm at its start pose.
            wrist_start = wrist_reading(grab_wrist)
            logging.info("wrist camera, arm at its start pose: %s", said_of_wrist(wrist_start))
            wrist_wrong = wrist_other_state(wrist_start)
            if wrist_start is None:
                print("\n  NOTE: the wrist camera's picture could not be judged (it is not the rest picture: the "
                      "jaws over the white floor). Its tone state is not watched in this session; the audit says "
                      "afterwards.")
            elif wrist_wrong and not args.any_wrist_state:
                logging.error("%s", wrist_wrong)
                log_say("Wrist camera in its other state. Nothing is recorded.", play_sounds)
                print(f"\n  Before anything is recorded, {wrist_wrong}.\n  Start the command again: the camera is "
                      "opened afresh. If it comes back the same, watch it by itself (scripts/rig/measure.py wrist) "
                      "while you try its plug and its port, or record all the same with --any-wrist-state.")
                events["stop_recording"] = True
                ended = 3
            elif wrist_wrong:
                logging.warning("%s (--any-wrist-state)", wrist_wrong)
            screen = RerunDisplay() if args.placement_view == "rerun" and args.display else None

            phase = PlacementPhase(view, grab, tolerance_mm=tolerance_mm, hold_s=args.hold,
                                   confirm=lambda: consume_go(events), accept_hint="press n to record",
                                   should_stop=lambda: bool(events["stop_recording"]),
                                   on_end=screen.clear if screen else None,
                                   show=screen.show if screen else None,
                                   wait_key=screen.wait_key if screen else None,
                                   grid=load_geometry()["grid"])

        with VideoEncodingManager(dataset):
            for episode in todo:
                if events["stop_recording"]:
                    break
                rerecords = 0
                while not events["stop_recording"]:
                    label = (f"episode {episode.episode_index}  cell {episode.cell}  "
                             f"({episode.target.x_mm:.0f}, {episode.target.y_mm:.0f}) mm")
                    log_say(f"Place the cube, cell {episode.cell}", play_sounds)
                    # A press of n left over from the reset must not confirm this placement.
                    events["exit_early"] = False
                    placed = phase.run(episode.target, label)
                    if not placed.accepted:
                        logging.warning("placement for episode %d ended as %s: stopping",
                                        episode.episode_index, placed.status)
                        events["stop_recording"] = True
                        break
                    logging.info("episode %d placed at (%.1f, %.1f) mm, %.1f mm from its target",
                                 episode.episode_index, placed.measured.x_mm, placed.measured.y_mm, placed.error_mm)
                    # A key pressed during the placement must not end or re-record the episode about to start.
                    events["exit_early"] = False
                    events["rerecord_episode"] = False

                    ep_ok = record_session.record_episode(
                        robot, teleop, procs, args.fps, args.episode_time, args.task, dataset, events,
                        play_sounds, gate, display_data=args.display,
                    )
                    if not ep_ok:
                        events["stop_recording"] = True
                        break
                    if events["stop_recording"]:
                        # The stop key ended this take before the demonstration did. A cut-off take must not become
                        # one of the 55 demonstrations: it is thrown away, and the episode is recorded next time.
                        logging.warning("episode %d: stopped during the take; it is NOT saved", episode.episode_index)
                        print(f"\n  Episode {episode.episode_index} was cut short by the stop key and is not saved.")
                        dataset.clear_episode_buffer()
                        break
                    frames = pending_frames(dataset)
                    travel = joint_travel(dataset)

                    # As in record_session: every episode ends into live teleop, never into the freeze.
                    if not events["stop_recording"] and not session.teleop_phase(
                        "Reset the environment", robot, teleop, procs, args.fps, args.reset_time, events,
                        play_sounds, gate, display_data=args.display,
                    ):
                        events["stop_recording"] = True
                    # The reset is over only when the arm rests about where it rested for the picture of the empty
                    # table. In a placement the arm holds and cannot be moved; an arm left far from its start pose
                    # may hang into the region where the cube is looked for, and the next episode would start from
                    # another pose than the rest.
                    for attempt in range(START_POSE_TRIES + 1):
                        if events["stop_recording"]:
                            break
                        away = off_start_pose(start_pose, robot.bus.sync_read("Present_Position", num_retry=3),
                                              args.start_pose_tolerance, args.free_joint_tolerance)
                        if away:
                            time.sleep(0.5)                 # a follower still arriving is not an arm left elsewhere
                            away = off_start_pose(start_pose, robot.bus.sync_read("Present_Position", num_retry=3),
                                                  args.start_pose_tolerance, args.free_joint_tolerance)
                        if not away:
                            break
                        logging.info("episode %d: the reset ended with the arm away from its start pose: %s",
                                     episode.episode_index, ", ".join(away))
                        if attempt == START_POSE_TRIES:
                            logging.warning("the arm is still not at its start pose after %d more resets (%s): "
                                            "stopping", START_POSE_TRIES, ", ".join(away))
                            print(f"\n  The arm is still not back at its start pose ({', '.join(away)}). Stopping "
                                  "here: what was recorded so far is kept. If the leader's rest has moved, say so.")
                            events["stop_recording"] = True
                            break
                        # the re-record key pressed in the first reset must survive this one; n must not
                        again = bool(events["rerecord_episode"])
                        events["exit_early"] = False
                        log_say("Arm back to the start pose", play_sounds)
                        print(f"\n  The arm is not back near its start pose ({', '.join(away)}).\n"
                              "  Put the leader on its rest, then press n.")
                        if not session.teleop_phase(
                            "Reset again: arm to the start pose", robot, teleop, procs, args.fps, args.reset_time,
                            events, play_sounds, gate, display_data=args.display,
                        ):
                            events["stop_recording"] = True
                        events["rerecord_episode"] = again or bool(events["rerecord_episode"])

                    # The take was recorded in the picture the camera gave, and that must still be the picture the
                    # session started on: the camera went back to its own zoom and exposure in the middle of a take
                    # once, and the take was saved (2026-10-06). Asked whatever ended the reset: a take in another
                    # picture is not one of the 55 either way. A hand or the arm in the way is waited out.
                    # When a stop has ended the reset the arm stands where it stood and cannot be led: then a
                    # hand can still be taken out of the picture, and is asked out, but it is not waited on for long.
                    another_picture = changed_since(view, grab, told=hands_out,
                                                    **({"patience_s": AFTER_STOP_PATIENCE_S}
                                                       if events["stop_recording"] else {}))
                    if another_picture:
                        logging.error("episode %d: %s; the take is NOT saved", episode.episode_index, another_picture)
                        log_say("Camera picture changed. Take not saved.", play_sounds)
                        print(f"\n  After that take, {another_picture}.\n  The take is not saved, and the sitting "
                              "stops here: what was saved before is kept.\n  Run scripts/rig/measure.py check, say "
                              "what it prints, and continue with --resume.")
                        dataset.clear_episode_buffer()
                        events["stop_recording"] = True
                        break

                    # ... and the wrist camera's, by its one number against this session's first reading. Only with
                    # the arm at its start pose: anywhere else its picture is not the rest picture and says nothing.
                    try:
                        # (the reset's own limits, and its usual ones where an option has switched them off: the
                        # wrist camera's number means nothing away from the start pose)
                        at_rest = wrist_start is not None and not off_start_pose(
                            start_pose, robot.bus.sync_read("Present_Position", num_retry=3),
                            args.start_pose_tolerance or 5.0, args.free_joint_tolerance or 12.0)
                    except Exception as error:  # noqa: BLE001 -- a finished take is not lost to one failed reading
                        logging.warning("episode %d: the joints could not be read after the take (%s): the wrist "
                                        "camera is not asked", episode.episode_index, error)
                        at_rest = False
                    wrist_now, wrist_wrong = wrist_changed_since(wrist_start, grab_wrist) if at_rest else (None, None)
                    logging.info("episode %d: wrist camera after the take: %s", episode.episode_index,
                                 said_of_wrist(wrist_now) if at_rest else "not asked (the arm is not at its start "
                                 "pose, or the session's first reading could not be judged)")
                    if wrist_wrong and not args.any_wrist_state:
                        logging.error("episode %d: %s; the take is NOT saved", episode.episode_index, wrist_wrong)
                        log_say("Wrist camera picture changed. Take not saved.", play_sounds)
                        print(f"\n  After that take, {wrist_wrong}.\n  The take is not saved, and the sitting stops "
                              "here: what was saved before is kept.\n  Continue with --resume: the camera is opened "
                              "afresh. scripts/rig/usb_errors.py says whether its link is dropping frames; "
                              "--any-wrist-state records all the same.")
                        dataset.clear_episode_buffer()
                        events["stop_recording"] = True
                        break
                    if wrist_wrong:
                        logging.warning("episode %d: %s (--any-wrist-state)", episode.episode_index, wrist_wrong)

                    slip = too_short(frames, args.fps, args.min_episode_seconds)
                    if slip:
                        logging.warning("episode %d: the take was %d frames (%.1f s), under %.1f s: not saved, "
                                        "recorded again", episode.episode_index, frames, frames / args.fps,
                                        args.min_episode_seconds)
                        print(f"\n  That take lasted {frames / args.fps:.1f} s: the go key pressed twice? "
                              "It is not saved. Same episode again.")
                    still = not slip and stood_still(travel, args.min_travel)
                    if still:
                        logging.warning("episode %d: no joint moved by %g degrees in the take (the most: %.1f): "
                                        "not saved, recorded again", episode.episode_index, args.min_travel, travel)
                        print(f"\n  The arm did not move in that take (no joint by more than {travel:.1f} degrees): "
                              "the go key pressed by mistake? It is not saved. Same episode again.")
                    if events["rerecord_episode"] or slip or still:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        rerecords += 1
                        if events["stop_recording"]:
                            break
                        continue

                    record_session.save_episode_announced(dataset, play_sounds)
                    append_placement(placement_log, {
                        "dataset": sitting.dataset,
                        "episode_index": episode.episode_index,
                        "round": episode.round,
                        "sitting": sitting.number,
                        "cell": episode.cell,
                        "target": {"x_mm": episode.target.x_mm, "y_mm": episode.target.y_mm},
                        "placed": {"x_mm": placed.measured.x_mm, "y_mm": placed.measured.y_mm},
                        "placement_error_mm": placed.error_mm,
                        "rerecords": rerecords,
                        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "session": stamp,
                    })
                    break

            logging.info("Encoding video -- the arm is still HOLDING position.")

        if not events["stop_recording"]:
            logging.info("Cool down: park the arm at its final rest pose, then press 'n'.")
            session.teleop_phase("Cool down", robot, teleop, procs, args.fps, args.cooldown_max, events,
                                 play_sounds, gate, display_data=args.display)
            events["rerecord_episode"] = False

    finally:
        log_say("Stop recording", play_sounds, blocking=True)
        try:
            dataset.finalize()
        except Exception:
            logging.exception("dataset.finalize() failed -- the dataset may be incomplete")

        arm_state = session.settle_arm(robot)
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        cv2.destroyAllWindows()

        logging.info("cleanup complete -- arm %s", arm_state)
        # Each error is a frame the system dropped, and now and then the stream comes out of one in another state
        # (2026-10-06). The arm is settled; the figure goes into this session's log.
        dropped = usb_errors.over_session(session_started, time.time(), {camera_list.FRONT_NAME: "front camera",
                                                                         camera_list.WRIST_NAME: "wrist camera"})
        logging.info("%s", dropped)
        print(f"\n{dropped}")
        print(f"\nlog:        {log_path}")
        print(f"dataset:    {Path(HF_LEROBOT_HOME) / repo_id}")
        print(f"placements: {placement_log}")
        print("\n" + session.arm_state_line(arm_state))

    return ended


def cli(argv: list[str] | None = None) -> int:
    """Turn a crash into an exit code with the traceback in the session log, as record_session's cli does."""
    try:
        rc = main(argv) or 0
        if rc == 0:
            logging.info("session ended cleanly")
        return rc
    except KeyboardInterrupt:
        logging.warning("interrupted by Ctrl-C")
        return 130
    except Exception:
        logging.exception("SESSION FAILED")
        print("\nSESSION FAILED -- traceback above, and in the session log.")
        return 2
