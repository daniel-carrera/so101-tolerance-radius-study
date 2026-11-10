"""Open-loop replay: one demonstration's recorded actions, played back through the loop a policy runs in.

The replay is the study's null model. It sends the joint targets that were recorded during one demonstration, in
order, at the recorded rate, and looks at nothing. With the cube where it was in that demonstration the replay should
succeed; with the cube displaced, how far it still succeeds measures what the gripper and the arm tolerate with no
policy at all.

A `ReplaySource` stands where a policy stands: the arm tooling's `attempt_cycle` is given the source as its `policy`
(for `reset()`) and the source's `predict` as its `predict`. Every gate then applies exactly as for a policy: the handoff
gate judges the first recorded action against the arm's present pose, the per-step clamp limits every command, and the
stall rule watches for a blocked arm. This module never sends a command itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class ReplayError(ValueError):
    """The recorded actions cannot be replayed as asked."""


class ReplaySource:
    """One episode's recorded actions, handed out one per control step.

    `actions` is the episode's action array, one row per recorded step, columns in the order of `joint_names`.
    `observe`, if given, is called once per step and its result passed to `on_observation`, as a policy's inference
    step does: the trial's video is recorded and the servo bus is read at the same rate as in a policy trial.
    `events` is the session's flag dictionary; when the last recorded action has been handed out, `exit_early` is set in
    it, which is how the control loop is told that the attempt is over.
    """

    def __init__(
        self,
        actions: np.ndarray,
        joint_names: Sequence[str],
        *,
        observe: Callable[[], Any] | None = None,
        on_observation: Callable[[Any], None] | None = None,
        events: MutableMapping[str, Any] | None = None,
    ) -> None:
        array = np.asarray(actions, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ReplayError(f"actions must be a non-empty table of steps by joints, got shape {array.shape}")
        if array.shape[1] != len(joint_names):
            raise ReplayError(f"{array.shape[1]} action columns but {len(joint_names)} joint names")
        if not np.all(np.isfinite(array)):
            raise ReplayError("the recorded actions contain values that are not finite")
        self._actions = array
        self._keys = tuple(name if name.endswith(".pos") else f"{name}.pos" for name in joint_names)
        self._observe, self._on_observation, self._events = observe, on_observation, events
        self._step = 0

    def __len__(self) -> int:
        return self._actions.shape[0]

    @property
    def steps_handed_out(self) -> int:
        return self._step

    def reset(self) -> None:
        """Rewind to the first recorded action. Called by the handoff gate before it judges the opening move."""
        self._step = 0

    def predict(self) -> dict[str, float]:
        """The next recorded action, as the `{"<joint>.pos": value}` dictionary the follower accepts.

        After the last recorded action the same action is repeated (the arm holds its final recorded pose) until the
        control loop ends the attempt.
        """
        if self._observe is not None:
            observation = self._observe()
            if self._on_observation is not None:
                self._on_observation(observation)
        row = self._actions[min(self._step, len(self) - 1)]
        self._step += 1
        if self._step >= len(self) and self._events is not None:
            self._events["exit_early"] = True
        return {key: float(value) for key, value in zip(self._keys, row, strict=True)}

    def largest_step(self) -> float:
        """The largest change of any joint between two consecutive recorded actions: what the clamp will be asked for."""
        if len(self) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(self._actions, axis=0))))


def load_episode_actions(dataset_root: str | Path, episode_index: int) -> tuple[np.ndarray, list[str]]:
    """Read one episode's recorded actions from a LeRobot dataset on disk, with the joint names of its columns.

    Reads the tabular files only; no video is decoded. Rows are returned in frame order.
    """
    import pandas as pd  # the rig's environment has it; the core of the package does not need it

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ReplayError(f"no dataset at {root} (missing meta/info.json)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = info.get("features", {}).get("action", {}).get("names")
    if isinstance(names, dict):  # some versions nest the list under a key such as "motors"
        names = next(iter(names.values()))
    if not names:
        raise ReplayError(f"{info_path} does not name the action's columns")

    tables = sorted((root / "data").rglob("*.parquet"))
    if not tables:
        raise ReplayError(f"no data files under {root / 'data'}")
    rows = []
    for table in tables:
        # use_threads=False on purpose: with the reader's thread pool, a process in the rig's environment that has
        # read these files fails to exit about half the time (measured 2026-10-05: 3 of 5 runs hung at exit with
        # threads, 0 of 5 without). The files are small; one thread costs nothing.
        frame = pd.read_parquet(table, columns=["action", "episode_index", "frame_index"], use_threads=False)
        rows.append(frame[frame["episode_index"] == episode_index])
    episode = pd.concat(rows).sort_values("frame_index")
    if episode.empty:
        raise ReplayError(f"episode {episode_index} is not in the dataset at {root}")
    frames = episode["frame_index"].to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ReplayError(f"episode {episode_index}: its frames are not one unbroken run from 0")
    actions = np.stack([np.asarray(action, dtype=float) for action in episode["action"]])
    return actions, list(names)
