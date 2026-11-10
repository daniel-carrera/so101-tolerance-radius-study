"""The adaptive rule that chooses the next distance to test.

A staircase moves the test distance up after a success and down after a failure, so its trials gather around the
distance at which success is 50 %: the place where each trial says the most about the tolerance radius. Trials spent
where the policy always works, or never does, say little.

The rule is a pure function of the staircase's own history, so the planner can rebuild its state from the trial log
after any interruption.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StaircaseParams:
    """One staircase: where it starts, how long it runs, how far it steps and where it may go."""

    start_level_mm: float
    n_trials: int
    #: Step used until the staircase first reverses direction: large, to reach the interesting region quickly.
    step_initial_mm: float
    #: Step used after the first reversal: small, to resolve the threshold.
    step_final_mm: float
    min_level_mm: float
    max_level_mm: float

    def __post_init__(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("a staircase runs at least one trial")
        if self.step_initial_mm <= 0 or self.step_final_mm <= 0:
            raise ValueError("steps are positive")
        if not self.min_level_mm <= self.start_level_mm <= self.max_level_mm:
            raise ValueError("the starting level lies between the limits")


def next_level(params: StaircaseParams, history: Sequence[tuple[float, bool]]) -> float | None:
    """The distance to request for this staircase's next trial, or None when it has run all its trials.

    `history` holds this staircase's scored trials, oldest first, as `(requested level in mm, success)`. Void trials
    are left out by the caller, so a void trial is simply asked again.

    One up, one down: a success moves the level farther from the demonstrations (harder), a failure moves it closer.
    The level never leaves `[min_level_mm, max_level_mm]`; at a limit it stays there until the outcome turns it back.
    """
    if len(history) >= params.n_trials:
        return None
    if not history:
        return params.start_level_mm
    last_level, last_success = history[-1]
    step = params.step_final_mm if reversal_indices(history) else params.step_initial_mm
    proposed = last_level + step if last_success else last_level - step
    return min(max(proposed, params.min_level_mm), params.max_level_mm)


def reversal_indices(history: Sequence[tuple[float, bool]]) -> list[int]:
    """Indices of the trials at which the staircase changed direction.

    Trial `i` is a reversal when its outcome differs from the outcome of trial `i - 1`: the staircase had been moving
    one way and now moves the other.
    """
    return [i for i in range(1, len(history)) if history[i][1] != history[i - 1][1]]
