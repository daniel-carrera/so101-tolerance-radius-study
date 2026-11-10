"""What a trial session must satisfy before it may run, as pure functions.

Two things are checked before the arm is connected, so that a problem costs a sentence and not a block of trials:

* **The execution settings.** How a policy or a replay is run (control rate, clamp, action smoothing, handoff
  tolerances, stall rule, time limit, how far the camera's picture may have crept) is part of the protocol (`[execution]` in `config/protocol.toml`). A counted
  session must run with exactly those values. In an earlier evaluation on this rig one such setting was off for part
  of the trials and nobody noticed, because the setting lived in a command line's default.
* **The blocks.** A staircase asks for whatever distance its rule gives; if the table has no point that far from the
  demonstrations as they were really recorded, the planner stops in the middle of the block. `planner.check_block`
  finds that beforehand.

No hardware, no OpenCV: the trial runner calls these, and they are tested everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tolerance_radius import planner
from tolerance_radius.schema import DemoStart

#: The execution settings the protocol states, in the order of its `[execution]` section.
EXECUTION_KEYS = (
    "control_hz",
    "time_limit_s",
    "clamp_deg",
    "temporal_ensemble_coeff",
    "handoff_tolerance_deg",
    "replay_handoff_tolerance_deg",
    "max_handoff_retries",
    "min_attempt_s",
    "stall_steps",
    "stall_progress_deg",
    "stall_fraction",
    "view_crept_limit_mm",
)


def _same(first: Any, second: Any) -> bool:
    """True when two settings are the same value. 30 and 30.0 are; None and a number are not."""
    numbers = (int, float)
    if isinstance(first, bool) or isinstance(second, bool):
        return first is second
    if isinstance(first, numbers) and isinstance(second, numbers):
        return math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def execution_problems(protocol: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Where a session's execution settings differ from the protocol's, as sentences; empty when they agree.

    A setting the session lacks, or has switched off (`None`), is a difference. So is a setting the protocol does not
    state: what runs at the rig and what the protocol describes must be the same list.
    """
    stated = protocol.get("execution")
    if not stated:
        return ["the protocol has no [execution] section, so there is nothing to hold this session's settings to"]
    problems = []
    for key in stated:
        if key not in settings:
            problems.append(f"{key}: the protocol states {stated[key]!r}, and this session does not set it")
        elif not _same(settings[key], stated[key]):
            problems.append(f"{key} is {settings[key]!r} in this session and {stated[key]!r} in the protocol")
    for key in settings:
        if key not in stated:
            problems.append(f"{key} is set to {settings[key]!r} in this session, and the protocol does not state it")
    return problems


def resolved_execution(protocol: Mapping[str, Any], given: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The settings a session runs with, and which of them were changed by hand.

    `given` holds what the command line set, with `None` for "not given". A setting that was not given takes the
    protocol's value. The second result names the settings whose given value differs from the protocol's.
    """
    stated = dict(protocol.get("execution") or {})
    unknown = sorted(set(given) - set(stated))
    if unknown:
        raise KeyError(f"not an execution setting of the protocol: {unknown}")
    merged = dict(stated)
    changed = []
    for key in stated:
        value = given.get(key)
        if value is None:
            continue
        merged[key] = value
        if not _same(value, stated[key]):
            changed.append(key)
    return merged, changed


def block_problems(
    blocks: Sequence[str], protocol: Mapping[str, Any], demo_starts: Sequence[DemoStart]
) -> list[str]:
    """What would stop any of these blocks before its end, each sentence led by the block's name."""
    problems = []
    for block in blocks:
        for problem in planner.check_block(block, protocol, demo_starts):
            problems.append(problem if block in problem else f"{block}: {problem}")
    return problems
