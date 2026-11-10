"""How many USB transfer errors the system logged on each camera's video link, minute by minute.

On 2026-10-06 the front camera's link, through the hub it had been on all day, gave one transfer error every two to
three seconds while it streamed. Each is a video frame that fails to decode and is dropped; and twice that evening
the stream came out of it in another state (the camera on its own zoom and exposure 3.4 seconds into a take, the tone
range stretched a few takes later). On another hub the same camera gave none in half an hour. Nothing at the rig
showed the errors: they are lines of the system's own log.

This reads those lines (`log show`, the kernel's `transaction error` on a camera's video endpoint). It opens no camera
and needs no arm. A link is good when a camera that has been streaming for some minutes has no line at all.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping

#: What the kernel writes for one failed transfer, with the device's name and the port it is plugged into.
ERROR = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d):(\d\d):\d\d\.\d+ .*\(([^()@]+)@([0-9a-fA-F]+)\) endpoint 0x[0-9a-fA-F]+: "
                   r"status 0x[0-9a-fA-F]+ \(transaction error\)")
PREDICATE = 'process == "kernel" AND eventMessage CONTAINS "transaction error"'


class LogError(RuntimeError):
    """The system's log could not be read."""


def system_log(minutes: int, timeout_s: float = 120.0) -> list[str]:
    """The kernel's transfer-error lines of the last `minutes` minutes."""
    try:
        done = subprocess.run(["/usr/bin/log", "show", "--last", f"{int(minutes)}m", "--style", "compact",
                               "--predicate", PREDICATE], capture_output=True, text=True, timeout=timeout_s,
                              check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LogError(f"the system's log could not be read: {error}") from error
    if done.returncode != 0:
        raise LogError(f"the system's log could not be read (exit {done.returncode})")
    return done.stdout.splitlines()


def per_minute(lines: Iterable[str], names: Iterable[str]) -> dict[str, Counter]:
    """For every device whose name is in `names`: the errors in each minute ("HH:MM") that has any.

    A device is counted under its name whatever port it was in; `ports` gives the ports seen for it.
    """
    wanted = set(names)
    counts: dict[str, Counter] = {name: Counter() for name in wanted}
    for line in lines:
        found = ERROR.match(line)
        if found and found.group(4) in wanted:
            counts[found.group(4)][f"{found.group(2)}:{found.group(3)}"] += 1
    return counts


def ports(lines: Iterable[str], names: Iterable[str]) -> dict[str, set[str]]:
    """The ports each named device was in when an error was logged for it."""
    wanted = set(names)
    seen: dict[str, set[str]] = {name: set() for name in wanted}
    for line in lines:
        found = ERROR.match(line)
        if found and found.group(4) in wanted:
            seen[found.group(4)].add(found.group(5))
    return seen


def since(lines: Iterable[str], name: str, start: str) -> int:
    """How many of the log's lines are transfer errors of the device `name` at or after `start`, a local time
    written as the log writes it ("2026-10-06 20:52:45")."""
    count = 0
    for line in lines:
        found = ERROR.match(line)
        if found and found.group(4) == name and line[:19] >= start:
            count += 1
    return count


def over_session(started_at: float, now: float, names: Mapping[str, str], timeout_s: float = 30.0) -> str:
    """One sentence on the errors of a session that began at `started_at` and is asked at `now` (both seconds of
    `time.time()`, given by the session): for each device in `names` (its name, and what to call it) how many.
    Never raises: a log that cannot be read is said.

    Asked by a session at its end, after the arm is settled, so that every session's log carries the figure.
    """
    minutes = max(1, math.ceil((now - started_at) / 60.0) + 1)
    try:
        counts = per_minute(system_log(minutes, timeout_s), names)
    except Exception as error:  # noqa: BLE001 -- a session's end must not fail on its last line
        return f"USB transfer errors on the cameras' links: not read ({error})"
    told = ", ".join(f"{role} {sum(counts[name].values())}" for name, role in names.items())
    return f"USB transfer errors on the cameras' links in the last {minutes} minutes: {told}"
