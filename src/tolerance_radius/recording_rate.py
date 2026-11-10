"""The rate a sitting's control loop held while it recorded, take by take, read from the session's own log.

An episode's frames carry nominal times (frame n at n / 30 s), so a loop that ran slow leaves no trace in the dataset:
in August 16 of 55 episodes were recorded at about 28 Hz and stamped 30. While every episode lasted a fixed time its
frame count showed that. Since 2026-10-06 the operator ends a take with a key, the episodes are not of one length and
the count shows nothing. The session's log does: the arm tooling writes one line per control cycle (the follower's
state read), with the clock's time to the millisecond.

Nothing here touches the rig, and nothing needs the arm tooling or LeRobot: text files and one table.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

#: One line of a session's log: the clock's time, the level, the message.
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) ([A-Z]+) (.*)$")
#: The recorder's loop begins a take with this line.
TAKE_BEGINS = re.compile(r"^Recording episode (\d+)$")
#: One control cycle: the follower's state is read once in every cycle of the loop.
CYCLE = re.compile(r"^\S+ SOFollower read state: ")
#: What follows a take in the recorder: the reset's first line, the stop key's note, or the session's end. Anything
#: else that is not a cycle (the clock's "5 seconds", a warning from the loop itself) belongs to the take.
TAKE_ENDS = ("Reset the environment", "Encoding video", "Stop recording", "SESSION FAILED", "interrupted by")
TAKE_CUT = re.compile(r"^episode \d+: stopped during the take")

#: A take that held less than this share of the rate asked for needs eyes: 2 %, the share the protocol gives the
#: trials' loop (section 6).
SHORTFALL = 0.02


@dataclass(frozen=True, slots=True)
class Take:
    """One run of the recording loop, as the log has it."""

    episode: int
    #: Control cycles the log shows between the take's first line and what ended it.
    cycles: int
    #: From the first cycle's line to the last one's, in seconds.
    seconds: float
    began: str

    @property
    def hz(self) -> float | None:
        """Cycles per second from the first cycle to the last; None for a take too short to have a rate."""
        if self.cycles < 2 or self.seconds <= 0:
            return None
        return (self.cycles - 1) / self.seconds


def _clock(stamp: str, millis: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp() + int(millis) / 1000.0


def takes_in(lines: Iterable[str]) -> list[Take]:
    """Every take a session's log holds, in order: also the ones recorded again or cut short."""
    takes: list[Take] = []
    episode: int | None = None
    began = ""
    first = last = 0.0
    cycles = 0

    def close() -> None:
        nonlocal episode
        if episode is not None:
            takes.append(Take(episode, cycles, last - first if cycles else 0.0, began))
        episode = None

    for line in lines:
        found = LINE.match(line.rstrip("\n"))
        if not found:
            continue
        stamp, millis, level, message = found.groups()
        begins = TAKE_BEGINS.match(message) if level != "DEBUG" else None
        if begins:
            close()
            episode, began, cycles = int(begins.group(1)), stamp, 0
            continue
        if episode is None:
            continue
        if level == "DEBUG":
            if CYCLE.match(message):
                last = _clock(stamp, millis)
                if cycles == 0:
                    first = last
                cycles += 1
        elif message.startswith(TAKE_ENDS) or TAKE_CUT.match(message):
            close()
    close()
    return takes


def saved_takes(placements: Sequence[dict[str, Any]], log_dir: Path) -> list[dict[str, Any]]:
    """The take that became each saved episode: the last one of that episode in the session that saved it.

    `placements` are the lines of a dataset's placement log. An episode whose session log is not there, or holds no
    take of it, comes back with `cycles` None: said, not passed over.
    """
    by_session: dict[str, list[Take]] = {}
    rows = []
    for entry in sorted(placements, key=lambda e: int(e["episode_index"])):
        session = str(entry["session"])
        if session not in by_session:
            path = log_dir / f"session_{session}.log"
            by_session[session] = takes_in(path.read_text(encoding="utf-8", errors="replace").splitlines()) \
                if path.is_file() else []
        mine = [take for take in by_session[session] if take.episode == int(entry["episode_index"])]
        row: dict[str, Any] = {"episode_index": int(entry["episode_index"]), "session": session,
                               "takes_in_session": len(mine), "cycles": None, "seconds": None, "hz": None}
        if mine:
            take = mine[-1]
            row.update(cycles=take.cycles, seconds=round(take.seconds, 3),
                       hz=None if take.hz is None else round(take.hz, 3))
        rows.append(row)
    return rows


def summary(rows: Sequence[dict[str, Any]], fps: float, lengths: dict[int, int] | None = None) -> dict[str, Any]:
    """What the rows say together: the median and the lowest rate, the takes that fell short, and what could not
    be read. `lengths` are the dataset's own frame counts by episode: a take whose cycles differ from its episode's
    frames is named, because then the log's take is not the one that was saved."""
    rates = [row["hz"] for row in rows if row["hz"] is not None]
    floor = fps * (1.0 - SHORTFALL)
    told: dict[str, Any] = {
        "fps_asked": fps,
        "episodes": len(rows),
        "with_a_rate": len(rates),
        "median_hz": round(statistics.median(rates), 3) if rates else None,
        "lowest_hz": round(min(rates), 3) if rates else None,
        "short_of": round(floor, 3),
        "fell_short": [row["episode_index"] for row in rows if row["hz"] is not None and row["hz"] < floor],
        "not_read": [row["episode_index"] for row in rows if row["hz"] is None],
    }
    if lengths is not None:
        told["frames_differ"] = [row["episode_index"] for row in rows
                                 if row["cycles"] is not None and lengths.get(row["episode_index"]) != row["cycles"]]
    return told


def read_placements(path: Path) -> list[dict[str, Any]]:
    """The lines of a placement log; an empty list when the file is not there."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
