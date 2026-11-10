"""A go key pressed by itself, told from one that came in the middle of typing.

The sessions hear their keys through a listener on the whole computer: n is the go wherever it is typed. On
2026-10-06 the operator typed a message in another program while a session ran, twice. In a recording the go starts a
take, and a take in which nobody moved the arm is thrown away afterwards. In a trial the go hands the motors to a
policy: a go that is a letter of somebody's sentence must not count.

A key typed in prose has neighbours: the keys before and after it come within a second or so. A go that is meant is
pressed alone. The guard notes the time of every key the listener hears, and calls a go alone when no other key came
in the two seconds before it or in the half second after it. A go therefore takes effect half a second after it is
pressed.

The guard only ever refuses a go; it cannot make one. When it has not heard the key at all (the go reached the
session another way), it says so and the go counts as it did before there was a guard.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, MutableMapping
from typing import Any

from tolerance_radius.rig.placement_phase import consume_go

#: The keys LeRobot's listener takes for "go on": the letter and the right arrow.
GO_KEYS = ("n", "right")
#: No other key in this long before a go, and in this long after it, or the go was typed.
QUIET_BEFORE_S = 2.0
QUIET_AFTER_S = 0.5
#: The session sees the go's flag, the guard has heard no go key: after this long the guard is taken to be deaf to it.
DEAF_AFTER_S = 1.5

WAIT, ALONE, TYPED, UNHEARD = "wait", "alone", "typed", "unheard"


class TypingGuard:
    """The times of the keys a listener heard, and what they say about the latest go."""

    def __init__(self, quiet_before_s: float = QUIET_BEFORE_S, quiet_after_s: float = QUIET_AFTER_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._before, self._after, self._clock = quiet_before_s, quiet_after_s, clock
        self._keys: deque[tuple[float, bool]] = deque(maxlen=512)  # when, and whether it was a go key
        self._judged_until = float("-inf")
        self._seen_until = float("-inf")
        self._flag_since: float | None = None
        #: How many goes were refused as typing, and how many were let through unheard.
        self.refused = 0
        self.unheard = 0

    def saw(self, key: Any) -> None:
        """Note a key the listener heard. Called on the listener's thread, for every key; never raises."""
        try:
            self._keys.append((self._clock(), str(key).lower() in GO_KEYS))
        except Exception:  # noqa: BLE001 -- whatever happens here, the key must still reach the session
            pass

    def judge(self) -> str:
        """What the keys heard say of the latest go.

        UNHEARD: no go key heard since the last judgement. WAIT: one was heard and its half second is not over.
        TYPED: another key came within the quiet time before the go or after it. ALONE: none did.

        The go is the last run of go keys that follow each other within the quiet time (a doubled n is one go). A
        go key from longer ago is not part of it: the n that ended a reset is never asked about here, and with it
        counted, any key pressed since the reset made the next placement's go a typed one (the review of
        2026-10-06: a stray key half a minute before a go pressed alone, and the go was refused).
        """
        now = self._clock()
        keys = list(self._keys)  # the listener's thread may add to it meanwhile
        goes = [when for when, go in keys if go and when > self._judged_until]
        if not goes:
            return UNHEARD
        first = last = goes[-1]
        for when in reversed(goes[:-1]):
            if first - when > self._before:
                break
            first = when
        self._seen_until = last
        if any(not go and first - self._before <= when <= last + self._after for when, go in keys):
            return TYPED
        return ALONE if now >= last + self._after else WAIT

    def judged(self) -> None:
        """The session has acted on the go keys `judge` last looked at: later ones are judged on their own.

        What is stamped is the last go key looked at, not the clock: a go key heard between the look and this call
        has not been judged, and stamped by the clock it never would be (it then counted as a go the guard had not
        heard, a second and a half later: the one way a typed n could get through, found in the review of
        2026-10-06).
        """
        self._judged_until = max(self._judged_until, self._seen_until)
        self._flag_since = None

    def forget(self) -> None:
        """Every go key heard so far is done with, whatever it was (the session is stopping, or recording again)."""
        self.judge()
        self.judged()

    def deaf_for(self) -> float:
        """How long the session has been showing a go the guard has not heard, in seconds."""
        now = self._clock()
        if self._flag_since is None:
            self._flag_since = now
        return now - self._flag_since

    def no_go_shown(self) -> None:
        """The session shows no go at the moment: a later unheard one is timed from its own beginning."""
        self._flag_since = None


def guarded_go(events: MutableMapping[str, Any], guard: TypingGuard | None,
               told: Callable[[str], None] | None = None) -> bool:
    """`consume_go`, with a go that came in the middle of typing refused.

    True once when the operator pressed the go key by itself. A stop or a re-record press is handled as
    `consume_go` handles it. While the guard cannot say yet (the half second after the key), the go's flag is left
    standing and this is asked again at the next frame. `told` is given the sentence for the operator when a go is
    refused. Without a guard this is `consume_go`.
    """
    if guard is None:
        return consume_go(events)
    if not events.get("exit_early"):
        guard.no_go_shown()
        return False
    if events.get("stop_recording") or events.get("rerecord_episode"):
        guard.forget()
        return consume_go(events)
    verdict = guard.judge()
    if verdict == WAIT:
        return False
    if verdict == UNHEARD:
        if guard.deaf_for() < DEAF_AFTER_S:
            return False
        guard.unheard += 1
        logging.warning("the typing guard did not hear the go key: the go counts as it would without the guard")
        guard.judged()
        return consume_go(events)
    guard.judged()
    took = consume_go(events)
    if verdict == ALONE:
        return took
    guard.refused += 1
    logging.warning("a go key came in the middle of typing: not taken as the go")
    if told is not None:
        told("That n came in the middle of typing (keys are heard from every program): it is not the go. "
             "Press n by itself.")
    return False


def forget_go(events: MutableMapping[str, Any], guard: TypingGuard | None) -> bool:
    """A go pressed while nothing can take it (the cube is not steady in its ring) is dropped now. True when there
    was one.

    `guarded_go` leaves a go standing for its half second; a go pressed up to that long before the cube became
    steady then started the trial (the review of 2026-10-06), where the recorder's rule forgets it. A stop is left
    set for the session, and a re-record press is forgotten, as `consume_go` does.
    """
    if not events.get("exit_early"):
        return False
    dropped = consume_go(events)
    if guard is not None:
        guard.forget()
    return dropped


#: How long `phase_go` waits for a verdict at most: a go key held down keeps its half second from ending.
PHASE_GO_WAIT_S = 10.0


def phase_go(guard: TypingGuard | None, told: Callable[[str], None] | None = None,
             sleep: Callable[[float], None] = time.sleep) -> bool:
    """Was the go that has just ended a phase pressed by itself? Waits out its half second, then says.

    For a phase whose end starts something that moves by itself: after a refused handoff the operator brings the
    arm closer and presses n, and the policy is given the motors next. That n is not a placement's go and was not
    looked at (the review of 2026-10-06: a typed sentence started the policy there). False when the key came in
    the middle of typing, or would not settle; the caller then runs the phase again. Without a guard, and for a go
    the guard did not hear, True: as it was before there was a guard.
    """
    if guard is None:
        return True
    verdict = guard.judge()
    waited = 0.0
    while verdict == WAIT and waited < PHASE_GO_WAIT_S:
        sleep(0.05)
        waited += 0.05
        verdict = guard.judge()
    guard.judged()
    if verdict not in (TYPED, WAIT):
        return True
    guard.refused += 1
    logging.warning("the go that ended the phase came in the middle of typing: the phase is run again")
    if told is not None:
        told("That n came in the middle of typing (keys are heard from every program): it is not the go. "
             "Press n by itself.")
    return False


class Keyboard:
    """The session's keys for its whole length, from one listener that is started once.

    The arm tooling's `PausableKeyboard` silences the keys while the operator types (a score, a note) by stopping
    the listener and starting a new one afterwards. On this computer a listener started after a window has been
    shown ends the whole program: pynput reads the keyboard's layout on the new listener's own thread, and macOS
    stops a process that does that once it has shown a window (*measured* 2026-10-06 with the layout read alone:
    two reads before a window pass, the read after it ends the process with a trace trap,
    `TSMGetInputSourceProperty` under `dispatch_assert_queue`; not tried with a listener). A trial session shows
    the placement's window before its first score, so its first score would have been its last line, before the
    trial was logged and without the arm being settled. The recorder never met it: it starts one listener, before
    any window.

    Here the listener is started once, before any window, and stays. While the keys are paused it goes on hearing
    them and passes none on: no flag changes, and the typing guard is told nothing. What was set before the pause
    (a stop pressed during the attempt) is still set after it.

    The keys and what they set are LeRobot's (`lerobot.utils.keyboard_input.init_keyboard_listener`): n or the right
    arrow, go on; r or the left arrow, record again; q or Esc, stop; through LeRobot's own
    `apply_recording_control`. A test presses the same keys on both.
    """

    def __init__(self, guard: TypingGuard | None = None) -> None:
        from lerobot.utils import keyboard_input

        self._apply = keyboard_input.apply_recording_control
        self._guard = guard
        self._paused = False
        self.events: dict[str, bool] = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        self._listener = keyboard_input.create_key_listener(
            self._on_key, controls_help="Right/Left/Esc, or n=next, r=re-record, q=quit")

    @property
    def paused(self) -> bool:
        return self._paused

    def _on_key(self, name: Any) -> None:
        if self._paused:
            return
        if self._guard is not None:
            self._guard.saw(name)
        key = str(name).lower()
        if key in ("right", "n"):
            self._apply("right", self.events)
        elif key in ("left", "r"):
            self._apply("left", self.events)
        elif key in ("esc", "q"):
            self._apply("esc", self.events)

    def pause(self) -> None:
        """From now on no key reaches the session, until `resume`: the operator is typing."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        """End the listener, at the session's end."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
