"""The guard that tells a go key pressed by itself from one typed in a sentence (`tolerance_radius.rig.typing_guard`)."""

from __future__ import annotations

import contextlib
import io
import logging
import unittest

from tolerance_radius.rig import typing_guard
from tolerance_radius.rig.typing_guard import ALONE, TYPED, UNHEARD, WAIT, TypingGuard, guarded_go


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def typed(guard: TypingGuard, clock: Clock, text: str, at: float, gap: float = 0.15) -> float:
    """Type `text` from time `at`, one key every `gap` seconds, as a listener would hear it. Returns the end."""
    for index, char in enumerate(text):
        clock.now = at + index * gap
        guard.saw("space" if char == " " else char)
    return at + (len(text) - 1) * gap


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.guard = TypingGuard(clock=self.clock)

    def at(self, when: float) -> str:
        self.clock.now = when
        return self.guard.judge()

    def test_nothing_heard_is_unheard(self):
        self.assertEqual(self.at(100.0), UNHEARD)
        typed(self.guard, self.clock, "hello", 100.0)
        self.assertEqual(self.at(105.0), UNHEARD)                  # keys, and no go among them

    def test_a_go_pressed_by_itself_is_alone_once_its_half_second_is_over(self):
        typed(self.guard, self.clock, "n", 110.0)
        self.assertEqual(self.at(110.2), WAIT)
        self.assertEqual(self.at(110.49), WAIT)
        self.assertEqual(self.at(110.5), ALONE)
        self.assertEqual(self.at(140.0), ALONE)                    # and stays so until the session has acted on it

    def test_the_right_arrow_is_a_go_too_and_a_capital_letter(self):
        for key in ("right", "N"):
            with self.subTest(key=key):
                guard = TypingGuard(clock=self.clock)
                self.clock.now = 110.0
                guard.saw(key)
                self.clock.now = 111.0
                self.assertEqual(guard.judge(), ALONE)

    def test_the_message_typed_during_sitting_1_would_not_have_been_a_go(self):
        # 2026-10-06, 18:01, typed in another program while the placement of episode 3 waited.
        end = typed(self.guard, self.clock, "how to stop recording? I need to go. Actually I think it is exiting "
                                            "already since I pressed n lots of times typing this", 110.0)
        self.assertEqual(self.at(end + 0.01), TYPED)
        self.assertEqual(self.at(end + 30.0), TYPED)

    def test_a_letter_n_is_typed_whatever_its_place_in_the_word(self):
        cases = {"first of a word after a pause": ("now", 0), "last of a word, then a pause": ("in", 1),
                 "a word of its own": ("x n y", 2), "inside a word": ("and", 1)}
        for name, (text, _place) in cases.items():
            with self.subTest(case=name):
                guard = TypingGuard(clock=self.clock)
                end = typed(guard, self.clock, text, 200.0)
                self.clock.now = end + 5.0
                self.assertEqual(guard.judge(), TYPED)

    def test_another_key_shortly_before_or_after_makes_it_typed(self):
        for other_at, expected in ((108.1, TYPED), (107.9, ALONE), (110.4, TYPED), (110.6, ALONE)):
            with self.subTest(other_at=other_at):
                guard = TypingGuard(clock=self.clock)
                for when, key in sorted(((other_at, "x"), (110.0, "n"))):
                    self.clock.now = when
                    guard.saw(key)
                self.clock.now = 112.0
                self.assertEqual(guard.judge(), expected)

    def test_a_key_after_the_go_refuses_it_at_once_without_waiting(self):
        typed(self.guard, self.clock, "no", 110.0)
        self.assertEqual(self.at(110.2), TYPED)

    def test_a_doubled_go_with_nothing_else_is_alone(self):
        typed(self.guard, self.clock, "nn", 110.0, gap=0.2)
        self.assertEqual(self.at(110.4), WAIT)                      # the half second runs from the last of them
        self.assertEqual(self.at(110.7), ALONE)

    def test_after_the_session_has_acted_a_later_go_is_judged_on_its_own(self):
        end = typed(self.guard, self.clock, "running now", 110.0)
        self.assertEqual(self.at(end + 1.0), TYPED)
        self.guard.judged()
        self.assertEqual(self.at(end + 2.0), UNHEARD)
        typed(self.guard, self.clock, "n", end + 10.0)
        self.assertEqual(self.at(end + 11.0), ALONE)
        # ... but a go typed right after other keys is not excused by the earlier judgement
        self.guard.judged()
        typed(self.guard, self.clock, "x", end + 20.0)
        typed(self.guard, self.clock, "n", end + 21.0)
        self.assertEqual(self.at(end + 22.0), TYPED)

    def test_a_go_that_ended_an_earlier_phase_does_not_make_the_next_go_a_typed_one(self):
        # The review of 2026-10-06: the n that ends a reset is never asked about; half a minute later one other key
        # somewhere; a minute after the reset the operator presses n by itself. It was refused as typing.
        typed(self.guard, self.clock, "n", 100.0)
        typed(self.guard, self.clock, "x", 130.0)
        typed(self.guard, self.clock, "n", 160.0)
        self.assertEqual(self.at(160.1), WAIT)
        self.assertEqual(self.at(160.6), ALONE)

    def test_the_go_is_the_last_run_of_go_keys_and_a_typed_word_is_one_run(self):
        end = typed(self.guard, self.clock, "running", 110.0)       # three n, each within the quiet time of the next
        self.assertEqual(self.at(end + 5.0), TYPED)
        # two goes more than the quiet time apart are two goes: a key between them, near neither, refuses neither
        guard = TypingGuard(clock=self.clock)
        for when, key in ((200.0, "n"), (205.0, "x"), (210.0, "n")):
            self.clock.now = when
            guard.saw(key)
        self.clock.now = 211.0
        self.assertEqual(guard.judge(), ALONE)
        # ... and near the earlier one only, it still does not refuse the later
        guard = TypingGuard(clock=self.clock)
        for when, key in ((300.0, "x"), (300.2, "n"), (310.0, "n")):
            self.clock.now = when
            guard.saw(key)
        self.clock.now = 311.0
        self.assertEqual(guard.judge(), ALONE)

    def test_a_go_heard_after_the_look_and_before_the_stamp_is_still_judged(self):
        # The one interleaving of the two threads in which a typed n counted (the review of 2026-10-06): the second
        # n of "running" is heard after `judge` has copied the keys and before `judged` is called.
        for when, key in ((110.00, "r"), (110.10, "u"), (110.20, "n")):
            self.clock.now = when
            self.guard.saw(key)
        self.assertEqual(self.at(110.3199), TYPED)
        self.clock.now = 110.3200
        self.guard.saw("n")                                         # the listener's thread
        self.clock.now = 110.3201
        self.guard.judged()                                         # the session's thread
        self.assertEqual(self.at(110.36), TYPED, "the second n must not pass as a go nobody heard")
        self.assertEqual(self.at(112.0), TYPED)

    def test_forgetting_puts_every_go_heard_so_far_behind(self):
        typed(self.guard, self.clock, "n", 110.0)
        self.clock.now = 110.1
        self.guard.forget()
        self.assertEqual(self.at(111.0), UNHEARD)
        typed(self.guard, self.clock, "n", 120.0)
        self.assertEqual(self.at(121.0), ALONE)

    def test_a_key_the_guard_cannot_read_does_no_harm(self):
        class Odd:
            def __str__(self):
                raise RuntimeError("no name")

        self.guard.saw(Odd())
        self.guard.saw(None)
        self.guard.saw(7)
        self.assertEqual(self.at(101.0), UNHEARD)


class GuardedGoTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.guard = TypingGuard(clock=self.clock)
        self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        self.said: list[str] = []
        logging.disable(logging.CRITICAL)                          # the refusals are logged; not into the test's output
        self.addCleanup(logging.disable, logging.NOTSET)

    def press(self, key: str, when: float) -> None:
        """A key as LeRobot's listener and the guard hear it together."""
        self.clock.now = when
        self.guard.saw(key)
        if key in ("n", "right"):
            self.events["exit_early"] = True
        elif key in ("r", "left"):
            self.events.update(rerecord_episode=True, exit_early=True)
        elif key in ("q", "esc"):
            self.events.update(stop_recording=True, exit_early=True)

    def go(self, when: float) -> bool:
        self.clock.now = when
        return guarded_go(self.events, self.guard, self.said.append)

    def test_a_go_by_itself_counts_half_a_second_after_it_was_pressed_and_only_once(self):
        self.assertFalse(self.go(109.0))
        self.press("n", 110.0)
        self.assertFalse(self.go(110.1))
        self.assertTrue(self.events["exit_early"], "the go is kept while it cannot be judged yet")
        self.assertTrue(self.go(110.55))
        self.assertFalse(self.events["exit_early"])
        self.assertFalse(self.go(110.6))
        self.assertEqual((self.guard.refused, self.guard.unheard, self.said), (0, 0, []))

    def test_a_go_in_the_middle_of_typing_is_refused_cleared_and_said(self):
        for index, char in enumerate("I need to go"):
            self.press("space" if char == " " else char, 110.0 + 0.15 * index)
        self.assertFalse(self.go(112.0))
        self.assertFalse(self.events["exit_early"], "a refused go must not fire later")
        self.assertEqual(self.guard.refused, 1)
        self.assertEqual(len(self.said), 1)
        self.assertIn("middle of typing", self.said[0])
        self.assertFalse(self.go(130.0))
        # the operator then presses n by itself: that one counts
        self.press("n", 140.0)
        self.assertTrue(self.go(140.6))

    def test_every_n_of_a_long_message_is_refused_however_often_the_session_asks(self):
        text = "how to stop recording? I need to go. Actually I think it is exiting already"
        started = 0
        for index, char in enumerate(text):
            when = 110.0 + 0.15 * index
            self.press("space" if char == " " else char, when)
            for frame in range(3):                                  # the placement asks at every frame
                started += self.go(when + 0.04 * (frame + 1))
        for frame in range(100):
            started += self.go(110.0 + 0.15 * len(text) + 0.05 * frame)
        self.assertEqual(started, 0)
        self.assertGreater(self.guard.refused, 0)

    def test_a_stop_and_a_re_record_are_handled_as_before(self):
        self.press("q", 110.0)
        self.assertFalse(self.go(110.1))
        self.assertTrue(self.events["stop_recording"], "a stop is left for the session to act on")
        self.events.update(stop_recording=False)
        self.press("r", 120.0)
        self.assertFalse(self.go(120.1))
        self.assertFalse(self.events["rerecord_episode"])
        self.assertEqual(self.guard.refused, 0)

    def test_a_go_the_guard_never_heard_counts_as_it_did_without_a_guard_after_a_short_wait(self):
        self.events["exit_early"] = True                           # set by another way than the listener
        self.assertFalse(self.go(110.0))
        self.assertFalse(self.go(111.0))
        self.assertTrue(self.go(111.6))
        self.assertEqual((self.guard.unheard, self.guard.refused), (1, 0))

    def test_an_unheard_go_is_timed_from_its_own_beginning(self):
        self.events["exit_early"] = True
        self.assertFalse(self.go(110.0))
        self.events["exit_early"] = False                          # the session cleared it (a new placement)
        self.assertFalse(self.go(120.0))
        self.events["exit_early"] = True
        self.assertFalse(self.go(130.0), "the wait starts again")
        self.assertTrue(self.go(131.6))

    def test_without_a_guard_it_is_consume_go(self):
        self.events["exit_early"] = True
        self.assertTrue(guarded_go(self.events, None))
        self.assertFalse(guarded_go(self.events, None))

    def test_a_typed_n_heard_between_the_look_and_the_stamp_never_counts(self):
        for when, key in ((110.00, "r"), (110.10, "u"), (110.20, "n")):
            self.press(key, when)
        self.events["rerecord_episode"] = False                    # the r's own flags were dealt with earlier
        stamp = self.guard.judged

        def stamp_with_a_key_arriving() -> None:
            self.guard.judged = stamp
            self.clock.now = 110.3200
            self.guard.saw("n")                                     # the second n of "running" is noted ...
            self.clock.now = 110.3201
            stamp()                                                 # ... before the session stamps its judgement

        self.guard.judged = stamp_with_a_key_arriving
        self.assertFalse(self.go(110.3199))
        self.events["exit_early"] = True                           # ... and the listener sets the second n's flag
        for when in (110.36, 110.9, 111.5, 111.9, 113.0):
            self.assertFalse(self.go(when), f"a typed n counted at {when}")
        self.assertEqual(self.guard.unheard, 0)
        self.assertEqual(self.guard.refused, 2)


class ForgetGoTests(unittest.TestCase):
    """A go pressed while the cube is not steady is dropped at once, and cannot count half a second later."""

    def setUp(self):
        self.clock = Clock()
        self.guard = TypingGuard(clock=self.clock)
        self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_a_go_is_dropped_and_does_not_come_back(self):
        self.assertFalse(typing_guard.forget_go(self.events, self.guard))          # nothing to drop
        self.clock.now = 110.0
        self.guard.saw("n")
        self.events["exit_early"] = True
        self.clock.now = 110.1
        self.assertTrue(typing_guard.forget_go(self.events, self.guard))
        self.assertFalse(self.events["exit_early"])
        for when in (110.2, 110.6, 112.0):                           # the cube steady from here on
            self.clock.now = when
            self.assertFalse(guarded_go(self.events, self.guard))
        self.assertEqual((self.guard.refused, self.guard.unheard), (0, 0))
        # ... and the next one, pressed with the cube steady, counts
        self.clock.now = 120.0
        self.guard.saw("n")
        self.events["exit_early"] = True
        self.clock.now = 120.6
        self.assertTrue(guarded_go(self.events, self.guard))

    def test_a_stop_is_left_for_the_session_and_a_re_record_press_is_forgotten(self):
        self.events.update(stop_recording=True, exit_early=True)
        self.assertFalse(typing_guard.forget_go(self.events, self.guard))
        self.assertTrue(self.events["stop_recording"])
        self.events.update(stop_recording=False, rerecord_episode=True, exit_early=True)
        self.assertFalse(typing_guard.forget_go(self.events, self.guard))
        self.assertEqual(self.events, {"exit_early": False, "rerecord_episode": False, "stop_recording": False})
        self.events["exit_early"] = True
        self.assertTrue(typing_guard.forget_go(self.events, None))                 # without a guard: `consume_go`


class PhaseGoTests(unittest.TestCase):
    """The go that ends a phase after which the policy starts (the reposition after a refused handoff)."""

    def setUp(self):
        self.clock = Clock()
        self.guard = TypingGuard(clock=self.clock)
        self.said: list[str] = []
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def sleep(self, seconds: float) -> None:
        self.clock.now += seconds

    def test_a_go_pressed_by_itself_passes_after_its_half_second(self):
        typed(self.guard, self.clock, "n", 110.0)
        self.clock.now = 110.02
        self.assertTrue(typing_guard.phase_go(self.guard, self.said.append, sleep=self.sleep))
        self.assertGreaterEqual(self.clock.now, 110.5)
        self.assertLess(self.clock.now, 110.7)
        self.assertEqual((self.said, self.guard.refused), ([], 0))
        self.assertEqual(self.guard.judge(), UNHEARD, "the go is behind the session once it was asked about")

    def test_a_go_in_a_sentence_does_not_pass_and_is_said(self):
        end = typed(self.guard, self.clock, "I need to go", 110.0)
        self.clock.now = end + 0.01
        self.assertFalse(typing_guard.phase_go(self.guard, self.said.append, sleep=self.sleep))
        self.assertEqual(self.guard.refused, 1)
        self.assertIn("middle of typing", self.said[0])

    def test_a_key_right_after_the_go_refuses_it(self):
        typed(self.guard, self.clock, "n", 110.0)

        def sleep_and_type(seconds: float) -> None:
            self.clock.now += seconds
            if not self.said and self.clock.now > 110.2 and len(self.guard._keys) == 1:
                self.guard.saw("o")

        self.clock.now = 110.01
        self.assertFalse(typing_guard.phase_go(self.guard, self.said.append, sleep=sleep_and_type))

    def test_a_go_nobody_heard_and_no_guard_pass_as_before(self):
        self.assertTrue(typing_guard.phase_go(self.guard, self.said.append, sleep=self.sleep))
        self.assertTrue(typing_guard.phase_go(None))
        self.assertEqual(self.said, [])

    def test_a_go_key_held_down_does_not_wait_for_ever(self):
        typed(self.guard, self.clock, "n", 110.0)

        def held(seconds: float) -> None:
            self.clock.now += seconds
            self.guard.saw("n")                                     # the key repeats for as long as it is held

        self.clock.now = 110.01
        self.assertFalse(typing_guard.phase_go(self.guard, self.said.append, sleep=held))
        self.assertLess(self.clock.now, 110.0 + typing_guard.PHASE_GO_WAIT_S + 1.0)


class KeyboardTests(unittest.TestCase):
    """The trial runner's keyboard: LeRobot's keys from one listener, paused without starting another."""

    KEYS = ("x", "up", "down", "enter", "space", "n", "N", "right", "r", "R", "left", "q", "Q", "esc")

    def setUp(self):
        try:
            from lerobot.utils import keyboard_input
        except ImportError:
            self.skipTest("needs LeRobot (run with the rig's interpreter)")
        self.keyboard_input = keyboard_input
        self.started: list = []
        self.stopped = 0

        class Listener:
            def stop(inner) -> None:  # noqa: N805
                self.stopped += 1

        real_create = keyboard_input.create_key_listener
        keyboard_input.create_key_listener = lambda dispatch, **_kwargs: self.started.append(dispatch) or Listener()
        self.addCleanup(setattr, keyboard_input, "create_key_listener", real_create)

    def press(self, dispatch, key) -> None:
        with contextlib.redirect_stdout(io.StringIO()):         # LeRobot prints what each control key does
            dispatch(key)

    def test_the_same_keys_set_the_same_flags_as_lerobot_s_listener(self):
        _listener, theirs = self.keyboard_input.init_keyboard_listener()
        guard = TypingGuard()
        keyboard = typing_guard.Keyboard(guard)
        their_key, our_key = self.started
        self.assertEqual(keyboard.events, theirs)
        for key in self.KEYS:
            for flags in (theirs, keyboard.events):
                flags.update(exit_early=False, rerecord_episode=False, stop_recording=False)
            self.press(their_key, key)
            self.press(our_key, key)
            self.assertEqual(keyboard.events, theirs, key)
        self.assertEqual(len(guard._keys), len(self.KEYS), "and the guard heard every one of them")

    def test_while_paused_no_key_reaches_the_flags_or_the_guard_and_no_listener_is_started(self):
        guard = TypingGuard()
        keyboard = typing_guard.Keyboard(guard)
        (key,) = self.started
        keyboard.pause()
        self.assertTrue(keyboard.paused)
        for char in "grasp:corner, ran into the basket, quit trying":       # a note full of n, r and q
            self.press(key, "space" if char == " " else char)
        self.press(key, "esc")
        self.assertEqual(keyboard.events, {"exit_early": False, "rerecord_episode": False, "stop_recording": False})
        self.assertEqual(len(guard._keys), 0)
        keyboard.resume()
        self.assertFalse(keyboard.paused)
        self.press(key, "n")
        self.assertTrue(keyboard.events["exit_early"])
        self.assertEqual((len(self.started), self.stopped), (1, 0), "one listener, from the session's start on")

    def test_a_stop_pressed_before_the_pause_is_still_set_after_it(self):
        keyboard = typing_guard.Keyboard()
        (key,) = self.started
        self.press(key, "q")
        keyboard.pause()
        self.press(key, "5")
        keyboard.resume()
        self.assertTrue(keyboard.events["stop_recording"])

    def test_stopping_ends_the_listener_once(self):
        keyboard = typing_guard.Keyboard()
        keyboard.stop()
        keyboard.stop()
        self.assertEqual(self.stopped, 1)

    def test_without_a_guard_the_keys_work_the_same(self):
        keyboard = typing_guard.Keyboard(None)
        (key,) = self.started
        self.press(key, "r")
        self.assertEqual(keyboard.events, {"exit_early": True, "rerecord_episode": True, "stop_recording": False})


if __name__ == "__main__":
    unittest.main()
