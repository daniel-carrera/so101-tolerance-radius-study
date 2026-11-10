"""The recording loop's rate, read from a session's log (`tolerance_radius.recording_rate`).

The logs here are made up in the form the arm tooling writes them: one line for the take's start, three or four lines
per control cycle, the clock's "5 seconds", and the reset's first line after the take.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from tolerance_radius import recording_rate

START = datetime(2026, 10, 6, 17, 54, 54, 241000)


def stamp(seconds: float) -> str:
    moment = START + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%d %H:%M:%S") + f",{moment.microsecond // 1000:03d}"


def take(episode: int, at: float, cycles: int, hz: float, *, ends: str = "Reset the environment -- leader already "
         "aligned (worst wrist_flex +0.0 deg)", level: str = "INFO", inside: tuple[str, ...] = ()) -> list[str]:
    """The log of one take: `cycles` control cycles at `hz`, then the line that ends it."""
    lines = [f"{stamp(at - 0.003)} INFO Episode {episode} -- leader already aligned (worst wrist_flex +0.0 deg)",
             f"{stamp(at)} INFO Recording episode {episode}"]
    for cycle in range(cycles):
        now = at + 0.037 + cycle / hz
        lines += [f"{stamp(now)} DEBUG my_follower SOFollower read state: 1.1ms",
                  f"{stamp(now + 0.001)} DEBUG my_follower SOFollower read front: 0.0ms",
                  f"{stamp(now + 0.001)} DEBUG my_follower SOFollower read wrist: 0.0ms",
                  f"{stamp(now + 0.002)} DEBUG my_leader SOLeader read action: 1.1ms"]
        if cycle == cycles // 2:
            lines += [f"{stamp(now + 0.003)} {line}" for line in inside]
    end = at + 0.037 + (cycles - 1) / hz + 0.035
    lines.append(f"{stamp(end)} {level} {ends}")
    # the reset that follows is teleoperation too: the same cycle lines, and none of them the take's
    lines += [f"{stamp(end + 0.03 * n)} DEBUG my_follower SOFollower read state: 1.0ms" for n in range(1, 40)]
    return lines


class TakesInALogTests(unittest.TestCase):
    def test_a_take_s_cycles_and_its_rate_are_read_between_its_first_line_and_the_reset(self):
        found = recording_rate.takes_in(take(0, 0.0, 973, 30.0))
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].episode, found[0].cycles, found[0].began), (0, 973, "2026-10-06 17:54:54"))
        self.assertAlmostEqual(found[0].seconds, 972 / 30.0, delta=0.002)
        self.assertAlmostEqual(found[0].hz, 30.0, delta=0.005)

    def test_a_loop_that_ran_slow_is_seen_although_its_frames_would_be_stamped_at_the_rate_asked_for(self):
        # August: 16 of 55 episodes at about 28 Hz, about 700 frames in 25 s, every frame stamped at 30.
        found = recording_rate.takes_in(take(26, 0.0, 700, 28.0))
        self.assertAlmostEqual(found[0].hz, 28.0, delta=0.005)

    def test_the_clock_s_announcement_and_a_warning_from_the_loop_belong_to_the_take(self):
        lines = take(0, 0.0, 600, 30.0, inside=("INFO 5 seconds",
                                                "WARNING Control loop is running slower (27.9 Hz) than the target"))
        found = recording_rate.takes_in(lines)
        self.assertEqual([(t.episode, t.cycles) for t in found], [(0, 600)])

    def test_the_cycles_of_the_placement_before_and_of_the_reset_after_are_not_the_take_s(self):
        before = [f"{stamp(-5.0 + 0.05 * n)} DEBUG my_follower SOFollower read state: 1.1ms" for n in range(60)]
        found = recording_rate.takes_in(before + take(3, 0.0, 300, 30.0))
        self.assertEqual(found[0].cycles, 300)

    def test_takes_recorded_again_and_a_take_cut_by_the_stop_key_are_all_there_in_order(self):
        lines = (take(4, 0.0, 300, 30.0) + take(4, 60.0, 750, 30.0)
                 + take(5, 120.0, 90, 30.0, ends="episode 5: stopped during the take; it is NOT saved",
                        level="WARNING"))
        found = recording_rate.takes_in(lines)
        self.assertEqual([(t.episode, t.cycles) for t in found], [(4, 300), (4, 750), (5, 90)])

    def test_a_take_the_log_ends_in_and_lines_that_are_not_log_lines_do_no_harm(self):
        lines = take(7, 0.0, 120, 30.0)[:-40] + ["Traceback (most recent call last):", "", "  File ..."]
        found = recording_rate.takes_in(lines)
        self.assertEqual([(t.episode, t.cycles) for t in found], [(7, 120)])
        self.assertEqual(recording_rate.takes_in(["nothing of a log", ""]), [])

    def test_a_take_of_one_cycle_has_no_rate(self):
        found = recording_rate.takes_in(take(8, 0.0, 1, 30.0))
        self.assertEqual(found[0].cycles, 1)
        self.assertIsNone(found[0].hz)


class SavedTakesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.logs = Path(directory.name)
        # Session A: episode 0, episode 1 recorded twice (the second kept). Session B, after a stop: episode 2, slow.
        (self.logs / "session_A.log").write_text(
            "\n".join(take(0, 0.0, 973, 30.0) + take(1, 60.0, 200, 30.0) + take(1, 120.0, 790, 30.0)) + "\n")
        (self.logs / "session_B.log").write_text("\n".join(take(2, 0.0, 700, 28.0)) + "\n")
        self.placements = [{"episode_index": 1, "session": "A"}, {"episode_index": 0, "session": "A"},
                           {"episode_index": 2, "session": "B"}, {"episode_index": 3, "session": "C"}]

    def test_each_saved_episode_gets_the_last_take_of_it_in_the_session_that_saved_it(self):
        rows = recording_rate.saved_takes(self.placements, self.logs)
        self.assertEqual([(row["episode_index"], row["session"], row["cycles"], row["takes_in_session"])
                          for row in rows], [(0, "A", 973, 1), (1, "A", 790, 2), (2, "B", 700, 1), (3, "C", None, 0)])
        self.assertAlmostEqual(rows[2]["hz"], 28.0, delta=0.005)
        self.assertIsNone(rows[3]["hz"])                       # its session's log is not there: said, not passed over

    def test_the_summary_names_what_fell_short_what_was_not_read_and_what_is_not_the_dataset_s(self):
        rows = recording_rate.saved_takes(self.placements, self.logs)
        told = recording_rate.summary(rows, 30.0, lengths={0: 973, 1: 791, 2: 700})
        self.assertEqual((told["episodes"], told["with_a_rate"]), (4, 3))
        self.assertEqual(told["fell_short"], [2])
        self.assertEqual(told["not_read"], [3])
        self.assertEqual(told["frames_differ"], [1])            # 790 cycles in the log, 791 frames in the dataset
        self.assertAlmostEqual(told["median_hz"], 30.0, delta=0.005)
        self.assertAlmostEqual(told["lowest_hz"], 28.0, delta=0.005)
        self.assertAlmostEqual(told["short_of"], 29.4)
        self.assertNotIn("frames_differ", recording_rate.summary(rows, 30.0))

    def test_a_rate_two_per_cent_under_is_the_line(self):
        for hz, short in ((29.5, False), (29.3, True)):
            with self.subTest(hz=hz):
                (self.logs / "session_D.log").write_text("\n".join(take(0, 0.0, 800, hz)) + "\n")
                rows = recording_rate.saved_takes([{"episode_index": 0, "session": "D"}], self.logs)
                self.assertEqual(recording_rate.summary(rows, 30.0)["fell_short"], [0] if short else [])

    def test_a_placement_log_is_read_line_by_line_and_a_missing_one_is_empty(self):
        path = self.logs / "clustered_placements.jsonl"
        path.write_text(json.dumps({"episode_index": 0, "session": "A"}) + "\n\n"
                        + json.dumps({"episode_index": 1, "session": "A"}) + "\n")
        self.assertEqual([entry["episode_index"] for entry in recording_rate.read_placements(path)], [0, 1])
        self.assertEqual(recording_rate.read_placements(self.logs / "none.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
