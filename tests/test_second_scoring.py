"""The second scoring from video: which trials are drawn, what the scorer is shown, and how the two scores compare.

Every trial here is invented, and so is every video: a file of a few bytes, a few small frames written to a temporary
folder where OpenCV is installed, or, in one test, what the real recorder writes on the rig made of fakes. No trial
data of the repository is read and nothing is written to it; its runbook, its page of data contracts and its protocol
file are read where a test checks what they claim. No window is opened and no camera.

The draws made here use a seed of their own and invented logs. None is the draw of the experiment: that one depends
on the bytes of the real log, which do not exist before the last trial.
"""

import contextlib
import importlib.util
import io
import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # the development environment has no OpenCV
    cv2 = None

import fakes_rig

from tolerance_radius import schema, second_scoring
from tolerance_radius.config import load_protocol
from tolerance_radius.intervals import wilson_interval
from tolerance_radius.schema import PlannedTrial, Point, TrialRecord
from tolerance_radius.second_scoring import SecondScoringError
from tolerance_radius.seeds import derived_rng, shuffled

NEEDS_OPENCV = "needs OpenCV (run with the rig's interpreter)"
REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("second_scoring_command", REPO / "scripts" / "second_scoring.py")
command = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(command)

SESSION = "20261007_090000"
#: The experiment as planned on 2026-10-05: 34 + 22 + 2 x 26 + 24 scored trials.
PLANNED = {("T1a", "act_clustered"): 34, ("T1b", "replay"): 22, ("T2", "act_clustered"): 26,
           ("T2", "act_spread"): 26, ("T3", "act_spread"): 24}
#: The seed these tests draw with, and what stands for the digest of the logs where no log is read.
SEED = 4242
SALT = "the digest of an invented log"
MAIN = "data/trials/main.jsonl"


def video_name(trial_id: str, view: str, session: str = SESSION) -> str:
    """A trial's video as the trial runner names it: the session's time, the trial's own name, the camera."""
    return f"data/recordings/{session}/{session}_{trial_id}_{view}.mp4"


def trial(trial_id: str, block: str, condition: str, outcome: str = "success", *, pair_id: str | None = None,
          status: str = "frozen", views: tuple[str, ...] = ("front", "wrist"), scorer: str = "op1",
          steps_sent: int | None = 300, number: int = 0) -> TrialRecord:
    """One invented trial with the fields the trial runner writes."""
    replay = condition == "replay"
    planned = PlannedTrial(trial_id=trial_id, block=block, condition=condition, blind_code=f"K{number % 100:02d}",
                           target=Point(75.0, 25.0), pair_id=pair_id, replay_episode=3 if replay else None)
    common = dict(planned=planned, ts_start="2026-10-07T09:00:00-07:00", ts_end="2026-10-07T09:00:30-07:00",
                  operator="op1", scorer=scorer, policy_repo=None if replay else condition,
                  checkpoint=None if replay else "act-run-v1@080000",
                  execution={} if steps_sent is None else {"steps_sent": steps_sent},
                  versions={"protocol_version": "1.0", "protocol_status": status},
                  media={view: video_name(trial_id, view) for view in views})
    if outcome == "void":
        return TrialRecord(outcome="void", void_reason="bumped the table", **common)
    success = outcome == "success"
    return TrialRecord(outcome=outcome, subgoal=5 if success else 2, failure=None if success else "grasp",
                       placed=Point(76.0, 25.5), nearest_demo_episode=3, nearest_demo_distance_mm=17.25, **common)


def experiment(sizes: dict[tuple[str, str], int] = PLANNED, *, voids: int = 2) -> list[TrialRecord]:
    """A whole invented experiment: each block's scored trials in the order run, with some void trials among them.

    Outcomes alternate, so that successes and failures are both there. The two trials of a workspace pair share a
    `pair_id`.
    """
    records: list[TrialRecord] = []
    for block in schema.BLOCKS:
        conditions = [condition for (name, condition) in sizes if name == block]
        count = sum(sizes[(block, condition)] for condition in conditions)
        number = 0
        for index in range(count):
            condition = conditions[index % len(conditions)]
            pair_id = f"{block}-p{index // 2 + 1:02d}" if len(conditions) == 2 else None
            if index < voids:  # a void trial, and then the same place asked again under the next number
                number += 1
                records.append(trial(f"{block}-{number:03d}", block, condition, "void", pair_id=pair_id, number=number))
            number += 1
            outcome = "success" if (index // len(conditions)) % 2 == 0 else "failure"
            records.append(trial(f"{block}-{number:03d}", block, condition, outcome, pair_id=pair_id, number=number))
    return records


def candidates_of(records: list[TrialRecord], log: str = MAIN) -> list[second_scoring.Candidate]:
    return [second_scoring.Candidate(log, record) for record in records if record.scored]


def write_log(path: Path, records: list[TrialRecord]) -> Path:
    for record in records:
        schema.append_trial(path, record)
    return path


def drawn_ids(draw: second_scoring.Draw) -> list[str]:
    return [item.candidate.trial_id for item in draw.items]


def blocks_of(draw: second_scoring.Draw) -> list[str]:
    return [item.candidate.block for item in draw.items]


def planned_draw(log: Path, root: Path, seed: int = SEED) -> second_scoring.Draw:
    """The draw `make_draw` makes from a log when every video can be used: the seed, and the log's own digest."""
    population = second_scoring.read_population([log], root)
    return second_scoring.draw_items(population.candidates, seed, population.digest)


def read_key(folder: Path) -> list[tuple[str, str, str, str]]:
    """The key of a draw: for each item its code, its trial, the trial's log and the digest of its copies."""
    lines = (folder / "key" / "key.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "item,trial_id,log,videos_sha256", lines[0]
    return [tuple(line.split(",")) for line in lines[1:]]


class QuotaTests(unittest.TestCase):
    def test_a_fifth_of_a_group_rounded_up(self):
        self.assertEqual([second_scoring.quota(n) for n in (34, 22, 26, 24, 20)], [7, 5, 6, 5, 4])

    def test_a_group_with_any_trial_gives_at_least_one(self):
        self.assertEqual([second_scoring.quota(n) for n in (0, 1, 4, 5, 6)], [0, 1, 1, 1, 2])

    def test_a_count_that_is_not_a_whole_number_of_trials_is_refused(self):
        for bad in (-1, 2.5, True):
            with self.subTest(n=bad), self.assertRaises(ValueError):
                second_scoring.quota(bad)


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.trials = self.root / "data" / "trials"

    def test_the_scored_trials_of_the_counted_logs_and_no_void_one(self):
        records = experiment()
        log = write_log(self.trials / "main.jsonl", records)
        population = second_scoring.read_population([log], self.root)
        self.assertEqual([c.trial_id for c in population.candidates],
                         [r.planned.trial_id for r in records if r.scored])
        self.assertEqual(len(population.candidates), 132)
        self.assertEqual({c.log for c in population.candidates}, {MAIN})
        (facts,) = population.logs
        self.assertEqual((facts.file, facts.n_records, facts.n_scored), (MAIN, len(records), 132))
        self.assertRegex(facts.sha256, r"^[0-9a-f]{64}$")

    def test_several_logs_are_read_in_name_order_whatever_order_they_are_given_in(self):
        first = write_log(self.trials / "a.jsonl", [trial("T1a-001", "T1a", "act_clustered")])
        second = write_log(self.trials / "b.jsonl", [trial("T1b-001", "T1b", "replay")])
        population = second_scoring.read_population([second, first], self.root)
        self.assertEqual([c.trial_id for c in population.candidates], ["T1a-001", "T1b-001"])
        self.assertEqual([facts.file for facts in population.logs], ["data/trials/a.jsonl", "data/trials/b.jsonl"])
        self.assertEqual(population.digest, second_scoring.read_population([first, second], self.root).digest)

    def test_the_digest_of_the_logs_changes_with_every_line_a_log_gains(self):
        log = write_log(self.trials / "main.jsonl", [trial("T1a-001", "T1a", "act_clustered")])
        digests = [second_scoring.read_population([log], self.root).digest]
        self.assertRegex(digests[0], r"^[0-9a-f]{64}$")
        self.assertEqual(digests[0], second_scoring.read_population([log], self.root).digest)
        schema.append_trial(log, trial("T1a-002", "T1a", "act_clustered", "void"))
        digests.append(second_scoring.read_population([log], self.root).digest)
        schema.append_trial(log, trial("T1a-003", "T1a", "act_clustered", "failure"))
        digests.append(second_scoring.read_population([log], self.root).digest)
        other = write_log(self.trials / "other.jsonl", [trial("T1b-001", "T1b", "replay")])
        digests.append(second_scoring.read_population([log, other], self.root).digest)
        self.assertEqual(len(set(digests)), 4)

    def test_a_pilot_log_is_refused(self):
        log = write_log(self.trials / "pilot_policy.jsonl", [trial("T1a-001", "T1a", "act_clustered", status="pilot")])
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.read_population([log], self.root)
        self.assertIn("pilot", str(refusal.exception))
        self.assertIn("--rehearsal", str(refusal.exception))

    def test_a_trial_that_was_not_run_under_a_frozen_protocol_is_refused(self):
        log = write_log(self.trials / "main.jsonl", [trial("T1a-001", "T1a", "act_clustered"),
                                                    trial("T1a-002", "T1a", "act_clustered", status="draft")])
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.read_population([log], self.root)
        self.assertIn("frozen", str(refusal.exception))
        self.assertIn("T1a-002", str(refusal.exception))

    def test_a_rehearsal_takes_pilot_trials(self):
        log = write_log(self.trials / "pilot_policy.jsonl",
                        [trial(f"T1a-{n:03d}", "T1a", "act_clustered", status="pilot") for n in range(1, 10)])
        population = second_scoring.read_population([log], self.root, rehearsal=True)
        self.assertEqual(len(population.candidates), 9)

    def test_a_rehearsal_takes_no_counted_trial(self):
        log = write_log(self.trials / "main.jsonl", [trial("T1a-001", "T1a", "act_clustered", status="draft"),
                                                    trial("T1a-002", "T1a", "act_clustered")])
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.read_population([log], self.root, rehearsal=True)
        self.assertIn("holds counted trials", str(refusal.exception))
        draft = write_log(self.trials / "draft.jsonl", [trial("T1a-003", "T1a", "act_clustered", status="draft")])
        self.assertEqual(len(second_scoring.read_population([draft], self.root, rehearsal=True).candidates), 1)

    def test_a_trial_name_that_two_logs_share_is_refused(self):
        first = write_log(self.trials / "a.jsonl", [trial("T1a-001", "T1a", "act_clustered")])
        second = write_log(self.trials / "b.jsonl", [trial("T1a-001", "T1a", "act_clustered")])
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.read_population([first, second], self.root)
        self.assertIn("T1a-001", str(refusal.exception))
        self.assertIn("twice", str(refusal.exception))

    def test_a_log_that_is_not_there_or_holds_no_scored_trial_is_refused(self):
        with self.assertRaises(SecondScoringError) as missing:
            second_scoring.read_population([self.trials / "main.jsonl"], self.root)
        self.assertIn("does not exist", str(missing.exception))
        log = write_log(self.trials / "main.jsonl", [trial("T1a-001", "T1a", "act_clustered", "void")])
        with self.assertRaises(SecondScoringError) as empty:
            second_scoring.read_population([log], self.root)
        self.assertIn("no scored trial", str(empty.exception))
        with self.assertRaises(SecondScoringError):
            second_scoring.read_population([], self.root)

    def test_a_log_that_cannot_be_read_is_refused_without_naming_the_machine(self):
        self.trials.mkdir(parents=True)
        log = self.trials / "main.jsonl"
        log.write_text('{"not": "a trial"}\n', encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.read_population([log], self.root)
        self.assertIn("data/trials/main.jsonl:1", str(refusal.exception))
        self.assertNotIn(str(self.root), str(refusal.exception))
        self.assertNotIn(str(self.root.resolve()), str(refusal.exception))


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.candidates = candidates_of(experiment())

    def draw(self, seed: int = SEED, salt: str = SALT, **options) -> second_scoring.Draw:
        return second_scoring.draw_items(self.candidates, seed, salt, **options)

    def test_every_group_gives_a_fifth_of_its_scored_trials_rounded_up(self):
        draw = self.draw()
        self.assertEqual({(g.block, g.condition): (g.n_scored, g.n_drawn) for g in draw.groups},
                         {("T1a", "act_clustered"): (34, 7), ("T1b", "replay"): (22, 5),
                          ("T2", "act_clustered"): (26, 6), ("T2", "act_spread"): (26, 6),
                          ("T3", "act_spread"): (24, 5)})
        self.assertEqual((draw.n_scored, len(draw.items)), (132, 29))
        drawn = {}
        for item in draw.items:
            key = (item.candidate.block, item.candidate.condition)
            drawn[key] = drawn.get(key, 0) + 1
        self.assertEqual(drawn, {key: second_scoring.quota(n) for key, n in PLANNED.items()})
        self.assertEqual((draw.seed, draw.salt, draw.one_in, draw.whole_pairs), (SEED, SALT, 5, ()))

    def test_the_groups_are_listed_in_the_protocol_s_order(self):
        self.assertEqual([(g.block, g.condition) for g in self.draw().groups],
                         [("T1a", "act_clustered"), ("T1b", "replay"), ("T2", "act_clustered"),
                          ("T2", "act_spread"), ("T3", "act_spread")])

    def test_the_same_trials_seed_and_salt_give_the_same_draw_and_another_seed_another(self):
        first = self.draw()
        again = second_scoring.draw_items(list(reversed(self.candidates)), SEED, SALT)
        other = self.draw(seed=SEED + 1)
        self.assertEqual(first, again)
        self.assertNotEqual(drawn_ids(first), drawn_ids(other))
        self.assertNotEqual(sorted(drawn_ids(first)), sorted(drawn_ids(other)))

    def test_which_trials_are_drawn_cannot_be_told_from_the_seed_before_the_log_is_whole(self):
        """The seed is public and fixed in advance. With it alone, the places of a group that get a second score
        would be known before the first trial, and those trials could be scored with more care at the rig."""
        places = set()
        group = sorted(c.trial_id for c in self.candidates if c.block == "T1a")
        for number in range(8):
            draw = self.draw(salt=f"the digest of log {number}")
            places.add(tuple(sorted(group.index(trial_id) for trial_id in drawn_ids(draw) if trial_id in group)))
        self.assertGreater(len(places), 6, "the same places of a group are drawn whatever the log holds")

    def test_an_item_s_place_does_not_give_its_block_to_someone_who_has_the_seed(self):
        """With the seed and the number of items alone, the order shown would be one known shuffle of the drawn
        trials in name order, and an item's place among those says which block it is from."""
        sizes = [0, 7, 12, 24, 29]  # where each block's drawn trials end, in name order
        guessed = shuffled(list(range(29)), derived_rng(SEED, second_scoring.STREAM, "order"))
        guess = [schema.BLOCKS[sum(place >= end for end in sizes[1:])] for place in guessed]
        right = []
        for number in range(8):
            draw = self.draw(salt=f"the digest of log {number}")
            right.append(sum(a == b for a, b in zip(guess, blocks_of(draw))))
        self.assertLess(max(right), 24, "the blocks of the items follow from the seed and their number")
        self.assertGreater(len({tuple(blocks_of(self.draw(salt=str(n)))) for n in range(8)}), 6)

    def test_no_trial_is_drawn_twice_and_every_one_is_a_scored_trial_of_the_log(self):
        ids = drawn_ids(self.draw())
        self.assertEqual(len(set(ids)), len(ids))
        self.assertLessEqual(set(ids), {c.trial_id for c in self.candidates})

    def test_the_items_have_neutral_codes_in_the_order_shown(self):
        self.assertEqual([item.code for item in self.draw().items], [f"item-{n:02d}" for n in range(1, 30)])

    def test_the_order_shown_is_not_the_order_run_and_does_not_keep_a_group_together(self):
        draw = self.draw()
        ids = drawn_ids(draw)
        self.assertNotEqual(ids, sorted(ids))
        blocks = blocks_of(draw)
        changes = sum(1 for before, after in zip(blocks, blocks[1:]) if before != after)
        self.assertGreater(changes, 4, "the blocks come one after the other, as they would unshuffled")

    def test_the_two_trials_of_a_pair_are_never_both_drawn(self):
        for number in range(40):
            with self.subTest(log=number):
                draw = self.draw(salt=str(number))
                pairs = [item.candidate.pair_id for item in draw.items if item.candidate.pair_id is not None]
                self.assertEqual(len(pairs), 12)
                self.assertEqual(len(set(pairs)), len(pairs))
                self.assertEqual(draw.whole_pairs, ())

    def test_every_trial_of_a_group_has_the_same_chance_whichever_policy_of_a_pair_is_drawn_first(self):
        logs = 3000
        times = {c.trial_id: 0 for c in self.candidates}
        for number in range(logs):
            for trial_id in drawn_ids(self.draw(salt=str(number))):
                times[trial_id] += 1
        for candidate in self.candidates:
            size = PLANNED[(candidate.block, candidate.condition)]
            share = second_scoring.quota(size) / size
            spread = (logs * share * (1 - share)) ** 0.5
            with self.subTest(trial=candidate.trial_id):
                # 4.5 standard deviations: about 0.17 to 0.24 where 0.21 is expected, 0.20 to 0.27 around 0.23.
                self.assertLess(abs(times[candidate.trial_id] - logs * share), 4.5 * spread)

    def test_a_pair_s_second_trial_is_taken_when_its_group_has_nothing_else_and_the_draw_says_so(self):
        only_pair = [trial("T2-001", "T2", "act_clustered", pair_id="T2-p01"),
                     trial("T2-002", "T2", "act_spread", pair_id="T2-p01")]
        draw = second_scoring.draw_items(candidates_of(only_pair), 3, SALT)
        self.assertEqual(sorted(drawn_ids(draw)), ["T2-001", "T2-002"])
        self.assertEqual(draw.whole_pairs, ("T2-p01",))

    def test_a_trial_whose_video_cannot_be_used_gives_its_place_to_the_next_of_the_same_order(self):
        plain = self.draw()
        lost = next(item.candidate for item in plain.items if item.candidate.block == "T1b")
        draw = self.draw(check=lambda c: "its front video is not there" if c.trial_id == lost.trial_id else None)
        self.assertEqual([(p.trial_id, p.log, p.reason) for p in draw.passed_over],
                         [(lost.trial_id, lost.log, "its front video is not there")])
        self.assertEqual(len(draw.items), len(plain.items))
        before, after = set(drawn_ids(plain)), set(drawn_ids(draw))
        self.assertEqual(before - after, {lost.trial_id})
        (replacement,) = after - before
        self.assertTrue(replacement.startswith("T1b-"), "the place went to a trial of another group")
        self.assertEqual(plain.passed_over, ())

    def test_only_the_trials_that_come_up_are_checked(self):
        asked = []

        def check(candidate):
            asked.append(candidate.trial_id)
            return None

        draw = self.draw(check=check)
        self.assertEqual(sorted(asked), sorted(drawn_ids(draw)))

    def test_a_group_that_cannot_be_filled_stops_the_draw_and_names_every_trial_without_a_video(self):
        def check(candidate):
            return "its front video is not there" if candidate.block == "T1b" else None

        with self.assertRaises(SecondScoringError) as refusal:
            self.draw(check=check)
        text = str(refusal.exception)
        self.assertIn("T1b", text)
        self.assertIn("5 of its 22", text)
        self.assertIn("its front video is not there", text)
        self.assertIn("22 of them", text)

    def test_no_trial_to_draw_from_and_a_trial_listed_twice_are_refused(self):
        with self.assertRaises(SecondScoringError):
            second_scoring.draw_items([], 1, SALT)
        twice = candidates_of([trial("T1a-001", "T1a", "act_clustered")]) * 2
        with self.assertRaises(SecondScoringError):
            second_scoring.draw_items(twice, 1, SALT)

    def test_the_salt_is_text(self):
        for bad in (None, 7, b"digest"):
            with self.subTest(salt=bad), self.assertRaises(TypeError):
                second_scoring.draw_items(self.candidates, SEED, bad)

    def test_one_trial_in_another_number_than_five_can_be_asked_for(self):
        draw = self.draw(one_in=2)
        self.assertEqual([g.n_drawn for g in draw.groups], [17, 11, 13, 13, 12])
        self.assertEqual(draw.one_in, 2)


# --------------------------------------------------------------------------------------------------------------------
# Videos that are a few bytes of text, and a probe that reads them
# --------------------------------------------------------------------------------------------------------------------


def fake_video(root: Path, record: TrialRecord, view: str, frames: int | str = 300) -> Path:
    """Write a stand-in for one of a trial's videos: text that says how many frames it has, and whose it is."""
    path = root / record.media[view]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"frames={frames} {record.planned.trial_id} {view}\n", encoding="utf-8")
    return path


def fake_videos(root: Path, records: list[TrialRecord], frames: int = 300) -> None:
    for record in records:
        for view in record.media:
            fake_video(root, record, view, frames)


def fake_probe(path: Path) -> int:
    """Count the frames of a stand-in video, as `probe_video` counts those of a real one."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("frames="):
        raise second_scoring.VideoError("does not open as a video")
    count = text.split()[0].removeprefix("frames=")
    if not count.isdigit():
        raise second_scoring.VideoError("does not open as a video")
    return int(count)


def files_below(folder: Path) -> dict[str, bytes]:
    """Every file below a folder, by its name relative to the folder, with its bytes."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()
    }


def identity_words(records: list[TrialRecord]) -> set[str]:
    """What must never reach the scorer: every trial's name, code and pair, the blocks, the conditions, the logs."""
    words = set(schema.BLOCKS) | set(schema.CONDITIONS) | {"clustered", "spread", "replay", "policy", "pilot",
                                                           "main.jsonl", "recordings", SESSION}
    for record in records:
        words |= {record.planned.trial_id, record.planned.blind_code}
        if record.planned.pair_id:
            words.add(record.planned.pair_id)
    return words


class VideoCheckTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.record = trial("T1a-017", "T1a", "act_clustered", steps_sent=300)
        self.candidate = second_scoring.Candidate(MAIN, self.record)
        self.check = second_scoring.VideoCheck(self.root, probe=fake_probe)

    def test_a_trial_whose_two_videos_decode_can_be_used(self):
        fake_videos(self.root, [self.record])
        self.assertIsNone(self.check(self.candidate))
        self.assertEqual(self.check.sources(self.candidate),
                         {view: self.root / video_name("T1a-017", view) for view in ("front", "wrist")})

    def test_a_video_the_record_does_not_name_or_that_is_not_there(self):
        fake_videos(self.root, [self.record])
        (self.root / video_name("T1a-017", "wrist")).unlink()
        self.assertEqual(self.check(self.candidate), "its wrist video is not there")
        front_only = second_scoring.Candidate(MAIN, trial("T1a-018", "T1a", "act_clustered", views=("front",)))
        fake_videos(self.root, [front_only.record])
        self.assertEqual(self.check(front_only), "its record names no wrist video")

    def test_a_file_that_is_not_a_video_or_holds_no_picture(self):
        fake_videos(self.root, [self.record])
        fake_video(self.root, self.record, "front", "torn")
        self.assertEqual(self.check(self.candidate), "its front video does not open as a video")
        fake_video(self.root, self.record, "front", 0)
        self.assertEqual(self.check(self.candidate), "its front video holds no picture")

    def test_a_video_named_by_an_absolute_path_or_outside_the_repository_is_not_used(self):
        elsewhere = Path(self.enterContext(tempfile.TemporaryDirectory())) / "front.mp4"
        elsewhere.write_text("frames=300\n", encoding="utf-8")
        for name, reason in ((str(elsewhere), "its record names its front video by an absolute path"),
                             ("data/../../front.mp4", "its record names its front video outside the repository")):
            with self.subTest(name=name):
                record = trial("T1a-019", "T1a", "act_clustered")
                record.media["front"] = name
                self.assertEqual(self.check(second_scoring.Candidate(MAIN, record)), reason)

    def test_a_folder_of_recordings_that_is_a_link_to_another_disk_is_used(self):
        elsewhere = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "data").mkdir()
        (self.root / "data" / "recordings").symlink_to(elsewhere, target_is_directory=True)
        fake_videos(self.root, [self.record])
        self.assertTrue((elsewhere / SESSION).is_dir())
        self.assertIsNone(self.check(self.candidate))

    def test_only_the_views_asked_for_are_looked_at(self):
        fake_video(self.root, self.record, "front")
        front = second_scoring.VideoCheck(self.root, views=("front",), probe=fake_probe)
        self.assertIsNone(front(self.candidate))
        self.assertEqual(list(front.sources(self.candidate)), ["front"])

    def test_a_video_with_fewer_frames_than_the_trial_sent_steps_is_not_used(self):
        """It has lost frames, and nothing says where. Cut at its end it would show a cube still in the gripper."""
        fake_video(self.root, self.record, "front", 300)
        fake_video(self.root, self.record, "wrist", 212)
        self.assertEqual(self.check(self.candidate), "its wrist video holds 212 frames and the trial sent 300 steps")
        fake_video(self.root, self.record, "wrist", 299)
        self.assertEqual(self.check(self.candidate), "its wrist video holds 299 frames and the trial sent 300 steps")

    def test_a_video_with_a_frame_for_every_step_or_more_is_used(self):
        fake_video(self.root, self.record, "front", 300)
        fake_video(self.root, self.record, "wrist", 302)  # each start the handoff gate refused adds a frame
        self.assertIsNone(self.check(self.candidate))

    def test_a_record_that_does_not_say_how_many_steps_were_sent_is_not_held_to_it(self):
        record = trial("T1a-020", "T1a", "act_clustered", steps_sent=None)
        fake_videos(self.root, [record], frames=5)
        self.assertIsNone(self.check(second_scoring.Candidate(MAIN, record)))

    def test_views_are_plain_names_each_given_once(self):
        for views in ((), ("front", "front"), ("item",), ("../front",), ("Front",)):
            with self.subTest(views=views), self.assertRaises(SecondScoringError):
                second_scoring.VideoCheck(self.root, views=views, probe=fake_probe)


class MakeDrawTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.records = experiment()
        self.log = write_log(self.root / "data" / "trials" / "main.jsonl", self.records)
        fake_videos(self.root, self.records)
        self.out = self.root / "data" / "second_scoring" / "main"

    def make(self, out: Path | None = None, **options):
        options.setdefault("probe", fake_probe)
        return second_scoring.make_draw([self.log], SEED, out or self.out, root=self.root, **options)

    def key(self, out: Path | None = None) -> dict[str, str]:
        return {code: trial_id for code, trial_id, _, _ in read_key(out or self.out)}

    def test_it_writes_the_items_the_copies_and_the_key_kept_apart(self):
        record = self.make()
        written = files_below(self.out)
        codes = [f"item-{n:02d}" for n in range(1, 30)]
        expected = {"items.csv", "key/key.csv", "key/draw.json"}
        expected |= {f"videos/{code}_{view}.mp4" for code in codes for view in ("front", "wrist")}
        self.assertEqual(set(written), expected)
        self.assertEqual(record, json.loads(written["key/draw.json"]))

    def test_what_the_scorer_s_tool_reads_holds_only_codes_and_neutral_names(self):
        self.make()
        lines = (self.out / "items.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "item,front,wrist")
        for number, line in enumerate(lines[1:], start=1):
            code = f"item-{number:02d}"
            self.assertEqual(line, f"{code},videos/{code}_front.mp4,videos/{code}_wrist.mp4")
        self.assertEqual(len(lines), 30)
        text = "\n".join(lines)
        for word in identity_words(self.records):
            self.assertNotIn(word, text)

    def test_each_copy_is_the_video_of_the_trial_the_key_names_and_the_key_holds_its_digest(self):
        self.make()
        key = read_key(self.out)
        self.assertEqual(len(key), 29)
        for code, trial_id, log, digest in key:
            copies = [self.out / "videos" / f"{code}_{view}.mp4" for view in ("front", "wrist")]
            with self.subTest(item=code):
                self.assertEqual([copy.read_bytes() for copy in copies],
                                 [(self.root / video_name(trial_id, view)).read_bytes() for view in ("front", "wrist")])
                self.assertEqual(digest, second_scoring.videos_digest(copies))
                self.assertEqual(log, MAIN)
        self.assertEqual(len({digest for _, _, _, digest in key}), 29)

    def test_the_key_is_what_the_seed_draws_from_the_log_s_own_bytes(self):
        self.make()
        draw = planned_draw(self.log, self.root)
        self.assertEqual(self.key(), {item.code: item.candidate.trial_id for item in draw.items})
        unsalted = second_scoring.draw_items(candidates_of(self.records), SEED, "")
        self.assertNotEqual(self.key(), {item.code: item.candidate.trial_id for item in unsalted.items})

    def test_one_more_line_in_the_log_gives_another_draw(self):
        """So nobody can work the draw out before the last trial is logged."""
        self.make()
        other_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        longer = write_log(other_root / "data" / "trials" / "main.jsonl",
                           [*self.records, trial("T3-090", "T3", "act_spread", "void", number=90)])
        fake_videos(other_root, self.records)
        other = other_root / "data" / "second_scoring" / "main"
        second_scoring.make_draw([longer], SEED, other, root=other_root, probe=fake_probe)
        self.assertNotEqual(self.key(), self.key(other))
        self.assertNotEqual(sorted(self.key().values()), sorted(self.key(other).values()))

    def test_the_record_of_the_draw_says_what_it_was_made_from(self):
        record = self.make()
        self.assertEqual(set(record), {"schema_version", "rehearsal", "seed", "one_in", "views", "logs", "logs_digest",
                                       "n_scored", "n_drawn", "groups", "passed_over", "whole_pairs"})
        self.assertEqual((record["schema_version"], record["rehearsal"], record["seed"], record["one_in"]),
                         (1, False, SEED, 5))
        self.assertEqual(record["views"], ["front", "wrist"])
        self.assertEqual((record["n_scored"], record["n_drawn"]), (132, 29))
        (log,) = record["logs"]
        self.assertEqual((log["file"], log["n_records"], log["n_scored"]), (MAIN, len(self.records), 132))
        self.assertRegex(log["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(record["logs_digest"], second_scoring.read_population([self.log], self.root).digest)
        self.assertEqual(record["groups"][0], {"block": "T1a", "condition": "act_clustered", "n_scored": 34,
                                               "n_drawn": 7})
        self.assertEqual((record["passed_over"], record["whole_pairs"]), ([], []))

    def test_nothing_written_names_the_machine(self):
        self.make()
        for name, content in files_below(self.out).items():
            if name.startswith("videos/"):
                continue
            with self.subTest(file=name):
                text = content.decode("utf-8")
                self.assertNotIn(str(self.root), text)
                self.assertNotIn(str(self.root.resolve()), text)
                self.assertNotIn("\r", text)

    def test_the_same_log_and_seed_write_the_same_bytes(self):
        self.make()
        other = self.root / "data" / "second_scoring" / "again"
        self.make(other)
        self.assertEqual(files_below(self.out), files_below(other))

    def test_the_copies_are_made_in_the_order_shown_so_that_their_dates_say_nothing(self):
        made = []

        def copy(source, destination):
            made.append(Path(destination).name)
            Path(destination).write_bytes(Path(source).read_bytes())

        self.make(copy=copy)
        self.assertEqual(made, [f"item-{n:02d}_{view}.mp4" for n in range(1, 30) for view in ("front", "wrist")])

    def test_a_copy_that_fails_ends_the_draw_without_naming_the_trial_it_was_copying(self):
        """The system's own message puts the trial's file beside the item's. It must not reach the terminal."""
        def full_disk(source, destination):
            raise OSError(28, "No space left on device", str(source), None, str(destination))

        with self.assertRaises(SecondScoringError) as refusal:
            self.make(copy=full_disk)
        text = str(refusal.exception)
        self.assertIn("No space left on device", text)
        self.assertIn("data/second_scoring/main", text)
        self.assertIsNone(refusal.exception.__cause__)
        self.assertTrue(refusal.exception.__suppress_context__)
        for word in identity_words(self.records) - {"main.jsonl"}:
            self.assertNotIn(word, text)
        self.assertFalse((self.out / "items.csv").exists())

    def test_a_drawn_trial_without_a_usable_video_is_replaced_and_listed(self):
        lost = planned_draw(self.log, self.root).items[0].candidate
        (self.root / video_name(lost.trial_id, "wrist")).unlink()
        record = self.make()
        self.assertEqual(record["passed_over"], [{"trial_id": lost.trial_id, "log": MAIN,
                                                  "reason": "its wrist video is not there"}])
        self.assertEqual(record["n_drawn"], 29)
        self.assertNotIn(lost.trial_id, self.key().values())

    def test_a_drawn_trial_whose_video_lost_frames_is_replaced_and_listed(self):
        short = planned_draw(self.log, self.root).items[3].candidate
        fake_video(self.root, short.record, "front", 120)
        record = self.make()
        self.assertEqual(record["passed_over"], [{
            "trial_id": short.trial_id, "log": MAIN,
            "reason": "its front video holds 120 frames and the trial sent 300 steps"}])
        self.assertNotIn(short.trial_id, self.key().values())

    def test_one_view_alone_can_be_asked_for(self):
        record = self.make(views=("front",))
        self.assertEqual(record["views"], ["front"])
        lines = (self.out / "items.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[:2], ["item,front", "item-01,videos/item-01_front.mp4"])
        self.assertEqual(sum(name.startswith("videos/") for name in files_below(self.out)), 29)

    def test_a_folder_that_already_has_scores_is_refused_and_left_as_it_was(self):
        self.make()
        (self.out / "scores.csv").write_text("item,score,scorer,watched_again,videos_sha256\n", encoding="utf-8")
        before = files_below(self.out)
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.make_draw([self.log], SEED + 1, self.out, root=self.root, probe=fake_probe)
        self.assertIn("already holds scores", str(refusal.exception))
        self.assertIn("data/second_scoring/main", str(refusal.exception))
        self.assertNotIn(str(self.root), str(refusal.exception))
        self.assertEqual(files_below(self.out), before)

    def test_a_folder_that_already_holds_a_draw_or_anything_else_is_refused(self):
        self.make()
        before = files_below(self.out)
        with self.assertRaises(SecondScoringError) as refusal:
            self.make()
        self.assertIn("already holds a draw", str(refusal.exception))
        self.assertEqual(files_below(self.out), before)
        other = self.root / "data" / "second_scoring" / "other"
        other.mkdir(parents=True)
        (other / "notes.txt").write_text("something", encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            self.make(other)
        self.assertIn("is not empty", str(refusal.exception))

    def test_a_folder_left_by_a_draw_that_was_cut_short_is_refused_too(self):
        (self.out / "videos").mkdir(parents=True)
        (self.out / "videos" / "item-01_front.mp4").write_text("frames=300\n", encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            self.make()
        self.assertIn("is not empty", str(refusal.exception))
        self.assertIn("cut short", str(refusal.exception))

    def test_an_empty_folder_that_exists_is_fine(self):
        self.out.mkdir(parents=True)
        self.make()
        self.assertTrue((self.out / "items.csv").is_file())

    def test_a_draw_that_is_refused_writes_nothing(self):
        for record in self.records:
            if record.planned.block == "T1b":
                (self.root / video_name(record.planned.trial_id, "front")).unlink()
        with self.assertRaises(SecondScoringError):
            self.make()
        self.assertFalse(self.out.exists())

    def test_a_rehearsal_on_pilot_trials_is_marked_as_one(self):
        pilots = [trial(f"T1a-{n:03d}", "T1a", "act_clustered", status="pilot") for n in range(1, 10)]
        log = write_log(self.root / "data" / "trials" / "pilot_policy.jsonl", pilots)
        fake_videos(self.root, pilots)
        out = self.root / "data" / "second_scoring" / "rehearsal"
        with self.assertRaises(SecondScoringError):
            second_scoring.make_draw([log], 1, out, root=self.root, probe=fake_probe)
        record = second_scoring.make_draw([log], 1, out, root=self.root, probe=fake_probe, rehearsal=True)
        self.assertEqual((record["rehearsal"], record["n_scored"], record["n_drawn"]), (True, 9, 2))
        self.assertTrue(second_scoring.draw_lines(record, "data/second_scoring/rehearsal")[0].startswith("REHEARSAL"))

    def test_what_the_draw_prints_counts_trials_and_names_none(self):
        (self.root / video_name(planned_draw(self.log, self.root).items[0].candidate.trial_id, "wrist")).unlink()
        record = self.make()
        text = "\n".join(second_scoring.draw_lines(record, "data/second_scoring/main"))
        self.assertIn("29 of the 132 scored trials", text)
        self.assertIn(f"seed {SEED}", text)
        self.assertIn("T1a  act_clustered  7 of 34", text)
        self.assertIn("1 trial came up and was passed over", text)
        self.assertNotIn("every drawn video", text)
        self.assertIn("scripts/second_scoring.py score --items data/second_scoring/main/items.csv", text)
        for record_ in self.records:
            self.assertNotIn(record_.planned.trial_id, text)

    def test_what_the_draw_prints_when_nothing_was_amiss(self):
        text = "\n".join(second_scoring.draw_lines(self.make(), "data/second_scoring/main"))
        self.assertIn("every drawn video could be read to its end", text)
        self.assertNotIn("passed over", text)
        self.assertNotIn("REHEARSAL", text)
        self.assertNotIn("both trials", text)

    def test_what_the_draw_prints_when_it_had_to_take_both_trials_of_a_pair(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        pair = [trial("T2-001", "T2", "act_clustered", pair_id="T2-p01"),
                trial("T2-002", "T2", "act_spread", pair_id="T2-p01")]
        log = write_log(root / "data" / "trials" / "main.jsonl", pair)
        fake_videos(root, pair)
        record = second_scoring.make_draw([log], SEED, root / "out", root=root, probe=fake_probe)
        self.assertEqual(record["whole_pairs"], ["T2-p01"])
        text = "\n".join(second_scoring.draw_lines(record, "out"))
        self.assertIn("both trials of 1 pair were drawn", text)
        self.assertNotIn("T2-p01", text)


# --------------------------------------------------------------------------------------------------------------------
# The scoring, with a player and a screen made of fakes
# --------------------------------------------------------------------------------------------------------------------

#: How many pictures a stand-in video has, and the keys of a scorer who lets one run to its end and touches nothing.
FRAMES = 3
WATCH = [None] * FRAMES


def scored(key: str) -> list:
    """The keys of a scorer who starts a video, watches it to its end, chooses, and records the choice."""
    return [" ", *WATCH, key, "enter"]


class FakePlayer:
    """Hands out an item's pictures as `(code, index)` pairs and remembers which items it was asked for."""

    def __init__(self, frames: int = FRAMES, empty: tuple[str, ...] = ()) -> None:
        self._frames, self._empty = frames, empty
        self.asked: list[str] = []

    def frame_ms(self, item) -> int:
        return 33

    def frames(self, item):
        self.asked.append(item.code)
        return iter([] if item.code in self._empty else [(item.code, index) for index in range(self._frames)])


class FakeScreen:
    """Shows nothing. Each wait for a key is answered from a script, and a script that runs out is an error.

    A script entry is None (no key), a letter or a space, "enter", "esc", a key's code as the window system gives
    it, or a function called at that moment whose result is the entry: a test uses one to look at the files while
    the scoring is going on.
    """

    CODES = {"enter": 13, "esc": 27}

    def __init__(self, keys: list) -> None:
        self.keys = list(keys)
        self.shown: list[tuple[object, tuple[str, ...]]] = []
        self.waits: list[int] = []
        self.closed = 0

    def show(self, picture, lines) -> None:
        self.shown.append((picture, tuple(lines)))

    def wait_key(self, ms: int) -> int:
        self.waits.append(ms)
        if not self.keys:
            raise AssertionError("the scorer's script ran out while the tool was still waiting for a key")
        key = self.keys.pop(0)
        if callable(key):
            key = key()
        if key is None:
            return -1
        if isinstance(key, int):
            return key
        return self.CODES[key] if key in self.CODES else ord(key)

    def close(self) -> None:
        self.closed += 1


class ScoringTests(unittest.TestCase):
    HEADING = "item,score,scorer,watched_again,videos_sha256"

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.records = experiment()
        log = write_log(self.root / "data" / "trials" / "main.jsonl", self.records)
        fake_videos(self.root, self.records)
        self.out = self.root / "data" / "second_scoring" / "main"
        second_scoring.make_draw([log], SEED, self.out, root=self.root, probe=fake_probe)
        self.items = self.out / "items.csv"
        self.scores = self.out / "scores.csv"
        self.said: list[str] = []

    def sit(self, keys: list, *, scorer: str = "sc2", player: FakePlayer | None = None):
        """One sitting: the scorer's keys in order. Returns what the tool returned, the player and the screen."""
        player, screen = player or FakePlayer(), FakeScreen(keys)
        result = second_scoring.score_items(self.items, scorer, player=player, screen=screen, say=self.said.append)
        return result, player, screen

    def scored_so_far(self) -> list[str]:
        """The scores in the file, each without the digest of its videos."""
        lines = self.scores.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], self.HEADING)
        return [",".join(line.split(",")[:4]) for line in lines[1:]]

    def test_each_item_is_started_shown_to_its_end_and_scored_by_a_choice_and_enter(self):
        keys = []
        for number in range(29):
            keys += scored("sfc"[number % 3])
        result, player, screen = self.sit(keys)
        self.assertEqual(result, (29, 0))
        self.assertEqual(player.asked, [f"item-{n:02d}" for n in range(1, 30)])
        words = ("success", "failure", "cannot_tell")
        self.assertEqual(self.scored_so_far(), [f"item-{n + 1:02d},{words[n % 3]},sc2,0" for n in range(29)])
        self.assertEqual((screen.keys, screen.closed), ([], 1))

    def test_an_item_waits_on_its_first_picture_until_the_space_bar(self):
        """So that no video runs while the scorer is still reading, or looking for the window."""
        _, player, screen = self.sit([None, "s", "enter", "r", None, " ", *WATCH, "x"])
        waiting = screen.shown[:6]
        self.assertEqual({picture for picture, _ in waiting}, {("item-01", 0)})
        for _, lines in waiting:
            self.assertIn("space: start the video", " ".join(lines))
        self.assertEqual([picture for picture, _ in screen.shown[6:]],
                         [("item-01", 0), ("item-01", 1), ("item-01", 2), ("item-01", 2)])
        self.assertEqual(player.asked, ["item-01"])
        self.assertFalse(self.scores.exists())

    def test_every_picture_of_a_video_is_shown_in_order_and_the_last_one_stays(self):
        _, _, screen = self.sit([*scored("s"), "x"])
        first = [picture for picture, _ in screen.shown if picture[0] == "item-01"]
        self.assertEqual(first, [("item-01", 0), ("item-01", 0), ("item-01", 1), ("item-01", 2), ("item-01", 2),
                                 ("item-01", 2)])

    def test_a_score_is_in_the_file_before_the_next_item_is_shown(self):
        seen = []

        def look():
            seen.append(self.scored_so_far())
            return None

        self.sit([*scored("f"), look, "x"])
        self.assertEqual(seen, [["item-01,failure,sc2,0"]])

    def test_while_the_video_runs_only_the_stop_key_does_anything(self):
        _, player, screen = self.sit([" ", "s", "enter", "r", "c", "enter", "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,cannot_tell,sc2,0"])
        self.assertEqual(player.asked, ["item-01", "item-02"])
        for _, lines in screen.shown[1:1 + FRAMES]:
            self.assertIn("watch to the end", " ".join(lines))
            self.assertNotIn("success", " ".join(lines))
        self.sit([" ", None, "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,cannot_tell,sc2,0"])

    def test_enter_alone_records_nothing_and_a_choice_can_be_changed_before_enter(self):
        self.sit([" ", *WATCH, "enter", None, "s", "f", "enter", "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,failure,sc2,0"])

    def test_the_choice_is_shown_before_it_is_recorded(self):
        _, _, screen = self.sit([" ", *WATCH, "c", "x"])
        self.assertFalse(self.scores.exists())
        self.assertIn("CANNOT TELL", " ".join(screen.shown[-1][1]))
        self.assertIn("Enter", " ".join(screen.shown[-1][1]))

    def test_watching_again_shows_the_whole_video_again_and_is_counted(self):
        _, player, screen = self.sit([" ", *WATCH, "r", *WATCH, "r", *WATCH, "s", "enter", "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,success,sc2,2"])
        self.assertEqual(player.asked, ["item-01", "item-01", "item-01", "item-02"])
        again = [picture for picture, _ in screen.shown[1 + FRAMES + 1:1 + FRAMES + 1 + FRAMES]]
        self.assertEqual(again, [("item-01", 0), ("item-01", 1), ("item-01", 2)])

    def test_a_key_held_down_counts_one_more_showing_and_not_one_for_each_repeat(self):
        self.sit([" ", *WATCH, "r", "r", "r", "r", "s", "enter", "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,success,sc2,1"])

    def test_a_choice_made_before_watching_again_is_forgotten(self):
        self.sit([" ", *WATCH, "s", "r", *WATCH, "enter", None, "f", "enter", "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,failure,sc2,1"])

    def test_stopping_writes_nothing_for_the_item_in_hand_and_the_next_sitting_carries_on_there(self):
        result, _, screen = self.sit([*scored("s"), None, "x"])
        self.assertEqual((result, screen.closed), ((1, 28), 1))
        self.assertEqual(self.scored_so_far(), ["item-01,success,sc2,0"])
        result, player, _ = self.sit([*scored("f"), " ", None, "x"])  # stopped while the third video runs
        self.assertEqual(result, (1, 27))
        self.assertEqual(player.asked, ["item-02", "item-03"])
        result, player, _ = self.sit([" ", *WATCH, "s", "esc"])  # stopped with a choice not yet recorded
        self.assertEqual((result, player.asked), ((0, 27), ["item-03"]))
        self.assertEqual(self.scored_so_far(), ["item-01,success,sc2,0", "item-02,failure,sc2,0"])
        keys = []
        for _ in range(27):
            keys += scored("c")
        result, player, _ = self.sit(keys, scorer="op1")
        self.assertEqual(result, (27, 0))
        self.assertEqual(player.asked[0], "item-03")
        self.assertEqual(self.scored_so_far()[-1], "item-29,cannot_tell,op1,0")
        self.assertEqual(len(self.scored_so_far()), 29)

    def test_when_every_item_is_scored_nothing_more_is_shown(self):
        keys = []
        for _ in range(29):
            keys += scored("s")
        self.sit(keys)
        before = self.scores.read_bytes()
        self.said.clear()
        result, player, screen = self.sit([])
        self.assertEqual((result, player.asked, screen.shown), ((0, 0), [], []))
        self.assertEqual(self.scores.read_bytes(), before)
        self.assertIn("all 29 items are scored", " ".join(self.said))

    def test_capital_letters_are_the_same_keys(self):
        self.sit([" ", *WATCH, "F", "enter", " ", *WATCH, "S", "enter", "X"])
        self.assertEqual(self.scored_so_far(), ["item-01,failure,sc2,0", "item-02,success,sc2,0"])

    def test_an_arrow_or_a_function_key_records_nothing(self):
        right_arrow, f7, f10 = 0xF703, 0xF70A, 0xF70D  # as a Mac's window reports them
        self.assertEqual((right_arrow & 0xFF, f7 & 0xFF, f10 & 0xFF), (3, 10, 13),
                         "cut down to eight bits these three are the keypad's Enter, a line feed and Return")
        self.sit([" ", *WATCH, "s", right_arrow, f7, f10, 0xF708, 255, 0, "x"])
        self.assertFalse(self.scores.exists())

    def test_only_the_return_key_records_a_choice(self):
        keypad_enter_or_control_c, line_feed = 3, 10
        self.sit([" ", *WATCH, "s", keypad_enter_or_control_c, line_feed, " ", None, "x"])
        self.assertFalse(self.scores.exists())
        self.sit([" ", *WATCH, "s", 13, "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,success,sc2,0"])

    def test_a_key_that_comes_with_shift_or_a_lock_is_still_its_key(self):
        num_lock, shift, caps = 0x10 << 16, 0x01 << 16, 0x02 << 16  # as GTK reports what was held with a key
        self.sit([num_lock | ord(" "), *WATCH, num_lock | ord("f"), num_lock | 13,
                  " ", *WATCH, shift | ord("S"), caps | shift | 13, "x"])
        self.assertEqual(self.scored_so_far(), ["item-01,failure,sc2,0", "item-02,success,sc2,0"])

    def test_a_key_pressed_with_control_alt_or_command_is_no_key(self):
        control, alt, command = 0x04 << 16, 0x08 << 16, 0x40 << 16
        self.sit([control | ord(" "), " ", *WATCH, control | ord("s"), alt | ord("f"), command | ord("c"),
                  control | 13, "s", control | 13, alt | 13, 0x250000, "x"])
        self.assertFalse(self.scores.exists())

    def test_the_pictures_follow_one_another_at_the_video_s_own_rate(self):
        _, _, screen = self.sit([" ", *WATCH, "x"])
        self.assertEqual(screen.waits, [second_scoring.HOLD_MS, 33, 33, 33, second_scoring.HOLD_MS])

    def test_each_score_carries_the_digest_of_the_files_that_were_shown_for_it(self):
        self.sit([*scored("s"), *scored("f"), "x"])
        lines = self.scores.read_text(encoding="utf-8").splitlines()[1:]
        for number, line in enumerate(lines, start=1):
            copies = [self.out / "videos" / f"item-{number:02d}_{view}.mp4" for view in ("front", "wrist")]
            self.assertEqual(line.split(",")[4], second_scoring.videos_digest(copies))
        self.assertEqual([line.split(",")[4] for line in lines], [digest for _, _, _, digest in read_key(self.out)[:2]])

    def test_nothing_the_scorer_sees_says_which_trial_an_item_is(self):
        keys = []
        for number in range(29):
            keys += [" ", *WATCH, "r", *WATCH, "sfc"[number % 3], "enter"]
        _, _, screen = self.sit(keys)
        seen = [line for _, lines in screen.shown for line in lines] + self.said
        seen += [second_scoring.WINDOW_TITLE, self.items.read_text(encoding="utf-8")]
        seen += [path.name for path in (self.out / "videos").iterdir()]
        self.assertGreater(len(seen), 200)
        text = "\n".join(seen)
        for word in identity_words(self.records):
            self.assertNotIn(word, text)
        self.assertNotIn(str(self.root), text)

    def test_the_scorer_is_told_the_rule_and_the_keys_in_the_first_scoring_s_words(self):
        self.sit([" ", "x"])
        text = "\n".join(self.said)
        self.assertIn("pick up the orange cube and place it in the basket", text)
        self.assertIn("released in the basket", text)
        self.assertIn("the fixed camera, the camera on the gripper", text)
        for key in ("space ", "s ", "f ", "c ", "r ", "x ", "Enter", "Return key"):
            self.assertIn(key, text)
        self.assertIn("29 items", text)

    def test_a_video_that_gives_no_picture_stops_the_sitting_and_scores_nothing_for_it(self):
        player, screen = FakePlayer(empty=("item-02",)), FakeScreen(scored("s"))
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.score_items(self.items, "sc2", player=player, screen=screen, say=self.said.append)
        self.assertIn("item-02", str(refusal.exception))
        self.assertEqual(self.scored_so_far(), ["item-01,success,sc2,0"])
        self.assertEqual(screen.closed, 1)

    def test_a_video_missing_from_the_folder_is_noticed_before_anything_is_shown(self):
        (self.out / "videos" / "item-05_wrist.mp4").unlink()
        player, screen = FakePlayer(), FakeScreen([])
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.score_items(self.items, "sc2", player=player, screen=screen, say=self.said.append)
        self.assertIn("item-05", str(refusal.exception))
        self.assertEqual((player.asked, screen.shown, self.scores.exists()), ([], [], False))

    def test_two_sittings_at_once_cannot_score_an_item_twice(self):
        def the_other_sitting():
            second_scoring.append_score(self.scores, second_scoring.Score("item-01", "failure", "sc2", 0, "0" * 64))
            return None

        with self.assertRaises(SecondScoringError) as refusal:
            self.sit([" ", *WATCH, the_other_sitting, "s", "enter"])
        self.assertIn("another sitting", str(refusal.exception))
        self.assertEqual(self.scored_so_far(), ["item-01,failure,sc2,0"])

    def test_the_scorer_is_named_by_a_role_code_never_by_a_name_or_by_initials(self):
        for bad in ("", "sc", "2", "Sc2", "someone", "some one", "sc2,op1", "sc2\n", "dan1", "bob2", "dc1", "jd07",
                    "sc123"):
            with self.subTest(scorer=bad):
                with self.assertRaises(SecondScoringError) as refusal:
                    self.sit([], scorer=bad)
                self.assertIn("role code", str(refusal.exception))
        self.assertFalse(self.scores.exists())
        for good in ("sc2", "op1", "sc10", "op2"):
            self.assertEqual(second_scoring.check_role_code(good), good)

    def test_scores_that_cannot_be_those_of_these_items_are_refused(self):
        digest = "0" * 64
        for body, why in ((f"item-99,success,sc2,0,{digest}\n", "item-99"),
                          (f"item-01,success,sc2,0,{digest}\nitem-01,failure,sc2,0,{digest}\n", "twice"),
                          (f"item-01,maybe,sc2,0,{digest}\n", "maybe"),
                          (f"item-01,success,someone,0,{digest}\n", "role code"),
                          (f"item-01,success,sc2,often,{digest}\n", "watched_again"),
                          ("item-01,success,sc2,0,abc\n", "videos_sha256"),
                          ("item-01,success,sc2,0\n", "line 2")):
            with self.subTest(body=body):
                self.scores.write_text(self.HEADING + "\n" + body, encoding="utf-8")
                with self.assertRaises(SecondScoringError) as refusal:
                    self.sit([])
                self.assertIn(why, str(refusal.exception))
        self.scores.write_text("item,score,scorer,watched_again\nitem-01,success,sc2,0\n", encoding="utf-8")
        with self.assertRaises(SecondScoringError):
            self.sit([])


class ItemsFileTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory())) / "main"
        self.folder.mkdir()
        self.path = self.folder / "items.csv"

    def write(self, text: str) -> Path:
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_an_item_is_a_code_and_a_video_for_each_view_inside_the_folder(self):
        items = second_scoring.read_items(self.write(
            "item,front,wrist\nitem-01,videos/item-01_front.mp4,videos/item-01_wrist.mp4\n"
            "item-02,videos/item-02_front.mp4,videos/item-02_wrist.mp4\n"))
        self.assertEqual([item.code for item in items], ["item-01", "item-02"])
        self.assertEqual(items[0].videos, (("front", self.folder / "videos" / "item-01_front.mp4"),
                                           ("wrist", self.folder / "videos" / "item-01_wrist.mp4")))

    def test_a_file_that_is_not_an_item_list_is_refused(self):
        for text, why in (("", "no items"),
                          ("item,front\n", "no items"),
                          ("code,front\nitem-01,videos/a.mp4\n", "first line"),
                          ("item\nitem-01\n", "first line"),
                          ("item,front\nitem-01,videos/a.mp4\nitem-01,videos/b.mp4\n", "twice"),
                          ("item,front,wrist\nitem-01,videos/a.mp4\n", "line 2"),
                          ("item,front\nitem-01,/somewhere/a.mp4\n", "inside the folder"),
                          ("item,front\nitem-01,../a.mp4\n", "inside the folder"),
                          ("item,front\n,videos/a.mp4\n", "line 2")):
            with self.subTest(text=text):
                with self.assertRaises(SecondScoringError) as refusal:
                    second_scoring.read_items(self.write(text))
                self.assertIn(why, str(refusal.exception))

    def test_a_missing_item_list_says_to_make_the_draw(self):
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.read_items(self.path)
        self.assertIn("main/items.csv", str(refusal.exception))
        self.assertIn("draw", str(refusal.exception))
        self.assertNotIn(str(self.folder.parent), str(refusal.exception))

    def test_scores_are_appended_one_line_each_under_one_heading(self):
        scores = self.folder / "scores.csv"
        self.assertEqual(second_scoring.read_scores(scores), [])
        first = second_scoring.Score("item-01", "success", "sc2", 0, "a" * 64)
        second = second_scoring.Score("item-02", "cannot_tell", "sc2", 3, "b" * 64)
        second_scoring.append_score(scores, first)
        second_scoring.append_score(scores, second)
        self.assertEqual(scores.read_text(encoding="utf-8"),
                         "item,score,scorer,watched_again,videos_sha256\n"
                         f"item-01,success,sc2,0,{'a' * 64}\nitem-02,cannot_tell,sc2,3,{'b' * 64}\n")
        self.assertEqual(second_scoring.read_scores(scores), [first, second])

    def test_the_digest_of_an_item_s_videos_depends_on_their_bytes_and_their_order(self):
        front, wrist = self.folder / "front.mp4", self.folder / "wrist.mp4"
        front.write_bytes(b"one")
        wrist.write_bytes(b"two")
        digest = second_scoring.videos_digest([front, wrist])
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, second_scoring.videos_digest([str(front), str(wrist)]))
        self.assertNotEqual(digest, second_scoring.videos_digest([wrist, front]))
        wrist.write_bytes(b"twO")
        self.assertNotEqual(digest, second_scoring.videos_digest([front, wrist]))


# --------------------------------------------------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------------------------------------------------


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.records = experiment()
        self.by_id = {record.planned.trial_id: record for record in self.records}
        self.log = write_log(self.root / "data" / "trials" / "main.jsonl", self.records)
        fake_videos(self.root, self.records)
        self.out = self.root / "data" / "second_scoring" / "main"

    def draw(self, **options) -> dict[str, str]:
        """Make the draw and return its key: which trial each item is."""
        second_scoring.make_draw([self.log], SEED, self.out, root=self.root, probe=fake_probe, **options)
        return self.key()

    def key(self) -> dict[str, str]:
        return {code: trial_id for code, trial_id, _, _ in read_key(self.out)}

    def score(self, key: dict[str, str], changed: dict[str, str] | None = None, *, scorer: str = "sc2",
              watched_again: dict[str, int] | None = None, only: int | None = None) -> None:
        """Give every item the first score of its trial, except the trials in `changed`, on the draw's own copies."""
        shown = {code: digest for code, _, _, digest in read_key(self.out)}
        for code, trial_id in list(key.items())[:only]:
            second = (changed or {}).get(trial_id, self.by_id[trial_id].outcome)
            second_scoring.append_score(self.out / "scores.csv", second_scoring.Score(
                code, second, scorer, (watched_again or {}).get(trial_id, 0), shown[code]))

    def compare(self, logs: list[Path] | None = None) -> dict:
        return second_scoring.compare(self.out, logs or [self.log], root=self.root)

    def drawn(self, key: dict[str, str], *, condition: str, outcome: str) -> str:
        """The first drawn trial, in name order, of a condition and a first score."""
        return next(trial_id for trial_id in sorted(key.values())
                    if self.by_id[trial_id].planned.condition == condition
                    and self.by_id[trial_id].outcome == outcome)

    def test_when_the_second_scores_are_the_first_ones_every_item_agrees(self):
        key = self.draw()
        self.score(key)
        agreement = self.compare()
        whole = agreement["all"]
        self.assertEqual((whole["n_items"], whole["n_agree"], whole["n_disagree"], whole["n_cannot_tell"],
                          whole["n_compared"]), (29, 29, 0, 0, 29))
        self.assertEqual(whole["agreement"], 1.0)
        low, high = wilson_interval(29, 29)
        self.assertAlmostEqual(whole["wilson_lo"], low, places=9)
        self.assertEqual(whole["wilson_hi"], high)
        self.assertEqual((agreement["disagreements"], agreement["cannot_tell"]), ([], []))
        first = [self.by_id[trial_id].outcome for trial_id in key.values()]
        self.assertEqual(whole["table"], {"success_success": first.count("success"), "success_failure": 0,
                                          "failure_success": 0, "failure_failure": first.count("failure")})
        self.assertEqual(agreement["interval_level"], 0.95)

    def test_a_disagreement_and_a_cannot_tell_are_counted_and_listed_by_trial(self):
        key = self.draw()
        differs = self.drawn(key, condition="act_clustered", outcome="failure")
        unclear = self.drawn(key, condition="replay", outcome="success")
        self.score(key, {differs: "success", unclear: "cannot_tell"}, watched_again={differs: 2})
        agreement = self.compare()
        whole = agreement["all"]
        self.assertEqual((whole["n_items"], whole["n_agree"], whole["n_disagree"], whole["n_cannot_tell"],
                          whole["n_compared"]), (29, 27, 1, 1, 28))
        self.assertAlmostEqual(whole["agreement"], 27 / 28, places=9)
        low, high = wilson_interval(27, 28)
        self.assertAlmostEqual(whole["wilson_lo"], low, places=9)
        self.assertAlmostEqual(whole["wilson_hi"], high, places=9)
        self.assertEqual((whole["table"]["failure_success"], whole["table"]["success_failure"]), (1, 0))
        self.assertEqual(whole["cannot_tell_first"], {"success": 1, "failure": 0})
        code = next(code for code, trial_id in key.items() if trial_id == differs)
        self.assertEqual(agreement["disagreements"], [{
            "trial_id": differs, "item": code, "block": self.by_id[differs].planned.block,
            "condition": "act_clustered", "kind": "policy", "first": "failure", "first_subgoal": 2,
            "first_failure": "grasp", "second": "success", "watched_again": 2}])
        (listed,) = agreement["cannot_tell"]
        self.assertEqual((listed["trial_id"], listed["kind"], listed["first"], listed["first_subgoal"],
                          listed["first_failure"], listed["second"]),
                         (unclear, "replay", "success", 5, None, "cannot_tell"))
        policy, replay = agreement["by_kind"]["policy"], agreement["by_kind"]["replay"]
        self.assertEqual((policy["n_items"], policy["n_agree"], policy["n_disagree"], policy["n_cannot_tell"]),
                         (24, 23, 1, 0))
        self.assertEqual((replay["n_items"], replay["n_agree"], replay["n_disagree"], replay["n_cannot_tell"]),
                         (5, 4, 0, 1))
        self.assertEqual(replay["wilson_lo"], round_9(wilson_interval(4, 4)[0]))
        self.assertEqual(agreement["watched_again"], {"n_items": 1, "total": 2})

    def test_the_two_kinds_add_up_to_the_whole(self):
        key = self.draw()
        changed = {trial_id: "failure" if self.by_id[trial_id].outcome == "success" else "success"
                   for trial_id in sorted(key.values())[::4]}
        self.score(key, changed)
        agreement = self.compare()
        for count in ("n_items", "n_agree", "n_disagree", "n_cannot_tell", "n_compared"):
            self.assertEqual(agreement["all"][count], sum(agreement["by_kind"][kind][count]
                                                           for kind in ("policy", "replay")))
        self.assertEqual(agreement["all"]["n_disagree"], len(changed))
        self.assertEqual([entry["trial_id"] for entry in agreement["disagreements"]], sorted(changed))
        self.assertEqual(sum(agreement["all"]["table"].values()), 29)

    def test_it_waits_until_every_item_is_scored_and_names_no_trial_meanwhile(self):
        key = self.draw()
        for scored_so_far in (0, 12):
            with self.subTest(scored=scored_so_far):
                self.score(key, only=scored_so_far)
                with self.assertRaises(SecondScoringError) as refusal:
                    self.compare()
                self.assertIn(f"{scored_so_far} of 29", str(refusal.exception))
                for trial_id in key.values():
                    self.assertNotIn(trial_id, str(refusal.exception))
        self.assertFalse((self.out / "agreement.json").exists())

    def test_a_key_that_is_not_what_the_seed_draws_from_the_logs_is_refused(self):
        key = self.draw()
        self.score(key)
        path = self.out / "key" / "key.csv"
        honest = path.read_text(encoding="utf-8")
        first, second = list(key.values())[:2]
        swapped = honest.replace(first, "?").replace(second, first).replace("?", second)
        path.write_text(swapped, encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            self.compare()
        self.assertIn(f"is not what seed {SEED} draws", str(refusal.exception))
        path.write_text(honest, encoding="utf-8")
        self.assertEqual(self.compare()["all"]["n_items"], 29)

    def test_a_trial_logged_after_the_draw_is_noticed_whether_it_was_scored_or_void(self):
        for outcome in ("failure", "void"):
            with self.subTest(outcome=outcome):
                root = Path(self.enterContext(tempfile.TemporaryDirectory()))
                log = write_log(root / "data" / "trials" / "main.jsonl", self.records)
                fake_videos(root, self.records)
                self.out = root / "data" / "second_scoring" / "main"
                second_scoring.make_draw([log], SEED, self.out, root=root, probe=fake_probe)
                self.score(self.key())
                schema.append_trial(log, trial("T3-090", "T3", "act_spread", outcome, number=90))
                with self.assertRaises(SecondScoringError) as refusal:
                    second_scoring.compare(self.out, [log], root=root)
                self.assertIn("data/trials/main.jsonl has changed since the draw", str(refusal.exception))

    def test_other_logs_than_the_draw_s_are_noticed(self):
        key = self.draw()
        self.score(key)
        other = write_log(self.root / "data" / "trials" / "other.jsonl", [
            trial(f"T1a-{n:03d}", "T1a", "act_clustered") for n in range(1, 11)])
        for logs in ([other], [self.log, other]):
            with self.subTest(logs=len(logs)), self.assertRaises(SecondScoringError) as refusal:
                self.compare(logs)
            self.assertIn("the draw was made from data/trials/main.jsonl", str(refusal.exception))

    def test_a_trial_passed_over_at_the_draw_is_passed_over_again_and_reported(self):
        lost = planned_draw(self.log, self.root).items[0].candidate.trial_id
        (self.root / video_name(lost, "wrist")).unlink()
        key = self.draw()
        self.assertNotIn(lost, key.values())
        self.score(key)
        fake_video(self.root, self.by_id[lost], "wrist")  # found again later: the draw stands as it was made
        agreement = self.compare()
        self.assertEqual(agreement["all"]["n_items"], 29)
        self.assertEqual(agreement["provenance"]["passed_over"], [
            {"trial_id": lost, "log": MAIN, "reason": "its wrist video is not there"}])
        self.assertIn(f"{lost} (its wrist video is not there)", "\n".join(second_scoring.agreement_lines(agreement)))

    def test_the_items_listed_must_be_the_key_s(self):
        key = self.draw()
        self.score(key)
        items = self.out / "items.csv"
        lines = items.read_text(encoding="utf-8").splitlines()
        items.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            self.compare()
        self.assertIn("items.csv", str(refusal.exception))

    def test_scores_that_belong_to_another_draw_are_refused(self):
        """A trial is logged after the draw, a new draw is made into a new folder, and the first draw's scores are
        put beside it. The codes are the same; the videos behind them are not."""
        key = self.draw()
        self.score(key)
        first_scores = (self.out / "scores.csv").read_text(encoding="utf-8")
        schema.append_trial(self.log, trial("T3-090", "T3", "act_spread", "void", number=90))
        first, self.out = self.out, self.root / "data" / "second_scoring" / "again"
        second_scoring.make_draw([self.log], SEED, self.out, root=self.root, probe=fake_probe)
        self.assertNotEqual(read_key(first), read_key(self.out))
        (self.out / "scores.csv").write_text(first_scores, encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            self.compare()
        self.assertIn("other video files", str(refusal.exception))
        self.assertNotIn("T3-", str(refusal.exception))

    def test_copies_that_were_swapped_in_the_folder_are_noticed(self):
        key = self.draw()
        videos = self.out / "videos"
        for view in ("front", "wrist"):
            one, two = videos / f"item-01_{view}.mp4", videos / f"item-02_{view}.mp4"
            kept = one.read_bytes()
            one.write_bytes(two.read_bytes())
            two.write_bytes(kept)
        keys = []
        for _ in key:
            keys += scored("s")
        second_scoring.score_items(self.out / "items.csv", "sc2", player=FakePlayer(), screen=FakeScreen(keys),
                                   say=lambda _line: None)
        with self.assertRaises(SecondScoringError) as refusal:
            self.compare()
        self.assertIn("the score of item-01, item-02 was given on other video files", str(refusal.exception))

    def test_a_record_of_the_draw_that_is_not_one_is_refused_in_one_sentence(self):
        key = self.draw()
        self.score(key)
        path = self.out / "key" / "draw.json"
        honest = json.loads(path.read_text(encoding="utf-8"))
        for change in ({"seed": str(SEED)}, {"seed": True}, {"one_in": 0}, {"logs": MAIN},
                       {"passed_over": ["T1a-001"]}, {"schema_version": 2}):
            with self.subTest(change=change):
                path.write_text(json.dumps({**honest, **change}), encoding="utf-8")
                with self.assertRaises(SecondScoringError) as refusal:
                    self.compare()
                self.assertIn("is not the record of a draw", str(refusal.exception))
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(SecondScoringError):
            self.compare()
        path.write_text(json.dumps({**honest, "seed": SEED + 1}), encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            self.compare()
        self.assertIn(f"is not what seed {SEED + 1} draws", str(refusal.exception))

    def test_what_the_record_says_of_the_draw_must_be_what_the_draw_was(self):
        """Everything the comparison reports from the record is first made again from the logs."""
        key = self.draw()
        self.score(key)
        path = self.out / "key" / "draw.json"
        honest = json.loads(path.read_text(encoding="utf-8"))
        never_met = next(c.trial_id for c in candidates_of(self.records) if c.trial_id not in key.values())
        invented = {"trial_id": never_met, "log": MAIN, "reason": "its front video is not there"}
        for change in ({"passed_over": [invented]}, {"n_scored": 999}, {"n_drawn": 1},
                       {"groups": honest["groups"][:-1]}, {"whole_pairs": ["T2-p01"]}, {"logs_digest": "0" * 64},
                       {"note": "a field the tool does not write"}):
            with self.subTest(change=list(change)):
                path.write_text(json.dumps({**honest, **change}), encoding="utf-8")
                with self.assertRaises(SecondScoringError) as refusal:
                    self.compare()
                self.assertIn(f"draw.json does not say what seed {SEED} draws", str(refusal.exception))
        path.write_text(json.dumps(honest), encoding="utf-8")
        self.assertEqual(self.compare()["provenance"]["n_scored"], 132)

    def test_a_folder_without_a_draw_is_refused(self):
        with self.assertRaises(SecondScoringError) as refusal:
            self.compare()
        self.assertIn("holds no draw", str(refusal.exception))
        self.assertNotIn(str(self.root), str(refusal.exception))

    def test_it_says_where_the_draw_came_from_and_who_scored(self):
        key = self.draw()
        self.score(key)
        agreement = self.compare()
        source = agreement["provenance"]
        self.assertEqual((source["folder"], source["logs"], source["seed"], source["one_in"], source["views"]),
                         ("data/second_scoring/main", [MAIN], SEED, 5, ["front", "wrist"]))
        self.assertEqual((source["n_scored"], source["n_drawn"], len(source["groups"])), (132, 29, 5))
        self.assertEqual((source["passed_over"], source["whole_pairs"]), ([], []))
        self.assertEqual(source["logs_digest"], second_scoring.read_population([self.log], self.root).digest)
        self.assertEqual(agreement["scorers"], {"first": ["op1"], "second": ["sc2"], "n_same_role": 0})
        self.assertEqual((agreement["schema_version"], agreement["rehearsal"]), (1, False))

    def test_the_operator_scoring_again_is_said_to_be_the_same_person(self):
        key = self.draw()
        self.score(key, scorer="op1")
        agreement = self.compare()
        self.assertEqual(agreement["scorers"], {"first": ["op1"], "second": ["op1"], "n_same_role": 29})
        self.assertIn("the same person", "\n".join(second_scoring.agreement_lines(agreement)))

    def test_when_no_item_could_be_told_there_is_no_share_to_report(self):
        key = self.draw()
        self.score(key, dict.fromkeys(key.values(), "cannot_tell"))
        agreement = self.compare()
        whole = agreement["all"]
        self.assertEqual((whole["n_cannot_tell"], whole["n_compared"]), (29, 0))
        self.assertEqual((whole["agreement"], whole["wilson_lo"], whole["wilson_hi"]), (None, None, None))
        self.assertIn("could tell none of the 29 items", "\n".join(second_scoring.agreement_lines(agreement)))

    def test_a_rehearsal_stays_marked_as_one(self):
        pilots = [trial(f"T1a-{n:03d}", "T1a", "act_clustered", status="pilot") for n in range(1, 10)]
        log = write_log(self.root / "data" / "trials" / "pilot_policy.jsonl", pilots)
        fake_videos(self.root, pilots)
        second_scoring.make_draw([log], 1, self.out, root=self.root, probe=fake_probe, rehearsal=True)
        self.by_id = {record.planned.trial_id: record for record in pilots}
        self.score(self.key())
        agreement = second_scoring.compare(self.out, [log], root=self.root)
        self.assertEqual((agreement["rehearsal"], agreement["all"]["n_items"]), (True, 2))
        self.assertTrue(second_scoring.agreement_lines(agreement)[0].startswith("REHEARSAL"))

    def test_the_file_written_is_the_same_bytes_each_time_and_names_no_machine(self):
        key = self.draw()
        differs = self.drawn(key, condition="act_spread", outcome="success")
        self.score(key, {differs: "failure"})
        path = second_scoring.write_agreement(self.out, self.compare())
        self.assertEqual(path, self.out / "agreement.json")
        first = path.read_bytes()
        second_scoring.write_agreement(self.out, self.compare())
        self.assertEqual(path.read_bytes(), first)
        text = first.decode("utf-8")
        self.assertEqual(json.loads(text), self.compare())
        self.assertNotIn(str(self.root), text)
        self.assertNotIn(str(self.root.resolve()), text)
        self.assertTrue(text.endswith("}\n"))

    def test_what_is_printed_gives_the_counts_and_names_the_trials_that_differ(self):
        key = self.draw()
        differs = self.drawn(key, condition="act_clustered", outcome="failure")
        unclear = self.drawn(key, condition="replay", outcome="success")
        self.score(key, {differs: "success", unclear: "cannot_tell"}, watched_again={differs: 2})
        agreement = self.compare()
        text = "\n".join(second_scoring.agreement_lines(agreement))
        self.assertIn("29 of the 132 scored trials", text)
        self.assertIn(f"seed {SEED}", text)
        self.assertIn("first scored at the rig by op1, scored again from video by sc2", text)
        self.assertIn("agree in 27 of the 28 items", text)
        self.assertIn("96 %", text)
        table = agreement["all"]["table"]
        self.assertIn(f"first score against second: success and success {table['success_success']}, success and "
                      f"failure 0, failure and success 1, failure and failure {table['failure_failure']}", text)
        self.assertIn(f"{differs}  act_clustered  first failure (grade 2, grasp), second success", text)
        self.assertIn(f"{unclear}  replay  first success", text)
        self.assertIn("policy  23 of 24 agree", text)
        self.assertIn("replay  4 of 4 agree, 1 cannot tell", text)
        self.assertNotIn("REHEARSAL", text)


def round_9(value: float) -> float:
    """A number as the agreement file keeps it: to 9 significant digits."""
    return float(f"{value:.9g}")


# --------------------------------------------------------------------------------------------------------------------
# Real video files, where OpenCV is installed
# --------------------------------------------------------------------------------------------------------------------


def write_video(path: Path, frames: int, *, size: tuple[int, int] = (64, 48), grey: int = 150) -> Path:
    """Write a small video the way the trial runner's recorder does: MP4, `mp4v`, 30 frames a second.

    Frame `i` is a plain grey picture with an orange square that moves one pixel to the right per frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    try:
        for index in range(frames):
            picture = np.full((size[1], size[0], 3), grey, np.uint8)
            picture[10:20, 5 + index:15 + index] = (0, 128, 255)
            writer.write(picture)
    finally:
        writer.release()
    return path


@unittest.skipUnless(cv2 is not None, NEEDS_OPENCV)
class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_it_counts_the_frames_of_a_video_written_as_the_recorder_writes_them(self):
        self.assertEqual(second_scoring.probe_video(write_video(self.folder / "a.mp4", 12)), 12)
        self.assertEqual(second_scoring.probe_video(str(write_video(self.folder / "b.mp4", 3))), 3)

    def test_a_file_cut_short_an_empty_one_and_one_that_is_not_a_video_do_not_open(self):
        whole = write_video(self.folder / "whole.mp4", 40).read_bytes()
        (self.folder / "cut.mp4").write_bytes(whole[: len(whole) * 6 // 10])  # no index at its end: never closed
        (self.folder / "empty.mp4").write_bytes(b"")
        (self.folder / "text.mp4").write_text("not a video", encoding="utf-8")
        for name in ("cut.mp4", "empty.mp4", "text.mp4", "missing.mp4"):
            with self.subTest(file=name), self.assertRaises(second_scoring.VideoError) as error:
                second_scoring.probe_video(self.folder / name)
            self.assertEqual(str(error.exception), "does not open as a video")

    def test_it_takes_a_file_and_never_a_camera_s_number(self):
        with self.assertRaises(TypeError):
            second_scoring.probe_video(0)

    def test_the_check_of_a_trial_uses_it_unless_another_probe_is_given(self):
        record = trial("T1a-017", "T1a", "act_clustered", steps_sent=12)
        for view in ("front", "wrist"):
            write_video(self.folder / video_name("T1a-017", view), 12)
        check = second_scoring.VideoCheck(self.folder)
        candidate = second_scoring.Candidate(MAIN, record)
        self.assertIsNone(check(candidate))
        write_video(self.folder / video_name("T1a-017", "wrist"), 9)
        self.assertEqual(check(candidate), "its wrist video holds 9 frames and the trial sent 12 steps")


@unittest.skipUnless(cv2 is not None, NEEDS_OPENCV)
class PlayerAndWindowTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.front = write_video(self.folder / "videos" / "item-01_front.mp4", 12, grey=150)
        self.wrist = write_video(self.folder / "videos" / "item-01_wrist.mp4", 9, grey=60)
        self.item = second_scoring.Item("item-01", (("front", self.front), ("wrist", self.wrist)))
        self.player = second_scoring.VideoPlayer()

    def test_the_views_are_laid_side_by_side_frame_by_frame(self):
        pictures = list(self.player.frames(self.item))
        self.assertEqual(len(pictures), 12)
        for picture in pictures:
            self.assertEqual(picture.shape, (48, 128, 3))
        first = pictures[0]
        self.assertAlmostEqual(float(first[40:, :60].mean()), 150, delta=6)  # the front view on the left
        self.assertAlmostEqual(float(first[40:, 68:].mean()), 60, delta=6)  # the wrist view on the right

    def test_a_view_that_ends_first_keeps_its_last_picture(self):
        pictures = list(self.player.frames(self.item))
        self.assertTrue(np.array_equal(pictures[8][:, 64:], pictures[11][:, 64:]))
        self.assertFalse(np.array_equal(pictures[5][:, 64:], pictures[8][:, 64:]))
        self.assertFalse(np.array_equal(pictures[8][:, :64], pictures[11][:, :64]))

    def test_every_call_plays_the_video_from_its_first_frame(self):
        first = [picture.copy() for picture in self.player.frames(self.item)]
        partly = self.player.frames(self.item)
        next(partly)
        partly.close()
        again = list(self.player.frames(self.item))
        self.assertEqual(len(again), len(first))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, again)))

    def test_one_view_alone_is_shown_as_it_is(self):
        alone = second_scoring.Item("item-01", (("front", self.front),))
        self.assertEqual([picture.shape for picture in self.player.frames(alone)], [(48, 64, 3)] * 12)

    def test_the_pace_is_the_video_s_own_rate(self):
        self.assertEqual(self.player.frame_ms(self.item), 33)

    def test_a_view_that_does_not_open_is_an_error_that_names_the_item_only(self):
        self.wrist.write_text("not a video", encoding="utf-8")
        with self.assertRaises(SecondScoringError) as refusal:
            list(self.player.frames(self.item))
        self.assertIn("item-01", str(refusal.exception))
        self.assertNotIn(str(self.folder), str(refusal.exception))
        both = second_scoring.Item("item-02", (("front", self.wrist),))
        self.assertEqual(list(self.player.frames(both)), [])

    def test_pictures_of_different_heights_are_brought_to_the_first_one_s(self):
        row = second_scoring.side_by_side([np.zeros((48, 64, 3), np.uint8), np.full((96, 128, 3), 255, np.uint8)])
        self.assertEqual(row.shape, (48, 128, 3))
        self.assertEqual((int(row[:, :64].max()), int(row[:, 64:].min())), (0, 255))

    def test_the_text_goes_under_the_picture_and_never_over_it(self):
        picture = np.full((480, 1280, 3), 90, np.uint8)
        lines = ["item-07     7 of 29", "the video has ended.   s: success   f: failure   c: cannot tell"]
        canvas = second_scoring.with_lines(picture, lines)
        self.assertEqual(canvas.shape[1], 1280)
        self.assertGreater(canvas.shape[0], 480 + 40)
        self.assertTrue(np.array_equal(canvas[:480], picture))
        self.assertGreater(int(canvas[480:].max()), 200, "no text was drawn under the picture")
        self.assertTrue(np.array_equal(second_scoring.with_lines(picture, ["", ""])[480:],
                                       np.zeros_like(canvas[480:])))

    def test_a_narrow_picture_is_centred_on_a_canvas_wide_enough_for_the_text(self):
        picture = np.full((480, 640, 3), 90, np.uint8)
        canvas = second_scoring.with_lines(picture, ["item-07     7 of 29"])
        width = second_scoring.MIN_WINDOW_WIDTH_PX
        left = (width - 640) // 2
        self.assertEqual(canvas.shape[1], width)
        self.assertTrue(np.array_equal(canvas[:480, left:left + 640], picture))
        self.assertEqual(int(canvas[:480, :left].max()), 0)

    def with_opencv_replaced(self, **functions):
        """Put stand-ins in place of OpenCV's window functions for the length of a test."""
        real = {name: getattr(cv2, name) for name in functions}
        for name, function in functions.items():
            setattr(cv2, name, function)
        self.addCleanup(lambda: [setattr(cv2, name, function) for name, function in real.items()])

    def test_the_window_shows_under_one_title_and_hands_back_the_key_s_whole_code(self):
        calls = []

        def cut_down(_ms):
            raise AssertionError("waitKey hands back only eight bits of a key's code: the window must not use it")

        self.with_opencv_replaced(
            imshow=lambda title, canvas: calls.append(("show", title, canvas.shape)),
            waitKeyEx=lambda ms: calls.append(("wait", ms)) or 0xF703,
            waitKey=cut_down,
            destroyAllWindows=lambda: calls.append(("close",)),
        )
        window = second_scoring.Window()
        window.show(np.zeros((480, 1280, 3), np.uint8), ["item-07     7 of 29", "watch to the end"])
        key = window.wait_key(0)
        window.close()
        self.assertEqual(key, 0xF703, "the right arrow must come back whole, or it reads as the keypad's Enter")
        self.assertEqual(calls[0], ("show", "second scoring", (480 + 12 + 56, 1280, 3)))
        self.assertEqual(calls[1], ("wait", 1))
        self.assertEqual(calls[2], ("close",))

    def test_the_right_arrow_pressed_in_the_real_window_s_place_records_nothing(self):
        """The whole way from the window to the file: a choice, then the right arrow as a Mac's window reports it."""
        # Space to start, twelve pictures, a choice, three keys that must not record it, stop; then the window closes.
        keys = [ord(" "), *([-1] * 12), ord("f"), 0xF703, 0xF70D, 3, ord("x"), -1]
        self.with_opencv_replaced(imshow=lambda _title, _canvas: None, waitKeyEx=lambda _ms: keys.pop(0),
                                  destroyAllWindows=lambda: None)
        items = self.folder / "items.csv"
        items.write_text("item,front,wrist\nitem-01,videos/item-01_front.mp4,videos/item-01_wrist.mp4\n",
                         encoding="utf-8")
        result = second_scoring.score_items(items, "sc2", player=self.player, screen=second_scoring.Window(),
                                            say=lambda _line: None)
        self.assertEqual((result, keys), ((0, 1), []))
        self.assertFalse((self.folder / "scores.csv").exists())

    def test_return_pressed_in_the_real_window_s_place_records_the_choice(self):
        keys = [ord(" "), *([-1] * 12), ord("f"), 13, -1]
        self.with_opencv_replaced(imshow=lambda _title, _canvas: None, waitKeyEx=lambda _ms: keys.pop(0),
                                  destroyAllWindows=lambda: None)
        items = self.folder / "items.csv"
        items.write_text("item,front,wrist\nitem-01,videos/item-01_front.mp4,videos/item-01_wrist.mp4\n",
                         encoding="utf-8")
        result = second_scoring.score_items(items, "sc2", player=self.player, screen=second_scoring.Window(),
                                            say=lambda _line: None)
        self.assertEqual((result, keys), ((1, 0), []))
        line = (self.folder / "scores.csv").read_text(encoding="utf-8").splitlines()[1]
        self.assertEqual(line, f"item-01,failure,sc2,0,{second_scoring.videos_digest([self.front, self.wrist])}")

    def test_an_opencv_without_a_window_is_refused_in_one_sentence(self):
        def no_window(_title, _canvas):
            raise cv2.error("The function is not implemented. Rebuild the library with Cocoa support.")

        self.with_opencv_replaced(imshow=no_window)
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.Window().show(np.zeros((48, 64, 3), np.uint8), ["item-01"])
        self.assertIn("cannot open a window", str(refusal.exception))

    def test_nothing_in_the_tool_reads_a_key_through_the_call_that_cuts_its_code_down(self):
        for name in ("src/tolerance_radius/second_scoring.py", "scripts/second_scoring.py"):
            source = (REPO / name).read_text(encoding="utf-8")
            self.assertNotIn("waitKey(", source.replace("waitKeyEx(", ""), name)
            self.assertNotIn("& 0xFF)", source, name)


@unittest.skipUnless(cv2 is not None, NEEDS_OPENCV)
class WholeFlowOnRealVideosTests(unittest.TestCase):
    """Draw, score and compare with nothing faked but the scorer's keys and the window."""

    FRAMES = 6

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        sizes = {("T1a", "act_clustered"): 6, ("T1b", "replay"): 4}
        self.records = [trial(r.planned.trial_id, r.planned.block, r.planned.condition, r.outcome,
                              steps_sent=self.FRAMES) for r in experiment(sizes, voids=1)]
        self.log = write_log(self.root / "data" / "trials" / "main.jsonl", self.records)
        for record in self.records:
            if record.scored:
                for view in ("front", "wrist"):
                    write_video(self.root / record.media[view], self.FRAMES)
        self.out = self.root / "data" / "second_scoring" / "main"

    def test_the_three_steps_follow_one_another(self):
        record = second_scoring.make_draw([self.log], SEED, self.out, root=self.root)
        self.assertEqual((record["n_scored"], record["n_drawn"], record["passed_over"]), (10, 3, []))
        for name in ("item-01_front.mp4", "item-03_wrist.mp4"):
            self.assertEqual(second_scoring.probe_video(self.out / "videos" / name), self.FRAMES)

        first = {r.planned.trial_id: r.outcome for r in self.records}
        keys = []
        for _, trial_id, _, _ in read_key(self.out):
            keys += [" ", *([None] * self.FRAMES), first[trial_id][0], "enter"]
        screen = FakeScreen(keys)
        result = second_scoring.score_items(self.out / "items.csv", "sc2", player=second_scoring.VideoPlayer(),
                                            screen=screen, say=lambda _line: None)
        self.assertEqual(result, (3, 0))
        self.assertEqual({picture.shape for picture, _ in screen.shown}, {(48, 128, 3)})
        self.assertEqual(screen.waits[1:1 + self.FRAMES], [33] * self.FRAMES)

        agreement = second_scoring.compare(self.out, [self.log], root=self.root)
        self.assertEqual((agreement["all"]["n_items"], agreement["all"]["n_agree"]), (3, 3))

    def test_a_video_that_was_never_closed_is_passed_over_at_the_draw(self):
        torn = planned_draw(self.log, self.root).items[0].candidate
        path = self.root / torn.record.media["front"]
        path.write_bytes(path.read_bytes()[:400])
        record = second_scoring.make_draw([self.log], SEED, self.out, root=self.root)
        self.assertEqual(record["passed_over"], [{"trial_id": torn.trial_id, "log": MAIN,
                                                  "reason": "its front video does not open as a video"}])
        self.assertEqual(record["n_drawn"], 3)

    def test_a_video_with_fewer_frames_than_steps_is_passed_over_at_the_draw(self):
        short = planned_draw(self.log, self.root).items[1].candidate
        write_video(self.root / short.record.media["wrist"], self.FRAMES - 2)
        record = second_scoring.make_draw([self.log], SEED, self.out, root=self.root)
        self.assertEqual(record["passed_over"], [{
            "trial_id": short.trial_id, "log": MAIN,
            "reason": f"its wrist video holds {self.FRAMES - 2} frames and the trial sent {self.FRAMES} steps"}])
        self.assertNotIn(short.trial_id, [trial_id for _, trial_id, _, _ in read_key(self.out)])


# --------------------------------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------------------------------


class CommandTests(unittest.TestCase):
    """`scripts/second_scoring.py`, run on a temporary folder as its root: never on the repository."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.records = experiment()
        self.by_id = {record.planned.trial_id: record for record in self.records}
        self.log = write_log(self.root / "data" / "trials" / "main.jsonl", self.records)
        fake_videos(self.root, self.records)
        self.out = self.root / "data" / "second_scoring" / "main"

    def run_command(self, *arguments: str, **stand_ins) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = command.main([str(argument) for argument in arguments], **stand_ins)
        return code, out.getvalue(), err.getvalue()

    def refused(self, *arguments: str, **stand_ins) -> str:
        """The one sentence a refused run prints, after checking that it wrote nothing."""
        before = files_below(self.root)
        code, out, err = self.run_command(*arguments, **stand_ins)
        self.assertEqual(code, 1)
        self.assertEqual(files_below(self.root), before, "a refused run writes nothing")
        self.assertRegex(err, r"^second_scoring: refused: [^\n]+\.\n$")
        self.assertNotIn(str(self.root), err)
        return out + err

    def drawn(self) -> list[tuple[str, str, str, str]]:
        second_scoring.make_draw([self.log], SEED, self.out, root=self.root, probe=fake_probe)
        return read_key(self.out)

    def score_all(self, key: list[tuple[str, str, str, str]], only: int | None = None) -> None:
        for code, trial_id, _, digest in key[:only]:
            second_scoring.append_score(self.out / "scores.csv", second_scoring.Score(
                code, self.by_id[trial_id].outcome, "sc2", 0, digest))

    def test_score_runs_a_sitting_and_tells_the_scorer_the_rule(self):
        self.drawn()
        screen = FakeScreen([*scored("s"), *scored("f"), None, "x"])
        code, out, err = self.run_command("score", "--items", self.out / "items.csv", "--scorer", "sc2",
                                          player=FakePlayer(), screen=screen)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("released in the basket", out)
        self.assertIn("  item-01: success\n  item-02: failure\n", out)
        self.assertIn("stopped at item-03", out)
        lines = (self.out / "scores.csv").read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual([",".join(line.split(",")[:4]) for line in lines],
                         ["item-01,success,sc2,0", "item-02,failure,sc2,0"])
        for word in identity_words(self.records):
            self.assertNotIn(word, out)

    def test_score_refuses_a_name_before_anything_is_shown(self):
        self.drawn()
        screen = FakeScreen([])
        text = self.refused("score", "--items", self.out / "items.csv", "--scorer", "someone",
                            player=FakePlayer(), screen=screen)
        self.assertIn("role code", text)
        self.assertEqual(screen.shown, [])

    def test_compare_prints_the_agreement_and_writes_it(self):
        self.score_all(self.drawn())
        code, out, err = self.run_command("compare", "--dir", self.out, "--log", self.log, "--root", self.root)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("the two scores agree in 29 of the 29 items", out)
        self.assertTrue(out.endswith("wrote data/second_scoring/main/agreement.json\n"))
        written = json.loads((self.out / "agreement.json").read_text(encoding="utf-8"))
        self.assertEqual(written["all"]["n_agree"], 29)
        self.assertNotIn(str(self.root), out)

    def test_compare_is_refused_in_one_sentence_while_items_are_left(self):
        key = self.drawn()
        self.score_all(key, only=1)
        text = self.refused("compare", "--dir", self.out, "--log", self.log, "--root", self.root)
        self.assertIn("1 of 29 items are scored", text)
        self.assertNotIn(key[0][1], text)

    def test_draw_is_refused_into_a_folder_that_has_scores(self):
        self.drawn()
        (self.out / "scores.csv").write_text("item,score,scorer,watched_again,videos_sha256\n", encoding="utf-8")
        text = self.refused("draw", "--log", self.log, "--seed", SEED, "--out", self.out, "--root", self.root)
        self.assertIn("already holds scores", text)

    def test_draw_refuses_a_pilot_log_unless_it_is_a_rehearsal(self):
        pilot = write_log(self.root / "data" / "trials" / "pilot_policy.jsonl",
                          [trial("T1a-101", "T1a", "act_clustered", status="pilot")])
        text = self.refused("draw", "--log", pilot, "--seed", "1", "--out", self.out, "--root", self.root)
        self.assertIn("pilot", text)

    def test_a_folder_outside_the_repository_is_named_so_that_it_can_be_found_again(self):
        elsewhere = Path(self.enterContext(tempfile.TemporaryDirectory())) / "handed-over"
        self.assertEqual(command.shown(elsewhere, self.root.resolve()), str(elsewhere))
        self.assertEqual(command.shown(self.out, self.root.resolve()), "data/second_scoring/main")

    def test_stopping_with_control_c_says_what_that_leaves_for_each_step(self):
        def interrupted(*_arguments, **_options):
            raise KeyboardInterrupt

        real = second_scoring.make_draw
        second_scoring.make_draw = interrupted
        self.addCleanup(setattr, second_scoring, "make_draw", real)
        code, _, err = self.run_command("draw", "--log", self.log, "--seed", SEED, "--out", self.out,
                                        "--root", self.root)
        self.assertEqual(code, 130)
        self.assertIn("the draw was cut short", err)
        self.assertNotIn("Scores", err)

    def test_the_command_starts_with_the_rig_s_interpreter_and_names_no_machine(self):
        text = (REPO / "scripts" / "second_scoring.py").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], "#!/bin/sh")
        self.assertIn("TOLRAD_RIG_PYTHON", text.splitlines()[2])
        for example in ("draw --log data/trials/main.jsonl --seed 20261006 --out data/second_scoring/main",
                        "score --items data/second_scoring/main/items.csv --scorer sc2",
                        "compare --dir data/second_scoring/main --log data/trials/main.jsonl"):
            self.assertIn(f"scripts/second_scoring.py {example}", command.__doc__)

    @unittest.skipUnless(cv2 is not None, NEEDS_OPENCV)
    def test_draw_reads_real_videos_and_writes_the_folder(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        records = [trial(f"T1a-{n:03d}", "T1a", "act_clustered", "success" if n % 2 else "failure", steps_sent=5)
                   for n in range(1, 8)]
        log = write_log(root / "data" / "trials" / "main.jsonl", records)
        for record in records:
            for view in ("front", "wrist"):
                write_video(root / record.media[view], 5)
        out = root / "data" / "second_scoring" / "main"
        code, text, err = self.run_command("draw", "--log", log, "--seed", SEED, "--out", out, "--root", root)
        self.assertEqual((code, err), (0, ""))
        self.assertIn(f"drew 2 of the 7 scored trials of data/trials/main.jsonl (seed {SEED})", text)
        self.assertIn("every drawn video could be read to its end", text)
        self.assertEqual(sorted(path.name for path in (out / "videos").iterdir()),
                         ["item-01_front.mp4", "item-01_wrist.mp4", "item-02_front.mp4", "item-02_wrist.mp4"])
        for record in records:
            self.assertNotIn(record.planned.trial_id, text)
        self.assertNotIn(str(root), text)

    @unittest.skipIf(cv2 is not None, "OpenCV is installed here")
    def test_without_opencv_draw_and_score_say_which_interpreter_to_use(self):
        text = self.refused("draw", "--log", self.log, "--seed", SEED, "--out", self.out, "--root", self.root)
        self.assertIn("OpenCV", text)
        self.assertFalse(self.out.exists())
        self.drawn()
        text = self.refused("score", "--items", self.out / "items.csv", "--scorer", "sc2")
        self.assertIn("rig's interpreter", text)


# --------------------------------------------------------------------------------------------------------------------
# The page that describes the files, and the runbook
# --------------------------------------------------------------------------------------------------------------------


def fields(value, prefix: str = "") -> set[str]:
    """Where the fields of a JSON value are: `a.b` for a field of an object, `[]` for the entries of a list."""
    found = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(f"{prefix}{key}")
            found |= fields(inner, f"{prefix}{key}.")
    elif isinstance(value, list):
        for inner in value:
            found |= fields(inner, f"{prefix}[].")
    return found


class ContractPageTests(unittest.TestCase):
    """`protocol/data-contracts.md` shows each file by an example. The examples must have the fields the code writes."""

    @classmethod
    def setUpClass(cls):
        text = (REPO / "protocol" / "data-contracts.md").read_text(encoding="utf-8")
        cls.section = text[text.index("## The second scoring"):]
        cls.draw_example, cls.agreement_example = (
            json.loads(block) for block in re.findall(r"```json\n(\{.*?\n\})\n```", cls.section, flags=re.S))

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.records = experiment()
        self.log = write_log(self.root / "data" / "trials" / "main.jsonl", self.records)
        fake_videos(self.root, self.records)
        lost = planned_draw(self.log, self.root).items[0].candidate.trial_id
        (self.root / video_name(lost, "wrist")).unlink()  # so that one trial is passed over
        self.out = self.root / "data" / "second_scoring" / "main"
        self.record = second_scoring.make_draw([self.log], SEED, self.out, root=self.root, probe=fake_probe)

    def test_the_section_is_the_last_one_and_its_examples_are_not_taken_for_the_results_contract(self):
        self.assertNotIn("\n## ", self.section[3:], "the second scoring is no longer the page's last section")
        self.assertEqual(re.findall(r"```\n\{", self.section), [],
                         "a plain fenced object here is read by tests/test_report.py as part of results.json")

    def test_the_record_of_a_draw_has_the_fields_of_its_example(self):
        self.assertEqual(len(self.record["passed_over"]), 1)
        self.assertEqual(fields(self.record), fields(self.draw_example))

    def test_the_headings_of_the_three_tables_are_the_ones_the_page_gives(self):
        for heading in ("item,front,wrist", ",".join(second_scoring.SCORE_HEADER), ",".join(second_scoring.KEY_HEADER)):
            self.assertIn(f"`{heading}`", self.section)
        self.assertEqual((self.out / "items.csv").read_text(encoding="utf-8").splitlines()[0], "item,front,wrist")

    def test_the_agreement_has_the_fields_of_its_example(self):
        by_id = {record.planned.trial_id: record for record in self.records}
        for number, (code, trial_id, _, digest) in enumerate(read_key(self.out)):
            first = by_id[trial_id].outcome
            second = {0: "cannot_tell", 1: "failure" if first == "success" else "success"}.get(number, first)
            second_scoring.append_score(self.out / "scores.csv",
                                        second_scoring.Score(code, second, "sc2", number % 2, digest))
        agreement = second_scoring.compare(self.out, [self.log], root=self.root)
        example = json.loads(json.dumps(self.agreement_example))
        # What the example leaves to "the same" or to another file, written out.
        example["provenance"]["groups"] = self.draw_example["groups"]
        example["provenance"]["passed_over"] = self.draw_example["passed_over"]
        example["by_kind"] = {"policy": example["all"], "replay": example["all"]}
        example["cannot_tell"] = example["disagreements"]
        self.assertEqual(fields(agreement), fields(example))

    def test_the_example_s_numbers_are_what_the_code_gives_for_its_counts(self):
        whole = self.agreement_example["all"]
        low, high = wilson_interval(whole["n_agree"], whole["n_compared"])
        self.assertEqual((whole["agreement"], whole["wilson_lo"], whole["wilson_hi"]),
                         (round_9(whole["n_agree"] / whole["n_compared"]), round_9(low), round_9(high)))
        self.assertEqual(whole["n_items"], whole["n_agree"] + whole["n_disagree"] + whole["n_cannot_tell"])
        self.assertEqual(sum(whole["table"].values()), whole["n_compared"])
        self.assertEqual(sum(whole["cannot_tell_first"].values()), whole["n_cannot_tell"])
        self.assertEqual(self.draw_example["n_drawn"], self.agreement_example["provenance"]["n_drawn"])


class RunbookTests(unittest.TestCase):
    """`docs/runbook/05-second-scoring.md` gives commands to type and lines to expect. Both must be the tool's."""

    @classmethod
    def setUpClass(cls):
        runbook = REPO / "docs" / "runbook" / "05-second-scoring.md"
        if not runbook.is_file():
            raise unittest.SkipTest("the rig's runbooks are not part of this copy of the repository")
        cls.text = runbook.read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", cls.text, flags=re.S)
        cls.commands = [shlex.split(line) for block in blocks for line in block.splitlines() if line.strip()]

    def test_every_command_of_the_runbook_is_one_the_tool_takes(self):
        self.assertEqual([words[:2] for words in self.commands],
                         [["scripts/second_scoring.py", name] for name in ("draw", "score", "compare") * 2])
        parsed = [command.build_parser().parse_args(words[1:]) for words in self.commands]
        rehearsal, counted = parsed[0], parsed[3]
        self.assertEqual((rehearsal.rehearsal, [log.name for log in rehearsal.log]), (True, ["pilot_policy.jsonl"]))
        self.assertEqual((counted.rehearsal, counted.seed, counted.out.as_posix()),
                         (False, 20261006, "data/second_scoring/main"))
        self.assertEqual([log.as_posix() for log in counted.log], [MAIN])
        self.assertEqual((parsed[4].items.as_posix(), parsed[4].scorer),
                         ("data/second_scoring/main/items.csv", "sc2"))
        self.assertEqual((parsed[5].dir.as_posix(), [log.as_posix() for log in parsed[5].log]),
                         ("data/second_scoring/main", [MAIN]))

    def test_the_three_counted_commands_are_the_ones_in_the_tool_s_own_help(self):
        for words in self.commands[3:]:
            self.assertIn(" ".join(words), command.__doc__)

    def test_the_seed_it_names_is_the_protocol_s_own(self):
        self.assertEqual(load_protocol()["datasets"]["layout_seed"], 20261006)
        self.assertIn("`[datasets].layout_seed`", self.text)

    def test_the_lines_it_quotes_are_lines_the_tool_prints(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        records = experiment()
        log = write_log(root / "data" / "trials" / "main.jsonl", records)
        fake_videos(root, records)
        out = root / "data" / "second_scoring" / "main"
        record = second_scoring.make_draw([log], SEED, out, root=root, probe=fake_probe)
        printed = "\n".join(second_scoring.draw_lines(record, "data/second_scoring/main"))
        clean = "every drawn video could be read to its end, and none has fewer frames than its trial sent steps"
        self.assertIn(clean, printed)
        self.assertIn(f"`{clean}`", " ".join(self.text.split()))
        self.assertIn("29 of the 132", self.text)
        self.assertIn("drew 29 of the 132 scored trials", printed)
        source = (REPO / "src" / "tolerance_radius" / "second_scoring.py").read_text(encoding="utf-8")
        for quoted in ("already holds a draw", "already holds scores", "is not empty", "has changed since the draw",
                       "is not there", "record names no front video", "other video files", "another sitting",
                       "cannot open a window"):
            self.assertIn(f"`{quoted}`", self.text)
            self.assertIn(quoted.replace("front", "{view}"), source)

    def test_the_keys_it_names_are_the_tool_s(self):
        self.assertEqual((second_scoring.KEY_START, second_scoring.KEY_AGAIN, second_scoring.KEY_STOP), (" ", "r", "x"))
        self.assertEqual(second_scoring.KEY_OF_SCORE, {"success": "s", "failure": "f", "cannot_tell": "c"})
        for key in ("**space bar**", "**s**", "**f**", "**c**", "**Enter**", "**r**", "**x**"):
            self.assertIn(key, self.text)


# --------------------------------------------------------------------------------------------------------------------
# The trial runner's own videos
# --------------------------------------------------------------------------------------------------------------------


@unittest.skipUnless(fakes_rig.READY, fakes_rig.WHY_NOT)
class RecordedByTheTrialRunnerTests(unittest.TestCase):
    """What the second scoring takes for granted about a trial's videos, checked on videos the trial runner wrote.

    The session runs on the rig made of fakes, with the arm tooling's real attempt recorder. If the runner or the
    recorder comes to name, place or fill its videos differently, this fails here and not on the day of the scoring.
    """

    def test_a_trial_s_videos_are_named_in_its_record_and_hold_a_frame_for_every_step(self):
        with fakes_rig.FakeRig(real_recorder=True) as rig:
            # A placement that nothing ends waits ten minutes in a real session. Here it must fail the test instead.
            rig.placement_timeout_s = 10.0
            rig.write_audits()
            log = rig.trial_log("pilot_test.jsonl")
            run = rig.run_trials(["--blocks", "T1a", "--log", str(log), "--pilot", "--policy",
                                  f"act_clustered={rig.policy_dir('clustered')}", "--limit", "2", "--no-display"])
            self.assertEqual((run.rc, run.error), (0, None))
            records = schema.read_trials(log)
            self.assertEqual([record.outcome for record in records], ["success", "success"])
            check = second_scoring.VideoCheck(rig.repo_root)
            for record in records:
                candidate = second_scoring.Candidate("pilot_test.jsonl", record)
                with self.subTest(trial=candidate.trial_id):
                    self.assertEqual(sorted(record.media), ["front", "wrist"])
                    # Usable as it is: named, there, decodable, and with a frame for every step sent.
                    self.assertIsNone(check(candidate))
                    steps = record.execution["steps_sent"]
                    for view, name in record.media.items():
                        self.assertFalse(Path(name).is_absolute())
                        # The file is named after the trial, block and all: why the scorer is given a copy.
                        self.assertIn(f"_{candidate.trial_id}_{view}.mp4", name)
                        frames = second_scoring.probe_video(rig.repo_root / name)
                        self.assertGreaterEqual(frames, steps, "the video lacks a frame for some step")
                        self.assertLessEqual(frames, steps + 2)
            item = second_scoring.Item("item-01", tuple(
                (view, rig.repo_root / records[0].media[view]) for view in second_scoring.VIEWS))
            self.assertEqual({picture.shape for picture in second_scoring.VideoPlayer().frames(item)},
                             {(480, 1280, 3)})


@unittest.skipIf(cv2 is not None, "OpenCV is installed here")
class WithoutOpenCVTests(unittest.TestCase):
    def test_reading_a_video_says_which_interpreter_has_what_it_needs(self):
        with self.assertRaises(SecondScoringError) as refusal:
            second_scoring.probe_video(Path("anything.mp4"))
        self.assertIn("OpenCV", str(refusal.exception))
        self.assertIn("rig's interpreter", str(refusal.exception))


if __name__ == "__main__":
    unittest.main()
