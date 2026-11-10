"""The two arm sessions and the audit, as far as they can be tested without an arm.

Three kinds of test: the pure helpers (which episodes a sitting records, which trial comes next, what a trial's record
holds); the audit on synthetic pictures; and structural checks that read the session sources and fail if a later edit
removes a safety (a control loop reached without its gate, an arm left without its settle step).
"""

import ast
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tolerance_radius import layouts, schema
from tolerance_radius.config import REPO_ROOT, load_protocol
from tolerance_radius.geometry import Grid
from tolerance_radius.rig import record_layout, trial_session
from tolerance_radius.schema import DemoStart, PlannedTrial, Point, TrialRecord

from fixtures_pictures import wrist_rest_picture

try:
    from test_rig_vision import PX_TO_MM, READY, scene
except ImportError:  # pragma: no cover -- the vision test module is always beside this one
    READY = False

PROTOCOL = load_protocol()
LAYOUT_DIR = REPO_ROOT / "config" / "layouts"
RIG_SOURCES = REPO_ROOT / "src" / "tolerance_radius" / "rig"


# ---------------------------------------------------------------------------------------------------- the recorder


class SittingTests(unittest.TestCase):
    def test_the_four_sittings_cover_each_dataset_once_in_recording_order(self):
        expected = {1: ("clustered", 0, 33), 2: ("spread", 0, 33), 3: ("spread", 33, 55), 4: ("clustered", 33, 55)}
        for number, (dataset, first, end) in expected.items():
            with self.subTest(sitting=number):
                sitting = record_layout.sitting_from_protocol(PROTOCOL, number, LAYOUT_DIR)
                self.assertEqual((sitting.dataset, sitting.first_index, sitting.end_index), (dataset, first, end))
                self.assertEqual([e.episode_index for e in sitting.episodes], list(range(first, end)))
                self.assertTrue(all(e.dataset == dataset for e in sitting.episodes))

    def test_a_sitting_outside_the_plan_is_refused(self):
        for number in (0, 5):
            with self.subTest(sitting=number), self.assertRaises(ValueError):
                record_layout.sitting_from_protocol(PROTOCOL, number, LAYOUT_DIR)

    def test_a_fresh_dataset_records_the_whole_first_sitting(self):
        sitting = record_layout.sitting_from_protocol(PROTOCOL, 1, LAYOUT_DIR)
        todo = record_layout.episodes_to_record(sitting, already_recorded=0)
        self.assertEqual([e.episode_index for e in todo], list(range(33)))

    def test_a_sitting_stopped_early_continues_at_the_next_unrecorded_episode(self):
        sitting = record_layout.sitting_from_protocol(PROTOCOL, 1, LAYOUT_DIR)
        todo = record_layout.episodes_to_record(sitting, already_recorded=20)
        self.assertEqual(todo[0].episode_index, 20)
        self.assertEqual(todo[-1].episode_index, 32)

    def test_a_later_sitting_is_refused_until_the_earlier_one_is_complete(self):
        sitting = record_layout.sitting_from_protocol(PROTOCOL, 4, LAYOUT_DIR)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            record_layout.episodes_to_record(sitting, already_recorded=30)
        self.assertEqual(record_layout.episodes_to_record(sitting, already_recorded=33)[0].episode_index, 33)

    def test_a_sitting_already_recorded_is_refused(self):
        sitting = record_layout.sitting_from_protocol(PROTOCOL, 2, LAYOUT_DIR)
        with self.assertRaisesRegex(ValueError, "nothing left"):
            record_layout.episodes_to_record(sitting, already_recorded=33)

    def test_a_sitting_of_the_other_set_that_comes_first_and_is_not_all_recorded_is_said(self):
        # The dry run of 2026-10-06: sitting 2 started with sitting 1 six episodes short, and nothing said so.
        def note(number, **held):
            sitting = record_layout.sitting_from_protocol(PROTOCOL, number, LAYOUT_DIR)
            return record_layout.out_of_turn(PROTOCOL, sitting, held, LAYOUT_DIR)

        self.assertIsNone(note(1, clustered=0, spread=0))
        self.assertIsNone(note(1, clustered=27, spread=0))
        told = note(2, clustered=27, spread=0)
        self.assertIn("sitting 2 is meant to come after the sittings before it", told)
        self.assertIn("sitting 1 ends at episode 32 of the clustered set, which holds 27 episodes", told)
        self.assertIsNone(note(2, clustered=33, spread=0))
        self.assertIsNone(note(2, clustered=33, spread=20))            # its own set's count is another rule's
        self.assertIsNone(note(3, clustered=33, spread=33))
        self.assertIn("sitting 1 ends at episode 32", note(3, clustered=30, spread=33))
        # the last sitting of the clustered set waits for both sittings of the spread set
        told = note(4, clustered=33, spread=33)
        self.assertIn("sitting 3 ends at episode 54 of the spread set, which holds 33 episodes", told)
        self.assertNotIn("sitting 2", told)
        self.assertIsNone(note(4, clustered=33, spread=55))
        self.assertIn("which holds 0 episodes", note(4, clustered=33))  # a set that is not there holds nothing

    def test_placements_are_appended_one_line_each(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording" / "clustered_placements.jsonl"
            record_layout.append_placement(path, {"episode_index": 0, "placed": {"x_mm": 1.0, "y_mm": 2.0}})
            record_layout.append_placement(path, {"episode_index": 1, "placed": {"x_mm": 3.0, "y_mm": 4.0}})
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([line["episode_index"] for line in lines], [0, 1])


class ResumeTests(unittest.TestCase):
    """A set that a session left without closing must not be appended to (the dry run of 2026-10-06: killed after
    two saves, `info.json` said 29 episodes, the episode table had 27 rows and a table file had no end; the next
    session went on at episode 29, wrote over that file, and the set no longer loaded)."""

    def test_a_set_whose_counts_agree_may_be_appended_to(self):
        sound = {"said": 27, "rows": 27, "unreadable": []}
        self.assertIsNone(record_layout.resume_problem(sound, 27, "clustered_placements.jsonl"))
        # a set left empty by a refused start: nothing saved, nothing logged
        self.assertIsNone(record_layout.resume_problem({"said": 0, "rows": 0, "unreadable": []}, 0, "x.jsonl"))
        # where the table's rows cannot be counted (no parquet reader), the other two are still held to each other
        self.assertIsNone(record_layout.resume_problem({"said": 27, "rows": None, "unreadable": []}, 27, "x.jsonl"))

    def test_what_a_session_that_died_leaves_is_refused_and_named(self):
        died = {"said": 29, "rows": 27, "unreadable": ["data/chunk-000/file-002.parquet"]}
        told = record_layout.resume_problem(died, 29, "clustered_placements.jsonl")
        self.assertIn("1 of its table files cannot be read (data/chunk-000/file-002.parquet)", told)
        self.assertIn("a session ended without closing the set", told)
        told = record_layout.resume_problem({"said": 29, "rows": 27, "unreadable": []}, 29, "x.jsonl")
        self.assertIn("it says it holds 29 episodes and its episode table has 27", told)

    def test_pictures_of_a_take_that_was_never_saved_are_refused_though_the_counts_agree(self):
        # A session cut off in the middle of a take, nothing saved since the set was closed: the counts agree, and
        # the next save failed on a half-written picture or took the dead take's frames with it (the review of the
        # night of 2026-10-06, with LeRobot's own writer on copies).
        told = record_layout.resume_problem({"said": 4, "rows": 4, "unreadable": [], "pictures": 44}, 4, "x.jsonl")
        self.assertIn("44 pictures of a take that was never saved are left in its images folder", told)
        self.assertIn("a session ended in the middle of a take", told)
        self.assertIsNone(record_layout.resume_problem({"said": 4, "rows": 4, "unreadable": [], "pictures": 0}, 4, "x"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(record_layout.dataset_counts(root)["pictures"], 0)
            folder = root / "images" / "observation.images.front" / "episode-000004"
            folder.mkdir(parents=True)
            self.assertEqual(record_layout.dataset_counts(root)["pictures"], 0, "empty folders are not pictures")
            for index in range(3):
                (folder / f"frame-{index:06d}.png").write_bytes(b"x")
            self.assertEqual(record_layout.dataset_counts(root)["pictures"], 3)

    def test_a_placement_log_that_does_not_match_the_set_is_refused(self):
        sound = {"said": 27, "rows": 27, "unreadable": []}
        for lines in (0, 26, 28):                        # the log gone; a line short; a line of another set left in it
            with self.subTest(lines=lines):
                told = record_layout.resume_problem(sound, lines, "clustered_placements.jsonl")
                self.assertIn(f"it holds 27 episodes and the placement log clustered_placements.jsonl has {lines} lines",
                              told)
        self.assertIn("meta/info.json cannot be read",
                      record_layout.resume_problem({"said": None, "rows": 0, "unreadable": []}, 0, "x.jsonl"))

    def test_a_placement_log_s_lines_are_counted_and_no_log_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clustered_placements.jsonl"
            self.assertEqual(record_layout.placement_lines(path), 0)
            path.write_text('{"episode_index": 0}\n{"episode_index": 1}\n\n', encoding="utf-8")
            self.assertEqual(record_layout.placement_lines(path), 2)
            # bytes that are not text are still lines to count: this is asked before a session starts
            path.write_bytes(b'{"episode_index": 0}\n\xff\xfe\x00garbled\n')
            self.assertEqual(record_layout.placement_lines(path), 2)

    def test_a_dataset_s_counts_are_read_from_its_files(self):
        try:
            import pyarrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("needs pyarrow (the rig's interpreter)")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(record_layout.dataset_counts(root),
                             {"said": None, "rows": 0, "unreadable": [], "pictures": 0})
            (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 5}), encoding="utf-8")
            for number, rows in ((0, 1), (1, 2)):
                parquet.write_table(pyarrow.table({"episode_index": list(range(rows))}),
                                    root / "meta" / "episodes" / "chunk-000" / f"file-{number:03d}.parquet")
            parquet.write_table(pyarrow.table({"index": list(range(40))}), root / "data" / "chunk-000" / "file-000.parquet")
            self.assertEqual(record_layout.dataset_counts(root),
                             {"said": 5, "rows": 3, "unreadable": [], "pictures": 0})
            # a table file whose writer was never closed: what was written of it, and no end
            whole = (root / "data" / "chunk-000" / "file-000.parquet").read_bytes()
            (root / "data" / "chunk-000" / "file-001.parquet").write_bytes(whole[: len(whole) // 2])
            counts = record_layout.dataset_counts(root)
            self.assertEqual(counts["unreadable"], ["data/chunk-000/file-001.parquet"])
            self.assertIn("cannot be read", record_layout.resume_problem(counts, 5, "x.jsonl"))


class SetAsideTests(unittest.TestCase):
    """A recorded set and its placement log are moved out of the recorder's way together, and nothing is deleted."""

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        root = Path(self._folder.name)
        self.dataset = root / "home" / "owner" / "tolrad-clustered-v1"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta" / "info.json").write_text('{"total_episodes": 27}', encoding="utf-8")
        self.log = root / "recording" / "clustered_placements.jsonl"
        self.log.parent.mkdir()
        self.log.write_text('{"episode_index": 0}\n' * 27, encoding="utf-8")

    def test_the_folder_and_the_log_get_one_new_name_and_keep_what_they_hold(self):
        moved = record_layout.set_aside(self.dataset, self.log, "attempt2")
        new_dir = self.dataset.with_name("tolrad-clustered-v1-attempt2")
        new_log = self.log.with_name("tolrad-clustered-v1-attempt2_placements.jsonl")
        self.assertEqual(moved, [(self.dataset, new_dir), (self.log, new_log)])
        self.assertFalse(self.dataset.exists() or self.log.exists())
        self.assertEqual(json.loads((new_dir / "meta" / "info.json").read_text())["total_episodes"], 27)
        self.assertEqual(record_layout.placement_lines(new_log), 27)
        # the log's new name is the one a set of that name has for its log
        self.assertEqual(record_layout.placement_log_path("clustered", new_dir.name, self.log.parent), new_log)

    def test_a_name_that_is_taken_renames_nothing(self):
        self.dataset.with_name("tolrad-clustered-v1-attempt1").mkdir()
        with self.assertRaisesRegex(ValueError, "the name is taken"):
            record_layout.set_aside(self.dataset, self.log, "attempt1")
        self.assertTrue(self.dataset.exists() and self.log.exists())
        self.log.with_name("tolrad-clustered-v1-attempt3_placements.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "the name is taken"):
            record_layout.set_aside(self.dataset, self.log, "attempt3")
        self.assertTrue(self.dataset.exists() and self.log.exists())

    def test_a_log_left_without_its_set_is_set_aside_by_itself_and_nothing_at_all_is_refused(self):
        import shutil

        shutil.rmtree(self.dataset)
        # a log with no set beside it is what a mistyped owner name looks like: nothing is renamed without a word
        with self.assertRaisesRegex(ValueError, "is the owner's name right"):
            record_layout.set_aside(self.dataset, self.log, "leftover")
        self.assertTrue(self.log.exists())
        moved = record_layout.set_aside(self.dataset, self.log, "leftover", log_only=True)
        self.assertEqual([old for old, _new in moved], [self.log])
        with self.assertRaisesRegex(ValueError, "nothing to set aside"):
            record_layout.set_aside(self.dataset, self.log, "again")

    def test_a_label_that_is_not_a_plain_word_is_refused(self):
        for label in ("", "../elsewhere", "two words", "-x", "a/b"):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "not a plain word"):
                record_layout.set_aside(self.dataset, self.log, label)
        self.assertTrue(self.dataset.exists() and self.log.exists())


class WristStateTests(unittest.TestCase):
    """The wrist camera has no tag: one number of its rest picture tells its two tone states apart."""

    def setUp(self):
        import numpy as np

        from tolerance_radius.rig import vision

        self.np, self.vision = np, vision

    def test_the_number_is_the_floor_over_the_jaws_green_at_one_floor_level(self):
        vision = self.vision
        # at the reference floor it is the plain ratio
        self.assertAlmostEqual(vision.wrist_contrast(wrist_rest_picture(floor=195.0, jaws_bgr=(14.0, 76.0, 192.0))),
                               195.0 / 76.0, places=3)
        # elsewhere the green is first brought along the usual state's line to that floor
        self.assertAlmostEqual(vision.wrist_contrast(wrist_rest_picture(floor=200.0)),
                               195.0 / (79.0 - 0.659 * 5.0), places=3)
        self.assertEqual((vision.WRIST_SLOPE, vision.WRIST_FLOOR_REFERENCE), (0.659, 195.0))
        self.assertEqual((vision.WRIST_USUAL_READS, vision.WRIST_OTHER_READS), ((2.50, 2.64), (3.08, 3.35)))
        # the line lies between the two, the same share from either
        self.assertEqual(vision.WRIST_OTHER_ABOVE, 2.85)
        self.assertAlmostEqual(vision.WRIST_OTHER_ABOVE / 2.64, 3.08 / vision.WRIST_OTHER_ABOVE, places=2)

    def test_the_stretched_tone_range_moves_it_and_another_exposure_does_not(self):
        usual = self.vision.wrist_contrast(wrist_rest_picture())
        other = self.vision.wrist_contrast(wrist_rest_picture(stretched=True))
        self.assertAlmostEqual(usual, 2.576, places=2)
        self.assertAlmostEqual(other, 3.22, places=2)                # inside what the other state read
        self.assertIsNone(self.vision.wrist_other_state(usual))
        self.assertIn("reads 3.2", self.vision.wrist_other_state(other))
        self.assertIn("2.50 to 2.64 in the usual tone state, 3.08 to 3.35 in the other",
                      self.vision.wrist_other_state(other))
        # The camera's own exposure wandered by 16 levels of the floor in the usual state on 2026-10-06, and the
        # jaws' green went with it along one line. The plain ratio fell by 0.009 a level: a floor 12 levels darker
        # than any seen would have read as the other state. Along the line the number stays where it is.
        for brighter in (-25.0, -12.0, 0.0, 8.0, 20.0):
            with self.subTest(brighter=brighter):
                self.assertAlmostEqual(self.vision.wrist_contrast(wrist_rest_picture(brighter=brighter)), usual,
                                       delta=0.02)
                self.assertGreater(self.vision.wrist_contrast(wrist_rest_picture(brighter=brighter, stretched=True)),
                                   3.08)
        # ... where the plain ratio of the same pictures crosses its old line
        darkest = wrist_rest_picture(brighter=-25.0)
        self.assertGreater(175.0 / (79.0 - 0.659 * 25.0), 2.74)
        self.assertIsNone(self.vision.wrist_other_state(self.vision.wrist_contrast(darkest)))

    def test_the_cube_at_the_top_of_the_picture_does_not_move_it(self):
        self.assertAlmostEqual(self.vision.wrist_contrast(wrist_rest_picture(cube=True)),
                               self.vision.wrist_contrast(wrist_rest_picture()), places=3)

    def test_a_picture_that_is_not_the_rest_picture_is_not_judged(self):
        np = self.np
        cases = {
            "the cover left on": np.zeros((480, 640, 3), np.uint8),
            "no jaws: the arm is elsewhere": np.full((480, 640, 3), 200, np.uint8),
            "less of the jaws in the picture (parked looking up, 2026-10-06 17:23)": wrist_rest_picture(jaw_share=0.055),
            "more orange than the jaws: the cube between them": wrist_rest_picture(jaw_share=0.10),
            "a grey picture": np.full((480, 640), 200, np.uint8),
        }
        nothing_below = wrist_rest_picture()
        nothing_below[180:300, 200:440] = (20, 90, 230)             # no floor in the window above the jaws
        cases["the window above the jaws is not floor"] = nothing_below
        # ... and with the cube over most of that window, the even floor that is left of it is not enough to go by
        mostly_cube = wrist_rest_picture()
        mostly_cube[180:300, 200:350] = (20, 90, 230)               # 150 of the window's 240 columns
        cases["the cube over most of the window"] = mostly_cube
        # Parked looking at the box's wall (16:46 on 2026-10-06): the jaws as at rest, light grey above them, and by
        # chance a number of the usual state. What gives it away is that the grey is not an even floor.
        wall = wrist_rest_picture(floor=215.0, jaws_bgr=(14.0, 84.0, 200.0))
        wall[180:300:6, 200:440] = 40
        wall[180:300, 300:316] = 250
        cases["the window above the jaws is not an even floor"] = wall
        # ... and the jaws lit another way over a plain floor: a number that neither state gives
        cases["a number far below both states"] = wrist_rest_picture(jaws_bgr=(14.0, 120.0, 200.0))
        cases["a number far above both states"] = wrist_rest_picture(jaws_bgr=(14.0, 52.0, 200.0))
        for name, picture in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self.vision.wrist_contrast(picture))
        self.assertIsNone(self.vision.wrist_other_state(None))

    def test_a_change_is_a_reading_on_the_other_side_of_the_line_and_a_step_away(self):
        changed = self.vision.wrist_changed
        self.assertIn("reads 3.20 now and read 2.55 then", changed(2.55, 3.20))
        self.assertIn("reads 2.55 now and read 3.20 then", changed(3.20, 2.55))     # back to the usual is a change too
        for start, now in ((2.50, 2.64), (3.08, 3.35), (3.35, 3.08),              # one state, its own scatter
                           (2.80, 2.90),                                           # across the line by a hair
                           (2.55, 2.21),                                           # elsewhere, on the same side
                           (None, 3.20), (2.55, None), (None, None)):              # not judged: nothing is said
            with self.subTest(start=start, now=now):
                self.assertIsNone(changed(start, now))
        # the nearest two frames of different states lay 16.7 % apart, the furthest two of one state 8.8 %
        self.assertIsNotNone(changed(2.64, 3.08))
        self.assertLess(3.35 / 3.08, self.vision.WRIST_STEP)
        self.assertLess(self.vision.WRIST_STEP, 3.08 / 2.64)


class StartPoseTests(unittest.TestCase):
    """The reset is over only with the arm back about where it rested when the empty table was pictured."""

    #: The arm's rest at the start of sitting 1 (2026-10-06), and where it rested at the end of that sitting's
    #: three takes: *read* from the dataset's first and last frames.
    START = {"shoulder_pan": -4.1, "shoulder_lift": -102.9, "elbow_flex": 96.7, "wrist_flex": 75.4,
             "wrist_roll": -93.8, "gripper": 0.6}
    AFTER = [
        {"shoulder_pan": -3.0, "shoulder_lift": -102.9, "elbow_flex": 96.4, "wrist_flex": 75.7, "wrist_roll": -89.7,
         "gripper": 0.8},
        {"shoulder_pan": 1.7, "shoulder_lift": -102.7, "elbow_flex": 96.7, "wrist_flex": 74.9, "wrist_roll": -96.7,
         "gripper": 0.8},
        {"shoulder_pan": -3.0, "shoulder_lift": -102.9, "elbow_flex": 96.8, "wrist_flex": 74.8, "wrist_roll": -100.4,
         "gripper": 0.8},
    ]

    def test_at_five_degrees_for_every_joint_two_of_sitting_1_s_three_resets_were_asked_for_again(self):
        away = [record_layout.off_start_pose(self.START, now, 5.0) for now in self.AFTER]
        self.assertEqual(away, [[], ["shoulder_pan +5.8 deg (allowed 5)"], ["wrist_roll -6.6 deg (allowed 5)"]])

    def test_the_two_joints_the_folded_arm_leaves_free_have_their_own_tolerance(self):
        self.assertEqual(record_layout.FREE_JOINTS, ("shoulder_pan", "wrist_roll"))
        for now in self.AFTER:
            self.assertEqual(record_layout.off_start_pose(self.START, now, 5.0, 12.0), [])
        # ... and it ends: a wrist turned further than that is asked back
        turned = dict(self.START, wrist_roll=-108.0)
        self.assertEqual(record_layout.off_start_pose(self.START, turned, 5.0, 12.0),
                         ["wrist_roll -14.2 deg (allowed 12)"])
        swivelled = dict(self.START, shoulder_pan=9.0)
        self.assertEqual(record_layout.off_start_pose(self.START, swivelled, 5.0, 12.0),
                         ["shoulder_pan +13.1 deg (allowed 12)"])

    def test_a_joint_the_folded_arm_holds_is_still_held_to_the_tighter_tolerance(self):
        # An arm left with its wrist not folded reaches into the region where the cube is looked for.
        for joint in ("shoulder_lift", "elbow_flex", "wrist_flex", "gripper"):
            with self.subTest(joint=joint):
                now = dict(self.START)
                now[joint] += 6.4
                self.assertEqual(record_layout.off_start_pose(self.START, now, 5.0, 12.0),
                                 [f"{joint} +6.4 deg (allowed 5)"])

    def test_the_check_and_its_free_part_can_each_be_switched_off(self):
        far = {joint: value + 40.0 for joint, value in self.START.items()}
        self.assertEqual(record_layout.off_start_pose(self.START, far, 0.0, 12.0), [])
        only_the_held = record_layout.off_start_pose(self.START, far, 5.0, 0.0)
        self.assertEqual(len(only_the_held), 4)
        self.assertFalse(any(joint in told for told in only_the_held for joint in record_layout.FREE_JOINTS))

    def test_a_joint_the_arm_does_not_report_is_passed_over(self):
        self.assertEqual(record_layout.off_start_pose(self.START, {"shoulder_pan": -4.0}, 5.0, 12.0), [])

    def test_the_recorder_s_own_tolerances(self):
        args = record_layout.build_parser().parse_args(["--sitting", "1"])
        self.assertEqual((args.start_pose_tolerance, args.free_joint_tolerance), (5.0, 12.0))


class StillTakeTests(unittest.TestCase):
    """A take in which the arm did not move is not a demonstration: nobody was at the leader."""

    NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
             "gripper.pos"]

    @staticmethod
    def dataset(states, names=None):
        """A dataset whose writer holds a take's joint readings as LeRobot's does: a list, one entry per frame.
        With `names`, it names its joints as LeRobot's dataset does (the recorded sets': `NAMES`)."""
        class Writer:
            episode_buffer = {"size": len(states), "task": ["t"] * len(states), "observation.state": states}

        class Dataset:
            writer = Writer()

        if names is not None:
            Dataset.features = {"observation.state": {"dtype": "float32", "names": list(names), "shape": [len(names)]}}
        return Dataset()

    def test_the_travel_is_asked_of_the_joints_the_folded_arm_holds(self):
        # The review of 2026-10-06: a hand resting on the leader swivels the base, turns the wrist and squeezes the
        # gripper by more than the limit, and nothing else moves. That was a demonstration by the rule as it was.
        rest = [-4.1, -102.9, 96.7, 75.4, -93.8, 0.6]
        fidget = [[-4.1 + 40.0, -102.5, 96.9, 75.1, -93.8 + 35.0, 60.0], rest, [-4.1 - 5.0, -102.9, 96.7, 76.6, -93.8, 0.6]]
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset(fidget, self.NAMES)), 1.5)
        self.assertTrue(record_layout.stood_still(record_layout.joint_travel(self.dataset(fidget, self.NAMES)), 30.0))
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset(fidget)), 59.4)   # unnamed joints: all of them
        # a demonstration is still one: the lift goes out and comes back
        out = [56.0, 17.1, 20.0, 40.0, -80.0, 45.0]
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset([rest, out, rest], self.NAMES)), 120.0)
        # names that are not the arm's, or not as many as the readings: all the joints, as before
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset(fidget, ["a", "b", "c", "d", "e", "f"])), 59.4)
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset(fidget, self.NAMES[:5])), 59.4)
        self.assertEqual(record_layout.HELD_JOINTS, ("shoulder_lift", "elbow_flex", "wrist_flex"))

    def test_the_travel_is_the_largest_distance_any_one_joint_covered(self):
        rest = [-4.1, -102.9, 96.7, 75.4, -93.8, 0.6]
        # a demonstration: the shoulder lift goes out 120 degrees and comes back; the pan 60
        out = [56.0, 17.1, 20.0, 40.0, -80.0, 45.0]
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset([rest, out, rest])), 120.0)
        # an arm at rest with a hand on the leader: a degree here, two there
        idle = [[v + d for v, d in zip(rest, (0.0, 0.1, 0.0, 0.3, 1.9, 0.2), strict=True)], rest,
                [v - d for v, d in zip(rest, (1.2, 0.0, 0.1, 0.0, 0.4, 0.0), strict=True)]]
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset(idle)), 2.3)
        self.assertEqual(record_layout.joint_travel(self.dataset([rest])), 0.0)

    def test_numpy_readings_are_read_as_the_recorder_gets_them(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("needs numpy")
        states = [np.array([0.0, -100.0, 96.0, 75.0, -90.0, 1.0], dtype=np.float32) + step for step in (0, 20, 50, 10)]
        self.assertAlmostEqual(record_layout.joint_travel(self.dataset(states)), 50.0)

    def test_a_dataset_that_cannot_say_gives_no_travel(self):
        class Bare:
            pass

        class NoBuffer:
            writer = Bare()

        for dataset in (Bare(), NoBuffer(), self.dataset([]), self.dataset([[1.0, "x"]]), self.dataset([1.0, 2.0])):
            with self.subTest(dataset=type(dataset).__name__):
                self.assertIsNone(record_layout.joint_travel(dataset))

    def test_only_a_known_travel_under_the_line_is_a_still_take(self):
        self.assertTrue(record_layout.stood_still(4.0, 30.0))
        self.assertFalse(record_layout.stood_still(105.3, 30.0))       # the least any recorded demonstration moved
        self.assertFalse(record_layout.stood_still(30.0, 30.0))
        self.assertFalse(record_layout.stood_still(None, 30.0))        # not known: the take is kept, as before
        self.assertFalse(record_layout.stood_still(4.0, 0.0))          # switched off

    def test_the_recorder_s_own_line(self):
        args = record_layout.build_parser().parse_args(["--sitting", "1"])
        self.assertEqual((args.min_travel, record_layout.MIN_TRAVEL_DEG), (30.0, 30.0))


# -------------------------------------------------------------------------------------------------- the trial runner


def demos() -> list[DemoStart]:
    return [
        DemoStart("clustered", 0, "B1", measured=Point(75.0, 25.0)),
        DemoStart("clustered", 1, "C2", measured=Point(125.0, 75.0)),
        DemoStart("clustered", 2, "C2", measured=Point(126.0, 75.0), trained=False),
        DemoStart("spread", 0, "B1", measured=Point(60.0, 30.0)),
    ]


def planned(**changes) -> PlannedTrial:
    base = PlannedTrial(trial_id="T1a-001", block="T1a", condition="act_clustered", blind_code="K7",
                        target=Point(85.0, 25.0), level_mm=10.0, staircase_id="up", staircase_index=0)
    return replace(base, **changes)


def record_of(trial: PlannedTrial, outcome: str = "success") -> TrialRecord:
    common = dict(planned=trial, ts_start="2026-10-07T09:00:00-07:00", ts_end="2026-10-07T09:00:30-07:00",
                  policy_repo=None if trial.condition == "replay" else trial.condition,
                  checkpoint=None if trial.condition == "replay" else "080000")
    if outcome == "void":
        return TrialRecord(outcome="void", void_reason="test", **common)
    return TrialRecord(outcome=outcome, placed=trial.target, nearest_demo_episode=0, nearest_demo_distance_mm=1.0,
                       subgoal=5 if outcome == "success" else 1, failure=None if outcome == "success" else "grasp",
                       **common)


class AuditOfTheSetTests(unittest.TestCase):
    """A trial session measures every distance from the audited starts and replays demonstrations by their number:
    the audit must be the whole set's, and the replay's set the audited one."""

    @staticmethod
    def starts(indices, dataset="clustered"):
        return [DemoStart(dataset, index, "B1", measured=Point(75.0, 25.0)) for index in indices]

    def test_an_audit_is_whole_when_it_holds_every_episode_by_its_number_not_by_their_count(self):
        gaps = trial_session.audit_gaps
        self.assertEqual(gaps(self.starts(range(55)), {"clustered": 55}), [])
        self.assertEqual(gaps(self.starts(range(33)), {"clustered": 55}),
                         ["the clustered audit holds 33 of the set's 55 episodes"])
        # 55 starts, and not the set's 55: one twice over, or one from elsewhere
        self.assertEqual(len(gaps(self.starts([*range(54), 60]), {"clustered": 55})), 1)
        self.assertEqual(len(gaps(self.starts([*range(20), *range(21, 55)]), {"clustered": 55})), 1)

    def test_the_replay_s_set_is_the_audited_one_when_it_ends_where_the_audit_ends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tolrad-clustered-v1"
            (root / "meta").mkdir(parents=True)
            problem = trial_session.replay_set_problem
            self.assertIn("cannot be read", problem(root, self.starts(range(55))))
            (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 55}), encoding="utf-8")
            self.assertIsNone(problem(root, self.starts(range(55))))
            # a start the audit could not measure is a gap in the audit, not another recording (it said so of a
            # whole set with one unmeasured start, in a pilot too: the review of the night of 2026-10-06)
            self.assertIsNone(problem(root, self.starts([*range(20), *range(21, 55)])))
            # an audit of a part of the set, and a set larger than the audit
            self.assertIn("holds 55 episodes and the clustered audit ends at episode 32", problem(root, self.starts(range(33))))
            (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 27}), encoding="utf-8")
            self.assertIn("holds 27 episodes and the clustered audit ends at episode 54", problem(root, self.starts(range(55))))
            # the spread set's starts say nothing of the replay's set
            self.assertIn("ends at episode -1", problem(root, self.starts(range(55), dataset="spread")))


class PlanningTests(unittest.TestCase):
    def test_planned_totals_match_the_protocol_text(self):
        # The trial budget of protocol/protocol.md, section 8: a block's row gives its total in bold after a colon, and
        # the sentence under the table gives the sum. The parameter file and the text must plan the same numbers.
        text = (REPO_ROOT / "protocol" / "protocol.md").read_text(encoding="utf-8")
        rows = re.findall(r"(?m)^\| \*\*(T\w+)\*\* \|[^|]*\|[^|]*: \*\*(\d+)\*\* \|", text)
        stated = {block: int(total) for block, total in rows}
        totals = {name: trial_session.planned_total(block) for name, block in PROTOCOL["blocks"].items()}
        self.assertEqual(stated, totals, "the budget table of protocol/protocol.md and config/protocol.toml disagree")
        self.assertEqual(len(rows), len(totals), "a block has two rows in the budget table")
        self.assertIn(f"Total {sum(totals.values())} counted trials.", text)

    def test_block_conditions_come_from_either_form(self):
        self.assertEqual(trial_session.block_conditions(PROTOCOL["blocks"]["T1b"]), ["replay"])
        self.assertEqual(trial_session.block_conditions(PROTOCOL["blocks"]["T2"]), ["act_clustered", "act_spread"])

    def test_the_interleaved_order_holds_every_planned_trial_and_never_changes(self):
        order = trial_session.block_order(["T1a", "T1b"], PROTOCOL)
        self.assertEqual((order.count("T1a"), order.count("T1b")), (34, 22))
        self.assertEqual(order, trial_session.block_order(["T1a", "T1b"], PROTOCOL))
        self.assertNotEqual(order, sorted(order))  # interleaved, not one block after the other

    def test_one_block_is_asked_directly(self):
        asked = []

        def plan(block, *_):
            asked.append(block)
            return planned()

        self.assertEqual(trial_session.next_planned(["T1a"], PROTOCOL, demos(), [], plan=plan), planned())
        self.assertEqual(asked, ["T1a"])

    def test_interleaved_blocks_follow_the_order_and_void_trials_do_not_advance_it(self):
        order = trial_session.block_order(["T1a", "T1b"], PROTOCOL)

        def plan(block, *_):
            condition = "replay" if block == "T1b" else "act_clustered"
            return planned(block=block, trial_id=f"{block}-x", condition=condition,
                           replay_episode=0 if condition == "replay" else None)

        first = trial_session.next_planned(["T1a", "T1b"], PROTOCOL, demos(), [], plan=plan)
        self.assertEqual(first.block, order[0])
        void_only = [record_of(first, "void")]
        self.assertEqual(trial_session.next_planned(["T1a", "T1b"], PROTOCOL, demos(), void_only, plan=plan).block,
                         order[0])
        one_scored = [record_of(first, "success")]
        self.assertEqual(trial_session.next_planned(["T1a", "T1b"], PROTOCOL, demos(), one_scored, plan=plan).block,
                         order[1])

    def test_when_the_due_block_is_complete_the_other_one_is_asked_and_then_nothing(self):
        def only_t1b(block, *_):
            return planned(block="T1b", trial_id="T1b-x", condition="replay", replay_episode=0) if block == "T1b" else None

        self.assertEqual(trial_session.next_planned(["T1a", "T1b"], PROTOCOL, demos(), [], plan=only_t1b).block, "T1b")
        self.assertIsNone(trial_session.next_planned(["T1a", "T1b"], PROTOCOL, demos(), [], plan=lambda *_: None))


class RecordTests(unittest.TestCase):
    COMMON = dict(ts_start="2026-10-07T09:00:00-07:00", ts_end="2026-10-07T09:00:30-07:00", operator="op1",
                  scorer="sc1", execution={"clamp_deg": 15.0}, versions={"protocol_version": "0.1"}, media={})

    def build(self, trial=None, **changes):
        trial = trial or planned()
        policy = None if trial.condition == "replay" else trial.condition
        values = dict(placed=Point(85.0, 25.0), demo_starts=demos(), score={"subgoal": 5, "failure": None, "notes": ""},
                      void_reason=None, policy_label=policy, checkpoint=None if policy is None else "080000")
        return trial_session.build_record(trial, **(self.COMMON | values | changes))

    def test_a_success_is_recorded_with_its_distance_to_the_nearest_trained_demonstration(self):
        record = self.build()
        self.assertTrue(record.success)
        self.assertEqual(record.nearest_demo_episode, 0)
        self.assertAlmostEqual(record.nearest_demo_distance_mm, 10.0)

    def test_an_untrained_demonstration_is_never_the_nearest(self):
        record = self.build(placed=Point(126.0, 75.0))
        self.assertEqual(record.nearest_demo_episode, 1)
        self.assertAlmostEqual(record.nearest_demo_distance_mm, 1.0)

    def test_a_failure_without_a_label_is_marked_and_not_guessed(self):
        record = self.build(score={"subgoal": 2, "failure": None, "notes": "dropped"})
        self.assertEqual((record.outcome, record.failure, record.notes), ("failure", trial_session.UNLABELLED, "dropped"))

    def test_a_replay_is_measured_to_the_demonstration_it_replays(self):
        replay = planned(block="T1b", trial_id="T1b-001", condition="replay", replay_episode=1)
        record = self.build(replay, placed=Point(75.0, 25.0))
        self.assertEqual(record.nearest_demo_episode, 1)  # not episode 0, although the cube sits exactly on it
        self.assertAlmostEqual(record.nearest_demo_distance_mm, Point(75.0, 25.0).distance_to(Point(125.0, 75.0)))
        self.assertIsNone(record.checkpoint)

    def test_a_refused_start_an_unscored_trial_and_a_declared_void_are_all_void(self):
        for changes, reason in (
            ({"void_reason": "the handoff gate refused the start pose", "score": None}, "handoff"),
            ({"score": None}, "unscored"),
            ({"void_reason": "bumped the table"}, "bumped"),
        ):
            with self.subTest(reason=reason):
                record = self.build(**changes)
                self.assertEqual(record.outcome, "void")
                self.assertIn(reason, record.void_reason)
                self.assertFalse(record.scored)

    def test_a_record_survives_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trials.jsonl"
            schema.append_trial(log, self.build())
            self.assertEqual(schema.read_trials(log), [self.build()])


class LogRulesTests(unittest.TestCase):
    FROZEN = {"protocol": {"version": "1.0", "status": "frozen"}}
    DRAFT = {"protocol": {"version": "0.1", "status": "draft"}}
    PILOT = {"protocol": {"version": "pilot", "status": "pilot"}}

    def problem(self, protocol, name, log=(), pilot=False):
        return trial_session.log_problem(protocol, Path("data/trials") / name, list(log), pilot=pilot)

    def test_counted_trials_need_a_frozen_protocol(self):
        self.assertIn("frozen", self.problem(self.DRAFT, "main.jsonl"))
        self.assertIsNone(self.problem(self.FROZEN, "main.jsonl"))

    def test_pilot_trials_go_to_a_pilot_file_and_counted_trials_never_do(self):
        self.assertIsNone(self.problem(self.PILOT, "pilot_replay.jsonl", pilot=True))
        self.assertIn("pilot", self.problem(self.PILOT, "main.jsonl", pilot=True))
        self.assertIn("--pilot", self.problem(self.FROZEN, "pilot_replay.jsonl"))

    def test_a_log_never_mixes_protocol_versions(self):
        old = replace(record_of(planned()), versions={"protocol_version": "0.9"})
        self.assertIn("0.9", self.problem(self.FROZEN, "main.jsonl", log=[old]))
        same = replace(record_of(planned()), versions={"protocol_version": "1.0"})
        self.assertIsNone(self.problem(self.FROZEN, "main.jsonl", log=[same]))


class ScoreSheetTests(unittest.TestCase):
    def test_answers_pass_through_until_void_is_typed(self):
        answers = iter(["5", "", "clean"])
        sheet = trial_session.ScoreSheet(lambda _prompt: next(answers))
        self.assertEqual([sheet("score"), sheet("failure"), sheet("notes")], ["5", "", "clean"])
        self.assertIsNone(sheet.void_reason)

    def test_void_with_a_reason_ends_the_prompts_without_asking_again(self):
        asked = []

        def ask(prompt):
            asked.append(prompt)
            return "void: bumped the table"

        sheet = trial_session.ScoreSheet(ask)
        self.assertEqual([sheet("score"), sheet("score"), sheet("score")], ["", "", ""])
        self.assertEqual(sheet.void_reason, "bumped the table")
        self.assertEqual(len(asked), 1)

    def test_void_alone_still_carries_a_reason(self):
        sheet = trial_session.ScoreSheet(lambda _prompt: "void")
        sheet("score")
        self.assertTrue(sheet.void_reason)


class SmallHelperTests(unittest.TestCase):
    def test_a_checkpoint_label_carries_no_account_or_machine_name(self):
        self.assertEqual(trial_session.checkpoint_label("/Users/someone/models/ns/run/080000/pretrained_model"), "080000")
        self.assertEqual(trial_session.checkpoint_label("/tmp/act-final"), "act-final")

    def test_a_checkpoint_label_names_the_run_when_the_folders_show_it(self):
        label = trial_session.checkpoint_label
        downloaded = "data/checkpoints/act-tolrad-clustered-v1/checkpoints/080000/pretrained_model"
        self.assertEqual(label(downloaded), "act-tolrad-clustered-v1@080000")
        self.assertEqual(label("/home/someone/ns/act-tolrad-spread-v1-s2/checkpoints/040000"),
                         "act-tolrad-spread-v1-s2@040000")
        # The owner's folder above the run is never part of it.
        self.assertNotIn("/", label("/home/someone/ns/act-tolrad-spread-v1-s2/checkpoints/040000"))

    def test_a_missing_audit_says_what_to_run(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(FileNotFoundError, "audit_dataset"):
            trial_session.read_audits(["clustered"], Path(directory))


# ------------------------------------------------------------------------------------------------------- the audit


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class AuditTests(unittest.TestCase):
    def setUp(self):
        from tolerance_radius.rig import audit
        from tolerance_radius.rig.vision import TableView

        self.audit = audit
        self.view = TableView(PX_TO_MM)
        self.grid = Grid.from_protocol(PROTOCOL)
        self.planned = layouts.read_layout(LAYOUT_DIR / "clustered.json")

    def frames(self, moved=None, missing=(), dropped=()):
        moved = moved or {}
        result = []
        for episode in self.planned:
            if episode.episode_index in dropped:
                continue
            if episode.episode_index in missing:
                result.append((episode.episode_index, scene()))
                continue
            where = moved.get(episode.episode_index, episode.target)
            result.append((episode.episode_index, scene(where)))
        return result

    def run_audit(self, **kwargs):
        return self.audit.audit_frames("clustered", self.frames(**kwargs), self.planned, self.view, self.grid)

    def test_a_dataset_recorded_as_planned_is_clean_and_every_start_is_measured(self):
        result = self.run_audit()
        self.assertTrue(result.clean, result.problems)
        self.assertEqual(len(result.starts), 55)
        self.assertLess(result.max_error_mm, 1.0)
        self.assertTrue(all(start.trained and start.planned is not None for start in result.starts))

    def test_a_cube_off_its_target_is_reported_with_its_distance(self):
        target = self.planned[7].target
        result = self.run_audit(moved={7: Point(target.x_mm + 6.0, target.y_mm)})
        self.assertFalse(result.clean)
        self.assertTrue(any("episode 7" in p and "from its target" in p for p in result.problems), result.problems)
        self.assertAlmostEqual(result.errors_mm[7], 6.0, delta=0.6)
        self.assertEqual(len(result.starts), 55)  # measured where it was, not dropped

    def test_an_episode_with_no_cube_in_view_is_a_problem_not_a_missing_row(self):
        result = self.run_audit(missing=(3,))
        self.assertTrue(any("episode 3" in p and "no single clean cube" in p for p in result.problems))
        self.assertEqual(len(result.starts), 54)

    def test_a_cube_in_the_wrong_cell_or_a_held_out_cell_is_named(self):
        held_out = self.grid.cell_centre("B2")
        result = self.run_audit(moved={0: held_out})
        text = " | ".join(result.problems)
        self.assertIn("held-out cell B2", text)
        self.assertIn("planned for cell", text)

    def test_a_dataset_with_a_missing_episode_does_not_match_the_plan(self):
        result = self.run_audit(dropped=(54,))
        self.assertTrue(any("do not match the plan" in p for p in result.problems))


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class AuditThroughTheTagsTests(unittest.TestCase):
    """The audit on pictures of the rig's own view, with the floor tags in them and a camera that moves.

    On 2026-10-06 the camera's picture stood 2 mm from its calibration within the hour, the floor tags where they
    had been. Measured through one conversion, every start recorded after that would have been 2 mm off.
    """

    def setUp(self):
        import cv2
        import numpy as np
        from fixtures_pictures import RIG_VIEW, to_px
        from test_rig_registration import picture, shifted

        from tolerance_radius.rig import audit
        from tolerance_radius.rig.vision import TableView, floor_levels, grid_probes_px

        from test_rig_registration import register

        self.audit, self.np, self.cv2 = audit, np, cv2
        self.rig_view, self.shifted, self.picture, self.to_px = RIG_VIEW, shifted, picture, to_px
        # The table geometry as the rig has it: the floor tags' corners as a registration stored them, through the
        # sheet's frame. (Their true corners would do worse: a tag's outline is read a few tenths of a pixel off by
        # an amount that the registration takes into what it stores.)
        self.geometry = register(picture()).geometry
        self.assertEqual(sorted(self.geometry["tags"]), ["0", "1"])
        self.grid = Grid.from_protocol(PROTOCOL)
        self.planned = layouts.read_layout(LAYOUT_DIR / "clustered.json")[:8]
        empty = picture(RIG_VIEW, sheet_ids=())
        self.view = TableView.from_frames([empty] * 3, self.geometry)      # the stored calibration
        quads = [to_px(RIG_VIEW, tag["corners"]) for tag in self.geometry["tags"].values()]
        self.view.floor_levels = floor_levels(empty, grid_probes_px(self.view, self.geometry)[:4], quads)

    def frame(self, where: Point, view=None):
        """The floor with its two tags seen through `view`, and a cube's footprint at `where`."""
        view = self.rig_view if view is None else view
        image = self.picture(view, sheet_ids=())
        half = 15.0
        foot = [(where.x_mm - half, where.y_mm - half), (where.x_mm + half, where.y_mm - half),
                (where.x_mm + half, where.y_mm + half), (where.x_mm - half, where.y_mm + half)]
        self.cv2.fillConvexPoly(image, self.np.round(self.to_px(view, foot)).astype(self.np.int32), (0, 128, 255))
        return image

    def frames(self, moved_from: int = 4, dx: float = 0.0, dy: float = 1.4):
        """The first frames of eight episodes; from episode `moved_from` on the camera's picture has moved."""
        later = self.shifted(self.rig_view, dx=dx, dy=dy)
        return [(episode.episode_index, self.frame(episode.target, later if place >= moved_from else None))
                for place, episode in enumerate(self.planned)]

    def test_every_start_is_measured_through_the_tags_of_its_own_frame(self):
        frames = self.frames()
        result = self.audit.audit_frames("clustered", frames, self.planned, self.view, self.grid, geometry=self.geometry)
        self.assertTrue(result.clean, result.problems)
        self.assertEqual(len(result.starts), 8)
        self.assertLess(result.max_error_mm, 0.8)
        # Through the one stored conversion the starts recorded after the camera moved are off by its movement.
        through_one = self.audit.audit_frames("clustered", frames, self.planned, self.view, self.grid)
        late = [through_one.errors_mm[episode.episode_index] for episode in self.planned[4:]]
        self.assertGreater(min(late), 1.2)
        # What the camera did is written down for every episode, and said.
        moved = [result.camera[episode.episode_index]["camera_moved_mm"] for episode in self.planned]
        self.assertLess(max(moved[:4]), 0.5)
        self.assertGreater(min(moved[4:]), 1.2)
        self.assertEqual(result.view_disagreement_mm, max(moved))
        self.assertIn("mm from the stored calibration's over the episodes", result.notes[0])
        self.assertIn("grey", result.camera[self.planned[0].episode_index])
        self.assertIsNotNone(result.tag_noise_px)
        self.assertLess(result.tag_noise_px, 0.2)

    def test_a_camera_that_jumped_is_not_averaged_across_the_jump(self):
        # Episodes either side of a jump lie in two views: an episode's tags are averaged only with neighbours in
        # its own.
        frames = self.frames(moved_from=4, dy=3.0)
        result = self.audit.audit_frames("clustered", frames, self.planned, self.view, self.grid, geometry=self.geometry)
        self.assertLess(result.max_error_mm, 0.8)
        used = [result.camera[episode.episode_index]["frames"] for episode in self.planned]
        self.assertEqual(used, [3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 3.0])

    def test_an_episode_whose_tag_is_hidden_is_measured_through_its_neighbour_s_view_and_named(self):
        frames = self.frames(moved_from=99)
        index, frame = frames[2]
        corners = self.np.round(self.to_px(self.rig_view, self.geometry["tags"]["1"]["corners"])).astype(int)
        (x0, y0), (x1, y1) = corners.min(axis=0) - 6, corners.max(axis=0) + 6
        frame[y0:y1, x0:x1] = frame[5, 5]                      # a sheet of paper over tag 1
        result = self.audit.audit_frames("clustered", frames, self.planned, self.view, self.grid, geometry=self.geometry)
        self.assertEqual(len(result.starts), 8)
        self.assertTrue(any(f"episodes [{index}]" in problem and "nearest episode" in problem for problem in result.problems),
                        result.problems)
        self.assertNotIn(index, result.camera)
        self.assertLess(result.errors_mm[index], 0.8)

    def test_frames_are_read_as_the_reader_gives_them(self):
        # Until the evening of 2026-10-06 the audit "undid a stretch by LeRobot's readers". There is none: the videos
        # it had been measured on were recorded while the camera's stream was in a stretched state. Put through that
        # undoing, a recording in the usual state came out with the floor at 188 for 201 and was "another picture".
        from unittest import mock

        given = [(0, self.frame(self.planned[0].target)), (1, self.frame(self.planned[1].target))]
        loader = mock.Mock(_load_episode_first_frames=lambda repo_id, camera: iter(given))
        with mock.patch.object(self.audit.lerobot101, "load", lambda name: loader):
            read = self.audit.first_frames("owner/set")
        self.assertEqual([index for index, _ in read], [0, 1])
        for (_, frame), (_, original) in zip(read, given, strict=True):
            self.assertTrue(self.np.array_equal(frame, original))
        self.assertFalse(hasattr(self.audit, "as_the_camera_gave"))

    # ------------------------------------------------------------------------------------- inside the takes

    def take(self, episode: int, *frames):
        """One take's frames as the audit looks at them: one a second, and the last."""
        return [(episode, 30 * place, frame) for place, frame in enumerate(frames)]

    def test_a_take_recorded_in_one_picture_has_nothing_to_say(self):
        usual = self.frame(self.planned[0].target)
        crept = self.frame(self.planned[0].target, self.shifted(self.rig_view, dx=0.0, dy=0.7))   # this rig's camera
        seen, problems = self.audit.takes_through(self.take(4, usual, usual, crept, crept), self.geometry)
        self.assertEqual(problems, [])
        self.assertEqual((seen[4]["frames"], seen[4]["frames_with_tags"]), (4, 4))
        self.assertLess(seen[4]["tags_moved_px"], 1.5)

    def test_a_camera_that_changes_its_picture_inside_a_take_is_named(self):
        # Episode 3 of the set that was put aside: its first 101 frames as they should be, then the camera on its
        # own zoom to the take's end. The start-state audit saw nothing: a take's first frame is all it looked at.
        usual = self.frame(self.planned[0].target)
        jumped = self.frame(self.planned[0].target, self.shifted(self.rig_view, dx=40.0, dy=30.0))
        frames = self.take(2, usual, usual, usual) + self.take(3, usual, jumped, jumped, jumped)
        seen, problems = self.audit.takes_through(frames, self.geometry)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("episodes [3]: the front camera's picture changes inside the take", problems[0])
        self.assertGreater(seen[3]["tags_moved_px"], 40.0)
        self.assertLess(seen[2]["tags_moved_px"], 0.5)

    def test_a_tone_range_that_changes_inside_a_take_is_named(self):
        # the stream gone into its stretched state: the tags' black at 19 for 32, and nothing has moved
        raw = self.frame(self.planned[0].target)
        # (levels at which this fixture's small tags read a black of about 37, near the rig's 32)
        usual = (18.0 + raw.astype(float) * 0.4).astype(self.np.uint8)
        stretched = self.np.clip((usual.astype(float) - 16.0) * 255.0 / 219.0, 0, 255).astype(self.np.uint8)
        seen, problems = self.audit.takes_through(self.take(5, usual, usual, stretched, stretched), self.geometry)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("episodes [5]: the front camera's tone range changes inside the take", problems[0])
        self.assertLess(seen[5]["black_low"], 0.7)
        self.assertLess(seen[5]["tags_moved_px"], 1.0)

    def test_a_frame_in_which_the_arm_hides_a_tag_is_passed_over(self):
        usual = self.frame(self.planned[0].target)
        hidden = usual.copy()
        corners = self.np.round(self.to_px(self.rig_view, self.geometry["tags"]["0"]["corners"])).astype(int)
        x0, y0 = corners.min(axis=0) - 8
        x1, y1 = corners.max(axis=0) + 8
        hidden[y0:y1, x0:x1] = (0, 128, 255)                      # the arm, over the tag
        seen, problems = self.audit.takes_through(self.take(6, usual, hidden, hidden, usual), self.geometry)
        self.assertEqual(problems, [])
        self.assertEqual((seen[6]["frames"], seen[6]["frames_with_tags"]), (4, 2))

    def test_a_take_whose_tags_are_never_in_sight_is_said_and_not_passed(self):
        blank = self.np.full_like(self.frame(self.planned[0].target), 200)
        _seen, problems = self.audit.takes_through(self.take(7, blank, blank, blank), self.geometry)
        self.assertEqual(len(problems), 1)
        self.assertIn("episodes [7]", problems[0])
        self.assertIn("cannot be said", problems[0])

    # ------------------------------------------------------------------------------------- the wrist camera

    def test_the_wrist_camera_s_other_state_is_named_with_its_episodes(self):
        # 2026-10-06, the first 27 episodes of the clustered set: six in the usual state, 21 in the other.
        other = {0, 7, 8, 9, 15, 16}
        frames = [(e, wrist_rest_picture(floor=198.0 + e % 4, stretched=e in other)) for e in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16)]
        levels, problems = self.audit.wrist_states(frames)
        self.assertAlmostEqual(levels[1]["contrast"], 195.0 / (79.0 - 0.659 * 4.0), places=2)
        self.assertGreater(levels[7]["contrast"], 3.08)
        self.assertIn("brightest", levels[1])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("in its other tone state in 6 of 12 episodes", problems[0])
        self.assertIn("[0, 7, 8, 9, 15, 16]", problems[0])
        self.assertRegex(problems[0], r"3\.\d\d to 3\.\d\d there, 2\.\d\d to 2\.\d\d in the rest")
        self.assertIn("the line is 2.85", problems[0])

    def test_a_set_recorded_wholly_in_the_other_state_is_named_all_the_same(self):
        _levels, problems = self.audit.wrist_states([(e, wrist_rest_picture(stretched=True)) for e in range(5)])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("in 5 of 5 episodes", problems[0])

    def test_one_state_at_several_exposures_is_not_two(self):
        # The usual state's brightest hundredth read 204 to 210 at the takes' starts and 196 to 201 at their ends:
        # a rule that goes by that number alone names the darker ones.
        frames = [(e, wrist_rest_picture(brighter=by)) for e, by in enumerate((-12.0, -6.0, 0.0, 3.0, 16.0))]
        levels, problems = self.audit.wrist_states(frames)
        self.assertEqual(problems, [])
        self.assertLess(max(v["contrast"] for v in levels.values()) / min(v["contrast"] for v in levels.values()), 1.02)
        self.assertEqual(self.audit.wrist_states([]), ({}, []))

    def test_first_frames_that_are_not_the_rest_picture_are_said_not_judged(self):
        away = self.np.full((480, 640, 3), 200, self.np.uint8)                 # no jaws where they rest
        levels, problems = self.audit.wrist_states([(0, wrist_rest_picture()), (1, away), (2, away)])
        self.assertIsNone(levels[1]["contrast"])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("could not be judged in 2 of 3 episodes", problems[0])
        self.assertIn("[1, 2]", problems[0])

    def test_two_groups_on_one_side_of_the_line_are_still_said(self):
        # another rest pose than the one the line was measured in: the number sits elsewhere, a step is still a step
        frames = [(e, wrist_rest_picture(jaws_bgr=(14.0, 87.0, 192.0), stretched=e >= 3)) for e in range(6)]
        levels, problems = self.audit.wrist_states(frames)
        self.assertLess(max(v["contrast"] for v in levels.values()), 2.85)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("fall into two groups", problems[0])
        self.assertIn("[0, 1, 2]", problems[0])
        self.assertIn("[3, 4, 5]", problems[0])

    def test_episodes_recorded_in_another_picture_state_are_named(self):
        # The same picture handed over stretched from video range to full range: a state the camera's stream was in
        # for every recording of the afternoon of 2026-10-06, minutes after each session's start had seen the usual.
        frames = self.frames(moved_from=99)
        for place in (5, 6):
            index, frame = frames[place]
            stretched = self.np.clip((frame.astype(float) - 16.0) * 255.0 / 219.0, 0, 255).astype(self.np.uint8)
            frames[place] = (index, stretched)
        normal = [(index, (16.0 + frame.astype(float) * 219.0 / 255.0).astype(self.np.uint8)) if place not in (5, 6)
                  else (index, frame) for place, (index, frame) in enumerate(frames)]
        self.view.floor_levels = None
        reference = self.audit.audit_frames("clustered", normal, self.planned, self.view, self.grid, geometry=self.geometry)
        first = self.planned[0].episode_index
        self.view.floor_levels = {key: reference.camera[first][key] for key in ("grey", "tag_black", "red_over_blue")}
        result = self.audit.audit_frames("clustered", normal, self.planned, self.view, self.grid, geometry=self.geometry)
        named = [problem for problem in result.problems if "another picture state" in problem]
        self.assertEqual(len(named), 1, result.problems)
        self.assertIn(str([self.planned[5].episode_index, self.planned[6].episode_index]), named[0])


# -------------------------------------------------------------------------------------------------- structural checks


def _calls(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.append(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


class SessionSafetyStructureTests(unittest.TestCase):
    """These read the source. They fail when an edit takes a safety out, even if nothing else notices."""

    SESSIONS = ("record_layout.py", "trial_session.py")

    def source(self, name: str) -> tuple[str, ast.AST]:
        text = (RIG_SOURCES / name).read_text(encoding="utf-8")
        return text, ast.parse(text)

    def test_no_rig_module_reaches_a_control_loop_or_a_motor_except_through_a_gate(self):
        forbidden = {"run_attempt", "record_loop", "send_action", "sync_write", "enable_torque", "disable_torque"}
        for path in sorted(RIG_SOURCES.glob("*.py")):
            with self.subTest(module=path.name):
                used = forbidden & set(_calls(ast.parse(path.read_text(encoding="utf-8"))))
                self.assertFalse(used, f"{path.name} calls {sorted(used)} directly")

    def test_each_session_settles_the_arm_in_a_finally(self):
        for name in self.SESSIONS:
            with self.subTest(session=name):
                _, tree = self.source(name)
                settled = any(
                    "settle_arm" in _calls(ast.Module(body=node.finalbody, type_ignores=[]))
                    for node in ast.walk(tree) if isinstance(node, ast.Try) and node.finalbody
                )
                self.assertTrue(settled, f"{name}: settle_arm is not called from a finally block")

    def test_each_session_builds_the_follower_so_that_it_never_goes_limp(self):
        for name in self.SESSIONS:
            with self.subTest(session=name):
                _, tree = self.source(name)
                keeps_torque = any(
                    keyword.arg == "disable_torque_on_disconnect"
                    and isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                    for node in ast.walk(tree) if isinstance(node, ast.Call) for keyword in node.keywords
                )
                self.assertTrue(keeps_torque, f"{name}: the follower is not built with disable_torque_on_disconnect=False")

    def test_the_trial_session_clamps_every_step(self):
        _, tree = self.source("trial_session.py")
        clamped = any(keyword.arg == "max_relative_target" for node in ast.walk(tree) if isinstance(node, ast.Call)
                      for keyword in node.keywords)
        self.assertTrue(clamped, "trial_session.py: the follower is built without max_relative_target")

    def test_each_session_runs_the_preflight_before_it_connects(self):
        for name in self.SESSIONS:
            with self.subTest(session=name):
                text, _ = self.source(name)
                self.assertIn("run_preflight(", text)
                self.assertLess(text.index("run_preflight("), text.index("robot.connect()"))

    def test_each_session_pins_lerobot_before_importing_it(self):
        for name in self.SESSIONS:
            with self.subTest(session=name):
                text, _ = self.source(name)
                self.assertLess(text.index("use_pinned_lerobot("), text.index("from lerobot"))

    def test_a_placement_in_a_session_is_confirmed_by_the_operator_never_automatic(self):
        for name in self.SESSIONS:
            with self.subTest(session=name):
                text, _ = self.source(name)
                self.assertNotIn("auto_accept=True", text)
                self.assertNotIn("def go_pressed", text)
        # The one shared rule for what counts as the go: the go key alone, never the stop or re-record key.
        recorder, _ = self.source("record_layout.py")
        self.assertIn("confirm=lambda: consume_go(events)", recorder)
        # A trial's go hands the motors to a policy: the same rule, and the key must also have come by itself, not
        # in the middle of typing. Every way through `guarded_go` that can say yes ends in `consume_go`.
        trials, _ = self.source("trial_session.py")
        self.assertIn("confirm=lambda: typing_guard.guarded_go(", trials)
        guard = (RIG_SOURCES / "typing_guard.py").read_text(encoding="utf-8")
        body = guard[guard.index("def guarded_go("):guard.index("def phase_go(")]
        self.assertNotIn("return True", body)
        self.assertEqual(body.count("return consume_go(events)") + body.count("took = consume_go(events)"), 4)
        # The other key after which the policy starts: the n that ends the reposition after a refused start. It is
        # asked about by the same guard, and the phase is run again when it was typed.
        self.assertIn('starts_the_policy = label.endswith("-- reposition")', trials)
        body = trials[trials.index("def reenter_teleop("):trials.index("if not reenter_teleop(WARM_UP")]
        self.assertIn("if typing_guard.phase_go(guard,", body)
        # ... and the re-record key, which ends the phase too and is no go key, is turned away before the guard is
        # asked: it had heard no go, and said yes.
        self.assertLess(body.index('if events.get("rerecord_episode"):'), body.index("if typing_guard.phase_go(guard,"))
        self.assertEqual(body.count("return True"), 2)        # a phase that starts no policy; and the go, pressed alone

    def test_the_trial_runner_starts_one_key_listener_and_never_the_tooling_s_pausable_one(self):
        # A listener started after a window has been shown ends the program on this computer (the dry run of
        # 2026-10-06): the arm tooling's PausableKeyboard starts a new one after every score.
        text, _ = self.source("trial_session.py")
        self.assertEqual(text.count("typing_guard.Keyboard("), 1)
        self.assertNotIn("PausableKeyboard(", text)
        self.assertNotIn("init_keyboard_listener", text)
        # ... and it is started before the placement's window can exist
        self.assertLess(text.index("typing_guard.Keyboard("), text.index("PlacementPhase("))

    def test_what_the_scorer_sees_carries_the_blind_code_and_not_the_trial_s_name(self):
        text, _ = self.source("trial_session.py")
        self.assertIn('label = f"Trial {done + 1} [{trial.blind_code}]"', text)
        self.assertNotIn('f"Trial {trial.trial_id}', text)

    def test_teleoperation_in_the_trial_session_never_ends_on_a_timer(self):
        text, _ = self.source("trial_session.py")
        self.assertIn('float("inf")', text)
        self.assertNotIn("args.align_timeout, events", text)  # August's rollout used the align timeout as a phase length


if __name__ == "__main__":
    unittest.main()
