"""The recorder and the trial runner, run from start to finish on a rig made of fakes.

Each flow test runs the real `record_layout.main` or `trial_session.main` against `fakes_rig.FakeRig` and asserts on
the ordered trace of what happened. `fakes_rig` says what is real in such a run and what is faked. In short: the
session code, the arm tooling's gates and LeRobot's own control loop are real; the arm, the cameras, the dataset on
disk, the policies, the keyboard and the person are not.

Every flow also ends with `self.kept(run)`: the promises a session must keep however it ends (no motor command
outside a gated loop, no jump on a loop's first cycle, the arm settled and holding, the listener stopped, the dataset
and its placement file in step).

Three more groups follow the flows: the arm tooling's own `attempt_cycle` given a replay in place of a policy, with
that project's test fakes for the robot; a test of the placement loop's stop rule by itself; and checks that read the
sessions' sources and LeRobot's (which also run without the rig's interpreter).

Tests named `test_known_limit_...` pin behaviour that is not a promise: it is what the code does today, inherited
from the August tooling, written down so that it is met in a test before it is met at the rig.

The flows need OpenCV, LeRobot and the arm tooling, so they run with the rig's interpreter and skip elsewhere. They
find the arm tooling and the LeRobot checkout also from a worktree (`fakes_rig.located_tooling`).
"""

import ast
import importlib.util
import site
import time
import re
import threading
import unittest
from pathlib import Path

import fakes_rig
import numpy as np
from fakes_rig import FPS, READY, TASK, WHY_NOT, FakeRig, measured_in, read_sidecar

from tolerance_radius import layouts, schema
from tolerance_radius.config import REPO_ROOT, load_protocol
from tolerance_radius.rig import lerobot101
from tolerance_radius.schema import Point


def setUpModule():
    if not READY:
        return
    try:
        fakes_rig.prepare()
    except lerobot101.ToolingError as error:
        raise unittest.SkipTest(f"the arm tooling or LeRobot is not the tested version here: {error}") from error


def planned(dataset: str) -> list:
    return layouts.read_layout(REPO_ROOT / "config" / "layouts" / f"{dataset}.json")


class FlowCase(unittest.TestCase):
    maxDiff = None

    def rig(self, **options) -> FakeRig:
        rig = FakeRig(**options)
        self.addCleanup(rig.close)
        return rig

    def kept(self, run) -> None:
        """The session kept its promises (see `FakeRig._problems`)."""
        self.assertEqual(run.problems, [], "\n".join(run.story()))


# ====================================================================================================================
# The recorder
# ====================================================================================================================


@unittest.skipUnless(READY, WHY_NOT)
class RecorderNormalFlowTests(FlowCase):
    def test_a_two_episode_sitting_runs_every_phase_in_order(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertIsNone(run.error)
        self.assertEqual(run.rc, 0)
        self.kept(run)
        first, second = planned("clustered")[:2]
        self.assertEqual(run.story(), [
            "pin arm tooling",
            "pin lerobot",
            "preflight ok",
            "follower built: clamp=None, id=my_follower, torque_off_on_disconnect=False",
            f"dataset:create tester/tolrad-clustered-v1 {FPS}",
            "leader:connect",
            "follower:connect",
            "stream ok",
            "view confirmed",
            "listener:start",
            "viewer record",
            "teleop [Warm up] ran",
            "background table empty",
            f"placement [episode 0  cell {first.cell}  ({first.target.x_mm:.0f}, {first.target.y_mm:.0f}) mm] "
            f"at ({first.target.x_mm:.1f}, {first.target.y_mm:.1f}): accepted",
            "episode 0 recorded",
            "teleop [Reset the environment] ran",
            "dataset:save 0 4",
            "sidecar 0 clustered_placements.jsonl",
            f"placement [episode 1  cell {second.cell}  ({second.target.x_mm:.0f}, {second.target.y_mm:.0f}) mm] "
            f"at ({second.target.x_mm:.1f}, {second.target.y_mm:.1f}): accepted",
            "episode 1 recorded",
            "teleop [Reset the environment] ran",
            "dataset:save 1 4",
            "sidecar 1 clustered_placements.jsonl",
            "dataset:finalize 0",
            "teleop [Cool down] ran",
            "follower:disconnect torque on",
            "settle seeded",
            "leader:disconnect",
            "listener:stop",
        ])

    def test_what_the_operator_hears_in_order(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        cell = planned("clustered")[0].cell
        self.assertEqual(run.said(), [
            "Warm up", "Cube off the table", f"Place the cube, cell {cell}", "Recording episode 0",
            "Reset the environment", "Hold still", "Cool down", "Stop recording",
        ])

    def test_a_placement_happens_while_no_control_loop_runs_and_every_loop_is_behind_its_gate(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)  # includes: no command during a placement, no loop without a gate
        kinds = [event[0] for event in run.trace]
        placements = [i for i, kind in enumerate(kinds) if kind == "placement"]
        self.assertEqual(len(placements), 2)
        for index in placements:
            loops = [kind for kind in kinds[:index] if kind in ("loop", "loop:end")]
            self.assertEqual(loops[-1], "loop:end", "a placement ended while a control loop was open")
        for index in (i for i, kind in enumerate(kinds) if kind == "loop"):
            self.assertEqual(kinds[index - 2], "gate", f"a loop with no gate just before it: {run.trace[index]}")
        # Seven loops: warm-up, two episodes, two resets, cool-down... and each sent the leader's pose every cycle.
        self.assertEqual([event[1] for event in run.events("loop")],
                         ["teleop", "episode", "teleop", "episode", "teleop", "teleop"])
        self.assertTrue(all(event[2] > 0 for event in run.events("loop:end")))

    def test_the_sidecar_holds_what_was_measured_and_the_dataset_s_first_frame_shows_the_same(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        lines = read_sidecar(rig, "clustered")
        self.assertEqual([line["episode_index"] for line in lines], [0, 1])
        store = rig.disk[rig.hf_home / rig.repo_id("clustered")]
        for line, episode, plan in zip(lines, store.episodes, planned("clustered"), strict=False):
            placed = Point(**line["placed"])
            self.assertEqual(line["target"], {"x_mm": plan.target.x_mm, "y_mm": plan.target.y_mm})
            self.assertEqual((line["dataset"], line["cell"], line["round"], line["sitting"], line["rerecords"]),
                             ("clustered", plan.cell, plan.round, 1, 0))
            # What was written is the measurement, which is not the target (the careful operator is 0.7 mm off)...
            self.assertGreater(placed.distance_to(plan.target), 0.3)
            self.assertLessEqual(line["placement_error_mm"], 2.0)
            self.assertAlmostEqual(line["placement_error_mm"], placed.distance_to(plan.target))
            # ...and it is what the start-state audit will measure in the episode's own first frame.
            self.assertLess(measured_in(episode["first_front_rgb"]).distance_to(placed), 0.05)
            self.assertEqual(episode["task"], TASK)
        self.assertNotIn("tester", "".join(str(line) for line in lines), "the owner's name went into the sidecar")

    def test_the_cube_is_seen_through_the_robot_s_own_rgb_frames(self):
        """LeRobot hands out RGB and the detector wants BGR: with the conversion missing, no cube is ever measured."""
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        (placement,) = run.events("placement")
        self.assertEqual(placement[3], "accepted")
        self.assertIsNotNone(measured_in(rig.front_frame()), "the fake camera itself is wrong")
        self.assertIsNone(measured_in(rig.front_frame()[:, :, ::-1].copy()), "colour order would not matter")

    def test_an_episode_that_ends_on_its_timer_goes_on_to_the_reset(self):
        rig = self.rig()
        rig.operator.quiet("episode", what="end")
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "-t", "0.04"])
        self.kept(run)
        self.assertEqual(run.story("episode", "teleop", "dataset:save")[1:3],
                         ["episode 0 recorded", "teleop [Reset the environment] ran"])
        (saved,) = run.events("dataset:save")
        self.assertGreater(saved[2], 2)

    def test_the_whole_session_through_cli_ends_cleanly(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1"], through_cli=True)
        self.assertEqual(run.rc, 0)
        self.assertIn(("INFO", "session ended cleanly"), rig.logged)
        self.assertIn("arm: torque ON, goal seeded to present", run.out)


@unittest.skipUnless(READY, WHY_NOT)
class RecorderKeysTests(FlowCase):
    def test_a_re_record_repeats_the_placement_and_writes_no_line_for_the_discarded_take(self):
        rig = self.rig()
        rig.operator.at("teleop", "Reset", nth=0, tick=1).press("r")
        # The second time the cube lands somewhere else inside the tolerance: that is the start that must be kept.
        rig.operator.quiet("placement", nth=1, what="setup")
        rig.operator.at("placement", nth=1, tick=0).cube("target", offset_mm=(-1.0, 1.0))
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.story("placement", "episode", "teleop", "dataset:clear", "dataset:save", "sidecar")[1:], [
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): accepted",
            "episode 0 recorded",
            "teleop [Reset the environment] ran",
            "dataset:clear 4",
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): accepted",
            "episode 0 recorded",
            "teleop [Reset the environment] ran",
            "dataset:save 0 4",
            "sidecar 0 clustered_placements.jsonl",
            "teleop [Cool down] ran",
        ])
        (line,) = read_sidecar(rig, "clustered")
        self.assertEqual(line["rerecords"], 1)
        second_placement = run.events("placement")[1]
        self.assertEqual((line["placed"]["x_mm"], line["placed"]["y_mm"]), second_placement[4])
        self.assertLess(line["placed"]["x_mm"], 125.0)  # the second placement, left of the target, not the first
        self.assertIn("Re-record episode", run.said())

    def test_r_during_the_episode_itself_discards_it_after_the_reset(self):
        rig = self.rig()
        rig.operator.at("episode", nth=0, tick=1).press("r")
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.story("episode", "teleop", "dataset:clear", "dataset:save")[1:6], [
            "episode 0 recorded", "teleop [Reset the environment] ran", "dataset:clear 2",
            "episode 0 recorded", "teleop [Reset the environment] ran",
        ])
        self.assertEqual(len(read_sidecar(rig, "clustered")), 1)

    def test_q_during_a_placement_stops_the_sitting_with_nothing_recorded(self):
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="all")  # the cube never reaches the target
        rig.operator.at("placement", nth=0, tick=3).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("placement", "episode", "teleop", "dataset:save", "dataset:finalize", "settle"), [
            "teleop [Warm up] ran",
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): stopped",
            "dataset:finalize 0",
            "settle seeded",
        ])
        self.assertEqual(read_sidecar(rig, "clustered"), [])

    def test_q_while_the_cube_is_steady_is_a_stop_and_never_the_go(self):
        """The stop key sets the same flag as the go key (LeRobot's listener: q is "stop" and "end this phase").

        A placement that is steady when q is pressed must end as stopped. It used to end as accepted, and the session
        went on to the episode, where only the alignment gate's own look at the stop flag kept the loop from starting.
        """
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=12).press("q")  # the cube has been steady since frame 7
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("placement", "episode", "gate", "dataset:save", "settle")[1:], [
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): stopped",
            "settle seeded",
        ])
        self.assertFalse(any("placed at" in message for _level, message in rig.logged),
                         "a placement that was stopped was logged as placed")

    def test_a_reset_ended_with_the_arm_away_is_asked_for_again_before_any_placement(self):
        # The recorder's second run at the rig: the reset ended with the arm away from its start pose. The next
        # placement showed "no cube" for a cube that lay in plain view (the arm, resting elsewhere, is a second
        # orange thing), and in a placement the arm holds: there was no way on but to stop the sitting.
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset the environment", nth=0, what="setup")     # the leader is not brought back
        rig.operator.at("teleop", "Reset the environment", nth=0, tick=1).press("n")
        # (tighter tolerances than the recorder's own: the test rig's arm ends an episode 4 degrees away, in its pan)
        run = rig.record_layout(["--sitting", "1", "--limit", "2", "--start-pose-tolerance", "2",
                                 "--free-joint-tolerance", "2"])
        self.kept(run)
        story = run.story("placement", "episode", "teleop", "dataset:save")[1:]
        self.assertEqual(story[:5], [
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): accepted",
            "episode 0 recorded",
            "teleop [Reset the environment] ran",
            "teleop [Reset again: arm to the start pose] ran",
            "dataset:save 0 4",
        ])
        self.assertIn("The arm is not back near its start pose (shoulder_pan", run.out)
        self.assertIn("(allowed 2)", run.out)
        self.assertIn("Arm back to the start pose", run.said())
        # The log says for which joint: after sitting 1 it had to be read out of the recorded frames.
        self.assertTrue(any("the reset ended with the arm away from its start pose: shoulder_pan" in message
                            for _level, message in rig.logged), rig.logged)
        # The second episode is placed and recorded as usual.
        self.assertIn("episode 1 recorded", story)

    def test_the_pan_and_the_roll_may_rest_further_from_the_start_pose_than_the_joints_the_arm_holds(self):
        # Sitting 1 (2026-10-06): at one tolerance for every joint the reset was asked for again after two of three
        # episodes, for the base's swivel and the wrist's turn, which the folded arm does not hold by itself.
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset the environment", nth=0, what="setup")     # the leader is not brought back
        rig.operator.at("teleop", "Reset the environment", nth=0, tick=1).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "2", "--start-pose-tolerance", "2"])   # pan: 4 degrees
        self.kept(run)
        self.assertNotIn("teleop [Reset again: arm to the start pose] ran", run.story("teleop"))
        self.assertNotIn("The arm is not back near its start pose", run.out)
        self.assertIn("episode 1 recorded", run.story("episode"))

    def test_a_joint_the_folded_arm_holds_is_asked_back_at_the_recorder_s_own_tolerances(self):
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset the environment", nth=0, what="setup")
        rig.operator.at("teleop", "Reset the environment", nth=0, tick=0).leader(wrist_flex=7.0)
        rig.operator.at("teleop", "Reset the environment", nth=0, tick=1).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertIn("teleop [Reset again: arm to the start pose] ran", run.story("teleop"))
        self.assertIn("The arm is not back near its start pose (wrist_flex", run.out)
        self.assertIn("(allowed 5)", run.out)
        self.assertNotIn("shoulder_pan", run.out.split("The arm is not back near its start pose")[1].split("\n")[0])
        self.assertIn("episode 1 recorded", run.story("episode"))

    def test_the_start_pose_is_written_to_the_log(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        told = [message for _level, message in rig.logged if message.startswith("start pose: ")]
        self.assertEqual(len(told), 1, rig.logged)
        for joint in fakes_rig.JOINTS:
            self.assertIn(joint, told[0])

    def test_an_arm_that_never_comes_back_stops_the_sitting_and_keeps_what_was_recorded(self):
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset", what="setup")
        for nth in range(6):
            rig.operator.at("teleop", "Reset", nth=nth, tick=1).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "2", "--start-pose-tolerance", "2",
                                 "--free-joint-tolerance", "2"])
        self.kept(run)
        story = run.story("placement", "episode", "teleop", "dataset:save")[1:]
        self.assertEqual(story.count("teleop [Reset again: arm to the start pose] ran"), 3)
        self.assertIn("dataset:save 0 4", story)                       # the finished take is kept
        self.assertNotIn("episode 1 recorded", story)
        self.assertIn("still not back at its start pose", run.out)

    def test_the_start_pose_check_can_be_switched_off(self):
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset the environment", nth=0, what="setup")
        rig.operator.at("teleop", "Reset the environment", nth=0, tick=1).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--start-pose-tolerance", "0"])
        self.kept(run)
        self.assertNotIn("teleop [Reset again: arm to the start pose] ran", run.story("teleop"))

    def test_q_during_a_reset_keeps_the_finished_episode_and_stops(self):
        rig = self.rig()
        rig.operator.at("teleop", "Reset", nth=0, tick=1).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(run.story("placement", "episode", "teleop", "dataset:save", "sidecar", "settle")[1:], [
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): accepted",
            "episode 0 recorded",
            "teleop [Reset the environment] ran",
            "dataset:save 0 4",
            "sidecar 0 clustered_placements.jsonl",
            "settle seeded",
        ])

    def test_r_then_q_discards_the_take_and_stops(self):
        rig = self.rig()
        rig.operator.at("episode", nth=0, tick=1).press("r")
        rig.operator.at("teleop", "Reset", nth=0, tick=1).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(run.events("dataset:save"), [])
        self.assertEqual(run.story("dataset:clear", "dataset:finalize", "settle"),
                         ["dataset:clear 2", "dataset:finalize 0", "settle seeded"])
        self.assertEqual(read_sidecar(rig, "clustered"), [])

    def test_a_key_pressed_before_the_cube_is_steady_does_not_start_an_episode(self):
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=2).press("n")  # three good frames so far: not steady
        rig.operator.at("placement", nth=0, tick=39).cube(None)  # 37 steady frames later the cube is taken away
        rig.operator.at("placement", nth=0, tick=40).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.events("episode"), [])
        (placement,) = run.events("placement")
        self.assertEqual((placement[3], placement[5]), ("stopped", 41))

    def test_the_early_press_is_forgotten_and_a_later_one_starts_the_episode(self):
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=2).press("n")
        rig.operator.at("placement", nth=0, tick=14).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        (placement,) = run.events("placement")
        self.assertEqual((placement[3], placement[5]), ("accepted", 15))  # at the second press, not when first steady

    def test_a_press_of_n_left_over_from_the_save_does_not_confirm_the_next_placement(self):
        rig = self.rig()
        rig.operator.at("save", nth=0).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        frames = [event[5] for event in run.events("placement")]
        self.assertEqual(frames, [11, 11])  # both accepted by the operator's own key on the eleventh frame

    def test_a_hand_in_the_picture_keeps_the_placement_from_being_accepted(self):
        rig = self.rig()
        rig.operator.at("placement", nth=0, tick=0).second_cube(Point(40.0, 120.0))  # something else orange
        rig.operator.at("placement", nth=0, tick=20).second_cube(None)
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=None).press("n")  # the key is hammered throughout
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        (placement,) = run.events("placement")
        # Eight clean frames after the hand left at frame 20: accepted on frame 27 (the 28th), never before.
        self.assertEqual((placement[3], placement[5]), ("accepted", 28))

    def test_typing_at_the_cube_off_prompt_does_not_confirm_a_placement(self):
        rig = self.rig()
        rig.operator.answer("OFF the table", "n")  # typed into the terminal, heard by the global listener
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertIn(("typing-overheard", "n"), run.trace)
        (placement,) = run.events("placement")
        self.assertEqual((placement[3], placement[5]), ("accepted", 11))


@unittest.skipUnless(READY, WHY_NOT)
class RecorderPlacementViewTests(FlowCase):
    def test_the_placement_is_shown_in_a_window_of_its_own_and_the_viewer_keeps_the_cameras(self):
        # The default since the recorder's first run at the rig: shown in the viewer, the placement picture left
        # the live cameras standing still.
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.events("placement:built"), [("placement:built", "window")])
        self.assertGreater(rig.windows_shown, 0)
        self.assertEqual(run.events("placement:cleared"), [])
        self.assertEqual({event[3] for event in run.events("loop")}, {True})

    def test_asked_for_the_viewer_the_placement_opens_no_window_in_the_session(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--placement-view", "rerun"])
        self.kept(run)
        self.assertEqual(run.events("placement:built"), [("placement:built", "viewer or none")])
        self.assertEqual(rig.windows_shown, 0)
        # Every loop of a recording shows the cameras in the viewer, as record_session does.
        self.assertEqual({event[3] for event in run.events("loop")}, {True})

    def test_without_the_viewer_the_placement_is_shown_in_a_window_and_no_loop_feeds_the_viewer(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--no-display"])
        self.kept(run)
        self.assertEqual(run.events("viewer"), [])
        self.assertEqual(run.events("placement:built"), [("placement:built", "window")])
        self.assertEqual(rig.windows_shown, 11)
        self.assertEqual({event[3] for event in run.events("loop")}, {False})

    def test_in_the_window_space_accepts_a_steady_cube(self):
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.window_keys = [-1] * 12 + [ord(" ")]
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--placement-view", "window"])
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "accepted")
        self.assertEqual(len(run.events("dataset:save")), 1)

    def test_a_placement_that_is_skipped_aborted_or_times_out_ends_the_sitting_with_nothing_recorded(self):
        for status, set_up in (
            ("skipped", lambda rig: setattr(rig, "window_keys", [-1] * 12 + [ord("v")])),
            ("aborted", lambda rig: setattr(rig, "window_keys", [-1] * 12 + [ord("x")])),
            ("timeout", lambda rig: setattr(rig, "placement_timeout_s", 0.05)),
        ):
            with self.subTest(status=status):
                rig = self.rig()
                rig.operator.quiet("placement", nth=0, what="end")
                set_up(rig)
                run = rig.record_layout(["--sitting", "1", "--limit", "2", "--placement-view", "window"])
                self.assertEqual(run.rc, 0)
                self.kept(run)
                self.assertEqual(run.events("placement")[0][3], status)
                self.assertEqual(run.events("episode", "dataset:save", "sidecar"), [])
                self.assertEqual(run.story("teleop", "settle"), ["teleop [Warm up] ran", "settle seeded"])


@unittest.skipUnless(READY, WHY_NOT)
class RecorderKnownLimitTests(FlowCase):
    """Behaviour that is the August recorder's too, pinned so that it is known and not met by surprise."""

    def test_q_in_the_middle_of_an_episode_throws_the_cut_off_take_away(self):
        """Unlike record_session and LeRobot's own recorder, which save it: a take cut short by the stop key must not
        become one of the 55 demonstrations."""
        rig = self.rig()
        rig.operator.at("episode", nth=0, tick=1).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(run.events("dataset:save", "sidecar"), [])
        self.assertEqual(run.events("dataset:clear"), [("dataset:clear", 2)])
        self.assertIn("is not saved", run.out)
        self.assertEqual(run.story("episode", "teleop", "settle")[1:], ["episode 0 recorded", "settle seeded"])
        # The next run of the same sitting records episode 0 again.
        again = rig.record_layout(["--sitting", "1", "--resume", "--limit", "1"])
        self.kept(again)
        self.assertEqual([event[1] for event in again.events("dataset:save")], [0])

    def test_a_second_n_as_the_episode_starts_does_not_save_a_stub(self):
        """The go key of the placement is also the key that ends an episode. A take shorter than the shortest
        demonstration is a slip: it is thrown away and the same episode is placed and recorded again."""
        rig = self.rig()
        rig.operator.at("episode", nth=0, tick=0).press("n")
        # The flows run at 200 Hz: 0.01 s is two frames, and an ordinary take here is four.
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--min-episode-seconds", "0.01"])
        self.kept(run)
        self.assertEqual(run.events("dataset:clear"), [("dataset:clear", 1)])
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])
        self.assertIn("go key pressed twice", run.out)
        self.assertEqual(len(run.events("placement")), 2)
        (line,) = read_sidecar(rig, "clustered")
        self.assertEqual((line["episode_index"], line["rerecords"]), (0, 1))

    def test_a_take_in_which_the_arm_stood_still_is_thrown_away_and_the_episode_recorded_again(self):
        """The keys are read from every program. On 2026-10-06 the operator typed a message elsewhere while a
        placement waited; had the cube been steady in the ring, the first n of that message would have started the
        take, and with nobody at the leader the recorder would have saved an arm at rest as a demonstration."""
        rig = self.rig()
        rig.still_takes = {0}
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.events("dataset:clear"), [("dataset:clear", 4)])
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])
        self.assertIn("The arm did not move in that take", run.out)
        self.assertTrue(any(level == "WARNING" and "no joint moved by 30 degrees" in message
                            for level, message in rig.logged), rig.logged)
        self.assertEqual(len(run.events("placement")), 2)
        # every take ends into the reset, the thrown one too: the arm is never left in the freeze
        self.assertEqual(run.story("teleop").count("teleop [Reset the environment] ran"), 2)
        (line,) = read_sidecar(rig, "clustered")
        self.assertEqual((line["episode_index"], line["rerecords"]), (0, 1))

    def test_a_take_after_which_the_picture_is_another_is_not_saved_and_the_sitting_stops(self):
        """2026-10-06, episode 3 of the clustered set: the front camera went back to its own zoom and exposure 3.4
        seconds into the take. The take was saved, and it showed only at the next placement."""
        rig = self.rig()
        rig.picture_changed_after = {1: "the front camera's picture stands 108 mm from the one this session "
                                        "started on (allowed 5)"}
        run = rig.record_layout(["--sitting", "1", "--limit", "3"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])      # the take before it is kept
        self.assertEqual(run.events("dataset:clear"), [("dataset:clear", 4)])       # the take in another picture
        self.assertEqual(len(run.events("placement")), 2, "no placement is asked for in a picture that changed")
        self.assertIn("stands 108 mm from the one this session started on", run.out)
        self.assertIn("The take is not saved", run.out)
        self.assertTrue(any(level == "ERROR" and "the take is NOT saved" in message for level, message in rig.logged))
        self.assertEqual([line["episode_index"] for line in read_sidecar(rig, "clustered")], [0])
        self.assertEqual(run.story("settle"), ["settle seeded"])
        # the next run of the sitting takes that episode up again
        rig.picture_changed_after = {}
        again = rig.record_layout(["--sitting", "1", "--resume", "--limit", "1"])
        self.kept(again)
        self.assertEqual([event[1] for event in again.events("dataset:save")], [1])

    def test_a_wrist_camera_in_its_other_tone_state_is_said_at_once_and_refused_at_the_start_pose(self):
        """2026-10-06: 21 of the first 27 episodes of the clustered set were recorded with the wrist camera's
        picture in a state no earlier recording was in. No check looked at that camera."""
        rig = self.rig()
        rig.wrist_readings = {0: 3.20, 1: 3.20}
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 3)
        self.kept(run)
        self.assertIn("Wrist camera in its other state. Nothing is recorded.", run.said())
        # before the warm-up it only says: the arm rests where the last session left it
        self.assertIn("NOTE: the wrist camera's picture is not in its usual tone state: the floor over the jaws' "
                      "green reads 3.20", run.out)
        self.assertIn("press q in the warm-up", run.out)
        # at the start pose nothing is recorded in that state
        self.assertIn("Before anything is recorded, the wrist camera's picture is not in its usual tone state", run.out)
        self.assertIn("--any-wrist-state", run.out)
        self.assertEqual(run.events("placement", "episode", "dataset:save"), [])
        self.assertEqual(run.story("settle"), ["settle seeded"])

    def test_a_first_look_that_read_the_other_state_with_the_arm_elsewhere_stops_nothing(self):
        # Away from the start pose the number is not to be trusted: of the frames from the middle of the takes of
        # 2026-10-06 that still showed the jaws over a plain floor, some of each state read as the other.
        rig = self.rig()
        rig.wrist_readings = {0: 3.20}                # at the start pose, after the warm-up, the usual state
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertIn("NOTE: the wrist camera's picture is not in its usual tone state", run.out)
        self.assertNotIn("Before anything is recorded", run.out)
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4)])

    def test_a_wrist_camera_that_changed_during_the_warm_up_is_refused_before_anything_is_recorded(self):
        rig = self.rig()
        rig.wrist_readings = {1: 3.20}
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 3)
        self.kept(run)
        self.assertNotIn("NOTE: the wrist camera's picture is not", run.out)
        # "Start the command again" is what it says, and the same command does go on: the set it left is empty
        rig.wrist_readings, rig.wrist_read = {}, 0
        again = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual(again.rc, 0)
        self.kept(again)
        self.assertIn("an empty set is there", again.out)
        self.assertEqual(again.events("dataset:create"), [])
        self.assertEqual([event[1:] for event in again.events("dataset:save")], [(0, 4)])
        self.assertIn("Before anything is recorded, the wrist camera's picture is not in its usual tone state", run.out)
        self.assertEqual(run.events("placement", "episode", "dataset:save"), [])
        self.assertEqual(run.story("settle"), ["settle seeded"])

    def test_a_take_after_which_the_wrist_camera_has_changed_its_state_is_not_saved_and_the_sitting_stops(self):
        rig = self.rig()
        rig.wrist_readings = {3: 3.20, 4: 3.21}       # after the second take, and the second reading that confirms it
        run = rig.record_layout(["--sitting", "1", "--limit", "3"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])
        self.assertEqual(run.events("dataset:clear"), [("dataset:clear", 4)])
        self.assertIn("After that take, the wrist camera's picture has changed its tone state since this session "
                      "began: the floor over the jaws' green reads 3.21 now and read 2.55 then", run.out)
        self.assertIn("--any-wrist-state", run.out)
        self.assertEqual(len(run.events("placement")), 2)
        self.assertEqual([line["episode_index"] for line in read_sidecar(rig, "clustered")], [0])
        self.assertTrue(any(level == "ERROR" and "the take is NOT saved" in message for level, message in rig.logged))

    def test_the_sessions_read_the_wrist_camera_s_own_picture_the_right_way_round(self):
        """The robot hands its pictures over as red, green, blue; the number is made of the jaws' orange and is read
        from blue, green, red. Read the wrong way round the jaws are not found, the picture is "not judged", and the
        look is silently off. Here the stand-in's wrist camera shows a drawn rest picture and nothing is faked
        between the robot's frame and the verdict."""
        import cv2

        from fixtures_pictures import wrist_rest_picture

        for word, start in (("record", lambda rig: rig.record_layout(["--sitting", "1", "--limit", "1"])),
                            ("trials", lambda rig: rig.run_trials(TrialCase.argv(
                                self, rig, "T1a", policies=["clustered"], extra=["--limit", "1"])))):
            with self.subTest(session=word, picture="the usual state"):
                rig = self.rig()
                rig.write_audits()
                rig.wrist_real = True
                rig.wrist_frame = cv2.cvtColor(wrist_rest_picture(), cv2.COLOR_BGR2RGB)
                run = start(rig)
                self.assertEqual(run.rc, 0)
                self.assertTrue(any(message == "wrist camera, arm at its start pose: the floor over the jaws' green "
                                               "reads 2.58" for _level, message in rig.logged), rig.logged)
                self.assertNotIn("could not be judged", run.out)
                self.assertNotIn("not in its usual tone state", run.out)
            with self.subTest(session=word, picture="the other state"):
                rig = self.rig()
                rig.write_audits()
                rig.wrist_real = True
                rig.wrist_frame = cv2.cvtColor(wrist_rest_picture(stretched=True), cv2.COLOR_BGR2RGB)
                run = start(rig)
                self.assertEqual(run.rc, 3)
                self.assertRegex(run.out, r"NOTE: the wrist camera's picture is not in its usual tone state: the "
                                          r"floor over the jaws' green reads 3\.2\d")
                self.assertRegex(run.out, r"Before (anything is recorded|any trial), the wrist camera's picture is "
                                          r"not in its usual tone state")
                self.assertEqual(run.events("placement", "episode", "run", "dataset:save"), [])

    def test_a_wrist_reading_that_does_not_come_twice_is_not_a_change(self):
        rig = self.rig()
        rig.wrist_readings = {3: 3.20}                # the second reading, a moment later, is the usual state's
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4), (1, 4)])
        self.assertEqual(run.events("dataset:clear"), [])

    def test_the_operator_can_record_in_either_state_of_the_wrist_camera_and_the_log_says_so(self):
        rig = self.rig()
        rig.wrist_readings = {0: 3.20, 1: 3.20, 2: 3.20}      # the other state from the start; back after take 2
        run = rig.record_layout(["--sitting", "1", "--limit", "2", "--any-wrist-state"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4), (1, 4)])
        said = [message for level, message in rig.logged if level == "WARNING" and "(--any-wrist-state)" in message]
        self.assertEqual(len(said), 2, said)
        self.assertIn("NOTE: --any-wrist-state.", run.out)
        self.assertIn("is not in its usual tone state", said[0])                       # at the start
        self.assertIn("has changed its tone state since this session began", said[1])   # after the second take
        self.assertTrue(any(message == "wrist camera, before the warm-up: the floor over the jaws' green reads 3.20"
                            for _level, message in rig.logged))

    def test_a_wrist_picture_that_cannot_be_judged_is_said_and_stops_nothing(self):
        # another rest pose than the one the number was measured in, or the arm parked elsewhere
        rig = self.rig()
        rig.wrist_readings = {0: None, 1: None, 2: 3.20, 3: 3.20}
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertIn("NOTE: the wrist camera's picture could not be judged", run.out)
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4), (1, 4)])
        self.assertEqual(rig.wrist_read, 2, "with no first reading there is nothing to hold a take against")

    def test_the_wrist_camera_is_not_asked_when_a_stop_left_the_arm_away_from_its_start_pose(self):
        # Away from rest the wrist camera's picture is not the rest picture, and its number says nothing: a take
        # must not be lost to it. q in the reset ends the reset wherever the arm is.
        rig = self.rig()
        rig.wrist_readings = {2: 3.20, 3: 3.20}
        rig.operator.quiet("teleop", "Reset", nth=0, what="all")          # the leader is not brought back
        rig.operator.at("teleop", "Reset", nth=0, tick=1).press("q")
        # (tighter tolerances than the recorder's own: the test rig's arm ends an episode 4 degrees away, in its pan)
        run = rig.record_layout(["--sitting", "1", "--limit", "2", "--start-pose-tolerance", "2",
                                 "--free-joint-tolerance", "2"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4)], "the finished take is kept")
        self.assertEqual(rig.wrist_read, 2)
        self.assertTrue(any("wrist camera after the take: not asked" in message for _level, message in rig.logged))

    def test_the_limit_for_one_run_reaches_the_check_of_the_view_in_both_sessions(self):
        # --tags-relative-limit is for the day the camera has turned and the tags moved with the picture. Dropped
        # on its way to the check, the option did nothing and no test said so (the review of the night of
        # 2026-10-06).
        for word, start in (("record", lambda rig, extra: rig.record_layout(["--sitting", "1", "--limit", "1", *extra])),
                            ("trials", lambda rig, extra: rig.run_trials(TrialCase.argv(
                                self, rig, "T1a", policies=["clustered"], extra=["--limit", "1", *extra])))):
            for extra, expected in (([], None), (["--tags-relative-limit", "5"], 5.0)):
                with self.subTest(session=word, extra=extra):
                    rig = self.rig()
                    rig.write_audits()
                    run = start(rig, extra)
                    self.assertEqual(run.rc, 0)
                    self.assertEqual(rig.view_asked, [{"relative_limit_mm": expected}])

    def test_a_session_s_log_ends_with_the_usb_errors_of_its_cameras(self):
        line = ("2026-10-06 20:52:45.144 E  kernel[0:1d0114c] (IOUSBHostFamily) AppleUSBIORequest: "
                "AppleUSBIORequest::complete: device 1 (USB2.0_CAM1@02100000) endpoint 0x81: status 0xe00002ed "
                "(transaction error)")
        for word, start in (("record", lambda rig: rig.record_layout(["--sitting", "1", "--limit", "1"])),
                            ("trials", lambda rig: rig.run_trials(TrialCase.argv(
                                self, rig, "T1a", policies=["clustered"], extra=["--limit", "1"])))):
            with self.subTest(session=word):
                rig = self.rig()
                rig.write_audits()
                rig.usb_log_lines = [line, line, line]
                run = start(rig)
                self.assertEqual(run.rc, 0)
                told = [message for _level, message in rig.logged if message.startswith("USB transfer errors")]
                self.assertEqual(len(told), 1, rig.logged)
                self.assertIn("front camera 0, wrist camera 3", told[0])
                self.assertIn("front camera 0, wrist camera 3", run.out)
                # asked once, after the arm was settled
                kinds = [event[0] for event in run.trace if event[0] in ("settle", "usb:asked")]
                self.assertEqual(kinds, ["settle", "usb:asked"])

    def test_the_wrist_camera_is_looked_at_at_the_start_and_after_every_take(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(rig.wrist_read, 4, "before the warm-up, at the start pose, and after each of two takes")
        self.assertTrue(any(message == "wrist camera, arm at its start pose: the floor over the jaws' green reads 2.55"
                            for _level, message in rig.logged))
        self.assertEqual(sum("wrist camera after the take: the floor over the jaws' green reads 2.55" in message
                             for _level, message in rig.logged), 2)

    def test_the_picture_is_looked_at_after_every_take(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(rig.pictures_checked, 2)
        self.assertEqual(len(run.events("picture:checked")), 2)
        # ... after the reset, with the arm back at rest, and before the take is saved
        story = [line for line in run.story("episode", "teleop", "picture:checked", "dataset:save")
                 if "Warm up" not in line and "Cool down" not in line]
        self.assertEqual(story, [
            "episode 0 recorded", "teleop [Reset the environment] ran", "picture:checked", "dataset:save 0 4",
            "episode 1 recorded", "teleop [Reset the environment] ran", "picture:checked", "dataset:save 1 4",
        ])

    def test_the_look_after_a_take_is_given_its_announcer_and_what_it_finds_is_spoken(self):
        rig = self.rig()
        rig.picture_changed_after = {1: "the front camera's picture is in another tone state: white 219 and black 20"}
        run = rig.record_layout(["--sitting", "1", "--limit", "3"])
        self.kept(run)
        self.assertEqual(rig.picture_looks, [{"told": "announcer"}] * 2, "a hand in the way must be asked out")
        self.assertIn("Camera picture changed. Take not saved.", run.said())
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4)])
        # ... and when the wrist camera's is the one that changed
        rig = self.rig()
        rig.wrist_readings = {3: 3.20, 4: 3.21}
        run = rig.record_layout(["--sitting", "1", "--limit", "3"])
        self.kept(run)
        self.assertIn("Wrist camera picture changed. Take not saved.", run.said())

    def test_after_a_stop_the_look_asks_hands_out_and_does_not_wait_long(self):
        # q ended the reset: the arm stands where it stood and cannot be led, so fifteen seconds of "Hands out of
        # the picture" asked for what nobody could give; a hand at the basket can still be withdrawn, and with no
        # word at all a finished take was lost to it after a second and a half (two reviews, the night of
        # 2026-10-06).
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset", nth=0, what="end")
        rig.operator.at("teleop", "Reset", nth=0, tick=3).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        # a hand can still be taken out, and is asked out; the arm cannot be led, so the asking is short
        self.assertEqual(rig.picture_looks, [{"patience_s": 4.0, "told": "announcer"}])

    def test_the_picture_is_looked_at_when_the_stop_key_ended_the_reset_too(self):
        # A take recorded in another picture is not one of the 55 whatever ended its reset (the review of
        # 2026-10-06: with the look left out on that way, no test said so).
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset", nth=0, what="end")
        rig.operator.at("teleop", "Reset", nth=0, tick=3).press("q")
        rig.picture_changed_after = {0: "the front camera's picture stands 60 mm from the one this session started on"}
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(rig.pictures_checked, 1)
        self.assertEqual(run.events("dataset:save"), [])
        self.assertEqual(run.events("dataset:clear"), [("dataset:clear", 4)])
        # ... and with the picture as it was, the finished take is kept when q ends the reset
        rig = self.rig()
        rig.operator.quiet("teleop", "Reset", nth=0, what="end")
        rig.operator.at("teleop", "Reset", nth=0, tick=3).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(rig.pictures_checked, 1)
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4)])

    def test_takes_in_which_the_arm_moved_are_kept(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.kept(run)
        self.assertEqual(rig.takes_asked, 2)
        self.assertEqual(run.events("dataset:clear"), [])
        self.assertNotIn("did not move", run.out)

    def test_the_still_arm_rule_can_be_switched_off(self):
        rig = self.rig()
        rig.still_takes = {0}
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--min-travel", "0"])
        self.kept(run)
        self.assertEqual(run.events("dataset:clear"), [])
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])

    def test_a_take_that_is_too_short_is_named_as_that_and_not_as_a_still_arm(self):
        rig = self.rig()
        rig.still_takes = {0}
        rig.operator.at("episode", nth=0, tick=0).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--min-episode-seconds", "0.01"])
        self.kept(run)
        self.assertIn("go key pressed twice", run.out)
        self.assertNotIn("did not move", run.out)
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])

    def test_a_take_of_ordinary_length_is_kept_under_the_shortest_take_rule(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--limit", "2", "--min-episode-seconds", "0.01"])
        self.kept(run)
        self.assertEqual(run.events("dataset:clear"), [])
        self.assertEqual([event[1:] for event in run.events("dataset:save")], [(0, 4), (1, 4)])

    def test_r_or_the_left_arrow_during_a_placement_is_not_the_go(self):
        """LeRobot's listener makes r "end this phase and re-record". During a placement it means nothing: what
        follows a go is an arm that moves, so only the go key is one."""
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=12).press("r")
        rig.operator.at("placement", nth=0, tick=30).cube(None)
        rig.operator.at("placement", nth=0, tick=31).press("q")
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "stopped")
        self.assertEqual(run.events("episode", "dataset:save", "sidecar"), [])

    def test_an_r_pressed_during_a_placement_does_not_discard_the_episode_that_follows(self):
        rig = self.rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=12).press("r")
        rig.operator.at("placement", nth=0, tick=20).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)
        self.assertEqual(run.events("dataset:clear"), [])
        self.assertEqual(run.events("dataset:save"), [("dataset:save", 0, 4)])

    def test_known_limit_a_key_that_arrives_as_the_episode_s_clock_runs_out_skips_the_reset(self):
        """`teleop_phase` clears the phase key when it ends, not when it begins, and the recorder (like the August
        one) does not clear it before the reset: a press that lands between the episode's last cycle and the reset's
        first is taken by the reset. The arm then holds wherever the episode left it through the save and the next
        placement. The trial runner clears the key before every phase it enters."""
        rig = self.rig()
        rig.operator.quiet("episode", what="end")
        rig.operator.at("gate", "Reset", nth=0, tick=0).press("n")
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "-t", "0.03"])
        self.kept(run)
        reset = next(event for event in run.events("teleop") if event[1] == "Reset the environment")
        self.assertEqual(reset[2:], (True, 0))  # it "ran", for no cycle at all
        self.assertEqual(len(run.events("dataset:save")), 1)

    def test_known_limit_a_dataset_recorded_at_another_rate_is_only_refused_at_the_first_episode(self):
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=2, fps=30)
        run = rig.record_layout(["--sitting", "1", "--resume", "--limit", "1"])  # the flows run at 200 Hz
        self.assertIsInstance(run.error, ValueError)
        self.assertIn("dataset fps should be equal to requested fps", str(run.error))
        self.kept(run)  # the arm is settled all the same
        self.assertEqual(run.story("teleop", "placement", "episode", "settle"), [
            "teleop [Warm up] ran",
            "placement [episode 2  cell A3  (25, 125) mm] at (25.0, 125.0): accepted",
            "settle seeded",
        ])

    def test_a_smoke_test_keeps_its_placements_out_of_the_real_dataset_s_file(self):
        """A dataset recorded under another name (a smoke test, the practice block) gets a placement file of its own,
        so its lines cannot pass for the real set's episodes."""
        rig = self.rig()
        rig.record_layout(["--sitting", "1", "--name", "tolrad-smoke-{dataset}", "--limit", "1"])
        self.assertEqual(read_sidecar(rig, "clustered"), [])
        rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual([(line["dataset"], line["episode_index"]) for line in read_sidecar(rig, "clustered")],
                         [("clustered", 0)])
        self.assertEqual(len(read_sidecar(rig, "tolrad-smoke-clustered")), 1)


@unittest.skipUnless(READY, WHY_NOT)
class RecorderRefusalTests(FlowCase):
    def assert_arm_never_touched(self, run):
        self.assertEqual(run.events("follower:connect", "leader:connect", "loop", "listener:start"), [])

    def test_refusals_on_paper_come_before_anything_is_opened(self):
        for argv, namespace, said in (
            (["--sitting", "9"], True, "sitting must be 1 to 4"),
            (["--sitting", "1"], False, "no dataset owner"),
        ):
            with self.subTest(argv=argv):
                rig = self.rig()
                run = rig.record_layout(argv, namespace=namespace)
                self.assertEqual(run.rc, 1)
                self.assertIn(said, run.err)
                self.assertEqual(run.events("bus:open", "dataset:create", "pin"), [])
                self.kept(run)

    def test_plan_only_prints_the_sitting_and_opens_nothing(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "3", "--plan-only"], namespace=False)
        self.assertEqual(run.rc, 0)
        self.assertIn("sitting 3: dataset spread, rounds [4, 5], episodes 33 to 54", run.out)
        self.assertEqual(run.trace, [("exit", 0)])

    def test_arm_tooling_that_is_not_the_tested_version_is_refused(self):
        rig = self.rig()
        rig.pin_error = "the arm tooling is not the tested version: it has uncommitted changes to tracked files"
        run = rig.record_layout(["--sitting", "1"])
        self.assertEqual(run.rc, 1)
        self.assertIn("not the tested version", run.err)
        self.assertEqual(run.events("bus:open"), [])

    def test_a_refused_preflight_opens_only_the_bus_and_leaves_the_arm_as_it_was(self):
        for fault in ("torque off", "stale goal"):
            with self.subTest(fault=fault):
                rig = self.rig()
                if fault == "torque off":
                    rig.arm.torque = False
                else:
                    rig.arm.blocked = True  # held by hand: the goal is 20 degrees away and the joint does not go
                    rig.arm.goal["elbow_flex"] += 20.0
                goal_before, present_before = dict(rig.arm.goal), dict(rig.arm.present)
                run = rig.record_layout(["--sitting", "1"])
                self.assertEqual(run.rc, 1)
                self.assertIn("REFUSING TO RECORD", run.out)
                self.assertEqual(run.story("bus:open", "bus:close", "preflight"),
                                 ["bus:open the preflight", "bus:close the preflight torque kept", "preflight REFUSED"])
                self.assert_arm_never_touched(run)
                self.assertEqual(run.events("dataset:create"), [], "a refused start left a dataset behind")
                self.assertEqual((rig.arm.goal, rig.arm.present, rig.arm.torque),
                                 (goal_before, present_before, fault != "torque off"))

    def test_a_camera_that_will_not_stream_ends_the_session_before_any_loop(self):
        rig = self.rig()
        rig.dead_camera = "wrist"
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 3)
        self.kept(run)
        self.assertIn("a camera connected but will not stream", run.out)
        self.assertIn("wrist", run.out)
        self.assertEqual(run.events("loop", "listener:start", "placement"), [])
        self.assertEqual(run.story("stream", "dataset:finalize", "settle", "follower:disconnect"),
                         ["stream DEAD", "dataset:finalize 0", "follower:disconnect torque on", "settle seeded"])

    def test_after_a_camera_fault_the_same_command_goes_on_in_the_empty_set_it_left(self):
        # The fault's own sentence says to start the command again. It used to be refused then ("a dataset already
        # exists", holding nothing), and only --resume carried on.
        for extra in ([], ["--resume"]):
            with self.subTest(extra=extra):
                rig = self.rig()
                rig.dead_camera = "wrist"
                rig.record_layout(["--sitting", "1", "--limit", "1"])
                rig.dead_camera = None
                again = rig.record_layout(["--sitting", "1", "--limit", "1", *extra])
                self.assertEqual(again.rc, 0)
                self.kept(again)
                self.assertEqual(again.story("dataset:create", "dataset:resume", "dataset:save"),
                                 ["dataset:resume tester/tolrad-clustered-v1 0", "dataset:save 0 4"])
                self.assertEqual("an empty set is there" in again.out, not extra)

    def test_a_camera_that_cannot_open_leaves_the_arm_untouched(self):
        rig = self.rig()
        rig.camera_that_cannot_open = "wrist"
        run = rig.record_layout(["--sitting", "1"], through_cli=True)
        self.assertEqual(run.rc, 2)
        self.assertEqual(run.events("settle"), [("settle", "untouched")])
        self.assertIn("arm: unchanged", run.out)
        self.assertTrue(rig.arm.torque)

    def test_a_view_that_no_longer_matches_the_stored_calibration_ends_the_session(self):
        rig = self.rig()
        rig.view_agrees = False
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertEqual(run.rc, 3)
        self.kept(run)
        self.assertIn("REFUSING TO RECORD -- the camera's picture is no longer the calibration's", run.out)
        self.assertEqual(run.events("loop", "listener:start", "viewer", "placement", "dataset:save"), [])
        self.assertEqual(run.story("view", "settle"), ["view REFUSED", "settle seeded"])

    def test_a_gate_that_does_not_open_at_the_warm_up_stops_the_session_with_no_motion(self):
        rig = self.rig()
        rig.leader_pose["elbow_flex"] += 30.0  # the leader was left somewhere else
        rig.operator.quiet("gate", what="setup")  # and nobody brings it back
        run = rig.record_layout(["--sitting", "1", "--align-timeout", "0.3"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("teleop", "placement", "settle"), ["teleop [Warm up] NOT RUN", "settle seeded"])
        self.assertEqual(run.events("loop"), [])
        self.assertEqual(rig.commands, [])

    def test_a_leader_moved_during_the_placement_is_held_at_the_gate_until_it_is_brought_back(self):
        rig = self.rig()
        rig.operator.at("placement", nth=0, tick=5).leader(elbow_flex=20.0)
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.kept(run)  # includes: no jump on the first cycle of any loop
        gate = next(event for event in run.events("gate") if event[1] == "Episode 0")
        self.assertTrue(gate[2])
        self.assertGreaterEqual(gate[3], 2, "the gate did not hold while the leader was 20 degrees away")
        self.assertEqual(run.story("episode", "dataset:save"), ["episode 0 recorded", "dataset:save 0 4"])

    def test_a_gate_that_does_not_open_before_an_episode_stops_without_recording_it(self):
        rig = self.rig()
        rig.operator.at("placement", nth=0, tick=5).leader(elbow_flex=20.0)
        rig.operator.quiet("gate", "Episode", what="setup")
        run = rig.record_layout(["--sitting", "1", "--limit", "1", "--align-timeout", "0.3"])
        self.kept(run)
        self.assertEqual(run.story("placement", "episode", "teleop", "dataset:save", "settle")[1:], [
            "placement [episode 0  cell C2  (125, 75) mm] at (125.0, 75.0): accepted",
            "episode 0 NOT RECORDED",
            "settle seeded",
        ])
        self.assertEqual(read_sidecar(rig, "clustered"), [])


@unittest.skipUnless(READY, WHY_NOT)
class RecorderResumeTests(FlowCase):
    def test_resume_carries_on_at_the_next_unrecorded_episode(self):
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=2)
        run = rig.record_layout(["--sitting", "1", "--resume", "--limit", "1"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        third = planned("clustered")[2]
        self.assertEqual(run.story("dataset:create", "dataset:resume", "placement", "dataset:save", "sidecar"), [
            "dataset:resume tester/tolrad-clustered-v1 2",
            f"placement [episode 2  cell {third.cell}  ({third.target.x_mm:.0f}, {third.target.y_mm:.0f}) mm] "
            f"at ({third.target.x_mm:.1f}, {third.target.y_mm:.1f}): accepted",
            "dataset:save 2 4",
            "sidecar 2 clustered_placements.jsonl",
        ])
        self.assertIn("(resumed)", run.out)

    def test_two_sessions_in_a_row_as_the_smoke_test_runs_them(self):
        rig = self.rig()
        first = rig.record_layout(["--sitting", "1", "--name", "tolrad-smoke-{dataset}", "--limit", "2"])
        second = rig.record_layout(["--sitting", "1", "--name", "tolrad-smoke-{dataset}", "--resume", "--limit", "1"])
        self.kept(first)
        self.kept(second)  # includes: the second session's preflight found the arm as the first left it
        self.assertEqual((first.rc, second.rc), (0, 0))
        self.assertEqual(second.story("preflight", "dataset:resume", "dataset:save"),
                         ["preflight ok", "dataset:resume tester/tolrad-smoke-clustered 2", "dataset:save 2 4"])
        self.assertEqual(len(rig.disk[rig.hf_home / "tester" / "tolrad-smoke-clustered"].episodes), 3)

    def test_the_later_sittings_append_to_the_dataset_the_earlier_ones_made(self):
        for sitting, dataset, first_index in ((3, "spread", 33), (4, "clustered", 33)):
            with self.subTest(sitting=sitting):
                rig = self.rig()
                rig.existing_dataset(dataset, episodes=33)
                run = rig.record_layout(["--sitting", str(sitting), "--resume", "--limit", "1"])
                self.assertEqual(run.rc, 0)
                self.kept(run)
                self.assertEqual(run.story("dataset:save", "sidecar"),
                                 [f"dataset:save {first_index} 4", f"sidecar {first_index} {dataset}_placements.jsonl"])
                target = planned(dataset)[first_index].target
                self.assertEqual(run.events("placement")[0][2], (target.x_mm, target.y_mm))

    def test_a_dataset_whose_count_does_not_fit_the_sitting_is_refused_before_the_arm_is_connected(self):
        for sitting, dataset, episodes, said in (
            (4, "clustered", 30, "an earlier sitting of this dataset is incomplete"),
            (3, "spread", 0, "an earlier sitting of this dataset is incomplete"),
            (2, "spread", 33, "nothing left to record"),
            (1, "clustered", 55, "nothing left to record"),
        ):
            with self.subTest(sitting=sitting, episodes=episodes):
                rig = self.rig()
                rig.existing_dataset(dataset, episodes=episodes)
                run = rig.record_layout(["--sitting", str(sitting), "--resume"])
                self.assertEqual(run.rc, 1)
                self.assertIn(said, run.err)
                self.assertEqual(run.events("follower:connect", "leader:connect"), [])

    def test_resume_without_a_dataset_and_a_new_recording_over_one_are_both_refused(self):
        rig = self.rig()
        run = rig.record_layout(["--sitting", "1", "--resume"])
        self.assertEqual(run.rc, 1)
        self.assertIn("--resume, but there is no dataset", run.err)
        rig.existing_dataset("clustered", episodes=5)
        run = rig.record_layout(["--sitting", "1"])
        self.assertEqual(run.rc, 1)
        self.assertIn("a dataset already exists", run.err)
        self.assertIn("(it holds 5 episodes)", run.err)
        self.assertIn("scripts/rig/set_aside.py", run.err)
        self.assertEqual(run.events("dataset:create", "dataset:resume", "follower:connect"), [])

    def test_a_set_that_a_session_left_without_closing_is_not_appended_to(self):
        """The dry run of 2026-10-06, with LeRobot's own writer on a copy: a session killed after two saves left
        `info.json` at 29 episodes, 27 rows in the episode table and a table file without its end. The next
        `--resume` went on at episode 29, wrote over that file, and the set no longer loaded."""
        died = {"said": 29, "rows": 27, "unreadable": ["data/chunk-000/file-002.parquet"]}
        for counts, logged, said in (
            (died, 29, "1 of its table files cannot be read (data/chunk-000/file-002.parquet)"),
            ({"said": 29, "rows": 27, "unreadable": []}, 29, "it says it holds 29 episodes and its episode table has 27"),
            (None, 26, "it holds 27 episodes and the placement log clustered_placements.jsonl has 26 lines"),
            ({"said": 27, "rows": 27, "unreadable": [], "pictures": 118}, 27,
             "118 pictures of a take that was never saved are left in its images folder"),
            (None, 0, "it holds 27 episodes and the placement log clustered_placements.jsonl has 0 lines"),
        ):
            with self.subTest(said=said):
                rig = self.rig()
                rig.existing_dataset("clustered", episodes=27, logged=logged)
                rig.counts_on_disk = counts
                run = rig.record_layout(["--sitting", "1", "--resume"])
                self.assertEqual(run.rc, 1)
                self.assertIn("does not add up: " + said, run.err)
                self.assertIn("Nothing was changed", run.err)
                self.assertIn("scripts/rig/repair_set.py clustered", run.err)
                self.assertEqual(run.events("dataset:resume", "dataset:create", "follower:connect", "leader:connect"), [],
                                 "the set was opened for writing, or the arm connected")

    def test_a_set_left_empty_by_a_start_that_recorded_nothing_is_gone_on_in(self):
        # A start refused after the set was created (the wrist camera, the picture, a stop in the warm-up) leaves
        # an empty set. The sentence that followed, "Start the command again", was then refused: "a dataset already
        # exists (it holds 0 episodes)" (the review of the night of 2026-10-06).
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=0)
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertIn("an empty set is there", run.out)
        self.assertEqual(run.events("dataset:create"), [])
        self.assertEqual(run.story("dataset:resume", "dataset:save"),
                         ["dataset:resume tester/tolrad-clustered-v1 0", "dataset:save 0 4"])
        # ... but not when that empty set does not add up, or holds something
        for counts, said in (({"said": 0, "rows": 0, "unreadable": [], "pictures": 44}, "a dataset already exists"),
                             ({"said": 0, "rows": 0, "unreadable": ["data/chunk-000/file-000.parquet"]},
                              "a dataset already exists")):
            rig = self.rig()
            rig.existing_dataset("clustered", episodes=0)
            rig.counts_on_disk = counts
            run = rig.record_layout(["--sitting", "1", "--limit", "1"])
            self.assertEqual(run.rc, 1)
            self.assertIn(said, run.err)
            self.assertEqual(run.events("dataset:resume", "dataset:create", "follower:connect"), [])

    def test_a_set_whose_counts_agree_is_looked_through_before_it_is_appended_to(self):
        # A save cut short in a session that then closed the set leaves the take's rows in a data file: every count
        # agrees, and the next episode was read from those rows beside its own pictures (measured on copies in the
        # night of 2026-10-06). The repair's own look sees it.
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=27)
        rig.deeper_look_says = "its files hold rows of a take that was never saved in data/chunk-000/file-002.parquet"
        run = rig.record_layout(["--sitting", "1", "--resume"])
        self.assertEqual(run.rc, 1)
        self.assertIn("does not add up: its files hold rows of a take that was never saved", run.err)
        self.assertEqual(run.events("dataset:resume", "follower:connect"), [])
        # ... and a set that is sound is looked through once, and gone on in
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=27)
        run = rig.record_layout(["--sitting", "1", "--resume", "--limit", "1"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.events("set:looked-through"), [("set:looked-through", "tolrad-clustered-v1")])
        self.assertEqual([event[1] for event in run.events("dataset:save")], [27])

    def test_a_new_set_is_not_begun_under_a_placement_log_that_holds_another_s_lines(self):
        # The clustered set was begun again on 2026-10-06 (its first four episodes set aside): had its log stayed,
        # the new set's lines would have followed the old one's.
        rig = self.rig()
        rig.placement_dir.mkdir(parents=True, exist_ok=True)
        (rig.placement_dir / "clustered_placements.jsonl").write_text('{"episode_index": 0}\n' * 4, encoding="utf-8")
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual(run.rc, 1)
        self.assertIn("the placement log clustered_placements.jsonl holds 4 lines of an earlier one", run.err)
        # the set is not there, so the way out is the one that sets a log aside by itself
        self.assertIn("scripts/rig/set_aside.py clustered --as NAME --log-only", run.err)
        self.assertEqual(run.events("dataset:create", "follower:connect"), [])

    def test_a_sitting_begun_before_the_other_set_s_earlier_sitting_is_finished_is_said_and_goes_on(self):
        # The dry run of 2026-10-06: sitting 2 started with sitting 1 six episodes short, and nothing said so.
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=27)
        run = rig.record_layout(["--sitting", "2", "--limit", "1"])
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertIn("NOTE: sitting 2 is meant to come after the sittings before it: sitting 1 ends at episode 32 of "
                      "the clustered set, which holds 27 episodes", run.out)
        self.assertEqual([event[1] for event in run.events("dataset:save")], [0])
        self.assertTrue(any(level == "WARNING" and "meant to come after" in message for level, message in rig.logged))
        # ... and with sitting 1 done there is nothing to say
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=33)
        run = rig.record_layout(["--sitting", "2", "--limit", "1"])
        self.assertNotIn("meant to come after", run.out)
        # a rehearsal under another name is not held to the study's sittings
        rig = self.rig()
        run = rig.record_layout(["--sitting", "2", "--name", "tolrad-smoke-{dataset}", "--limit", "1"])
        self.assertNotIn("meant to come after", run.out)

    def test_a_new_sitting_that_is_refused_for_its_plan_creates_no_dataset(self):
        """Sitting 4 asked for with no dataset and no --resume: refused, and nothing may be left in the way.

        The dataset used to be created first and the plan checked second. The empty directory then made the right
        command (sitting 1) fail with "a dataset already exists".
        """
        rig = self.rig()
        run = rig.record_layout(["--sitting", "4"])
        self.assertEqual(run.rc, 1)
        self.assertIn("an earlier sitting of this dataset is incomplete", run.err)
        self.assertEqual(run.events("dataset:create", "follower:connect"), [])
        self.assertFalse((rig.hf_home / rig.repo_id("clustered")).exists())
        right = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual(right.rc, 0)
        self.kept(right)

    def test_resume_with_another_task_string_is_refused(self):
        rig = self.rig()
        rig.existing_dataset("clustered", episodes=2, task="pick up the orange cube")
        run = rig.record_layout(["--sitting", "1", "--resume"])
        self.assertEqual(run.rc, 1)
        self.assertIn("is not the task this dataset was recorded with", run.err)
        self.assertEqual(run.events("follower:connect"), [])


@unittest.skipUnless(READY, WHY_NOT)
class RecorderFaultTests(FlowCase):
    def test_an_exception_in_the_middle_of_an_episode_still_settles_the_arm(self):
        rig = self.rig()
        rig.operator.at("episode", nth=0, tick=1).fail_read(ConnectionError("Failed to sync read 'Present_Position'"))
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertIsInstance(run.error, ConnectionError)
        self.kept(run)
        self.assertEqual(
            run.story("dataset:save", "dataset:finalize", "follower:disconnect", "settle", "listener:stop"),
            ["dataset:finalize 2", "follower:disconnect torque on", "settle seeded", "listener:stop"])
        self.assertEqual(read_sidecar(rig, "clustered"), [])
        self.assertEqual(rig.preflight_problems(), [])

    def test_the_same_fault_through_cli_is_exit_code_2_with_the_traceback_logged(self):
        rig = self.rig()
        rig.operator.at("episode", nth=0, tick=1).fail_read(ConnectionError("Failed to sync read 'Present_Position'"))
        run = rig.record_layout(["--sitting", "1", "--limit", "2"], through_cli=True)
        self.assertEqual(run.rc, 2)
        self.assertIn(("ERROR", "SESSION FAILED"), rig.logged)
        self.assertIn("arm: torque ON", run.out)

    def test_an_exception_in_the_save_settles_the_arm_and_writes_no_placement(self):
        rig = self.rig()
        rig.save_fault = OSError("No space left on device")
        run = rig.record_layout(["--sitting", "1", "--limit", "2"])
        self.assertIsInstance(run.error, OSError)
        self.kept(run)
        self.assertEqual(read_sidecar(rig, "clustered"), [])
        self.assertEqual(run.events("settle"), [("settle", "seeded")])

    def test_ctrl_c_in_a_reset_settles_the_arm_and_is_exit_code_130(self):
        rig = self.rig()

        def interrupt(_rig, _phase):
            raise KeyboardInterrupt

        rig.operator.at("teleop", "Reset", nth=0, tick=1).do(interrupt)
        run = rig.record_layout(["--sitting", "1", "--limit", "2"], through_cli=True)
        self.assertEqual(run.rc, 130)
        self.kept(run)
        self.assertEqual(run.events("settle"), [("settle", "seeded")])

    def test_when_the_goal_cannot_be_seeded_the_arm_is_released_and_the_operator_is_told(self):
        rig = self.rig()
        rig.seed_fails = True
        run = rig.record_layout(["--sitting", "1", "--limit", "1"])
        self.assertEqual(run.events("settle"), [("settle", "torque-disabled")])
        self.assertFalse(rig.arm.torque)
        self.assertIn("arm: torque OFF", run.out)
        self.assertIn("--hold", run.out)


@unittest.skipUnless(READY, WHY_NOT)
class PlacementStopTests(unittest.TestCase):
    """The placement loop by itself: a stop and a confirmation that arrive together."""

    def run_placement(self, press_at: int, *, stop: bool, go: bool) -> object:
        import cv2

        from tolerance_radius.rig.placement_phase import PlacementPhase
        from tolerance_radius.rig.vision import TableView

        target = Point(75.0, 25.0)
        on_target = cv2.cvtColor(fakes_rig.scene_rgb(target), cv2.COLOR_RGB2BGR)
        view = TableView(fakes_rig.PX_TO_MM, cv2.cvtColor(fakes_rig.scene_rgb(), cv2.COLOR_RGB2BGR))
        state = {"frame": -1, "go": False, "stop": False, "now": 0.0}

        def grab():
            state["frame"] += 1
            state["now"] += 0.03
            if state["frame"] == press_at:
                state["go"], state["stop"] = go, stop
            return on_target

        def confirm():
            pressed, state["go"] = state["go"], False
            return pressed

        phase = PlacementPhase(view, grab, tolerance_mm=2.0, stable_frames=3, confirm=confirm,
                               should_stop=lambda: state["stop"], show=lambda *_: None, wait_key=lambda _ms: -1,
                               clock=lambda: state["now"], timeout_s=5.0)
        with fakes_rig.located_tooling():
            return phase.run(target)

    def test_a_stop_that_arrives_with_the_confirmation_ends_the_placement_as_stopped(self):
        result = self.run_placement(6, stop=True, go=True)  # what the session's q key does: both flags at once
        self.assertEqual(result.status, "stopped")
        self.assertIsNone(result.measured)

    def test_a_confirmation_alone_is_still_accepted_and_a_stop_alone_still_stops(self):
        self.assertEqual(self.run_placement(6, stop=False, go=True).status, "accepted")
        self.assertEqual(self.run_placement(6, stop=True, go=False).status, "stopped")


# ====================================================================================================================
# The trial runner
# ====================================================================================================================


class TrialCase(FlowCase):
    def trial_rig(self, **options) -> FakeRig:
        """A rig with both datasets audited (every start where it was planned) and the clustered set's tables."""
        rig = self.rig(**options)
        rig.write_audits()
        rig.write_replay_dataset()
        return rig

    def argv(self, rig, *blocks, log="pilot_test.jsonl", pilot=True, policies=(), replay=False, extra=()):
        argv = ["--blocks", *blocks, "--log", str(rig.trial_log(log))]
        if pilot:
            argv.append("--pilot")
        for name in policies:
            argv += ["--policy", f"act_{name}={rig.policy_dir(name)}"]
        if replay:
            argv += ["--replay-root", str(rig.replay_root)]
        return [*argv, *extra]

    def records(self, rig, log="pilot_test.jsonl"):
        """The trial log, read back the way the planner and the analysis read it."""
        return schema.read_trials(rig.trial_log(log))

    def kept(self, run) -> None:
        """As for every flow, and every trial log the rig holds is valid by `schema.read_trials`."""
        super().kept(run)
        for log in sorted((run.rig.tmp / "trials").glob("*.jsonl")):
            for record in schema.read_trials(log):  # raises on a line that does not satisfy the schema
                self.assertIn(record.outcome, schema.OUTCOMES)

    def moved_by(self, rig, run_phase) -> str:
        """Which joint the commands of one attempt moved: the signature of the executor that sent them."""
        asked = [command["asked"] for command in rig.commands
                 if command["loop"] == "run" and command["phase"] == run_phase.serial]
        spans = {joint: max(a[f"{joint}.pos"] for a in asked) - min(a[f"{joint}.pos"] for a in asked)
                 for joint in fakes_rig.JOINTS}
        return max(spans, key=spans.get)


#: The joint each kind of executor moves on the fake rig (see `FakePolicy` and `FakeRig.replay_actions`).
SIGNATURE = {"act_clustered": "shoulder_pan", "act_spread": "elbow_flex", "replay": "wrist_roll"}
RESET = "Reset"
WARM_UP = "Warm up"
COOL_DOWN = "Cool down"


@unittest.skipUnless(READY, WHY_NOT)
class TrialNormalFlowTests(TrialCase):
    def test_one_policy_trial_runs_every_phase_in_order(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.assertIsNone(run.error)
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story(), [
            "plan T1a-001 act_clustered",          # on paper, before anything is opened
            "pin arm tooling",
            "pin lerobot",
            "preflight ok",
            "policy:load policy:clustered cpu",    # loaded and checked before the robot is built
            "follower built: clamp=15.0, id=my_follower, torque_off_on_disconnect=False",
            "leader:connect",
            "follower:connect",
            "stream ok",
            "cameras confirmed",
            "view confirmed",
            "listener:start",
            "viewer trials",
            f"teleop [{WARM_UP}] ran",
            "keys:pause",                          # typing at the prompt must not reach the session's keys
            "keys:resume",
            "background table empty",
            "plan T1a-001 act_clustered",
            "placement [Trial 1 [PB89]] at (75.0, 25.0): accepted",
            "handoff ok",
            "run policy:clustered: 7 steps",
            "keys:pause",                          # scored with the keys paused: the one listener stays
            "keys:resume",
            "score 5",
            "record T1a-001 success",
            f"teleop [{RESET}] ran",
            f"teleop [{COOL_DOWN}] ran",
            "follower:disconnect torque on",
            "settle seeded",
            "leader:disconnect",
            "listener:stop",
        ])
        (record,) = self.records(rig)
        self.assertEqual((record.outcome, record.subgoal, record.policy_repo, record.checkpoint),
                         ("success", 5, "act_clustered", "080000"))
        self.assertEqual(record.execution["steps_sent"], 7)
        self.assertEqual(record.execution["handoff_tolerance_deg"], 8.0)
        self.assertEqual(record.placed, Point(*run.events("placement")[0][4]))
        self.assertAlmostEqual(record.nearest_demo_distance_mm, record.placed.distance_to(record.planned.target))
        self.assertEqual(record.versions["protocol_version"], "pilot")
        self.assertEqual(set(record.media), {"front", "wrist"})
        self.assertTrue(all(not path.startswith("/") for path in record.media.values()), record.media)

    def test_one_replay_trial_sends_the_recorded_actions_in_order_and_ends_after_the_last(self):
        rig = self.trial_rig()
        rig.operator.quiet("run")  # nobody touches a key: the replay ends the attempt itself
        run = rig.run_trials(self.argv(rig, "T1b", replay=True, extra=["--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        (record,) = self.records(rig)
        episode = record.planned.replay_episode
        self.assertEqual(run.story("replay:load", "handoff", "run", "score", "record"), [
            f"replay:load {episode}", "handoff ok", f"run replay:{episode}: 8 steps", "score 5",
            f"record {record.planned.trial_id} success",
        ])
        sent = [[command["goal"][joint] for joint in fakes_rig.JOINTS] for command in rig.commands
                if command["loop"] == "run"]
        self.assertEqual(len(sent), 8)
        for step, (got, recorded) in enumerate(zip(sent, FakeRig.replay_actions(episode), strict=True)):
            for joint, a, b in zip(fakes_rig.JOINTS, got, recorded, strict=True):
                self.assertAlmostEqual(a, b, places=6, msg=f"step {step}, {joint}")
        self.assertEqual(run.events("handoff")[0][3], 5.0)  # the tighter tolerance of a replay
        self.assertEqual((record.policy_repo, record.checkpoint, record.nearest_demo_episode), (None, None, episode))
        self.assertEqual(record.execution["steps_sent"], 8)

    def test_a_replay_is_rewound_when_the_same_demonstration_is_played_again(self):
        rig = self.trial_rig()
        rig.operator.answer("score 0-5", "void the cube was knocked")  # so the same trial comes back
        run = rig.run_trials(self.argv(rig, "T1b", replay=True, extra=["--limit", "2"]))
        self.kept(run)
        first, second = self.records(rig)
        self.assertEqual((first.outcome, second.outcome), ("void", "success"))
        self.assertEqual(first.planned.replay_episode, second.planned.replay_episode)
        self.assertEqual(len(run.events("replay:load")), 1, "the same demonstration was read from disk twice")
        runs = [phase for phase in rig.history if phase.kind == "run"]
        self.assertEqual(len(runs), 2)
        for phase in runs:
            wrist = [command["asked"]["wrist_roll.pos"] for command in rig.commands
                     if command["loop"] == "run" and command["phase"] == phase.serial]
            self.assertEqual(wrist, list(FakeRig.replay_actions(0)[:, fakes_rig.JOINTS.index("wrist_roll")]))

    def test_every_teleoperation_phase_is_unbounded_ends_on_a_key_and_stays_out_of_the_viewer(self):
        """`float("inf")` goes through the pinned `teleop_phase`, `PhaseClock` and LeRobot's `record_loop`."""
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        teleop_loops = [event for event in run.events("loop") if event[1] == "teleop"]
        self.assertEqual(len(teleop_loops), 3)  # warm-up, reset, cool-down
        self.assertEqual({(event[2], event[3]) for event in teleop_loops}, {(float("inf"), False)})
        self.assertTrue(all(event[3] >= 3 for event in run.events("teleop")), "a phase did not wait for its key")
        ends = [i for i, event in enumerate(run.trace) if event[:2] == ("loop:end", "teleop")]
        self.assertTrue(all(run.trace[i - 1] == ("key", "n") for i in ends), "a phase ended without its key")
        self.assertIn("elapsed -- press 'n' to end", run.out)
        self.assertNotIn("5 seconds", run.said())  # no countdown is spoken for a phase with no end

    def test_a_placement_that_times_out_ends_the_trials_and_still_parks_the_arm(self):
        rig = self.trial_rig()
        rig.operator.quiet("placement", nth=0)
        rig.placement_timeout_s = 0.05
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("placement", "run", "record", "teleop", "settle")[1:], [
            "placement [Trial 1 [PB89]] at (75.0, 25.0): timeout", f"teleop [{COOL_DOWN}] ran", "settle seeded"])

    def test_limit_stops_after_that_many_trials(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.kept(run)
        self.assertEqual([record.planned.trial_id for record in self.records(rig)], ["T1a-001", "T1a-002", "T1a-003"])
        self.assertEqual(len(run.events("run")), 3)
        self.assertEqual(run.story("teleop")[-1], f"teleop [{COOL_DOWN}] ran")

    def test_a_small_block_runs_to_its_end_and_a_second_session_finds_nothing_to_do(self):
        rig = self.trial_rig()

        def small(protocol):
            block = {**protocol["blocks"]["T1b"], "anchors_zero": 1,
                     "staircases": [{"id": "up", "start_level_mm": 5.0, "trials": 1},
                                    {"id": "down", "start_level_mm": 25.0, "trials": 1}]}
            return {**protocol, "blocks": {**protocol["blocks"], "T1b": block}}

        rig.protocol_changes = small
        run = rig.run_trials(self.argv(rig, "T1b", replay=True))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(len(self.records(rig)), 3)
        self.assertIn("All planned trials of these blocks are done.", run.out)
        self.assertEqual(run.story("teleop")[-1], f"teleop [{COOL_DOWN}] ran")
        again = rig.run_trials(self.argv(rig, "T1b", replay=True))
        self.assertEqual(again.rc, 0)
        self.assertIn("Nothing to do", again.out)
        self.assertEqual(again.events("pin", "bus:open", "follower:connect"), [])

    def test_the_real_attempt_recorder_writes_the_videos_the_record_points_to(self):
        rig = self.trial_rig(real_recorder=True)
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1", "--no-display"]))
        self.kept(run)
        (record,) = self.records(rig)
        self.assertEqual(set(record.media), {"front", "wrist"})
        for path in record.media.values():
            self.assertTrue((rig.repo_root / path).is_file(), path)
            self.assertIn("T1a-001", path)

    def test_an_attempt_that_runs_out_its_clock_is_scored_like_any_other(self):
        rig = self.trial_rig()
        rig.operator.quiet("run")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"],
                                       extra=["--limit", "1", "--attempt-seconds", "0.05"]))
        self.kept(run)
        (record,) = self.records(rig)
        self.assertEqual(record.outcome, "success")
        self.assertGreater(record.execution["steps_sent"], 3)


@unittest.skipUnless(READY, WHY_NOT)
class TrialExecutorTests(TrialCase):
    def check_each_trial_ran_what_was_planned(self, rig, run, records):
        runs = [phase for phase in rig.history if phase.kind == "run"]
        self.assertEqual(len(runs), len(records))
        for record, phase in zip(records, runs, strict=True):
            condition = record.planned.condition
            with self.subTest(trial=record.planned.trial_id, condition=condition):
                expected = (f"replay:{record.planned.replay_episode}" if condition == "replay"
                            else "policy:" + condition.removeprefix("act_"))
                self.assertEqual(phase.info["executor"], expected)
                self.assertEqual(self.moved_by(rig, phase), SIGNATURE[condition])
                self.assertEqual(record.policy_repo, None if condition == "replay" else condition)

    def test_two_policies_in_one_block_each_trial_is_run_by_its_planned_policy(self):
        rig = self.trial_rig()
        rig.protocol_changes = fakes_rig.frozen
        run = rig.run_trials(self.argv(rig, "T2", log="main.jsonl", pilot=False, policies=["clustered", "spread"],
                                       extra=["--limit", "6"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        records = self.records(rig, "main.jsonl")
        self.assertEqual(len(records), 6)
        conditions = [record.planned.condition for record in records]
        for first, second in zip(conditions[0::2], conditions[1::2], strict=True):
            self.assertEqual({first, second}, {"act_clustered", "act_spread"})  # the two trials of a pair
        self.assertIn("act_spread", conditions[0::2], "the second policy never went first in three pairs")
        self.check_each_trial_ran_what_was_planned(rig, run, records)
        self.assertEqual([record.planned.target for record in records[0::2]],
                         [record.planned.target for record in records[1::2]])

    def test_a_policy_block_and_the_replay_block_interleaved(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", "T1b", policies=["clustered"], replay=True, extra=["--limit", "6"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        records = self.records(rig)
        self.assertEqual({record.planned.condition for record in records}, {"act_clustered", "replay"})
        self.check_each_trial_ran_what_was_planned(rig, run, records)
        tolerances = {record.planned.condition: record.execution["handoff_tolerance_used_deg"] for record in records}
        self.assertEqual(tolerances, {"act_clustered": 8.0, "replay": 5.0})
        # Every record carries the session's whole list of execution settings, keyed as the protocol keys them, and
        # where the arm started from.
        for record in records:
            self.assertEqual(set(fakes_rig.pilot_protocol()["execution"]) - set(record.execution), set())
            self.assertEqual(record.execution["replay_handoff_tolerance_deg"], 5.0)
            self.assertEqual(record.execution["control_hz"], FPS)
            self.assertEqual(record.execution["view_crept_limit_mm"], 1.0, "the limit the creep was held to")
            self.assertEqual(len(record.execution["start_pose_deg"]), 6)
            # How long the control loop ran, from the first step sent to the last: seconds, never negative. The
            # clock here is the real one and an attempt lasts a few cycles, so no more is asked of the number.
            self.assertTrue(record.scored)
            self.assertIsInstance(record.execution["loop_s"], float)
            self.assertGreaterEqual(record.execution["loop_s"], 0.0)

    def test_each_policy_is_given_its_own_processors_and_the_session_s_task(self):
        rig = self.trial_rig()
        rig.protocol_changes = fakes_rig.frozen
        run = rig.run_trials(self.argv(rig, "T2", log="main.jsonl", pilot=False, policies=["clustered", "spread"],
                                       extra=["--limit", "1"]))
        self.kept(run)  # includes: no policy was handed another policy's processors
        self.assertEqual(sorted(event[1:3] for event in run.events("predict:built")),
                         [("policy:clustered", TASK), ("policy:spread", TASK)])
        self.assertEqual(sorted(event[1:] for event in run.events("policy:overrides")),
                         [("policy:clustered", 0.01), ("policy:spread", 0.01)])


@unittest.skipUnless(READY, WHY_NOT)
class TrialVoidTests(TrialCase):
    def test_a_start_refused_by_the_handoff_gate_is_logged_void_and_the_same_trial_is_asked_again(self):
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0, 30.0, 30.0]  # three opening moves 30 degrees away
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("plan", "handoff", "teleop", "run", "score", "record")[2:], [
            "plan T1a-001 act_clustered",
            "handoff REFUSED",
            "teleop [Trial 1 [PB89] -- reposition] ran",
            "handoff REFUSED",
            "teleop [Trial 1 [PB89] -- reposition] ran",
            "handoff REFUSED",
            "record T1a-001 void (the handoff gate refused the start pose)",
            f"teleop [{RESET}] ran",
            "plan T1a-002 act_clustered",
            "handoff ok",
            "run policy:clustered: 7 steps",
            "score 5",
            "record T1a-002 success",
            f"teleop [{RESET}] ran",
            f"teleop [{COOL_DOWN}] ran",
        ])
        void, scored = self.records(rig)
        self.assertEqual((void.outcome, void.subgoal, void.execution["steps_sent"]), ("void", None, 0))
        self.assertEqual(void.execution["loop_s"], 0.0, "no step was sent: the loop has no duration")
        # No attempt began, so there is no pose it began from: the void record holds the pose read before the gate's
        # first look, and the next trial's record its own (the pose its gate passed at, not one kept from before).
        looks = [(ok, {name: round(value, 2) for name, value in sorted(present.items())})
                 for ok, _action, present in rig.gate_looks]
        self.assertEqual([ok for ok, _pose in looks], [False, False, False, True])
        self.assertEqual(void.execution["start_pose_deg"], looks[0][1])
        self.assertEqual(scored.execution["start_pose_deg"], looks[3][1])
        self.assertNotEqual(looks[0][1], looks[3][1], "the test cannot tell the two poses apart")
        self.assertEqual(void.planned.target, scored.planned.target)       # the same slot of the plan...
        self.assertNotEqual(void.planned.blind_code, scored.planned.blind_code)  # ...under a new code
        self.assertEqual([command["loop"] for command in rig.commands
                          if command["phase"] in {p.serial for p in rig.history if "Trial 1" in p.label}],
                         ["teleop"] * 6, "an executor commanded the arm in the trial whose start was refused")

    def test_a_void_after_a_trial_that_ran_does_not_carry_that_trial_s_pose(self):
        # The gate is one for all the trials of a kind: what it kept of the first must not reach the second's record.
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [0.0, 30.0, 30.0, 30.0]    # the first runs, the second never
        rest = {**fakes_rig.START_POSE, "shoulder_pan": fakes_rig.START_POSE["shoulder_pan"] + 1.303,
                "wrist_roll": fakes_rig.START_POSE["wrist_roll"] - 2.211}
        rig.operator.quiet("teleop", "Reset", nth=0, what="all")
        rig.operator.at("teleop", "Reset", nth=0, tick=1).do(lambda rig, _phase: rig.leader_pose.update(rest))
        rig.operator.at("teleop", "Reset", nth=0, tick=6).press("n")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        ran, void = self.records(rig)
        self.assertEqual((ran.outcome, void.outcome, void.void_reason),
                         ("success", "void", "the handoff gate refused the start pose"))
        looks = [(ok, {name: round(value, 2) for name, value in sorted(present.items())})
                 for ok, _action, present in rig.gate_looks]
        self.assertEqual([ok for ok, _pose in looks], [True, False, False, False])
        self.assertEqual(ran.execution["start_pose_deg"], looks[0][1])
        self.assertEqual(void.execution["start_pose_deg"], looks[1][1], "the pose read before its own first look")
        self.assertNotEqual(looks[0][1], looks[1][1], "the test cannot tell the two poses apart")

    def test_a_start_refused_once_runs_after_the_arm_is_repositioned(self):
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0]
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(run.story("handoff", "run", "record"),
                         ["handoff REFUSED", "handoff ok", "run policy:clustered: 7 steps", "record T1a-001 success"])

    def test_the_correction_after_a_refused_start_says_where_the_joint_is_and_where_it_is_wanted(self):
        """2026-10-07, the first pilot: the gate refused a replay three times for the wrist's turn, by 7.8, 8.4
        and 9.8 units. The line said by how much and not which way, and every correction went the wrong way."""
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0]
        # The operator's correction: the wrist turned by 9 units and left there (the stand-in's careful operator
        # would bring the leader back to the start pose), then the go. In the loop's second cycle: its first command
        # is the aligned leader's, and must ask for no jump.
        rig.operator.quiet("teleop", "reposition", what="all")
        rig.operator.at("teleop", "reposition", nth=0, tick=1).leader(wrist_roll=9.0)
        rig.operator.at("teleop", "reposition", nth=0, tick=4).press("n")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(run.story("handoff", "run", "record"),
                         ["handoff REFUSED", "handoff ok", "run policy:clustered: 7 steps", "record T1a-001 success"])
        told = re.search(r"wrist roll: it is at ([+-]\d+\.\d) and is wanted at ([+-]\d+\.\d) \(within 8\)", run.out)
        self.assertIsNotNone(told, run.out[-2000:])
        refused, action, present = rig.gate_looks[0]
        self.assertFalse(refused)
        # which way: where it is, then where the first action wants it, 30 units on
        self.assertEqual((float(told[1]), float(told[2])),
                         (round(present["wrist_roll.pos"], 1), round(action["wrist_roll.pos"], 1)))
        self.assertAlmostEqual(float(told[2]) - float(told[1]), 30.0, places=1)
        self.assertIn("the line below shows the joint that is furthest from where it is wanted", run.out)
        # the phase is still the tooling's "-- reposition" by its name: the log, the speech and the go's rule see it
        self.assertEqual(len([event for event in run.events("teleop") if str(event[1]).endswith("-- reposition")]), 1)
        # while it runs its name is live, the arm's reads are watched, and the readout holds what they gave: every
        # joint the gate will judge, not only the one it refused for
        correction = [look for look in rig.phase_looks if look["label"].endswith("-- reposition")]
        others = [look for look in rig.phase_looks if not look["label"].endswith("-- reposition")]
        self.assertEqual([(look["live"], look["watched"]) for look in correction], [(True, True)])
        self.assertRegex(correction[0]["readout"], r"^[a-z ]+ now [+-]\d+\.\d, wanted [+-]\d+\.\d( ok)?$")
        self.assertTrue(others)
        self.assertEqual({(look["live"], look["watched"]) for look in others}, {(None, False)})
        # ... and nothing is left on the arm's object when the correction is over
        self.assertNotIn("get_observation", vars(rig.followers[-1]))
        # the record's start pose is the pose the gate passed at, not the one it had refused
        passed, _, passed_from = rig.gate_looks[-1]
        self.assertTrue(passed)
        record = self.records(rig)[0]
        self.assertEqual(record.execution["start_pose_deg"],
                         {name: round(value, 2) for name, value in sorted(passed_from.items())})
        self.assertAlmostEqual(record.execution["start_pose_deg"]["wrist_roll.pos"] - present["wrist_roll.pos"], 9.0,
                               places=1, msg="the pose after the correction, 9 units on in the wrist's turn")
        # the attempt's countdown stands aside for the correction and begins anew after it: never two at once
        mine = [(event, name) for event, name in rig.clock_events if name.startswith("Trial 1 [")]
        attempt = next(name for _, name in mine if not name.endswith("-- reposition"))
        self.assertEqual(mine, [("on", attempt), ("off", attempt), ("on", attempt + " -- reposition"),
                                ("off", attempt + " -- reposition"), ("on", attempt), ("off", attempt)])

    def test_the_correction_goes_on_to_the_joint_that_is_furthest_once_the_refused_one_is_brought_home(self):
        """2026-10-07, 19:36: the wrist was refused, the base stood 2.6 from where it was wanted and was not shown,
        and the next look refused for the base. The readout watches every joint the gate judges."""
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0]
        rig.operator.quiet("teleop", "reposition", nth=0, what="all")
        for tick in range(1, 9):
            rig.operator.at("teleop", "reposition", nth=0, tick=tick).leader(wrist_roll=3.5)    # 28 of the 30
        rig.operator.at("teleop", "reposition", nth=0, tick=12).press("n")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        (correction,) = [look for look in rig.phase_looks if look["label"].endswith("-- reposition")]
        # the wrist is 2 from where it is wanted now; the hand on the leader let the base wander further than that
        self.assertRegex(correction["readout"], r"^shoulder pan now [+-]\d+\.\d, wanted [+-]\d+\.\d ok$")
        # the lines printed once name the joint the gate refused for, and no other
        self.assertEqual(run.out.count(": it is at "), 1)
        self.assertIn("wrist roll: it is at ", run.out)
        self.assertEqual(run.story("handoff", "run"), ["handoff REFUSED", "handoff ok", "run policy:clustered: 7 steps"])

    def test_a_replay_s_correction_is_held_to_the_replay_s_own_tolerance(self):
        # 5 for a replay, 8 for a policy: a wrist left 6 from where the demonstration began is not ok.
        rig = self.trial_rig()
        rig.operator.at("teleop", "Warm up", tick=1).leader(wrist_roll=7.0)      # the arm rests 7 off in the wrist
        rig.operator.quiet("teleop", "reposition", nth=0, what="all")
        rig.operator.at("teleop", "reposition", nth=0, tick=1).leader(wrist_roll=-1.0)
        rig.operator.at("teleop", "reposition", nth=0, tick=4).press("n")
        run = rig.run_trials(self.argv(rig, "T1b", replay=True, extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(run.story("handoff")[:3], ["handoff REFUSED", "handoff REFUSED", "handoff ok"])
        first, second = [look for look in rig.phase_looks if look["label"].endswith("-- reposition")]
        self.assertRegex(first["readout"], r"^wrist roll now [+-]\d+\.\d, wanted [+-]\d+\.\d$")   # 6 off: no ok
        self.assertTrue(second["readout"].endswith(" ok"), second["readout"])
        self.assertIn("(within 5)", run.out)
        self.assertNotIn("(within 8)", run.out)
        (record,) = self.records(rig)
        passed, _action, passed_from = rig.gate_looks[-1]
        self.assertTrue(passed)
        self.assertEqual(record.execution["start_pose_deg"],
                         {name: round(value, 2) for name, value in sorted(passed_from.items())})
        self.assertEqual(record.execution["handoff_tolerance_used_deg"], 5.0)

    def test_every_record_of_an_interleaved_session_carries_the_pose_its_own_gate_passed_at(self):
        """A policy's trials and the replays alternate in one session, each kind through a gate of its own. Every
        reset leaves the arm a little elsewhere, and two of the policy's starts are refused and corrected."""
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0, 30.0]
        for k in range(8):
            # ... by no round figure, and near enough to the start pose for a replay's five degrees
            rest = {**fakes_rig.START_POSE, "shoulder_pan": fakes_rig.START_POSE["shoulder_pan"] + 0.303 * (k + 1),
                    "wrist_roll": fakes_rig.START_POSE["wrist_roll"] - 0.211 * (k + 1)}
            rig.operator.quiet("teleop", "Reset", nth=k, what="all")
            rig.operator.at("teleop", "Reset", nth=k, tick=1).do(
                lambda rig, _phase, rest=rest: rig.leader_pose.update(rest))
            rig.operator.at("teleop", "Reset", nth=k, tick=6).press("n")
        for k in range(2):
            rig.operator.quiet("teleop", "reposition", nth=k, what="all")
            rig.operator.at("teleop", "reposition", nth=k, tick=1).leader(wrist_roll=9.0)
            rig.operator.at("teleop", "reposition", nth=k, tick=4).press("n")
        run = rig.run_trials(self.argv(rig, "T1a", "T1b", policies=["clustered"], replay=True,
                                       extra=["--limit", "6"]))
        self.kept(run)
        records = self.records(rig)
        self.assertEqual({record.planned.condition for record in records}, {"act_clustered", "replay"})
        self.assertEqual([record.outcome for record in records], ["success"] * 6)
        passing = [present for ok, _action, present in rig.gate_looks if ok]
        self.assertEqual(len([ok for ok, _action, _present in rig.gate_looks if not ok]), 2)
        self.assertEqual(len(passing), 6)
        for record, present in zip(records, passing, strict=True):
            self.assertEqual(record.execution["start_pose_deg"],
                             {name: round(value, 2) for name, value in sorted(present.items())},
                             record.planned.trial_id)
        poses = [tuple(sorted(record.execution["start_pose_deg"].items())) for record in records]
        self.assertEqual(len(set(poses)), 6, "every trial began from a pose of its own, or the test tells nothing")
        # ... to a hundredth, as a record without a correction has it
        self.assertTrue(any(round(value, 2) != value for present in passing for value in present.values()))

    def test_a_first_action_that_is_no_number_is_refused_although_the_tooling_s_gate_passes_it(self):
        """The tooling's gate asks whether the largest move is over its tolerance, and a value that is no number is
        over nothing (the second review of the night of 2026-10-07)."""
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [float("nan")] * 3
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(run.story("handoff"), ["handoff ok"] * 3, "the stand-in hears the tooling's own verdict")
        self.assertEqual(run.events("run"), [], "an action that is no number reached an attempt")
        self.assertEqual({command["loop"] for command in rig.commands}, {"teleop"}, "an executor commanded the arm")
        # ... and the record does not blame the arm's pose, nor is the operator sent to move the arm
        self.assertEqual([(record.outcome, record.void_reason) for record in self.records(rig)],
                         [("void", "the first action was not a number; no attempt was made")])
        said = [message for _level, message in rig.logged if "the first action is not a number" in message]
        self.assertEqual(len(said), 3)
        self.assertIn("wrist_roll.pos", said[0])
        self.assertEqual(run.out.count("The first action was not a number (wrist_roll.pos): moving the arm cannot "
                                       "help."), 2, "said at each of the two corrections")
        self.assertNotIn(": it is at ", run.out)
        # ... and a start that is a number again runs, from the pose that look passed at
        once = self.trial_rig()
        once.wild_first_actions["policy:clustered"] = [float("nan")]
        run = once.run_trials(self.argv(once, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(len(run.events("run")), 1)
        (record,) = self.records(once)
        self.assertEqual(record.outcome, "success")
        self.assertEqual(record.execution["start_pose_deg"],
                         {name: round(value, 2) for name, value in sorted(once.gate_looks[-1][2].items())})

    def test_a_first_action_is_asked_for_numbers_in_every_name_it_holds(self):
        # ... also in one the arm does not report, which the tooling's gate has nothing to compare with; and an
        # action that cannot be read as numbers at all is none to start from.
        from unittest import mock

        for extra in (float("nan"), "fast"):
            with self.subTest(extra=extra):
                rig = self.trial_rig()
                real = fakes_rig.FakePolicy.select_action

                def with_a_seventh(policy, observation, real=real, extra=extra):
                    return {**real(policy, observation), "speed": extra}

                with mock.patch.object(fakes_rig.FakePolicy, "select_action", with_a_seventh):
                    run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
                self.assertEqual(run.events("run"), [], "an action with a value that is no number reached an attempt")
                self.assertEqual([(record.outcome, record.void_reason) for record in self.records(rig)],
                                 [("void", "the first action was not a number; no attempt was made")])

    def test_a_readout_reads_the_joint_furthest_from_where_it_is_wanted_and_fits_the_countdown_line(self):
        from tolerance_radius.rig import trial_session as real

        wanted = {"wrist_roll.pos": -86.4, "shoulder_pan.pos": 4.0}
        self.assertEqual(real.handoff_readout(wanted, {}, 5.0), "wrist roll wanted -86.4")
        seen = {"wrist_roll.pos": -94.2, "shoulder_pan.pos": 3.1}
        self.assertEqual(real.handoff_readout(wanted, seen, 5.0), "wrist roll now -94.2, wanted -86.4")
        seen["wrist_roll.pos"] = -87.0                      # within the tolerance now: every refused joint is
        self.assertEqual(real.handoff_readout(wanted, seen, 5.0), "shoulder pan now +3.1, wanted +4.0 ok")
        seen["shoulder_pan.pos"] = -4.0                     # ... and the other one moved away meanwhile
        self.assertEqual(real.handoff_readout(wanted, seen, 5.0), "shoulder pan now -4.0, wanted +4.0")
        self.assertEqual(real.handoff_readout({}, seen, 5.0), "")
        longest = real.handoff_readout({"shoulder_lift.pos": -100.0}, {"shoulder_lift.pos": -102.9}, 5.0)
        self.assertEqual(longest, "shoulder lift now -102.9, wanted -100.0 ok")
        self.assertLessEqual(len(longest), real.READOUT_WIDTH)
        # ... in a correction of under ten minutes, which is every correction: the countdown then reads m:ss
        self.assertLessEqual(2 + real.READOUT_WIDTH + len(": 9:59 elapsed -- press 'n' to end"), 78)

    def test_a_live_label_reads_as_its_name_except_in_the_countdown_line_of_its_own_phase(self):
        from tolerance_radius.rig import trial_session as real

        said = ["wrist roll now -94.2, wanted -86.4"]
        label = real.LiveLabel("Trial 4 [PP75] -- reposition", lambda: said[0])
        self.assertEqual(label, "Trial 4 [PP75] -- reposition")
        self.assertTrue(label.endswith("-- reposition"))
        self.assertEqual(f"  {label}: 0:05 elapsed", "  Trial 4 [PP75] -- reposition: 0:05 elapsed")
        def drawn(name: str, clock: str = "0:05") -> str:
            """What the tooling's countdown draws: it formats the name in a thread of its own."""
            lines: list[str] = []
            thread = threading.Thread(target=lambda: lines.append(f"  {name}: {clock} elapsed"))
            thread.start()
            thread.join()
            return lines[0]

        self.assertEqual(drawn(label), "  Trial 4 [PP75] -- reposition: 0:05 elapsed")       # not live yet
        label.live = True
        self.assertEqual(drawn(label), "  wrist roll now -94.2, wanted -86.4: 0:05 elapsed")
        # in the session's own thread it stays the name: the tooling's alignment gate prints it there
        self.assertEqual(f"Alignment gate before '{label}'", "Alignment gate before 'Trial 4 [PP75] -- reposition'")
        self.assertEqual("%s -- teleop live" % label, "Trial 4 [PP75] -- reposition -- teleop live")   # the log
        self.assertEqual(str(label), "Trial 4 [PP75] -- reposition")
        said[0] = "wrist roll now -87.0, wanted -86.4 ok"
        self.assertIn("ok: 0:06", drawn(label, "0:06"))

        def broken() -> str:
            raise KeyError("wrist_roll.pos")

        failing = real.LiveLabel("Reset", broken)
        failing.live = True
        self.assertEqual(drawn(failing), "  Reset: 0:05 elapsed", "a readout that fails leaves the phase's name")

    def test_void_typed_at_the_score_prompt_voids_the_trial_with_its_reason(self):
        rig = self.trial_rig()
        rig.operator.answer("score 0-5", "void bumped the table")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        void, scored = self.records(rig)
        self.assertEqual((void.outcome, void.void_reason, void.subgoal), ("void", "bumped the table", None))
        self.assertEqual(void.execution["steps_sent"], 7)
        self.assertEqual((scored.outcome, scored.planned.target), ("success", void.planned.target))
        asked = [event[1] for event in run.events("ask") if "score" in event[1]]
        self.assertEqual(len(asked), 2, "the scorer was asked again after typing void")

    def test_a_trial_nobody_scores_is_void_and_asked_again(self):
        rig = self.trial_rig()
        rig.operator.answer("score 0-5", *["?"] * 6)
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        void, scored = self.records(rig)
        self.assertEqual((void.outcome, void.void_reason), ("void", "left unscored"))
        self.assertEqual(scored.planned.target, void.planned.target)

    def test_a_failure_keeps_its_label_and_notes_and_typing_them_does_not_drive_the_session(self):
        rig = self.trial_rig()
        rig.operator.answer("score 0-5", "2")
        rig.operator.answer("failure", "grasp")
        rig.operator.answer("notes", "grasp:corner, ran into the basket, quit trying")  # full of n, r and q
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        (record,) = self.records(rig)
        self.assertEqual((record.outcome, record.subgoal, record.failure, record.notes),
                         ("failure", 2, "grasp", "grasp:corner, ran into the basket, quit trying"))
        self.assertEqual(run.events("typing-overheard"), [])
        self.assertEqual(run.story("teleop")[-2:], [f"teleop [{RESET}] ran", f"teleop [{COOL_DOWN}] ran"])


@unittest.skipUnless(READY, WHY_NOT)
class TrialStopTests(TrialCase):
    def executor_commands(self, rig) -> list:
        return [command for command in rig.commands if command["loop"] == "run"]

    def test_q_during_a_placement_starts_no_trial(self):
        rig = self.trial_rig()
        rig.operator.quiet("placement", nth=0)  # the cube never reaches the target
        rig.operator.at("placement", nth=0, tick=3).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("placement", "handoff", "run", "score", "record", "teleop", "settle"), [
            f"teleop [{WARM_UP}] ran",
            "placement [Trial 1 [PB89]] at (75.0, 25.0): stopped",
            "settle seeded",
        ])
        self.assertEqual(self.records(rig), [])
        self.assertEqual(self.executor_commands(rig), [])

    def test_q_while_the_cube_is_steady_starts_no_trial_and_moves_nothing(self):
        """The defect this pins: q pressed with the cube steady on its target was taken as the go.

        The stop key sets the same flag as the go key. The placement ended as accepted, the handoff gate passed, the
        executor's first action went to the motors (the arm moved, after the operator had asked to stop, with hands
        possibly still at the cube), and the scorer was asked to score a trial that never ran.
        """
        for executor, argv in (("policy", {"policies": ["clustered"]}), ("replay", {"replay": True})):
            with self.subTest(executor=executor):
                rig = self.trial_rig()
                rig.operator.quiet("placement", nth=0, what="end")
                rig.operator.at("placement", nth=0, tick=12).press("q")  # steady since frame 7
                block = "T1a" if executor == "policy" else "T1b"
                run = rig.run_trials(self.argv(rig, block, **argv))
                self.assertEqual(run.rc, 0)
                self.kept(run)
                self.assertEqual(run.events("placement")[0][3], "stopped")
                self.assertEqual(run.events("attempt", "handoff", "run", "score", "record"), [])
                self.assertEqual(self.executor_commands(rig), [], "an executor's action reached the motors")
                self.assertEqual([event for event in run.events("ask") if "score" in event[1]], [])
                self.assertEqual(self.records(rig), [])
                self.assertEqual(run.story("teleop", "settle")[-2:], [f"teleop [{WARM_UP}] ran", "settle seeded"])

    def test_q_after_the_placement_was_accepted_but_before_the_attempt_starts_no_trial(self):
        """The first time a demonstration is replayed its recorded actions are read from disk, after the go.

        A stop pressed in that moment used to be ignored until the executor's first action had gone to the motors.
        """
        rig = self.trial_rig()
        rig.operator.at("load", nth=0).press("q")
        run = rig.run_trials(self.argv(rig, "T1b", replay=True))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "accepted")
        self.assertEqual(run.events("attempt", "handoff", "run", "score", "record"), [])
        self.assertEqual(self.executor_commands(rig), [], "an executor's action reached the motors after the stop")
        self.assertEqual(self.records(rig), [])
        self.assertEqual(run.story("teleop", "settle")[-2:], [f"teleop [{WARM_UP}] ran", "settle seeded"])

    def test_q_while_repositioning_after_a_refused_start_ends_the_trial_without_handing_over_the_motors(self):
        """After a refused start the operator moves the arm in a teleoperation phase, and the cycle arms again.

        A q pressed in that phase ended it like any key; the cycle then judged the new pose, found it close enough,
        and sent the executor's first action before anything looked at the stop.
        """
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0]  # refused once; the second opening move would pass
        rig.operator.at("teleop", "reposition", nth=0, tick=1).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("handoff", "run", "score", "record", "settle"), [
            "handoff REFUSED",
            "record T1a-001 void (the session was stopped while the start pose was being corrected after a refusal; "
            "no attempt was made)",
            "settle seeded",
        ])
        self.assertEqual(self.executor_commands(rig), [], "an executor's action reached the motors after the stop")
        self.assertEqual([record.outcome for record in self.records(rig)], ["void"])

    def test_r_or_the_left_arrow_during_a_steady_placement_does_not_start_the_trial(self):
        """LeRobot's listener makes r and the left arrow "end this phase", as n and the right arrow are. Here only
        the go key confirms a placement: what follows is an arm that moves."""
        rig = self.trial_rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=12).press("r")
        rig.operator.at("placement", nth=0, tick=30).cube(None)
        rig.operator.at("placement", nth=0, tick=31).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "stopped")
        self.assertEqual(run.events("attempt", "run", "record"), [])

    def test_an_attempt_ended_by_a_key_as_it_starts_is_void_and_asked_again(self):
        """The go key of the placement is also the key that ends an attempt. An attempt that lasted under the
        protocol's shortest attempt tested nothing: it is logged void, with no score asked, and its place in the plan
        is asked again."""
        rig = self.trial_rig()
        rig.operator.at("run", nth=0, tick=1).press("n")
        # The flows run at 200 Hz: 0.02 s is four steps, and an ordinary attempt here is seven.
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"],
                                       extra=["--limit", "2", "--min-attempt-seconds", "0.02"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        first, second = self.records(rig)
        self.assertEqual(first.outcome, "void")
        self.assertIn("ended by a key", first.void_reason)
        self.assertEqual((second.outcome, second.planned.target), ("success", first.planned.target))
        self.assertEqual(len(run.events("score")), 1)

    def test_q_during_an_attempt_ends_it_at_once_and_then_the_session(self):
        rig = self.trial_rig()
        rig.operator.at("run", nth=0, tick=2).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        # The August runner offered the stopped attempt for scoring, and so did this one until the dry run of
        # 2026-10-06, where a replay stopped after 11 of 40 steps was scored 5 and logged a success. What the stop
        # key cut short is void, is not scored, and is asked again; then nothing else moves.
        self.assertEqual(run.story("run", "score", "record", "teleop", "settle")[1:],
                         ["run policy:clustered: 3 steps",
                          "record T1a-001 void (the session was stopped during the attempt, after 0.0 s)",
                          "settle seeded"])
        self.assertEqual(run.events("score"), [])
        self.assertEqual(len(self.executor_commands(rig)), 3)
        self.assertIn("Logged void; it will be asked again", run.out)
        (record,) = self.records(rig)
        self.assertEqual(record.outcome, "void")
        # ... and the next session asks for the same place again, as a trial of its own
        again = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(again)
        void, asked_again = self.records(rig)
        self.assertEqual((void.outcome, asked_again.outcome), ("void", "success"))
        self.assertEqual((asked_again.planned.target, asked_again.planned.level_mm, asked_again.planned.staircase_id),
                         (void.planned.target, void.planned.level_mm, void.planned.staircase_id))

    def test_a_key_pressed_before_the_cube_is_steady_does_not_start_a_trial(self):
        rig = self.trial_rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=2).press("n")
        rig.operator.at("placement", nth=0, tick=39).cube(None)
        rig.operator.at("placement", nth=0, tick=40).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.kept(run)
        self.assertEqual(run.events("attempt", "run", "record"), [])
        self.assertEqual(run.events("placement")[0][3], "stopped")

    def test_a_go_key_left_over_from_the_reset_or_the_score_does_not_start_the_next_trial(self):
        rig = self.trial_rig()
        rig.operator.at("score", nth=0).press("n")                    # while the listener is paused: unheard
        rig.operator.at("teleop", "Reset", nth=0, tick=2).press("n")  # twice: the second press is left over
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        self.assertEqual([event[5] for event in run.events("placement")], [11, 11])
        self.assertEqual(len(self.records(rig)), 2)

    def test_cameras_that_are_not_confirmed_end_the_session_before_any_loop(self):
        rig = self.trial_rig()
        rig.operator.answer("match their names", "n")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 1)
        self.kept(run)
        self.assertEqual(run.story("cameras", "listener:start", "settle"), ["cameras NOT CONFIRMED", "settle seeded"])
        self.assertEqual(rig.commands, [])

    def test_a_view_that_no_longer_matches_and_a_dead_camera_end_the_session_with_code_3(self):
        for fault in ("view", "camera"):
            with self.subTest(fault=fault):
                rig = self.trial_rig()
                if fault == "view":
                    rig.view_agrees = False
                else:
                    rig.dead_camera = "front"
                run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
                self.assertEqual(run.rc, 3)
                self.kept(run)
                self.assertIn("REFUSING TO RUN", run.out)
                self.assertEqual(run.events("loop", "listener:start"), [])
                self.assertEqual(run.events("settle"), [("settle", "seeded")])

    def test_a_refused_preflight_loads_no_policy_and_opens_nothing_else(self):
        rig = self.trial_rig()
        rig.arm.torque = False
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 1)
        self.assertEqual(run.events("policy:load", "follower:built", "follower:connect", "leader:connect"), [])
        self.assertFalse(rig.arm.torque)

    def test_a_policy_that_does_not_match_the_rig_is_refused_before_the_robot_is_built(self):
        rig = self.trial_rig()
        rig.policy_features["policy:clustered"] = {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 240, 320]}}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 1)
        self.assertIn("does not match this rig's cameras or joints", run.err)
        self.assertEqual(run.events("follower:built", "follower:connect", "leader:connect"), [])

    def test_a_blocked_arm_ends_the_attempt_by_the_stall_rule(self):
        rig = self.trial_rig()
        rig.policy_rate = 4.0  # six degrees a step: the goal runs away from an arm that does not move
        rig.operator.quiet("run")
        rig.operator.at("run", nth=0, tick=0).block_arm()
        rig.operator.at("teleop", "Reset", nth=0, tick=0).block_arm(False)
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"],
                                       extra=["--limit", "1", "--stall-steps", "5"]))
        self.kept(run)
        (ran,) = run.events("run")
        self.assertLess(ran[3], 12, "the attempt went on pushing against the obstacle")
        self.assertIn("STALLED", run.out)
        self.assertEqual(len(self.records(rig)), 1)


LAPTOP_S_OWN = ("MacBook Pro Camera", "6C707041-0000")
FRONT_ON_A_HUB = ("HD Pro Webcam C920", "0x1100000aaaa")
WRIST_ON_A_HUB = ("USB2.0_CAM1", "0x1140000bbbb")
WRIST_IN_THE_COMPUTER_S_PORT = ("USB2.0_CAM1", "0x100000bbbb")


@unittest.skipUnless(READY, WHY_NOT)
class CamerasByNameTests(TrialCase):
    """The sessions take their cameras by number, and a number belongs to whatever camera is plugged in.

    2026-10-06, 19:22: the wrist camera had not started, number 1 was the laptop's own camera, and a recording
    session took it for the wrist. The operator saw it in the viewer; the session said nothing.
    """

    def refused(self, run, word: str) -> None:
        self.assertEqual(run.rc, 3)
        self.assertIn(f"REFUSING TO {word}", run.out)
        story = run.story()
        self.assertNotIn("preflight ok", story)
        self.assertFalse(any(line.startswith(("follower:connect", "leader:connect", "listener:start"))
                             for line in story), story)

    def sessions(self, rig):
        yield "RECORD", lambda extra=(): rig.record_layout(["--sitting", "1", "--limit", "1", *extra])
        yield "RUN", lambda extra=(): rig.run_trials(
            self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1", *extra]))

    def test_a_wrist_camera_that_did_not_start_is_refused_before_anything_is_opened(self):
        for word, _ in self.sessions(self.trial_rig()):
            with self.subTest(session=word):
                rig = self.trial_rig()
                rig.cameras_offered = [LAPTOP_S_OWN, FRONT_ON_A_HUB]
                run = dict(self.sessions(rig))[word]()
                self.refused(run, word)
                self.assertIn("the wrist camera ('USB2.0_CAM1') is not among the cameras the system offers", run.out)
                self.assertIn("camera 1 is now 'MacBook Pro Camera'", run.out)
                self.assertTrue(any("cameras by name: camera 1: MacBook Pro Camera" in message and "WRONG" in message
                                    for _level, message in rig.logged), rig.logged)

    def test_two_cameras_at_each_other_s_numbers_are_refused(self):
        # 19:41 the same evening: the wrist camera in the computer's own port, which comes first.
        for word, _ in self.sessions(self.trial_rig()):
            with self.subTest(session=word):
                rig = self.trial_rig()
                rig.cameras_offered = [LAPTOP_S_OWN, FRONT_ON_A_HUB, WRIST_IN_THE_COMPUTER_S_PORT]
                run = dict(self.sessions(rig))[word]()
                self.refused(run, word)
                self.assertIn("the front camera ('HD Pro Webcam C920') looks to be camera 1", run.out)

    def test_the_rig_as_it_should_be_is_written_into_the_log_and_the_session_goes_on(self):
        for word, _ in self.sessions(self.trial_rig()):
            with self.subTest(session=word):
                rig = self.trial_rig()
                run = dict(self.sessions(rig))[word]()
                self.assertEqual(run.rc, 0)
                self.kept(run)
                told = [message for _level, message in rig.logged if message.startswith("cameras by name: ")]
                self.assertEqual(told, [
                    "cameras by name: camera 0: HD Pro Webcam C920  (the session's front camera)",
                    "cameras by name: camera 1: USB2.0_CAM1  (the session's wrist camera)",
                    "cameras by name: camera 2: MacBook Pro Camera  (not used)",
                ])

    def test_a_list_that_cannot_be_read_stops_nothing_and_is_said(self):
        from tolerance_radius.rig import camera_list

        for word, _ in self.sessions(self.trial_rig()):
            with self.subTest(session=word):
                rig = self.trial_rig()
                rig.cameras_offered = camera_list.CameraListError("the system's camera list could not be read")
                run = dict(self.sessions(rig))[word]()
                self.assertEqual(run.rc, 0)
                self.kept(run)
                self.assertTrue(any(message.startswith("cameras by name: not checked")
                                    for _level, message in rig.logged), rig.logged)

    def test_the_operator_can_run_all_the_same_and_the_log_says_so(self):
        for word, _ in self.sessions(self.trial_rig()):
            with self.subTest(session=word):
                rig = self.trial_rig()
                rig.cameras_offered = [LAPTOP_S_OWN, FRONT_ON_A_HUB, WRIST_IN_THE_COMPUTER_S_PORT]
                run = dict(self.sessions(rig))[word](("--any-cameras",))
                self.assertEqual(run.rc, 0)
                self.assertTrue(any(level == "WARNING" and "(--any-cameras)" in message
                                    for level, message in rig.logged), rig.logged)


@unittest.skipUnless(READY, WHY_NOT)
class TrialTypingGuardTests(TrialCase):
    """A trial's go hands the motors to a policy. The keys are heard from every program: a go that is a letter of
    somebody's sentence must start nothing. (2026-10-06: a message typed elsewhere during a recording session,
    twice; the second one while a placement waited.)"""

    #: Quiet times short enough for a test and long against the flows' keys, which come within microseconds.
    GUARD = ("--typing-guard", "0.3", "0.1")
    SENTENCE = "I need to go"

    def typed_while_the_cube_is_steady(self, rig) -> None:
        rig.operator.quiet("placement", nth=0, what="end")     # the careful operator's own n is held back
        for char in self.SENTENCE:                               # the cube has been steady since frame 7
            rig.operator.at("placement", nth=0, tick=12).press("space" if char == " " else char)

    def executor_commands(self, rig) -> list:
        return [command for command in rig.commands if command["loop"] == "run"]

    def test_a_go_typed_in_a_sentence_starts_no_trial_and_moves_nothing(self):
        for executor, argv in (("policy", {"policies": ["clustered"]}), ("replay", {"replay": True})):
            with self.subTest(executor=executor):
                rig = self.trial_rig()
                self.typed_while_the_cube_is_steady(rig)
                rig.operator.at("placement", nth=0, tick=60).press("q")
                block = "T1a" if executor == "policy" else "T1b"
                run = rig.run_trials(self.argv(rig, block, **argv, extra=self.GUARD))
                self.assertEqual(run.rc, 0)
                self.kept(run)
                self.assertEqual(run.events("placement")[0][3], "stopped")
                self.assertEqual(self.records(rig), [])
                self.assertEqual(self.executor_commands(rig), [], "the arm moved on a key typed in a sentence")
                self.assertIn("middle of typing", run.out)
                self.assertTrue(any("1 go key(s) refused as typing" in message for _level, message in rig.logged))

    def test_without_the_guard_the_same_sentence_starts_the_trial(self):
        """What the guard is for, shown: the listener's own rule takes the sentence's n for the go."""
        rig = self.trial_rig()
        self.typed_while_the_cube_is_steady(rig)
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--no-typing-guard", "--limit", "1"]))
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "accepted")
        self.assertNotEqual(self.executor_commands(rig), [])
        self.assertTrue(any(message == "typing guard: off" for _level, message in rig.logged))

    def test_after_a_refused_go_an_n_pressed_by_itself_starts_the_trial(self):
        rig = self.trial_rig()
        self.typed_while_the_cube_is_steady(rig)
        rig.operator.at("placement", nth=0, tick=13).do(lambda _rig, _phase: time.sleep(0.4))   # the typing is over
        rig.operator.at("placement", nth=0, tick=14).press("n")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=[*self.GUARD, "--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.out.count("middle of typing"), 1)
        self.assertEqual(run.events("placement")[0][3], "accepted")
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])

    def test_an_n_pressed_by_itself_starts_the_trial_as_before(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=[*self.GUARD, "--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertNotIn("middle of typing", run.out)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])
        self.assertTrue(any(message.startswith("typing guard: a go starts a trial when no other key came 0.3 s")
                            for _level, message in rig.logged))

    def test_a_go_pressed_just_before_the_cube_is_steady_starts_nothing_with_the_guard_s_own_times(self):
        # The guard holds a go for its quiet time after the key. A go pressed while the cube was not yet steady was
        # still standing when it became steady, and started the trial (the review of 2026-10-06). The flows' other
        # tests of an early press run with quiet times of nought, which hid it.
        rig = self.trial_rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.operator.at("placement", nth=0, tick=2).press("n")        # the cube is steady from frame 7 on
        rig.operator.at("placement", nth=0, tick=60).do(lambda _rig, _phase: time.sleep(0.3))
        rig.operator.at("placement", nth=0, tick=61).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=self.GUARD))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "stopped")
        self.assertEqual(self.executor_commands(rig), [], "a go pressed before the cube was steady started the trial")
        self.assertEqual(self.records(rig), [])
        self.assertEqual(run.out.count("came while the cube was not steady"), 1)

    def test_a_go_typed_in_a_sentence_while_repositioning_does_not_start_the_policy(self):
        # After a refused start the n that ends the reposition arms the policy again: it is no placement's go, and
        # nothing looked at it (the review of 2026-10-06: a typed sentence there started the policy).
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0]              # refused once; the second start would pass
        rig.operator.quiet("teleop", "reposition", nth=0, what="end")    # the careful operator's own n is held back
        for char in self.SENTENCE:
            rig.operator.at("teleop", "reposition", nth=0, tick=9).press("space" if char == " " else char)
        rig.operator.at("teleop", "reposition", nth=1, tick=0).do(lambda _rig, _phase: time.sleep(0.4))
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=[*self.GUARD, "--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        story = run.story("handoff", "teleop", "run")
        self.assertEqual(story[story.index("handoff REFUSED"):][:5], [
            "handoff REFUSED",
            "teleop [Trial 1 [PB89] -- reposition] ran",       # ended by the sentence's n: asked for again
            "teleop [Trial 1 [PB89] -- reposition] ran",       # ended by an n pressed by itself
            "handoff ok",
            "run policy:clustered: 7 steps",
        ])
        self.assertEqual(run.out.count("middle of typing"), 1)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])

    def test_the_runner_s_own_quiet_times(self):
        from tolerance_radius.rig import trial_session

        args = trial_session.build_parser().parse_args(["--blocks", "T1a", "--log", "x.jsonl"])
        self.assertEqual((args.typing_guard, args.no_typing_guard), ([2.0, 0.5], False))


@unittest.skipUnless(READY, WHY_NOT)
class TrialDryRunFindingsTests(TrialCase):
    """What a dry run of runbook 3 against this runner found on 2026-10-06, each pinned where it was put right."""

    def test_an_audit_of_a_part_of_a_set_is_said_in_a_pilot_and_refused_in_a_counted_session(self):
        # After one sitting the clustered set holds 33 episodes. Audited as it is, a session started without a word,
        # with other demonstrations to replay and other distances than the finished set gives.
        rig = self.rig()
        rig.write_audits("clustered", episodes=33)
        rig.write_audits("spread")
        rig.write_replay_dataset(episodes=33)
        pilot = rig.run_trials(self.argv(rig, "T1b", replay=True, extra=["--limit", "1"]))
        self.assertEqual(pilot.rc, 0)
        self.kept(pilot)
        self.assertIn("NOTE -- the clustered audit holds 33 of the set's 55 episodes", pilot.out + pilot.err)
        rig.protocol_changes = fakes_rig.frozen
        counted = rig.run_trials(self.argv(rig, "T1b", log="main.jsonl", pilot=False, replay=True))
        self.assertEqual(counted.rc, 1)
        self.assertIn("the clustered audit holds 33 of the set's 55 episodes", counted.err)
        self.assertIn("a counted session needs the finished sets' audits", counted.err)
        self.assertEqual(counted.events("follower:connect"), [])

    def test_a_replay_set_that_is_not_the_audited_recording_is_refused(self):
        rig = self.rig()
        rig.write_audits()
        rig.write_replay_dataset(episodes=27)                 # another recording than the 55 the audit measured
        run = rig.run_trials(self.argv(rig, "T1b", replay=True))
        self.assertEqual(run.rc, 1)
        self.assertIn("holds 27 episodes and the clustered audit ends at episode 54: they are not the same "
                      "recording", run.err)
        self.assertEqual(run.events("follower:connect"), [])

    def test_the_placement_window_s_own_keys_start_and_end_nothing_in_a_trial(self):
        # Space in the window was a second go, and v and x ended the session: keys of the camera tools. In a trial
        # the one go is the session's own key, which the typing guard looks at.
        from tolerance_radius.rig.placement_phase import KEY_ABORT, KEY_ACCEPT, KEY_SKIP

        rig = self.trial_rig()
        rig.operator.quiet("placement", nth=0, what="end")
        rig.window_keys = [-1] * 12 + [KEY_ACCEPT, KEY_SKIP, KEY_ABORT, KEY_ACCEPT]    # steady from frame 7 on
        rig.operator.at("placement", nth=0, tick=40).press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.events("placement")[0][3], "stopped", "the window's keys must not accept, skip or abort")
        self.assertEqual(run.events("run", "record"), [])
        self.assertEqual(rig.window_keys, [], "the window was still given its turns")

    def test_a_pilot_says_how_far_each_trial_is_from_a_demonstration_and_a_counted_session_does_not(self):
        rig = self.trial_rig()
        pilot = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(pilot)
        self.assertRegex(pilot.out, r"Trial 1 \[\w+\]: target \([\d.]+, [\d.]+\) mm   \(pilot: [\d.]+ mm from the nearest")
        rig = self.trial_rig()
        rig.protocol_changes = fakes_rig.frozen
        counted = rig.run_trials(self.argv(rig, "T1a", log="main.jsonl", pilot=False, policies=["clustered"],
                                           extra=["--limit", "1"]))
        self.kept(counted)
        self.assertIn("Trial 1 [", counted.out)
        self.assertNotIn("from the nearest demonstration", counted.out)
        # The same trial of the same block is shown under another code in a pilot than in the counted session: a
        # block is one condition, so a pilot under the counted codes would tell which of them is a replay and which
        # a policy (the walk through the runbooks, 2026-10-07).
        self.assertIn("Trial 1 [PB89]", pilot.out)
        self.assertIn("Trial 1 [W57]", counted.out)
        self.assertNotIn("W57", pilot.out)

    def test_an_attempt_is_not_made_in_a_picture_that_is_no_longer_the_session_s(self):
        """The front camera's stream changed its zoom, its exposure and its tone range in the middle of sessions on
        2026-10-06. A policy must not be run, and a trial must not be scored, in a picture it was not trained in."""
        rig = self.trial_rig()
        changed = "the front camera's picture stands 108 mm from the one this session started on (allowed 5)"
        rig.picture_changed_after = {1: changed, 2: changed, 3: changed}      # before the second attempt, and it stays
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.kept(run)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])   # the first trial only
        # the trial is sent back to its placement twice, and the third look that is not clean ends the session
        self.assertEqual([event[3] for event in run.events("placement")], ["accepted"] * 4)
        self.assertEqual(len(run.events("run")), 1, "no attempt in a picture that changed")
        self.assertEqual(run.out.count("then press n again"), 2)
        self.assertEqual(run.rc, 3, "a session that a changed picture ended does not end as one that went well")
        self.assertIn("Before the attempt, the front camera's picture stands 108 mm", run.out)
        self.assertIn("nothing is logged; the session stops here", run.out)
        # ... at once: there is no cool-down to wait in
        self.assertEqual([line for line in run.story("teleop") if "Cool down" in line], [])
        self.assertEqual(run.story("settle"), ["settle seeded"])
        # the same command again goes on with the trial that was not made
        rig.picture_changed_after = {}
        again = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(again)
        self.assertEqual(len(self.records(rig)), 2)

    def test_a_phase_s_countdown_line_fits_the_terminal_and_what_to_do_is_said_once(self):
        """The first run at the rig, 2026-10-07: with the instruction in a phase's name the tooling's countdown
        line was 95 characters, wrapped in a terminal 87 wide, and every rewrite left a row behind."""
        from tolerance_radius.rig import trial_session as real

        for name in (real.WARM_UP, real.RESET, real.COOL_DOWN):
            self.assertLessEqual(len(name) + real.COUNTDOWN_BESIDE_NAME, 78, name)
        self.assertEqual((WARM_UP, RESET, COOL_DOWN), (real.WARM_UP, real.RESET, real.COOL_DOWN))
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        self.assertEqual(run.out.count("Bring the arm to the start pose, then press n."), 1)
        self.assertEqual(run.out.count("Cube out of the basket, arm back to the start pose, then press n."), 2)
        self.assertEqual(run.out.count("Park the arm at its final rest pose, then press n."), 1)

    def test_a_camera_that_has_crept_too_far_ends_the_session_before_the_attempt_and_the_next_goes_on(self):
        """2026-10-07: the front camera moves as it warms, 1.3 and 1.5 mm in half an hour from cold, and a session
        measures every placement through the conversion it solved at its start. The live positions of two
        recording sessions came to stand 1.2 and 1.7 mm from where the episodes' own frames have the cube; a
        trial's start is what the session measured, and would be off by as much."""
        rig = self.trial_rig()
        rig.crept_after = {0: 0.4, 1: 1.3, 2: 1.24}             # over the limit, and over it again at the new go
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.kept(run)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])   # the first trial only
        self.assertEqual(self.records(rig)[0].execution["view_crept_mm"], 0.4, "the figure is in the trial's record")
        self.assertEqual(len(run.events("run")), 1, "no attempt from a start that is off by the creep")
        self.assertEqual(run.rc, 3)
        self.assertEqual(rig.creeps_asked, 3, "a reading over the limit is asked once more before it stops a session")
        # ... at a new go, and not by itself: the second trial's placement was asked for twice
        self.assertEqual(len([event for event in run.events("placement") if "Trial 2" in str(event[1])]), 2)
        self.assertIn("the front camera's picture reads 1.30 mm from where this session solved its conversion "
                      "(allowed 1)", run.out)
        self.assertIn("then press n again: a second reading over the limit ends the session", run.out)
        self.assertIn("has crept 1.24 mm since this session solved its conversion (allowed 1), after 1.30 mm at "
                      "an earlier go of this trial", run.out)
        self.assertIn("nothing is logged; the session stops here", run.out)
        told = run.out.split("has crept 1.24 mm")[1]
        self.assertIn("Take the cube off the table, then run the same command again: a new session solves the "
                      "conversion afresh", told)
        self.assertNotIn("measure.py check", told, "a camera that crept is not a camera to be looked into")
        # ... at once, as with a picture that changed
        self.assertEqual([line for line in run.story("teleop") if "Cool down" in line], [])
        self.assertEqual(run.story("settle"), ["settle seeded"])
        # the same command again solves the conversion afresh and goes on with the trial that was not made
        rig.crept_after = {}
        again = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(again)
        self.assertEqual(again.rc, 0)
        self.assertEqual(len(self.records(rig)), 2)

    def test_a_creep_under_the_limit_stops_nothing_and_its_figure_is_in_the_record_to_a_hundredth(self):
        rig = self.trial_rig()
        rig.crept_after = {0: 0.9, 1: 0.447, 2: 1.0}            # ... and at the limit is not over it
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.kept(run)
        self.assertEqual(run.rc, 0)
        self.assertEqual(rig.creeps_asked, 3, "asked once before every attempt, after the look at the picture")
        self.assertEqual([record.execution["view_crept_mm"] for record in self.records(rig)], [0.9, 0.45, 1.0])
        self.assertEqual(len(run.events("placement")), 3, "no placement was asked for twice")
        self.assertNotIn("press n again", run.out)

    def test_a_creep_that_cannot_be_read_sends_the_trial_back_to_its_placement(self):
        # No attempt from a start whose measurement nobody can vouch for: a look that cannot say is a look that is
        # not clean (until the night of 2026-10-07 the attempt went on, and its record said null).
        rig = self.trial_rig()
        rig.crept_after = {0: None, 1: 0.3}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual((run.rc, rig.creeps_asked), (0, 2))
        self.assertEqual(len(run.events("placement")), 2, "the go was asked for again")
        self.assertEqual(len(run.events("run")), 1)
        self.assertEqual([record.execution["view_crept_mm"] for record in self.records(rig)], [0.3])
        self.assertIn("could not be read (a floor tag hidden for a moment, or no frame).\n  Nothing has moved. "
                      "Hands and the arm out of the picture, then press n again.", run.out)
        # ... and three such looks in a row end the session, as three looks at a picture that is not the session's
        blind = self.trial_rig()
        blind.crept_after = {0: None, 1: None, 2: None}
        run = blind.run_trials(self.argv(blind, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual((run.rc, blind.creeps_asked, len(run.events("run"))), (3, 3, 0))
        self.assertEqual(len(self.records(blind)), 0)
        self.assertIn("No attempt is made and nothing is logged", run.out)
        self.assertIn("measure.py check", run.out, "a picture that cannot be read is one to be looked into")
        self.assertNotIn("Take the cube off the table", run.out, "... and not a camera that has crept")

    def test_a_creep_reading_over_the_limit_that_is_under_it_at_the_next_go_stops_nothing(self):
        """The reading follows what stands in the picture by a few tenths: a shadow's edge along a floor tag's side
        read 0.2 to 0.9 mm on drawn tags with nothing else amiss (the review of the night of 2026-10-07). A
        reading that changes between two looks is something standing in the picture, so nothing moves on its
        account without a new go (the second review, the same night)."""
        rig = self.trial_rig()
        rig.crept_after = {0: 1.3, 1: 0.8, 2: 1.1}     # trial 1: over, then under; trial 2: over, then nothing at all
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        self.assertEqual(run.rc, 0)
        self.assertEqual(rig.creeps_asked, 4)
        self.assertNotIn("Take the cube off the table", run.out, "no session was ended for the creep")
        self.assertEqual(run.out.count("then press n again: a second reading over the limit ends the session"), 2)
        self.assertEqual(len(run.events("placement")), 4, "each trial's go was asked for again")
        # nothing ran between the reading over the limit and the new go
        order = [event[0] for event in run.events("placement", "run")]
        self.assertEqual(order, ["placement", "placement", "run", "placement", "placement", "run"])
        # the record carries the reading of the look that let the attempt go
        self.assertEqual([record.execution["view_crept_mm"] for record in self.records(rig)], [0.8, 0.0])

    def test_a_reading_over_the_limit_then_one_that_cannot_say_then_one_over_ends_the_session(self):
        # A look that cannot say takes nothing back: two readings over the limit, and nothing under it between.
        rig = self.trial_rig()
        rig.crept_after = {0: 1.3, 1: None, 2: 1.2}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual((run.rc, rig.creeps_asked, len(run.events("run"))), (3, 3, 0))
        self.assertEqual(len(self.records(rig)), 0)
        self.assertIn("has crept 1.20 mm since this session solved its conversion (allowed 1), after 1.30 mm at "
                      "an earlier go of this trial", run.out)

    HAND = ("the floor tags could not be seen (no tag 0 in the picture): whether the front camera's picture is "
            "still the session's cannot be said")

    def test_a_look_with_a_hand_in_it_does_not_take_back_a_reading_over_the_limit(self):
        # over, a hand, over: the session ends at the third go, and no fourth is asked for
        rig = self.trial_rig()
        rig.crept_after = {0: 1.3, 1: 1.2}
        rig.picture_changed_after = {1: self.HAND}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual((run.rc, rig.creeps_asked, len(run.events("placement")), len(run.events("run"))),
                         (3, 2, 3, 0))
        self.assertIn("has crept 1.20 mm since this session solved its conversion (allowed 1), after 1.30 mm at "
                      "an earlier go of this trial", run.out)

    def test_a_reading_over_the_limit_is_not_one_of_the_three_looks_at_the_picture(self):
        # a hand, over, a hand, under: the attempt is made at the fourth go. Were the reading over the limit one of
        # the three looks that end a session, the second hand would have been the third.
        rig = self.trial_rig()
        rig.picture_changed_after = {0: self.HAND, 2: self.HAND}
        rig.crept_after = {0: 1.3, 1: 0.2}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual((run.rc, len(run.events("placement")), len(run.events("run"))), (0, 4, 1))
        self.assertEqual([record.execution["view_crept_mm"] for record in self.records(rig)], [0.2])

    def test_the_looks_that_are_not_clean_are_counted_for_each_trial_anew(self):
        # two in the first trial and two in the second: four in the session, and never three for one trial
        rig = self.trial_rig()
        rig.picture_changed_after = {0: self.HAND, 1: self.HAND, 3: self.HAND, 4: self.HAND}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        self.assertEqual((run.rc, len(run.events("placement")), len(run.events("run"))), (0, 6, 2))

    def test_the_wrist_camera_is_looked_at_after_a_creep_reading_that_was_asked_again(self):
        # over, under, and then the wrist camera in its other tone state, twice: no attempt in that picture
        rig = self.trial_rig()
        rig.crept_after = {0: 1.3, 1: 0.6}
        rig.wrist_readings = {2: 3.20, 3: 3.20}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual((run.rc, len(run.events("run"))), (3, 0))
        self.assertEqual(len(self.records(rig)), 0)
        self.assertIn("wrist camera", run.out)

    def test_a_session_says_at_its_start_that_the_cube_must_be_off_the_table(self):
        # Before the answer that the conversion is solved after, and so before anything is measured.
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        said = run.out.index("The cube must be OFF the table now")
        self.assertLess(said, run.out.index("Camera snapshots"))
        self.assertEqual(run.out.count("The cube must be OFF the table now"), 1)

    def test_another_limit_for_the_creep_is_the_one_that_is_held(self):
        rig = self.trial_rig()
        rig.crept_after = {0: 1.3}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"],
                                       extra=["--limit", "1", "--view-crept-limit", "2"]))
        self.kept(run)
        self.assertEqual((run.rc, rig.creeps_asked, len(self.records(rig))), (0, 1, 1))
        # a pilot may, says so, and its record carries the limit it ran with
        self.assertIn("view_crept_limit_mm is 2.0 in this session and 1.0 in the protocol", run.out)
        self.assertEqual(self.records(rig)[0].execution["view_crept_limit_mm"], 2.0)
        tighter = self.trial_rig()
        tighter.crept_after = {0: 0.6, 1: 0.6}
        run = tighter.run_trials(self.argv(tighter, "T1a", policies=["clustered"],
                                           extra=["--limit", "1", "--view-crept-limit", "0.5"]))
        self.assertEqual((run.rc, tighter.creeps_asked), (3, 2))
        self.assertEqual(len(self.records(tighter)), 0)
        self.assertIn("has crept 0.60 mm since this session solved its conversion (allowed 0.5)", run.out)

    def test_the_creep_is_not_asked_of_a_picture_that_is_not_the_session_s(self):
        rig = self.trial_rig()
        rig.picture_changed_after = {0: "the floor tags could not be seen (no tag 0 in the picture): whether the "
                                        "front camera's picture is still the session's cannot be said"}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.kept(run)
        self.assertEqual(rig.creeps_asked, 1, "only after the look that was clean")

    def test_the_stop_for_a_creep_can_be_switched_off_and_the_session_says_so(self):
        rig = self.trial_rig()
        rig.crept_after = {0: 2.4, 1: None}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"],
                                       extra=["--limit", "2", "--view-crept-limit", "0"]))
        self.kept(run)
        self.assertEqual(run.rc, 0)
        self.assertIn("NOTE: --view-crept-limit 0", run.out)
        # ... and then it is read once for each trial, stops nothing, and is in the record all the same
        self.assertEqual(rig.creeps_asked, 2)
        self.assertEqual(len(run.events("placement")), 2)
        self.assertEqual([record.execution["view_crept_mm"] for record in self.records(rig)], [2.4, None])

    def test_a_look_that_is_not_clean_sends_the_trial_back_to_its_placement_and_nothing_starts_by_itself(self):
        """A hand over a floor tag when the go is pressed. The look before the attempt used to wait until it was
        clean and then went straight on: the arm started the instant the hand was lifted, up to twenty seconds
        after the go and with nobody having said so (the review of the night of 2026-10-06)."""
        rig = self.trial_rig()
        rig.picture_changed_after = {0: "the floor tags could not be seen (no tag 0 in the picture): whether the "
                                        "front camera's picture is still the session's cannot be said"}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        kinds = [event[0] for event in run.trace if event[0] in ("placement", "picture:checked", "run")]
        self.assertEqual(kinds, ["placement", "picture:checked", "placement", "picture:checked", "run"],
                         "the attempt must follow a go of its own, pressed after the look that was not clean")
        self.assertEqual(run.out.count("Nothing has moved. Hands and the arm out of the picture, then press n again"), 1)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])
        # the look is one look, made at that moment: no patience, no announcer to wait with
        self.assertEqual(rig.picture_looks, [{"tries": 1, "patience_s": 0.0}] * 2)

    def test_the_re_record_key_does_not_end_a_reposition_into_the_policy(self):
        """After a refused start the n that ends the reposition arms the policy again. r and the left arrow end
        the phase too and are no go keys: the guard has heard no go, and it said yes (the review of the night of
        2026-10-06: "sorry" typed there, or the left arrow alone, gave the policy the motors without a word)."""
        for key, extra in (("r", TrialTypingGuardTests.GUARD), ("left", TrialTypingGuardTests.GUARD), ("r", ("--no-typing-guard",))):
            with self.subTest(key=key, guard=extra[0]):
                rig = self.trial_rig()
                rig.wild_first_actions["policy:clustered"] = [30.0]              # refused once
                rig.operator.quiet("teleop", "reposition", nth=0, what="end")
                rig.operator.at("teleop", "reposition", nth=0, tick=9).press(key)
                rig.operator.at("teleop", "reposition", nth=1, tick=0).do(lambda _rig, _phase: time.sleep(0.4))
                run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=[*extra, "--limit", "1"]))
                self.assertEqual(run.rc, 0)
                self.kept(run)
                story = run.story("handoff", "teleop", "run")
                self.assertEqual(story[story.index("handoff REFUSED"):][:5], [
                    "handoff REFUSED",
                    "teleop [Trial 1 [PB89] -- reposition] ran",       # ended by the key that is no go
                    "teleop [Trial 1 [PB89] -- reposition] ran",       # ended by n
                    "handoff ok",
                    "run policy:clustered: 7 steps",
                ])
                self.assertEqual(run.out.count("That key is not the go"), 1)
                self.assertEqual([record.outcome for record in self.records(rig)], ["success"])

    def test_a_doubled_go_at_the_end_of_a_reposition_does_not_cut_the_attempt_short(self):
        rig = self.trial_rig()
        rig.wild_first_actions["policy:clustered"] = [30.0]
        rig.operator.quiet("teleop", "reposition", nth=0, what="end")
        rig.operator.at("teleop", "reposition", nth=0, tick=9).press("n")

        def second_press_while_the_go_is_looked_at(rig_, _phase):
            # the guard holds the go for its quiet time after the key: the second n lands in that wait
            import threading

            threading.Timer(0.03, lambda: rig_.press("n")).start()

        rig.operator.at("teleop", "reposition", nth=0, tick=9).do(second_press_while_the_go_is_looked_at)
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=[*TrialTypingGuardTests.GUARD, "--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("run"), ["run policy:clustered: 7 steps"])
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])

    def test_a_re_record_key_pressed_in_the_reset_does_not_swallow_the_next_trial_s_go(self):
        # r ends a reset like any key and leaves its own flag set; the next placement's n, pressed alone, was then
        # dropped without a word, and the placement waited.
        rig = self.trial_rig()
        rig.operator.quiet("teleop", "Reset", nth=0, what="end")
        rig.operator.at("teleop", "Reset", nth=0, tick=3).press("r")
        rig.operator.at("placement", nth=1, tick=0).do(lambda _rig, _phase: time.sleep(0.4))   # the r is long past
        rig.operator.at("placement", nth=1, tick=60).press("q")       # if the go was swallowed: do not wait ten minutes
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=[*TrialTypingGuardTests.GUARD, "--limit", "2"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(len(run.events("run")), 2, "the second trial's go was swallowed")
        self.assertEqual([record.outcome for record in self.records(rig)], ["success", "success"])

    def test_a_stop_pressed_after_the_attempt_is_over_does_not_void_it(self):
        # q while the videos are being finished: the attempt was complete, and it is scored. Read after the videos,
        # the stop voided it (the review of the night of 2026-10-06).
        rig = self.trial_rig()
        rig.on_video_finish = lambda: rig.press("q")
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"], "scored, then the session ends")

    def test_an_attempt_is_not_made_with_the_wrist_camera_in_its_other_tone_state(self):
        rig = self.trial_rig()
        rig.wrist_readings = {3: 3.20, 4: 3.20}       # before the second attempt, and the reading that confirms it
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.assertEqual(run.rc, 3)
        self.kept(run)
        self.assertEqual(len(run.events("run")), 1)
        self.assertEqual([record.outcome for record in self.records(rig)], ["success"])
        self.assertIn("Before the attempt, the wrist camera's picture has changed its tone state since this session "
                      "began", run.out)
        # ... and a session is not begun in the other state: said before the warm-up, refused at the start pose
        rig = self.trial_rig()
        rig.wrist_readings = {0: 3.20, 1: 3.20}
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "3"]))
        self.assertEqual(run.rc, 3)
        self.kept(run)
        self.assertIn("Wrist camera in its other state. No trial is run.", run.said())
        self.assertIn("NOTE: the wrist camera's picture is not in its usual tone state", run.out)
        self.assertIn("Before any trial, the wrist camera's picture is not in its usual tone state", run.out)
        self.assertEqual(run.events("placement", "run"), [])
        self.assertEqual(self.records(rig), [])
        # ... unless the operator says that either state will do: then it is written down and nothing is stopped
        rig = self.trial_rig()
        rig.wrist_readings = {0: 3.20, 1: 3.20, 2: 3.20}       # the other state from the start; back before trial 2
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2", "--any-wrist-state"]))
        self.kept(run)
        self.assertEqual(len(run.events("run")), 2)
        self.assertEqual(sum(level == "WARNING" and "(--any-wrist-state)" in message for level, message in rig.logged), 2)
        # the option is said at the start of every session that takes it: a trial's record does not carry it
        self.assertIn("NOTE: --any-wrist-state.", run.out)
        self.assertTrue(any(level == "WARNING" and message.startswith("--any-wrist-state:")
                            for level, message in rig.logged))

    def test_the_wrist_camera_is_not_asked_before_an_attempt_with_the_arm_away_from_the_session_s_first_pose(self):
        # Away from that pose the wrist camera's number says nothing (frames from the middle of the takes of
        # 2026-10-06 read as either state): a trial must not be lost to it.
        from unittest import mock

        from tolerance_radius.rig import record_layout, trial_session

        rig = self.trial_rig()
        rig.wrist_readings = {2: 3.20, 3: 3.20, 4: 3.20, 5: 3.20}         # would be "changed", were it asked
        asked = []

        def away(reference, now, tolerance_deg, free_tolerance_deg=None):   # the recorder's rule says: not there
            asked.append((tolerance_deg, free_tolerance_deg))
            return ["shoulder_lift +9.0 deg (allowed 5)"]

        with mock.patch.object(record_layout, "off_start_pose", away):
            run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.assertEqual(asked, [trial_session.WRIST_ASKED_WITHIN_DEG] * 2)
        self.assertEqual(trial_session.WRIST_ASKED_WITHIN_DEG, (5.0, 12.0))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(len(run.events("run")), 2)
        self.assertEqual(rig.wrist_read, 2, "asked before the warm-up and at the start pose only")
        self.assertTrue(any("wrist camera before the attempt: not asked" in message for _level, message in rig.logged))

    def test_the_picture_is_looked_at_before_every_attempt(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "2"]))
        self.kept(run)
        kinds = [event[0] for event in run.trace if event[0] in ("picture:checked", "run")]
        self.assertEqual(kinds, ["picture:checked", "run"] * 2)


@unittest.skipUnless(READY, WHY_NOT)
class TrialDryRunTests(TrialCase):
    def test_a_dry_run_gives_the_motors_to_no_executor_and_writes_nothing(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1b", replay=True, extra=["--dry-run", "--limit", "2"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertEqual(run.story("plan", "placement", "handoff", "run", "score", "record", "video:start")[1:], [
            # The pilot's first trial: one of its eleven exact starts, here the demonstration at the centre of C2.
            "plan T1b-001 replay", "placement [Trial 1 [PL94]] at (125.0, 75.0): accepted", "handoff ok",
            "plan T1b-001 replay", "placement [Trial 2 [PL94]] at (125.0, 75.0): accepted", "handoff ok",
        ])
        self.assertEqual({command["loop"] for command in rig.commands}, {"teleop"})
        self.assertFalse(rig.trial_log().exists())
        self.assertFalse((rig.repo_root / "data").exists(), "a dry run left recordings behind")
        self.assertTrue(any("DRY RUN -- would command" in message for _level, message in rig.logged))
        self.assertEqual([event for event in run.events("ask") if "score" in event[1]], [])

    def test_a_dry_run_is_allowed_before_the_protocol_is_frozen_and_into_any_file(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1b", log="main.jsonl", pilot=False, replay=True,
                                       extra=["--dry-run", "--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertFalse(rig.trial_log("main.jsonl").exists())


@unittest.skipUnless(READY, WHY_NOT)
class TrialLogRuleTests(TrialCase):
    """Which sessions may write to which log (the rule added with `--pilot`), through `main` itself."""

    def refused(self, rig, argv, said):
        run = rig.run_trials(argv)
        self.assertEqual(run.rc, 1)
        self.assertIn(said, run.err)
        self.assertEqual(run.events("pin", "bus:open", "policy:load", "follower:connect"), [])

    def test_a_counted_session_is_refused_when_an_execution_setting_differs_from_the_protocol(self):
        rig = self.trial_rig()
        rig.protocol_changes = fakes_rig.frozen
        self.refused(rig, self.argv(rig, "T1b", log="main.jsonl", pilot=False, replay=True, extra=["--clamp", "10"]),
                     "clamp_deg is 10.0 in this session and 15.0 in the protocol")

    def test_a_counted_session_cannot_run_with_another_limit_for_the_camera_s_creep(self):
        # Until the freeze the limit was an option's default: a counted session could have switched the stop off,
        # and no record would have said so.
        for given, shown in (("0", "0.0"), ("2", "2.0")):
            with self.subTest(limit=given):
                rig = self.trial_rig()
                rig.protocol_changes = fakes_rig.frozen
                self.refused(rig, self.argv(rig, "T1b", log="main.jsonl", pilot=False, replay=True,
                                            extra=["--view-crept-limit", given]),
                             f"view_crept_limit_mm is {shown} in this session and 1.0 in the protocol")

    def test_a_pilot_may_change_a_setting_and_says_so(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1b", replay=True, extra=["--clamp", "10", "--limit", "1"]))
        self.assertEqual(run.rc, 0)
        self.kept(run)
        self.assertIn("clamp_deg is 10.0 in this session and 15.0 in the protocol", run.out)
        self.assertEqual(self.records(rig)[0].execution["clamp_deg"], 10.0)

    def test_counted_trials_are_refused_until_the_protocol_is_frozen(self):
        rig = self.trial_rig()
        self.refused(rig, self.argv(rig, "T1b", log="main.jsonl", pilot=False, replay=True),
                     "counted trials are run only after it is frozen")

    def test_a_pilot_session_logs_only_to_a_pilot_file_and_a_counted_one_never_does(self):
        rig = self.trial_rig()
        self.refused(rig, self.argv(rig, "T1b", log="main.jsonl", replay=True),
                     "a pilot session logs to a file whose name starts with 'pilot'")
        rig.protocol_changes = fakes_rig.frozen
        self.refused(rig, self.argv(rig, "T1b", log="pilot_replay.jsonl", pilot=False, replay=True),
                     "pilot_replay.jsonl is a pilot log")

    def test_a_log_holding_trials_of_another_protocol_version_is_refused(self):
        rig = self.trial_rig()
        first = rig.run_trials(self.argv(rig, "T1b", log="pilot_a.jsonl", replay=True, extra=["--limit", "1"]))
        self.assertEqual(first.rc, 0)

        def later_version(protocol):
            return {**protocol, "protocol": {**protocol["protocol"], "version": "pilot-2"}}

        rig.protocol_changes = later_version
        self.refused(rig, self.argv(rig, "T1b", log="pilot_a.jsonl", replay=True), "already holds trials run under")

    def test_a_pilot_session_and_a_counted_session_under_a_frozen_protocol_both_run(self):
        rig = self.trial_rig()
        pilot = rig.run_trials(self.argv(rig, "T1b", log="pilot_replay.jsonl", replay=True, extra=["--limit", "1"]))
        self.assertEqual(pilot.rc, 0)
        self.kept(pilot)
        self.assertEqual(self.records(rig, "pilot_replay.jsonl")[0].versions["protocol_status"], "pilot")
        rig.protocol_changes = fakes_rig.frozen
        counted = rig.run_trials(self.argv(rig, "T1b", log="main.jsonl", pilot=False, replay=True,
                                           extra=["--limit", "1"]))
        self.assertEqual(counted.rc, 0)
        self.kept(counted)
        (record,) = self.records(rig, "main.jsonl")
        self.assertEqual((record.versions["protocol_status"], record.versions["protocol_version"]),
                         ("frozen", str(load_protocol()["protocol"]["version"])))

    def test_a_pilot_block_has_the_pilot_s_size(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "9"]))
        self.kept(run)
        records = self.records(rig)
        self.assertEqual(len(records), 9)
        self.assertTrue(all(record.planned.level_mm == 0.0 for record in records))  # nine exact starts, no staircase
        again = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertIn("Nothing to do", again.out)

    def test_other_refusals_on_paper(self):
        for build, said in (
            (lambda rig: self.argv(rig, "T9", replay=True), "unknown blocks"),
            (lambda rig: self.argv(rig, "T1a"), "no usable --policy act_clustered=DIR"),
            (lambda rig: self.argv(rig, "T1b"), "replay trials need --replay-root"),
        ):
            with self.subTest(said=said):
                rig = self.trial_rig()
                self.refused(rig, build(rig), said)
        rig = self.rig()  # no audit written
        rig.write_replay_dataset()
        self.refused(rig, self.argv(rig, "T1b", replay=True), "no start-state audit")


@unittest.skipUnless(READY, WHY_NOT)
class TrialBlindTests(TrialCase):
    def windows(self, run) -> list[str]:
        """What could be seen or heard from the moment each trial was planned until its score was entered."""
        texts, current = [], None
        for channel, text in run.perceived:
            if channel == "mark" and text.startswith("plan "):
                current = []
            elif channel == "mark" and text == "score entered":
                texts.append("\n".join(current or []))
                current = None
            elif current is not None:
                current.append(text)
        return texts

    def test_between_two_policies_nothing_shown_or_said_before_the_score_tells_which_one_runs(self):
        rig = self.trial_rig()
        rig.protocol_changes = fakes_rig.frozen
        run = rig.run_trials(self.argv(rig, "T2", log="main.jsonl", pilot=False, policies=["clustered", "spread"],
                                       extra=["--limit", "4"]))
        self.kept(run)
        records = self.records(rig, "main.jsonl")
        windows = self.windows(run)
        self.assertEqual(len(windows), 4)
        for record, seen in zip(records, windows, strict=True):
            with self.subTest(trial=record.planned.trial_id):
                self.assertIn(record.planned.blind_code, seen)  # the window is the right one, and not empty
                for word in ("act_clustered", "act_spread", "clustered", "spread", "T2", "080000", "pretrained"):
                    self.assertNotIn(word, seen)

    def test_with_a_replay_in_the_session_the_block_and_the_condition_are_not_named_either(self):
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", "T1b", policies=["clustered"], replay=True, extra=["--limit", "6"]))
        self.kept(run)
        windows = self.windows(run)
        self.assertEqual(len(windows), 6)
        for record, seen in zip(self.records(rig), windows, strict=True):
            with self.subTest(trial=record.planned.trial_id):
                self.assertIn(record.planned.blind_code, seen)
                for word in ("act_clustered", "clustered", "replay", "Replay", "T1a", "T1b", "080000", "episode"):
                    self.assertNotIn(word, seen)

    def test_known_limit_the_handoff_line_shows_a_tolerance_that_is_5_for_a_replay_and_8_for_a_policy(self):
        """Pinned so that nobody takes the terminal for blind between a policy and the replay.

        The handoff gate is the arm tooling's own and logs its tolerance. A replay is held to 5 degrees and a policy to
        8, so the line on the terminal tells the two apart. Between two policies (block T2) it tells nothing.
        """
        rig = self.trial_rig()
        run = rig.run_trials(self.argv(rig, "T1a", "T1b", policies=["clustered"], replay=True, extra=["--limit", "6"]))
        for record, seen in zip(self.records(rig), self.windows(run), strict=True):
            told = "(tolerance 5.0)" if record.planned.condition == "replay" else "(tolerance 8.0)"
            self.assertIn(told, seen)


@unittest.skipUnless(READY, WHY_NOT)
class TrialFaultTests(TrialCase):
    def test_an_exception_in_the_middle_of_an_attempt_still_settles_the_arm(self):
        rig = self.trial_rig()
        rig.operator.at("run", nth=0, tick=2).fail_read(ConnectionError("Failed to sync read 'Present_Position'"))
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertIsInstance(run.error, ConnectionError)
        self.kept(run)
        self.assertEqual(run.story("record", "follower:disconnect", "settle", "listener:stop")[-3:],
                         ["follower:disconnect torque on", "settle seeded", "listener:stop"])
        self.assertEqual(run.events("record"), [])
        self.assertEqual(self.records(rig), [])
        self.assertEqual(rig.preflight_problems(), [])

    def test_the_same_fault_through_cli_is_an_exit_code_with_the_traceback_logged(self):
        """The launcher ends with `leave(cli())`: an exception that escapes `cli` skips that way out.

        The process would then leave through the interpreter's own shutdown, which the launcher's comment says can
        hang on the viewer, and the traceback would be on the terminal only, not in the session log.
        """
        rig = self.trial_rig()
        rig.operator.at("run", nth=0, tick=2).fail_read(ConnectionError("Failed to sync read 'Present_Position'"))
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]), through_cli=True)
        self.assertIsNone(run.error, "the fault escaped cli()")
        self.assertEqual(run.rc, 2)
        self.kept(run)
        self.assertIn(("ERROR", "SESSION FAILED"), rig.logged)
        self.assertIn("arm: torque ON", run.out)

    def test_an_exception_in_the_middle_of_a_placement_still_settles_the_arm(self):
        rig = self.trial_rig()
        rig.operator.at("placement", nth=0, tick=4).fail_read(TimeoutError("latest frame is too old: 612.0 ms"))
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertIsInstance(run.error, TimeoutError)
        self.kept(run)
        self.assertEqual(run.events("settle"), [("settle", "seeded")])

    def test_ctrl_c_at_the_score_prompt_settles_the_arm_and_logs_nothing(self):
        rig = self.trial_rig()
        rig.operator.answer("score 0-5", KeyboardInterrupt())
        run = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]), through_cli=True)
        self.assertEqual(run.rc, 130)
        self.kept(run)
        self.assertEqual(self.records(rig), [])
        self.assertEqual(run.events("settle"), [("settle", "seeded")])

    def test_after_a_session_that_died_the_next_one_starts_and_asks_the_same_trial(self):
        rig = self.trial_rig()
        rig.operator.at("run", nth=0, tick=2).fail_read(ConnectionError("Failed to sync read 'Present_Position'"))
        died = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"]))
        self.assertIsNotNone(died.error)
        rig.operator = fakes_rig.Operator()
        again = rig.run_trials(self.argv(rig, "T1a", policies=["clustered"], extra=["--limit", "1"]))
        self.assertEqual(again.rc, 0)
        self.kept(again)
        self.assertEqual(again.story("preflight", "record"), ["preflight ok", "record T1a-001 success"])


# ====================================================================================================================
# The arm tooling's own attempt cycle, given a replay in place of a policy
# ====================================================================================================================


@unittest.skipUnless(READY, WHY_NOT)
class ReplayThroughTheProvenCycleTests(unittest.TestCase):
    """The real `attempt_cycle`, `first_action_for_handoff` and `run_attempt`, with a `ReplaySource` as the policy.

    The robot is the arm tooling's own test fake (`FakeRobot`, and `MovingArm`, which travels toward its goal), loaded
    from its test suite, so that what is exercised here is the proven cycle against the proven fakes.
    """

    JOINTS = list(fakes_rig.JOINTS)

    @classmethod
    def setUpClass(cls):
        from tolerance_radius.rig.replay import ReplaySource

        cls.rollout = fakes_rig.prepare()["rollout_session"]
        spec = importlib.util.spec_from_file_location(
            "arm_tooling_rollout_gate_tests", fakes_rig.TOOLING_ROOT / "tests" / "test_rollout_session_gates.py")
        cls.proven = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.proven)
        cls.ReplaySource = ReplaySource

    def recording(self, steps: int = 12, per_step: float = 1.0) -> np.ndarray:
        """A recorded episode that starts at the neutral pose and bends the elbow by `per_step` each step."""
        actions = np.zeros((steps, len(self.JOINTS)))
        actions[:, self.JOINTS.index("elbow_flex")] = per_step * np.arange(steps)
        return actions

    def cycle(self, robot, source, events, *, tolerance=5.0, seconds=30.0, reenter=None, dry_run=False):
        started = time.monotonic()
        with self.assertLogs(level="INFO"):  # the cycle reports through logging: keep it off the test output
            ran = self.rollout.attempt_cycle(
                robot=robot, policy=source, predict=source.predict, gate=self.rollout.PolicyHandoffGate(tolerance),
                fps=500, seconds=seconds, label="replay", events=events, max_retries=3,
                reenter_teleop=reenter or (lambda _label: True), dry_run=dry_run)
        return ran, time.monotonic() - started

    def events(self):
        return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def elbows(self, robot):
        return [action["elbow_flex.pos"] for action in robot.sent]

    def test_a_start_pose_too_far_from_the_recorded_first_pose_is_refused_and_nothing_is_sent(self):
        robot = self.proven.FakeRobot(present=self.proven.pose(elbow_flex=6.0))  # six degrees from the recording
        events = self.events()
        source = self.ReplaySource(self.recording(), self.JOINTS, observe=robot.get_observation, events=events)
        repositions = []
        ran, _ = self.cycle(robot, source, events, reenter=lambda label: repositions.append(label) or True)
        self.assertFalse(ran)
        self.assertEqual(robot.sent, [])
        self.assertEqual(len(repositions), 2)       # two chances to move the arm, then it gives up
        self.assertEqual(source.steps_handed_out, 1)  # rewound each time: only the first action was ever judged
        self.assertFalse(events["exit_early"])

    def test_the_same_start_pose_would_pass_a_policy_s_looser_gate(self):
        robot = self.proven.FakeRobot(present=self.proven.pose(elbow_flex=6.0))
        events = self.events()
        source = self.ReplaySource(self.recording(3), self.JOINTS, observe=robot.get_observation, events=events)
        ran, _ = self.cycle(robot, source, events, tolerance=8.0)
        self.assertTrue(ran)

    def test_the_attempt_ends_when_the_recording_is_exhausted_long_before_the_time_limit(self):
        robot = self.proven.MovingArm(speed=8.0)
        events = self.events()
        source = self.ReplaySource(self.recording(12), self.JOINTS, observe=robot.get_observation, events=events)
        ran, took = self.cycle(robot, source, events, seconds=30.0)
        self.assertTrue(ran)
        self.assertEqual(self.elbows(robot), [float(step) for step in range(12)])  # every action, once, in order
        self.assertLess(took, 5.0)
        self.assertFalse(events["exit_early"], "the replay's end signal was left for the next phase to trip on")

    def test_after_a_refusal_and_a_reposition_the_replay_starts_again_from_its_first_action(self):
        robot = self.proven.FakeRobot(present=self.proven.pose(elbow_flex=12.0))
        events = self.events()
        source = self.ReplaySource(self.recording(6), self.JOINTS, observe=robot.get_observation, events=events)

        def reposition(_label):
            robot.present = self.proven.pose()  # the operator brings the arm to the recorded first pose
            return True

        ran, _ = self.cycle(robot, source, events, reenter=reposition)
        self.assertTrue(ran)
        self.assertEqual(self.elbows(robot), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # not from the second action

    def test_the_clamp_is_the_robot_s_own_the_replay_hands_over_the_recorded_values_unchanged(self):
        robot = self.proven.FakeRobot(present=self.proven.pose())
        events = self.events()
        actions = self.recording(4)
        actions[2:, self.JOINTS.index("elbow_flex")] += 40.0  # a recorded jump of 41 degrees between two steps
        source = self.ReplaySource(actions, self.JOINTS, observe=robot.get_observation, events=events)
        ran, _ = self.cycle(robot, source, events)
        self.assertTrue(ran)
        self.assertEqual(self.elbows(robot), [0.0, 1.0, 42.0, 43.0])  # asked for as recorded
        self.assertAlmostEqual(source.largest_step(), 41.0)
        # What the follower does with such a jump: the same helper the real `send_action` calls caps it at the clamp.
        with FakeRig() as rig:
            config = rig._follower_config(port="none", id="my_follower", max_relative_target=15.0,
                                          disable_torque_on_disconnect=False,
                                          cameras=fakes_rig.prepare()["so101_session"].CAMERAS)
            follower = fakes_rig.FakeFollower(rig, config)
            follower.connect()
            before = rig.arm.present["elbow_flex"]
            with self.assertLogs("lerobot.robots.utils", level="WARNING"):  # LeRobot says so when it clamps
                asked = {f"{joint}.pos": rig.arm.present[joint] + (41.0 if joint == "elbow_flex" else 0.0)
                         for joint in self.JOINTS}
                sent = follower.send_action(asked)
            self.assertAlmostEqual(sent["elbow_flex.pos"], before + 15.0)

    def test_a_dry_run_sends_nothing_and_the_replay_is_whole_for_the_next_use(self):
        robot = self.proven.MovingArm(speed=8.0)
        events = self.events()
        source = self.ReplaySource(self.recording(5), self.JOINTS, observe=robot.get_observation, events=events)
        ran, _ = self.cycle(robot, source, events, dry_run=True)
        self.assertTrue(ran)
        self.assertEqual(robot.sent, [])
        ran, _ = self.cycle(robot, source, events)
        self.assertEqual(self.elbows(robot), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_a_recording_whose_joints_are_not_the_robot_s_is_refused_not_played(self):
        robot = self.proven.FakeRobot(present=self.proven.pose())
        events = self.events()
        source = self.ReplaySource(self.recording(4), [f"motor_{i}" for i in range(6)],
                                   observe=robot.get_observation, events=events)
        ran, _ = self.cycle(robot, source, events)
        self.assertFalse(ran)
        self.assertEqual(robot.sent, [])

    def test_a_stop_ends_a_replay_at_once_and_the_stop_is_kept_for_the_session(self):
        robot = self.proven.MovingArm(speed=8.0)
        events = self.events()
        source = self.ReplaySource(self.recording(400), self.JOINTS, observe=robot.get_observation, events=events)
        original = source.predict

        def predict():
            if source.steps_handed_out == 5:
                events["stop_recording"] = events["exit_early"] = True  # what q does
            return original()

        source.predict = predict
        ran, _ = self.cycle(robot, source, events)
        self.assertTrue(ran)
        self.assertLess(len(robot.sent), 10)
        self.assertEqual((events["exit_early"], events["stop_recording"]), (False, True))


# ====================================================================================================================
# What can be read from the sources without running anything
# ====================================================================================================================


def _method_parameters(source: Path, class_name: str, method: str) -> tuple[list[str], set[str]]:
    """A method's parameter names in order (without `self` or `cls`), and those that have no default."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    positional = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args]][1:]
    required = set(positional[: len(positional) - len(function.args.defaults)])
    keyword_only = [arg.arg for arg in function.args.kwonlyargs]
    required |= {name for name, default in zip(keyword_only, function.args.kw_defaults, strict=True) if default is None}
    return [*positional, *keyword_only], required


def _calls_in(source: Path, receiver: str, method: str) -> list[ast.Call]:
    return [node for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == method
            and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver]


RIG_SOURCES = REPO_ROOT / "src" / "tolerance_radius" / "rig"


class DatasetCallTests(unittest.TestCase):
    """The recorder opens its dataset the way LeRobot wants: both the checkout the sessions import and the copy
    installed in the rig's environment (0.6.0), which is what runs if the pin is ever bypassed."""

    def lerobot_sources(self) -> dict[str, Path]:
        found = {}
        if fakes_rig.LEROBOT_ROOT is not None:
            found["the pinned checkout"] = (fakes_rig.LEROBOT_ROOT / "src" / "lerobot" / "datasets"
                                            / "lerobot_dataset.py")
        for folder in site.getsitepackages():
            installed = Path(folder) / "lerobot" / "datasets" / "lerobot_dataset.py"
            if installed.is_file():
                found["the environment's copy"] = installed
        return found

    def test_create_and_resume_are_called_with_arguments_both_lerobot_versions_accept(self):
        sources = self.lerobot_sources()
        if not sources:
            self.skipTest("no LeRobot source on this machine")
        for method in ("create", "resume"):
            (call,) = _calls_in(RIG_SOURCES / "record_layout.py", "LeRobotDataset", method)
            keywords = {keyword.arg for keyword in call.keywords}
            for which, source in sources.items():
                with self.subTest(method=method, lerobot=which):
                    names, required = _method_parameters(source, "LeRobotDataset", method)
                    self.assertLessEqual(len(call.args), len(names))
                    self.assertFalse(keywords - set(names), f"{method}() does not take {sorted(keywords - set(names))}")
                    given = set(names[: len(call.args)]) | keywords
                    self.assertFalse(required - given, f"{method}() also needs {sorted(required - given)}")

    def test_resume_is_given_the_dataset_s_directory(self):
        """LeRobot refuses `resume` without `root` (the August recorder's own --resume omits it and cannot work)."""
        (call,) = _calls_in(RIG_SOURCES / "record_layout.py", "LeRobotDataset", "resume")
        self.assertIn("root", {keyword.arg for keyword in call.keywords})


class SessionStructureTests(unittest.TestCase):
    """More checks that read the two sessions' sources (see also `test_rig_sessions.SessionSafetyStructureTests`)."""

    def main_of(self, name: str) -> ast.FunctionDef:
        tree = ast.parse((RIG_SOURCES / name).read_text(encoding="utf-8"))
        return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main")

    @staticmethod
    def called(node: ast.AST) -> list[str]:
        return [n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                for n in ast.walk(node) if isinstance(n, ast.Call)]

    def test_the_arm_is_connected_only_inside_the_try_whose_finally_settles_it(self):
        for name in ("record_layout.py", "trial_session.py"):
            with self.subTest(session=name):
                main = self.main_of(name)
                guarded = [node for node in ast.walk(main) if isinstance(node, ast.Try) and node.finalbody
                           and "settle_arm" in self.called(ast.Module(body=node.finalbody, type_ignores=[]))]
                self.assertEqual(len(guarded), 1)
                inside = self.called(ast.Module(body=guarded[0].body, type_ignores=[]))
                self.assertEqual(inside.count("connect"), 2)                 # the leader and the follower
                self.assertEqual(self.called(main).count("connect"), 2)      # and nowhere else in main
                first = guarded[0].body[:2]
                self.assertEqual([self.called(statement) for statement in first], [["connect"], ["connect"]],
                                 "something runs inside the try before the two arms are connected")

    def test_the_trial_session_enters_teleoperation_only_through_reenter_teleop(self):
        main = self.main_of("trial_session.py")
        reenter = next(node for node in ast.walk(main) if isinstance(node, ast.FunctionDef)
                       and node.name == "reenter_teleop")
        self.assertEqual(self.called(main).count("teleop_phase"), 1)
        self.assertEqual(self.called(reenter).count("teleop_phase"), 1)
        source = ast.unparse(reenter)
        self.assertLess(source.index("events['exit_early'] = False"), source.index("teleop_phase("))
        self.assertIn("display_data=False", source)
        self.assertIn("float('inf')", source)

    def test_the_trial_session_scores_with_the_listener_paused_and_never_asks_with_it_live(self):
        main = self.main_of("trial_session.py")
        self.assertEqual(self.called(main).count("score_quietly"), 1)
        self.assertNotIn("score_attempt", self.called(main))
        source = ast.unparse(main)
        self.assertEqual(source.count("input("), 1)  # the cube-off prompt; ScoreSheet(input) only passes it on
        self.assertLess(source.index("kb.pause()"), source.index("input("))

    def test_the_recorder_saves_only_through_the_announced_save(self):
        main = self.main_of("record_layout.py")
        self.assertNotIn("save_episode", self.called(main))
        self.assertEqual(self.called(main).count("save_episode_announced"), 1)

    def test_neither_session_offers_a_way_around_a_gate(self):
        for name in ("record_layout.py", "trial_session.py"):
            with self.subTest(session=name):
                text = (RIG_SOURCES / name).read_text(encoding="utf-8")
                for bypass in ("skip-preflight", "skip-alignment", "allow-input-mismatch", "enabled=False"):
                    self.assertNotIn(bypass, text)


if __name__ == "__main__":
    unittest.main()
