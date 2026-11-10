"""The data contracts: what a valid record is, what is refused, and that a log survives a round trip."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tolerance_radius import schema
from tolerance_radius.schema import DemoStart, PlannedTrial, Point, SchemaError, TrialRecord


def planned(**changes) -> PlannedTrial:
    base = PlannedTrial(
        trial_id="T1a-001",
        block="T1a",
        condition="act_clustered",
        blind_code="K7",
        target=Point(85.0, 25.0),
        level_mm=10.0,
        staircase_id="T1a-up",
        staircase_index=0,
    )
    return replace(base, **changes)


def record(**changes) -> TrialRecord:
    base = TrialRecord(
        planned=planned(),
        outcome="success",
        ts_start="2026-10-07T09:00:00-07:00",
        ts_end="2026-10-07T09:00:41-07:00",
        placed=Point(85.6, 24.7),
        nearest_demo_episode=3,
        nearest_demo_distance_mm=10.4,
        subgoal=5,
        operator="op1",
        scorer="sc1",
        policy_repo="anonymous/act-clustered",
        checkpoint="080000",
    )
    return replace(base, **changes)


class PointTests(unittest.TestCase):
    def test_distance_is_euclidean(self):
        self.assertAlmostEqual(Point(0, 0).distance_to(Point(30, 40)), 50.0)

    def test_refuses_values_that_are_not_finite_numbers(self):
        for bad in (float("nan"), float("inf"), "12", None, True):
            with self.subTest(bad=bad), self.assertRaises(SchemaError):
                Point(bad, 0.0)


class PlannedTrialTests(unittest.TestCase):
    def test_dataset_follows_the_condition(self):
        self.assertEqual(planned().dataset, "clustered")
        self.assertEqual(planned(condition="act_spread").dataset, "spread")
        replay = planned(condition="replay", replay_episode=4)
        self.assertEqual(replay.dataset, "clustered")

    def test_a_replay_names_its_episode_and_a_policy_trial_does_not(self):
        with self.assertRaises(SchemaError):
            planned(condition="replay")
        with self.assertRaises(SchemaError):
            planned(replay_episode=4)

    def test_staircase_fields_come_together(self):
        with self.assertRaises(SchemaError):
            planned(staircase_index=None)
        planned(staircase_id=None, staircase_index=None)

    def test_refuses_unknown_block_condition_and_negative_level(self):
        for changes in ({"block": "T9"}, {"condition": "diffusion"}, {"level_mm": -1.0}, {"trial_id": ""}):
            with self.subTest(changes=changes), self.assertRaises(SchemaError):
                planned(**changes)


class TrialRecordTests(unittest.TestCase):
    def test_a_success_has_the_top_subgoal_and_no_failure_label(self):
        self.assertTrue(record().success)
        with self.assertRaises(SchemaError):
            record(subgoal=4)
        with self.assertRaises(SchemaError):
            record(failure="grasp")

    def test_a_failure_names_its_kind(self):
        failed = record(outcome="failure", subgoal=1, failure="perception")
        self.assertTrue(failed.scored)
        self.assertFalse(failed.success)
        with self.assertRaises(SchemaError):
            record(outcome="failure", subgoal=1)
        with self.assertRaises(SchemaError):
            record(outcome="failure", subgoal=5, failure="grasp")

    def test_a_void_trial_states_its_reason_and_has_no_score(self):
        void = record(outcome="void", subgoal=None, void_reason="camera dropped", placed=None)
        self.assertFalse(void.scored)
        self.assertIsNone(void.placement_error_mm)
        with self.assertRaises(SchemaError):
            record(outcome="void", subgoal=None)
        with self.assertRaises(SchemaError):
            record(outcome="void", subgoal=2, void_reason="camera dropped")

    def test_a_scored_trial_records_the_placement_and_the_nearest_demonstration(self):
        with self.assertRaises(SchemaError):
            record(placed=None)
        with self.assertRaises(SchemaError):
            record(nearest_demo_distance_mm=None)

    def test_placement_error_is_the_distance_from_target_to_placed(self):
        self.assertAlmostEqual(record().placement_error_mm, Point(85.0, 25.0).distance_to(Point(85.6, 24.7)))

    def test_a_replay_has_no_policy_and_a_policy_trial_names_one(self):
        replay = planned(condition="replay", replay_episode=4, block="T1b", trial_id="T1b-001")
        record(planned=replay, policy_repo=None, checkpoint=None)
        with self.assertRaises(SchemaError):
            record(planned=replay)
        with self.assertRaises(SchemaError):
            record(checkpoint=None)

    def test_subgoal_must_be_an_integer_on_the_scale(self):
        for bad in (6, -1, 2.5, True):
            with self.subTest(bad=bad), self.assertRaises(SchemaError):
                record(outcome="failure", subgoal=bad, failure="grasp")


class JsonTests(unittest.TestCase):
    def test_a_trial_survives_a_round_trip_through_json(self):
        original = record(execution={"clamp_deg": 15.0}, versions={"protocol_version": "1.0"}, notes="clean grasp")
        restored = schema.trial_record_from_json(json.loads(json.dumps(schema.to_json(original))))
        self.assertEqual(restored, original)

    def test_a_demo_start_survives_a_round_trip_through_json(self):
        original = DemoStart("spread", 12, "C2", measured=Point(121.3, 80.9), planned=Point(121.0, 81.0), trained=False)
        restored = schema.demo_start_from_json(json.loads(json.dumps(schema.to_json(original))))
        self.assertEqual(restored, original)

    def test_unknown_fields_are_refused(self):
        data = schema.to_json(record())
        data["sucess"] = True
        with self.assertRaises(SchemaError):
            schema.trial_record_from_json(data)

    def test_a_point_needs_exactly_its_two_coordinates(self):
        data = schema.to_json(record())
        data["placed"] = {"x_mm": 1.0}
        with self.assertRaises(SchemaError):
            schema.trial_record_from_json(data)


class FileTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_appended_trials_are_read_back_in_order(self):
        log = self.dir / "trials" / "session.jsonl"
        first = record()
        second = record(planned=planned(trial_id="T1a-002", staircase_index=1), outcome="failure", subgoal=1, failure="grasp")
        schema.append_trial(log, first)
        schema.append_trial(log, second)
        self.assertEqual(schema.read_trials(log), [first, second])

    def test_a_missing_log_is_an_empty_log(self):
        self.assertEqual(schema.read_trials(self.dir / "none.jsonl"), [])

    def test_a_malformed_line_is_reported_with_its_line_number(self):
        log = self.dir / "session.jsonl"
        schema.append_trial(log, record())
        with open(log, "a", encoding="utf-8") as handle:
            handle.write('{"outcome": "success"}\n')
        with self.assertRaisesRegex(SchemaError, r"session\.jsonl:2"):
            schema.read_trials(log)

    def test_demo_starts_are_written_and_read_back(self):
        path = self.dir / "audit" / "clustered.json"
        starts = [DemoStart("clustered", i, "B1", measured=Point(75.0 + i, 25.0)) for i in range(3)]
        schema.write_demo_starts(path, starts)
        self.assertEqual(schema.read_demo_starts(path), starts)


if __name__ == "__main__":
    unittest.main()
