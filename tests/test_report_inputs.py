"""What the report reads and what it refuses: pilot logs, unfrozen trials, distances that do not match the audit,
trials of a condition that were not all run the same way."""

import ast
import json
import re
import shutil
import tempfile
import types
import unittest
from pathlib import Path

import fixtures_experiment as fixtures
from tolerance_radius import schema, second_scoring
from tolerance_radius.report import bookkeeping, inputs
from tolerance_radius.report.inputs import ReportError, load_inputs

#: The trial runner's source. It is read here, never imported: the analysis may not depend on the rig's modules, and
#: the rig's modules do not import where there is no camera library.
RUNNER = fixtures.REPO / "src" / "tolerance_radius" / "rig" / "trial_session.py"


class ExperimentOnDisk(unittest.TestCase):
    """A synthetic experiment written once for the class; each test reads it, or a changed copy of its log."""

    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        cls.experiment = fixtures.build_experiment(Path(cls._folder.name) / "experiment", coarse=True)
        cls.root = cls.experiment.root
        cls._variants = 0

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def changed_log(self, change, name: str | None = None) -> Path:
        """A copy of the counted log with `change(index, entry)` applied to every line, outside `data/trials`."""
        type(self)._variants += 1
        path = self.root / "changed" / (name or f"log_{self._variants}.jsonl")
        path.parent.mkdir(exist_ok=True)
        shutil.copy(self.experiment.log, path)
        fixtures.rewrite_log(path, change)
        return path

    def refusal(self, **arguments) -> str:
        """The sentence with which the given inputs are refused."""
        with self.assertRaises(ReportError) as caught:
            load_inputs(self.root, **arguments)
        message = str(caught.exception)
        self.assertNotIn(str(self.root), message, "a refusal names files relative to the root")
        self.assertNotIn("\n", message)
        return message

    def index_of(self, **wanted) -> int:
        """The position in the log of the first scored record with the given planned fields."""
        for index, record in enumerate(self.experiment.records):
            if record.scored and all(getattr(record.planned, key) == value for key, value in wanted.items()):
                return index
        raise AssertionError(f"no scored record with {wanted}")


class ReadingTests(ExperimentOnDisk):
    def test_it_reads_the_counted_log_and_leaves_the_pilot_log_alone(self):
        read = load_inputs(self.root)
        self.assertEqual(read.logs, ("data/trials/main.jsonl",))
        self.assertEqual(read.pilot_logs, ("data/trials/pilot_replay.jsonl",))
        self.assertEqual([trial.record for trial in read.trials], list(self.experiment.records))
        self.assertEqual([trial.number for trial in read.trials], list(range(1, len(self.experiment.records) + 1)))
        self.assertEqual(len(read.scored), sum(record.scored for record in self.experiment.records))
        self.assertFalse(read.draft)

    def test_every_scored_trial_has_its_recomputed_distance(self):
        read = load_inputs(self.root)
        self.assertEqual(set(read.nearest), {record.planned.trial_id for record in self.experiment.scored()})
        for record in self.experiment.scored():
            episode, distance = read.nearest[record.planned.trial_id]
            self.assertEqual(episode, record.nearest_demo_episode)
            self.assertAlmostEqual(distance, record.nearest_demo_distance_mm, places=9)

    def test_everything_else_is_found_where_the_tools_put_it(self):
        read = load_inputs(self.root)
        self.assertEqual(read.protocol_file, "config/protocol.toml")
        self.assertEqual(len(read.protocol_sha256), 64)
        self.assertEqual(
            read.audit_files, {"clustered": "data/audit/clustered.json", "spread": "data/audit/spread.json"}
        )
        episodes = [len(fixtures.audited_starts(name)) for name in schema.DATASETS]
        self.assertEqual([len(read.audits[name]) for name in schema.DATASETS], episodes)
        self.assertEqual(
            [run.label for run in read.scale_runs], ["ruler_x", "ruler_x_corner", "ruler_y", "ruler_y_corner"]
        )
        self.assertEqual({len(run.readings) for run in read.scale_runs}, {6})
        self.assertEqual([len(log.readings) for log in read.repeat_sets], [10])
        self.assertEqual(len(read.by_eye), 20)
        self.assertEqual({reset.label for reset in read.by_eye}, {"op1"})
        self.assertEqual(len(read.placements["clustered"]), episodes[0] + 3, "three lines of a smoke test come first")
        self.assertEqual(len(read.placements["spread"]), episodes[1])
        self.assertEqual(read.deviations, ())
        self.assertEqual(read.decisions_file, "docs/decisions.md")
        self.assertEqual(read.simulation.protocol_digest, read.protocol_sha256[:12])

    def test_no_name_it_keeps_is_a_place_on_this_machine(self):
        read = load_inputs(self.root)
        names = [*read.logs, *read.pilot_logs, read.protocol_file, read.decisions_file, *read.by_eye_files]
        names += [*read.audit_files.values(), *read.placement_files.values(), *read.simulation.files]
        names += [log.file for log in (*read.scale_runs, *read.repeat_sets)]
        for name in names:
            self.assertFalse(Path(name).is_absolute(), name)
            self.assertNotIn(str(self.root), name)

    def test_what_is_not_there_is_left_empty_and_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            bare = Path(folder)
            (bare / "config").mkdir()
            shutil.copy(self.root / "config" / "protocol.toml", bare / "config" / "protocol.toml")
            read = load_inputs(bare)
        self.assertEqual((read.logs, read.trials, read.scale_runs, read.repeat_sets, read.by_eye), ((), (), (), (), ()))
        self.assertEqual(read.audits, {"clustered": None, "spread": None})
        self.assertEqual(read.placements, {"clustered": None, "spread": None})
        self.assertIsNone(read.deviations)
        self.assertIsNone(read.simulation)
        self.assertEqual((read.second_scoring, read.logs_digest), (None, None))
        self.assertEqual(read.scenes, {})

    def test_the_note_is_kept_for_the_outputs(self):
        self.assertEqual(load_inputs(self.root, draft=True, note=fixtures.SYNTHETIC_NOTE).note, fixtures.SYNTHETIC_NOTE)
        self.assertIsNone(load_inputs(self.root).note)


class PilotAndFrozenTests(ExperimentOnDisk):
    def test_a_pilot_log_is_refused_when_it_is_named(self):
        message = self.refusal(trials=[self.root / "data" / "trials" / "pilot_replay.jsonl"])
        self.assertIn("data/trials/pilot_replay.jsonl is a pilot log", message)

    def test_a_pilot_log_is_refused_in_a_draft_too(self):
        message = self.refusal(draft=True, trials=[self.root / "data" / "trials" / "pilot_replay.jsonl"])
        self.assertIn("pilot", message)

    def test_any_log_whose_name_starts_with_pilot_is_one(self):
        renamed = self.changed_log(lambda index, entry: entry, name="pilot2.jsonl")
        self.assertIn("pilot2.jsonl is a pilot log", self.refusal(trials=[renamed]))
        self.assertTrue(inputs.is_pilot_log("data/trials/pilot.jsonl"))
        self.assertFalse(inputs.is_pilot_log("data/trials/main_pilot.jsonl"))

    def test_a_record_that_was_not_run_under_a_frozen_protocol_is_refused(self):
        def unfreeze(index, entry):
            if index == 4:
                entry["versions"]["protocol_status"] = "draft"

        path = self.changed_log(unfreeze)
        message = self.refusal(trials=[path])
        self.assertIn(self.experiment.records[4].planned.trial_id, message)
        self.assertIn("not run under a frozen protocol", message)
        self.assertIn("'draft'", message)
        self.assertEqual(len(load_inputs(self.root, draft=True, trials=[path]).trials), len(self.experiment.records))

    def test_a_record_that_does_not_say_its_status_is_refused(self):
        def forget(index, entry):
            if index == 0:
                del entry["versions"]["protocol_status"]

        self.assertIn("not run under a frozen protocol", self.refusal(trials=[self.changed_log(forget)]))

    def test_a_counted_report_needs_a_frozen_protocol_file(self):
        with tempfile.TemporaryDirectory() as folder:
            draft = fixtures.build_experiment(folder, status="draft", version="0.9", coarse=True, blocks=("T1b",))
            with self.assertRaises(ReportError) as caught:
                load_inputs(draft.root)
            self.assertIn("config/protocol.toml has status 'draft'", str(caught.exception))
            self.assertEqual(len(load_inputs(draft.root, draft=True).trials), len(draft.records))

    def test_two_protocol_versions_in_a_log_are_refused(self):
        def other_version(index, entry):
            if index >= 100:
                entry["versions"]["protocol_version"] = "1.1"

        message = self.refusal(trials=[self.changed_log(other_version)])
        self.assertIn("more than one protocol version", message)
        self.assertIn("'1.0'", message)
        self.assertIn("'1.1'", message)
        self.assertIn(self.experiment.records[100].planned.trial_id, message)

    def test_two_versions_are_refused_in_a_draft_as_well(self):
        def other_version(index, entry):
            if index == 0:
                entry["versions"]["protocol_version"] = "0.2"

        self.assertIn(
            "more than one protocol version", self.refusal(draft=True, trials=[self.changed_log(other_version)])
        )

    def test_trials_of_another_version_than_the_protocol_file_are_refused(self):
        def other_version(index, entry):
            entry["versions"]["protocol_version"] = "0.9"

        message = self.refusal(trials=[self.changed_log(other_version)])
        self.assertIn("'0.9'", message)
        self.assertIn("config/protocol.toml is version '1.0'", message)


class LogShapeTests(ExperimentOnDisk):
    def test_a_trial_name_that_appears_twice_is_refused(self):
        twice = self.changed_log(lambda index, entry: entry)
        message = self.refusal(trials=[self.experiment.log, twice])
        self.assertIn("appears twice", message)
        self.assertIn(self.experiment.records[0].planned.trial_id, message)

    def test_a_line_that_is_not_a_record_is_refused_with_its_place(self):
        path = self.changed_log(lambda index, entry: entry)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[6] = lines[6][:40]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        message = self.refusal(trials=[path])
        self.assertIn(f"changed/{path.name}:7", message)

    def test_a_log_that_does_not_exist_is_refused(self):
        self.assertIn("does not exist", self.refusal(trials=[self.root / "data" / "trials" / "nothing.jsonl"]))

    def test_a_trial_of_a_condition_its_block_does_not_run_is_refused(self):
        at = self.index_of(block="T3")

        def other_block(index, entry):
            if index == at:
                entry["planned"]["block"] = "T1a"

        message = self.refusal(trials=[self.changed_log(other_block)])
        self.assertIn("act_spread in block T1a", message)

    def test_a_time_that_is_not_one_is_refused(self):
        def no_time(index, entry):
            if index == 2:
                entry["ts_end"] = "after lunch"

        self.assertIn("'after lunch'", self.refusal(trials=[self.changed_log(no_time)]))
        self.assertIsNone(inputs.parse_time("2026-10-07T09:00:00"))  # no offset: which day it was is not known
        self.assertIsNotNone(inputs.parse_time("2026-10-07T09:00:00-07:00"))

    def test_two_scored_trials_of_one_policy_at_one_placement_are_refused(self):
        first = self.index_of(block="T2", condition="act_clustered")
        pair = self.experiment.records[first].planned.pair_id
        second = next(
            index
            for index, record in enumerate(self.experiment.records)
            if index > first
            and record.scored
            and record.planned.block == "T2"
            and record.planned.condition == "act_clustered"
        )

        def same_pair(index, entry):
            if index == second:
                entry["planned"]["pair_id"] = pair

        message = self.refusal(trials=[self.changed_log(same_pair)])
        self.assertIn(f"placement {pair}", message)
        self.assertIn("tested once with each policy", message)

    def test_a_scored_trial_of_the_paired_block_without_a_pair_is_refused(self):
        at = self.index_of(block="T2")

        def no_pair(index, entry):
            if index == at:
                entry["planned"]["pair_id"] = None

        self.assertIn("without a pair", self.refusal(trials=[self.changed_log(no_pair)]))


class DistanceTests(ExperimentOnDisk):
    def test_a_distance_that_is_not_what_the_audit_gives_is_refused(self):
        at = self.index_of(block="T1a")
        name = self.experiment.records[at].planned.trial_id

        def farther(index, entry):
            if index == at:
                entry["nearest_demo_distance_mm"] += 0.5

        message = self.refusal(trials=[self.changed_log(farther)])
        self.assertIn(name, message)
        self.assertIn("the audit of the clustered set gives", message)

    def test_a_tiny_difference_is_refused_and_rounding_is_not(self):
        at = self.index_of(block="T1a")

        def nudged(by):
            def change(index, entry):
                if index == at:
                    entry["nearest_demo_distance_mm"] += by

            return change

        self.refusal(trials=[self.changed_log(nudged(1e-4))])
        self.assertEqual(
            len(load_inputs(self.root, trials=[self.changed_log(nudged(1e-9))]).trials), len(self.experiment.records)
        )

    def test_another_demonstration_than_the_nearest_is_refused(self):
        at = self.index_of(block="T1a")
        other = (self.experiment.records[at].nearest_demo_episode + 20) % len(fixtures.audited_starts("clustered"))

        def other_episode(index, entry):
            if index == at:
                entry["nearest_demo_episode"] = other

        self.assertIn(f"episode {other}", self.refusal(trials=[self.changed_log(other_episode)]))

    def test_a_replay_s_distance_is_to_the_episode_it_replays_not_to_the_nearest(self):
        at = self.index_of(block="T1b", staircase_id="T1b-down")
        record = self.experiment.records[at]
        self.assertEqual(record.nearest_demo_episode, record.planned.replay_episode)
        read = load_inputs(self.root)
        self.assertAlmostEqual(read.nearest[record.planned.trial_id][1], record.nearest_demo_distance_mm, places=9)

        def to_the_nearest(index, entry):
            if index == at:
                entry["nearest_demo_distance_mm"] = 0.25

        self.assertIn(record.planned.trial_id, self.refusal(trials=[self.changed_log(to_the_nearest)]))

    def test_an_audit_that_changed_after_the_trials_is_noticed(self):
        with tempfile.TemporaryDirectory() as folder:
            audits = Path(folder)
            for dataset in schema.DATASETS:
                shutil.copy(self.root / "data" / "audit" / f"{dataset}.json", audits / f"{dataset}.json")
            payload = json.loads((audits / "clustered.json").read_text(encoding="utf-8"))
            for start in payload["demo_starts"]:
                start["measured"]["x_mm"] += 1.0
            (audits / "clustered.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ReportError) as caught:
                load_inputs(self.root, audit_dir=audits)
        self.assertIn("the audit changed after the trial was run, or the record was altered", str(caught.exception))

    def test_a_replayed_episode_that_was_not_trained_on_is_refused(self):
        replayed = self.experiment.scored(block="T1b")[0].planned.replay_episode
        with tempfile.TemporaryDirectory() as folder:
            audits = Path(folder)
            for dataset in schema.DATASETS:
                shutil.copy(self.root / "data" / "audit" / f"{dataset}.json", audits / f"{dataset}.json")
            payload = json.loads((audits / "clustered.json").read_text(encoding="utf-8"))
            for start in payload["demo_starts"]:
                if start["episode_index"] == replayed:
                    start["trained"] = False
            (audits / "clustered.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ReportError) as caught:
                load_inputs(self.root, audit_dir=audits)
        self.assertIn(f"episode {replayed}", str(caught.exception))

    def test_trials_without_the_audit_of_their_set_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(self.root / "data" / "audit" / "clustered.json", Path(folder) / "clustered.json")
            with self.assertRaises(ReportError) as caught:
                load_inputs(self.root, audit_dir=folder)
        self.assertIn("there is no audit of that set", str(caught.exception))

    def test_an_audit_of_the_wrong_set_or_with_an_episode_twice_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            audits = Path(folder)
            shutil.copy(self.root / "data" / "audit" / "spread.json", audits / "clustered.json")
            with self.assertRaises(ReportError) as caught:
                inputs.read_audit(audits / "clustered.json", "clustered", audits)
            self.assertIn("episodes of the spread set", str(caught.exception))
            payload = json.loads((self.root / "data" / "audit" / "spread.json").read_text(encoding="utf-8"))
            payload["demo_starts"].append(payload["demo_starts"][0])
            (audits / "spread.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ReportError) as caught:
                inputs.read_audit(audits / "spread.json", "spread", audits)
            self.assertIn("lists an episode twice", str(caught.exception))


class ExecutionTests(ExperimentOnDisk):
    """The guard against a repeat of the earlier evaluation: a setting that was off for part of the trials."""

    def scored_indices(self, condition: str) -> list[int]:
        return [
            i
            for i, record in enumerate(self.experiment.records)
            if record.scored and record.planned.condition == condition
        ]

    def set_execution(self, indices, **settings):
        chosen = set(indices)

        def change(index, entry):
            if index in chosen:
                entry["execution"].update(settings)

        return self.changed_log(change)

    def test_a_setting_that_was_off_for_part_of_a_condition_s_trials_is_refused_and_the_trials_named(self):
        clustered = self.scored_indices("act_clustered")
        off = clustered[10:13]
        message = self.refusal(trials=[self.set_execution(off, temporal_ensemble_coeff=0.0)])
        self.assertIn(
            "execution setting temporal_ensemble_coeff is not the same in every scored act_clustered trial", message
        )
        for index in off:
            self.assertIn(self.experiment.records[index].planned.trial_id, message)
        self.assertIn("0.01 in", message)
        self.assertIn("0 in", message)

    def test_it_is_refused_in_a_draft_too(self):
        clustered = self.scored_indices("act_clustered")
        self.assertIn("clamp_deg", self.refusal(draft=True, trials=[self.set_execution(clustered[:1], clamp_deg=10.0)]))

    def test_a_setting_that_is_the_same_everywhere_but_not_the_protocol_s_is_refused(self):
        everything = range(len(self.experiment.records))
        message = self.refusal(trials=[self.set_execution(everything, clamp_deg=10.0)])
        self.assertIn("clamp_deg = 10", message)
        self.assertIn("the protocol's [execution] section says clamp_deg = 15", message)

    def test_every_setting_the_protocol_names_is_held_to_it(self):
        everything = range(len(self.experiment.records))
        stated = self.experiment.protocol["execution"]
        self.assertGreaterEqual(len(stated), 10, "the control rate, the clamp, the gates, the stall rule and the rest")
        for key, value in stated.items():
            other = value + 1 if isinstance(value, int) else value * 0.5
            with self.subTest(key=key):
                message = self.refusal(trials=[self.set_execution(everything, **{key: other})])
                self.assertIn(f"ran with {key} = {other:g}", message)
                self.assertIn(f"says {key} = {value:g}", message)

    def test_every_record_carries_every_setting_of_the_protocol(self):
        stated = self.experiment.protocol["execution"]
        for record in self.experiment.records:
            self.assertEqual({key: record.execution[key] for key in stated}, stated)
            self.assertEqual(set(record.execution) - set(stated), set(fixtures.PER_TRIAL_KEYS))

    def test_a_setting_of_the_protocol_that_no_record_of_a_condition_carries_is_refused(self):
        replay = set(self.scored_indices("replay"))

        def forget(index, entry):
            if index in replay:
                del entry["execution"]["stall_fraction"]

        message = self.refusal(trials=[self.changed_log(forget)])
        self.assertIn("the scored replay trials do not record stall_fraction", message)
        self.assertIn(f"says stall_fraction = {self.experiment.protocol['execution']['stall_fraction']:g}", message)
        self.assertIn(self.experiment.records[min(replay)].planned.trial_id, message)

    def test_a_replay_is_held_to_both_handoff_tolerances_like_every_other_record(self):
        stated = self.experiment.protocol["execution"]
        self.assertNotEqual(stated["handoff_tolerance_deg"], stated["replay_handoff_tolerance_deg"])
        read = load_inputs(self.root)
        replay = [trial for trial in read.scored if trial.condition == "replay"]
        policy, own = stated["handoff_tolerance_deg"], stated["replay_handoff_tolerance_deg"]
        self.assertEqual({trial.record.execution["handoff_tolerance_deg"] for trial in replay}, {policy})
        self.assertEqual({trial.record.execution["replay_handoff_tolerance_deg"] for trial in replay}, {own})
        self.assertEqual({trial.record.execution["handoff_tolerance_used_deg"] for trial in replay}, {own})
        self.assertEqual(read.execution_notes, ())
        wrong = self.set_execution(self.scored_indices("replay"), replay_handoff_tolerance_deg=own + 1.0)
        message = self.refusal(trials=[wrong])
        self.assertIn(f"the scored replay trials ran with replay_handoff_tolerance_deg = {own + 1.0:g}", message)
        self.assertIn(f"says replay_handoff_tolerance_deg = {own:g}", message)

    def test_a_gate_that_used_another_tolerance_than_its_kind_of_trial_has_is_noted_and_not_refused(self):
        # What belongs to the one trial stops nothing. A replay that went through the policies' looser gate is said.
        stated = self.experiment.protocol["execution"]
        policy, own = stated["handoff_tolerance_deg"], stated["replay_handoff_tolerance_deg"]
        replay = self.scored_indices("replay")
        read = load_inputs(self.root, trials=[self.set_execution(replay[:2], handoff_tolerance_used_deg=policy)])
        self.assertEqual(len(read.trials), len(self.experiment.records))
        self.assertEqual(len(read.execution_notes), 1)
        note = read.execution_notes[0]
        for index in replay[:2]:
            self.assertIn(self.experiment.records[index].planned.trial_id, note)
        self.assertIn(f"used a tolerance of {policy:g} degrees", note)
        self.assertIn(f"replay_handoff_tolerance_deg = {own:g} for a scored replay trial", note)
        tighter = self.set_execution(self.scored_indices("act_spread")[:1], handoff_tolerance_used_deg=own)
        notes = load_inputs(self.root, trials=[tighter]).execution_notes
        self.assertIn(f"handoff_tolerance_deg = {policy:g} for a scored act_spread trial", notes[0])

    def test_what_belongs_to_one_trial_is_not_a_setting(self):
        self.assertEqual(
            inputs.PER_TRIAL_KEYS,
            {"steps_sent", "loop_s", "device", "handoff_tolerance_used_deg", "start_pose_deg", "view_crept_mm"},
        )
        self.assertEqual(inputs.PER_TRIAL_KEYS, set(fixtures.PER_TRIAL_KEYS))
        clustered = self.experiment.scored(condition="act_clustered")
        self.assertGreater(len({record.execution["steps_sent"] for record in clustered}), 1)
        self.assertGreater(len({record.execution["loop_s"] for record in clustered}), 1)
        self.assertGreater(len({json.dumps(record.execution["start_pose_deg"]) for record in clustered}), 1)
        self.assertGreater(len({record.execution["view_crept_mm"] for record in clustered}), 1)
        settings = inputs.execution_settings(self.experiment.records[0])
        self.assertEqual(settings, self.experiment.protocol["execution"])
        # Another device, another start pose and another count of steps in some trials of a condition: all read.
        some = self.scored_indices("act_clustered")[2:9]
        path = self.set_execution(
            some, device="cpu", steps_sent=1, loop_s=0.0, start_pose_deg={"shoulder_pan.pos": 40.0}
        )
        read = load_inputs(self.root, trials=[path])
        self.assertEqual(len(read.trials), len(self.experiment.records))
        self.assertEqual(read.execution_notes, ())

    def test_a_record_written_before_the_runner_kept_the_loop_s_duration_is_read(self):
        spread = set(self.scored_indices("act_spread")[:6])

        def forget(index, entry):
            if index in spread:
                del entry["execution"]["loop_s"]

        read = load_inputs(self.root, trials=[self.changed_log(forget)])
        self.assertEqual(len(read.trials), len(self.experiment.records), "not a setting: its absence stops nothing")
        self.assertEqual(read.execution_notes, ())

    def test_a_setting_that_some_trials_do_not_record_is_refused(self):
        clustered = set(self.scored_indices("act_clustered")[:5])

        def forget(index, entry):
            if index in clustered:
                del entry["execution"]["stall_steps"]

        message = self.refusal(trials=[self.changed_log(forget)])
        self.assertIn("execution setting stall_steps", message)
        self.assertIn("not recorded in", message)
        for index in clustered:
            self.assertIn(self.experiment.records[index].planned.trial_id, message)

    def test_a_setting_the_protocol_does_not_name_must_still_be_the_same(self):
        clustered = self.scored_indices("act_clustered")
        self.assertIn(
            "execution setting new_setting", self.refusal(trials=[self.set_execution(clustered[-2:], new_setting="on")])
        )
        everywhere = self.set_execution(range(len(self.experiment.records)), new_setting="on")
        self.assertEqual(len(load_inputs(self.root, trials=[everywhere]).trials), len(self.experiment.records))

    def test_two_conditions_may_differ_in_a_setting_the_protocol_does_not_name(self):
        spread = [i for i, record in enumerate(self.experiment.records) if record.planned.condition == "act_spread"]
        path = self.set_execution(spread, new_setting="on")
        self.assertEqual(len(load_inputs(self.root, trials=[path]).trials), len(self.experiment.records))

    def test_scored_policy_trials_of_two_training_steps_are_refused(self):
        spread = self.scored_indices("act_spread")[:4]

        def earlier(index, entry):
            if index in spread:
                entry["checkpoint"] = fixtures.checkpoint_name("act_spread", step="060000")

        message = self.refusal(trials=[self.changed_log(earlier)])
        self.assertIn("the scored policy trials were not all run at one training step", message)
        self.assertIn("step 060000 in", message)
        self.assertIn("step 080000 in", message)
        for index in spread:
            self.assertIn(self.experiment.records[index].planned.trial_id, message)
        self.assertIn("one training step", self.refusal(draft=True, trials=[self.changed_log(earlier)]))

    def test_the_step_is_held_across_the_two_policies_and_the_run_is_not(self):
        spread = set(self.scored_indices("act_spread"))

        def whole_policy_earlier(index, entry):
            if index in spread:
                entry["checkpoint"] = fixtures.checkpoint_name("act_spread", step="060000")

        self.assertIn("one training step", self.refusal(trials=[self.changed_log(whole_policy_earlier)]))

        pairs = sorted({r.planned.pair_id for r in self.experiment.scored(block="T2")})[::2]

        def second_runs(index, entry):
            if entry["planned"]["pair_id"] in pairs and entry["checkpoint"] is not None:
                entry["checkpoint"] = fixtures.checkpoint_name(entry["planned"]["condition"], run="v1-s2")

        read = load_inputs(self.root, trials=[self.changed_log(second_runs)])
        self.assertEqual(len(read.trials), len(self.experiment.records))
        self.assertEqual(read.execution_notes, (), "two runs in the workspace test are allowed and need no remark")

    def test_a_checkpoint_names_a_run_and_a_step_or_a_step_alone(self):
        self.assertEqual(
            inputs.split_checkpoint("act-tolrad-clustered-v1@080000"), ("act-tolrad-clustered-v1", "080000")
        )
        self.assertEqual(inputs.split_checkpoint("080000"), (None, "080000"))
        self.assertEqual(inputs.split_checkpoint("act-final"), (None, "act-final"))
        self.assertEqual(inputs.split_checkpoint("a@b@040000"), ("a@b", "040000"))
        mixed = self.scored_indices("act_clustered")[:3]

        def bare(index, entry):
            if index in mixed:
                entry["checkpoint"] = "080000"

        read = load_inputs(self.root, trials=[self.changed_log(bare)])
        self.assertEqual(len(read.trials), len(self.experiment.records), "the same step, named with and without a run")

    def test_a_void_trial_of_another_step_is_noted(self):
        void = next(
            i for i, record in enumerate(self.experiment.records) if not record.scored and record.checkpoint is not None
        )

        def earlier(index, entry):
            if index == void:
                entry["checkpoint"] = entry["checkpoint"].replace("@080000", "@040000")

        read = load_inputs(self.root, trials=[self.changed_log(earlier)])
        name = self.experiment.records[void].planned.trial_id
        self.assertEqual(
            read.execution_notes, (f"the void trial {name} ran at training step 040000, unlike the scored trials",)
        )

    def test_a_void_trial_with_another_setting_is_not_refused_and_is_noted(self):
        void = next(i for i, record in enumerate(self.experiment.records) if not record.scored)
        read = load_inputs(self.root, trials=[self.set_execution([void], clamp_deg=10.0)])
        name = self.experiment.records[void].planned.trial_id
        self.assertEqual(len(read.execution_notes), 1)
        self.assertIn(f"the void trial {name} ran with clamp_deg = 10", read.execution_notes[0])
        self.assertEqual(load_inputs(self.root).execution_notes, ())

    def test_a_long_list_of_trials_is_cut_short_in_the_message(self):
        clustered = self.scored_indices("act_clustered")
        message = self.refusal(trials=[self.set_execution(clustered[: len(clustered) // 2], clamp_deg=10.0)])
        self.assertRegex(message, r"and \d+ more")
        self.assertLess(len(message), 900)


class InstrumentFileTests(ExperimentOnDisk):
    def instrument_copy(self) -> Path:
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copytree(self.root / "data" / "instrument", folder, dirs_exist_ok=True)
        return folder

    def test_a_scale_run_with_one_reading_is_refused(self):
        folder = self.instrument_copy()
        fixtures.write_position_csv(folder / "ruler_x_20261006_090000.csv", "ruler_x", [(60.0, 70.0)])
        message = self.refusal(instrument_dir=folder)
        self.assertIn("ruler_x_20261006_090000.csv holds 1 reading(s)", message)
        self.assertNotIn(str(folder), message)

    def test_a_file_without_its_columns_or_with_a_word_for_a_number_is_refused(self):
        folder = self.instrument_copy()
        (folder / "ruler_z_1.csv").write_text("index,label,x_mm\n0,ruler_z,1.0\n1,ruler_z,2.0\n", encoding="utf-8")
        self.assertIn("ruler_z_1.csv has no column y_mm", self.refusal(instrument_dir=folder))
        (folder / "ruler_z_1.csv").write_text("index,label,x_mm,y_mm\n0,z,1.0,2.0\n1,z,far,2.0\n", encoding="utf-8")
        self.assertIn("ruler_z_1.csv, row 2: x_mm is 'far'", self.refusal(instrument_dir=folder))

    def test_a_by_eye_error_that_is_not_the_distance_between_its_positions_is_refused(self):
        folder = self.instrument_copy()
        path = next(folder.glob("byeye_*.csv"))
        lines = path.read_text(encoding="utf-8").splitlines()
        cells = lines[3].split(",")
        cells[8] = f"{float(cells[8]) + 1.0:.2f}"
        lines[3] = ",".join(cells)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertIn("row 3: the error it states", self.refusal(instrument_dir=folder))

    def test_a_by_eye_reset_s_error_is_from_the_reference_placement(self):
        read = load_inputs(self.root)
        errors = [reset.error_mm for reset in read.by_eye]
        self.assertEqual(len(errors), len(self.experiment.by_eye_errors_mm))
        for found, written in zip(errors, self.experiment.by_eye_errors_mm, strict=True):
            self.assertAlmostEqual(found, written, places=12)

    def test_a_second_person_s_file_is_read_as_well(self):
        folder = self.instrument_copy()
        fixtures.write_by_eye(folder, label="op2", count=5)
        read = load_inputs(self.root, instrument_dir=folder)
        self.assertEqual(len(read.by_eye), 25)
        self.assertEqual({reset.label for reset in read.by_eye}, {"op1", "op2"})


class SceneAuditFileTests(ExperimentOnDisk):
    """The scene audit of the counted log: what is read of it, and what is refused."""

    def changed_audit(self, change) -> Path:
        """A copy of the audit folder in which `change(audit)` was applied to the log's scene audit."""
        folder = Path(self.enterContext(tempfile.TemporaryDirectory())) / "audit"
        shutil.copytree(self.root / "data" / "audit", folder)
        audit = json.loads(json.dumps(self.experiment.scene))
        change(audit)
        fixtures.write_scene_audit(folder, audit)
        return folder

    def line_of(self, audit: dict, **wanted) -> dict:
        """The first line of an audit whose trial has a reading and the given flags."""
        for line in audit["trials"]:
            read = line["basket_start"] is not None and line["wrist_start"] is not None
            if read and all(line[key] == value for key, value in wanted.items()):
                return line
        raise AssertionError(f"the synthetic audit has no line with {wanted}")

    def test_the_audit_of_a_log_is_read_beside_the_start_state_audits(self):
        read = load_inputs(self.root)
        self.assertEqual(list(read.scenes), ["data/trials/main.jsonl"])
        audit = read.scenes["data/trials/main.jsonl"]
        self.assertEqual((audit.file, audit.log), ("data/audit/scene_main.json", "main.jsonl"))
        self.assertEqual((audit.basket_named_mm, audit.wrist_named_px), (2.0, 2.0))
        written = self.experiment.scene["trials"]
        self.assertEqual([reading.trial_id for reading in audit.readings], [line["trial_id"] for line in written])
        self.assertEqual(len(audit.readings), len(self.experiment.records), "a line for every record, void ones too")
        for reading, line in zip(audit.readings, written, strict=True):
            basket, wrist = line["basket_start"], line["wrist_start"]
            self.assertEqual((reading.condition, reading.outcome), (line["condition"], line["outcome"]))
            self.assertEqual(reading.basket_start_mm, basket and (basket["across_mm"], basket["down_mm"]))
            self.assertEqual(reading.wrist_start_px, wrist and (wrist["across_px"], wrist["down_px"]))
            self.assertEqual(
                (reading.basket_moved_mm, reading.basket_was_moved, reading.named_basket, reading.named_wrist),
                (line["basket_moved_mm"], line["basket_was_moved"], line["named_basket"], line["named_wrist"]),
            )

    def test_an_offset_is_the_larger_of_across_and_down_whatever_its_sign(self):
        reading = inputs.SceneReading("T1a-001", "replay", "success", (-2.3, 0.4), (0.5, -1.9), 0.0, False, True, False)
        self.assertEqual((reading.basket_off_mm, reading.wrist_off_px), (2.3, 1.9))
        by_edge = inputs.SceneReading("T1a-002", "replay", "success", None, (-0.7, None), None, None, None, False)
        self.assertEqual((by_edge.basket_off_mm, by_edge.wrist_off_px), (None, 0.7))

    def test_a_log_without_an_audit_has_none_and_that_is_not_an_error(self):
        folder = self.changed_audit(lambda audit: None)
        (folder / "scene_main.json").unlink()
        self.assertEqual(load_inputs(self.root, audit_dir=folder).scenes, {"data/trials/main.jsonl": None})
        other = self.changed_log(lambda index, entry: None, "another.jsonl")
        self.assertEqual(load_inputs(self.root, trials=[other]).scenes, {"changed/another.jsonl": None})

    def test_a_file_that_is_not_a_scene_audit_is_refused(self):
        for change, said in (
            (lambda a: a.pop("basket_named_mm"), "it has no basket_named_mm"),
            (lambda a: a["trials"][0].pop("named_wrist"), "it has no named_wrist"),
            (lambda a: a["trials"][0].update(trial_id=None), "None is not a name"),
            (lambda a: a["trials"][2]["basket_start"].update(across_mm="left"), "'left' is not a number"),
            (lambda a: a["trials"][2].update(basket_moved_mm=-1.0), "-1.0 is a distance below zero"),
            (lambda a: a["trials"][2].update(named_basket="no"), "'no' is neither true nor false"),
            (lambda a: a.update(trials=None), "TypeError"),
        ):
            with self.subTest(said=said):
                message = self.refusal(audit_dir=self.changed_audit(change))
                self.assertIn("scene_main.json is not a scene audit as scripts/rig/scene_audit.py writes one", message)
                self.assertIn(said, message)

    def test_an_audit_of_a_log_of_another_name_is_refused(self):
        folder = self.changed_audit(lambda audit: None)
        audit = json.loads(json.dumps(self.experiment.scene))
        audit["log"] = "pilot_policy.jsonl"
        (folder / "scene_main.json").write_text(json.dumps(audit), encoding="utf-8")
        message = self.refusal(audit_dir=folder)
        said = "scene_main.json is the scene audit of pilot_policy.jsonl, and it was read for main.jsonl"
        self.assertIn(said, message)

    def test_a_trial_listed_twice_is_refused(self):
        folder = self.changed_audit(lambda audit: audit["trials"].append(dict(audit["trials"][0])))
        self.assertIn(f"lists {self.experiment.records[0].planned.trial_id} twice", self.refusal(audit_dir=folder))

    def test_a_flag_that_is_not_what_its_own_reading_and_limit_give_is_refused(self):
        named = self.line_of(self.experiment.scene, named_basket=True)["trial_id"]
        unnamed = self.line_of(self.experiment.scene, named_wrist=False)["trial_id"]

        def flag(trial_id: str, **flags):
            def change(audit):
                next(line for line in audit["trials"] if line["trial_id"] == trial_id).update(flags)

            return change

        message = self.refusal(audit_dir=self.changed_audit(flag(named, named_basket=False)))
        self.assertIn(f"gives {named} named_basket = false, and its own reading at the start is 2.", message)
        self.assertIn(" mm against a limit of 2 mm", message)
        message = self.refusal(audit_dir=self.changed_audit(flag(unnamed, named_wrist=True)))
        self.assertIn(f"gives {unnamed} named_wrist = true, and its own reading", message)
        message = self.refusal(audit_dir=self.changed_audit(flag(unnamed, named_wrist=None)))
        self.assertIn(f"gives {unnamed} named_wrist = null", message)
        # A lower limit names more trials than the flags say: the file no longer holds together.
        folder = self.changed_audit(lambda audit: audit.update(basket_named_mm=1.0))
        self.assertIn("named_basket = false", self.refusal(audit_dir=folder))

    def test_a_reading_on_the_limit_is_not_beyond_it(self):
        line = self.line_of(self.experiment.scene, named_basket=False, named_wrist=False)["trial_id"]

        def on_the_line(audit):
            found = next(entry for entry in audit["trials"] if entry["trial_id"] == line)
            found["basket_start"].update(across_mm=-2.0, down_mm=0.3)
            found.update(basket_end=None, basket_moved_mm=None, basket_was_moved=None)

        read = load_inputs(self.root, audit_dir=self.changed_audit(on_the_line))
        reading = next(r for r in read.scenes["data/trials/main.jsonl"].readings if r.trial_id == line)
        self.assertEqual((reading.basket_off_mm, reading.named_basket), (2.0, False))

    def test_an_audit_that_was_made_from_another_log_is_refused(self):
        def stranger(audit):
            audit["trials"].append({**audit["trials"][0], "trial_id": "T9-001"})

        message = self.refusal(audit_dir=self.changed_audit(stranger))
        self.assertIn("holds T9-001, which data/trials/main.jsonl does not: the audit was made from", message)
        scored = self.index_of(block="T1b")
        record = self.experiment.records[scored]
        other = "failure" if record.outcome == "success" else "success"
        message = self.refusal(audit_dir=self.changed_audit(lambda a: a["trials"][scored].update(outcome=other)))
        self.assertIn(
            f"gives {record.planned.trial_id} as {other} of replay, and data/trials/main.jsonl as {record.outcome} "
            "of replay",
            message,
        )
        folder = self.changed_audit(lambda a: a["trials"][scored].update(condition="act_spread"))
        message = self.refusal(audit_dir=folder)
        self.assertIn(f"gives {record.planned.trial_id} as {record.outcome} of act_spread", message)

    def test_a_log_that_gained_a_trial_since_its_audit_is_not_refused(self):
        folder = self.changed_audit(lambda audit: audit["trials"].pop())
        read = load_inputs(self.root, audit_dir=folder)
        self.assertEqual(len(read.scenes["data/trials/main.jsonl"].readings), len(self.experiment.records) - 1)

    def test_audits_of_two_logs_must_name_trials_beyond_the_same_limits(self):
        read = load_inputs(self.root)
        audit = read.scenes["data/trials/main.jsonl"]
        looser = inputs.SceneAudit("data/audit/scene_later.json", "later.jsonl", 3.0, 2.0, ())
        inputs.check_scenes(read.trials, {"data/trials/main.jsonl": audit, "data/trials/later.jsonl": None})
        with self.assertRaises(ReportError) as caught:
            inputs.check_scenes(read.trials, {"data/trials/main.jsonl": audit, "data/trials/later.jsonl": looser})
        self.assertIn("do not all name a trial beyond the same limits", str(caught.exception))

    def test_the_repository_s_own_scene_audit_is_read_and_holds_together(self):
        path = fixtures.REPO / "data" / "audit" / "scene_main.json"
        if not path.is_file():
            self.skipTest("the counted log has no scene audit yet")
        audit = inputs.read_scene_audit(path, "main.jsonl", fixtures.REPO)
        self.assertEqual((audit.basket_named_mm, audit.wrist_named_px), (2.0, 2.0), "the limits of protocol section 8")
        self.assertEqual(len({reading.trial_id for reading in audit.readings}), len(audit.readings))


class SecondScoringFileTests(ExperimentOnDisk):
    """The agreement of the second scoring from video: what is read of it, and what is refused."""

    def changed_agreement(self, change) -> Path:
        """A folder that holds the experiment's agreement with `change(agreement)` applied to it."""
        folder = Path(self.enterContext(tempfile.TemporaryDirectory())) / "scored_again"
        agreement = json.loads(json.dumps(self.experiment.second_scoring))
        change(agreement)
        fixtures.write_second_scoring(folder, agreement)
        return folder

    def test_the_agreement_is_read_where_the_comparison_writes_it(self):
        read = load_inputs(self.root).second_scoring
        written = self.experiment.second_scoring
        self.assertEqual(read.file, "data/second_scoring/main/agreement.json")
        self.assertEqual((read.rehearsal, read.logs), (False, ("data/trials/main.jsonl",)))
        self.assertEqual((read.seed, read.one_in), (fixtures.SECOND_SEED, fixtures.SECOND_ONE_IN))
        self.assertEqual(read.n_scored, len(self.experiment.scored()))
        self.assertEqual(read.n_drawn, written["all"]["n_items"])
        self.assertEqual(read.tallies, {"all": written["all"], **written["by_kind"]})
        self.assertEqual((read.first_scorers, read.second_scorers, read.n_same_role), (("op1",), ("sc2",), 0))
        self.assertEqual(read.interval_level, 0.95)
        self.assertEqual(
            [trial for trial, _, _ in read.disagreements], [entry["trial_id"] for entry in written["disagreements"]]
        )
        self.assertEqual(len(read.disagreements), 1)
        self.assertEqual({second for _, _, second in read.cannot_tell}, {"cannot_tell"})

    def test_the_synthetic_draw_takes_one_scored_trial_in_five_of_each_condition_of_each_block(self):
        groups = self.experiment.second_scoring["provenance"]["groups"]
        self.assertEqual(len(groups), 5, "T1a, T1b, the two policies of T2, and T3")
        for group in groups:
            scored = len(self.experiment.scored(block=group["block"], condition=group["condition"]))
            self.assertEqual((group["n_scored"], group["n_drawn"]), (scored, second_scoring.quota(scored)))

    def test_the_digest_of_the_logs_is_the_one_a_draw_would_record(self):
        read = load_inputs(self.root)
        population = second_scoring.read_population([self.experiment.log], self.root)
        self.assertEqual(read.logs_digest, population.digest)
        self.assertEqual(read.second_scoring.logs_digest, population.digest, "the agreement is of this very log")
        two = [self.experiment.log, self.changed_log(lambda index, entry: None, "z_other.jsonl")]
        self.assertNotEqual(inputs.logs_digest(two, self.root), read.logs_digest)
        self.assertEqual(inputs.logs_digest(two, self.root), inputs.logs_digest(two[::-1], self.root))
        self.assertIsNone(inputs.logs_digest([], self.root))

    def test_the_names_of_the_file_and_of_its_kinds_are_the_comparison_s_own(self):
        self.assertEqual(inputs.AGREEMENT_FILE, second_scoring.AGREEMENT_FILE)
        self.assertEqual(sorted(load_inputs(self.root).second_scoring.tallies), ["all", "policy", "replay"])

    def test_another_folder_can_be_named_and_one_without_the_file_gives_none(self):
        folder = self.changed_agreement(lambda agreement: agreement.update(interval_level=0.9))
        self.assertEqual(load_inputs(self.root, second_scoring_dir=folder).second_scoring.interval_level, 0.9)
        self.assertEqual(load_inputs(self.root, second_scoring_dir=folder).second_scoring.file, "agreement.json")
        self.assertIsNone(load_inputs(self.root, second_scoring_dir=folder.parent).second_scoring)

    def test_a_file_that_is_not_an_agreement_is_refused(self):
        folder = self.changed_agreement(lambda agreement: agreement.pop("scorers"))
        message = self.refusal(second_scoring_dir=folder)
        self.assertIn("agreement.json is not an agreement as scripts/second_scoring.py compare writes one", message)
        self.assertIn("it has no scorers", message)
        self.assertNotIn(str(folder), message)
        (folder / "agreement.json").write_text("not json", encoding="utf-8")
        self.assertIn("JSONDecodeError", self.refusal(second_scoring_dir=folder))
        for change, said in (
            (lambda a: a["all"].update(n_agree="27"), "'27' is not a count"),
            (lambda a: a["all"].update(n_agree=-1), "-1 is not a count"),
            (lambda a: a["all"].update(n_agree=True), "True is not a count"),
            (lambda a: a.update(rehearsal="no"), "'no' is neither true nor false"),
            (lambda a: a["by_kind"].pop("replay"), "it has no replay"),
            (lambda a: a["all"]["table"].pop("failure_success"), "it has no failure_success"),
            (lambda a: a.update(disagreements=[{"trial_id": "T1a-001"}]), "it has no first"),
        ):
            with self.subTest(said=said):
                self.assertIn(said, self.refusal(second_scoring_dir=self.changed_agreement(change)))

    def test_counts_that_do_not_add_up_are_refused(self):
        def one_more(*path):
            def change(agreement):
                target = agreement
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] += 1

            return change

        cases = (
            (one_more("all", "n_agree"), "of all items, those that agree and those that differ are not the"),
            (one_more("all", "n_items"), "of all items, the compared and those that could not be told are not the"),
            (one_more("by_kind", "policy", "table", "success_failure"), "the table of the policy trials holds"),
            (one_more("by_kind", "replay", "cannot_tell_first", "failure"), "of the replay trials, the items that"),
            (one_more("provenance", "n_drawn"), "trials were drawn"),
            (lambda a: a["scorers"].update(n_same_role=a["all"]["n_items"] + 1), "both scores are by one person in"),
            (lambda a: a["disagreements"].clear(), "the trials it lists as differing or as not told"),
            (lambda a: a["all"].update(wilson_lo=None), "there is a share that agree exactly when"),
            (lambda a: a.update(interval_level=95), "its interval level is 95"),
        )
        for change, said in cases:
            with self.subTest(said=said):
                message = self.refusal(second_scoring_dir=self.changed_agreement(change))
                self.assertIn("the second scoring's agreement agreement.json does not add up: ", message)
                self.assertIn(said, message)

        def moved(agreement):  # each tally adds up on its own, and the two kinds are no longer all the items
            for tally in (agreement["all"], agreement["by_kind"]["policy"]):
                tally["table"]["success_success"] -= 1
                tally["table"]["success_failure"] += 1
                tally["n_agree"] -= 1
                tally["n_disagree"] += 1
            agreement["by_kind"]["policy"]["n_items"] += 1
            agreement["by_kind"]["policy"]["n_compared"] += 1
            agreement["by_kind"]["policy"]["n_agree"] += 1
            agreement["by_kind"]["policy"]["table"]["success_success"] += 1

        message = self.refusal(second_scoring_dir=self.changed_agreement(moved))
        self.assertIn("of the policy and the replay trials together is not that of all items", message)

    def test_a_diagonal_that_is_not_the_agreeing_items_is_refused(self):
        def swapped(agreement):
            table = agreement["all"]["table"]
            table["success_success"] -= 1
            table["success_failure"] += 1

        message = self.refusal(second_scoring_dir=self.changed_agreement(swapped))
        self.assertIn("the table of all items does not hold its", message)

    def test_a_scorer_who_is_not_named_by_a_role_code_is_refused_and_the_name_is_not_repeated(self):
        for field in ("first", "second"):
            folder = self.changed_agreement(lambda a, field=field: a["scorers"].update({field: ["Someone Named"]}))
            message = self.refusal(second_scoring_dir=folder)
            self.assertIn("names 1 scorer(s) otherwise than by a role code", message)
            self.assertNotIn("Someone", message)
        self.assertTrue(second_scoring.ROLE_CODE.fullmatch("op1") and second_scoring.ROLE_CODE.fullmatch("sc12"))

    def test_the_repository_s_own_agreement_is_read_and_adds_up(self):
        path = fixtures.REPO / "data" / "second_scoring" / "main" / second_scoring.AGREEMENT_FILE
        if not path.is_file():
            self.skipTest("the counted trials have no second scoring yet")
        read = inputs.read_second_scoring(path, fixtures.REPO)
        self.assertFalse(read.rehearsal)
        self.assertEqual(read.tallies["all"]["n_items"], read.n_drawn)
        logs = [fixtures.REPO / name for name in read.logs]
        self.assertEqual(inputs.logs_digest(logs, fixtures.REPO), read.logs_digest, "drawn from the logs as they are")


class OtherFileTests(ExperimentOnDisk):
    def test_a_placement_log_keeps_the_order_written_smoke_test_included(self):
        placements = load_inputs(self.root).placements["clustered"]
        self.assertEqual([p.episode_index for p in placements[:4]], [0, 1, 2, 0])
        self.assertEqual({p.session for p in placements[:3]}, {fixtures.SMOKE_SESSION})
        takes = sum(fixtures.takes_before_saving(start.episode_index) for start in fixtures.audited_starts("clustered"))
        self.assertEqual(sum(p.rerecords for p in placements[3:]), takes)

    def test_a_placement_line_the_recorder_does_not_write_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            recording = Path(folder)
            shutil.copytree(self.root / "data" / "recording", recording, dirs_exist_ok=True)
            with open(recording / "spread_placements.jsonl", "a", encoding="utf-8") as handle:
                handle.write('{"dataset": "spread", "episode_index": 3}\n')
            message = self.refusal(recording_dir=recording)
            self.assertIn("spread_placements.jsonl, line 56", message)
            entry = json.loads((recording / "clustered_placements.jsonl").read_text(encoding="utf-8").splitlines()[0])
            (recording / "spread_placements.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
            self.assertIn(
                "a placement of the clustered set in the spread set's log", self.refusal(recording_dir=recording)
            )

    def test_only_rows_of_the_decision_log_that_are_marked_count_as_deviations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.md"
            path.write_text(fixtures.decisions_text(deviations=2), encoding="utf-8")
            self.assertIn("**DEVIATION**", path.read_text(encoding="utf-8").split("|")[0], "the heading names the mark")
            found = inputs.read_deviations(path)
            self.assertEqual(len(found), 2)
            self.assertEqual(found[0].date, "2026-10-07")
            self.assertIn("block T3 was cut short", found[0].decision)
            self.assertEqual(found[0].why, "No time left")
            self.assertEqual(len(load_inputs(self.root, decisions=path).deviations), 2)

    def test_the_repository_s_own_decision_log_marks_the_three_deviations_of_the_day_of_the_trials(self):
        """Until 2026-10-08 the log marked none. That evening a review of the paper against the protocol found two
        (the repeatability check run after the trials, the tag check at a block's end not run as its own step), and
        the omission of block T3 was counted as a third. The paper's appendix lists them: a fourth row, or one that
        goes, has to be met here on purpose."""
        log = fixtures.REPO / "docs" / "decisions.md"
        if not log.is_file():
            self.skipTest("the decision log is not part of this copy of the repository")
        found = inputs.read_deviations(log)
        self.assertEqual([deviation.date for deviation in found], ["2026-10-08"] * 3)
        self.assertIn("block 3 (T3", found[0].decision)
        self.assertIn("repeatability check", found[1].decision)
        self.assertIn("tag check at the end of block 1", found[2].decision)

    def test_the_simulation_s_report_says_which_protocol_file_it_read(self):
        with tempfile.TemporaryDirectory() as folder:
            tables = Path(folder)
            protocol = self.root / "config" / "protocol.toml"
            fixtures.write_simulation(tables, protocol, 20, stale=True, rule=None)
            read = inputs.read_simulation(tables, tables)
            self.assertEqual(read.protocol_digest, "0123456789ab")
            self.assertEqual(read.files, ("procedure.csv", "paired_power.csv", "procedure.md"))
            self.assertEqual((read.rule, read.rule_digest), (None, None))
            self.assertEqual(read.sweep[1]["design"], "protocol")
            (tables / "procedure.md").unlink()
            self.assertIsNone(inputs.read_simulation(tables, tables).protocol_digest)
            (tables / "paired_power.csv").unlink()
            self.assertIsNone(inputs.read_simulation(tables, tables))

    def test_the_comparison_rule_s_table_is_read_with_the_digest_of_its_own_report(self):
        with tempfile.TemporaryDirectory() as folder:
            tables = Path(folder)
            protocol = self.root / "config" / "protocol.toml"
            fixtures.write_simulation(tables, protocol, 20, stale=True, rule="fresh")
            read = inputs.read_simulation(tables, tables)
            self.assertEqual(read.protocol_digest, "0123456789ab")
            self.assertEqual(read.rule_digest, load_inputs(self.root).protocol_sha256[:12], "each report has its own")
            self.assertEqual(len(read.rule), 18)
            self.assertEqual(read.rule[4]["share_larger"], "1.0000")
            self.assertEqual(read.files[3:], ("comparison_rule.csv", "comparison_rule.md"))
            (tables / "comparison_rule.md").unlink()
            read = inputs.read_simulation(tables, tables)
            self.assertEqual((len(read.rule), read.rule_digest), (18, None))

    def test_the_repository_s_own_comparison_rule_names_a_protocol_file_across_a_line_break_or_not(self):
        read = inputs.read_simulation(fixtures.REPO / "analysis" / "simulation", fixtures.REPO)
        self.assertRegex(read.rule_digest, r"^[0-9a-f]{12}$")
        self.assertEqual(len(read.rule), 18)
        self.assertIn("analysis/simulation/comparison_rule.md", read.files)

    def test_the_repository_s_own_simulation_names_a_protocol_file(self):
        read = inputs.read_simulation(fixtures.REPO / "analysis" / "simulation", fixtures.REPO)
        self.assertRegex(read.protocol_digest, r"^[0-9a-f]{12}$")
        self.assertIn("analysis/simulation/procedure.csv", read.files)

    def test_a_protocol_file_that_cannot_be_read_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ReportError) as caught:
                load_inputs(folder)
            self.assertIn("config/protocol.toml cannot be read", str(caught.exception))
            self.assertNotIn(folder, str(caught.exception))


class RunnerFormatTests(unittest.TestCase):
    """What the reader takes for granted about a record, held against the trial runner's own source.

    The runner cannot be run here, so its source is read. If it comes to write a record differently, these fail
    before a counted log is refused for it.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_what_the_runner_adds_to_a_record_s_settings_is_what_the_reader_takes_as_belonging_to_one_trial(self):
        written = []
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_record"):
                continue
            execution = next((keyword.value for keyword in node.keywords if keyword.arg == "execution"), None)
            if isinstance(execution, ast.Dict):
                written.append(execution)
        self.assertEqual(len(written), 1, "the runner builds a record in one place, with its execution written out")
        (execution,) = written
        unpacked = [value for key, value in zip(execution.keys, execution.values, strict=True) if key is None]
        self.assertEqual([getattr(value, "id", None) for value in unpacked], ["execution_settings"])
        own = {key.value for key in execution.keys if key is not None}
        self.assertEqual(own, set(inputs.PER_TRIAL_KEYS))
        self.assertIn(inputs.HANDOFF_USED_KEY, own)
        self.assertEqual(own, set(fixtures.PER_TRIAL_KEYS), "and the synthetic records are written the same way")

    def test_the_settings_the_runner_writes_are_the_protocol_s_own_section(self):
        # `settle_execution` returns what `session_rules.resolved_execution` makes of the protocol's [execution].
        from tolerance_radius import session_rules

        protocol = self_protocol()
        settings, changed = session_rules.resolved_execution(protocol, {})
        self.assertEqual((settings, changed), (protocol["execution"], []))
        self.assertEqual(session_rules.execution_problems(protocol, settings), [])
        self.assertIn("settings, changed = session_rules.resolved_execution(protocol, given)", self.source)
        self.assertIn("execution_settings = settle_execution(", self.source)

    def test_a_key_that_ended_an_attempt_at_once_is_written_as_the_reader_expects(self):
        sentence = r'f"(ended by a key \{[^}]+\} s after it began )"\s*\n\s*f"(\(under \{[^}]+\} s\))"'
        pieces = re.findall(sentence, self.source)
        self.assertEqual(len(pieces), 1, "the runner's sentence for a slip of the key is no longer the one read here")
        template = re.sub(r"\{[^}]+\}", "{}", "".join(pieces[0]))
        for seconds, limit in (("0.4", "1"), ("0.0", "1"), ("1.9", "2.5")):
            written = template.format(seconds, limit)
            self.assertEqual(
                bookkeeping.reason_shown(written), f"ended by a key less than {limit} s after it began", written
            )
        self.assertEqual(bookkeeping.reason_shown("ended by a keystroke"), "ended by a keystroke")

    def test_a_checkpoint_is_labelled_run_at_step(self):
        functions = [node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)]
        function = next(node for node in functions if node.name == "checkpoint_label")
        scope: dict = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER.name), "exec"), scope)
        label = scope["checkpoint_label"]
        downloaded = label("models/act-tolrad-spread-v1-s2/checkpoints/040000/pretrained_model")
        self.assertEqual(inputs.split_checkpoint(downloaded), ("act-tolrad-spread-v1-s2", "040000"))
        self.assertEqual(inputs.split_checkpoint(label("models/ns/run/080000/pretrained_model")), (None, "080000"))

    def test_the_two_times_of_a_record_span_more_than_the_control_loop(self):
        # Why the rate is made from `loop_s` and not from the record's two times: ts_start is taken before the gated
        # cycle and ts_end after the videos, and both are whole seconds.
        start = self.source.index("ts_start = now()")
        cycle = self.source.index("rollout.attempt_cycle(", start)
        videos = self.source.index("recorder.finish()", cycle)
        end = self.source.index("ts_end = now()", videos)
        self.assertLess(start, cycle)
        self.assertLess(videos, end)
        self.assertIn('isoformat(timespec="seconds")', self.source, "and both are whole seconds")

    def step_callback(self, clock):
        """The runner's own `on_step`, taken out of its source and given a clock: `(on_step, steps, sent_at)`."""
        functions = [node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)]
        (on_step,) = [node for node in functions if node.name == "on_step"]
        scope = {"steps": [0], "sent_at": [0.0, 0.0], "time": types.SimpleNamespace(monotonic=clock)}
        exec(compile(ast.Module(body=[on_step], type_ignores=[]), RUNNER.name, "exec"), scope)
        return scope["on_step"], scope["steps"], scope["sent_at"]

    def loop_seconds(self, steps: list[int], sent_at: list[float]) -> float:
        """What the runner writes as `loop_s`, by its own expression, for a count of steps and two moments."""
        execution = self.execution_written()
        (written,) = [value for key, value in zip(execution.keys, execution.values, strict=True)
                      if key is not None and key.value == "loop_s"]  # fmt: skip
        return eval(compile(ast.Expression(body=written), RUNNER.name, "eval"), {"steps": steps, "sent_at": sent_at})

    def execution_written(self) -> ast.Dict:
        (call,) = [node for node in ast.walk(self.tree)
                   if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_record"]  # fmt: skip
        return next(keyword.value for keyword in call.keywords if keyword.arg == "execution")

    def test_the_runner_times_the_first_step_sent_and_the_latest_one_on_a_monotonic_clock(self):
        moments = iter([100.0, 100.031, 100.066, 100.1])
        on_step, steps, sent_at = self.step_callback(lambda: next(moments))
        self.assertEqual((steps, sent_at), ([0], [0.0, 0.0]))
        on_step(0, {})
        self.assertEqual((steps, sent_at), ([1], [100.0, 100.0]))
        on_step(1, {})
        on_step(2, {})
        on_step(3, {})
        self.assertEqual((steps, sent_at), ([4], [100.0, 100.1]), "the first moment stays, the latest follows")
        self.assertIn("import time\n", self.source)

    def test_the_loop_s_duration_it_writes_is_the_latest_step_minus_the_first_to_a_millisecond(self):
        self.assertEqual(self.loop_seconds([4], [100.0, 100.1]), 0.1)
        self.assertEqual(self.loop_seconds([301], [5000.25, 5010.2504]), 10.0)
        self.assertEqual(self.loop_seconds([301], [5000.25, 5010.7716]), 10.522)
        for steps, sent_at in (([0], [0.0, 0.0]), ([1], [100.0, 100.0]), ([1], [100.0, 100.4])):
            self.assertEqual(self.loop_seconds(steps, sent_at), 0.0, "fewer than two steps: no duration")
        self.assertIsInstance(self.loop_seconds([0], [0.0, 0.0]), float)

    def test_the_rate_the_reader_makes_of_what_the_runner_writes_is_the_rate_of_the_steps(self):
        # 301 steps sent 1/30 s apart: the runner's callback and its expression, then the reader's formula.
        moments = iter([50.0 + index / 30.0 for index in range(301)])
        on_step, steps, sent_at = self.step_callback(lambda: next(moments))
        for index in range(301):
            on_step(index, {})
        loop_s = self.loop_seconds(steps, sent_at)
        self.assertEqual((steps[0], loop_s), (301, 10.0))
        self.assertEqual(bookkeeping.achieved_rate_hz(steps[0], loop_s), 30.0)
        # A loop that takes 1/28 s a step, stamped or not: 28 Hz, to the millisecond the duration is kept to.
        moments = iter([7.5 + index / 28.0 for index in range(561)])
        on_step, steps, sent_at = self.step_callback(lambda: next(moments))
        for index in range(561):
            on_step(index, {})
        rate = bookkeeping.achieved_rate_hz(steps[0], self.loop_seconds(steps, sent_at))
        self.assertAlmostEqual(rate, 28.0, places=3)


def self_protocol() -> dict:
    """The repository's protocol file as the session reads it."""
    from tolerance_radius import config

    return config.load_protocol()


if __name__ == "__main__":
    unittest.main()
