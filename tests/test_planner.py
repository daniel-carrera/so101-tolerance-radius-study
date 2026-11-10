"""The trial planner: which trial comes next, given the protocol, the audited demonstrations and the log so far."""

import copy
import math
import re
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from functools import cache
from pathlib import Path

from fixtures_demos import GRID, PROTOCOL, audit
from tolerance_radius import config, geometry, planner, schema, staircase, targets
from tolerance_radius.planner import PlannerError
from tolerance_radius.schema import DemoStart, PlannedTrial, Point, TrialRecord
from tolerance_radius.seeds import derived_rng

#: Both audits together, as the planner is given them: it has to pick the set of the condition itself.
AUDITS = (*audit("clustered"), *audit("spread"))
#: One demonstration at the centre of each trained cell, in both sets: enough to plan with, and quick.
SPARSE = tuple(
    DemoStart(dataset, index, cell, measured=GRID.cell_centre(cell))
    for dataset in schema.DATASETS
    for index, cell in enumerate(GRID.trained_cells)
)
THRESHOLD_BLOCKS = ("T1a", "T1b", "T3")
#: The placements of the paired block, as the repository's protocol states them.
PAIRS = PROTOCOL["blocks"]["T2"]["pairs"]
MATCH = PROTOCOL["placement"]["level_match_mm"]
TIME = "2026-10-07T09:00:00-07:00"


def record(planned: PlannedTrial, outcome: str, demos=AUDITS) -> TrialRecord:
    """The record a trial would leave: the cube placed exactly on the target, scored as given."""
    if outcome == "void":
        return TrialRecord(planned=planned, outcome="void", ts_start=TIME, ts_end=TIME, void_reason="camera dropped")
    own = geometry.trained_demos(demos, planned.dataset)
    index, distance = geometry.nearest(planned.target, own)
    policy = {} if planned.condition == "replay" else {"policy_repo": "anonymous/policy", "checkpoint": "080000"}
    return TrialRecord(
        planned=planned,
        outcome=outcome,
        ts_start=TIME,
        ts_end=TIME,
        placed=planned.target,
        nearest_demo_episode=own[index].episode_index,
        nearest_demo_distance_mm=distance,
        subgoal=schema.SUBGOAL_MAX if outcome == "success" else 2,
        failure=None if outcome == "success" else "grasp",
        operator="op1",
        scorer="sc1",
        **policy,
    )


def by_distance(planned: PlannedTrial, number: int) -> str:
    """A responder whose success gets rarer with distance: half its plateau at 20 mm."""
    level = planned.level_mm if planned.level_mm is not None else 10.0
    chance = 0.97 / (1.0 + math.exp((level - 20.0) / 4.0))
    return "success" if derived_rng(99, planned.block, number).random() < chance else "failure"


def always(outcome: str):
    return lambda planned, number: outcome


def run(block: str, protocol=PROTOCOL, demos=AUDITS, outcome=by_distance, voids=(), log=()) -> list[TrialRecord]:
    """Run a block to its end against a responder. `voids` are the positions in the log that come out void."""
    log = list(log)
    while (planned := planner.next_trial(block, protocol, demos, log)) is not None:
        result = "void" if len(log) in voids else outcome(planned, len(log))
        log.append(record(planned, result, demos))
        if len(log) > 400:
            raise AssertionError(f"block {block} does not end")
    return log


@cache
def finished(block: str) -> tuple[TrialRecord, ...]:
    """The repository's protocol, the full audits, the responder by distance: run once, looked at by many tests."""
    return tuple(run(block))


def with_block(block: str, **changes) -> dict:
    """The repository's protocol with entries of one block changed."""
    protocol = copy.deepcopy(PROTOCOL)
    protocol["blocks"][block].update(changes)
    return protocol


def planned_total(block: str, protocol=PROTOCOL) -> int:
    """The scored trials a block plans, added up from its entry in the protocol and not by the planner."""
    entry = protocol["blocks"][block]
    if "pairs" in entry:
        return entry["pairs"] * len(entry["conditions"])
    in_staircases = sum(staircase["trials"] for staircase in entry["staircases"])
    return entry.get("anchors_zero", 0) + entry.get("anchors_far", 0) + in_staircases


def kind(planned: PlannedTrial) -> str:
    """Which part of a threshold block a trial belongs to."""
    if planned.staircase_id is not None:
        return planned.staircase_id
    return "zero" if planned.level_mm == 0 else "far"


@cache
def usual_order(block: str, seed: int | None) -> tuple[str, ...]:
    """The parts of a block's trials in the order they are run, on the sparse audit: looked at by several tests."""
    protocol = PROTOCOL if seed is None else with_block(block, seed=seed)
    return tuple(kind(r.planned) for r in run(block, protocol=protocol, demos=SPARSE))


def levels_of(log, part: str) -> list[float]:
    """The distances the scored trials of one part of a block asked for, in order."""
    return [r.planned.level_mm for r in log if r.scored and kind(r.planned) == part]


def targets_of(log, part: str) -> list[Point]:
    """The targets of the trials of one part of a block, in order."""
    return [r.planned.target for r in log if kind(r.planned) == part]


def differs_only_in_its_labels(first: PlannedTrial, second: PlannedTrial) -> bool:
    return replace(first, trial_id="-", blind_code="-") == replace(second, trial_id="-", blind_code="-")


class ThresholdBlockCountTests(unittest.TestCase):
    def test_each_block_runs_to_its_end_with_the_planned_numbers_of_trials(self):
        expected = {
            "T1a": {"zero": 6, "far": 4, "T1a-up": 12, "T1a-down": 12},
            "T1b": {"zero": 4, "T1b-up": 9, "T1b-down": 9},
            "T3": {"zero": 4, "T3-up": 10, "T3-down": 10},
        }
        for block in THRESHOLD_BLOCKS:
            log = finished(block)
            with self.subTest(block=block):
                self.assertEqual(Counter(kind(r.planned) for r in log), expected[block])
                self.assertTrue(all(r.planned.block == block for r in log))
                self.assertIsNone(planner.next_trial(block, PROTOCOL, AUDITS, log))

    def test_every_trial_has_the_condition_of_its_block(self):
        for block, condition in (("T1a", "act_clustered"), ("T1b", "replay"), ("T3", "act_spread")):
            for trial in finished(block):
                self.assertEqual(trial.planned.condition, condition)
                self.assertIsNone(trial.planned.pair_id)
                self.assertEqual(trial.planned.replay_episode is not None, condition == "replay")

    def test_a_block_ends_whatever_the_outcomes(self):
        for block in THRESHOLD_BLOCKS:
            for outcome in ("success", "failure"):
                with self.subTest(block=block, outcome=outcome):
                    log = run(block, demos=SPARSE, outcome=always(outcome))
                    self.assertEqual(len(log), len(finished(block)))
                    self.assertIsNone(planner.next_trial(block, PROTOCOL, SPARSE, log))


class StaircaseTests(unittest.TestCase):
    def test_parameters_come_from_the_block_and_the_staircase_section(self):
        params = planner.staircase_params("T1a", PROTOCOL)
        self.assertEqual(list(params), ["T1a-up", "T1a-down"])
        self.assertEqual(params["T1a-up"], staircase.StaircaseParams(10.0, 12, 10.0, 5.0, 0.0, 50.0))
        self.assertEqual(params["T1a-down"], staircase.StaircaseParams(40.0, 12, 10.0, 5.0, 0.0, 50.0))
        self.assertEqual(planner.staircase_params("T1b", PROTOCOL)["T1b-down"].max_level_mm,
                         PROTOCOL["blocks"]["T1b"]["max_level_mm"])
        self.assertEqual(planner.staircase_params("T3", PROTOCOL)["T3-up"].n_trials, 10)

    def test_a_block_without_staircases_or_an_unknown_rule_is_refused(self):
        with self.assertRaises(PlannerError):
            planner.staircase_params("T2", PROTOCOL)
        protocol = copy.deepcopy(PROTOCOL)
        protocol["staircase"]["rule"] = "2up1down"
        with self.assertRaises(PlannerError):
            planner.staircase_params("T1a", protocol)

    def test_levels_follow_the_staircase_rule_applied_to_the_staircase_s_own_history(self):
        for block in THRESHOLD_BLOCKS:
            for staircase_id, params in planner.staircase_params(block, PROTOCOL).items():
                trials = [r for r in finished(block) if r.planned.staircase_id == staircase_id]
                history: list[tuple[float, bool]] = []
                for index, trial in enumerate(trials):
                    with self.subTest(staircase=staircase_id, index=index):
                        self.assertEqual(trial.planned.staircase_index, index)
                        self.assertEqual(trial.planned.level_mm, staircase.next_level(params, history))
                    history.append((trial.planned.level_mm, trial.success))
                self.assertIsNone(staircase.next_level(params, history))

    def test_the_levels_respond_to_the_outcomes(self):
        succeeding = run("T3", demos=SPARSE, outcome=always("success"))
        self.assertEqual(levels_of(succeeding, "T3-up")[:4], [5.0, 15.0, 25.0, 35.0])
        failing = run("T3", demos=SPARSE, outcome=always("failure"))
        self.assertEqual(levels_of(failing, "T3-down")[:4], [25.0, 15.0, 5.0, 0.0])

    def test_void_trials_are_left_out_of_a_staircase_s_history(self):
        clean = run("T3", demos=SPARSE, outcome=always("success"))
        voided = run("T3", demos=SPARSE, outcome=always("success"), voids={3, 4, 9})
        self.assertEqual(len(voided), len(clean) + 3)
        for part in ("T3-up", "T3-down"):
            self.assertEqual(levels_of(voided, part), levels_of(clean, part))


class PlacementTests(unittest.TestCase):
    def test_policy_staircase_targets_have_the_requested_distance_and_lie_in_the_test_region(self):
        for block in ("T1a", "T3"):
            own = geometry.trained_demos(AUDITS, schema.DATASET_OF_CONDITION[PROTOCOL["blocks"][block]["condition"]])
            for trial in finished(block):
                planned = trial.planned
                with self.subTest(trial=planned.trial_id):
                    self.assertTrue(GRID.in_test_region(planned.target))
                    if planned.staircase_id is not None or kind(planned) == "far":
                        distance = geometry.nearest(planned.target, own)[1]
                        self.assertLessEqual(abs(distance - planned.level_mm), MATCH + 1e-9)

    def test_zero_anchors_sit_exactly_on_measured_starts_of_trained_demonstrations_in_distinct_cells(self):
        for block, dataset in (("T1a", "clustered"), ("T3", "spread")):
            own = {d.measured: d for d in geometry.trained_demos(AUDITS, dataset)}
            anchors = [r.planned for r in finished(block) if kind(r.planned) == "zero"]
            with self.subTest(block=block):
                self.assertTrue(all(a.target in own and a.level_mm == 0.0 for a in anchors))
                cells = [own[a.target].cell for a in anchors]
                self.assertEqual(len(set(cells)), len(cells))

    def test_far_anchors_are_at_the_far_level(self):
        own = geometry.trained_demos(AUDITS, "clustered")
        far = [r.planned for r in finished("T1a") if kind(r.planned) == "far"]
        self.assertEqual(len(far), 4)
        far_level = PROTOCOL["blocks"]["T1a"]["far_level_mm"]  # read from the protocol: the mechanism is under test
        for planned in far:
            self.assertEqual(planned.level_mm, far_level)
            self.assertAlmostEqual(geometry.nearest(planned.target, own)[1], far_level, delta=MATCH + 1e-9)
        self.assertEqual(len({p.target for p in far}), 4)

    def test_two_trials_at_the_same_level_get_different_targets(self):
        log = run("T3", outcome=always("failure"))  # both staircases sink to 0 mm and stay there
        at_zero = [r.planned.target for r in log if r.planned.staircase_id is not None and r.planned.level_mm == 0.0]
        self.assertGreater(len(at_zero), 8)
        self.assertEqual(len(set(at_zero)), len(at_zero))

    def test_a_demonstration_left_out_of_training_is_never_an_anchor_and_distances_ignore_it(self):
        left_out = {0, 7, 19, 33, 54}
        demos = tuple(replace(d, trained=d.episode_index not in left_out) for d in audit("spread"))
        trained = geometry.trained_demos(demos, "spread")
        measured = {d.measured for d in trained}
        for trial in run("T3", demos=demos):
            planned = trial.planned
            if kind(planned) == "zero":
                self.assertIn(planned.target, measured)
            else:
                self.assertLessEqual(abs(geometry.nearest(planned.target, trained)[1] - planned.level_mm), MATCH + 1e-9)

    def test_a_demonstration_measured_in_the_margin_is_never_an_anchor_nor_replayed(self):
        # Test placements stay out of the margin, and a trial at distance 0 is a test placement. Here four cells have
        # their demonstration at the centre and the seven others have it 2 mm from the grid's outer edge.
        inside = {"A2", "B1", "C2", "D3"}

        def start(cell: str) -> Point:
            rect, centre = GRID.cell_rect(cell), GRID.cell_centre(cell)
            if cell in inside:
                return centre
            if rect.x_min_mm == 0.0 or rect.x_max_mm == GRID.width_mm:
                return Point(2.0 if rect.x_min_mm == 0.0 else GRID.width_mm - 2.0, centre.y_mm)
            return Point(centre.x_mm, 2.0 if rect.y_min_mm == 0.0 else GRID.height_mm - 2.0)

        demos = [
            DemoStart(dataset, index, cell, measured=start(cell))
            for dataset in schema.DATASETS
            for index, cell in enumerate(GRID.trained_cells)
        ]
        self.assertEqual(sum(GRID.in_test_region(d.measured) for d in demos), 8)
        for block in ("T1a", "T3"):
            anchors = targets_of(run(block, with_block(block, anchors_zero=4), demos), "zero")
            self.assertEqual(set(anchors), {GRID.cell_centre(cell) for cell in inside})
        four = with_block("T1b", replay_demos=4)
        self.assertEqual({d.cell for d in planner.replay_demos("T1b", four, demos)}, inside)
        with self.assertRaises(PlannerError):
            planner.replay_demos("T1b", with_block("T1b", replay_demos=5), demos)
        # The protocol asks for a demonstration from every trained cell: here only four cells can give one, and the
        # check made before a block's first trial says so.
        with self.assertRaisesRegex(PlannerError, "4 cells"):
            planner.replay_demos("T1b", PROTOCOL, demos)
        (problem,) = planner.check_block("T1b", PROTOCOL, demos)
        self.assertIn("4 cells", problem)

    def test_a_level_that_cannot_be_placed_raises_an_error_that_says_what_can(self):
        staircases = [{"id": name, "start_level_mm": 190.0, "trials": 3} for name in ("up", "down")]
        protocol = with_block("T3", max_level_mm=200.0, staircases=staircases, anchors_zero=0)
        largest = targets.max_feasible_level(audit("spread"), GRID)
        with self.assertRaises(PlannerError) as caught:
            planner.next_trial("T3", protocol, AUDITS, [])
        message = str(caught.exception)
        self.assertIn("190", message)
        self.assertIn(f"{largest:.1f}", message)
        self.assertIn("T3", message)

    def test_a_far_level_that_cannot_be_placed_raises_too(self):
        staircases = [{"id": "up", "start_level_mm": 10.0, "trials": 1}]
        protocol = with_block("T1a", far_level_mm=120.0, anchors_zero=0, staircases=staircases)
        with self.assertRaisesRegex(PlannerError, "120"):
            run("T1a", protocol=protocol, demos=SPARSE)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.log = finished("T1b")
        self.by_episode = {d.episode_index: d for d in geometry.trained_demos(AUDITS, "clustered")}

    def test_the_replayed_demonstrations_are_a_fixed_set_one_from_each_trained_cell(self):
        chosen = planner.replay_demos("T1b", PROTOCOL, AUDITS)
        self.assertEqual(sorted(d.cell for d in chosen), sorted(GRID.trained_cells))
        self.assertEqual(len(chosen), PROTOCOL["blocks"]["T1b"]["replay_demos"])
        self.assertTrue(all(d.dataset == "clustered" and d.trained for d in chosen))
        self.assertEqual({r.planned.replay_episode for r in self.log}, {d.episode_index for d in chosen})
        self.assertEqual(chosen, planner.replay_demos("T1b", PROTOCOL, tuple(reversed(AUDITS))))

    def test_the_protocol_s_block_replays_each_of_its_demonstrations_twice_in_two_different_parts(self):
        # What protocol/protocol.md (section 6) and the paper's appendix say of the block as it is planned. It holds
        # because the block has two trials for each demonstration: change one number and the texts change too.
        wanted = PROTOCOL["blocks"]["T1b"]["replay_demos"]
        self.assertEqual(planned_total("T1b"), 2 * wanted)
        used = Counter(r.planned.replay_episode for r in self.log)
        self.assertEqual((len(used), set(used.values())), (wanted, {2}))
        for part in ("zero", "T1b-up", "T1b-down"):
            in_part = [r.planned.replay_episode for r in self.log if kind(r.planned) == part]
            with self.subTest(part=part):
                self.assertEqual(len(set(in_part)), len(in_part))

    def test_the_pilot_replays_each_of_the_block_s_demonstrations_once_at_its_own_start(self):
        # Tuesday's replay pilot (`config/pilot.toml`): the counted block tests only its four anchors' demonstrations
        # at their own start, so the pilot is the one look at each of the eleven.
        pilot = config.load_protocol(overlay=config.PILOT_OVERLAY_PATH)
        wanted = PROTOCOL["blocks"]["T1b"]["replay_demos"]
        self.assertEqual(pilot["blocks"]["T1b"]["anchors_zero"], wanted)
        log = run("T1b", protocol=pilot)
        at_own_start = [r.planned.replay_episode for r in log if kind(r.planned) == "zero"]
        self.assertEqual(len(at_own_start), wanted)
        self.assertEqual(len(set(at_own_start)), wanted)
        # and they are the demonstrations the counted block replays
        self.assertEqual(set(at_own_start), {d.episode_index for d in planner.replay_demos("T1b", PROTOCOL, AUDITS)})
        self.assertEqual(len(log), wanted + sum(s["trials"] for s in pilot["blocks"]["T1b"]["staircases"]))

    def test_the_size_of_the_set_can_be_given_in_the_protocol(self):
        chosen = planner.replay_demos("T1b", with_block("T1b", replay_demos=6), AUDITS)
        self.assertEqual(len({d.cell for d in chosen}), 6)
        with self.assertRaises(PlannerError):
            planner.replay_demos("T1b", with_block("T1b", replay_demos=12), AUDITS)  # there are 11 trained cells
        with self.assertRaises(PlannerError):
            planner.replay_demos("T1a", PROTOCOL, AUDITS)  # not a replay block

    def test_a_zero_anchor_is_exactly_the_start_of_the_demonstration_it_replays(self):
        anchors = [r.planned for r in self.log if kind(r.planned) == "zero"]
        self.assertEqual(len(anchors), PROTOCOL["blocks"]["T1b"]["anchors_zero"])
        for planned in anchors:
            self.assertEqual(planned.target, self.by_episode[planned.replay_episode].measured)
            self.assertEqual(planned.level_mm, 0.0)

    def test_a_staircase_target_is_at_the_requested_distance_from_the_demonstration_it_replays(self):
        for trial in self.log:
            planned = trial.planned
            if planned.staircase_id is not None:
                start = self.by_episode[planned.replay_episode].measured
                self.assertAlmostEqual(planned.target.distance_to(start), planned.level_mm, places=9)
                self.assertTrue(GRID.in_test_region(planned.target))

    def uses(self, protocol, demos=SPARSE) -> tuple[dict[str, Counter], Counter]:
        """How often each demonstration of the replay set is replayed, in each part of the block and in all of it.

        A demonstration that a part never replays is counted there with 0, so that "as often as any other" is asked
        of the whole set and not only of the demonstrations a part happens to use.
        """
        chosen = [demo.episode_index for demo in planner.replay_demos("T1b", protocol, demos)]
        log = run("T1b", protocol, demos, outcome=always("failure"))
        by_part: dict[str, Counter] = {}
        for trial in log:
            used = by_part.setdefault(kind(trial.planned), Counter(dict.fromkeys(chosen, 0)))
            used[trial.planned.replay_episode] += 1
        overall = Counter(dict.fromkeys(chosen, 0))
        for used in by_part.values():
            overall.update(used)
        return by_part, overall

    def test_the_demonstrations_are_used_in_balance_in_every_part_and_over_the_block(self):
        # Whatever the size of the set: no demonstration is replayed twice in a part while another has not been
        # replayed there as often, and the same holds over the whole block.
        for size in (1, 3, 4, 6, len(GRID.trained_cells)):
            for seed in (1, 2, 3):
                by_part, overall = self.uses(with_block("T1b", replay_demos=size, seed=seed))
                with self.subTest(size=size, seed=seed):
                    self.assertEqual(set(by_part), {"zero", "T1b-up", "T1b-down"})
                    for used in (*by_part.values(), overall):
                        self.assertEqual(len(used), size)
                        self.assertLessEqual(max(used.values()) - min(used.values()), 1)

    def test_a_set_smaller_than_every_part_is_replayed_in_every_part(self):
        # Four demonstrations, four anchors and staircases of nine: each demonstration has one anchor, and two or
        # three trials of each staircase.
        by_part, overall = self.uses(with_block("T1b", replay_demos=4))
        self.assertEqual(sorted(by_part["zero"].values()), [1, 1, 1, 1])
        for staircase_id in ("T1b-up", "T1b-down"):
            self.assertEqual(sorted(by_part[staircase_id].values()), [2, 2, 2, 3])
        self.assertEqual(sorted(overall.values()), [5, 5, 6, 6])

    def test_a_demonstration_from_every_trained_cell_is_replayed_twice_in_two_different_parts(self):
        # Eleven demonstrations over four anchors and two staircases of nine, 22 trials: the anchors are four
        # different demonstrations, a staircase replays nine different ones, and each demonstration is replayed
        # exactly twice in the block, never twice in the same part.
        cells = len(GRID.trained_cells)
        sizes = {"zero": 4, "T1b-up": 9, "T1b-down": 9}
        self.assertEqual(sum(sizes.values()), 2 * cells, "the test is written for a block of two trials for each cell")
        starts = (("up", 5.0), ("down", 25.0))
        staircases = [{"id": name, "start_level_mm": start, "trials": 9} for name, start in starts]
        for seed in range(1, 9):
            block = with_block("T1b", replay_demos=cells, seed=seed, anchors_zero=4, staircases=staircases)
            by_part, overall = self.uses(block)
            with self.subTest(seed=seed):
                for part, size in sizes.items():
                    self.assertEqual(sum(by_part[part].values()), size)
                    self.assertEqual(max(by_part[part].values()), 1)
                self.assertEqual(set(overall.values()), {2})

    def test_which_demonstrations_have_an_anchor_depends_on_the_seed(self):
        cells = len(GRID.trained_cells)
        anchored = set()
        for seed in range(1, 9):
            by_part, _ = self.uses(with_block("T1b", replay_demos=cells, seed=seed))
            anchored.add(frozenset(episode for episode, count in by_part["zero"].items() if count))
        self.assertGreater(len(anchored), 4)

    def test_a_short_block_replays_no_demonstration_twice_before_every_other_was_replayed(self):
        # A block with fewer trials than twice its demonstrations (the pilot's replay block is one): 3 anchors and
        # two staircases of 5 over eleven demonstrations replay nine of them once and two of them twice.
        cells = len(GRID.trained_cells)
        starts = (("up", 5.0), ("down", 25.0))
        staircases = [{"id": name, "start_level_mm": start, "trials": 5} for name, start in starts]
        by_part, overall = self.uses(with_block("T1b", replay_demos=cells, anchors_zero=3, staircases=staircases))
        self.assertEqual(sorted(overall.values()), [1] * 9 + [2] * 2)
        self.assertTrue(all(max(used.values()) == 1 for used in by_part.values()))

    def test_the_pilot_s_shorter_block_replays_demonstrations_of_the_counted_block_s_own_set(self):
        # The pilot file changes the block's size and nothing about which demonstrations are played back: a
        # demonstration tried in the pilot is one the counted block will replay, given the same audit.
        pilot = config.load_protocol(overlay=config.PILOT_OVERLAY_PATH)
        counted = planner.replay_demos("T1b", PROTOCOL, AUDITS)
        self.assertEqual(planner.replay_demos("T1b", pilot, AUDITS), counted)
        tried = {r.planned.replay_episode for r in run("T1b", pilot, AUDITS, outcome=always("success"))}
        self.assertLessEqual(tried, {demo.episode_index for demo in counted})

    def test_without_an_entry_the_set_has_the_planner_s_default_size(self):
        protocol = copy.deepcopy(PROTOCOL)
        del protocol["blocks"]["T1b"]["replay_demos"]
        self.assertEqual(len(planner.replay_demos("T1b", protocol, AUDITS)), planner.DEFAULT_REPLAY_DEMOS)

    def test_the_demonstration_replayed_does_not_depend_on_the_outcomes(self):
        failing = run("T1b", outcome=always("failure"))
        self.assertEqual([r.planned.replay_episode for r in failing], [r.planned.replay_episode for r in self.log])

    def test_directions_vary_from_trial_to_trial(self):
        angles = []
        for trial in run("T1b", outcome=always("success")):
            planned = trial.planned
            if planned.staircase_id is not None:
                start = self.by_episode[planned.replay_episode].measured
                angles.append(math.atan2(planned.target.y_mm - start.y_mm, planned.target.x_mm - start.x_mm))
        self.assertEqual(len({round(angle, 6) for angle in angles}), len(angles))
        self.assertTrue(any(angle > 0 for angle in angles) and any(angle < 0 for angle in angles))

    def test_a_distance_at_which_no_target_fits_the_table_raises_a_clear_error(self):
        staircases = [{"id": "up", "start_level_mm": 400.0, "trials": 2}]
        protocol = with_block("T1b", max_level_mm=400.0, staircases=staircases, anchors_zero=0)
        with self.assertRaisesRegex(PlannerError, "400"):
            planner.next_trial("T1b", protocol, AUDITS, [])


class OrderTests(unittest.TestCase):
    def order(self, block: str, seed: int | None = None, outcome=None) -> list[str]:
        if outcome is None:
            return list(usual_order(block, seed))
        protocol = PROTOCOL if seed is None else with_block(block, seed=seed)
        return [kind(r.planned) for r in run(block, protocol=protocol, demos=SPARSE, outcome=outcome)]

    def test_the_order_is_fixed_by_the_seed_whatever_the_outcomes(self):
        for block in THRESHOLD_BLOCKS:
            with self.subTest(block=block):
                succeeding = self.order(block, outcome=always("success"))
                self.assertEqual(succeeding, self.order(block, outcome=always("failure")))
                self.assertEqual(self.order(block), [kind(r.planned) for r in finished(block)])

    def test_another_seed_gives_another_order(self):
        for block in THRESHOLD_BLOCKS:
            orders = {tuple(self.order(block, seed)) for seed in range(1, 7)}
            self.assertEqual(len(orders), 6)

    def test_anchors_are_spread_through_the_block(self):
        # The j-th of n anchors of a kind falls in the j-th n-th of the block, give or take one trial for each of
        # the block's parts (the bound below is exact).
        for block in THRESHOLD_BLOCKS:
            for seed in range(1, 13):
                order = self.order(block, seed)
                sizes = Counter(order)
                for part in ("zero", "far"):
                    positions = [index for index, name in enumerate(order) if name == part]
                    n = len(positions)
                    for j, position in enumerate(positions):
                        earliest = sum(size * j // n for size in sizes.values())
                        latest = sum(-(-size * (j + 1) // n) for size in sizes.values()) - 1
                        with self.subTest(block=block, seed=seed, part=part, j=j):
                            self.assertTrue(earliest <= position <= latest, (earliest, position, latest))

    def test_no_half_of_a_block_holds_all_the_anchors(self):
        for block in THRESHOLD_BLOCKS:
            for seed in range(1, 13):
                order = self.order(block, seed)
                half = len(order) // 2
                for part in {"zero", "far"} & set(order):
                    total = order.count(part)
                    first_half = order[:half].count(part)
                    with self.subTest(block=block, seed=seed, part=part):
                        self.assertTrue(total / 4 <= first_half <= 3 * total / 4, (first_half, total))

    def test_the_two_staircases_advance_together_in_an_order_that_cannot_be_foreseen(self):
        for block in THRESHOLD_BLOCKS:
            first_of_each_round = Counter()
            for seed in range(1, 13):
                steps = [name for name in self.order(block, seed) if name not in ("zero", "far")]
                up, down = f"{block}-up", f"{block}-down"
                for count in range(1, len(steps) + 1):
                    lead = steps[:count].count(up) - steps[:count].count(down)
                    self.assertLessEqual(abs(lead), 1)
                first_of_each_round.update(steps[::2])
            self.assertGreater(min(first_of_each_round.values()), 0.3 * sum(first_of_each_round.values()))

    def test_the_first_trial_of_a_block_is_not_always_of_the_same_kind(self):
        firsts = {self.order("T1a", seed)[0] for seed in range(1, 25)}
        self.assertGreaterEqual(len(firsts), 3)


class RestartTests(unittest.TestCase):
    def test_replaying_any_prefix_of_the_log_gives_the_trial_that_followed_it(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log = finished(block)
            for count in range(len(log) + 1):
                expected = log[count].planned if count < len(log) else None
                with self.subTest(block=block, count=count):
                    self.assertEqual(planner.next_trial(block, PROTOCOL, AUDITS, log[:count]), expected)

    def test_the_log_survives_a_round_trip_through_json_and_plans_the_same(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log = finished(block)[:11]
            reread = [schema.trial_record_from_json(schema.to_json(r)) for r in log]
            self.assertEqual(reread, list(log))
            self.assertEqual(planner.next_trial(block, PROTOCOL, AUDITS, reread), finished(block)[11].planned)

    def test_the_same_inputs_give_the_same_trial_and_are_left_untouched(self):
        protocol = copy.deepcopy(PROTOCOL)
        log = list(finished("T1a")[:9])
        demos = list(AUDITS)
        first = planner.next_trial("T1a", protocol, demos, log)
        self.assertEqual(planner.next_trial("T1a", protocol, demos, log), first)
        self.assertEqual(protocol, PROTOCOL)
        self.assertEqual(log, list(finished("T1a")[:9]))
        self.assertEqual(demos, list(AUDITS))

    def test_the_order_in_which_the_audit_lists_the_demonstrations_does_not_matter(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log = finished(block)[:7]
            shuffled_audit = tuple(reversed(AUDITS))
            self.assertEqual(planner.next_trial(block, PROTOCOL, shuffled_audit, log), finished(block)[7].planned)

    def test_trials_of_other_blocks_in_the_log_do_not_matter(self):
        others = [*finished("T1b"), *finished("T2")]
        log = finished("T1a")
        for count in (0, 5, 20):
            mixed = [*others[:10], *log[:count], *others[10:]]
            self.assertEqual(planner.next_trial("T1a", PROTOCOL, AUDITS, mixed), log[count].planned)
        self.assertIsNone(planner.next_trial("T1a", PROTOCOL, AUDITS, [*others, *log]))

    def test_the_demonstrations_of_the_other_set_do_not_matter(self):
        log = finished("T3")[:6]
        self.assertEqual(planner.next_trial("T3", PROTOCOL, audit("spread"), log), finished("T3")[6].planned)

    def test_the_plans_are_pinned_so_that_every_environment_plans_the_same(self):
        # Trials are planned with the rig's NumPy and looked at again with another. For each block: its first trial
        # (blind code, part, distance, target), the order of its parts or policies, and its first blind codes.
        pinned = {
            "T1a": ("W57", "zero", 0.0, (74.38030979409723, 24.831582709257603)),
            "T1b": ("V89", "T1b-up", 5.0, (224.59437820377693, 71.00774906614826)),
            "T2": ("L95", "pair", None, (15.45393225652828, 89.64261437177282)),
            "T3": ("E47", "T3-down", 25.0, (168.93146983615986, 83.26422369633167)),
        }
        orders = {
            "T1a": "ZUFDUDDZUFDUUDZDUUDUZDFUDZDUDZUFDU",
            "T1b": "UDUZDDUZUDDUDUDZUZUDDU",
            "T2": "sccssccsscsccscscscsscsccssccssccsscsccssccssccssccs",
            "T3": "DUZUDUDDZUUDDUUDDZUDUZDU",
        }
        codes = {
            "T1a": "W57 S63 N67 Z68 V27 B74",
            "T1b": "V89 K54 Y79 L56 K39 U25",
            "T2": "L95 N85 D22 X68 D93 H52",
            "T3": "E47 U63 C29 C36 E82 M22",
        }
        letter = {"zero": "Z", "far": "F", "up": "U", "down": "D", "act_clustered": "c", "act_spread": "s"}
        for block, (blind_code, part, level, (x, y)) in pinned.items():
            log = finished(block)
            planned = log[0].planned
            with self.subTest(block=block):
                self.assertEqual(planned.trial_id, f"{block}-001")
                self.assertEqual((planned.blind_code, planned.level_mm), (blind_code, level))
                self.assertEqual("pair" if planned.pair_id else kind(planned), part)
                self.assertAlmostEqual(planned.target.x_mm, x, places=9)
                self.assertAlmostEqual(planned.target.y_mm, y, places=9)
                names = [r.planned.condition if block == "T2" else kind(r.planned).split("-")[-1] for r in log]
                self.assertEqual("".join(letter[name] for name in names), orders[block])
                self.assertEqual(" ".join(r.planned.blind_code for r in log[:6]), codes[block])


class VoidTests(unittest.TestCase):
    def test_a_void_trial_is_asked_again_with_the_same_target(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            voids = {0, 3, 4, 10, 17, len(finished(block))}
            log = run(block, voids=voids)
            for position in sorted(voids):
                void, again = log[position].planned, log[position + 1].planned
                with self.subTest(block=block, position=position):
                    self.assertEqual(log[position].outcome, "void")
                    self.assertTrue(differs_only_in_its_labels(void, again))
                    self.assertEqual(void.target, again.target)
                    self.assertNotEqual(void.trial_id, again.trial_id)
                    self.assertNotEqual(void.blind_code, again.blind_code)

    def test_void_trials_do_not_count_towards_the_block(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log = run(block, voids={1, 2, 8, 20})
            scored = [r for r in log if r.scored]
            with self.subTest(block=block):
                self.assertEqual(len(log), len(finished(block)) + 4)
                expected = Counter(kind(r.planned) for r in finished(block))
                self.assertEqual(Counter(kind(r.planned) for r in scored), expected)

    def test_a_trial_that_is_void_again_and_again_keeps_being_asked(self):
        log: list[TrialRecord] = list(finished("T1a")[:5])
        expected = finished("T1a")[5].planned
        for _ in range(4):
            planned = planner.next_trial("T1a", PROTOCOL, AUDITS, log)
            self.assertTrue(differs_only_in_its_labels(planned, expected))
            log.append(record(planned, "void"))
        self.assertEqual([r.planned.trial_id for r in log[5:]], ["T1a-006", "T1a-007", "T1a-008", "T1a-009"])


class TrialIdTests(unittest.TestCase):
    def test_ids_count_every_trial_of_the_block_void_ones_too(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log = run(block, voids={2, 3, 9})
            self.assertEqual([r.planned.trial_id for r in log], [f"{block}-{n:03d}" for n in range(1, len(log) + 1)])

    def test_ids_are_unique_across_the_whole_log(self):
        log = [r for block in (*THRESHOLD_BLOCKS, "T2") for r in finished(block)]
        self.assertEqual(len({r.planned.trial_id for r in log}), len(log))

    def test_an_id_already_in_the_log_is_never_given_again(self):
        # A log whose third trial is missing: the count says "T1a-005" twice in a row unless the planner looks.
        log = [*finished("T1a")[:2], *finished("T1a")[3:5]]
        planned = planner.next_trial("T1a", PROTOCOL, AUDITS, log)
        self.assertNotIn(planned.trial_id, {r.planned.trial_id for r in log})


class PairedBlockTests(unittest.TestCase):
    def setUp(self):
        self.log = finished("T2")

    def test_every_placement_is_tested_once_with_each_policy(self):
        self.assertEqual(len(self.log), 2 * PAIRS)
        pairs: dict[str, list[PlannedTrial]] = {}
        for trial in self.log:
            pairs.setdefault(trial.planned.pair_id, []).append(trial.planned)
        self.assertEqual(len(pairs), PAIRS)
        for pair_id, (first, second) in pairs.items():
            with self.subTest(pair=pair_id):
                self.assertEqual(first.target, second.target)
                self.assertEqual({first.condition, second.condition}, {"act_clustered", "act_spread"})
        self.assertIsNone(planner.next_trial("T2", PROTOCOL, AUDITS, self.log))

    def test_the_two_trials_of_a_pair_come_back_to_back(self):
        ids = [trial.planned.pair_id for trial in self.log]
        self.assertEqual(ids[0::2], ids[1::2])
        self.assertEqual(len(set(ids[0::2])), PAIRS)
        self.assertEqual(ids[0], "T2-p01")
        self.assertEqual(ids[-1], f"T2-p{PAIRS:02d}")

    def test_which_policy_goes_first_varies_and_each_goes_first_in_half_of_the_pairs(self):
        first = [trial.planned.condition for trial in self.log[0::2]]
        # With an odd number of pairs the policy the protocol lists first would go first once more.
        self.assertEqual(Counter(first), {"act_clustered": PAIRS - PAIRS // 2, "act_spread": PAIRS // 2})
        self.assertNotEqual(first, sorted(first))
        self.assertNotEqual(first, sorted(first, reverse=True))
        self.assertNotEqual(first[0::2], first[1::2])

    def test_the_placements_are_drawn_over_the_protocol_s_regions_and_fixed_by_the_placement_seed(self):
        placements = planner.paired_placements("T2", PROTOCOL)
        self.assertEqual(len(placements), PAIRS)
        self.assertEqual([trial.planned.target for trial in self.log[0::2]], placements)
        self.assertEqual(len(set(placements)), PAIRS)
        # So many in each region as the protocol says, and they are that region's own seeded draws.
        seed = PROTOCOL["blocks"]["T2"]["placement_seed"]
        for stratum in PROTOCOL["blocks"]["T2"]["regions"]:
            region, count = stratum["region"], stratum["pairs"]
            inside = [point for point in placements if GRID.in_region(point, region)]
            drawn = targets.uniform_placements(GRID, count, derived_rng(seed, "T2", "placements", region), region)
            with self.subTest(region=region):
                self.assertEqual(len(inside), count)
                self.assertEqual(set(inside), set(drawn))
        other = planner.paired_placements("T2", with_block("T2", placement_seed=1))
        self.assertNotEqual(placements, other)
        # A seed of its own changes the block's order and nothing about where the cube goes.
        reseeded = with_block("T2", seed=5)
        self.assertEqual(planner.paired_placements("T2", reseeded), placements)
        self.assertNotEqual([r.planned.condition for r in run("T2", reseeded)], [r.planned.condition for r in self.log])

    def test_a_block_with_one_region_draws_every_placement_there(self):
        # The other form of the entry: `region` in place of the list. Without either, the trained cells.
        seed = PROTOCOL["blocks"]["T2"]["placement_seed"]
        for region in ("trained_cells", "held_out_cells", "test_region"):
            placements = planner.paired_placements("T2", single_region(20, region))
            expected = targets.uniform_placements(GRID, 20, derived_rng(seed, "T2", "placements"), region)
            with self.subTest(region=region):
                self.assertEqual(placements, expected)
                self.assertTrue(all(GRID.in_region(point, region) for point in placements))
        unnamed = single_region(20, "trained_cells")
        del unnamed["blocks"]["T2"]["region"]
        self.assertEqual(planner.paired_placements("T2", unnamed),
                         planner.paired_placements("T2", single_region(20, "trained_cells")))
        log = run("T2", single_region(20, "trained_cells"))
        self.assertEqual(len(log), 40)
        self.assertEqual(planner.block_status("T2", single_region(20, "trained_cells"), log)["pairs"],
                         {"done": 20, "planned": 20})

    def test_a_paired_trial_asks_for_no_distance_and_belongs_to_no_staircase(self):
        for trial in self.log:
            planned = trial.planned
            self.assertIsNone(planned.level_mm)
            self.assertIsNone(planned.staircase_id)
            self.assertIsNone(planned.staircase_index)
            self.assertIsNone(planned.replay_episode)

    def test_the_placements_do_not_wait_for_an_audit(self):
        self.assertEqual(planner.next_trial("T2", PROTOCOL, [], []), self.log[0].planned)

    def test_the_outcomes_change_nothing_in_a_paired_block(self):
        failing = run("T2", outcome=always("failure"))
        self.assertEqual([r.planned for r in failing], [r.planned for r in self.log])

    def test_a_void_second_trial_does_not_bring_the_first_back(self):
        log = run("T2", voids={1, 2})
        self.assertEqual([r.outcome == "void" for r in log[:5]], [False, True, True, False, False])
        self.assertEqual([r.planned.pair_id for r in log[:5]], ["T2-p01", "T2-p01", "T2-p01", "T2-p01", "T2-p02"])
        self.assertEqual(len({r.planned.condition for r in log[1:4]}), 1)
        self.assertNotEqual(log[0].planned.condition, log[1].planned.condition)


def single_region(pairs: int, region: str, **changes) -> dict:
    """The repository's protocol with the paired block's placements all drawn over one region, so many of them."""
    protocol = with_block("T2", pairs=pairs, region=region, **changes)
    protocol["blocks"]["T2"].pop("regions", None)
    return protocol


def with_regions(*strata: tuple[str, int], **changes) -> dict:
    """The repository's protocol with the paired block's placements drawn over several regions, so many in each."""
    protocol = with_block("T2", pairs=sum(count for _, count in strata),
                          regions=[{"region": region, "pairs": count} for region, count in strata], **changes)
    protocol["blocks"]["T2"].pop("region", None)
    return protocol


class PairedBlockRegionsTests(unittest.TestCase):
    """A paired block may draw a stated number of placements from each of several regions."""

    HALVES = (("trained_cells", 13), ("held_out_cells", 13))

    def test_each_region_gets_its_number_of_placements(self):
        placements = planner.paired_placements("T2", with_regions(*self.HALVES))
        self.assertEqual(len(placements), 26)
        self.assertEqual(len(set(placements)), 26)
        self.assertEqual(sum(GRID.in_region(point, "trained_cells") for point in placements), 13)
        self.assertEqual(sum(GRID.in_region(point, "held_out_cells") for point in placements), 13)

    def test_the_order_says_nothing_about_the_region(self):
        placements = planner.paired_placements("T2", with_regions(*self.HALVES))
        trained = [GRID.in_region(point, "trained_cells") for point in placements]
        self.assertNotEqual(trained, sorted(trained))
        self.assertNotEqual(trained, sorted(trained, reverse=True))
        self.assertNotEqual(trained[0::2], trained[1::2])  # nor do the regions simply alternate
        runs = 1 + sum(a != b for a, b in zip(trained, trained[1:]))
        self.assertGreater(runs, 6)

    def test_the_placements_are_fixed_by_the_placement_seed(self):
        protocol = with_regions(*self.HALVES)
        self.assertEqual(planner.paired_placements("T2", protocol), planner.paired_placements("T2", protocol))
        self.assertNotEqual(planner.paired_placements("T2", protocol),
                            planner.paired_placements("T2", with_regions(*self.HALVES, placement_seed=1)))

    def test_one_region_given_as_a_list_draws_the_same_kind_of_placements_as_the_single_entry(self):
        placements = planner.paired_placements("T2", with_regions(("trained_cells", 20)))
        self.assertEqual(len(placements), 20)
        self.assertTrue(all(GRID.in_region(point, "trained_cells") for point in placements))

    def test_the_whole_block_runs_with_each_placement_tested_once_by_each_policy(self):
        protocol = with_regions(*self.HALVES)
        log = run("T2", protocol)
        self.assertEqual(len(log), 52)
        pairs = Counter(trial.planned.pair_id for trial in log)
        self.assertEqual(len(pairs), 26)
        self.assertEqual(set(pairs.values()), {2})
        first = Counter(trial.planned.condition for trial in log[0::2])
        self.assertEqual(first, {"act_clustered": 13, "act_spread": 13})
        self.assertIsNone(planner.next_trial("T2", protocol, AUDITS, log))
        self.assertEqual(planner.check_block("T2", protocol, AUDITS), [])

    def test_counts_that_do_not_add_up_to_the_pairs_are_refused(self):
        protocol = with_regions(*self.HALVES)
        protocol["blocks"]["T2"]["pairs"] = 20
        with self.assertRaisesRegex(PlannerError, "26.*20|20.*26"):
            planner.paired_placements("T2", protocol)
        self.assertNotEqual(planner.check_block("T2", protocol, AUDITS), [])

    def test_a_block_with_both_a_region_and_a_list_of_regions_is_refused(self):
        protocol = with_regions(*self.HALVES)
        protocol["blocks"]["T2"]["region"] = "trained_cells"
        with self.assertRaisesRegex(PlannerError, "region"):
            planner.paired_placements("T2", protocol)

    def test_an_unknown_region_an_empty_list_or_a_count_that_is_not_a_positive_whole_number_is_refused(self):
        for strata in ((("the_tray", 13), ("held_out_cells", 13)), ()):
            with self.subTest(strata=strata), self.assertRaises((PlannerError, geometry.GeometryError)):
                planner.paired_placements("T2", with_regions(*strata))
        for count in (0, -1, 2.5, True):
            protocol = with_regions(*self.HALVES)
            protocol["blocks"]["T2"]["regions"][0]["pairs"] = count
            with self.subTest(count=count), self.assertRaises(PlannerError):
                planner.paired_placements("T2", protocol)

    def test_a_region_named_twice_is_refused(self):
        with self.assertRaisesRegex(PlannerError, "twice"):
            planner.paired_placements("T2", with_regions(("trained_cells", 10), ("trained_cells", 10)))


class BlindCodeTests(unittest.TestCase):
    def test_every_trial_has_a_code_of_the_same_shape_whatever_its_block(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            for trial in finished(block):
                self.assertRegex(trial.planned.blind_code, r"^[A-HJ-NP-Z][2-9][2-9]$")

    def test_a_pilot_s_trials_are_the_same_trials_under_codes_of_their_own(self):
        # A threshold block is one condition, and a code is fixed by its block and its number: a pilot of a block
        # under the counted codes would tell, for those numbers, which code is a replay and which a policy, and
        # the counted session that interleaves the blocks shows the same codes again (the walk through the
        # runbooks, 2026-10-07: 30 of block 1's 56 codes known after the pilots the runbook asks for).
        counted_codes, pilot_codes = [], []
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log: list[TrialRecord] = []
            for _ in range(12):
                counted = planner.next_trial(block, PROTOCOL, AUDITS, log)
                pilot = planner.next_trial(block, PROTOCOL, AUDITS, log, pilot=True)
                # the same trial in everything but its code
                self.assertEqual(replace(pilot, blind_code="-"), replace(counted, blind_code="-"))
                self.assertRegex(pilot.blind_code, r"^P[A-HJ-NP-Z][2-9][2-9]$")
                self.assertNotEqual(pilot.blind_code[1:], counted.blind_code)
                counted_codes.append(counted.blind_code)
                pilot_codes.append(pilot.blind_code)
                log.append(record(counted, "success", AUDITS))
        self.assertEqual(len(set(pilot_codes)), len(pilot_codes), "two trials of a pilot share a code")
        self.assertFalse(set(pilot_codes) & set(counted_codes))
        # ... and the counted codes are what they were before there were pilot codes
        self.assertEqual(planner.next_trial("T1a", PROTOCOL, AUDITS, []).blind_code, "W57")
        self.assertEqual(planner.next_trial("T1a", PROTOCOL, AUDITS, [], pilot=True).blind_code, "PB89")

    def test_codes_are_unique_across_all_blocks_void_trials_included(self):
        logs = [run(block, demos=SPARSE, voids={1, 5, 6}) for block in (*THRESHOLD_BLOCKS, "T2")]
        codes = [r.planned.blind_code for log in logs for r in log]
        planned = sum(planned_total(block) for block in (*THRESHOLD_BLOCKS, "T2"))
        self.assertEqual(len(codes), planned + 4 * 3)  # three void trials in each of the four blocks
        self.assertEqual(len(set(codes)), len(codes))

    def test_a_code_names_neither_the_condition_nor_the_block(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            for trial in finished(block):
                code = trial.planned.blind_code.lower()
                for word in ("act", "clu", "spr", "rep", "t1", "t2", "t3"):
                    self.assertNotIn(word, code)

    def test_swapping_the_two_policies_swaps_every_condition_and_changes_no_code(self):
        # If a code carried the condition, it would have to change when the conditions change places.
        swapped = run("T2", protocol=with_block("T2", conditions=["act_spread", "act_clustered"]))
        for trial, mirror in zip(finished("T2"), swapped, strict=True):
            self.assertEqual(trial.planned.blind_code, mirror.planned.blind_code)
            self.assertEqual(trial.planned.target, mirror.planned.target)
            self.assertNotEqual(trial.planned.condition, mirror.planned.condition)

    def test_no_part_of_a_code_tells_the_two_policies_apart(self):
        log = finished("T2")
        for position in range(3):
            seen: dict[str, set[str]] = {}
            for trial in log:
                seen.setdefault(trial.planned.blind_code[position], set()).add(trial.planned.condition)
            both = sum(len(conditions) == 2 for conditions in seen.values())
            with self.subTest(position=position):
                self.assertGreater(both, 0)
        clustered = sorted(r.planned.blind_code for r in log if r.planned.condition == "act_clustered")
        spread = sorted(r.planned.blind_code for r in log if r.planned.condition == "act_spread")
        self.assertFalse(clustered[-1] < spread[0] or spread[-1] < clustered[0])  # nor does their alphabetical order

    def test_codes_do_not_run_in_order(self):
        codes = [trial.planned.blind_code for trial in finished("T1a")]
        self.assertNotEqual(codes, sorted(codes))
        self.assertGreater(len({code[0] for code in codes}), 8)

    def test_the_codes_of_a_block_cannot_be_told_from_those_of_another_by_their_letters(self):
        letters = {block: {r.planned.blind_code[0] for r in finished(block)} for block in (*THRESHOLD_BLOCKS, "T2")}
        for first in letters:
            for second in letters:
                self.assertTrue(letters[first] & letters[second])


class StatusTests(unittest.TestCase):
    def test_before_the_first_trial_nothing_is_done(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            total = planned_total(block)
            status = planner.block_status(block, PROTOCOL, [])
            with self.subTest(block=block):
                self.assertEqual(status["block"], block)
                self.assertFalse(status["complete"])
                self.assertEqual(status["trials_planned"], total)
                self.assertEqual(status["trials_scored"], 0)
                self.assertEqual(status["trials_void"], 0)
                self.assertEqual(status["trials_remaining"], total)

    def test_the_parts_of_a_threshold_block_are_counted(self):
        status = planner.block_status("T1a", PROTOCOL, [])
        self.assertEqual(status["anchors_zero"], {"done": 0, "planned": 6})
        self.assertEqual(status["anchors_far"], {"done": 0, "planned": 4})
        not_started = {"done": 0, "planned": 12}
        self.assertEqual(status["staircases"], {"T1a-up": not_started, "T1a-down": not_started})
        self.assertEqual(planner.block_status("T3", PROTOCOL, [])["anchors_far"], {"done": 0, "planned": 0})

    def test_midway_the_counts_are_those_of_the_log(self):
        log = run("T1a", voids={2, 6})[:20]
        status = planner.block_status("T1a", PROTOCOL, log)
        scored = [r for r in log if r.scored]
        parts = Counter(kind(r.planned) for r in scored)
        self.assertEqual(status["trials_scored"], 18)
        self.assertEqual(status["trials_void"], 2)
        self.assertEqual(status["trials_remaining"], 34 - 18)
        self.assertEqual(status["anchors_zero"]["done"], parts["zero"])
        self.assertEqual(status["anchors_far"]["done"], parts["far"])
        self.assertEqual(status["staircases"]["T1a-up"]["done"], parts["T1a-up"])
        self.assertEqual(status["staircases"]["T1a-down"]["done"], parts["T1a-down"])
        self.assertFalse(status["complete"])

    def test_pairs_are_counted_in_a_paired_block(self):
        self.assertEqual(planner.block_status("T2", PROTOCOL, [])["pairs"], {"done": 0, "planned": PAIRS})
        status = planner.block_status("T2", PROTOCOL, finished("T2")[:7])
        self.assertEqual(status["pairs"], {"done": 3, "planned": PAIRS})
        self.assertEqual(status["trials_scored"], 7)
        self.assertEqual(status["trials_remaining"], 2 * PAIRS - 7)

    def test_a_block_is_complete_exactly_when_no_trial_is_left(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            log = finished(block)
            for count in (0, 1, len(log) // 2, len(log) - 1, len(log)):
                status = planner.block_status(block, PROTOCOL, log[:count])
                with self.subTest(block=block, count=count):
                    nothing_left = planner.next_trial(block, PROTOCOL, AUDITS, log[:count]) is None
                    self.assertEqual(status["complete"], nothing_left)
                    self.assertEqual(status["trials_remaining"], len(log) - count)

    def test_trials_of_other_blocks_are_not_counted(self):
        status = planner.block_status("T3", PROTOCOL, [*finished("T1a"), *finished("T3")[:5]])
        self.assertEqual(status["trials_scored"], 5)

    def test_the_status_does_not_give_away_what_comes_next(self):
        # It may be shown to whoever places and scores: no distance, no target, no condition, no order.
        midway = planner.block_status("T1a", PROTOCOL, finished("T1a")[:13])
        text = repr(midway) + repr(planner.block_status("T2", PROTOCOL, finished("T2")[:5]))
        for word in ("level", "target", "condition", "act_", "next", "x_mm"):
            self.assertNotIn(word, text)


class RefusalTests(unittest.TestCase):
    def test_an_unknown_block_is_refused(self):
        with self.assertRaises(PlannerError):
            planner.next_trial("T9", PROTOCOL, AUDITS, [])
        with self.assertRaises(PlannerError):
            planner.block_status("T9", PROTOCOL, [])

    def test_a_threshold_block_needs_the_audit_of_its_own_set(self):
        with self.assertRaisesRegex(PlannerError, "spread"):
            planner.next_trial("T3", PROTOCOL, audit("clustered"), [])
        with self.assertRaisesRegex(PlannerError, "clustered"):
            planner.next_trial("T1b", PROTOCOL, [], [])

    def test_a_log_with_a_staircase_the_block_does_not_have_is_refused(self):
        stray = replace(finished("T1a")[1].planned, staircase_id="T1a-middle", staircase_index=0, level_mm=20.0)
        with self.assertRaisesRegex(PlannerError, "T1a-middle"):
            planner.next_trial("T1a", PROTOCOL, AUDITS, [record(stray, "success")])
        with self.assertRaisesRegex(PlannerError, "T1a-middle"):
            planner.block_status("T1a", PROTOCOL, [record(stray, "success")])

    def test_a_log_with_a_trial_of_another_condition_is_refused(self):
        stray = replace(finished("T1a")[0].planned, condition="act_spread")
        with self.assertRaisesRegex(PlannerError, "act_spread"):
            planner.next_trial("T1a", PROTOCOL, AUDITS, [record(stray, "success")])

    def test_a_log_with_a_pair_the_block_does_not_have_is_refused(self):
        stray = replace(finished("T2")[0].planned, pair_id="T2-p99")
        with self.assertRaisesRegex(PlannerError, "T2-p99"):
            planner.next_trial("T2", PROTOCOL, AUDITS, [record(stray, "success")])

    def test_a_paired_block_compares_two_different_policies(self):
        one = ["act_clustered"]
        for conditions in ([], one, one * 2, [*one, "replay"], list(schema.CONDITIONS)):
            with self.subTest(conditions=conditions), self.assertRaises(PlannerError):
                planner.next_trial("T2", with_block("T2", conditions=conditions), AUDITS, [])

    def test_an_entry_the_block_needs_and_does_not_have_is_named(self):
        for block, key in (("T1a", "far_level_mm"), ("T1a", "max_level_mm"), ("T3", "condition"), ("T2", "conditions")):
            protocol = copy.deepcopy(PROTOCOL)
            del protocol["blocks"][block][key]
            with self.subTest(block=block, key=key), self.assertRaisesRegex(PlannerError, key):
                run(block, protocol=protocol, demos=SPARSE)
        protocol = copy.deepcopy(PROTOCOL)
        del protocol["placement"]["level_match_mm"]
        with self.assertRaisesRegex(PlannerError, "level_match_mm"):
            run("T3", protocol=protocol, demos=SPARSE)

    def test_a_block_that_is_neither_a_threshold_nor_a_paired_block_is_refused(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["blocks"]["T3"] = {"title": "nothing to run", "condition": "act_spread"}
        with self.assertRaises(PlannerError):
            planner.next_trial("T3", protocol, AUDITS, [])


class SeedTests(unittest.TestCase):
    def test_a_block_uses_its_own_seed_then_its_placement_seed_then_the_layout_seed(self):
        self.assertEqual(planner.block_seed("T2", PROTOCOL), 20261007)
        self.assertEqual(planner.block_seed("T1a", PROTOCOL), PROTOCOL["datasets"]["layout_seed"])
        self.assertEqual(planner.block_seed("T1a", with_block("T1a", seed=5)), 5)
        self.assertEqual(planner.block_seed("T2", with_block("T2", seed=5)), 5)

    def test_two_blocks_sharing_a_seed_do_not_share_their_draws(self):
        logs = {
            block: run(block, with_block(block, seed=3, anchors_zero=6), SPARSE, outcome=always("failure"))
            for block in ("T1a", "T3")
        }
        self.assertNotEqual(targets_of(logs["T1a"], "zero"), targets_of(logs["T3"], "zero"))
        drawn = {block: {r.planned.target for r in log if kind(r.planned) != "zero"} for block, log in logs.items()}
        self.assertFalse(drawn["T1a"] & drawn["T3"])


class CheckBlockTests(unittest.TestCase):
    def test_the_blocks_of_the_protocol_can_run_on_the_planned_layouts(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            self.assertEqual(planner.check_block(block, PROTOCOL, AUDITS), [])

    def test_a_staircase_limit_beyond_what_the_table_offers_is_reported_before_the_block_starts(self):
        largest = targets.max_feasible_level(audit("spread"), GRID)
        problems = planner.check_block("T3", with_block("T3", max_level_mm=80.0), AUDITS)
        self.assertEqual(len(problems), 1)
        self.assertIn("80", problems[0])
        self.assertIn(f"{largest:.1f}", problems[0])
        self.assertEqual(planner.check_block("T3", with_block("T3", max_level_mm=round(largest) - 1.0), AUDITS), [])
        # A limit past the largest distance by less than the match tolerance can still be placed.
        self.assertEqual(planner.check_block("T3", with_block("T3", max_level_mm=largest + 0.3), AUDITS), [])
        self.assertNotEqual(planner.check_block("T3", with_block("T3", max_level_mm=largest + 0.6), AUDITS), [])

    def test_a_far_level_beyond_what_the_table_offers_is_reported(self):
        problems = planner.check_block("T1a", with_block("T1a", far_level_mm=90.0), AUDITS)
        self.assertEqual(len(problems), 1)
        self.assertIn("90", problems[0])

    def test_a_replay_distance_that_does_not_fit_the_table_is_reported(self):
        problems = planner.check_block("T1b", with_block("T1b", max_level_mm=400.0), AUDITS)
        self.assertTrue(problems)
        self.assertTrue(all("400" in problem for problem in problems))

    def test_a_missing_audit_is_reported_not_raised(self):
        problems = planner.check_block("T3", PROTOCOL, audit("clustered"))
        self.assertEqual(len(problems), 1)
        self.assertIn("spread", problems[0])


class WholeExperimentTests(unittest.TestCase):
    """The way the rig uses the planner: everything read back from its file before every trial."""

    def test_the_four_blocks_run_from_the_files_on_disk_into_one_log(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for dataset in schema.DATASETS:
                schema.write_demo_starts(folder / f"{dataset}.json", audit(dataset))
            log_path = folder / "trials" / "all.jsonl"
            for block in ("T1b", "T1a", "T2", "T3"):  # the protocol's order of priority
                demos = [d for dataset in schema.DATASETS for d in schema.read_demo_starts(folder / f"{dataset}.json")]
                self.assertEqual(planner.check_block(block, PROTOCOL, demos), [])
                while True:
                    log = schema.read_trials(log_path)
                    planned = planner.next_trial(block, PROTOCOL, demos, log)
                    if planned is None:
                        break
                    outcome = "void" if len(log) % 17 == 5 else by_distance(planned, len(log))
                    schema.append_trial(log_path, record(planned, outcome, demos))
                    self.assertLess(len(log), 400)
                self.assertTrue(planner.block_status(block, PROTOCOL, schema.read_trials(log_path))["complete"])
            log = schema.read_trials(log_path)
        scored = [r for r in log if r.scored]
        self.assertEqual(Counter(r.planned.block for r in scored),
                         {block: planned_total(block) for block in ("T1b", "T1a", "T2", "T3")})
        self.assertEqual(len(log) - len(scored), sum(index % 17 == 5 for index in range(len(log))))
        self.assertEqual(len({r.planned.trial_id for r in log}), len(log))
        self.assertEqual(len({r.planned.blind_code for r in log}), len(log))
        # The same four blocks, each planned on its own from its own trials, are the same trials.
        for block in ("T1b", "T1a", "T2", "T3"):
            own = [r for r in log if r.planned.block == block]
            for count in (0, len(own) // 2, len(own)):
                expected = own[count].planned if count < len(own) else None
                self.assertEqual(planner.next_trial(block, PROTOCOL, AUDITS, own[:count]), expected)


class ReadabilityTests(unittest.TestCase):
    def test_ids_read_as_block_and_number(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            for trial in finished(block):
                self.assertTrue(re.fullmatch(rf"{block}-\d{{3}}", trial.planned.trial_id))

    def test_staircase_ids_carry_their_block(self):
        self.assertEqual({r.planned.staircase_id for r in finished("T3")} - {None}, {"T3-up", "T3-down"})

    def test_a_planned_trial_is_a_record_the_schema_accepts(self):
        for block in (*THRESHOLD_BLOCKS, "T2"):
            for trial in finished(block):
                self.assertEqual(schema.planned_trial_from_json(schema.to_json(trial.planned)), trial.planned)
                self.assertIsInstance(trial.planned.target, Point)


if __name__ == "__main__":
    unittest.main()
