"""The procedure simulation: a simulated block follows the protocol's rule, and the numbers reported about it hold."""

import contextlib
import csv
import io
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from tolerance_radius import config, simulate
from tolerance_radius.intervals import mcnemar_exact
from tolerance_radius.psychometric import ThresholdSummary, grids_from_protocol, success_probability
from tolerance_radius.simulate import (
    BlockDesign,
    StaircasePlan,
    StaircaseSteps,
    Truth,
    blocks_from_protocol,
    constant_stimuli,
    design_variants,
    evaluate_design,
    paired_power,
    run_sweep,
    simulate_block,
    steps_from_protocol,
    with_staircase_trials,
)
from tolerance_radius.staircase import StaircaseParams, next_level

PROTOCOL = config.load_protocol()
GRIDS = grids_from_protocol(PROTOCOL["analysis"])._replace(floor_values=np.array([0.0]))  # no floor in the model
FLOOR_GRIDS = GRIDS._replace(floor_values=np.array([0.0, 0.05, 0.10]))

# A block and steps written out here, so that these tests do not move when the protocol file does.
STEPS = StaircaseSteps(step_initial_mm=10.0, step_final_mm=5.0)
BLOCK = BlockDesign(
    name="demo",
    anchors_zero=6,
    staircases=(StaircasePlan("up", 10.0, 12), StaircasePlan("down", 40.0, 12)),
    min_level_mm=0.0,
    max_level_mm=50.0,
    anchors_far=4,
    far_level_mm=50.0,
)
EASY = Truth(r50_mm=25.0, width_mm=1.0, lapse=0.0)  # a cliff in the middle of the range


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def made_up_block(truth: Truth, median: float, low: float, high: float, lower: bool, upper: bool):
    """A block whose estimate is written by hand, to check how estimates are tallied."""
    summary = ThresholdSummary(
        level=0.95,
        n_trials=34,
        n_success=17,
        r50_median_mm=median,
        r50_low_mm=low,
        r50_high_mm=high,
        reaches_lower_edge=lower,
        reaches_upper_edge=upper,
        width_median_mm=3.0,
        width_low_mm=1.0,
        width_high_mm=9.0,
        lapse_mean=0.03,
    )
    nothing = np.array([])
    return simulate.SimulatedBlock(truth, summary, (), nothing, nothing, nothing.astype(bool))


def expected_designs_over_blocks(protocol: dict) -> int:
    """How many (block, design) combinations a sweep of this protocol runs: five per block, and a sixth for each
    block whose largest level is below the largest of all."""
    blocks = blocks_from_protocol(protocol)
    widest = max(block.max_level_mm for block in blocks)
    return sum(len(simulate.DESIGNS) + (block.max_level_mm < widest) for block in blocks)


def same_block(first: simulate.SimulatedBlock, second: simulate.SimulatedBlock) -> bool:
    return (
        first.kinds == second.kinds
        and np.array_equal(first.levels_mm, second.levels_mm)
        and np.array_equal(first.distances_mm, second.distances_mm)
        and np.array_equal(first.successes, second.successes)
        and first.summary == second.summary
    )


class BlockDesignTests(unittest.TestCase):
    def test_a_block_counts_all_its_trials(self):
        self.assertEqual(BLOCK.n_trials, 6 + 4 + 12 + 12)
        fixed = BlockDesign("fixed", 0, (), 0.0, 40.0, fixed_levels_mm=(0.0, 20.0, 40.0))
        self.assertEqual(fixed.n_trials, 3)

    def test_constant_stimuli_spread_the_same_number_of_trials_evenly_over_the_range(self):
        constant = constant_stimuli(BLOCK)
        self.assertEqual(constant.n_trials, BLOCK.n_trials)
        self.assertEqual((constant.anchors_zero, constant.anchors_far, constant.staircases), (0, 0, ()))
        levels = np.array(constant.fixed_levels_mm)
        self.assertEqual((levels[0], levels[-1]), (BLOCK.min_level_mm, BLOCK.max_level_mm))
        self.assertTrue(np.allclose(np.diff(levels), 50.0 / 33, rtol=1e-12))
        self.assertEqual((constant.name, constant.min_level_mm, constant.max_level_mm), ("demo", 0.0, 50.0))

    def test_staircases_can_be_made_longer_or_shorter(self):
        longer, shorter = with_staircase_trials(BLOCK, 4), with_staircase_trials(BLOCK, -4)
        self.assertEqual([plan.n_trials for plan in longer.staircases], [16, 16])
        self.assertEqual([plan.n_trials for plan in shorter.staircases], [8, 8])
        self.assertEqual((longer.n_trials, shorter.n_trials), (BLOCK.n_trials + 8, BLOCK.n_trials - 8))
        self.assertEqual(longer.anchors_zero, BLOCK.anchors_zero)
        with self.assertRaises(ValueError):
            with_staircase_trials(BLOCK, -12)

    def test_impossible_blocks_and_truths_are_refused(self):
        for change in (
            {"anchors_zero": -1},
            {"anchors_far": 2, "far_level_mm": None},  # far anchors need their distance
            {"max_level_mm": -5.0},
            {"staircases": (StaircasePlan("up", 60.0, 5),)},  # starts beyond the block's largest distance
            {"staircases": (StaircasePlan("up", 10.0, 5), StaircasePlan("up", 20.0, 5))},  # two with one name
            {"staircases": (StaircasePlan("zero", 10.0, 5),)},  # the name of an anchor
            {"fixed_levels_mm": (10.0, -1.0)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(BLOCK, **change)
        for values in ((20.0, 0.0, 0.0), (20.0, 3.0, 1.0), (20.0, 3.0, -0.1), (float("nan"), 3.0, 0.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Truth(*values)
        for lapse, floor in ((0.0, -0.01), (0.0, 1.0), (0.6, 0.4)):
            with self.subTest(lapse=lapse, floor=floor), self.assertRaises(ValueError):
                Truth(20.0, 3.0, lapse, floor)
        self.assertEqual(Truth(20.0, 3.0).floor, 0.0)
        with self.assertRaises(ValueError):
            StaircaseSteps(step_initial_mm=10.0, step_final_mm=0.0)


class ProtocolBlocksTests(unittest.TestCase):
    def test_every_block_with_staircases_is_read_as_the_file_states_it(self):
        blocks = blocks_from_protocol(PROTOCOL)
        with_staircases = {name: block for name, block in PROTOCOL["blocks"].items() if "staircases" in block}
        self.assertEqual([block.name for block in blocks], list(with_staircases))
        self.assertGreaterEqual(len(blocks), 1)
        for block in blocks:
            stated = with_staircases[block.name]
            with self.subTest(block=block.name):
                self.assertEqual(block.anchors_zero, stated.get("anchors_zero", 0))
                self.assertEqual(block.anchors_far, stated.get("anchors_far", 0))
                self.assertEqual(block.far_level_mm, stated.get("far_level_mm"))
                self.assertEqual(block.max_level_mm, stated["max_level_mm"])
                self.assertEqual(block.min_level_mm, PROTOCOL["staircase"]["min_level_mm"])
                self.assertEqual(
                    [(plan.id, plan.start_level_mm, plan.n_trials) for plan in block.staircases],
                    [(item["id"], item["start_level_mm"], item["trials"]) for item in stated["staircases"]],
                )
                planned = stated.get("anchors_zero", 0) + stated.get("anchors_far", 0)
                self.assertEqual(block.n_trials, planned + sum(item["trials"] for item in stated["staircases"]))

    def test_the_steps_are_the_file_s(self):
        steps = steps_from_protocol(PROTOCOL)
        self.assertEqual(steps.step_initial_mm, PROTOCOL["staircase"]["step_initial_mm"])
        self.assertEqual(steps.step_final_mm, PROTOCOL["staircase"]["step_final_mm"])

    def test_a_rule_other_than_one_up_one_down_is_refused(self):
        other = dict(PROTOCOL, staircase=dict(PROTOCOL["staircase"], rule="2up1down"))
        with self.assertRaises(ValueError):
            steps_from_protocol(other)

    def test_a_block_that_stops_short_of_the_widest_also_gets_a_wider_range(self):
        # The comparison's five designs keep the block's own limits. Where another block of the protocol goes farther
        # out, a sixth design asks what the same block would give with that limit.
        narrow = replace(BLOCK, max_level_mm=40.0, anchors_far=0, far_level_mm=None)
        variants = design_variants(narrow, STEPS, widest_level_mm=50.0)
        self.assertEqual(tuple(variants), (*simulate.DESIGNS, simulate.WIDER_RANGE))
        self.assertEqual(variants[simulate.WIDER_RANGE], (replace(narrow, max_level_mm=50.0), STEPS))
        self.assertEqual(tuple(design_variants(BLOCK, STEPS, widest_level_mm=50.0)), simulate.DESIGNS)

    def test_the_variants_are_the_five_designs_of_the_comparison(self):
        variants = design_variants(BLOCK, STEPS)
        self.assertEqual(tuple(variants), simulate.DESIGNS)
        self.assertEqual(variants["protocol"], (BLOCK, STEPS))
        self.assertEqual(variants["constant"], (constant_stimuli(BLOCK), STEPS))
        self.assertEqual(variants["fine_step"], (BLOCK, StaircaseSteps(step_initial_mm=10.0, step_final_mm=2.5)))
        self.assertEqual(variants["longer"], (with_staircase_trials(BLOCK, 4), STEPS))
        self.assertEqual(variants["shorter"], (with_staircase_trials(BLOCK, -4), STEPS))
        # The first three are compared at an equal number of trials.
        self.assertEqual({variants[name][0].n_trials for name in ("protocol", "constant", "fine_step")}, {34})


class SimulateBlockTests(unittest.TestCase):
    def test_the_same_seed_gives_the_same_block_and_another_seed_another(self):
        first = simulate_block(EASY, BLOCK, STEPS, rng(11), grids=GRIDS)
        again = simulate_block(EASY, BLOCK, STEPS, rng(11), grids=GRIDS)
        other = simulate_block(EASY, BLOCK, STEPS, rng(12), grids=GRIDS)
        self.assertTrue(same_block(first, again))
        self.assertFalse(same_block(first, other))

    def test_a_block_has_exactly_the_planned_number_of_trials(self):
        steps = steps_from_protocol(PROTOCOL)
        for block in [BLOCK, *blocks_from_protocol(PROTOCOL)]:
            for design, (variant, variant_steps) in design_variants(block, steps).items():
                with self.subTest(block=block.name, design=design):
                    result = simulate_block(Truth(15.0, 3.0, 0.05), variant, variant_steps, rng(5), grids=GRIDS)
                    self.assertEqual(len(result.kinds), variant.n_trials)
                    self.assertEqual(result.levels_mm.shape, (variant.n_trials,))
                    self.assertEqual(result.distances_mm.shape, (variant.n_trials,))
                    self.assertEqual(result.successes.shape, (variant.n_trials,))
                    self.assertEqual(result.summary.n_trials, variant.n_trials)
                    self.assertEqual(result.summary.n_success, int(result.successes.sum()))

    def test_anchors_are_requested_where_the_block_says(self):
        result = simulate_block(EASY, BLOCK, STEPS, rng(3), grids=GRIDS)
        kinds = np.array(result.kinds)
        counts = {kind: int((kinds == kind).sum()) for kind in set(result.kinds)}
        self.assertEqual(counts, {"zero": 6, "far": 4, "up": 12, "down": 12})
        self.assertTrue(np.all(result.levels_mm[kinds == "zero"] == 0.0))
        self.assertTrue(np.all(result.levels_mm[kinds == "far"] == 50.0))
        self.assertEqual(result.truth, EASY)

    def test_each_staircase_follows_the_protocol_s_rule_from_its_own_outcomes(self):
        for seed, truth in enumerate((EASY, Truth(5.0, 3.0, 0.05), Truth(40.0, 10.0, 0.0), Truth(15.0, 6.0, 0.05))):
            for steps in (STEPS, StaircaseSteps(step_initial_mm=10.0, step_final_mm=2.5)):
                result = simulate_block(truth, BLOCK, steps, rng(seed), grids=GRIDS)
                kinds = np.array(result.kinds)
                for plan in BLOCK.staircases:
                    with self.subTest(truth=truth, final_step=steps.step_final_mm, staircase=plan.id):
                        params = StaircaseParams(
                            start_level_mm=plan.start_level_mm,
                            n_trials=plan.n_trials,
                            step_initial_mm=steps.step_initial_mm,
                            step_final_mm=steps.step_final_mm,
                            min_level_mm=BLOCK.min_level_mm,
                            max_level_mm=BLOCK.max_level_mm,
                        )
                        history: list[tuple[float, bool]] = []
                        pairs = zip(result.levels_mm[kinds == plan.id], result.successes[kinds == plan.id], strict=True)
                        for level, success in pairs:
                            self.assertEqual(next_level(params, history), level)
                            history.append((float(level), bool(success)))
                        self.assertIsNone(next_level(params, history))

    def test_the_two_staircases_take_turns(self):
        result = simulate_block(EASY, BLOCK, STEPS, rng(3), grids=GRIDS)
        self.assertEqual([kind for kind in result.kinds if kind in ("up", "down")], ["up", "down"] * 12)

    def test_without_placement_noise_the_cube_is_where_it_was_asked(self):
        result = simulate_block(EASY, BLOCK, STEPS, rng(7), grids=GRIDS, placement_sd_mm=0.0)
        self.assertTrue(np.array_equal(result.distances_mm, result.levels_mm))

    def test_placement_noise_has_the_stated_spread_and_never_gives_a_negative_distance(self):
        many = BlockDesign("many", 400, (), 0.0, 50.0, fixed_levels_mm=(30.0,) * 4000)
        result = simulate_block(Truth(20.0, 3.0), many, STEPS, rng(8), grids=GRIDS, placement_sd_mm=1.5)
        kinds = np.array(result.kinds)
        error = result.distances_mm[kinds == "fixed"] - 30.0
        self.assertAlmostEqual(float(error.std()), 1.5, delta=0.1)
        self.assertAlmostEqual(float(error.mean()), 0.0, delta=0.1)
        at_zero = result.distances_mm[kinds == "zero"]
        self.assertTrue(np.all(at_zero >= 0.0))
        self.assertTrue(0.3 < float((at_zero == 0.0).mean()) < 0.7)  # about half the errors point inward

    def test_outcomes_are_drawn_from_the_model(self):
        # Many trials at four distances, placed exactly: the share of successes at each must match the model, plateau
        # included (at 0 mm the model gives 0.90 with these lapses, and 0.999 without).
        truth = Truth(r50_mm=20.0, width_mm=3.0, lapse=0.10)
        per_level = 2000
        levels = tuple(level for level in (0.0, 15.0, 20.0, 25.0) for _ in range(per_level))
        block = BlockDesign("check", 0, (), 0.0, 50.0, fixed_levels_mm=levels)
        result = simulate_block(truth, block, STEPS, rng(9), grids=GRIDS, placement_sd_mm=0.0)
        for level in (0.0, 15.0, 20.0, 25.0):
            with self.subTest(level=level):
                expected = float(success_probability(level, truth.r50_mm, truth.width_mm, truth.lapse))
                observed = float(result.successes[result.levels_mm == level].mean())
                four_standard_errors = 4.0 * math.sqrt(expected * (1.0 - expected) / per_level)
                self.assertAlmostEqual(observed, expected, delta=four_standard_errors)
        self.assertAlmostEqual(result.summary.r50_median_mm, 20.0, delta=1.0)

    def test_the_outcome_depends_on_where_the_cube_landed_not_on_where_it_was_asked(self):
        # A cliff at 20 mm, every trial requested at 20 mm, placements scattered by 2 mm: a trial succeeds when it
        # landed short of the cliff and fails when it landed beyond, whatever was requested.
        cliff = Truth(r50_mm=20.0, width_mm=0.01, lapse=0.0)
        block = BlockDesign("scatter", 0, (), 0.0, 50.0, fixed_levels_mm=(20.0,) * 400)
        result = simulate_block(cliff, block, STEPS, rng(10), grids=GRIDS, placement_sd_mm=2.0)
        clear = np.abs(result.distances_mm - 20.0) > 0.2  # twenty widths from the cliff: the outcome is certain
        self.assertGreater(int(clear.sum()), 300)
        self.assertTrue(np.array_equal(result.successes[clear], result.distances_mm[clear] < 20.0))
        self.assertTrue(0.3 < float(result.successes.mean()) < 0.7)

    def test_the_level_of_the_interval_is_passed_on(self):
        result = simulate_block(EASY, BLOCK, STEPS, rng(4), grids=GRIDS, level=0.5)
        self.assertEqual(result.summary.level, 0.5)

    def test_a_truth_with_a_floor_gives_lucky_successes_far_out(self):
        # A cliff at 20 mm, every trial placed exactly at 60 mm: without a floor none succeeds, with a floor of 0.08
        # about one in twelve does.
        far_out = BlockDesign("far", 0, (), 0.0, 80.0, fixed_levels_mm=(60.0,) * 3000)
        without = simulate_block(Truth(20.0, 1.0, 0.05), far_out, STEPS, rng(21), grids=GRIDS, placement_sd_mm=0.0)
        with_floor = simulate_block(
            Truth(20.0, 1.0, 0.05, floor=0.08), far_out, STEPS, rng(21), grids=GRIDS, placement_sd_mm=0.0
        )
        self.assertEqual(int(without.successes.sum()), 0)
        four_standard_errors = 4.0 * math.sqrt(0.08 * 0.92 / 3000)
        self.assertAlmostEqual(float(with_floor.successes.mean()), 0.08, delta=four_standard_errors)

    def test_the_fit_uses_the_floor_values_of_the_grids_it_is_given(self):
        truth = Truth(20.0, 3.0, 0.05, floor=0.08)
        plain = simulate_block(truth, BLOCK, STEPS, rng(22), grids=GRIDS)
        with_floors = simulate_block(truth, BLOCK, STEPS, rng(22), grids=FLOOR_GRIDS)
        # The same trials (same seed): only the analysis differs.
        self.assertTrue(np.array_equal(plain.distances_mm, with_floors.distances_mm))
        self.assertTrue(np.array_equal(plain.successes, with_floors.successes))
        self.assertEqual(plain.summary.floor_mean, 0.0)
        self.assertTrue(0.0 < with_floors.summary.floor_mean < 0.10)


class EvaluateDesignTests(unittest.TestCase):
    def test_a_cliff_in_the_middle_of_the_range_is_recovered(self):
        # Stated tolerance for an easy truth (r50 = 25 mm, width 1 mm, no lapses) and 34 trials, over 80 blocks: the
        # median is off by less than 0.5 mm on average and by less than 1.5 mm in root-mean-square, the interval is
        # under 6 mm wide on average, at least 90 % of the intervals contain the truth, and none reaches an end of
        # the grid.
        performance = evaluate_design(EASY, BLOCK, STEPS, rng(2026), grids=GRIDS, replications=80)
        self.assertEqual((performance.truth, performance.block, performance.n_trials), (EASY, "demo", 34))
        self.assertEqual(performance.replications, 80)
        self.assertLess(abs(performance.bias_mm), 0.5)
        self.assertLess(performance.rmse_mm, 1.5)
        self.assertLess(performance.mean_interval_width_mm, 6.0)
        self.assertGreaterEqual(performance.interval_coverage, 0.90)
        self.assertEqual(performance.any_edge_rate, 0.0)

    def test_the_same_seed_gives_the_same_numbers(self):
        first = evaluate_design(EASY, BLOCK, STEPS, rng(1), grids=GRIDS, replications=5)
        again = evaluate_design(EASY, BLOCK, STEPS, rng(1), grids=GRIDS, replications=5)
        other = evaluate_design(EASY, BLOCK, STEPS, rng(2), grids=GRIDS, replications=5)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_the_tally_counts_a_truth_on_a_bound_as_inside_and_an_edge_on_either_side(self):
        truth = Truth(20.0, 3.0)
        blocks = [
            made_up_block(truth, median=20.0, low=20.0, high=24.0, lower=False, upper=False),  # truth on the low bound
            made_up_block(truth, median=22.0, low=16.0, high=20.0, lower=True, upper=False),  # truth on the high bound
            made_up_block(truth, median=17.0, low=10.0, high=19.0, lower=False, upper=True),  # truth above
            made_up_block(truth, median=25.0, low=21.0, high=30.0, lower=False, upper=False),  # truth below
        ]
        with mock.patch.object(simulate, "simulate_block", side_effect=blocks):
            performance = evaluate_design(truth, BLOCK, STEPS, rng(0), grids=GRIDS, replications=4)
        self.assertAlmostEqual(performance.bias_mm, (0.0 + 2.0 - 3.0 + 5.0) / 4, places=12)
        self.assertAlmostEqual(performance.rmse_mm, math.sqrt((0.0 + 4.0 + 9.0 + 25.0) / 4), places=12)
        self.assertAlmostEqual(performance.mean_interval_width_mm, (4.0 + 4.0 + 9.0 + 9.0) / 4, places=12)
        self.assertEqual(performance.interval_coverage, 0.5)
        edges = (performance.lower_edge_rate, performance.upper_edge_rate, performance.any_edge_rate)
        self.assertEqual(edges, (0.25, 0.25, 0.5))

    def test_the_numbers_are_those_of_the_blocks_it_ran(self):
        for options in ({}, {"placement_sd_mm": 0.0, "level": 0.5}):
            with self.subTest(options=options):
                self.check_against_blocks(options)

    def check_against_blocks(self, options: dict):
        generator, twin = rng(33), rng(33)
        truth = Truth(15.0, 6.0, 0.05)
        performance = evaluate_design(truth, BLOCK, STEPS, generator, grids=GRIDS, replications=6, **options)
        blocks = [simulate_block(truth, BLOCK, STEPS, twin, grids=GRIDS, **options) for _ in range(6)]
        summaries = [block.summary for block in blocks]
        if options:
            self.assertTrue(all(np.array_equal(block.distances_mm, block.levels_mm) for block in blocks))
            self.assertTrue(all(summary.level == 0.5 for summary in summaries))
        errors = np.array([summary.r50_median_mm - 15.0 for summary in summaries])
        self.assertAlmostEqual(performance.bias_mm, float(errors.mean()), places=12)
        self.assertAlmostEqual(performance.rmse_mm, float(np.sqrt((errors**2).mean())), places=12)
        widths = [summary.r50_high_mm - summary.r50_low_mm for summary in summaries]
        self.assertAlmostEqual(performance.mean_interval_width_mm, float(np.mean(widths)), places=12)
        inside = [summary.r50_low_mm <= 15.0 <= summary.r50_high_mm for summary in summaries]
        self.assertAlmostEqual(performance.interval_coverage, float(np.mean(inside)), places=12)
        lower = [summary.reaches_lower_edge for summary in summaries]
        upper = [summary.reaches_upper_edge for summary in summaries]
        self.assertAlmostEqual(performance.lower_edge_rate, float(np.mean(lower)), places=12)
        self.assertAlmostEqual(performance.upper_edge_rate, float(np.mean(upper)), places=12)
        either = [low or high for low, high in zip(lower, upper, strict=True)]
        self.assertAlmostEqual(performance.any_edge_rate, float(np.mean(either)), places=12)

    def test_a_radius_beyond_the_largest_distance_tested_is_reported_as_unbounded(self):
        # Success out to 70 mm, tested out to 50 mm: nearly every trial succeeds and the data cannot say where
        # success ends. The intervals must say so.
        performance = evaluate_design(Truth(70.0, 1.0), BLOCK, STEPS, rng(6), grids=GRIDS, replications=20)
        self.assertGreaterEqual(performance.upper_edge_rate, 0.95)
        self.assertEqual(performance.lower_edge_rate, 0.0)

    def test_at_least_one_replication_is_needed(self):
        with self.assertRaises(ValueError):
            evaluate_design(EASY, BLOCK, STEPS, rng(1), grids=GRIDS, replications=0)


class EvaluateFitsTests(unittest.TestCase):
    """The same simulated blocks analysed in more than one way."""

    TRUTH = Truth(20.0, 3.0, 0.05, floor=0.08)

    def test_one_fit_alone_is_the_design_s_own_evaluation(self):
        alone = simulate.evaluate_fits(self.TRUTH, BLOCK, STEPS, rng(3), fits={"only": GRIDS}, replications=5)
        performance, floor = alone["only"]
        self.assertEqual(performance, evaluate_design(self.TRUTH, BLOCK, STEPS, rng(3), grids=GRIDS, replications=5))
        self.assertEqual(floor, 0.0)

    def test_every_fit_sees_the_same_blocks(self):
        fits = {"first": FLOOR_GRIDS, "plain": GRIDS, "second": FLOOR_GRIDS}
        results = simulate.evaluate_fits(self.TRUTH, BLOCK, STEPS, rng(4), fits=fits, replications=6)
        self.assertEqual(list(results), ["first", "plain", "second"])
        self.assertEqual(results["first"], results["second"])  # the same analysis of the same blocks
        self.assertNotEqual(results["first"][0], results["plain"][0])
        # The fit without a floor is what the same generator gives with that fit alone.
        alone = simulate.evaluate_fits(self.TRUTH, BLOCK, STEPS, rng(4), fits={"plain": GRIDS}, replications=6)
        self.assertEqual(results["plain"], alone["plain"])

    def test_the_mean_floor_is_the_mean_of_the_blocks_posterior_means(self):
        generator, twin = rng(5), rng(5)
        fits = {"floor": FLOOR_GRIDS}
        results = simulate.evaluate_fits(self.TRUTH, BLOCK, STEPS, generator, fits=fits, replications=4)
        blocks = [simulate_block(self.TRUTH, BLOCK, STEPS, twin, grids=FLOOR_GRIDS) for _ in range(4)]
        expected = float(np.mean([block.summary.floor_mean for block in blocks]))
        self.assertAlmostEqual(results["floor"][1], expected, places=12)
        self.assertTrue(0.0 < results["floor"][1] < 0.10)

    def test_at_least_one_fit_and_one_replication_are_needed(self):
        with self.assertRaises(ValueError):
            simulate.evaluate_fits(self.TRUTH, BLOCK, STEPS, rng(1), fits={}, replications=3)
        with self.assertRaises(ValueError):
            simulate.evaluate_fits(self.TRUTH, BLOCK, STEPS, rng(1), fits={"only": GRIDS}, replications=0)


class FloorCheckTests(unittest.TestCase):
    """What a floor in the truth does to a fit without one, and what a floor in the fit costs when there is none."""

    def test_the_blocks_are_one_with_far_anchors_and_one_without_which_also_gets_some(self):
        designs = simulate.floor_check_designs(PROTOCOL)
        blocks = blocks_from_protocol(PROTOCOL)
        with_far = [block for block in blocks if block.anchors_far > 0]
        without = [block for block in blocks if block.anchors_far == 0]
        self.assertTrue(with_far and without, "the repository protocol has blocks of both kinds")
        largest = max(without, key=lambda block: block.n_trials)
        self.assertEqual([(variant, block.name) for variant, block in designs][: len(with_far)],
                         [("as_written", block.name) for block in with_far])
        rest = designs[len(with_far):]
        self.assertEqual([variant for variant, _ in rest], ["as_written", "far_anchors", "longer_staircases"])
        as_written, far_anchors, longer = (block for _, block in rest)
        self.assertEqual(as_written, largest)
        self.assertEqual(far_anchors.anchors_far, simulate.FLOOR_CHECK_FAR_ANCHORS)
        self.assertEqual(far_anchors.far_level_mm, largest.max_level_mm)
        self.assertEqual(far_anchors.staircases, largest.staircases)
        self.assertEqual(far_anchors.n_trials, largest.n_trials + simulate.FLOOR_CHECK_FAR_ANCHORS)
        # The fair comparison for the far anchors: the same number of extra trials, spent on the staircases.
        self.assertEqual(longer.anchors_far, 0)
        self.assertEqual(longer.n_trials, far_anchors.n_trials)
        extra = [new.n_trials - old.n_trials for new, old in zip(longer.staircases, largest.staircases, strict=True)]
        self.assertEqual(sum(extra), simulate.FLOOR_CHECK_FAR_ANCHORS)
        self.assertLessEqual(max(extra) - min(extra), 1)

    def test_the_truths_have_a_floor_or_none_and_the_fits_are_with_and_without(self):
        truths = simulate.floor_check_truths()
        self.assertEqual({truth.floor for truth in truths}, {0.08, 0.0})
        floors = [truth for truth in truths if truth.floor > 0]
        plain = [truth for truth in truths if truth.floor == 0]
        self.assertEqual(
            [(truth.r50_mm, truth.width_mm, truth.lapse) for truth in floors],
            [(truth.r50_mm, truth.width_mm, truth.lapse) for truth in plain],
        )
        self.assertEqual(simulate.FLOOR_CHECK_FLOOR_VALUES, (0.0, 0.05, 0.10))

    def test_one_row_for_every_block_truth_and_fit(self):
        rows = simulate.run_floor_check(PROTOCOL, replications=2, seed=5)
        designs = simulate.floor_check_designs(PROTOCOL)
        truths = simulate.floor_check_truths()
        self.assertEqual(len(rows), len(designs) * len(truths) * 2)
        self.assertEqual({row.fit for row in rows}, {"no_floor", "floor"})
        keys = {(row.variant, row.performance.block, row.performance.truth, row.fit) for row in rows}
        self.assertEqual(len(keys), len(rows))
        for row in rows:
            self.assertEqual(row.performance.replications, 2)
            if row.fit == "no_floor":
                self.assertEqual(row.mean_floor_estimate, 0.0)
            else:
                self.assertTrue(0.0 < row.mean_floor_estimate < 0.10)
        trials = {(row.variant, row.performance.block): row.performance.n_trials for row in rows}
        self.assertEqual(trials, {(variant, block.name): block.n_trials for variant, block in designs})

    def test_the_two_fits_of_a_row_analyse_the_same_blocks_and_the_seed_decides_them(self):
        first = simulate.run_floor_check(PROTOCOL, replications=2, seed=5)
        again = simulate.run_floor_check(PROTOCOL, replications=2, seed=5)
        other = simulate.run_floor_check(PROTOCOL, replications=2, seed=6)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_the_check_ignores_the_floor_values_of_the_protocol_itself(self):
        # Whatever the protocol's own floor values are, the check compares no floor with the three values.
        with_floors = dict(PROTOCOL, analysis=dict(PROTOCOL["analysis"], floor_values=[0.0, 0.2]))
        analysis = {key: value for key, value in PROTOCOL["analysis"].items() if key != "floor_values"}
        without = dict(PROTOCOL, analysis=analysis)
        self.assertEqual(
            simulate.run_floor_check(with_floors, replications=1, seed=5),
            simulate.run_floor_check(without, replications=1, seed=5),
        )


class PairedPowerTests(unittest.TestCase):
    def test_values_worked_by_hand(self):
        # The first policy always succeeds and the second half the time: the pairs that differ all favour the first,
        # and with 6 pairs the test rejects only when all 6 differ (p = 2/64): probability (1/2)^6.
        self.assertAlmostEqual(paired_power(6, 1.0, 0.5), 1 / 64, places=12)
        # One always succeeds and the other never: every pair differs the same way. Six such pairs reach 5 %, five
        # do not.
        self.assertAlmostEqual(paired_power(6, 1.0, 0.0), 1.0, places=12)
        self.assertEqual(paired_power(5, 1.0, 0.0), 0.0)

    def test_two_equal_policies_are_declared_different_no_more_often_than_alpha(self):
        for pairs in (16, 20, 24):
            for rate in (0.1, 0.5, 0.9):
                with self.subTest(pairs=pairs, rate=rate):
                    self.assertLessEqual(paired_power(pairs, rate, rate), 0.05)

    def test_it_does_not_matter_which_policy_is_called_first(self):
        self.assertAlmostEqual(paired_power(20, 0.9, 0.4), paired_power(20, 0.4, 0.9), places=12)

    def test_a_larger_difference_is_detected_more_often(self):
        powers = [paired_power(20, 0.9, other) for other in (0.9, 0.7, 0.5, 0.3, 0.1)]
        self.assertEqual(powers, sorted(powers))
        self.assertGreater(powers[-1], 0.9)

    def test_it_matches_a_simulation_of_the_paired_test(self):
        generator = rng(44)
        for pairs, first, second in ((20, 0.9, 0.5), (16, 0.8, 0.3), (24, 0.6, 0.5)):
            with self.subTest(pairs=pairs, first=first, second=second):
                shares = [first * (1 - second), (1 - first) * second, first * second + (1 - first) * (1 - second)]
                tables = generator.multinomial(pairs, shares, size=20000)
                rejected = np.mean([mcnemar_exact(int(a), int(b)) < 0.05 for a, b, _ in tables])
                self.assertAlmostEqual(paired_power(pairs, first, second), float(rejected), delta=0.015)

    def test_impossible_inputs_are_refused(self):
        bad = ((0, 0.5, 0.5, 0.05), (20, 1.5, 0.5, 0.05), (20, 0.5, -0.1, 0.05), (20, 0.5, 0.5, 0.0))
        for pairs, first, second, alpha in bad:
            with self.subTest(pairs=pairs, first=first, second=second, alpha=alpha), self.assertRaises(ValueError):
                paired_power(pairs, first, second, alpha)


class SweepTests(unittest.TestCase):
    TRUTHS = (Truth(20.0, 1.0, 0.0), Truth(10.0, 6.0, 0.05))

    def test_one_row_for_every_truth_block_and_design(self):
        rows = run_sweep(PROTOCOL, replications=2, seed=5, truths=self.TRUTHS)
        self.assertEqual(len(rows), len(self.TRUTHS) * expected_designs_over_blocks(PROTOCOL))
        keys = {(row.design, row.performance.block, row.performance.truth) for row in rows}
        self.assertEqual(len(keys), len(rows))
        self.assertTrue(set(simulate.DESIGNS) <= {row.design for row in rows})
        self.assertTrue(all(row.performance.replications == 2 for row in rows))

    def test_the_wider_range_is_run_only_for_blocks_that_stop_short_of_the_widest(self):
        blocks = blocks_from_protocol(PROTOCOL)
        short = dict(PROTOCOL["blocks"][blocks[0].name], max_level_mm=30.0)
        far = dict(PROTOCOL["blocks"][blocks[0].name], max_level_mm=45.0)
        protocol = dict(PROTOCOL, blocks={"short": short, "far": far})
        rows = run_sweep(protocol, replications=1, seed=5, truths=self.TRUTHS[:1])
        wider = [row for row in rows if row.design == simulate.WIDER_RANGE]
        self.assertEqual([row.performance.block for row in wider], ["short"])
        self.assertEqual(len(rows), 2 * len(simulate.DESIGNS) + 1)

    def test_the_seed_decides_every_number(self):
        first = run_sweep(PROTOCOL, replications=2, seed=5, truths=self.TRUTHS[:1], designs=("protocol", "constant"))
        again = run_sweep(PROTOCOL, replications=2, seed=5, truths=self.TRUTHS[:1], designs=("protocol", "constant"))
        other = run_sweep(PROTOCOL, replications=2, seed=6, truths=self.TRUTHS[:1], designs=("protocol", "constant"))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_a_row_does_not_depend_on_which_other_rows_are_run(self):
        whole = run_sweep(PROTOCOL, replications=2, seed=5, truths=self.TRUTHS)
        part = run_sweep(PROTOCOL, replications=2, seed=5, truths=self.TRUTHS[1:], designs=("fine_step",))
        self.assertGreater(len(part), 0)
        for row in part:
            self.assertIn(row, whole)

    def test_rows_that_differ_in_anything_draw_their_own_random_numbers(self):
        # Two truths too close to change any outcome, two blocks that differ in name only, two designs that happen to
        # coincide: if their rows shared random numbers, their results would be identical.
        base = Truth(20.0, 3.0, 0.0)
        nearly = (
            Truth(20.0 + 1e-9, 3.0, 0.0),
            Truth(20.0, 3.0 + 1e-9, 0.0),
            Truth(20.0, 3.0, 1e-12),
            Truth(20.0, 3.0, 0.0, floor=1e-12),
        )
        rows = run_sweep(PROTOCOL, replications=5, seed=5, truths=(base, *nearly), designs=("protocol",))
        first_block = rows[0].performance.block

        def what_the_blocks_gave(performance: simulate.DesignPerformance) -> tuple:
            # The mean of the medians (the bias with the truth added back), so that the truth itself does not differ.
            return (
                round(performance.bias_mm + performance.truth.r50_mm, 6),
                performance.mean_interval_width_mm,
                performance.interval_coverage,
            )

        results = [what_the_blocks_gave(row.performance) for row in rows if row.performance.block == first_block]
        self.assertEqual(len(results), 5)
        self.assertEqual(len(set(results)), 5)

        one_block = next(iter(blocks_from_protocol(PROTOCOL))).name
        twins = dict(PROTOCOL, blocks={"first": PROTOCOL["blocks"][one_block], "second": PROTOCOL["blocks"][one_block]})
        by_name = run_sweep(twins, replications=3, seed=5, truths=(base,), designs=("protocol",))
        self.assertEqual([row.performance.block for row in by_name], ["first", "second"])
        self.assertNotEqual(replace(by_name[0].performance, block=""), replace(by_name[1].performance, block=""))

        already_fine = dict(PROTOCOL, staircase=dict(PROTOCOL["staircase"], step_final_mm=simulate.FINE_STEP_FINAL_MM))
        by_design = run_sweep(already_fine, replications=3, seed=5, truths=(base,), designs=("protocol", "fine_step"))
        by_design = [row for row in by_design if row.performance.block == first_block]
        self.assertEqual([row.design for row in by_design], ["protocol", "fine_step"])
        self.assertNotEqual(by_design[0].performance, by_design[1].performance)

    def test_the_interval_level_comes_from_the_protocol_and_the_placement_noise_from_the_caller(self):
        easy = (Truth(20.0, 3.0, 0.0),)
        usual = run_sweep(PROTOCOL, replications=4, seed=5, truths=easy, designs=("protocol",))
        half_level = dict(PROTOCOL, analysis=dict(PROTOCOL["analysis"], interval_level=0.5))
        narrower = run_sweep(half_level, replications=4, seed=5, truths=easy, designs=("protocol",))
        exact = run_sweep(PROTOCOL, replications=4, seed=5, truths=easy, designs=("protocol",), placement_sd_mm=0.0)
        for usual_row, narrow_row, exact_row in zip(usual, narrower, exact, strict=True):
            with self.subTest(block=usual_row.performance.block):
                # The same trials (same seed), a 50 % interval in place of the protocol's: narrower, same medians.
                self.assertLess(
                    narrow_row.performance.mean_interval_width_mm, usual_row.performance.mean_interval_width_mm
                )
                self.assertEqual(narrow_row.performance.bias_mm, usual_row.performance.bias_mm)
                self.assertNotEqual(exact_row.performance, usual_row.performance)

    def test_the_full_sweep_covers_the_stated_truths(self):
        truths = simulate.sweep_truths()
        self.assertEqual(len(truths), 6 * 4 * 2)
        self.assertEqual({truth.r50_mm for truth in truths}, {5.0, 10.0, 15.0, 20.0, 30.0, 40.0})
        self.assertEqual({truth.width_mm for truth in truths}, {1.0, 3.0, 6.0, 10.0})
        self.assertEqual({truth.lapse for truth in truths}, {0.0, 0.05})

    def test_an_unknown_design_is_refused(self):
        with self.assertRaises(ValueError):
            run_sweep(PROTOCOL, replications=1, seed=1, truths=self.TRUTHS[:1], designs=("protocol", "bigger"))


class PowerTableTests(unittest.TestCase):
    def test_one_row_for_every_number_of_pairs_and_every_pair_of_rates(self):
        table = simulate.paired_power_table((16, 20), rates=(0.3, 0.6, 0.9))
        self.assertEqual(len(table), 2 * 3 * 3)
        self.assertEqual({row.pairs for row in table}, {16, 20})
        for row in table:
            self.assertEqual(row.power, paired_power(row.pairs, row.p_first, row.p_second))

    def test_the_rate_needed_for_a_given_power_is_the_first_that_reaches_it(self):
        needed = simulate.rate_needed(20, 0.4, power=0.8)
        self.assertIsNotNone(needed)
        self.assertGreaterEqual(paired_power(20, 0.4, needed), 0.8)
        self.assertLess(paired_power(20, 0.4, round(needed - 0.05, 2)), 0.8)
        self.assertAlmostEqual(needed * 20, round(needed * 20), places=9)  # on the grid of twentieths
        # From 0.9 no higher rate exists that would be told apart: there is nothing to report.
        self.assertIsNone(simulate.rate_needed(20, 0.9, power=0.8))


class CommandTests(unittest.TestCase):
    """`python -m tolerance_radius.simulate`: the whole sweep on a tiny replication count, and what it writes."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.out = Path(cls._directory.name) / "first"
        cls.again = Path(cls._directory.name) / "again"
        cls.printed = cls.run_command(cls.out)
        cls.run_command(cls.again)
        cls.report = (cls.out / "procedure.md").read_text(encoding="utf-8")
        cls.words = " ".join(cls.report.split())  # the report without its line breaks, to look for phrases in

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    @staticmethod
    def run_command(out: Path, seed: int = 7) -> str:
        arguments = ["--protocol", str(config.DEFAULT_PROTOCOL_PATH), "--out", str(out), "--replications", "2"]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = simulate.main([*arguments, "--seed", str(seed)])
        assert status == 0
        return printed.getvalue()

    def test_it_writes_the_table_of_every_row_and_the_report(self):
        for name in ("procedure.csv", "procedure.md", "floor_check.csv", "paired_power.csv"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertIn("procedure.md", self.printed)

    def test_the_floor_check_has_one_line_for_every_block_truth_and_fit(self):
        with open(self.out / "floor_check.csv", newline="", encoding="utf-8") as handle:
            lines = list(csv.DictReader(handle))
        expected = len(simulate.floor_check_designs(PROTOCOL)) * len(simulate.floor_check_truths()) * 2
        self.assertEqual(len(lines), expected)
        self.assertEqual({line["fit"] for line in lines}, {"no_floor", "floor"})
        self.assertEqual({line["floor_true"] for line in lines}, {"0.08", "0"})
        self.assertEqual({line["variant"] for line in lines}, {"as_written", "far_anchors", "longer_staircases"})
        rows = simulate.run_floor_check(PROTOCOL, replications=2, seed=7)
        for line, row in zip(lines, rows, strict=True):
            written = (line["block"], line["variant"], line["fit"])
            self.assertEqual(written, (row.performance.block, row.variant, row.fit))
            self.assertAlmostEqual(float(line["rmse_mm"]), row.performance.rmse_mm, places=3)
            self.assertAlmostEqual(float(line["mean_floor_estimate"]), row.mean_floor_estimate, places=3)

    def test_the_table_has_one_line_for_every_row_of_the_sweep(self):
        with open(self.out / "procedure.csv", newline="", encoding="utf-8") as handle:
            lines = list(csv.DictReader(handle))
        self.assertEqual(len(lines), len(simulate.sweep_truths()) * expected_designs_over_blocks(PROTOCOL))
        self.assertEqual(
            list(lines[0]),
            [
                "design",
                "block",
                "n_trials",
                "r50_true_mm",
                "width_true_mm",
                "lapse_true",
                "replications",
                "bias_mm",
                "rmse_mm",
                "mean_interval_width_mm",
                "interval_coverage",
                "lower_edge_rate",
                "upper_edge_rate",
                "any_edge_rate",
            ],
        )
        self.assertEqual({line["replications"] for line in lines}, {"2"})
        self.assertTrue(set(simulate.DESIGNS) <= {line["design"] for line in lines})
        for line in lines:
            self.assertTrue(0.0 <= float(line["interval_coverage"]) <= 1.0)
            self.assertGreaterEqual(float(line["rmse_mm"]), abs(float(line["bias_mm"])) - 1e-6)

    def test_the_table_agrees_with_the_sweep_run_directly(self):
        rows = run_sweep(PROTOCOL, replications=2, seed=7, truths=simulate.sweep_truths()[:1], designs=("protocol",))
        with open(self.out / "procedure.csv", newline="", encoding="utf-8") as handle:
            columns = ("design", "block", "r50_true_mm", "width_true_mm", "lapse_true")
            lines = {tuple(line[column] for column in columns): line for line in csv.DictReader(handle)}
        for row in rows:
            performance, truth = row.performance, row.performance.truth
            key = (row.design, performance.block, f"{truth.r50_mm:g}", f"{truth.width_mm:g}", f"{truth.lapse:g}")
            line = lines[key]
            self.assertAlmostEqual(float(line["rmse_mm"]), performance.rmse_mm, places=3)
            self.assertAlmostEqual(float(line["mean_interval_width_mm"]), performance.mean_interval_width_mm, places=3)
            self.assertEqual(int(line["n_trials"]), performance.n_trials)

    def test_the_power_table_covers_the_protocol_s_pairs_and_four_more_or_fewer(self):
        pairs = next(block["pairs"] for block in PROTOCOL["blocks"].values() if "pairs" in block)
        with open(self.out / "paired_power.csv", newline="", encoding="utf-8") as handle:
            lines = list(csv.DictReader(handle))
        self.assertEqual(list(lines[0]), ["pairs", "p_first", "p_second", "power"])
        self.assertEqual({int(line["pairs"]) for line in lines}, {pairs - 4, pairs, pairs + 4})
        for line in lines[:: len(lines) // 7]:
            expected = paired_power(int(line["pairs"]), float(line["p_first"]), float(line["p_second"]))
            self.assertAlmostEqual(float(line["power"]), expected, places=4)

    def test_the_report_states_the_seed_the_replications_and_that_its_numbers_are_simulated(self):
        self.assertIn("Seed 7", self.words)
        self.assertIn("2 simulated blocks", self.words)
        self.assertIn("simulated", self.report.split("\n\n")[1])  # said at the top, before any number
        self.assertIn(f"version {PROTOCOL['protocol']['version']}", self.words)
        for heading in ("## Recommendation", "## Reading", "## What this simulation does not test", "## Terms"):
            self.assertIn(heading, self.report)
        for block in blocks_from_protocol(PROTOCOL):
            self.assertIn(block.name, self.report)

    def test_the_report_has_the_floor_check_and_says_what_to_do_with_the_floor(self):
        self.assertIn("## 6. A floor in the model", self.report)
        self.assertIn("floor_values", self.words)
        self.assertIn("far anchors", self.words)
        self.assertIn("Floor estimate", self.report)

    def test_every_table_of_the_report_is_labelled_simulated_or_computed(self):
        captions = [line for line in self.report.splitlines() if line.startswith("**Table ")]
        self.assertGreaterEqual(len(captions), 10)
        tables = self.report.split("**Table ")[1:]
        for table in tables:
            caption = " ".join(table.split("\n\n")[0].split())
            self.assertTrue("Simulated." in caption or "Computed." in caption, caption[:80])

    def test_the_report_names_no_person_and_no_place_on_this_machine(self):
        # The review is double blind and the report is committed: no home directory, no user name, no absolute path.
        home = str(Path.home())
        tables = [(self.out / name).read_text(encoding="utf-8") for name in ("procedure.csv", "floor_check.csv")]
        for output in (self.report, *tables):
            self.assertNotIn(home, output)
            self.assertNotIn(Path.home().name, output)
            self.assertNotIn(str(self.out), output)
            self.assertNotIn(str(config.DEFAULT_PROTOCOL_PATH), output)

    def test_running_it_again_writes_the_same_files_and_another_seed_different_ones(self):
        for name in ("procedure.csv", "procedure.md", "floor_check.csv", "paired_power.csv"):
            with self.subTest(name=name):
                self.assertEqual((self.out / name).read_bytes(), (self.again / name).read_bytes())
        other = Path(self._directory.name) / "other"
        self.run_command(other, seed=8)
        self.assertNotEqual((other / "procedure.csv").read_bytes(), (self.out / "procedure.csv").read_bytes())
        self.assertNotEqual((other / "floor_check.csv").read_bytes(), (self.out / "floor_check.csv").read_bytes())
        self.assertEqual((other / "paired_power.csv").read_bytes(), (self.out / "paired_power.csv").read_bytes())

    def test_no_line_of_the_report_is_left_with_an_unfilled_number(self):
        # Even with 2 simulated blocks, where some errors are exactly zero and some ratios do not exist.
        for marker in ("{", "}", "None"):
            self.assertNotIn(marker, self.report, marker)
        self.assertIsNone(re.search(r"\b(nan|inf)\b", self.report, flags=re.IGNORECASE))

    def test_it_runs_as_a_module_and_explains_itself(self):
        source = Path(simulate.__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=str(source))
        run = subprocess.run(
            [sys.executable, "-m", "tolerance_radius.simulate", "--help"],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        for option in ("--protocol", "--out", "--replications", "--seed"):
            self.assertIn(option, run.stdout)

    def test_a_replication_count_below_one_is_refused(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            protocol = str(config.DEFAULT_PROTOCOL_PATH)
            simulate.main(["--protocol", protocol, "--out", str(self.out), "--replications", "0"])


if __name__ == "__main__":
    unittest.main()
