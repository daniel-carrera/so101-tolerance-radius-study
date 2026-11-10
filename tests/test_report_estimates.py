"""The report's statistics: the scale rule, the bins, the difference of two radii, the predicted count."""

import itertools
import math
import unittest

import numpy as np

from tolerance_radius import config
from tolerance_radius.intervals import mcnemar_exact, wilson_interval
from tolerance_radius.psychometric import fit_threshold, grids_from_protocol, success_probability
from tolerance_radius.report import estimates

ANALYSIS = config.load_protocol()["analysis"]
GRIDS = grids_from_protocol(ANALYSIS)
#: The same span with fewer values: for tests of what a fit's summary holds, where its numbers do not matter.
COARSE = grids_from_protocol(
    ANALYSIS
    | {"r50_grid_mm": {"min": 0.0, "max": 80.0, "step": 2.0}, "width_grid_mm": {"min": 0.5, "max": 20.0, "count": 8}}
)
#: A grid small enough to walk through value by value.
SMALL = {
    "r50_grid_mm": [10.0, 20.0, 30.0],
    "width_grid_mm": [1.0, 5.0],
    "lapse_values": [0.0, 0.1],
    "floor_values": [0.0, 0.2],
}


def trials(seed: int, n: int, r50_mm: float, width_mm: float, largest_mm: float = 50.0):
    """Simulated trials at distances spread evenly at random."""
    rng = np.random.default_rng(seed)
    distances = rng.uniform(0.0, largest_mm, size=n)
    return distances, rng.random(n) < success_probability(distances, r50_mm, width_mm, 0.02)


class ScaleRuleTests(unittest.TestCase):
    def test_runs_within_one_point_are_divided_by_their_mean(self):
        rule = estimates.scale_rule([1.030, 1.027, 1.033, 1.031])
        self.assertTrue(rule["applied"])
        self.assertAlmostEqual(rule["factor"], 1.03025, places=12)
        self.assertEqual(rule["divisor"], rule["factor"])
        self.assertEqual((rule["factor_min"], rule["factor_max"], rule["n_runs"]), (1.027, 1.033, 4))
        self.assertAlmostEqual(rule["spread_points"], 0.6, places=9)

    def test_the_line_is_one_percentage_point_and_belongs_to_the_agreeing_side(self):
        just_inside = estimates.scale_rule([1.0200, 1.0299, 1.025, 1.025])
        on_the_line = estimates.scale_rule([1.02, 1.03, 1.025, 1.025])
        just_outside = estimates.scale_rule([1.0200, 1.0301, 1.025, 1.025])
        self.assertTrue(just_inside["applied"])
        self.assertTrue(on_the_line["applied"], "1.03 - 1.02 is one point, whatever its last binary digit")
        self.assertFalse(just_outside["applied"])

    def test_runs_that_disagree_leave_distances_as_read_and_still_report_the_factor(self):
        rule = estimates.scale_rule([1.020, 1.027, 1.033, 1.031])
        self.assertFalse(rule["applied"])
        self.assertEqual(rule["divisor"], 1.0)
        self.assertAlmostEqual(rule["factor"], 1.02775, places=12)
        self.assertEqual((rule["factor_min"], rule["factor_max"]), (1.020, 1.033))
        self.assertIn("differ", rule["why"])

    def test_no_run_and_one_run_make_no_correction(self):
        none = estimates.scale_rule([])
        self.assertEqual((none["applied"], none["factor"], none["n_runs"], none["divisor"]), (False, None, 0, 1.0))
        self.assertEqual((none["factor_min"], none["factor_max"]), (None, None))
        one = estimates.scale_rule([1.03])
        self.assertEqual((one["applied"], one["factor"], one["n_runs"], one["divisor"]), (False, 1.03, 1, 1.0))

    def test_a_factor_that_cannot_be_is_refused(self):
        for bad in ([0.0], [1.03, -1.0], [float("nan")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimates.scale_rule(bad)


class BinTests(unittest.TestCase):
    def test_a_bin_is_closed_below_and_open_above(self):
        expected = {0.0: 0, 9.999999: 0, 10.0: 1, 19.99: 1, 20.0: 2, 39.99: 3, 40.0: 4, 40.01: 4, 50.0: 5}
        for distance, index in expected.items():
            with self.subTest(distance=distance):
                self.assertEqual(estimates.bin_index(distance), index)

    def test_the_largest_number_below_an_edge_stays_below_it(self):
        for edge in (10.0, 30.0, 70.0):
            self.assertEqual(estimates.bin_index(math.nextafter(edge, 0.0)), int(edge // 10) - 1)
            self.assertEqual(estimates.bin_index(edge), int(edge // 10))

    def test_every_distance_lies_in_the_bin_it_is_given(self):
        rng = np.random.default_rng(3)
        for width in (10.0, 2.5, 0.3):
            for distance in rng.uniform(0.0, 90.0, size=300).tolist():
                index = estimates.bin_index(distance, width)
                self.assertLessEqual(index * width, distance)
                self.assertLess(distance, (index + 1) * width)

    def test_a_trial_at_exactly_ten_and_one_at_forty_point_oh_one(self):
        bins = estimates.distance_bins([10.0, 40.01, 3.0, 9.99], [True, False, True, True])
        self.assertEqual(
            [(b["lo_mm"], b["hi_mm"], b["n"], b["k"]) for b in bins],
            [(0.0, 10.0, 2, 2), (10.0, 20.0, 1, 1), (40.0, 50.0, 1, 0)],
        )

    def test_each_bin_carries_its_wilson_interval_and_empty_bins_are_left_out(self):
        distances = [1.0, 2.0, 3.0, 11.0, 12.0, 35.0]
        bins = estimates.distance_bins(distances, [True, True, False, True, False, False], level=0.9)
        self.assertEqual([b["lo_mm"] for b in bins], [0.0, 10.0, 30.0])
        for entry in bins:
            self.assertEqual((entry["wilson_lo"], entry["wilson_hi"]), wilson_interval(entry["k"], entry["n"], 0.9))
        self.assertEqual(estimates.distance_bins([], []), [])

    def test_the_farthest_success_and_the_nearest_failure(self):
        self.assertEqual(estimates.extremes([1.0, 22.0, 14.0, 30.0], [True, True, False, False]), (22.0, 14.0))
        self.assertEqual(estimates.extremes([1.0, 2.0], [True, True]), (2.0, None))
        self.assertEqual(estimates.extremes([1.0, 2.0], [False, False]), (None, 1.0))
        self.assertEqual(estimates.extremes([], []), (None, None))

    def test_a_distance_that_cannot_be_and_unequal_lists_are_refused(self):
        with self.assertRaises(ValueError):
            estimates.bin_index(-0.1)
        with self.assertRaises(ValueError):
            estimates.distance_bins([1.0, 2.0], [True])

    def test_a_proportion_without_trials_has_no_interval(self):
        self.assertEqual(estimates.proportion(0, 0), {"k": 0, "n": 0, "wilson_lo": None, "wilson_hi": None})
        low, high = wilson_interval(3, 7)
        self.assertEqual(estimates.proportion(3, 7), {"k": 3, "n": 7, "wilson_lo": low, "wilson_hi": high})


class FitSummaryTests(unittest.TestCase):
    def test_it_reports_the_fit_in_the_contract_s_words(self):
        distances, outcomes = trials(seed=1, n=34, r50_mm=21.0, width_mm=3.0)
        posterior = estimates.fit(distances, outcomes, COARSE)
        summary = estimates.fit_summary(posterior, ["T1a"], level=0.95, largest_tested_mm=float(distances.max()))
        told = posterior.summary(0.95)
        self.assertEqual(summary["blocks"], ["T1a"])
        self.assertEqual((summary["n"], summary["k"]), (34, int(outcomes.sum())))
        self.assertEqual(
            (summary["r50_median_mm"], summary["r50_lo_mm"], summary["r50_hi_mm"]),
            (told.r50_median_mm, told.r50_low_mm, told.r50_high_mm),
        )
        self.assertEqual(
            (summary["open_below"], summary["open_above"]), (told.reaches_lower_edge, told.reaches_upper_edge)
        )
        self.assertEqual((summary["lapse_mean"], summary["floor_mean"]), (told.lapse_mean, told.floor_mean))

    def test_the_curve_runs_from_zero_past_the_largest_distance_tested_in_half_millimetres(self):
        distances, outcomes = trials(seed=2, n=20, r50_mm=20.0, width_mm=3.0)
        posterior = estimates.fit(distances, outcomes, COARSE)
        for largest in (31.7, 80.0, 93.2):
            curve = estimates.fit_summary(posterior, ["T1a"], level=0.95, largest_tested_mm=largest)["curve"]
            d = np.array(curve["d_mm"])
            with self.subTest(largest=largest):
                self.assertEqual(d[0], 0.0)
                self.assertGreaterEqual(d[-1], max(largest, 80.0))
                self.assertTrue(np.allclose(np.diff(d), 0.5, rtol=0.0, atol=1e-12))
                self.assertEqual({len(curve[key]) for key in ("d_mm", "mean", "lo", "hi")}, {d.size})
                self.assertTrue(np.array_equal(curve["mean"], posterior.predict(d)))
                low, high = posterior.band(d, 0.95)
                self.assertTrue(np.array_equal(curve["lo"], low) and np.array_equal(curve["hi"], high))
                self.assertTrue(
                    np.all((np.array(curve["lo"]) <= curve["mean"]) & (np.array(curve["mean"]) <= curve["hi"]))
                )

    def test_all_successes_leave_the_interval_open_above(self):
        posterior = estimates.fit([0.0, 5.0, 10.0, 15.0], [True] * 4, COARSE)
        summary = estimates.fit_summary(posterior, ["T2"], level=0.95, largest_tested_mm=15.0)
        self.assertTrue(summary["open_above"])
        self.assertFalse(summary["open_below"])


class RadiusDifferenceTests(unittest.TestCase):
    def by_walking_every_pair(self, first, second, level):
        """The same comparison the slow way: every pair of grid values, one at a time, in plain Python."""
        first_grid, first_probability = (values.tolist() for values in first.r50_marginal())
        second_grid, second_probability = (values.tolist() for values in second.r50_marginal())
        mass: dict[float, float] = {}
        greater = equal = 0.0
        for a, pa in zip(first_grid, first_probability, strict=True):
            for b, pb in zip(second_grid, second_probability, strict=True):
                mass[a - b] = mass.get(a - b, 0.0) + pa * pb
                if a > b:
                    greater += pa * pb
                elif a == b:
                    equal += pa * pb
        total = sum(mass.values())
        tail = (1.0 - level) / 2.0
        running, quantiles = 0.0, {}
        for value in sorted(mass):
            running += mass[value]
            for name, mark in (("lo", tail), ("median", 0.5), ("hi", 1.0 - tail)):
                if name not in quantiles and running >= mark * total * (1.0 - 1e-12):
                    quantiles[name] = value
        return quantiles, greater / total, equal / total

    def test_it_matches_a_walk_through_every_pair_of_grid_values(self):
        policy = estimates.fit(*trials(seed=11, n=34, r50_mm=22.0, width_mm=3.0), GRIDS)
        replay = estimates.fit(*trials(seed=12, n=22, r50_mm=13.0, width_mm=2.0), GRIDS)
        for first, second in ((policy, replay), (replay, policy), (policy, policy)):
            for level in (0.95, 0.8):
                found = estimates.radius_difference(first, second, level)
                quantiles, greater, equal = self.by_walking_every_pair(first, second, level)
                with self.subTest(level=level):
                    self.assertEqual(
                        (found["lo_mm"], found["median_mm"], found["hi_mm"]),
                        (quantiles["lo"], quantiles["median"], quantiles["hi"]),
                    )
                    self.assertAlmostEqual(found["prob_greater"], greater + 0.5 * equal, places=9)
                    self.assertAlmostEqual(found["prob_equal"], equal, places=9)

    def test_the_differences_are_weighted_exactly_and_sum_to_one(self):
        policy = estimates.fit(*trials(seed=13, n=34, r50_mm=22.0, width_mm=3.0), GRIDS)
        replay = estimates.fit(*trials(seed=14, n=22, r50_mm=13.0, width_mm=2.0), GRIDS)
        values, probabilities = estimates.difference_distribution(policy, replay)
        self.assertTrue(np.all(np.diff(values) > 0.0))
        self.assertEqual((values[0], values[-1]), (-80.0, 80.0))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        mean = float(policy.r50_marginal()[1] @ policy.r50_grid_mm - replay.r50_marginal()[1] @ replay.r50_grid_mm)
        self.assertAlmostEqual(float(values @ probabilities), mean, places=9)

    def test_a_clearly_larger_radius_is_called_larger_and_the_reverse_smaller(self):
        policy = estimates.fit(*trials(seed=15, n=60, r50_mm=30.0, width_mm=2.0), GRIDS)
        replay = estimates.fit(*trials(seed=16, n=60, r50_mm=10.0, width_mm=2.0), GRIDS)
        forward = estimates.radius_difference(policy, replay)
        backward = estimates.radius_difference(replay, policy)
        self.assertEqual((forward["verdict"], backward["verdict"]), ("larger", "smaller"))
        self.assertGreater(forward["lo_mm"], 0.0)
        self.assertAlmostEqual(forward["prob_greater"] + backward["prob_greater"], 1.0, places=8)
        self.assertEqual(
            (forward["median_mm"], forward["lo_mm"], forward["hi_mm"]),
            (-backward["median_mm"], -backward["hi_mm"], -backward["lo_mm"]),
        )
        self.assertEqual(forward["fits"], "block")

    def test_a_radius_compared_with_itself_is_not_told_apart(self):
        posterior = estimates.fit(*trials(seed=17, n=34, r50_mm=20.0, width_mm=3.0), GRIDS)
        same = estimates.radius_difference(posterior, posterior)
        self.assertAlmostEqual(same["prob_greater"], 0.5, places=9)
        self.assertEqual((same["verdict"], same["median_mm"]), ("not told apart", 0.0))
        self.assertGreater(same["prob_equal"], 0.0)

    def test_two_posteriors_on_one_grid_value_tie_and_are_not_called_smaller(self):
        # A grid with one radius: the two fits cannot differ. Counting a tie as "not larger" would give probability 0.
        one_radius = SMALL | {"r50_grid_mm": [20.0]}
        first = fit_threshold([5.0, 30.0], [True, False], **one_radius)
        second = fit_threshold([8.0, 25.0], [True, False], **one_radius)
        same = estimates.radius_difference(first, second)
        self.assertEqual((same["prob_equal"], same["prob_greater"], same["verdict"]), (1.0, 0.5, "not told apart"))

    def test_the_rule_of_prediction_two(self):
        cases = {
            1.0: "larger",
            0.95: "larger",
            0.9499: "not told apart",
            0.5: "not told apart",
            0.0501: "not told apart",
            0.05: "smaller",
            0.0: "smaller",
        }
        for probability, expected in cases.items():
            with self.subTest(probability=probability):
                self.assertEqual(estimates.verdict(probability), expected)


class PairedTests(unittest.TestCase):
    def test_the_table_counts_the_four_kinds_of_pair(self):
        first = [True, True, False, False, True, False]
        second = [True, False, True, False, True, True]
        table = estimates.paired_table(first, second)
        self.assertEqual(table, {"both": 2, "neither": 1, "only_first": 1, "only_second": 2})
        self.assertEqual(estimates.paired_p_value(table), mcnemar_exact(1, 2))

    def test_no_pair_gives_no_p_value_and_unequal_lists_are_refused(self):
        self.assertIsNone(estimates.paired_p_value(estimates.paired_table([], [])))
        with self.assertRaises(ValueError):
            estimates.paired_table([True], [])


class CountIntervalTests(unittest.TestCase):
    def test_each_end_is_the_first_count_at_which_the_running_total_reaches_its_mark(self):
        probabilities = [0.01, 0.01, 0.03, 0.15, 0.30, 0.30, 0.15, 0.03, 0.01, 0.01]
        self.assertEqual(estimates.count_interval(probabilities, 0.95), (2, 7))
        self.assertEqual(estimates.count_interval(probabilities, 0.5), (4, 5))
        self.assertEqual(estimates.count_interval([1.0], 0.95), (0, 0))

    def test_it_leaves_less_than_its_share_below_and_at_most_its_share_above(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            probabilities = rng.dirichlet(np.ones(21) * 0.4)
            low, high = estimates.count_interval(probabilities, 0.95)
            self.assertLess(probabilities[:low].sum(), 0.025 + 1e-12)
            self.assertLessEqual(probabilities[high + 1 :].sum(), 0.025 + 1e-12)
            self.assertGreaterEqual(probabilities[: low + 1].sum(), 0.025 - 1e-12)
            self.assertGreaterEqual(probabilities[low : high + 1].sum(), 0.95 - 1e-12)

    def test_an_exact_tie_at_a_mark_does_not_depend_on_rounding(self):
        # 2.5 % lies exactly on count 0: the running total reaches the mark there.
        self.assertEqual(estimates.count_interval([0.025, 0.95, 0.025], 0.95), (0, 1))

    def test_what_is_not_a_distribution_is_refused(self):
        for bad in ([], [0.0, 0.0], [0.5, -0.5, 1.0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimates.count_interval(bad)
        with self.assertRaises(ValueError):
            estimates.count_interval([1.0], 1.0)

    def test_a_count_matches_inside_the_interval_ends_included(self):
        verdicts = [estimates.lookup_verdict(observed, 6, 16) for observed in (5, 6, 10, 16, 17)]
        self.assertEqual(verdicts, ["falls short", "matches", "matches", "matches", "exceeds"])


class LookupPredictionTests(unittest.TestCase):
    def setUp(self):
        self.posterior = estimates.fit(*trials(seed=21, n=34, r50_mm=21.0, width_mm=3.0), GRIDS)
        self.distances = np.random.default_rng(22).uniform(0.0, 36.0, size=26)

    def test_it_expects_the_sum_of_the_predicted_probabilities(self):
        found = estimates.lookup_prediction(self.posterior, self.distances, observed=12)
        self.assertAlmostEqual(found["expected"], float(self.posterior.predict(self.distances).sum()), places=12)
        self.assertEqual(found["observed"], 12)
        self.assertEqual(len(found["pmf"]), 27)
        self.assertAlmostEqual(sum(found["pmf"]), 1.0, places=12)

    def test_its_interval_is_that_of_the_count_s_whole_distribution(self):
        found = estimates.lookup_prediction(self.posterior, self.distances, observed=12)
        counts = self.posterior.count_distribution(self.distances)
        self.assertEqual((found["lo"], found["hi"]), estimates.count_interval(counts, 0.95))
        self.assertLess(found["lo"], found["expected"])
        self.assertGreater(found["hi"], found["expected"])

    def test_it_is_wider_than_the_chance_in_the_trials_alone_would_make_it(self):
        # Treating the posterior-mean probabilities as known leaves the fit's uncertainty out.
        found = estimates.lookup_prediction(self.posterior, self.distances, observed=12)
        chances = self.posterior.predict(self.distances).tolist()
        known = [1.0]
        for chance in chances:
            known = [a * (1.0 - chance) + b * chance for a, b in zip([*known, 0.0], [0.0, *known], strict=True)]
        low, high = estimates.count_interval(known, 0.95)
        self.assertGreaterEqual(found["hi"] - found["lo"], high - low)
        self.assertGreater(found["hi"] - found["lo"], 0)

    def test_the_verdict_follows_the_observed_count(self):
        ends = estimates.lookup_prediction(self.posterior, self.distances, observed=0)
        self.assertEqual(ends["verdict"], "falls short")
        self.assertEqual(estimates.lookup_prediction(self.posterior, self.distances, ends["lo"])["verdict"], "matches")
        self.assertEqual(estimates.lookup_prediction(self.posterior, self.distances, 26)["verdict"], "exceeds")
        with self.assertRaises(ValueError):
            estimates.lookup_prediction(self.posterior, self.distances, observed=27)


class TrialCountTests(unittest.TestCase):
    def test_the_counts_of_a_set_of_trials(self):
        distances = [0.4, 0.9, 12.0, 18.0, 31.0, 40.0]
        outcomes = [True, False, True, True, False, False]
        counts = estimates.trial_counts(distances, outcomes, [True, True, False, False, False, False])
        self.assertEqual(
            counts, {"max_success_d_mm": 18.0, "min_failure_d_mm": 0.9, "zero_k": 1, "zero_n": 2}
        )

    def test_trials_without_a_success_a_failure_or_an_anchor(self):
        counts = estimates.trial_counts([5.0, 9.0], [True, True], [False, False])
        self.assertEqual(counts, {"max_success_d_mm": 9.0, "min_failure_d_mm": None, "zero_k": 0, "zero_n": 0})
        with self.assertRaises(ValueError):
            estimates.trial_counts([5.0, 9.0], [True, True], [False])


class PlateauCheckTests(unittest.TestCase):
    """The same trials fitted with a longer list of lapse values (protocol section 10)."""

    WIDE = COARSE._replace(lapse_values=np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]))

    def low_plateau(self):
        """Trials of a condition that succeeds two times in three near a demonstration, and never beyond 22 mm."""
        distances, outcomes = [], []
        for index in range(36):
            distances.append(0.6 * index)
            outcomes.append(index % 3 != 0)
        for index in range(12):
            distances.append(24.0 + 2.0 * index)
            outcomes.append(False)
        return distances, outcomes

    def test_a_high_plateau_gives_two_fits_that_agree(self):
        distances, outcomes = trials(seed=3, n=60, r50_mm=20.0, width_mm=2.0)
        main = estimates.fit(distances, outcomes, COARSE).summary(0.95)
        check = estimates.plateau_check(main.r50_median_mm, estimates.fit(distances, outcomes, self.WIDE), 5.0)
        self.assertLessEqual(abs(check["median_shift_mm"]), 2.0)
        self.assertFalse(check["plateau_in_doubt"])
        self.assertEqual(check["lapse_values"], [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50])
        self.assertEqual(check["limit_mm"], 5.0)

    def test_a_low_plateau_pulls_the_short_list_s_radius_inward_and_is_called_in_doubt(self):
        distances, outcomes = self.low_plateau()
        main = estimates.fit(distances, outcomes, COARSE).summary(0.95)
        wide = estimates.fit(distances, outcomes, self.WIDE)
        check = estimates.plateau_check(main.r50_median_mm, wide, 5.0)
        told = wide.summary(0.95)
        self.assertEqual(
            (check["r50_median_mm"], check["r50_lo_mm"], check["r50_hi_mm"]),
            (told.r50_median_mm, told.r50_low_mm, told.r50_high_mm),
        )
        self.assertEqual((check["open_below"], check["open_above"]), (told.reaches_lower_edge, told.reaches_upper_edge))
        self.assertAlmostEqual(check["median_shift_mm"], told.r50_median_mm - main.r50_median_mm, places=9)
        self.assertGreater(check["median_shift_mm"], 5.0, "the longer list puts the radius farther out")
        self.assertTrue(check["plateau_in_doubt"])
        self.assertGreater(check["lapse_mean"], 0.2, "and explains the failures near a demonstration by the plateau")
        self.assertAlmostEqual(check["lapse_mean"], told.lapse_mean, places=12)

    def test_a_shift_of_exactly_the_limit_is_not_more_than_it(self):
        distances, outcomes = self.low_plateau()
        wide = estimates.fit(distances, outcomes, self.WIDE)
        median = wide.summary(0.95).r50_median_mm
        for shift, doubted in ((4.0, False), (6.0, True), (-6.0, True), (-4.0, False)):
            check = estimates.plateau_check(median - shift, wide, 5.0)
            self.assertEqual((check["median_shift_mm"], check["plateau_in_doubt"]), (shift, doubted))
        on_the_line = estimates.plateau_check(median - 6.0, wide, 6.0)
        self.assertFalse(on_the_line["plateau_in_doubt"])
        self.assertFalse(estimates.plateau_check(median + 6.0, wide, 6.0)["plateau_in_doubt"])

    def test_without_a_limit_nothing_is_called_in_doubt(self):
        distances, outcomes = self.low_plateau()
        check = estimates.plateau_check(0.0, estimates.fit(distances, outcomes, self.WIDE), None)
        self.assertEqual((check["limit_mm"], check["plateau_in_doubt"]), (None, None))
        self.assertGreater(check["median_shift_mm"], 5.0)


class LayoutOutcomeTests(unittest.TestCase):
    """The rule of prediction 3: the exact paired test gives p below 0.05 with the spread policy ahead."""

    def test_the_prediction_holds_below_the_level_with_the_spread_policy_ahead(self):
        self.assertEqual(estimates.LAYOUT_P_BELOW, 0.05)
        self.assertEqual(estimates.layout_outcome(mcnemar_exact(8, 1), 16, 9), "spread ahead")
        self.assertEqual(estimates.layout_outcome(0.049, 12, 6), "spread ahead")

    def test_the_level_itself_and_anything_above_it_do_not_decide(self):
        self.assertEqual(estimates.layout_outcome(0.05, 12, 6), "not decided")
        self.assertEqual(estimates.layout_outcome(mcnemar_exact(5, 1), 14, 10), "not decided")
        self.assertEqual(estimates.layout_outcome(1.0, 10, 10), "not decided")
        self.assertEqual(estimates.layout_outcome(None, 0, 0), "not decided")

    def test_a_test_that_decides_for_the_clustered_policy_is_said_to(self):
        self.assertEqual(estimates.layout_outcome(mcnemar_exact(1, 8), 9, 16), "clustered ahead")

    def test_six_discordant_pairs_one_way_are_the_fewest_that_decide(self):
        self.assertEqual(estimates.layout_outcome(mcnemar_exact(6, 0), 6, 0), "spread ahead")
        self.assertEqual(estimates.layout_outcome(mcnemar_exact(5, 0), 5, 0), "not decided")


class SmallHelperTests(unittest.TestCase):
    def test_the_mean_success_at_some_errors_is_the_curve_averaged_over_them(self):
        posterior = estimates.fit(*trials(seed=31, n=34, r50_mm=20.0, width_mm=3.0), GRIDS)
        errors = [0.5, 1.0, 8.0, 19.0]
        expected = sum(float(posterior.predict(error)) for error in errors) / 4
        self.assertAlmostEqual(estimates.mean_success_at(posterior, errors), expected, places=12)
        self.assertIsNone(estimates.mean_success_at(posterior, []))
        self.assertGreater(estimates.mean_success_at(posterior, [1.0]), estimates.mean_success_at(posterior, [19.0]))

    def test_median_and_percentile(self):
        self.assertEqual(estimates.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(estimates.median([4.0, 1.0, 2.0, 3.0]), 2.5)
        self.assertIsNone(estimates.median([]))
        values = np.random.default_rng(6).uniform(0.0, 20.0, size=20).tolist()
        for percent in (0.0, 50.0, 95.0, 100.0):
            self.assertAlmostEqual(
                estimates.percentile(values, percent), float(np.percentile(values, percent)), places=12
            )
        self.assertEqual(estimates.percentile([7.0], 95.0), 7.0)
        self.assertIsNone(estimates.percentile([], 95.0))
        with self.assertRaises(ValueError):
            estimates.percentile([1.0], 101.0)

    def test_numbers_are_kept_to_nine_significant_digits(self):
        self.assertEqual(estimates.stable(21.0), 21.0)
        self.assertEqual(estimates.stable(0.1 + 0.2), 0.3)
        self.assertEqual(estimates.stable(1.0 / 3.0), 0.333333333)
        self.assertEqual(estimates.stable(-12345.678912345), -12345.6789)
        self.assertEqual(estimates.stable(estimates.stable(math.pi)), estimates.stable(math.pi))
        # Two sums that differ in their last digits, as two builds of NumPy make them differ, are one number.
        self.assertEqual(estimates.stable(0.9981261823055716), estimates.stable(0.9981261823055717))

    def test_every_pair_of_small_grid_values_is_weighted_once(self):
        first = fit_threshold([5.0, 12.0, 18.0, 26.0], [True, True, False, False], **SMALL)
        second = fit_threshold([5.0, 12.0], [True, False], **SMALL)
        values, probabilities = estimates.difference_distribution(first, second)
        expected: dict[float, float] = {}
        pairs = itertools.product(zip(*first.r50_marginal(), strict=True), zip(*second.r50_marginal(), strict=True))
        for (a, pa), (b, pb) in pairs:
            expected[float(a - b)] = expected.get(float(a - b), 0.0) + float(pa * pb)
        self.assertEqual(values.tolist(), sorted(expected))
        self.assertTrue(
            np.allclose(probabilities, [expected[value] for value in values.tolist()], rtol=0.0, atol=1e-15)
        )


if __name__ == "__main__":
    unittest.main()
