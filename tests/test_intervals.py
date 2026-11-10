"""Intervals and exact tests for counts: known values, defining properties, and a cross-check against SciPy."""

import math
import unittest

from tolerance_radius.intervals import binomial_two_sided, mcnemar_exact, normal_quantile, wilson_interval

try:  # SciPy is a cross-check only. The rig's interpreter has none, and these tests then skip.
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def normal_cdf(x: float) -> float:
    """The standard normal distribution function, from the standard library's error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


class NormalQuantileTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(normal_quantile(0.5), 0.0)
        self.assertAlmostEqual(normal_quantile(0.975), 1.959963984540054, places=12)
        self.assertAlmostEqual(normal_quantile(0.995), 2.5758293035489004, places=12)
        self.assertAlmostEqual(normal_quantile(0.8413447460685429), 1.0, places=12)

    def test_it_inverts_the_distribution_function(self):
        for p in (1e-10, 1e-4, 0.01, 0.025, 0.2, 0.5, 0.7, 0.975, 0.999, 1 - 1e-9):
            with self.subTest(p=p):
                self.assertAlmostEqual(normal_cdf(normal_quantile(p)) / p, 1.0, places=9)

    def test_it_is_antisymmetric(self):
        for p in (0.001, 0.025, 0.3):
            with self.subTest(p=p):
                self.assertAlmostEqual(normal_quantile(p), -normal_quantile(1.0 - p), places=12)

    def test_a_probability_outside_the_open_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normal_quantile(bad)


class WilsonIntervalTests(unittest.TestCase):
    def test_published_values_at_95_percent(self):
        for successes, trials, low, high in ((5, 10, 0.2366, 0.7634), (1, 10, 0.0179, 0.4042), (0, 10, 0.0, 0.2775)):
            with self.subTest(successes=successes, trials=trials):
                got_low, got_high = wilson_interval(successes, trials)
                self.assertAlmostEqual(got_low, low, places=4)
                self.assertAlmostEqual(got_high, high, places=4)

    def test_no_trials_says_nothing(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_all_failures_start_at_zero_and_all_successes_end_at_one(self):
        self.assertEqual(wilson_interval(0, 7)[0], 0.0)
        self.assertEqual(wilson_interval(7, 7)[1], 1.0)
        self.assertLess(wilson_interval(0, 7)[1], 1.0)
        self.assertGreater(wilson_interval(7, 7)[0], 0.0)

    def test_each_bound_is_where_the_score_test_just_rejects(self):
        # The Wilson interval is the set of proportions p with |observed - p| <= z * sqrt(p (1 - p) / n).
        z = normal_quantile(0.975)
        for successes, trials in ((3, 20), (11, 34), (1, 4), (19, 20)):
            observed = successes / trials
            for bound in wilson_interval(successes, trials):
                with self.subTest(successes=successes, trials=trials, bound=bound):
                    allowed = z * math.sqrt(bound * (1 - bound) / trials)
                    self.assertAlmostEqual(abs(observed - bound), allowed, places=12)

    def test_it_contains_the_observed_share_and_mirrors_with_the_outcome(self):
        for successes, trials in ((0, 5), (2, 5), (5, 5), (17, 40)):
            with self.subTest(successes=successes, trials=trials):
                low, high = wilson_interval(successes, trials)
                self.assertTrue(0.0 <= low <= successes / trials <= high <= 1.0)
                mirrored_low, mirrored_high = wilson_interval(trials - successes, trials)
                self.assertAlmostEqual(low, 1.0 - mirrored_high, places=12)
                self.assertAlmostEqual(high, 1.0 - mirrored_low, places=12)

    def test_a_higher_level_gives_a_wider_interval_and_more_trials_a_narrower_one(self):
        low_90, high_90 = wilson_interval(6, 20, level=0.90)
        low_99, high_99 = wilson_interval(6, 20, level=0.99)
        self.assertTrue(low_99 < low_90 < high_90 < high_99)
        low_small, high_small = wilson_interval(6, 20)
        low_large, high_large = wilson_interval(60, 200)
        self.assertLess(high_large - low_large, high_small - low_small)

    def test_impossible_counts_and_levels_are_refused(self):
        bad = ((-1, 5, 0.95), (6, 5, 0.95), (2, -1, 0.95), (2, 5, 0.0), (2, 5, 1.0), (2.5, 5, 0.95))
        for successes, trials, level in bad:
            with self.subTest(successes=successes, trials=trials, level=level), self.assertRaises(ValueError):
                wilson_interval(successes, trials, level)


class BinomialTwoSidedTests(unittest.TestCase):
    def test_values_worked_by_hand(self):
        # 4 trials at p = 1/2: the outcomes 0..4 have probabilities 1, 4, 6, 4, 1 sixteenths.
        self.assertEqual(binomial_two_sided(0, 4, 0.5), 2 / 16)
        self.assertEqual(binomial_two_sided(1, 4, 0.5), 10 / 16)
        self.assertEqual(binomial_two_sided(2, 4, 0.5), 1.0)
        # 3 trials at p = 1/4: the outcomes 0..3 have probabilities 27, 27, 9, 1 sixty-fourths.
        self.assertEqual(binomial_two_sided(3, 3, 0.25), 1 / 64)
        self.assertEqual(binomial_two_sided(2, 3, 0.25), 10 / 64)
        self.assertEqual(binomial_two_sided(1, 3, 0.25), 1.0)
        self.assertEqual(binomial_two_sided(0, 3, 0.25), 1.0)

    def test_no_trials_is_no_evidence(self):
        self.assertEqual(binomial_two_sided(0, 0, 0.3), 1.0)

    def test_an_outcome_the_hypothesis_forbids_has_probability_zero(self):
        self.assertEqual(binomial_two_sided(1, 5, 0.0), 0.0)
        self.assertEqual(binomial_two_sided(0, 5, 0.0), 1.0)
        self.assertEqual(binomial_two_sided(4, 5, 1.0), 0.0)
        self.assertEqual(binomial_two_sided(5, 5, 1.0), 1.0)

    def test_it_is_a_probability_and_is_largest_at_the_most_likely_outcome(self):
        values = [binomial_two_sided(k, 30, 0.3) for k in range(31)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(max(values), 1.0)
        self.assertEqual(values.index(1.0), 9)  # 30 x 0.3

    def test_large_counts_do_not_overflow(self):
        self.assertAlmostEqual(binomial_two_sided(1500, 3000, 0.5), 1.0, places=12)
        self.assertLess(binomial_two_sided(1300, 3000, 0.5), 1e-10)

    def test_impossible_inputs_are_refused(self):
        bad = ((-1, 5, 0.5), (6, 5, 0.5), (2, 5, -0.1), (2, 5, 1.1), (2.0, 5, 0.5), (2, 5, float("nan")))
        for successes, trials, p in bad:
            with self.subTest(successes=successes, trials=trials, p=p), self.assertRaises(ValueError):
                binomial_two_sided(successes, trials, p)


class McNemarExactTests(unittest.TestCase):
    def test_values_worked_by_hand(self):
        self.assertEqual(mcnemar_exact(0, 5), 2 / 32)
        self.assertEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertEqual(mcnemar_exact(7, 0), 2 / 128)
        self.assertEqual(mcnemar_exact(1, 9), 2 * (1 + 10) / 1024)
        self.assertEqual(mcnemar_exact(2, 10), 2 * (1 + 12 + 66) / 4096)
        self.assertEqual(mcnemar_exact(4, 9), 2 * (1 + 13 + 78 + 286 + 715) / 8192)

    def test_no_discordant_pairs_and_equal_counts_give_one(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)

    def test_it_does_not_matter_which_policy_is_called_first(self):
        for only_first, only_second in ((0, 6), (2, 9), (5, 7)):
            with self.subTest(only_first=only_first, only_second=only_second):
                self.assertEqual(mcnemar_exact(only_first, only_second), mcnemar_exact(only_second, only_first))

    def test_six_one_sided_pairs_are_the_fewest_that_reach_five_percent(self):
        self.assertGreaterEqual(mcnemar_exact(0, 5), 0.05)
        self.assertLess(mcnemar_exact(0, 6), 0.05)

    def test_negative_counts_are_refused(self):
        with self.assertRaises(ValueError):
            mcnemar_exact(-1, 3)


@unittest.skipIf(scipy_stats is None, "SciPy is not installed in this environment")
class ScipyCrossCheckTests(unittest.TestCase):
    def test_normal_quantile_matches_scipy(self):
        for p in (1e-12, 1e-6, 0.001, 0.025, 0.1, 0.5, 0.75, 0.975, 0.9999, 1 - 1e-12):
            with self.subTest(p=p):
                expected = scipy_stats.norm.ppf(p)
                self.assertAlmostEqual(normal_quantile(p), expected, delta=1e-12 * max(1.0, abs(expected)))

    def test_wilson_interval_matches_scipy(self):
        for trials in (1, 4, 6, 20, 34, 200):
            for successes in sorted({0, 1, trials // 3, trials // 2, trials - 1, trials}):
                for level in (0.80, 0.95, 0.99):
                    with self.subTest(successes=successes, trials=trials, level=level):
                        expected = scipy_stats.binomtest(successes, trials).proportion_ci(level, method="wilson")
                        low, high = wilson_interval(successes, trials, level)
                        self.assertAlmostEqual(low, expected.low, places=12)
                        self.assertAlmostEqual(high, expected.high, places=12)

    def test_binomial_two_sided_matches_scipy(self):
        for trials in (1, 5, 12, 20, 33):
            for p in (0.5, 0.25, 0.3, 0.07, 0.9):
                for successes in range(trials + 1):
                    with self.subTest(successes=successes, trials=trials, p=p):
                        expected = scipy_stats.binomtest(successes, trials, p).pvalue
                        self.assertAlmostEqual(binomial_two_sided(successes, trials, p), expected, places=12)

    def test_mcnemar_exact_matches_scipy_on_every_table_of_up_to_24_pairs(self):
        for only_first in range(25):
            for only_second in range(25 - only_first):
                if only_first + only_second == 0:
                    continue
                with self.subTest(only_first=only_first, only_second=only_second):
                    expected = scipy_stats.binomtest(only_first, only_first + only_second, 0.5).pvalue
                    self.assertAlmostEqual(mcnemar_exact(only_first, only_second), expected, places=12)


if __name__ == "__main__":
    unittest.main()
