"""What a paired workspace test can tell apart, computed exactly from each placement's two success probabilities."""

import itertools
import unittest

import numpy as np

from tolerance_radius import config, geometry, workspace_power
from tolerance_radius.intervals import mcnemar_exact

PROTOCOL = config.load_protocol()
GRID = geometry.Grid.from_protocol(PROTOCOL)


def brute_force_power(p_first, p_second, alpha):
    """The same power by listing every outcome of every pair."""
    cells = []
    for a, b in zip(p_first, p_second, strict=True):
        cells.append(((1, 0, a * (1 - b)), (0, 1, b * (1 - a)), (0, 0, a * b + (1 - a) * (1 - b))))
    power = 0.0
    for outcome in itertools.product(*cells):
        only_first = sum(o[0] for o in outcome)
        only_second = sum(o[1] for o in outcome)
        if mcnemar_exact(only_first, only_second) < alpha:
            power += float(np.prod([o[2] for o in outcome]))
    return power


class DiscordantDistributionTests(unittest.TestCase):
    def test_it_is_a_distribution_over_the_two_counts(self):
        pmf = workspace_power.discordant_distribution([0.2, 0.9, 0.5], [0.7, 0.9, 0.1])
        self.assertEqual(pmf.shape, (4, 4))
        self.assertAlmostEqual(float(pmf.sum()), 1.0, places=12)
        self.assertTrue((pmf >= 0).all())
        # Three pairs cannot give more than three discordant ones.
        self.assertEqual(float(sum(pmf[a, b] for a in range(4) for b in range(4) if a + b > 3)), 0.0)

    def test_one_pair(self):
        pmf = workspace_power.discordant_distribution([0.25], [0.6])
        self.assertAlmostEqual(float(pmf[1, 0]), 0.25 * 0.4)  # only the first succeeded
        self.assertAlmostEqual(float(pmf[0, 1]), 0.6 * 0.75)
        self.assertAlmostEqual(float(pmf[0, 0]), 0.25 * 0.6 + 0.75 * 0.4)

    def test_lengths_must_agree_and_probabilities_must_be_probabilities(self):
        with self.assertRaises(ValueError):
            workspace_power.discordant_distribution([0.5, 0.5], [0.5])
        with self.assertRaises(ValueError):
            workspace_power.discordant_distribution([1.2], [0.5])


class PairedPowerTests(unittest.TestCase):
    def test_against_listing_every_outcome(self):
        rng = np.random.default_rng(3)
        for n in (1, 4, 7, 9):
            p_first, p_second = rng.random(n), rng.random(n)
            with self.subTest(n=n):
                self.assertAlmostEqual(workspace_power.paired_power(p_first, p_second),
                                       brute_force_power(p_first, p_second, 0.05), places=10)

    def test_equal_policies_are_rarely_told_apart_and_opposite_ones_always(self):
        same = np.full(20, 0.5)
        self.assertLess(workspace_power.paired_power(same, same), 0.05)
        self.assertAlmostEqual(workspace_power.paired_power(np.zeros(20), np.ones(20)), 1.0)
        # Six discordant pairs all one way are the fewest that reach p < 0.05.
        self.assertAlmostEqual(workspace_power.paired_power(np.zeros(6), np.ones(6)), 1.0)
        self.assertAlmostEqual(workspace_power.paired_power(np.zeros(5), np.ones(5)), 0.0)

    def test_more_pairs_at_the_same_rates_give_more_power(self):
        powers = [workspace_power.paired_power(np.full(n, 0.4), np.full(n, 0.9)) for n in (10, 20, 30)]
        self.assertEqual(powers, sorted(powers))
        self.assertGreater(powers[1], 0.75)  # `simulate.paired_power(20, 0.4, 0.9)` gives 0.88 for the same case


class DesignTests(unittest.TestCase):
    def test_a_design_draws_its_numbers_from_its_regions(self):
        design = (("trained_cells", 13), ("held_out_cells", 13))
        points = workspace_power.design_placements(GRID, design, seed=5)
        self.assertEqual(len(points), 26)
        self.assertEqual(sum(GRID.in_region(point, "trained_cells") for point in points), 13)
        self.assertEqual(points, workspace_power.design_placements(GRID, design, seed=5))
        self.assertNotEqual(points, workspace_power.design_placements(GRID, design, seed=6))

    def test_the_lookup_probability_follows_the_distance_to_the_nearest_demonstration(self):
        demos = np.array([[25.0, 25.0], [125.0, 75.0]])
        points = [geometry.Point(25.0, 25.0), geometry.Point(125.0, 95.0), geometry.Point(75.0, 50.0)]
        p = workspace_power.lookup_probabilities(points, demos, r50_mm=20.0, width_mm=1.0, lapse=0.02, floor=0.03)
        self.assertAlmostEqual(float(p[0]), 0.98, places=3)                      # on a demonstration: the plateau
        self.assertAlmostEqual(float(p[1]), 0.03 + (1 - 0.02 - 0.03) / 2, places=6)  # exactly at the radius: halfway
        self.assertAlmostEqual(float(p[2]), 0.03, places=3)                      # far from both: the floor

    def test_the_table_has_one_row_for_every_design_and_truth(self):
        designs = {"a": (("trained_cells", 6),), "b": (("trained_cells", 3), ("held_out_cells", 3))}
        truths = [(15.0, 3.0), (25.0, 3.0)]
        rows = workspace_power.compare_designs(PROTOCOL, designs, truths, draws=3, seed=1)
        self.assertEqual({(r["design"], r["r50_mm"], r["width_mm"]) for r in rows},
                         {(d, r, w) for d in designs for r, w in truths})
        for row in rows:
            self.assertTrue(0.0 <= row["power"] <= 1.0)
            self.assertTrue(0.0 <= row["expected_clustered"] <= row["pairs"])
        self.assertEqual(rows, workspace_power.compare_designs(PROTOCOL, designs, truths, draws=3, seed=1))


if __name__ == "__main__":
    unittest.main()
