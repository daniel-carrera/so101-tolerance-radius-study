"""The lookup prediction: a placement succeeds as far as its distance to the nearest demonstration allows."""

import math
import unittest
from dataclasses import replace

import numpy as np

from fixtures_demos import GRID, PROTOCOL, audit, demos_at
from tolerance_radius import coverage, geometry, layouts
from tolerance_radius.geometry import GeometryError
from tolerance_radius.schema import Point

TRAINED_AREA = GRID.region_area_mm2("trained_cells")
CLUSTERED_PLAN = layouts.planned_demo_starts(layouts.clustered_layout(PROTOCOL))
SPREAD_PLAN = layouts.planned_demo_starts(layouts.spread_layout(PROTOCOL))


def logistic(distance_mm):
    """A response curve like the one the analysis fits: half its plateau at 20 mm."""
    return 0.95 / (1.0 + np.exp((distance_mm - 20.0) / 4.0))


class CoveredFractionTests(unittest.TestCase):
    def test_a_radius_of_zero_covers_nothing(self):
        for demos in (CLUSTERED_PLAN, SPREAD_PLAN, audit("clustered"), audit("spread")):
            self.assertEqual(coverage.covered_fraction(demos, GRID, 0.0), 0.0)

    def test_36_mm_around_the_cell_centres_covers_all_trained_cells_and_35_mm_does_not(self):
        self.assertEqual(coverage.covered_fraction(CLUSTERED_PLAN, GRID, 36.0), 1.0)
        nearly = coverage.covered_fraction(CLUSTERED_PLAN, GRID, 35.0)
        self.assertLess(nearly, 1.0)
        self.assertGreater(nearly, 0.999)

    def test_one_small_disc_covers_its_share_of_the_area(self):
        demos = demos_at((125.0, 75.0))
        for radius in (5.0, 10.0, 20.0):
            disc = math.pi * radius**2
            with self.subTest(radius=radius):
                share = disc / TRAINED_AREA
                self.assertAlmostEqual(coverage.covered_fraction(demos, GRID, radius), share, delta=0.01 * share)
                whole = coverage.covered_fraction(demos, GRID, radius, region="test_region")
                self.assertAlmostEqual(whole, disc / 33600.0, delta=0.01 * disc / 33600.0)

    def test_only_the_part_of_a_disc_inside_the_region_counts(self):
        in_a_trained_cell = demos_at((125.0, 75.0))  # the centre of C2
        self.assertEqual(coverage.covered_fraction(in_a_trained_cell, GRID, 10.0, region="held_out_cells"), 0.0)
        on_a_border = demos_at((100.0, 75.0))  # between B2, held out, and C2
        half_disc = math.pi * 10.0**2 / 2.0
        trained = coverage.covered_fraction(on_a_border, GRID, 10.0)
        held_out = coverage.covered_fraction(on_a_border, GRID, 10.0, region="held_out_cells")
        self.assertAlmostEqual(trained, half_disc / TRAINED_AREA, delta=0.02 * half_disc / TRAINED_AREA)
        self.assertAlmostEqual(held_out, half_disc / 9050.0, delta=0.02 * half_disc / 9050.0)

    def test_the_margin_is_not_part_of_any_region(self):
        in_the_margin = demos_at((0.0, 75.0))  # on the grid's outer edge: 5 mm outside the test region
        self.assertEqual(coverage.covered_fraction(in_the_margin, GRID, 5.0, region="test_region"), 0.0)
        self.assertGreater(coverage.covered_fraction(in_the_margin, GRID, 8.0, region="test_region"), 0.0)

    def test_where_two_discs_overlap_the_area_counts_once(self):
        demos = demos_at((120.0, 75.0), (130.0, 75.0))
        lens = 2 * 10.0**2 * math.acos(0.5) - 5.0 * math.sqrt(300.0)
        union = 2 * math.pi * 10.0**2 - lens
        share = union / TRAINED_AREA
        self.assertAlmostEqual(coverage.covered_fraction(demos, GRID, 10.0), share, delta=0.01 * share)

    def test_coverage_grows_with_the_radius_and_is_complete_once_the_largest_gap_is_reached(self):
        demos = audit("spread")
        rects = GRID.region_rects("trained_cells")
        gap = geometry.largest_nearest_distance(geometry.trained_positions(demos), rects)[0]
        radii = (2.0, 6.0, 10.0, 14.0, gap - 1.0, gap + 0.01)
        fractions = [coverage.covered_fraction(demos, GRID, radius) for radius in radii]
        self.assertEqual(fractions, sorted(fractions))
        self.assertLess(fractions[-2], 1.0)
        self.assertEqual(fractions[-1], 1.0)

    def test_at_the_same_radius_the_spread_set_covers_more_than_the_clustered_set(self):
        clustered = coverage.covered_fraction(CLUSTERED_PLAN, GRID, 15.0)
        spread = coverage.covered_fraction(SPREAD_PLAN, GRID, 15.0)
        # Eleven whole discs of 15 mm around the cell centres, each inside its own cell.
        self.assertAlmostEqual(clustered, 11 * math.pi * 15.0**2 / TRAINED_AREA, delta=0.003)
        self.assertGreater(spread, 2.5 * clustered)

    def test_an_episode_left_out_of_training_covers_nothing(self):
        demos = [*demos_at((125.0, 75.0)), replace(demos_at((25.0, 75.0))[0], episode_index=1, trained=False)]
        self.assertEqual(coverage.covered_fraction(demos, GRID, 10.0), coverage.covered_fraction(demos[:1], GRID, 10.0))

    def test_a_finer_lattice_changes_the_answer_little(self):
        demos = audit("spread")
        coarse = coverage.covered_fraction(demos, GRID, 12.0)
        fine = coverage.covered_fraction(demos, GRID, 12.0, step_mm=0.25)
        self.assertAlmostEqual(coarse, fine, delta=0.003)
        self.assertNotEqual(coarse, fine)

    def test_a_sample_point_counts_for_the_area_it_stands_for(self):
        # With sample points 25 mm apart in whole cells and 22.5 mm apart in the cells cut by the margin, a radius of
        # 20 mm about the centre of C2 takes in the four sample points of C2 and no other: 2500 mm2.
        fraction = coverage.covered_fraction(demos_at((125.0, 75.0)), GRID, 20.0, step_mm=30.0)
        self.assertAlmostEqual(fraction, 2500.0 / TRAINED_AREA, places=12)

    def test_what_cannot_be_computed_is_refused(self):
        with self.assertRaises(ValueError):
            coverage.covered_fraction(CLUSTERED_PLAN, GRID, -1.0)
        with self.assertRaises(ValueError):
            coverage.covered_fraction(CLUSTERED_PLAN, GRID, math.nan)
        with self.assertRaises(GeometryError):
            coverage.covered_fraction(CLUSTERED_PLAN, GRID, 10.0, region="everywhere")
        with self.assertRaises(GeometryError):
            coverage.covered_fraction([*CLUSTERED_PLAN, *SPREAD_PLAN], GRID, 10.0)


class ExpectedSuccessTests(unittest.TestCase):
    POINTS = (Point(25.0, 75.0), Point(40.0, 75.0), Point(75.0, 75.0), Point(5.0, 5.0))

    def test_it_is_the_curve_at_the_distance_to_the_nearest_demonstration(self):
        demos = audit("clustered")
        result = coverage.expected_success(self.POINTS, demos, logistic)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        for point, value in zip(self.POINTS, result.tolist(), strict=True):
            self.assertAlmostEqual(value, float(logistic(geometry.nearest(point, demos)[1])), places=12)

    def test_success_falls_with_distance_under_a_falling_curve(self):
        # The centre of a trained cell, 15 mm from it, the centre of a held-out cell, the corner of the test region.
        result = coverage.expected_success(self.POINTS, CLUSTERED_PLAN, logistic).tolist()
        self.assertEqual(result, sorted(result, reverse=True))
        self.assertAlmostEqual(result[0], float(logistic(0.0)))
        self.assertAlmostEqual(result[2], float(logistic(50.0)))

    def test_points_may_be_given_as_an_array_of_x_and_y(self):
        as_points = coverage.expected_success(self.POINTS, CLUSTERED_PLAN, logistic)
        xy = np.array([[p.x_mm, p.y_mm] for p in self.POINTS])
        as_array = coverage.expected_success(xy, CLUSTERED_PLAN, logistic)
        self.assertEqual(as_points.tolist(), as_array.tolist())

    def test_a_curve_that_takes_one_distance_at_a_time_works_too(self):
        def one_at_a_time(distance_mm):
            return 1.0 if distance_mm < 20.0 else 0.0  # fails on an array: its truth value is ambiguous

        result = coverage.expected_success(self.POINTS, CLUSTERED_PLAN, one_at_a_time)
        self.assertEqual(result.tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_a_curve_that_gives_one_number_for_everything_works_too(self):
        result = coverage.expected_success(self.POINTS, CLUSTERED_PLAN, lambda distance_mm: 0.25)
        self.assertEqual(result.tolist(), [0.25] * 4)

    def test_no_points_no_probabilities(self):
        self.assertEqual(coverage.expected_success([], CLUSTERED_PLAN, logistic).shape, (0,))

    def test_a_curve_that_does_not_give_probabilities_is_refused(self):
        for bad in (lambda d: d, lambda d: -0.1 + 0.0 * d, lambda d: np.full(np.shape(d), np.nan)):
            with self.assertRaises(ValueError):
                coverage.expected_success(self.POINTS, CLUSTERED_PLAN, bad)

    def test_points_that_are_not_pairs_are_refused(self):
        with self.assertRaises(ValueError):
            coverage.expected_success(np.zeros((3, 3)), CLUSTERED_PLAN, logistic)

    def test_with_a_step_at_a_radius_its_mean_over_a_region_is_the_covered_fraction(self):
        demos = audit("spread")
        xs, ys, weights = GRID.region_lattice("trained_cells", 0.5)
        inside = coverage.expected_success(np.column_stack([xs, ys]), demos, lambda d: (d < 12.0).astype(float))
        mean = float((inside * weights).sum() / weights.sum())
        self.assertAlmostEqual(mean, coverage.covered_fraction(demos, GRID, 12.0), places=12)


if __name__ == "__main__":
    unittest.main()
