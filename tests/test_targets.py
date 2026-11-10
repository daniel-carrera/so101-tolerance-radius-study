"""Turning a request (a distance, a replayed demonstration, a region) into a place to put the cube."""

import math
import unittest
from collections import Counter
from dataclasses import replace

import numpy as np

from fixtures_demos import GRID, audit, demos_at
from tolerance_radius import geometry, targets
from tolerance_radius.geometry import GeometryError
from tolerance_radius.schema import Point
from tolerance_radius.seeds import derived_rng

MATCH = 0.5


def distance(point: Point, demos) -> float:
    return geometry.nearest(point, demos)[1]


class MaxFeasibleLevelTests(unittest.TestCase):
    def test_one_demonstration_in_the_middle_leaves_the_corners_of_the_test_region_farthest(self):
        self.assertAlmostEqual(targets.max_feasible_level(demos_at((125.0, 75.0)), GRID), math.hypot(120.0, 70.0))

    def test_cell_centres_leave_the_corners_of_the_two_held_out_corner_cells_farthest(self):
        centres = demos_at(*[(GRID.cell_centre(c).x_mm, GRID.cell_centre(c).y_mm) for c in GRID.trained_cells])
        # From (5, 5), the nearest centres are those of A2 and B1: 20 mm one way and 70 mm the other.
        self.assertAlmostEqual(targets.max_feasible_level(centres, GRID), math.hypot(20.0, 70.0))

    def test_it_matches_a_fine_lattice_search(self):
        for dataset in ("clustered", "spread"):
            demos = audit(dataset)
            xs, ys, _ = GRID.region_lattice("test_region", 0.5)
            lattice = float(geometry.nearest_distance_field(xs, ys, demos).max())
            exact = targets.max_feasible_level(demos, GRID)
            with self.subTest(dataset=dataset):
                self.assertGreaterEqual(exact, lattice - 1e-9)
                self.assertLessEqual(exact, lattice + 0.5 * math.sqrt(0.5) + 1e-9)

    def test_an_episode_left_out_of_training_does_not_count(self):
        demos = [*demos_at((125.0, 75.0)), replace(demos_at((6.0, 6.0))[0], episode_index=1, trained=False)]
        self.assertAlmostEqual(targets.max_feasible_level(demos, GRID), math.hypot(120.0, 70.0))


class PlacementAtDistanceTests(unittest.TestCase):
    def test_a_placement_has_the_requested_distance_and_lies_in_the_test_region(self):
        cases = {"clustered": (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0), "spread": (0.0, 5.0, 10.0, 20.0, 30.0, 45.0)}
        for dataset, levels in cases.items():
            demos = audit(dataset)
            for level in levels:
                rng = derived_rng(1, dataset, int(level))
                for _ in range(12):
                    point = targets.placement_at_distance(level, demos, GRID, rng, MATCH)
                    with self.subTest(dataset=dataset, level=level):
                        self.assertIsNotNone(point)
                        self.assertTrue(GRID.in_test_region(point))
                        self.assertLessEqual(abs(distance(point, demos) - level), MATCH + 1e-9)

    def test_a_distance_that_no_point_of_the_test_region_has_gives_nothing(self):
        self.assertIsNone(targets.placement_at_distance(80.0, audit("clustered"), GRID, derived_rng(1, "x"), MATCH))
        middle = demos_at((125.0, 75.0))
        self.assertIsNone(targets.placement_at_distance(150.0, middle, GRID, derived_rng(1, "x"), MATCH))

    def test_the_largest_distance_can_just_be_placed_and_a_little_more_cannot(self):
        for dataset in ("clustered", "spread"):
            demos = audit(dataset)
            largest = targets.max_feasible_level(demos, GRID)
            rng = derived_rng(2, dataset)
            with self.subTest(dataset=dataset):
                for level in (largest, largest + MATCH - 1e-6):
                    point = targets.placement_at_distance(level, demos, GRID, rng, MATCH)
                    self.assertIsNotNone(point)
                    self.assertTrue(GRID.in_test_region(point))
                    self.assertLessEqual(abs(distance(point, demos) - level), MATCH + 1e-9)
                self.assertIsNone(targets.placement_at_distance(largest + MATCH + 1e-3, demos, GRID, rng, MATCH))

    def test_a_distance_below_the_smallest_in_the_test_region_gives_nothing(self):
        # The only demonstration is measured in the margin, 3 mm outside the test region.
        demos = demos_at((2.0, 75.0))
        rng = derived_rng(3, "margin")
        self.assertIsNone(targets.placement_at_distance(1.0, demos, GRID, rng, MATCH))
        self.assertIsNone(targets.placement_at_distance(2.4, demos, GRID, rng, MATCH))
        for level in (2.5 + 1e-6, 2.6, 3.0, 10.0):
            point = targets.placement_at_distance(level, demos, GRID, rng, MATCH)
            self.assertIsNotNone(point)
            self.assertTrue(GRID.in_test_region(point))
            self.assertLessEqual(abs(distance(point, demos) - level), MATCH + 1e-9)

    def test_the_same_stream_gives_the_same_placement_and_another_stream_another(self):
        demos = audit("clustered")
        first = targets.placement_at_distance(20.0, demos, GRID, derived_rng(4, "a"), MATCH)
        again = targets.placement_at_distance(20.0, demos, GRID, derived_rng(4, "a"), MATCH)
        other = targets.placement_at_distance(20.0, demos, GRID, derived_rng(4, "b"), MATCH)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_placements_are_spread_evenly_around_a_demonstration_and_across_the_width_of_the_band(self):
        demos = demos_at((125.0, 75.0))
        rng = derived_rng(5, "ring")
        points = [targets.placement_at_distance(30.0, demos, GRID, rng, MATCH) for _ in range(400)]
        quadrants = Counter((p.x_mm >= 125.0, p.y_mm >= 75.0) for p in points)
        self.assertEqual(len(quadrants), 4)
        self.assertGreater(min(quadrants.values()), 70)
        outer = sum(distance(p, demos) > 30.0 for p in points)
        self.assertTrue(150 < outer < 250, outer)
        self.assertGreater(max(distance(p, demos) for p in points), 30.4)
        self.assertLess(min(distance(p, demos) for p in points), 29.6)

    def test_every_part_of_the_table_at_that_distance_gets_its_share(self):
        demos = audit("clustered")
        rng = derived_rng(6, "share")
        cells = Counter()
        for _ in range(660):
            point = targets.placement_at_distance(10.0, demos, GRID, rng, MATCH)
            cells[demos[geometry.nearest(point, demos)[0]].cell] += 1
        self.assertEqual(set(cells), set(GRID.trained_cells))
        self.assertGreater(min(cells.values()), 30)  # 60 expected around each of the 11 cell centres
        self.assertLess(max(cells.values()), 95)

    def test_an_episode_left_out_of_training_is_not_a_demonstration_to_keep_a_distance_from(self):
        trained = demos_at((125.0, 75.0))
        left_out = replace(demos_at((155.0, 75.0))[0], episode_index=1, trained=False)
        rng = derived_rng(7, "left-out")
        points = [targets.placement_at_distance(30.0, [*trained, left_out], GRID, rng, MATCH) for _ in range(200)]
        for point in points:
            self.assertLessEqual(abs(point.distance_to(Point(125.0, 75.0)) - 30.0), MATCH + 1e-9)
        self.assertTrue(any(point.distance_to(Point(155.0, 75.0)) < 5.0 for point in points))

    def test_demonstrations_of_two_sets_are_refused(self):
        demos = [*demos_at((125.0, 75.0)), *demos_at((25.0, 75.0), dataset="spread")]
        with self.assertRaises(GeometryError):
            targets.placement_at_distance(10.0, demos, GRID, derived_rng(1, "x"), MATCH)

    def test_the_level_and_the_tolerance_must_make_sense(self):
        demos = demos_at((125.0, 75.0))
        for level, match in ((-1.0, 0.5), (10.0, 0.0), (10.0, -0.5), (math.nan, 0.5), (math.inf, 0.5)):
            with self.subTest(level=level, match=match), self.assertRaises(ValueError):
                targets.placement_at_distance(level, demos, GRID, derived_rng(1, "x"), match)


class PlacementAroundTests(unittest.TestCase):
    def test_a_placement_is_at_the_requested_distance_from_the_demonstration_and_in_the_test_region(self):
        demos = audit("clustered")
        for demo in (demos[0], demos[7], demos[23], demos[41], demos[54]):
            for level in (2.0, 5.0, 15.0, 25.0, 40.0):
                rng = derived_rng(8, demo.episode_index, int(level))
                for _ in range(20):
                    point = targets.placement_around(demo, level, GRID, rng)
                    self.assertTrue(GRID.in_test_region(point))
                    self.assertAlmostEqual(point.distance_to(demo.measured), level, places=9)

    def test_distance_zero_is_the_demonstration_itself(self):
        demo = audit("clustered")[3]
        self.assertEqual(targets.placement_around(demo, 0.0, GRID, derived_rng(1, "x")), demo.measured)

    def test_every_direction_is_as_likely_when_the_whole_circle_fits(self):
        demo = demos_at((125.0, 75.0))[0]
        rng = derived_rng(9, "directions")
        points = [targets.placement_around(demo, 30.0, GRID, rng) for _ in range(400)]
        quadrants = Counter((p.x_mm >= 125.0, p.y_mm >= 75.0) for p in points)
        self.assertEqual(len(quadrants), 4)
        self.assertGreater(min(quadrants.values()), 70)

    def test_near_a_corner_only_the_directions_that_stay_inside_are_used_and_all_of_them(self):
        demo = demos_at((10.0, 10.0))[0]
        rng = derived_rng(10, "corner")
        angles = []
        for _ in range(500):
            point = targets.placement_around(demo, 30.0, GRID, rng)
            self.assertTrue(GRID.in_test_region(point))
            angles.append(math.degrees(math.atan2(point.y_mm - 10.0, point.x_mm - 10.0)))
        # Inside means x >= 5 and y >= 5: directions from -9.6 to 99.6 degrees.
        limit = math.degrees(math.asin(5.0 / 30.0))
        self.assertGreaterEqual(min(angles), -limit - 1e-6)
        self.assertLessEqual(max(angles), 90.0 + limit + 1e-6)
        self.assertLess(min(angles), -limit + 3.0)
        self.assertGreater(max(angles), 90.0 + limit - 3.0)
        self.assertAlmostEqual(sorted(angles)[250], 45.0, delta=8.0)

    def test_two_separate_arcs_are_both_used(self):
        # A circle of 100 mm about the middle of the table leaves the test region above and below.
        demo = demos_at((125.0, 75.0))[0]
        rng = derived_rng(11, "arcs")
        points = [targets.placement_around(demo, 100.0, GRID, rng) for _ in range(300)]
        for point in points:
            self.assertTrue(GRID.in_test_region(point))
            self.assertAlmostEqual(point.distance_to(demo.measured), 100.0, places=9)
        right = sum(point.x_mm > 125.0 for point in points)
        self.assertTrue(110 < right < 190, right)

    def test_a_circle_with_no_part_in_the_test_region_gives_nothing(self):
        middle = demos_at((125.0, 75.0))[0]
        self.assertIsNone(targets.placement_around(middle, 200.0, GRID, derived_rng(1, "x")))
        far_away = replace(middle, measured=Point(-100.0, -100.0))
        self.assertIsNone(targets.placement_around(far_away, 10.0, GRID, derived_rng(1, "x")))
        self.assertIsNone(targets.placement_around(far_away, 0.0, GRID, derived_rng(1, "x")))

    def test_the_same_stream_gives_the_same_placement(self):
        demo = audit("clustered")[12]
        first = targets.placement_around(demo, 25.0, GRID, derived_rng(12, "a"))
        self.assertEqual(first, targets.placement_around(demo, 25.0, GRID, derived_rng(12, "a")))
        self.assertNotEqual(first, targets.placement_around(demo, 25.0, GRID, derived_rng(12, "b")))

    def test_a_negative_distance_is_refused(self):
        with self.assertRaises(ValueError):
            targets.placement_around(audit("clustered")[0], -1.0, GRID, derived_rng(1, "x"))


class UniformPlacementTests(unittest.TestCase):
    def test_placements_all_fall_in_trained_cells_inside_the_test_region(self):
        points = targets.uniform_placements(GRID, 500, derived_rng(13, "uniform"))
        self.assertEqual(len(points), 500)
        for point in points:
            self.assertTrue(GRID.in_region(point, "trained_cells"))
            self.assertTrue(GRID.is_trained(GRID.cell_of(point)))
            self.assertTrue(GRID.in_test_region(point))

    def test_the_other_regions_can_be_drawn_from(self):
        for point in targets.uniform_placements(GRID, 200, derived_rng(13, "held-out"), region="held_out_cells"):
            self.assertTrue(GRID.in_region(point, "held_out_cells"))
            self.assertFalse(GRID.is_trained(GRID.cell_of(point)))
        whole = targets.uniform_placements(GRID, 400, derived_rng(13, "whole"), region="test_region")
        self.assertTrue(all(GRID.in_test_region(point) for point in whole))
        self.assertTrue(any(not GRID.is_trained(GRID.cell_of(point)) for point in whole))
        with self.assertRaises(GeometryError):
            targets.uniform_placements(GRID, 5, derived_rng(13, "x"), region="everywhere")

    def test_each_cell_gets_placements_in_proportion_to_its_area_inside_the_test_region(self):
        count = 40000  # enough to tell "by area" from "every cell equally often"
        cells = Counter(GRID.cell_of(p) for p in targets.uniform_placements(GRID, count, derived_rng(14, "area")))
        self.assertEqual(set(cells), set(GRID.trained_cells))
        total = GRID.region_area_mm2("trained_cells")
        for cell, area in (("C2", 2500.0), ("A2", 45.0 * 50.0), ("A3", 45.0 * 45.0), ("E1", 45.0 * 45.0)):
            expected = count * area / total
            with self.subTest(cell=cell):
                self.assertAlmostEqual(cells[cell], expected, delta=4.0 * math.sqrt(expected))

    def test_placements_fill_a_cell_evenly(self):
        points = [p for p in targets.uniform_placements(GRID, 6000, derived_rng(15, "fill")) if GRID.cell_of(p) == "C2"]
        xs = np.array([p.x_mm for p in points])
        ys = np.array([p.y_mm for p in points])
        self.assertAlmostEqual(float(xs.mean()), 125.0, delta=2.5)
        self.assertAlmostEqual(float(ys.mean()), 75.0, delta=2.5)
        self.assertLess(float(xs.min()), 102.0)
        self.assertGreater(float(xs.max()), 148.0)

    def test_the_same_stream_gives_the_same_placements(self):
        first = targets.uniform_placements(GRID, 20, derived_rng(16, "a"))
        self.assertEqual(first, targets.uniform_placements(GRID, 20, derived_rng(16, "a")))
        self.assertNotEqual(first, targets.uniform_placements(GRID, 20, derived_rng(16, "b")))

    def test_none_asked_for_none_given(self):
        self.assertEqual(targets.uniform_placements(GRID, 0, derived_rng(1, "x")), [])
        with self.assertRaises(ValueError):
            targets.uniform_placements(GRID, -1, derived_rng(1, "x"))

    def test_the_placements_are_pinned_so_that_every_environment_draws_the_same_ones(self):
        first, second = targets.uniform_placements(GRID, 2, derived_rng(20261007, "pinned"))
        self.assertEqual((first.x_mm, first.y_mm), (109.70204513386349, 108.08763002438754))
        self.assertEqual((second.x_mm, second.y_mm), (145.6630843248291, 40.52295973648266))
        rng = derived_rng(20261007, "pinned")
        at_distance = targets.placement_at_distance(20.0, audit("clustered"), GRID, rng, 0.5)
        self.assertAlmostEqual(at_distance.x_mm, 105.91326168649658, places=9)
        self.assertAlmostEqual(at_distance.y_mm, 65.78939910525517, places=9)
        around = targets.placement_around(audit("clustered")[12], 25.0, GRID, derived_rng(20261007, "pinned"))
        self.assertAlmostEqual(around.x_mm, 131.34404989630337, places=9)
        self.assertAlmostEqual(around.y_mm, 99.77910023512428, places=9)


if __name__ == "__main__":
    unittest.main()
