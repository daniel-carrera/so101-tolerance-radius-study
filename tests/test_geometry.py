"""The grid on the table, its regions, and distances from a point to the demonstrations."""

import itertools
import math
import unittest

import numpy as np

from tolerance_radius import config, geometry
from tolerance_radius.geometry import GeometryError, Grid, Rect
from tolerance_radius.schema import DemoStart, Point

GRID = Grid(
    cols=5, rows=3, cell_mm=50.0, held_out_cells=("A1", "B2", "D2", "E3"), cube_edge_mm=30.0, test_region_margin_mm=5.0
)


def demo(x: float, y: float, index: int = 0, dataset: str = "clustered", trained: bool = True) -> DemoStart:
    cell = GRID.cell_of(Point(x, y)) or "A2"
    return DemoStart(dataset, index, cell, measured=Point(x, y), trained=trained)


def demos_at(*positions: tuple[float, float], dataset: str = "clustered") -> list[DemoStart]:
    return [demo(x, y, index, dataset) for index, (x, y) in enumerate(positions)]


class GridFromProtocolTests(unittest.TestCase):
    def test_the_repository_protocol_gives_the_five_by_three_grid(self):
        grid = Grid.from_protocol(config.load_protocol())
        self.assertEqual(grid, GRID)

    def test_a_table_section_with_a_missing_key_is_refused(self):
        protocol = {"table": {"cols": 5, "rows": 3, "cell_mm": 50.0}}
        with self.assertRaises(GeometryError):
            Grid.from_protocol(protocol)

    def test_impossible_grids_are_refused(self):
        good = {"cols": 5, "rows": 3, "cell_mm": 50.0, "held_out_cells": ("A1",), "cube_edge_mm": 30.0}
        good["test_region_margin_mm"] = 5.0
        for bad in (
            {"cols": 0},
            {"cols": 27},
            {"rows": 0},
            {"cell_mm": 0.0},
            {"held_out_cells": ("F1",)},
            {"held_out_cells": ("A4",)},
            {"held_out_cells": ("A1", "A1")},
            {"cols": 1, "rows": 1},  # its only cell is held out: nothing is trained
            {"test_region_margin_mm": -1.0},
            {"test_region_margin_mm": 75.0},  # nothing would be left of a grid 150 mm high
            {"cube_edge_mm": 0.0},
        ):
            with self.subTest(bad=bad), self.assertRaises(GeometryError):
                Grid(**(good | bad))


class CellTests(unittest.TestCase):
    def test_labels_run_column_by_column(self):
        self.assertEqual(len(GRID.labels), 15)
        self.assertEqual(GRID.labels[:4], ("A1", "A2", "A3", "B1"))
        self.assertEqual(GRID.labels[-1], "E3")

    def test_centres(self):
        self.assertEqual(GRID.cell_centre("B1"), Point(75.0, 25.0))
        self.assertEqual(GRID.cell_centre("E3"), Point(225.0, 125.0))
        self.assertEqual(GRID.cell_centre("A1"), Point(25.0, 25.0))
        self.assertEqual(GRID.cell_centre("C2"), Point(125.0, 75.0))

    def test_a_cell_is_a_square_of_the_cell_size(self):
        self.assertEqual(GRID.cell_rect("B1"), Rect(50.0, 0.0, 100.0, 50.0))
        self.assertEqual(GRID.cell_rect("E3"), Rect(200.0, 100.0, 250.0, 150.0))

    def test_labels_that_are_not_on_the_grid_are_refused(self):
        for bad in ("F1", "A4", "A0", "a1", "", "AA1", "1A", "B"):
            with self.subTest(bad=bad), self.assertRaises(GeometryError):
                GRID.cell_centre(bad)

    def test_a_point_inside_a_cell_is_in_that_cell(self):
        self.assertEqual(GRID.cell_of(Point(75.0, 25.0)), "B1")
        self.assertEqual(GRID.cell_of(Point(249.0, 149.0)), "E3")
        self.assertEqual(GRID.cell_of(Point(1.0, 101.0)), "A3")

    def test_every_centre_is_in_its_own_cell(self):
        for label in GRID.labels:
            self.assertEqual(GRID.cell_of(GRID.cell_centre(label)), label)

    def test_a_point_on_a_border_belongs_to_the_cell_that_starts_there(self):
        self.assertEqual(GRID.cell_of(Point(50.0, 25.0)), "B1")
        self.assertEqual(GRID.cell_of(Point(25.0, 50.0)), "A2")
        self.assertEqual(GRID.cell_of(Point(50.0, 50.0)), "B2")
        self.assertEqual(GRID.cell_of(Point(0.0, 0.0)), "A1")

    def test_the_far_edges_of_the_grid_belong_to_the_last_cells(self):
        self.assertEqual(GRID.cell_of(Point(250.0, 25.0)), "E1")
        self.assertEqual(GRID.cell_of(Point(25.0, 150.0)), "A3")
        self.assertEqual(GRID.cell_of(Point(250.0, 150.0)), "E3")

    def test_a_point_outside_the_grid_is_in_no_cell(self):
        for x, y in ((-0.01, 25.0), (250.01, 25.0), (25.0, -0.01), (25.0, 150.01), (-10.0, -10.0), (300.0, 200.0)):
            with self.subTest(x=x, y=y):
                self.assertIsNone(GRID.cell_of(Point(x, y)))

    def test_trained_and_held_out_cells(self):
        self.assertEqual(GRID.held_out_cells, ("A1", "B2", "D2", "E3"))
        self.assertEqual(len(GRID.trained_cells), 11)
        self.assertEqual(set(GRID.trained_cells) | set(GRID.held_out_cells), set(GRID.labels))
        self.assertTrue(GRID.is_trained("C2"))
        self.assertFalse(GRID.is_trained("B2"))
        with self.assertRaises(GeometryError):
            GRID.is_trained("Z9")


class RegionTests(unittest.TestCase):
    def test_the_test_region_is_the_grid_shrunk_by_the_margin(self):
        self.assertEqual(GRID.outline, Rect(0.0, 0.0, 250.0, 150.0))
        self.assertEqual(GRID.test_region, Rect(5.0, 5.0, 245.0, 145.0))

    def test_the_edge_of_the_test_region_is_inside_it(self):
        self.assertTrue(GRID.in_test_region(Point(5.0, 5.0)))
        self.assertTrue(GRID.in_test_region(Point(245.0, 145.0)))
        self.assertTrue(GRID.in_test_region(Point(125.0, 75.0)))
        self.assertFalse(GRID.in_test_region(Point(4.99, 75.0)))
        self.assertFalse(GRID.in_test_region(Point(125.0, 145.01)))

    def test_the_trained_area_is_the_trained_cells_with_their_borders(self):
        self.assertTrue(GRID.in_trained_cells(Point(25.0, 75.0)))  # centre of A2
        self.assertFalse(GRID.in_trained_cells(Point(75.0, 75.0)))  # centre of B2, held out
        self.assertFalse(GRID.in_trained_cells(Point(25.0, 25.0)))  # centre of A1, held out
        self.assertTrue(GRID.in_trained_cells(Point(50.0, 75.0)))  # the border between A2 and B2
        self.assertTrue(GRID.in_trained_cells(Point(50.0, 50.0)))  # a corner shared by A2 and B1
        self.assertFalse(GRID.in_trained_cells(Point(0.0, 0.0)))  # the outer corner of A1
        self.assertTrue(GRID.in_trained_cells(Point(0.0, 150.0)))  # the outer corner of A3
        self.assertFalse(GRID.in_trained_cells(Point(-1.0, 75.0)))  # off the grid
        self.assertFalse(GRID.in_trained_cells(Point(125.0, 151.0)))

    def test_the_mask_agrees_with_the_point_test(self):
        rng = np.random.default_rng(0)
        xs = rng.random(400) * 280.0 - 15.0
        ys = rng.random(400) * 180.0 - 15.0
        mask = GRID.trained_cells_mask(xs, ys)
        self.assertEqual(mask.shape, (400,))
        for x, y, inside in zip(xs.tolist(), ys.tolist(), mask.tolist(), strict=True):
            self.assertEqual(inside, GRID.in_trained_cells(Point(x, y)))

    def test_regions_are_parts_of_the_test_region(self):
        self.assertEqual(geometry.REGIONS, ("trained_cells", "held_out_cells", "test_region"))
        self.assertAlmostEqual(GRID.region_area_mm2("test_region"), 240.0 * 140.0)
        # Two corner cells keep 45 mm x 45 mm inside the margin; B2 and D2 are whole.
        self.assertAlmostEqual(GRID.region_area_mm2("held_out_cells"), 2 * 45.0 * 45.0 + 2 * 2500.0)
        self.assertAlmostEqual(GRID.region_area_mm2("trained_cells"), 33600.0 - 9050.0)

    def test_region_rectangles_lie_in_the_test_region_and_do_not_overlap(self):
        for region in geometry.REGIONS:
            rects = GRID.region_rects(region)
            self.assertAlmostEqual(sum(rect.area_mm2 for rect in rects), GRID.region_area_mm2(region))
            for rect in rects:
                self.assertGreaterEqual(rect.x_min_mm, 5.0)
                self.assertLessEqual(rect.x_max_mm, 245.0)
                self.assertGreaterEqual(rect.y_min_mm, 5.0)
                self.assertLessEqual(rect.y_max_mm, 145.0)
            for first, second in itertools.combinations(rects, 2):
                overlap_x = min(first.x_max_mm, second.x_max_mm) - max(first.x_min_mm, second.x_min_mm)
                overlap_y = min(first.y_max_mm, second.y_max_mm) - max(first.y_min_mm, second.y_min_mm)
                self.assertFalse(overlap_x > 0 and overlap_y > 0)
        self.assertEqual(len(GRID.region_rects("trained_cells")), 11)

    def test_a_point_is_in_a_region_only_inside_the_test_region(self):
        self.assertTrue(GRID.in_region(Point(25.0, 75.0), "trained_cells"))
        self.assertFalse(GRID.in_region(Point(2.0, 75.0), "trained_cells"))  # a trained cell, but in the margin
        self.assertFalse(GRID.in_region(Point(75.0, 75.0), "trained_cells"))
        self.assertTrue(GRID.in_region(Point(75.0, 75.0), "held_out_cells"))
        self.assertFalse(GRID.in_region(Point(2.0, 2.0), "held_out_cells"))
        self.assertTrue(GRID.in_region(Point(75.0, 75.0), "test_region"))
        self.assertFalse(GRID.in_region(Point(50.0, 75.0), "held_out_cells"))  # the border counts as trained
        with self.assertRaises(GeometryError):
            GRID.in_region(Point(25.0, 75.0), "everywhere")

    def test_every_point_of_the_test_region_is_in_exactly_one_of_the_two_kinds_of_cell(self):
        rng = np.random.default_rng(1)
        xs = rng.random(500) * 250.0
        ys = rng.random(500) * 150.0
        trained = GRID.region_mask(xs, ys, "trained_cells")
        held_out = GRID.region_mask(xs, ys, "held_out_cells")
        whole = GRID.region_mask(xs, ys, "test_region")
        self.assertTrue(np.array_equal(trained ^ held_out, whole))
        self.assertFalse(np.any(trained & held_out))
        for x, y, flag in zip(xs.tolist(), ys.tolist(), trained.tolist(), strict=True):
            self.assertEqual(flag, GRID.in_region(Point(x, y), "trained_cells"))

    def test_a_lattice_of_sample_points_covers_a_region_with_its_area(self):
        for region in geometry.REGIONS:
            xs, ys, weights = GRID.region_lattice(region, step_mm=1.0)
            with self.subTest(region=region):
                self.assertEqual(xs.shape, ys.shape)
                self.assertEqual(xs.shape, weights.shape)
                self.assertAlmostEqual(float(weights.sum()), GRID.region_area_mm2(region))
                self.assertTrue(bool(np.all(GRID.region_mask(xs, ys, region))))
                self.assertLessEqual(float(weights.max()), 1.0 + 1e-9)

    def test_a_lattice_step_that_does_not_divide_the_cells_still_gives_the_right_area(self):
        xs, ys, weights = GRID.region_lattice("trained_cells", step_mm=0.7)
        self.assertAlmostEqual(float(weights.sum()), GRID.region_area_mm2("trained_cells"))
        self.assertLessEqual(float(weights.max()), 0.49 + 1e-9)
        with self.assertRaises(GeometryError):
            GRID.region_lattice("trained_cells", step_mm=0.0)


class RectTests(unittest.TestCase):
    def test_size_and_membership(self):
        rect = Rect(5.0, 5.0, 245.0, 145.0)
        self.assertEqual((rect.width_mm, rect.height_mm, rect.area_mm2), (240.0, 140.0, 33600.0))
        self.assertTrue(rect.contains(Point(5.0, 145.0)))
        self.assertFalse(rect.contains(Point(4.9, 100.0)))

    def test_an_inside_out_rectangle_is_refused(self):
        with self.assertRaises(GeometryError):
            Rect(10.0, 0.0, 5.0, 10.0)


class TrainedDemoTests(unittest.TestCase):
    def test_it_keeps_one_set_and_only_what_was_trained_on_in_episode_order(self):
        starts = [
            demo(25.0, 75.0, 2),
            demo(75.0, 25.0, 0),
            demo(125.0, 25.0, 1, trained=False),
            demo(40.0, 80.0, 0, dataset="spread"),
        ]
        kept = geometry.trained_demos(starts, "clustered")
        self.assertEqual([d.episode_index for d in kept], [0, 2])
        self.assertEqual([d.episode_index for d in geometry.trained_demos(starts, "spread")], [0])
        with self.assertRaises(GeometryError):
            geometry.trained_demos(starts, "scattered")

    def test_an_episode_listed_twice_is_refused(self):
        with self.assertRaises(GeometryError):
            geometry.trained_demos([demo(25.0, 75.0, 3), demo(75.0, 25.0, 3)], "clustered")

    def test_the_positions_distances_are_taken_to_are_the_measured_starts_of_the_trained_episodes(self):
        starts = [demo(25.0, 75.0, 0), demo(125.0, 25.0, 1, trained=False), demo(75.5, 24.5, 2)]
        positions = geometry.trained_positions(starts)
        self.assertEqual(positions.tolist(), [[25.0, 75.0], [75.5, 24.5]])
        with self.assertRaises(GeometryError):
            geometry.trained_positions([*starts, demo(40.0, 80.0, 0, dataset="spread")])
        with self.assertRaises(GeometryError):
            geometry.trained_positions([demo(125.0, 25.0, 1, trained=False)])


class NearestTests(unittest.TestCase):
    def test_it_returns_the_position_in_the_list_and_the_distance(self):
        demos = demos_at((25.0, 75.0), (75.0, 25.0), (125.0, 25.0))
        index, distance = geometry.nearest(Point(78.0, 29.0), demos)
        self.assertEqual(index, 1)
        self.assertAlmostEqual(distance, 5.0)
        self.assertIsInstance(index, int)
        self.assertIsInstance(distance, float)

    def test_a_point_on_a_demonstration_is_at_distance_zero(self):
        demos = demos_at((25.0, 75.0), (75.0, 25.0))
        self.assertEqual(geometry.nearest(Point(25.0, 75.0), demos), (0, 0.0))

    def test_between_two_equally_near_demonstrations_the_first_wins(self):
        demos = demos_at((0.0, 0.0), (10.0, 0.0))
        self.assertEqual(geometry.nearest(Point(5.0, 0.0), demos)[0], 0)

    def test_distances_are_to_measured_positions_not_planned_ones(self):
        start = DemoStart("clustered", 0, "B1", measured=Point(76.0, 25.0), planned=Point(75.0, 25.0))
        self.assertAlmostEqual(geometry.nearest(Point(80.0, 25.0), [start])[1], 4.0)

    def test_an_episode_left_out_of_training_is_never_the_nearest(self):
        demos = [demo(25.0, 75.0, 0), demo(100.0, 100.0, 1, trained=False), demo(200.0, 100.0, 2)]
        index, distance = geometry.nearest(Point(101.0, 100.0), demos)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(distance, math.hypot(76.0, 25.0))

    def test_demonstrations_of_two_sets_are_refused(self):
        demos = [demo(25.0, 75.0, 0), demo(30.0, 80.0, 0, dataset="spread")]
        with self.assertRaises(GeometryError):
            geometry.nearest(Point(0.0, 0.0), demos)

    def test_no_trained_demonstration_is_an_error(self):
        with self.assertRaises(GeometryError):
            geometry.nearest(Point(0.0, 0.0), [])
        with self.assertRaises(GeometryError):
            geometry.nearest(Point(0.0, 0.0), [demo(25.0, 75.0, 0, trained=False)])


class DistanceFieldTests(unittest.TestCase):
    def test_the_field_matches_a_brute_force_loop(self):
        rng = np.random.default_rng(3)
        demos = [demo(float(x), float(y), i) for i, (x, y) in enumerate(rng.random((23, 2)) * [250.0, 150.0])]
        xs = rng.random((7, 9)) * 250.0
        ys = rng.random((7, 9)) * 150.0
        field = geometry.nearest_distance_field(xs, ys, demos)
        self.assertEqual(field.shape, (7, 9))
        for i in range(7):
            for j in range(9):
                brute = min(math.hypot(xs[i, j] - d.measured.x_mm, ys[i, j] - d.measured.y_mm) for d in demos)
                self.assertAlmostEqual(float(field[i, j]), brute, places=9)

    def test_the_field_agrees_with_nearest_point_by_point(self):
        demos = demos_at((25.0, 75.0), (75.0, 25.0), (125.0, 25.0), (225.0, 75.0))
        xs = np.array([0.0, 80.0, 130.0, 249.0])
        ys = np.array([0.0, 30.0, 140.0, 75.0])
        field = geometry.nearest_distance_field(xs, ys, demos)
        for x, y, value in zip(xs.tolist(), ys.tolist(), field.tolist(), strict=True):
            self.assertAlmostEqual(value, geometry.nearest(Point(x, y), demos)[1], places=9)

    def test_x_and_y_broadcast_against_each_other(self):
        demos = demos_at((25.0, 75.0), (75.0, 25.0))
        xs = np.linspace(0.0, 250.0, 6)[:, None]
        ys = np.linspace(0.0, 150.0, 4)[None, :]
        field = geometry.nearest_distance_field(xs, ys, demos)
        self.assertEqual(field.shape, (6, 4))
        self.assertAlmostEqual(float(field[0, 0]), math.hypot(25.0, 75.0))

    def test_the_field_ignores_episodes_left_out_of_training(self):
        demos = [demo(25.0, 75.0, 0), demo(100.0, 100.0, 1, trained=False)]
        field = geometry.nearest_distance_field(np.array([100.0]), np.array([100.0]), demos)
        self.assertAlmostEqual(float(field[0]), math.hypot(75.0, 25.0))

    def test_plain_positions_can_be_used_in_place_of_demonstrations(self):
        positions = np.array([[25.0, 75.0], [75.0, 25.0]])
        distances = geometry.nearest_distances_xy(np.array([28.0, 75.0]), np.array([79.0, 20.0]), positions)
        self.assertEqual(distances.tolist(), [5.0, 5.0])


def in_some_triangle(point: tuple[float, float], positions: list[tuple[float, float]]) -> bool:
    """Brute force: a point is in the convex hull exactly when it is in a triangle of three of the positions."""

    def side(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    for a, b, c in itertools.combinations(positions, 3):
        signs = (side(a, b, point), side(b, c, point), side(c, a, point))
        if all(s >= 0 for s in signs) or all(s <= 0 for s in signs):
            return True
    return False


class ConvexHullTests(unittest.TestCase):
    SQUARE = ((25.0, 25.0), (125.0, 25.0), (125.0, 125.0), (25.0, 125.0))

    def test_inside_outside_and_on_the_edge_of_a_square_of_demonstrations(self):
        demos = demos_at(*self.SQUARE)
        self.assertTrue(geometry.in_convex_hull(Point(75.0, 75.0), demos))
        self.assertFalse(geometry.in_convex_hull(Point(130.0, 75.0), demos))
        self.assertFalse(geometry.in_convex_hull(Point(24.9, 75.0), demos))
        self.assertTrue(geometry.in_convex_hull(Point(25.0, 75.0), demos))  # on an edge
        self.assertTrue(geometry.in_convex_hull(Point(125.0, 125.0), demos))  # on a corner

    def test_demonstrations_inside_the_hull_change_nothing(self):
        demos = demos_at(*self.SQUARE, (75.0, 75.0), (60.0, 90.0))
        self.assertTrue(geometry.in_convex_hull(Point(120.0, 30.0), demos))
        self.assertFalse(geometry.in_convex_hull(Point(126.0, 30.0), demos))

    def test_an_episode_left_out_of_training_does_not_stretch_the_hull(self):
        demos = [*demos_at(*self.SQUARE), demo(225.0, 75.0, 9, trained=False)]
        self.assertFalse(geometry.in_convex_hull(Point(175.0, 75.0), demos))

    def test_demonstrations_in_a_line_have_a_hull_that_is_a_segment(self):
        demos = demos_at((25.0, 25.0), (75.0, 25.0), (125.0, 25.0))
        self.assertTrue(geometry.in_convex_hull(Point(100.0, 25.0), demos))
        self.assertFalse(geometry.in_convex_hull(Point(100.0, 25.5), demos))
        self.assertFalse(geometry.in_convex_hull(Point(130.0, 25.0), demos))

    def test_a_single_demonstration_is_its_own_hull(self):
        demos = demos_at((25.0, 25.0))
        self.assertTrue(geometry.in_convex_hull(Point(25.0, 25.0), demos))
        self.assertFalse(geometry.in_convex_hull(Point(25.0, 26.0), demos))

    def test_repeated_positions_are_harmless(self):
        demos = demos_at(*self.SQUARE, *self.SQUARE)
        self.assertTrue(geometry.in_convex_hull(Point(75.0, 75.0), demos))
        self.assertFalse(geometry.in_convex_hull(Point(130.0, 75.0), demos))

    def test_it_agrees_with_a_brute_force_search_over_triangles(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            positions = [(float(x), float(y)) for x, y in rng.random((int(rng.integers(3, 9)), 2)) * 100.0]
            demos = demos_at(*positions)
            for x, y in (rng.random((25, 2)) * 140.0 - 20.0).tolist():
                self.assertEqual(geometry.in_convex_hull(Point(x, y), demos), in_some_triangle((x, y), positions))

    def test_demonstrations_of_two_sets_are_refused(self):
        demos = [demo(25.0, 75.0, 0), demo(30.0, 80.0, 0, dataset="spread")]
        with self.assertRaises(GeometryError):
            geometry.in_convex_hull(Point(0.0, 0.0), demos)


class LargestGapTests(unittest.TestCase):
    def test_one_position_in_the_middle_of_a_square_leaves_its_corners_farthest(self):
        distance, where = geometry.largest_nearest_distance(np.array([[25.0, 25.0]]), [Rect(0.0, 0.0, 50.0, 50.0)])
        self.assertAlmostEqual(distance, 25.0 * math.sqrt(2.0))
        self.assertIn((where.x_mm, where.y_mm), {(0.0, 0.0), (0.0, 50.0), (50.0, 0.0), (50.0, 50.0)})

    def test_one_position_off_centre_leaves_the_far_corner(self):
        distance, where = geometry.largest_nearest_distance(np.array([[10.0, 20.0]]), [Rect(0.0, 0.0, 50.0, 50.0)])
        self.assertAlmostEqual(distance, math.hypot(40.0, 30.0))
        self.assertEqual(where, Point(50.0, 50.0))

    def test_two_positions_leave_a_point_of_the_edge_between_them(self):
        # In a 100 x 20 strip with a position at each end, the farthest points are where the bisector meets the edges.
        ends = np.array([[0.0, 10.0], [100.0, 10.0]])
        distance, where = geometry.largest_nearest_distance(ends, [Rect(0.0, 0.0, 100.0, 20.0)])
        self.assertAlmostEqual(distance, math.hypot(50.0, 10.0))
        self.assertAlmostEqual(where.x_mm, 50.0)
        self.assertIn(where.y_mm, (0.0, 20.0))

    def test_three_positions_leave_the_centre_of_their_circle(self):
        positions = np.array([[50.0, 10.0], [10.0, 80.0], [90.0, 80.0]])
        distance, where = geometry.largest_nearest_distance(positions, [Rect(40.0, 40.0, 60.0, 70.0)])
        # The point at equal distance from the three: on x = 50, where the distances to (50, 10) and (10, 80) agree.
        y = (80.0**2 + 40.0**2 - 10.0**2) / (2 * 70.0)
        self.assertAlmostEqual(where.x_mm, 50.0)
        self.assertAlmostEqual(where.y_mm, y)
        self.assertAlmostEqual(distance, y - 10.0)

    def test_the_centres_of_the_trained_cells_leave_a_cell_corner_35_mm_away(self):
        centres = np.array([[GRID.cell_centre(c).x_mm, GRID.cell_centre(c).y_mm] for c in GRID.trained_cells])
        rects = [GRID.cell_rect(c) for c in GRID.trained_cells]
        distance, where = geometry.largest_nearest_distance(centres, rects)
        self.assertAlmostEqual(distance, 25.0 * math.sqrt(2.0))
        for coordinate in (where.x_mm, where.y_mm):  # a corner of a cell: a whole number of cell sizes
            self.assertAlmostEqual(coordinate / 50.0, round(coordinate / 50.0), places=9)

    def test_of_several_equally_far_points_the_one_nearest_the_middle_is_returned(self):
        # So that the answer does not hang on the order of the search or on the last digit of a rounding.
        centres = np.array([[GRID.cell_centre(c).x_mm, GRID.cell_centre(c).y_mm] for c in GRID.trained_cells])
        rects = [GRID.cell_rect(c) for c in GRID.trained_cells]
        # Many cell corners are 35.4 mm from the nearest centre. Four of them are nearest the middle of the grid,
        # (125, 75); of those, the one with the lowest x, then the lowest y.
        self.assertEqual(geometry.largest_nearest_distance(centres, rects)[1], Point(100.0, 50.0))
        self.assertEqual(geometry.largest_nearest_distance(centres, rects[::-1])[1], Point(100.0, 50.0))
        one = np.array([[25.0, 25.0]])
        self.assertEqual(geometry.largest_nearest_distance(one, [Rect(0.0, 0.0, 50.0, 50.0)])[1], Point(0.0, 0.0))
        # The corner (60, 0) is as far as (0, 0) from the position, and nearer the middle of the two rectangles.
        strip = [Rect(0.0, 0.0, 60.0, 10.0), Rect(100.0, 0.0, 110.0, 10.0)]
        two = np.array([[30.0, 5.0], [105.0, 5.0]])
        self.assertEqual(geometry.largest_nearest_distance(two, strip)[1], Point(60.0, 0.0))

    def test_repeated_positions_and_positions_in_a_line_are_harmless(self):
        positions = np.array([[10.0, 10.0], [10.0, 10.0], [30.0, 10.0], [50.0, 10.0], [50.0, 10.0]])
        distance, where = geometry.largest_nearest_distance(positions, [Rect(0.0, 0.0, 60.0, 40.0)])
        self.assertAlmostEqual(distance, math.hypot(10.0, 30.0))
        self.assertEqual(where.y_mm, 40.0)

    def test_it_matches_a_fine_lattice_search_for_scattered_positions(self):
        rng = np.random.default_rng(7)
        rects = [GRID.cell_rect(c) for c in GRID.trained_cells]
        step = 0.5
        ticks_x, ticks_y = np.arange(0.0, 250.0 + step / 2, step), np.arange(0.0, 150.0 + step / 2, step)
        gx, gy = np.meshgrid(ticks_x, ticks_y, indexing="ij")
        inside = GRID.trained_cells_mask(gx, gy)
        for _ in range(5):
            positions = rng.random((30, 2)) * [250.0, 150.0]
            distance, where = geometry.largest_nearest_distance(positions, rects)
            lattice = float(geometry.nearest_distances_xy(gx[inside], gy[inside], positions).max())
            # The exact answer is at least what any lattice point sees, and no lattice point is farther than half a
            # lattice diagonal from the true farthest point.
            self.assertGreaterEqual(distance, lattice - 1e-9)
            self.assertLessEqual(distance, lattice + step * math.sqrt(0.5) + 1e-9)
            self.assertTrue(GRID.in_trained_cells(where))
            self.assertAlmostEqual(float(geometry.nearest_distances_xy(where.x_mm, where.y_mm, positions)), distance)

    def test_nothing_to_measure_is_an_error(self):
        with self.assertRaises(GeometryError):
            geometry.largest_nearest_distance(np.zeros((0, 2)), [Rect(0.0, 0.0, 1.0, 1.0)])
        with self.assertRaises(GeometryError):
            geometry.largest_nearest_distance(np.array([[1.0, 1.0]]), [])


if __name__ == "__main__":
    unittest.main()
