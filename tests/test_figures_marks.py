"""Where the figures put their marks: piles that hide nothing, stretches with no trial, curves' pieces, hatchings."""

import math
import unittest

import numpy as np

from tolerance_radius.figures import marks
from tolerance_radius.geometry import Rect

#: The distances of the 19 near trials and the 12 far trials of the earlier evaluation, to two decimals.
AUGUST_NEAR = [1.49, 1.95, 3.03, 2.43, 1.43, 2.34, 2.08, 5.63, 1.22, 0.63, 1.25, 2.76, 1.36, 3.02, 1.64, 5.57, 2.54,
               2.53, 0.80]  # fmt: skip
AUGUST_FAR = [38.11, 45.34, 35.37, 33.37, 38.63, 47.14, 31.01, 36.40, 40.01, 44.21, 34.31, 33.63]


class StackHeightsTests(unittest.TestCase):
    def assert_nothing_is_covered(self, xs, heights, diameter):
        for i in range(len(xs)):
            for j in range(i):
                apart = math.hypot(xs[i] - xs[j], heights[i] - heights[j])
                self.assertGreaterEqual(apart, diameter - 1e-6, f"marks {j} and {i} overlap")

    def test_marks_with_room_stay_on_the_line(self):
        self.assertEqual(marks.stack_heights([0.0, 2.0, 4.0, 10.0], 2.0), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(marks.stack_heights([], 2.0), [])
        self.assertEqual(marks.stack_heights([3.0], 2.0), [0.0])

    def test_marks_at_one_place_are_piled_one_diameter_apart(self):
        self.assertEqual(marks.stack_heights([5.0, 5.0, 5.0], 1.5), [0.0, 1.5, 3.0])

    def test_a_mark_rests_in_the_hollow_between_its_neighbours(self):
        # The second mark is half a diameter from the first: it rests on it. The third is a whole diameter from the
        # first and may touch the second, so it goes back down to the line.
        heights = marks.stack_heights([0.0, 1.0, 2.0], 2.0)
        self.assertAlmostEqual(heights[0], 0.0)
        self.assertAlmostEqual(heights[1], math.sqrt(3.0))
        self.assertAlmostEqual(heights[2], 0.0)

    def test_the_heights_come_back_in_the_order_given(self):
        xs = [9.0, 0.0, 0.2, 9.1]
        heights = marks.stack_heights(xs, 1.0)
        self.assertEqual(heights[1], 0.0)  # the smallest x is dropped first
        self.assertGreater(heights[2], 0.9)
        self.assertEqual(heights[0], 0.0)
        self.assertGreater(heights[3], 0.9)

    def test_no_mark_covers_another_and_every_mark_is_as_low_as_it_can_go(self):
        rng = np.random.default_rng(3)
        for count, spread in ((40, 6.0), (60, 20.0), (25, 0.5)):
            xs = (rng.random(count) * spread).tolist()
            heights = marks.stack_heights(xs, 1.4)
            self.assert_nothing_is_covered(xs, heights, 1.4)
            self.assertEqual(min(heights), 0.0)
            # A mark above the line rests on another one: it touches a mark that is lower than itself.
            for i, height in enumerate(heights):
                if height > 0.0:
                    touching = [
                        j
                        for j in range(count)
                        if j != i
                        and heights[j] < height
                        and abs(math.hypot(xs[i] - xs[j], height - heights[j]) - 1.4) < 1e-6
                    ]
                    self.assertTrue(touching, f"mark {i} hangs in the air")

    def test_the_same_marks_give_the_same_pile(self):
        self.assertEqual(marks.stack_heights(AUGUST_NEAR, 1.0), marks.stack_heights(list(AUGUST_NEAR), 1.0))

    def test_the_near_trials_of_the_earlier_evaluation_make_one_tall_pile(self):
        # 17 of the 19 lie within 2.4 mm of each other: on a 76 mm axis of 52 mm they cannot sit side by side.
        scale = 76.0 / 52.0
        xs = [d * scale for d in AUGUST_NEAR]
        heights = marks.stack_heights(xs, 1.5)
        self.assert_nothing_is_covered(xs, heights, 1.5)
        self.assertGreater(marks.stack_extent(heights, 1.5), 5.0)
        self.assertLess(marks.stack_extent(heights, 1.5), 12.0)

    def test_the_room_a_pile_takes(self):
        self.assertEqual(marks.stack_extent([], 1.5), 0.0)
        self.assertEqual(marks.stack_extent([0.0, 0.0], 1.5), 1.5)
        self.assertEqual(marks.stack_extent([0.0, 1.5, 3.0], 1.5), 4.5)

    def test_what_is_not_a_pile_is_refused(self):
        with self.assertRaises(ValueError):
            marks.stack_heights([1.0], 0.0)
        with self.assertRaises(ValueError):
            marks.stack_heights([float("nan")], 1.0)


class UntestedStretchesTests(unittest.TestCase):
    def test_the_earlier_evaluation_has_one_stretch_with_no_trial(self):
        stretches = marks.untested_stretches(AUGUST_NEAR + AUGUST_FAR, upto_mm=50.0, longer_than_mm=15.0)
        self.assertEqual(stretches, [(5.63, 31.01)])

    def test_the_ends_of_the_axis_count(self):
        stretches = marks.untested_stretches([20.0, 22.0], upto_mm=50.0, longer_than_mm=15.0)
        self.assertEqual(stretches, [(0.0, 20.0), (22.0, 50.0)])

    def test_trials_close_enough_leave_no_stretch(self):
        tested = [0.0, 10.0, 20.0, 30.0, 40.0]
        self.assertEqual(marks.untested_stretches(tested, upto_mm=50.0, longer_than_mm=15.0), [])
        self.assertEqual(marks.untested_stretches(tested, upto_mm=50.0, longer_than_mm=9.0), [(0.0, 10.0),
                         (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.0)])  # fmt: skip

    def test_no_trial_at_all_is_one_stretch_and_trials_off_the_axis_are_left_out(self):
        self.assertEqual(marks.untested_stretches([], upto_mm=50.0, longer_than_mm=15.0), [(0.0, 50.0)])
        self.assertEqual(marks.untested_stretches([70.0], upto_mm=50.0, longer_than_mm=15.0), [(0.0, 50.0)])
        with self.assertRaises(ValueError):
            marks.untested_stretches([1.0], upto_mm=0.0, longer_than_mm=15.0)


class CurvePiecesTests(unittest.TestCase):
    def test_a_curve_runs_from_the_nearest_trial_to_the_farthest(self):
        self.assertEqual(marks.curve_pieces([12.0, 0.5, 30.0, 21.0, 41.0], longer_than_mm=15.0), [(0.5, 41.0)])

    def test_it_is_interrupted_where_nothing_was_tested(self):
        pieces = marks.curve_pieces(AUGUST_NEAR + AUGUST_FAR, longer_than_mm=15.0)
        self.assertEqual(pieces, [(0.63, 5.63), (31.01, 47.14)])

    def test_a_trial_that_stands_alone_carries_no_curve(self):
        self.assertEqual(marks.curve_pieces([], longer_than_mm=15.0), [])
        self.assertEqual(marks.curve_pieces([10.0], longer_than_mm=15.0), [])
        self.assertEqual(marks.curve_pieces([10.0, 10.0], longer_than_mm=15.0), [])
        self.assertEqual(marks.curve_pieces([0.0, 2.0, 30.0], longer_than_mm=15.0), [(0.0, 2.0)])
        self.assertEqual(marks.curve_pieces([0.0, 30.0, 32.0, 60.0], longer_than_mm=15.0), [(30.0, 32.0)])


class SpreadApartTests(unittest.TestCase):
    def test_places_with_room_stay_where_they_are(self):
        self.assertEqual(marks.spread_apart([10.0, 3.0, 20.0], 2.0), [10.0, 3.0, 20.0])
        self.assertEqual(marks.spread_apart([], 2.0), [])
        self.assertEqual(marks.spread_apart([5.0], 2.0), [5.0])
        self.assertEqual(marks.spread_apart([5.0, 7.0], 2.0), [5.0, 7.0])  # exactly far enough

    def test_two_that_crowd_each_other_move_apart_by_equal_amounts(self):
        self.assertEqual(marks.spread_apart([10.0, 11.0], 2.0), [9.5, 11.5])
        self.assertEqual(marks.spread_apart([11.0, 10.0], 2.0), [11.5, 9.5])  # in the order given
        self.assertEqual(marks.spread_apart([4.0, 4.0], 1.0), [3.5, 4.5])  # the first given stays first

    def test_a_group_is_laid_out_about_the_mean_of_where_its_members_belong(self):
        moved = marks.spread_apart([10.0, 10.5, 11.0, 30.0], 2.0)
        self.assertEqual(moved, [8.5, 10.5, 12.5, 30.0])
        self.assertAlmostEqual(sum(moved[:3]) / 3.0, 10.5)

    def test_moving_one_group_can_draw_a_neighbour_in(self):
        # 10 and 11 alone would end at 9.5 and 11.5, which is too near 13: the three are set out together.
        moved = marks.spread_apart([10.0, 11.0, 13.0], 2.0)
        for first, second in zip(moved, moved[1:], strict=False):
            self.assertGreaterEqual(second - first, 2.0 - 1e-9)
        self.assertAlmostEqual(sum(moved) / 3.0, 34.0 / 3.0)
        self.assertEqual(marks.spread_apart([1.0, 2.0, 3.0], 0.0), [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            marks.spread_apart([1.0], -1.0)


class FreeSlotTests(unittest.TestCase):
    def test_with_nothing_in_the_way_the_place_is_at_the_high_end(self):
        self.assertEqual(marks.free_slot([], 10.0, low_mm=0.0, high_mm=50.0), 40.0)
        self.assertIsNone(marks.free_slot([], 60.0, low_mm=0.0, high_mm=50.0))
        self.assertEqual(marks.free_slot([], 50.0, low_mm=0.0, high_mm=50.0), 0.0)

    def test_it_is_the_place_nearest_the_high_end_that_keeps_clear(self):
        taken = [(45.0, 50.0), (10.0, 20.0)]
        self.assertEqual(marks.free_slot(taken, 10.0, low_mm=0.0, high_mm=50.0), 35.0)
        self.assertEqual(marks.free_slot(taken, 10.0, low_mm=0.0, high_mm=50.0, clear_mm=1.5), 33.5)
        self.assertEqual(marks.free_slot(taken, 24.0, low_mm=0.0, high_mm=50.0), 21.0)  # between the two
        self.assertEqual(marks.free_slot(taken, 26.0, low_mm=0.0, high_mm=50.0), None)
        self.assertEqual(marks.free_slot(taken, 9.0, low_mm=0.0, high_mm=50.0, clear_mm=13.0), None)
        self.assertEqual(marks.free_slot([(20.0, 50.0)], 9.0, low_mm=0.0, high_mm=50.0, clear_mm=1.0), 10.0)

    def test_stretches_that_run_into_each_other_leave_no_gap_between_them(self):
        self.assertEqual(marks.free_slot([(10.0, 30.0), (25.0, 50.0)], 5.0, low_mm=0.0, high_mm=50.0), 5.0)
        self.assertIsNone(marks.free_slot([(0.0, 50.0)], 5.0, low_mm=0.0, high_mm=50.0))


class HatchSegmentsTests(unittest.TestCase):
    RECT = Rect(50.0, 50.0, 100.0, 100.0)

    def test_every_line_runs_at_45_degrees_inside_the_rectangle(self):
        for rising in (True, False):
            segments = marks.hatch_segments(self.RECT, 7.0, rising=rising)
            self.assertGreater(len(segments), 8)
            for (x0, y0), (x1, y1) in segments:
                self.assertAlmostEqual((y1 - y0) / (x1 - x0), 1.0 if rising else -1.0)
                for x, y in ((x0, y0), (x1, y1)):
                    self.assertTrue(50.0 - 1e-9 <= x <= 100.0 + 1e-9 and 50.0 - 1e-9 <= y <= 100.0 + 1e-9)

    def test_the_lines_are_the_stated_distance_apart(self):
        segments = marks.hatch_segments(self.RECT, 7.0)
        offsets = sorted(y0 - x0 for (x0, y0), _ in segments)
        for first, second in zip(offsets, offsets[1:], strict=False):
            self.assertAlmostEqual((second - first) / math.sqrt(2.0), 7.0)

    def test_the_hatchings_of_two_rectangles_that_touch_run_through(self):
        left = marks.hatch_segments(Rect(0.0, 0.0, 50.0, 50.0), 7.0)
        right = marks.hatch_segments(Rect(50.0, 0.0, 100.0, 50.0), 7.0)
        ends_at_the_border = {round(y1, 6) for _, (x1, y1) in left if abs(x1 - 50.0) < 1e-9}
        starts_at_the_border = {round(y0, 6) for (x0, y0), _ in right if abs(x0 - 50.0) < 1e-9}
        self.assertTrue(ends_at_the_border)
        self.assertEqual(ends_at_the_border & starts_at_the_border, ends_at_the_border - {50.0})

    def test_the_lines_cover_the_rectangle_from_corner_to_corner(self):
        segments = marks.hatch_segments(self.RECT, 7.0)
        offsets = [y0 - x0 for (x0, y0), _ in segments]
        self.assertLess(min(offsets), -50.0 + 7.0 * math.sqrt(2.0))
        self.assertGreater(max(offsets), 50.0 - 7.0 * math.sqrt(2.0))
        with self.assertRaises(ValueError):
            marks.hatch_segments(self.RECT, 0.0)


if __name__ == "__main__":
    unittest.main()
