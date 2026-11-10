"""The instrument check: step errors, span error, straightness and repeatability from logged readings."""

import math
import unittest

from tolerance_radius.instrument import check_against_scale, scatter_mm
from tolerance_radius.schema import Point


class ScaleCheckTests(unittest.TestCase):
    def test_perfect_readings_have_no_error(self):
        check = check_against_scale([Point(20.0 + 10 * i, 40.0) for i in range(6)], true_step_mm=10.0)
        self.assertEqual(len(check.measured_steps_mm), 5)
        self.assertAlmostEqual(check.mean_abs_step_error_mm, 0.0)
        self.assertAlmostEqual(check.measured_span_mm, 50.0)
        self.assertAlmostEqual(check.span_error_mm, 0.0)
        self.assertAlmostEqual(check.max_off_line_mm, 0.0)
        self.assertTrue(check.passes(1.5, 3.0))

    def test_a_constant_offset_in_the_readings_does_not_matter(self):
        true = [Point(10.0 * i, 0.0) for i in range(6)]
        shifted = [Point(p.x_mm + 8.3, p.y_mm - 4.1) for p in true]
        self.assertAlmostEqual(check_against_scale(shifted, 10.0).mean_abs_step_error_mm, 0.0)

    def test_a_stretched_scale_shows_as_step_and_span_error(self):
        check = check_against_scale([Point(10.3 * i, 0.0) for i in range(6)], true_step_mm=10.0)
        self.assertAlmostEqual(check.mean_abs_step_error_mm, 0.3)
        self.assertAlmostEqual(check.span_error_mm, 1.5)
        self.assertAlmostEqual(check.scale_error_percent, 3.0)

    def test_the_direction_of_the_run_does_not_matter(self):
        diagonal = [Point(10.0 * i * math.cos(0.5), 10.0 * i * math.sin(0.5)) for i in range(4)]
        self.assertAlmostEqual(check_against_scale(diagonal, 10.0).max_abs_step_error_mm, 0.0)

    def test_one_bad_step_fails_the_worst_step_limit(self):
        readings = [Point(0, 0), Point(10, 0), Point(24, 0), Point(30, 0)]
        check = check_against_scale(readings, true_step_mm=10.0)
        self.assertAlmostEqual(check.max_abs_step_error_mm, 4.0)
        self.assertFalse(check.passes(1.5, 3.0))
        self.assertFalse(check.passes(5.0, 3.0))
        self.assertTrue(check.passes(5.0, 4.0))

    def test_a_reading_off_the_line_is_reported(self):
        check = check_against_scale([Point(0, 0), Point(10, 2), Point(20, 0)], true_step_mm=10.0)
        self.assertAlmostEqual(check.max_off_line_mm, 2.0)

    def test_it_needs_two_readings_and_a_positive_step(self):
        with self.assertRaises(ValueError):
            check_against_scale([Point(0, 0)], 10.0)
        with self.assertRaises(ValueError):
            check_against_scale([Point(0, 0), Point(10, 0)], 0.0)


class ScatterTests(unittest.TestCase):
    def test_scatter_is_measured_from_the_mean(self):
        rms, largest = scatter_mm([Point(0, 0), Point(2, 0), Point(0, 2), Point(2, 2)])
        self.assertAlmostEqual(rms, math.sqrt(2))
        self.assertAlmostEqual(largest, math.sqrt(2))

    def test_identical_readings_have_no_scatter(self):
        self.assertEqual(scatter_mm([Point(5, 5)] * 3), (0.0, 0.0))

    def test_no_readings_is_an_error(self):
        with self.assertRaises(ValueError):
            scatter_mm([])


if __name__ == "__main__":
    unittest.main()
