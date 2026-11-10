"""The frame of Figures 1 and 3: every trial a mark at its own distance, the strips, the rate band, the axes."""

import math
import unittest

from tolerance_radius.figures import ratechart, style
from tolerance_radius.figures.ratechart import MarkGroup, RateChart
from tolerance_radius.figures.style import MM_PER_PT, FigureError, Page

POLICY = MarkGroup(
    style.CLUSTERED,
    (0.4, 0.9, 1.3, 1.8, 0.2, 1.1, 10.2, 19.6, 20.4, 24.8, 30.3, 29.5, 40.2, 39.1, 41.0, 38.8, 35.2, 25.3),
    (True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, False, False,
     True),
)  # fmt: skip
REPLAY = MarkGroup(
    style.REPLAY,
    (0.3, 1.2, 0.8, 1.6, 5.4, 9.8, 14.7, 10.3, 15.2, 19.9, 25.4, 20.3, 15.6),
    (True, True, True, False, True, True, False, True, False, False, False, False, True),
)


def chart(groups=(POLICY, REPLAY), height_mm=46.0, **options) -> RateChart:
    options = {"x_max_mm": 50.0, "x_label": "Distance to the nearest demonstration, $d$ (mm)"} | options
    return RateChart(Page(height_mm), list(groups), **options)


def lines(drawn: RateChart, prefix: str) -> list:
    return [line for line in drawn.page.axes.lines if (line.get_gid() or "").startswith(prefix)]


def texts(drawn: RateChart, prefix: str) -> list:
    return [text for text in drawn.page.texts() if (text.get_gid() or "").startswith(prefix)]


class PlacesTests(unittest.TestCase):
    def test_distances_run_from_left_to_right_inside_the_axis(self):
        drawn = chart()
        self.assertGreater(drawn.x(0.0), drawn.left_mm)
        self.assertLess(drawn.x(50.0), drawn.right_mm)
        self.assertAlmostEqual(drawn.x(25.0), (drawn.x(0.0) + drawn.x(50.0)) / 2.0)
        self.assertGreater(drawn.right_mm - drawn.left_mm, 70.0)  # most of the column's 88 mm is the axis
        self.assertAlmostEqual(drawn.right_mm, 88.0 - ratechart.RIGHT_MM)

    def test_rates_run_from_the_bottom_of_the_band_to_its_top(self):
        drawn = chart()
        self.assertEqual(drawn.y(0.0), drawn.band_low_mm)
        self.assertEqual(drawn.y(1.0), drawn.band_high_mm)
        self.assertAlmostEqual(drawn.y(0.5), (drawn.band_low_mm + drawn.band_high_mm) / 2.0)
        self.assertGreaterEqual(drawn.band_high_mm - drawn.band_low_mm, ratechart.LEAST_BAND_MM)

    def test_everything_is_on_the_page(self):
        drawn = chart()
        self.assertLessEqual(drawn.strip_edge_mm[True], 46.0 - ratechart.TOP_MM + 1e-9)
        self.assertGreater(drawn.strip_edge_mm[False], ratechart.AXIS_Y_MM)
        for x, y in drawn.mark_places.values():
            self.assertTrue(drawn.left_mm < x < drawn.right_mm)
            self.assertTrue(ratechart.AXIS_Y_MM < y < 46.0)

    def test_a_trial_beyond_the_end_of_the_axis_is_refused_not_dropped(self):
        with self.assertRaisesRegex(FigureError, "beyond the axis"):
            chart(x_max_mm=40.0)
        with self.assertRaises(FigureError):
            chart(x_max_mm=0.0)

    def test_a_group_has_an_outcome_for_every_distance(self):
        with self.assertRaises(FigureError):
            MarkGroup(style.CLUSTERED, (1.0, 2.0), (True,))
        with self.assertRaises(FigureError):
            MarkGroup(style.CLUSTERED, (1.0, 2.0), (True, False), (True,))
        with self.assertRaises(FigureError):
            MarkGroup(style.CLUSTERED, (-1.0,), (True,))
        with self.assertRaises(FigureError):
            MarkGroup(style.CLUSTERED, (math.nan,), (True,))


class MarksTests(unittest.TestCase):
    def test_every_trial_has_one_mark_at_its_own_distance(self):
        drawn = chart()
        self.assertEqual(len(drawn.mark_places), len(POLICY.distances_mm) + len(REPLAY.distances_mm))
        for g, group in enumerate((POLICY, REPLAY)):
            for t, distance in enumerate(group.distances_mm):
                x, _ = drawn.mark_places[(g, t)]
                self.assertAlmostEqual(x, drawn.x(distance))  # a mark never moves sideways
        self.assertEqual(len(lines(drawn, "trial:")), len(drawn.mark_places))

    def test_successes_are_above_the_rate_band_and_failures_below(self):
        drawn = chart()
        for g, group in enumerate((POLICY, REPLAY)):
            for t, success in enumerate(group.successes):
                _, y = drawn.mark_places[(g, t)]
                if success:
                    self.assertGreater(y, drawn.band_high_mm + ratechart.BAND_GAP_MM)
                else:
                    self.assertLess(y, drawn.band_low_mm - ratechart.BAND_GAP_MM)

    def test_no_mark_hides_another(self):
        drawn = chart()
        pitch = (drawn.mark_pt + ratechart.PILE_PT - ratechart.MARK_PT) * MM_PER_PT
        places = list(drawn.mark_places.values())
        for index, (x1, y1) in enumerate(places):
            for x2, y2 in places[:index]:
                self.assertGreaterEqual(math.hypot(x1 - x2, y1 - y2), pitch - 1e-6)

    def test_each_group_has_a_lane_of_its_own_with_the_first_group_next_to_the_band(self):
        drawn = chart()
        for success in (True, False):
            first = [drawn.mark_places[(0, t)][1] for t, s in enumerate(POLICY.successes) if s == success]
            second = [drawn.mark_places[(1, t)][1] for t, s in enumerate(REPLAY.successes) if s == success]
            if success:
                self.assertLess(max(first), min(second))
            else:
                self.assertGreater(min(first), max(second))

    def test_groups_can_share_one_pile(self):
        drawn = chart(separate_lanes=False)
        lowest_success = [min(drawn.mark_places[(g, t)][1] for t, s in enumerate(group.successes) if s)
                          for g, group in enumerate((POLICY, REPLAY))]  # fmt: skip
        self.assertAlmostEqual(lowest_success[0], lowest_success[1])  # both start on the same line
        apart = chart()
        shared, in_lanes = (c.strip_edge_mm[True] - c.band_high_mm for c in (drawn, apart))
        self.assertLess(shared, in_lanes + 1e-9)

    def test_a_mark_wears_its_groups_marker_and_colour_and_a_white_ring(self):
        drawn = chart()
        for series in (style.CLUSTERED, style.REPLAY):
            for line in lines(drawn, f"trial:{series.label}:"):
                self.assertEqual(line.get_marker(), series.marker)
                self.assertEqual(matplotlib_hex(line.get_markerfacecolor()), series.color)
                self.assertEqual(matplotlib_hex(line.get_markeredgecolor()), "#ffffff")
        self.assertEqual(len(lines(drawn, "trial:clustered:success")), sum(POLICY.successes))
        self.assertEqual(len(lines(drawn, "trial:replay:failure")), len(REPLAY.successes) - sum(REPLAY.successes))

    def test_an_open_mark_is_white_inside(self):
        group = MarkGroup(style.CLUSTERED, (1.0, 6.0, 12.0), (True, True, True), (False, True, False))
        drawn = chart(groups=[group])
        faces = [matplotlib_hex(line.get_markerfacecolor()) for line in lines(drawn, "trial:")]
        self.assertEqual(sorted(faces), ["#1c5cab", "#1c5cab", "#ffffff"])
        marks_drawn = lines(drawn, "trial:")
        (open_mark,) = [line for line in marks_drawn if matplotlib_hex(line.get_markerfacecolor()) == "#ffffff"]
        self.assertEqual(matplotlib_hex(open_mark.get_markeredgecolor()), style.BLUE)
        self.assertAlmostEqual(open_mark.get_xdata()[0], drawn.x(6.0))

    def test_marks_shrink_when_their_piles_would_squeeze_the_rate_band(self):
        crowd = MarkGroup(style.CLUSTERED, tuple(0.05 * i for i in range(80)), (True,) * 80)
        drawn = chart(groups=[crowd], height_mm=40.0)
        self.assertLess(drawn.mark_pt, ratechart.MARK_PT)
        self.assertGreaterEqual(drawn.mark_pt, ratechart.SMALLEST_MARK_PT)
        self.assertEqual(chart().mark_pt, ratechart.MARK_PT)

    def test_piles_that_leave_no_room_for_the_band_are_an_error(self):
        crowd = MarkGroup(style.CLUSTERED, (5.0,) * 60, (True,) * 60)
        with self.assertRaisesRegex(FigureError, "no room"):
            chart(groups=[crowd], height_mm=40.0)

    def test_an_empty_strip_still_has_its_room_and_its_name(self):
        only_successes = MarkGroup(style.CLUSTERED, (1.0, 2.0, 30.0), (True, True, True))
        drawn = chart(groups=[only_successes])
        self.assertAlmostEqual(drawn.band_low_mm - ratechart.BAND_GAP_MM - drawn.strip_edge_mm[False],
                               ratechart.EMPTY_STRIP_MM)  # fmt: skip
        self.assertEqual([text.get_text() for text in texts(drawn, "strip:")], ["success", "failure"])


class FrameTests(unittest.TestCase):
    def test_the_distance_axis_has_a_number_every_ten_millimetres_and_its_label(self):
        drawn = chart()
        ticks = texts(drawn, "distance-tick:")
        self.assertEqual([tick.get_text() for tick in ticks], ["0", "10", "20", "30", "40", "50"])
        for tick, distance in zip(ticks, (0.0, 10.0, 20.0, 30.0, 40.0, 50.0), strict=True):
            self.assertAlmostEqual(tick.get_position()[0], drawn.x(distance))
            self.assertEqual(tick.get_ha(), "center")
        (label,) = texts(drawn, "distance-label")
        self.assertEqual(label.get_text(), "Distance to the nearest demonstration, $d$ (mm)")
        self.assertEqual(label.get_fontsize(), style.LABEL_PT)
        (axis,) = lines(drawn, "distance-axis")
        self.assertEqual(list(axis.get_xdata()), [drawn.left_mm, drawn.right_mm])

    def test_the_rate_axis_names_its_levels_and_the_strips_say_what_they_hold(self):
        drawn = chart()
        self.assertEqual([text.get_text() for text in texts(drawn, "rate-tick:")], ["0", "0.5", "1"])
        (label,) = texts(drawn, "rate-label")
        self.assertEqual((label.get_text(), label.get_rotation()), ("Success rate", 90.0))
        names = {text.get_text(): text.get_position()[1] for text in texts(drawn, "strip:")}
        self.assertGreater(names["success"], drawn.band_high_mm)
        self.assertLess(names["failure"], drawn.band_low_mm)
        self.assertEqual(len(lines(drawn, "level:")), 3)

    def test_the_label_of_the_distance_axis_fits_the_page(self):
        drawn = chart(x_label="Distance to the nearest demonstration, $d$ (mm, corrected for scale)")
        (label,) = texts(drawn, "distance-label")
        width = style.text_width_mm(label.get_text(), label.get_fontsize())
        self.assertLess(label.get_position()[0] + width / 2.0, 88.0)
        self.assertGreater(label.get_position()[0] - width / 2.0, 0.0)

    def test_rows_above_the_axis_take_room_from_the_band_and_lie_below_the_failures(self):
        plain, with_rows = chart(), chart(rows=2)
        self.assertEqual(plain.row_y_mm, [])
        self.assertEqual(len(with_rows.row_y_mm), 2)
        top_row, bottom_row = with_rows.row_y_mm
        self.assertGreater(top_row, bottom_row)
        self.assertGreater(bottom_row, ratechart.AXIS_Y_MM + 1.0)
        self.assertLess(top_row + ratechart.ROW_MM / 2.0, with_rows.strip_edge_mm[False])
        self.assertAlmostEqual(top_row - bottom_row, ratechart.ROW_MM)
        self.assertLess(with_rows.band_high_mm - with_rows.band_low_mm, plain.band_high_mm - plain.band_low_mm)

    def test_a_shaded_stretch_covers_its_distances_from_the_axis_to_the_top_and_carries_a_note(self):
        drawn = chart()
        drawn.shade(5.6, 31.0, ["no trial between", "5.6 and 31.0 mm"])
        (patch,) = [patch for patch in drawn.page.axes.patches if patch.get_gid() == "untested-stretch"]
        self.assertAlmostEqual(patch.get_x(), drawn.x(5.6))
        self.assertAlmostEqual(patch.get_x() + patch.get_width(), drawn.x(31.0))
        self.assertAlmostEqual(patch.get_y(), ratechart.AXIS_Y_MM)
        self.assertAlmostEqual(patch.get_y() + patch.get_height(), drawn.top_mm)
        notes = texts(drawn, "untested-note")
        self.assertEqual([note.get_text() for note in notes], ["no trial between", "5.6 and 31.0 mm"])
        for note in notes:
            self.assertAlmostEqual(note.get_position()[0], (drawn.x(5.6) + drawn.x(31.0)) / 2.0)
            self.assertTrue(drawn.band_low_mm < note.get_position()[1] < drawn.band_high_mm)
        self.assertLess(patch.get_zorder(), min(line.get_zorder() for line in drawn.page.axes.lines))

    def test_no_text_of_the_frame_is_below_seven_points(self):
        drawn = chart(rows=2)
        self.assertGreaterEqual(min(text.get_fontsize() for text in drawn.page.texts()), 7.0)


def matplotlib_hex(colour) -> str:
    from matplotlib.colors import to_hex

    return to_hex(colour)


if __name__ == "__main__":
    unittest.main()
