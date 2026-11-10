"""Figure 1: the earlier evaluation by distance. What is read from the CSV, what is drawn, and the file it makes."""

import csv
import tempfile
import unittest
from pathlib import Path

from matplotlib.colors import to_hex

import pdf_facts
from tolerance_radius.figures import HEIGHTS_MM, ratechart, style
from tolerance_radius.figures import earlier_evaluation as ee
from tolerance_radius.figures.earlier_evaluation import EarlierTrial
from tolerance_radius.figures.style import FigureError

CSV_PATH = pdf_facts.REPO / "analysis" / "pilot_august" / "trials_with_distance.csv"
TRIALS = ee.read_trials(CSV_PATH)


def gids(page, prefix: str) -> list:
    artists = [*page.axes.lines, *page.axes.patches, *page.axes.texts]
    return [artist for artist in artists if (artist.get_gid() or "").startswith(prefix)]


def rate_levels(page) -> dict[str, float]:
    """Page y of the rates 0, 0.5 and 1, read from the hairlines drawn at those levels."""
    return {line.get_gid().removeprefix("level:"): float(line.get_ydata()[0]) for line in gids(page, "level:")}


class ReadTrialsTests(unittest.TestCase):
    def test_the_file_holds_the_31_trials_in_two_groups(self):
        self.assertEqual(len(TRIALS), 31)
        self.assertEqual([trial.trial for trial in TRIALS], list(range(1, 32)))
        near = [trial for trial in TRIALS if not trial.held_out]
        far = [trial for trial in TRIALS if trial.held_out]
        self.assertEqual((len(near), sum(trial.success for trial in near)), (19, 19))
        self.assertEqual((len(far), sum(trial.success for trial in far)), (12, 1))
        self.assertEqual((min(t.distance_mm for t in near), max(t.distance_mm for t in near)), (0.63, 5.63))
        self.assertEqual((min(t.distance_mm for t in far), max(t.distance_mm for t in far)), (31.01, 47.14))

    def test_one_trial_has_its_position_from_a_still(self):
        from_still = [trial for trial in TRIALS if trial.from_still]
        self.assertEqual([(trial.trial, trial.distance_mm, trial.success, trial.held_out) for trial in from_still],
                         [(1, 1.49, True, False)])  # fmt: skip

    def test_the_distance_is_to_the_nearest_demonstration_the_policy_was_trained_on(self):
        # Trial 17 is the one trial for which the two distances of the file differ: 2.76 mm to the nearest of the 49
        # trained demonstrations, 1.11 mm to an episode that was held out of training.
        self.assertEqual(TRIALS[16].distance_mm, 2.76)

    def write(self, rows: list[dict]) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        self.addCleanup(lambda: Path(handle.name).unlink())
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return Path(handle.name)

    def row(self, **changes) -> dict:
        return {"trial": "1", "d_nearest_trained49_mm": "1.49", "success": "1", "kind": "id",
                "position_source": "video frame"} | changes  # fmt: skip

    def test_a_file_without_a_needed_column_is_refused(self):
        row = self.row()
        del row["kind"]
        with self.assertRaisesRegex(FigureError, "kind"):
            ee.read_trials(self.write([row]))

    def test_a_row_that_cannot_be_read_is_refused_with_its_line(self):
        for changes in ({"success": "yes"}, {"kind": "elsewhere"}, {"d_nearest_trained49_mm": ""},
                        {"d_nearest_trained49_mm": "-1"}, {"d_nearest_trained49_mm": "nan"}):  # fmt: skip
            with self.assertRaisesRegex(FigureError, "line 3"):
                ee.read_trials(self.write([self.row(), self.row(**changes)]))

    def test_a_file_with_no_trial_is_refused(self):
        path = self.write([self.row()])
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(FigureError, "no trial"):
            ee.read_trials(path)


class RatesTests(unittest.TestCase):
    def test_the_two_rates_are_the_ones_of_the_re_analysis(self):
        # analysis/pilot_august/README.md: 19 of 19 (Wilson 83.2 to 100 %) and 1 of 12 (Wilson 1.5 to 35.4 %).
        near = ee.group_rate([trial for trial in TRIALS if not trial.held_out])
        far = ee.group_rate([trial for trial in TRIALS if trial.held_out])
        self.assertEqual((near.successes, near.trials, near.rate), (19, 19, 1.0))
        self.assertEqual((round(100 * near.low, 1), near.high), (83.2, 1.0))
        self.assertEqual((far.successes, far.trials), (1, 12))
        self.assertEqual((round(100 * far.low, 1), round(100 * far.high, 1)), (1.5, 35.4))
        self.assertEqual((near.nearest_mm, near.farthest_mm), (0.63, 5.63))
        self.assertEqual((far.nearest_mm, far.farthest_mm), (31.01, 47.14))
        self.assertIsNone(ee.group_rate([]))

    def test_a_stretch_with_no_trial_is_named_by_its_ends(self):
        self.assertEqual(ee.stretch_note(5.63, 31.01, 50.0)[0], ["no trial between", "5.6 and 31.0 mm"])
        self.assertEqual(ee.stretch_note(0.0, 18.0, 50.0)[0], ["no trial", "below 18.0 mm"])
        self.assertEqual(ee.stretch_note(32.0, 50.0, 50.0)[0], ["no trial", "beyond 32.0 mm"])
        for notes in (ee.stretch_note(5.63, 31.01, 50.0), ee.stretch_note(0.0, 18.0, 50.0)):
            self.assertEqual(notes[-1], ["no trial"])  # the shortest way of saying it


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = ee.figure(TRIALS)
        cls.texts = [text.get_text() for text in cls.page.texts()]

    def test_it_is_a_column_wide_and_42_mm_high(self):
        self.assertEqual((self.page.width_mm, self.page.height_mm), (88.0, 42.0))
        self.assertEqual(HEIGHTS_MM[1], 42.0)

    def test_every_trial_is_one_mark_success_above_and_failure_below(self):
        marks_drawn = gids(self.page, "trial:")
        self.assertEqual(len(marks_drawn), 31)
        levels = rate_levels(self.page)
        successes = [mark for mark in marks_drawn if mark.get_gid().endswith(":success")]
        failures = [mark for mark in marks_drawn if mark.get_gid().endswith(":failure")]
        self.assertEqual((len(successes), len(failures)), (20, 11))
        self.assertGreater(min(mark.get_ydata()[0] for mark in successes), levels["1"])
        self.assertLess(max(mark.get_ydata()[0] for mark in failures), levels["0"])

    def test_the_two_kinds_of_cell_have_markers_of_their_own(self):
        circles = [mark for mark in gids(self.page, "trial:") if mark.get_marker() == "o"]
        triangles = [mark for mark in gids(self.page, "trial:") if mark.get_marker() == "^"]
        self.assertEqual((len(circles), len(triangles)), (19, 12))
        # Every circle lies left of the shaded stretch and every triangle right of it.
        (shade,) = gids(self.page, "untested-stretch")
        self.assertLess(max(mark.get_xdata()[0] for mark in circles), shade.get_x() + 1e-9)
        self.assertGreater(min(mark.get_xdata()[0] for mark in triangles), shade.get_x() + shade.get_width() - 1e-9)

    def test_the_trial_measured_from_a_still_has_the_one_open_mark(self):
        open_marks = [mark for mark in gids(self.page, "trial:") if to_hex(mark.get_markerfacecolor()) == "#ffffff"]
        self.assertEqual(len(open_marks), 1)
        self.assertEqual(open_marks[0].get_marker(), "o")
        self.assertEqual(to_hex(open_marks[0].get_markeredgecolor()), style.BLUE)

    def test_the_stretch_with_no_trial_is_shaded_from_5_6_to_31_0_mm_and_says_so(self):
        (shade,) = gids(self.page, "untested-stretch")
        # Read the distances back from the axis's own numbers.
        ticks = {float(text.get_text()): text.get_position()[0] for text in gids(self.page, "distance-tick:")}
        per_mm = (ticks[50.0] - ticks[0.0]) / 50.0
        self.assertAlmostEqual((shade.get_x() - ticks[0.0]) / per_mm, 5.63, places=6)
        self.assertAlmostEqual((shade.get_x() + shade.get_width() - ticks[0.0]) / per_mm, 31.01, places=6)
        self.assertEqual([text.get_text() for text in gids(self.page, "untested-note")],
                         ["no trial between", "5.6 and 31.0 mm"])  # fmt: skip
        self.assertEqual(to_hex(shade.get_facecolor()), style.SHADE)

    def test_no_mark_lies_in_the_shaded_stretch_not_even_the_legends(self):
        (shade,) = gids(self.page, "untested-stretch")
        for line in self.page.axes.lines:
            if line.get_marker() not in ("None", None, ""):
                x = line.get_xdata()[0]
                self.assertFalse(shade.get_x() < x < shade.get_x() + shade.get_width(), line.get_gid())

    def test_each_group_has_its_rate_its_interval_and_its_count(self):
        self.assertIn("19 of 19", self.texts)
        self.assertIn("1 of 12", self.texts)
        levels = rate_levels(self.page)
        height = levels["1"] - levels["0"]
        intervals = {line.get_gid(): [(y - levels["0"]) / height for y in line.get_ydata()]
                     for line in gids(self.page, "rate-interval:")}  # fmt: skip
        near = intervals["rate-interval:cell with demonstrations"]
        far = intervals["rate-interval:held-out cell"]
        self.assertAlmostEqual(near[0], 0.8318, places=3)
        self.assertAlmostEqual(near[1], 1.0)
        self.assertAlmostEqual(far[0], 0.0149, places=3)
        self.assertAlmostEqual(far[1], 0.3539, places=3)
        points = {line.get_gid(): (line.get_ydata()[0] - levels["0"]) / height for line in gids(self.page, "rate:")}
        self.assertAlmostEqual(points["rate:cell with demonstrations"], 1.0)
        self.assertAlmostEqual(points["rate:held-out cell"], 1.0 / 12.0)

    def test_the_far_group_is_one_rate_not_two_bins(self):
        # The one success sits at 40.01 mm, a hundredth of a millimetre above a bin's edge: the pooled 1 of 12 is
        # shown, and no rate is drawn for 30 to 40 or 40 to 50 mm.
        self.assertEqual(len(gids(self.page, "rate:")), 2)
        self.assertNotIn("0 of 8", self.texts)
        self.assertNotIn("1 of 4", self.texts)

    def test_no_curve_is_drawn(self):
        for line in self.page.axes.lines:
            self.assertLessEqual(len(line.get_xdata()), 2, f"{line.get_gid()} is drawn through more than two points")
        self.assertEqual(len(self.page.axes.collections), 0)

    def test_the_legend_names_the_three_marks_outside_the_strips(self):
        self.assertEqual([text.get_text() for text in gids(self.page, "legend-text")],
                         ["cell with demonstrations", "same, from a still image", "held-out cell"])  # fmt: skip
        levels = rate_levels(self.page)
        legend_marks = gids(self.page, "legend-mark")
        self.assertEqual([mark.get_marker() for mark in legend_marks], ["o", "o", "^"])
        for mark in legend_marks:
            self.assertTrue(levels["0"] < mark.get_ydata()[0] < levels["1"])  # in the rate band, where no trial is
        self.assertEqual(to_hex(legend_marks[1].get_markerfacecolor()), "#ffffff")

    def test_the_axis_is_in_plain_words_with_its_unit(self):
        self.assertIn("Distance to the nearest trained demonstration, $d$ (mm)", self.texts)
        self.assertIn("Success rate", self.texts)
        self.assertEqual([text.get_text() for text in gids(self.page, "distance-tick:")],
                         ["0", "10", "20", "30", "40", "50"])  # fmt: skip

    def test_it_has_no_title_no_stamp_and_no_small_text(self):
        self.assertEqual(self.page.stamps, [])
        self.assertGreaterEqual(min(text.get_fontsize() for text in self.page.texts()), 7.0)
        self.assertEqual(self.page.figure.texts, [])
        self.assertEqual(self.page.axes.get_title(), "")

    def test_no_text_runs_off_the_page(self):
        for text in self.page.texts():
            if text.get_rotation():
                continue
            width = style.text_width_mm(text.get_text(), text.get_fontsize())
            x = text.get_position()[0]
            left = {"left": x, "center": x - width / 2.0, "right": x - width}[text.get_ha()]
            self.assertGreaterEqual(left, 0.0, text.get_text())
            self.assertLessEqual(left + width, 88.0, text.get_text())

    def test_a_legend_with_no_free_corner_is_an_error_not_a_legend_over_the_data(self):
        # One group succeeds everywhere and the other fails everywhere: their rates run along the top and the bottom
        # of the rate band, from end to end.
        everywhere = [EarlierTrial(i, 4.0 * i, True, False, False) for i in range(13)]
        everywhere += [EarlierTrial(13 + i, 1.0 + 4.0 * i, False, True, False) for i in range(13)]
        with self.assertRaisesRegex(FigureError, "legend"):
            ee.figure(everywhere)

    def test_trials_without_a_long_untested_stretch_get_no_shading(self):
        even = [EarlierTrial(i, 4.0 * i, True, False, False) for i in range(8)]
        even += [EarlierTrial(8 + i, 32.0 + 4.0 * i, False, True, False) for i in range(4)]
        page = ee.figure(even)
        self.assertEqual(gids(page, "untested-stretch"), [])
        self.assertEqual(len(gids(page, "trial:")), 12)
        self.assertNotIn("same, from a still image", [text.get_text() for text in page.texts()])

    def test_no_trials_is_an_error(self):
        with self.assertRaises(FigureError):
            ee.figure([])


class FileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = style.pdf_bytes(ee.figure(TRIALS))

    def test_the_page_is_88_by_42_mm(self):
        width, height = pdf_facts.page_size_mm(self.data)
        self.assertAlmostEqual(width, 88.0, delta=0.5)
        self.assertAlmostEqual(height, 42.0, delta=0.5)

    def test_no_text_is_under_seven_points(self):
        sizes = pdf_facts.text_sizes_pt(self.data)
        self.assertGreater(len(sizes), 20)
        self.assertGreaterEqual(min(sizes), 7.0)
        self.assertLessEqual(max(sizes), 8.0)

    def test_the_bytes_say_nothing_of_the_maker(self):
        self.assertEqual(pdf_facts.forbidden_hits(self.data), [])
        self.assertEqual(pdf_facts.names_this_machine(self.data), [])
        self.assertNotIn(b"CreationDate", self.data)

    def test_the_same_trials_give_the_same_bytes(self):
        self.assertEqual(style.pdf_bytes(ee.figure(ee.read_trials(CSV_PATH))), self.data)

    def test_the_fonts_are_embedded_truetype(self):
        self.assertEqual(pdf_facts.font_kinds(self.data), {b"Type0", b"CIDFontType2"})
        self.assertIn(b"/FontFile2", self.data)

    def test_the_frames_layout_left_room_for_the_rate_band(self):
        distances, successes = tuple(t.distance_mm for t in TRIALS), tuple(t.success for t in TRIALS)
        chart = ratechart.RateChart(
            style.Page(42.0),
            [ratechart.MarkGroup(ee.IN_TRAINED_CELL, distances, successes)],
            x_max_mm=50.0, x_label="", separate_lanes=False,
        )  # fmt: skip
        self.assertEqual(chart.mark_pt, ratechart.MARK_PT)  # the marks are full size
        self.assertGreater(chart.band_high_mm - chart.band_low_mm, 15.0)


if __name__ == "__main__":
    unittest.main()
