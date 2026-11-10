"""Figure 3: the response curves of the clustered policy and of the replay, from synthetic analysis results."""

import unittest

from matplotlib.colors import to_hex

import fixtures_results
from tolerance_radius.figures import HEIGHTS_MM, ratechart, results, style
from tolerance_radius.figures import response_curves as rc
from tolerance_radius.figures.results import ResultsError
from tolerance_radius.figures.style import FigureError


def artists(page, prefix: str) -> list:
    """Everything drawn whose identifier starts with `prefix`."""
    drawn = [*page.axes.lines, *page.axes.patches, *page.axes.texts, *page.axes.collections]
    return [artist for artist in drawn if (artist.get_gid() or "").startswith(prefix)]


class Read:
    """A figure read back from what was drawn: distances from the axis's own numbers, rates from its levels."""

    def __init__(self, page):
        self.page = page
        ticks = {float(text.get_text()): text.get_position()[0] for text in artists(page, "distance-tick:")}
        self.x0 = ticks[0.0]
        self.per_mm = (ticks[10.0] - ticks[0.0]) / 10.0
        levels = {line.get_gid().removeprefix("level:"): float(line.get_ydata()[0]) for line in artists(page, "level:")}
        self.y0, self.y1 = levels["0"], levels["1"]
        (axis,) = artists(page, "distance-axis")
        self.axis_left, self.axis_right = (float(x) for x in axis.get_xdata())

    def d(self, x: float) -> float:
        """The distance at a page x."""
        return (x - self.x0) / self.per_mm

    def rate(self, y: float) -> float:
        """The success rate at a page y."""
        return (y - self.y0) / (self.y1 - self.y0)

    def axis_end(self) -> float:
        """The distance at which the axis ends (the right end of the axis less the room kept inside it)."""
        return self.d(self.axis_right - ratechart.INSET_MM)

    def texts(self, prefix: str = "") -> list[str]:
        return [text.get_text() for text in artists(self.page, prefix) if hasattr(text, "get_text")]


def drawn(name: str = "base", **changes) -> Read:
    return Read(rc.figure(fixtures_results.changed(name, **changes)))


def with_gap(low_mm: float, high_mm: float) -> dict:
    """The base results without the threshold trials between two distances, and the bins counted again."""
    made = fixtures_results.load("base")
    for condition in ("act_clustered", "replay"):
        entry = made["conditions"][condition]
        entry["trials"] = [trial for trial in entry["trials"] if not low_mm < trial["d_mm"] < high_mm]
        entry["bins"] = [
            {"lo_mm": b.lo_mm, "hi_mm": b.hi_mm, "n": b.n, "k": b.k, "wilson_lo": b.wilson_lo, "wilson_hi": b.wilson_hi}
            for b in results.binned(results.trials(made, condition), 10.0, 0.95)
        ]
    return made


class ShownTests(unittest.TestCase):
    def test_the_block_fit_is_drawn_with_the_trials_of_its_block_and_no_others(self):
        made = fixtures_results.load("base")
        policy, replay = rc.conditions_shown(made)
        self.assertEqual((policy.name, replay.name), ("policy", "replay"))
        self.assertEqual(len(results.trials(made, "act_clustered")), 54)  # 34 of its threshold block, 20 others
        self.assertEqual([trial.block for trial in policy.trials], ["T1a"] * 34)
        self.assertEqual([trial.block for trial in replay.trials], ["T1b"] * 22)
        self.assertEqual(policy.fit, results.fit(made, "act_clustered", "block"))
        self.assertEqual(replay.fit, results.fit(made, "replay", "block"))
        self.assertEqual((policy.series, replay.series), (style.CLUSTERED, style.REPLAY))

    def test_the_rates_are_counted_from_the_trials_drawn(self):
        made = fixtures_results.load("base")
        policy, replay = rc.conditions_shown(made)
        self.assertEqual(sum(bin_.n for bin_ in policy.bins), 34)
        self.assertEqual(list(policy.bins), results.binned(policy.trials, 10.0, 0.95))
        # The replay's only trials are its threshold block's: there the figure's rates are the file's own bins.
        self.assertTrue(results.same_bins(replay.bins, results.file_bins(made, "replay")))
        self.assertFalse(results.same_bins(policy.bins, results.file_bins(made, "act_clustered")))

    def test_a_rate_stands_at_the_mean_distance_of_its_trials(self):
        policy, _ = rc.conditions_shown(fixtures_results.load("no_spread_threshold"))
        for bin_ in policy.bins:
            inside = [trial.d_mm for trial in policy.trials if bin_.lo_mm <= trial.d_mm < bin_.hi_mm]
            self.assertEqual(len(inside), bin_.n)
            self.assertAlmostEqual(bin_.mean_d_mm, sum(inside) / len(inside))
        # The far anchors, asked for at 40 mm, fall on both sides of 40: the last stretch holds trials just beyond it.
        last = policy.bins[-1]
        self.assertEqual((last.lo_mm, last.hi_mm), (40.0, 50.0))
        self.assertLess(last.mean_d_mm, 41.0)

    def test_a_condition_without_a_block_fit_is_not_drawn_whatever_trials_it_has(self):
        made = fixtures_results.changed("base", **{"conditions.act_clustered.block_fit": None})
        self.assertEqual(len(results.trials(made, "act_clustered")), 54)
        self.assertEqual([item.name for item in rc.conditions_shown(made)], ["replay"])
        self.assertIsNone(rc.shown(made, "act_clustered", style.CLUSTERED, "policy"))

    def test_the_pooled_fit_can_be_drawn_with_all_the_trials_it_was_made_from(self):
        made = fixtures_results.load("base")
        policy, replay = rc.conditions_shown(made, "pooled")
        self.assertEqual(policy.fit.blocks, ("T1a", "T2"))
        self.assertEqual(len(policy.trials), 54)
        self.assertEqual(replay.fit.blocks, ("T1b",))  # the replay has no pooled fit and keeps its block fit
        self.assertTrue(results.same_bins(policy.bins, results.file_bins(made, "act_clustered")))
        with self.assertRaisesRegex(FigureError, "fits"):
            rc.conditions_shown(made, "other")

    def test_bins_that_this_codes_counting_does_not_give_are_refused_and_the_first_is_named(self):
        made = fixtures_results.load("base")
        bins = made["conditions"]["replay"]["bins"]
        bins[1]["k"] += 1
        with self.assertRaises(ResultsError) as caught:
            rc.figure(made)
        said = str(caught.exception)
        self.assertIn("conditions.replay.bins are not the counts of the trials listed", said)
        self.assertIn("in 10 to 20 mm", said)
        self.assertIn("stretches of 10 mm closed below", said)
        for change in (lambda b: b[0].update(wilson_hi=b[0]["wilson_hi"] - 0.01), lambda b: b.pop(), lambda b: b.pop(0),
                       lambda b: b[0].update(lo_mm=5.0)):  # fmt: skip
            made = fixtures_results.load("base")
            change(made["conditions"]["act_clustered"]["bins"])
            with self.assertRaisesRegex(ResultsError, r"conditions\.act_clustered\.bins"):
                rc.figure(made)

    def test_a_fit_whose_blocks_hold_no_trial_is_refused(self):
        made = fixtures_results.load("base")
        made["conditions"]["replay"]["block_fit"]["blocks"] = ["T9"]
        with self.assertRaisesRegex(ResultsError, "no trial is in them"):
            rc.figure(made)

    def test_there_is_something_to_draw_when_a_threshold_block_was_run(self):
        self.assertTrue(rc.has_something_to_draw(fixtures_results.load("base")))
        only_replay = fixtures_results.changed("base", **{"conditions.act_clustered.block_fit": None})
        self.assertTrue(rc.has_something_to_draw(only_replay))
        neither = fixtures_results.changed("base", **{"conditions.act_clustered.block_fit": None,
                                                      "conditions.replay.block_fit": None})  # fmt: skip
        self.assertFalse(rc.has_something_to_draw(neither))
        with self.assertRaisesRegex(FigureError, "no threshold block"):
            rc.figure(neither)
        nothing = fixtures_results.changed("base", **{"conditions.act_clustered": None, "conditions.replay": None})
        self.assertFalse(rc.has_something_to_draw(nothing))


class AxisTests(unittest.TestCase):
    def test_the_axis_ends_at_the_first_multiple_of_five_that_holds_every_trial(self):
        for name, end in (("base", 40.0), ("no_spread_threshold", 45.0), ("threshold_blocks_only", 50.0)):
            shown = rc.conditions_shown(fixtures_results.load(name))
            farthest = max(trial.d_mm for item in shown for trial in item.trials)
            self.assertEqual(rc.axis_end_mm(shown, 10.0), end, name)
            self.assertTrue(end - 5.0 < farthest <= end, name)
            self.assertAlmostEqual(drawn(name).axis_end(), end, places=6)

    def test_an_end_of_an_interval_that_the_trials_close_is_on_the_axis_and_an_open_one_is_not(self):
        shown = rc.conditions_shown(fixtures_results.load("open_above"))
        policy = shown[0].fit
        self.assertTrue(policy.open_above)
        self.assertGreater(policy.r50_hi_mm, 60.0)  # where the prior's grid ends, not where the radius is
        farthest = max(trial.d_mm for item in shown for trial in item.trials)
        self.assertLess(farthest, 50.0 + 1e-9)
        self.assertGreater(policy.r50_lo_mm, farthest)
        self.assertEqual(rc.axis_end_mm(shown, 10.0), 55.0)  # holds the lower end, 53 mm, and stops there

    def test_the_axis_is_never_shorter_than_one_bin(self):
        made = with_gap(3.0, 1000.0)
        for condition in ("act_clustered", "replay"):
            made["conditions"][condition]["block_fit"] |= {"r50_lo_mm": 0.5, "r50_median_mm": 1.0, "r50_hi_mm": 2.0}
        self.assertEqual(rc.axis_end_mm(rc.conditions_shown(made), 10.0), 10.0)

    def test_the_label_says_whether_distances_are_corrected_for_scale(self):
        self.assertEqual(drawn("base").texts("distance-label"),
                         ["Distance to the nearest demonstration, $d$ (mm, corrected for scale)"])  # fmt: skip
        self.assertEqual(drawn("no_spread_threshold").texts("distance-label"),
                         ["Distance to the nearest demonstration, $d$ (mm, as read)"])  # fmt: skip
        self.assertEqual(drawn("base").texts("rate-label"), ["Success rate"])
        self.assertEqual(drawn("base").texts("distance-tick:"), ["0", "10", "20", "30", "40"])
        self.assertEqual(drawn("no_spread_threshold").texts("distance-tick:"), ["0", "10", "20", "30", "40"])  # to 45


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.made = fixtures_results.load("base")
        cls.read = Read(rc.figure(cls.made))
        cls.page = cls.read.page
        cls.policy, cls.replay = rc.conditions_shown(cls.made)

    def test_it_is_a_column_wide_and_46_mm_high(self):
        self.assertEqual((self.page.width_mm, self.page.height_mm, HEIGHTS_MM[3]), (88.0, 46.0, 46.0))

    def test_every_trial_of_the_two_threshold_blocks_is_one_mark_at_its_distance(self):
        for item, count in ((self.policy, 34), (self.replay, 22)):
            marks_drawn = artists(self.page, f"trial:{item.series.label}:")
            self.assertEqual(len(marks_drawn), count)
            self.assertEqual(sorted(round(self.read.d(mark.get_xdata()[0]), 6) for mark in marks_drawn),
                             sorted(round(trial.d_mm, 6) for trial in item.trials))  # fmt: skip
            for mark in marks_drawn:
                self.assertEqual(mark.get_marker(), item.series.marker)
                self.assertEqual(to_hex(mark.get_markerfacecolor()), item.series.color)

    def test_successes_are_above_the_rate_band_and_failures_below(self):
        for item in (self.policy, self.replay):
            successes = artists(self.page, f"trial:{item.series.label}:success")
            failures = artists(self.page, f"trial:{item.series.label}:failure")
            self.assertEqual(len(successes), sum(trial.success for trial in item.trials))
            self.assertEqual(len(successes) + len(failures), len(item.trials))
            self.assertGreater(min(mark.get_ydata()[0] for mark in successes), self.read.y1)
            self.assertLess(max(mark.get_ydata()[0] for mark in failures), self.read.y0)

    def test_the_marks_are_no_smaller_than_the_smallest_a_figure_may_use(self):
        sizes = {mark.get_markersize() for mark in artists(self.page, "trial:")}
        self.assertGreaterEqual(min(sizes), style.marker_size("s", ratechart.SMALLEST_MARK_PT))
        self.assertGreaterEqual(self.read.y1 - self.read.y0, 13.0)  # and the rate band keeps a useful height

    def test_each_curve_runs_from_its_conditions_nearest_trial_to_its_farthest_and_no_further(self):
        for item in (self.policy, self.replay):
            (curve,) = artists(self.page, f"curve:{item.name}")
            distances = [trial.d_mm for trial in item.trials]
            self.assertAlmostEqual(self.read.d(curve.get_xdata()[0]), min(distances), places=6)
            self.assertAlmostEqual(self.read.d(curve.get_xdata()[-1]), max(distances), places=6)
            self.assertGreater(len(curve.get_xdata()), 30)
            for x, y in zip(curve.get_xdata(), curve.get_ydata(), strict=True):
                self.assertAlmostEqual(self.read.rate(y), item.fit.mean_at(self.read.d(x)), places=6)
            self.assertEqual(to_hex(curve.get_color()), item.series.color)
            self.assertEqual(len(artists(self.page, f"band:{item.name}")), 1)
        # The replay was not tested as far out as the policy: its curve stops earlier.
        self.assertLess(artists(self.page, "curve:replay")[0].get_xdata()[-1],
                        artists(self.page, "curve:policy")[0].get_xdata()[-1] - 10.0)  # fmt: skip

    def test_the_two_curves_differ_in_line_style_as_well_as_in_colour(self):
        (policy,), (replay,) = artists(self.page, "curve:policy"), artists(self.page, "curve:replay")
        self.assertTrue(policy.is_dashed() is False and replay.is_dashed() is True)
        self.assertNotEqual(to_hex(policy.get_color()), to_hex(replay.get_color()))

    def test_the_band_of_a_curve_is_the_fits_own(self):
        for item in (self.policy, self.replay):
            (band,) = artists(self.page, f"band:{item.name}")
            for x, y in band.get_paths()[0].vertices:
                d = self.read.d(x)
                low = results.interpolate(item.fit.curve_d_mm, item.fit.curve_lo, d)
                high = results.interpolate(item.fit.curve_d_mm, item.fit.curve_hi, d)
                self.assertTrue(low - 1e-6 <= self.read.rate(y) <= high + 1e-6)

    def test_each_stretch_of_distance_has_its_rate_and_wilson_interval_where_its_trials_are(self):
        for item in (self.policy, self.replay):
            points = artists(self.page, f"rate:{item.name}")
            intervals = artists(self.page, f"rate-interval:{item.name}")
            self.assertEqual((len(points), len(intervals)), (len(item.bins), len(item.bins)))
            for bin_, point, interval in zip(item.bins, points, intervals, strict=True):
                self.assertAlmostEqual(self.read.rate(point.get_ydata()[0]), bin_.rate, places=6)
                self.assertAlmostEqual(self.read.rate(interval.get_ydata()[0]), bin_.wilson_lo, places=6)
                self.assertAlmostEqual(self.read.rate(interval.get_ydata()[1]), bin_.wilson_hi, places=6)
                self.assertEqual(point.get_xdata()[0], interval.get_xdata()[0])
                # At the mean distance of the stretch's trials, give or take what keeps two marks apart.
                self.assertAlmostEqual(self.read.d(point.get_xdata()[0]), bin_.mean_d_mm, delta=1.0)
                self.assertTrue(bin_.lo_mm - 1.0 <= self.read.d(point.get_xdata()[0]) <= bin_.hi_mm + 1.0)
                self.assertGreater(point.get_markersize(), artists(self.page, "trial:")[0].get_markersize())

    def test_no_two_rates_cover_each_other(self):
        places = sorted(point.get_xdata()[0] for point in artists(self.page, "rate:"))
        for first, second in zip(places, places[1:], strict=False):
            self.assertGreaterEqual(second - first, rc.RATES_APART_MM - 1e-9)

    def test_each_radius_is_a_mark_on_a_bar_over_its_interval_with_a_line_up_to_the_curve(self):
        rows = {}
        for item in (self.policy, self.replay):
            fit = item.fit
            (bar,) = artists(self.page, f"radius-interval:{item.name}")
            self.assertAlmostEqual(self.read.d(bar.get_xdata()[0]), fit.r50_lo_mm, places=6)
            self.assertAlmostEqual(self.read.d(bar.get_xdata()[1]), fit.r50_hi_mm, places=6)
            ends = artists(self.page, f"radius-end:{item.name}")
            self.assertEqual(sorted(round(self.read.d(end.get_xdata()[0]), 6) for end in ends),
                             [fit.r50_lo_mm, fit.r50_hi_mm])  # fmt: skip
            self.assertEqual(artists(self.page, f"radius-open:{item.name}"), [])
            (mark,) = artists(self.page, f"radius:{item.name}")
            self.assertAlmostEqual(self.read.d(mark.get_xdata()[0]), fit.r50_median_mm, places=6)
            self.assertEqual(mark.get_marker(), item.series.marker)
            (line,) = artists(self.page, f"radius-line:{item.name}")
            self.assertEqual(list(line.get_xdata()), [mark.get_xdata()[0]] * 2)
            self.assertEqual(line.get_ydata()[0], mark.get_ydata()[0])
            self.assertAlmostEqual(self.read.rate(line.get_ydata()[1]), fit.mean_at(fit.r50_median_mm), places=6)
            rows[item.name] = mark.get_ydata()[0]
            self.assertEqual(bar.get_ydata()[0], rows[item.name])
        # The rows lie between the failures and the distance axis, the policy's above the replay's.
        failures = [mark for mark in artists(self.page, "trial:") if "failure" in mark.get_gid()]
        lowest_failure = min(mark.get_ydata()[0] for mark in failures)
        self.assertTrue(ratechart.AXIS_Y_MM < rows["replay"] < rows["policy"] < lowest_failure)

    def test_the_rows_are_named_with_each_conditions_mark_and_say_what_they_hold(self):
        self.assertEqual(self.read.texts("row-name:"), ["policy", "replay"])
        for item in (self.policy, self.replay):
            (mark,) = artists(self.page, f"row-mark:{item.name}")
            (name,) = artists(self.page, f"row-name:{item.name}")
            self.assertEqual(mark.get_marker(), item.series.marker)
            self.assertEqual(to_hex(mark.get_markerfacecolor()), item.series.color)
            self.assertLess(mark.get_xdata()[0], name.get_position()[0])
            self.assertLess(name.get_position()[0], self.read.axis_left)
            self.assertGreater(mark.get_xdata()[0], 0.8)
        (label,) = artists(self.page, "rows-name")
        self.assertEqual(label.get_text(), "tolerance radius")
        width = style.text_width_mm(label.get_text())
        self.assertAlmostEqual(label.get_position()[0] + width, self.read.axis_right)
        bars = [bar.get_xdata() for bar in artists(self.page, "radius-interval:")]
        self.assertTrue(all(end < label.get_position()[0] - 1.0 for _, end in bars))

    def test_nothing_was_untested_so_nothing_is_shaded(self):
        self.assertEqual(artists(self.page, "untested-"), [])
        self.assertEqual(rc.untested_stretches([self.policy, self.replay], 40.0, 10.0), [])

    def test_a_draft_says_draft_and_results_that_count_do_not(self):
        self.assertEqual(self.page.stamps, ["DRAFT"])
        self.assertEqual([text.get_text() for text in artists(self.page, "stamp:")], ["DRAFT"])
        counted = rc.figure(fixtures_results.changed("base", draft=False))
        self.assertEqual((counted.stamps, artists(counted, "stamp:")), ([], []))
        self.assertEqual(len(artists(counted, "trial:")), 56)

    def test_no_text_is_small_and_none_runs_off_the_page(self):
        for name in fixtures_results.CASES:
            page = rc.figure(fixtures_results.load(name))
            for text in page.texts():
                self.assertGreaterEqual(text.get_fontsize(), 7.0)
                if text.get_rotation() or text.get_gid().startswith("stamp:"):
                    continue
                width = style.text_width_mm(text.get_text(), text.get_fontsize())
                x = text.get_position()[0]
                left = {"left": x, "center": x - width / 2.0, "right": x - width}[text.get_ha()]
                self.assertGreaterEqual(left, 0.0, (name, text.get_text()))
                self.assertLessEqual(left + width, 88.0, (name, text.get_text()))
            for line in page.axes.lines:
                for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
                    self.assertTrue(0.0 <= x <= 88.0 and 0.0 <= y <= 46.0, (name, line.get_gid()))

    def test_the_file_has_no_title_and_the_same_results_give_the_same_file(self):
        self.assertEqual(self.page.axes.get_title(), "")
        self.assertEqual(self.page.figure.texts, [])
        self.assertEqual(style.pdf_bytes(rc.figure(fixtures_results.load("base"))), style.pdf_bytes(self.page))


class OpenIntervalTests(unittest.TestCase):
    def test_an_interval_open_above_runs_off_the_right_end_in_an_arrow_and_marks_no_radius(self):
        read = drawn("open_above")
        fit = rc.conditions_shown(fixtures_results.load("open_above"))[0].fit
        (bar,) = artists(read.page, "radius-interval:policy")
        self.assertAlmostEqual(read.d(bar.get_xdata()[0]), fit.r50_lo_mm, places=6)  # the end the trials do close
        (arrow,) = artists(read.page, "radius-open:policy")
        tip, *base = arrow.get_xy()[:3]
        self.assertAlmostEqual(tip[0], read.axis_right)  # the tip is at the very end of the axis
        self.assertTrue(all(x < tip[0] for x, _ in base))
        self.assertAlmostEqual(bar.get_xdata()[1], base[0][0])  # the bar ends where the arrowhead begins
        self.assertEqual(len(artists(read.page, "radius-end:policy")), 1)
        self.assertEqual(artists(read.page, "radius:policy"), [])  # no median: the trials do not locate the radius
        self.assertEqual(artists(read.page, "radius-line:policy"), [])
        # The replay beside it is bounded and drawn as usual.
        self.assertEqual(len(artists(read.page, "radius:replay")), 1)
        self.assertEqual(len(artists(read.page, "radius-end:replay")), 2)

    def test_what_lies_beyond_the_farthest_trial_is_shaded_and_says_not_tested(self):
        read = drawn("open_above")
        shown = rc.conditions_shown(fixtures_results.load("open_above"))
        farthest = max(trial.d_mm for item in shown for trial in item.trials)
        self.assertEqual(rc.untested_stretches(shown, 55.0, 10.0), [(farthest, 55.0)])
        (shade,) = artists(read.page, "untested-stretch")
        self.assertAlmostEqual(read.d(shade.get_x()), farthest, places=6)
        self.assertAlmostEqual(read.d(shade.get_x() + shade.get_width()), 55.0, places=6)
        (note,) = artists(read.page, "untested-note")
        self.assertEqual((note.get_text(), note.get_rotation()), ("not tested", 90.0))  # upright: the stretch is narrow
        self.assertTrue(shade.get_x() < note.get_position()[0] < shade.get_x() + shade.get_width())
        self.assertTrue(read.y0 < note.get_position()[1] < read.y1)
        # The lower end of the interval lies in that stretch: it rests on the model, and the picture shows it.
        (bar,) = artists(read.page, "radius-interval:policy")
        self.assertGreater(bar.get_xdata()[0], shade.get_x())
        # No curve, no rate and no trial is drawn in it.
        for curve in artists(read.page, "curve:"):
            self.assertLessEqual(curve.get_xdata()[-1], shade.get_x() + 1e-9)
        for mark in [*artists(read.page, "rate:"), *artists(read.page, "trial:")]:
            self.assertLessEqual(mark.get_xdata()[0], shade.get_x() + 1e-9)

    def test_an_open_interval_is_drawn_the_same_when_the_file_gives_null_for_what_it_does_not_establish(self):
        made = fixtures_results.load("open_above")
        with_numbers = style.pdf_bytes(rc.figure(made))
        made["conditions"]["act_clustered"]["block_fit"] |= {"r50_hi_mm": None, "r50_median_mm": None}
        self.assertEqual(style.pdf_bytes(rc.figure(made)), with_numbers)

    def test_an_interval_open_below_runs_off_the_left_end_in_an_arrow(self):
        read = drawn("threshold_blocks_only")
        fit = rc.conditions_shown(fixtures_results.load("threshold_blocks_only"))[1].fit
        self.assertEqual((fit.open_below, fit.open_above), (True, False))
        (bar,) = artists(read.page, "radius-interval:replay")
        self.assertAlmostEqual(read.d(bar.get_xdata()[1]), fit.r50_hi_mm, places=6)
        (arrow,) = artists(read.page, "radius-open:replay")
        tip, *base = arrow.get_xy()[:3]
        self.assertAlmostEqual(tip[0], read.axis_left)
        self.assertTrue(all(x > tip[0] for x, _ in base))
        self.assertEqual(artists(read.page, "radius:replay"), [])
        self.assertEqual(artists(read.page, "radius-line:replay"), [])
        self.assertEqual(len(artists(read.page, "radius:policy")), 1)
        self.assertEqual(read.texts("row-name:"), ["policy", "replay"])  # the row is still named

    def test_an_interval_open_on_both_sides_runs_from_end_to_end(self):
        made = fixtures_results.load("base")
        made["conditions"]["replay"]["block_fit"] |= {"open_below": True, "open_above": True}
        read = Read(rc.figure(made))
        self.assertEqual(len(artists(read.page, "radius-open:replay")), 2)
        self.assertEqual(artists(read.page, "radius-end:replay"), [])
        self.assertEqual(artists(read.page, "radius:replay"), [])
        self.assertAlmostEqual(read.axis_end(), 40.0, places=6)  # no end of its interval asks for room on the axis

    def test_where_the_bars_leave_no_room_the_rows_go_unnamed_rather_than_written_over_a_bar(self):
        made = fixtures_results.load("base")
        made["conditions"]["act_clustered"]["block_fit"] |= {"r50_hi_mm": 58.0}
        page = rc.figure(made)
        self.assertEqual(artists(page, "rows-name"), [])
        # With room only between the bars, the words go there.
        read = drawn("open_above")
        (label,) = artists(read.page, "rows-name")
        (policy,) = artists(read.page, "radius-interval:policy")
        (replay,) = artists(read.page, "radius-interval:replay")
        self.assertGreater(label.get_position()[0], replay.get_xdata()[1] + 1.0)
        self.assertLess(label.get_position()[0] + style.text_width_mm("tolerance radius"), policy.get_xdata()[0] - 1.0)


class UntestedTests(unittest.TestCase):
    def test_an_end_of_a_closed_interval_beyond_the_trials_lies_in_a_shaded_stretch(self):
        made = fixtures_results.load("base")
        made["conditions"]["act_clustered"]["block_fit"] |= {"r50_hi_mm": 58.0}
        read = Read(rc.figure(made))
        self.assertAlmostEqual(read.axis_end(), 60.0, places=6)
        (shade,) = artists(read.page, "untested-stretch")
        (bar,) = artists(read.page, "radius-interval:policy")
        self.assertTrue(shade.get_x() < bar.get_xdata()[1] < shade.get_x() + shade.get_width())
        (note,) = artists(read.page, "untested-note")
        self.assertEqual((note.get_text(), note.get_rotation()), ("not tested", 0.0))  # written across: there is room

    def test_no_curve_is_drawn_through_a_stretch_that_was_not_tested(self):
        made = with_gap(12.0, 33.0)
        read = Read(rc.figure(made))
        shown = rc.conditions_shown(made)
        (stretch,) = rc.untested_stretches(shown, 40.0, 10.0)
        self.assertGreater(stretch[1] - stretch[0], 15.0)
        (shade,) = artists(read.page, "untested-stretch")
        self.assertAlmostEqual(read.d(shade.get_x()), stretch[0], places=6)
        self.assertAlmostEqual(read.d(shade.get_x() + shade.get_width()), stretch[1], places=6)
        self.assertEqual([note.get_text() for note in artists(read.page, "untested-note")], ["not tested"])
        pieces = artists(read.page, "curve:policy")
        self.assertEqual(len(pieces), 2)  # before the stretch and after it
        low, high = shade.get_x() + 1e-9, shade.get_x() + shade.get_width() - 1e-9
        for piece in [*pieces, *artists(read.page, "curve:replay")]:
            self.assertEqual([x for x in piece.get_xdata() if low < x < high], [], piece.get_gid())
        for mark in [*artists(read.page, "rate:"), *artists(read.page, "trial:")]:
            self.assertFalse(low < mark.get_xdata()[0] < high, mark.get_gid())
        # The policy's radius lies in the stretch: its mark is drawn on its bar, with no line up to a curve.
        self.assertEqual(len(artists(read.page, "radius:policy")), 1)
        self.assertEqual(artists(read.page, "radius-line:policy"), [])

    def test_a_short_gap_between_trials_is_not_called_untested(self):
        made = with_gap(26.0, 37.0)
        shown = rc.conditions_shown(made)
        self.assertEqual(rc.untested_stretches(shown, 40.0, 10.0), [])
        self.assertEqual(len(artists(rc.figure(made), "curve:policy")), 1)


class OneConditionTests(unittest.TestCase):
    def test_with_one_threshold_block_only_that_condition_is_drawn_and_the_page_says_why(self):
        made = fixtures_results.changed("base", **{"conditions.act_clustered.block_fit": None})
        page = rc.figure(made)
        self.assertEqual({mark.get_gid().split(":")[1] for mark in artists(page, "trial:")}, {"replay"})
        self.assertEqual([text.get_text() for text in artists(page, "row-name:")], ["replay"])
        self.assertEqual(page.notes, ["the policy has trials and no block fit in the results: none of them is drawn"])
        self.assertEqual(rc.figure(fixtures_results.load("base")).notes, [])

    def test_a_condition_that_was_never_run_is_left_out_and_the_page_says_that_one_is_missing(self):
        made = fixtures_results.changed("base", **{"conditions.replay": None})
        page = rc.figure(made)
        self.assertEqual([text.get_text() for text in artists(page, "row-name:")], ["policy"])
        self.assertEqual(page.notes, ["the replay has no trial in the results: the figure shows one condition only"])

    def test_the_pooled_picture_holds_the_workspace_trials_too(self):
        page = rc.figure(fixtures_results.load("base"), fits="pooled")
        self.assertEqual(len(artists(page, "trial:clustered:")), 54)
        self.assertEqual(len(artists(page, "trial:replay:")), 22)

    def test_results_of_another_version_are_refused(self):
        with self.assertRaisesRegex(ResultsError, "schema_version"):
            rc.figure(fixtures_results.changed("base", schema_version=2))


if __name__ == "__main__":
    unittest.main()
