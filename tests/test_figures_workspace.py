"""Figure 4: the workspace test and the lookup prediction, from synthetic analysis results."""

import math
import unittest

from matplotlib.colors import to_hex

import fixtures_results
from fixtures_demos import GRID
from tolerance_radius.figures import HEIGHTS_MM, gridmap, results, style
from tolerance_radius.figures import workspace as ws
from tolerance_radius.figures.results import ResultsError
from tolerance_radius.figures.style import FigureError
from tolerance_radius.schema import Point


def artists(page, prefix: str) -> list:
    """Everything drawn whose identifier starts with `prefix`."""
    drawn = [*page.axes.lines, *page.axes.patches, *page.axes.texts, *page.axes.collections]
    return [artist for artist in drawn if (artist.get_gid() or "").startswith(prefix)]


def texts(page, prefix: str) -> list[str]:
    """The words of every text whose identifier starts with `prefix`, in the order written."""
    return [text.get_text() for text in artists(page, prefix) if hasattr(text, "get_text")]


def panels(page) -> list[gridmap.Panel]:
    """The two pictures of the grid, found again from the outlines drawn."""
    return [gridmap.Panel(outline.get_x(), outline.get_y(), outline.get_width() / GRID.width_mm, GRID)
            for outline in artists(page, "grid-outline")]  # fmt: skip


def on_table(panel: gridmap.Panel, mark) -> tuple[float, float]:
    """Where a mark drawn in a picture is on the table."""
    return ((mark.get_xdata()[0] - panel.x0_mm) / panel.scale, (mark.get_ydata()[0] - panel.y0_mm) / panel.scale)


def in_panel(panel: gridmap.Panel, marks: list) -> list:
    """The marks that lie in a picture and not in the other one or in the lines of text below."""
    return [mark for mark in marks
            if panel.x0_mm - 1e-6 <= mark.get_xdata()[0] <= panel.right_mm + 1e-6
            and panel.y0_mm - 1e-6 <= mark.get_ydata()[0] <= panel.top_mm + 1e-6]  # fmt: skip


def nearest_mm(position: tuple[float, float], demos) -> float:
    return min(math.hypot(position[0] - demo.x_mm, position[1] - demo.y_mm) for demo in demos)


def with_fit(name: str = "base", **changes) -> dict:
    """A case's results with fields of the clustered policy's block fit replaced."""
    made = fixtures_results.load(name)
    made["conditions"]["act_clustered"]["block_fit"] |= changes
    return made


class DiscsTests(unittest.TestCase):
    def test_a_bounded_fit_gives_discs_of_its_radius(self):
        fit = results.fit(fixtures_results.load("base"), "act_clustered")
        self.assertEqual(ws.discs_of(fit), ws.Discs(fit.r50_median_mm, ws.RADIUS))

    def test_an_interval_open_above_gives_discs_of_its_lower_end_the_smallest_radius_the_trials_allow(self):
        fit = results.fit(fixtures_results.load("open_above"), "act_clustered")
        self.assertEqual(ws.discs_of(fit), ws.Discs(fit.r50_lo_mm, ws.SMALLEST))
        self.assertLess(fit.r50_lo_mm, fit.r50_median_mm)

    def test_an_interval_open_below_gives_discs_of_its_upper_end_the_largest_radius_the_trials_allow(self):
        fit = results.fit(with_fit(open_below=True), "act_clustered")
        self.assertEqual(ws.discs_of(fit), ws.Discs(fit.r50_hi_mm, ws.LARGEST))

    def test_no_fit_or_an_interval_open_on_both_sides_gives_no_disc(self):
        self.assertIsNone(ws.discs_of(None))
        self.assertIsNone(ws.discs_of(results.fit(with_fit(open_below=True, open_above=True), "act_clustered")))


class CountRowsTests(unittest.TestCase):
    def test_placements_in_one_region_have_one_line_for_all(self):
        made = fixtures_results.load("base")
        rows, notes = ws.count_rows(made, "act_spread")
        self.assertEqual(notes, [])
        self.assertEqual(rows, [("all", results.success_count(made, "act_spread"), results.lookup(made, "act_spread"))])

    def test_placements_in_two_regions_have_a_line_for_each_after_the_line_for_all(self):
        made = fixtures_results.load("held_out_placements")
        rows, notes = ws.count_rows(made, "act_clustered")
        self.assertEqual(notes, [])
        self.assertEqual([name for name, _, _ in rows], ["all", "trained_cells", "held_out_cells"])
        self.assertEqual([count.n for _, count, _ in rows], [26, 13, 13])
        self.assertIsNotNone(rows[0][2])
        self.assertEqual([lookup for _, _, lookup in rows[1:]], [None, None])  # the interval belongs to the total
        self.assertEqual(rows[0][1].k, rows[1][1].k + rows[2][1].k)

    def test_pairs_that_all_name_one_region_add_no_line(self):
        made = fixtures_results.load("base")
        for pair in made["workspace"]["pairs"]:
            pair["region"] = "trained_cells"
        self.assertEqual([name for name, _, _ in ws.count_rows(made, "act_clustered")[0]], ["all"])

    def test_a_file_cut_down_to_the_contract_has_the_line_for_all_only(self):
        rows, notes = ws.count_rows(fixtures_results.contract_only("held_out_placements"), "act_clustered")
        self.assertEqual(([name for name, _, _ in rows], notes), (["all"], []))


class FigureTests(unittest.TestCase):
    """The case that the experiment is likely to be: 13 placements in trained cells and 13 in held-out cells."""

    @classmethod
    def setUpClass(cls):
        cls.made = fixtures_results.load("held_out_placements")
        cls.page = ws.figure(cls.made, GRID)
        cls.left, cls.right = panels(cls.page)
        cls.fit = results.fit(cls.made, "act_clustered")

    def test_it_is_a_column_wide_and_40_mm_high(self):
        self.assertEqual((self.page.width_mm, self.page.height_mm, HEIGHTS_MM[4]), (88.0, 40.0, 40.0))

    def test_the_grid_is_drawn_twice_at_one_scale_the_clustered_policy_on_the_left(self):
        self.assertAlmostEqual(self.left.scale, self.right.scale)
        self.assertLess(self.left.right_mm, self.right.x0_mm)
        self.assertAlmostEqual(self.left.y0_mm, self.right.y0_mm)
        self.assertEqual(texts(self.page, "panel-title"), ["Clustered policy", "Spread policy"])
        self.assertEqual(len(artists(self.page, "held-out-hatching")), 2)
        self.assertGreater(self.left.right_mm - self.left.x0_mm, 34.0)  # still large with four lines of counts below
        self.assertLessEqual(self.right.right_mm, 88.0 - gridmap.SIDE_MM + 1e-9)
        self.assertGreaterEqual(self.left.x0_mm, gridmap.SIDE_MM - 1e-9)

    def test_no_cell_is_named(self):
        self.assertEqual(artists(self.page, "column-name") + artists(self.page, "row-name"), [])

    def test_each_picture_shows_its_own_policys_demonstrations_where_they_were_measured(self):
        for panel, dataset, series in ((self.left, "clustered", style.CLUSTERED), (self.right, "spread", style.SPREAD)):
            dots = artists(self.page, f"demonstration:{dataset}")
            self.assertEqual(len(dots), 55)
            measured = sorted((demo.x_mm, demo.y_mm) for demo in results.demonstrations(self.made, dataset))
            for (x, y), (mx, my) in zip(sorted(on_table(panel, dot) for dot in dots), measured, strict=True):
                self.assertAlmostEqual(x, mx, places=6)
                self.assertAlmostEqual(y, my, places=6)
            for dot in dots:
                self.assertEqual(dot.get_marker(), series.marker)
                self.assertEqual(to_hex(dot.get_markerfacecolor()), series.color)
                self.assertLess(dot.get_markersize(), 3.0)  # smaller than a placement: the background of the test

    def test_every_placement_is_one_mark_in_each_picture_filled_for_a_success_and_open_for_a_failure(self):
        for panel, policy in ((self.left, "act_clustered"), (self.right, "act_spread")):
            placed = results.placements(self.made, policy)
            successes = in_panel(panel, artists(self.page, "placement:success"))
            failures = in_panel(panel, artists(self.page, "placement:failure"))
            self.assertEqual(len(successes) + len(failures), 26)
            self.assertEqual(len(successes), sum(placement.success for placement in placed))
            drawn = {(round(x, 6), round(y, 6)): True for x, y in (on_table(panel, mark) for mark in successes)}
            drawn |= {(round(x, 6), round(y, 6)): False for x, y in (on_table(panel, mark) for mark in failures)}
            self.assertEqual(drawn, {(round(p.x_mm, 6), round(p.y_mm, 6)): p.success for p in placed})
            for mark in successes:
                self.assertEqual((to_hex(mark.get_markerfacecolor()), to_hex(mark.get_markeredgecolor())),
                                 (style.INK, "#ffffff"))  # fmt: skip
            for mark in failures:
                self.assertEqual((to_hex(mark.get_markerfacecolor()), to_hex(mark.get_markeredgecolor())),
                                 ("#ffffff", style.INK))  # fmt: skip

    def test_a_placement_in_a_hatched_cell_stands_clear_of_the_hatching(self):
        (hatching, _) = artists(self.page, "held-out-hatching")
        in_held_out = 0
        for mark in in_panel(self.left, artists(self.page, "placement:")):
            cell = GRID.cell_of(Point(*on_table(self.left, mark)))
            if not GRID.is_trained(cell):
                in_held_out += 1
                self.assertGreater(mark.get_zorder(), hatching.get_zorder())  # drawn over the lines
                # White around a filled mark and inside an open one: no hatch line touches the ink or shows through.
                self.assertIn("#ffffff", (to_hex(mark.get_markerfacecolor()), to_hex(mark.get_markeredgecolor())))
                self.assertGreaterEqual(mark.get_markeredgewidth(), 0.5)
        self.assertEqual(in_held_out, 13)
        self.assertGreaterEqual(min(mark.get_markersize() for mark in artists(self.page, "placement:")), 3.0)

    def test_a_disc_of_the_clustered_policys_radius_lies_around_every_demonstration_in_both_pictures(self):
        divisor = results.distance_divisor(self.made)
        self.assertEqual(divisor, 1.031)
        for dataset, series in (("clustered", style.CLUSTERED), ("spread", style.SPREAD)):
            discs = artists(self.page, f"disc:{dataset}")
            self.assertEqual(len(discs), 55)
            for disc in discs:
                # Positions are as read and the radius is in corrected millimetres: on the table it is longer.
                self.assertAlmostEqual(disc.get_radius(), self.fit.r50_median_mm * divisor)
                self.assertEqual(to_hex(disc.get_facecolor()), series.tint)
            centres = sorted(disc.center for disc in discs)
            measured = sorted((demo.x_mm, demo.y_mm) for demo in results.demonstrations(self.made, dataset))
            self.assertEqual(centres, measured)
            (outline,) = artists(self.page, f"disc-outline:{dataset}")
            self.assertTrue(outline.get_gid().endswith(":radius"))
            self.assertEqual(outline.get_linestyles()[0][1], None)  # a solid line: the trials bound this radius

    def test_a_placement_is_drawn_inside_a_disc_exactly_when_its_distance_is_within_the_radius(self):
        divisor = results.distance_divisor(self.made)
        inside_and_failed = 0
        for panel, policy, dataset in ((self.left, "act_clustered", "clustered"), (self.right, "act_spread", "spread")):
            demos = results.demonstrations(self.made, dataset)
            by_place = {(round(p.x_mm, 6), round(p.y_mm, 6)): p for p in results.placements(self.made, policy)}
            for mark in in_panel(panel, artists(self.page, "placement:")):
                x, y = on_table(panel, mark)
                placement = by_place[(round(x, 6), round(y, 6))]
                drawn_mm = nearest_mm((x, y), demos)
                self.assertAlmostEqual(drawn_mm / divisor, placement.d_mm, delta=0.05)
                inside = drawn_mm <= self.fit.r50_median_mm * divisor
                self.assertEqual(inside, placement.d_mm <= self.fit.r50_median_mm)
                inside_and_failed += inside and not placement.success
        self.assertGreater(inside_and_failed, 0)  # the picture can show where the prediction does not hold
        self.assertEqual(self.page.notes, [])

    def test_the_counts_stand_below_each_picture_by_region(self):
        self.assertEqual(texts(self.page, "counts-heading"), ["cells", "observed", "predicted"] * 2)
        self.assertEqual(texts(self.page, "counts-name:"), ["all", "trained", "held-out"] * 2)
        for index, policy in enumerate(("act_clustered", "act_spread")):
            rows, _ = ws.count_rows(self.made, policy)
            lookup = results.lookup(self.made, policy)
            self.assertEqual(texts(self.page, "counts-success:")[3 * index : 3 * index + 3],
                             [str(count.k) for _, count, _ in rows])  # fmt: skip
            self.assertEqual(texts(self.page, "counts-failure:")[3 * index : 3 * index + 3],
                             [str(count.n - count.k) for _, count, _ in rows])  # fmt: skip
            self.assertEqual(texts(self.page, "counts-predicted:")[3 * index : 3 * index + 3],
                             [f"{lookup.expected:.1f} ({lookup.lo} to {lookup.hi})",
                              f"{rows[1][1].lookup_expected:.1f}", f"{rows[2][1].lookup_expected:.1f}"])  # fmt: skip
        for text in artists(self.page, "counts-"):
            if hasattr(text, "get_text"):
                of_left = text.get_position()[0] < self.right.x0_mm - 1.0
                self.assertLess(text.get_position()[1], self.left.y0_mm)  # below the pictures
                # The left table ends before the right one begins, and the right one before the page does.
                end = text.get_position()[0] + style.text_width_mm(text.get_text())
                self.assertLessEqual(end, self.right.x0_mm - 1.0 if of_left else 88.0 - gridmap.SIDE_MM)

    def test_the_number_of_filled_marks_in_a_picture_is_the_number_written_below_it(self):
        for index, panel in enumerate((self.left, self.right)):
            filled = len(in_panel(panel, artists(self.page, "placement:success")))
            opened = len(in_panel(panel, artists(self.page, "placement:failure")))
            self.assertEqual(texts(self.page, "counts-success:all")[index], str(filled))
            self.assertEqual(texts(self.page, "counts-failure:all")[index], str(opened))

    def test_the_last_line_says_what_the_marks_and_the_discs_are(self):
        self.assertEqual(texts(self.page, "key-text"), ["success", "failure", "within the radius of a demonstration"])
        success, failure = artists(self.page, "key-mark")
        self.assertEqual((to_hex(success.get_markerfacecolor()), to_hex(failure.get_markerfacecolor())),
                         (style.INK, "#ffffff"))  # fmt: skip
        swatches = artists(self.page, "key-disc:radius")
        self.assertEqual([to_hex(swatch.get_facecolor()) for swatch in swatches], [style.BLUE_TINT, style.ORANGE_TINT])
        self.assertEqual([to_hex(swatch.get_edgecolor()) for swatch in swatches], [style.BLUE, style.ORANGE])
        self.assertEqual({swatch.get_linestyle() for swatch in swatches}, {"solid"})
        key_y = {round(text.get_position()[1], 6) for text in artists(self.page, "key-text")}
        self.assertEqual(len(key_y), 1)  # one line
        counts = [text for text in artists(self.page, "counts-") if hasattr(text, "get_text")]
        lowest_count = min(text.get_position()[1] for text in counts)
        self.assertLess(key_y.pop() + 2.4, lowest_count)  # below the last line of counts, clear of it
        last = artists(self.page, "key-text")[-1]
        self.assertLess(last.get_position()[0] + style.text_width_mm(last.get_text()), 88.0)

    def test_a_draft_says_draft_and_results_that_count_do_not(self):
        self.assertEqual(self.page.stamps, ["DRAFT"])
        counted = ws.figure(fixtures_results.changed("held_out_placements", draft=False), GRID)
        self.assertEqual((counted.stamps, artists(counted, "stamp:")), ([], []))

    def test_no_text_is_small_and_none_runs_off_the_page(self):
        cases = [fixtures_results.load(name) for name in ("base", "open_above", "held_out_placements")]
        cases.append(with_fit(open_below=True, open_above=True))
        for made in cases:
            page = ws.figure(made, GRID)
            for text in page.texts():
                self.assertGreaterEqual(text.get_fontsize(), 7.0)
                if text.get_gid().startswith("stamp:"):
                    continue
                bold = text.get_gid() == "panel-title"
                width = style.text_width_mm(text.get_text(), text.get_fontsize(), bold=bold)
                self.assertGreaterEqual(text.get_position()[0], 0.0, text.get_text())
                self.assertLessEqual(text.get_position()[0] + width, 88.0, text.get_text())
                self.assertTrue(0.2 <= text.get_position()[1] <= 40.0 - 2.0, text.get_text())
            for line in page.axes.lines:
                for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
                    self.assertTrue(0.0 <= x <= 88.0 and 0.0 <= y <= 40.0, line.get_gid())

    def test_the_same_results_give_the_same_file(self):
        self.assertEqual(style.pdf_bytes(ws.figure(fixtures_results.load("held_out_placements"), GRID)),
                         style.pdf_bytes(self.page))  # fmt: skip
        self.assertEqual(self.page.axes.get_title(), "")
        self.assertEqual(self.page.figure.texts, [])


class TwentyPlacementsTests(unittest.TestCase):
    """A workspace test with every placement in a trained cell and no counts by region, as the protocol had it
    until draft 0.2: 20 placements. A results file of that kind must still be drawn."""

    @classmethod
    def setUpClass(cls):
        cls.made = fixtures_results.load("base")
        cls.page = ws.figure(cls.made, GRID)
        cls.left, cls.right = panels(cls.page)

    def test_twenty_placements_are_drawn_in_each_picture_and_none_in_a_held_out_cell(self):
        for panel in (self.left, self.right):
            marks = in_panel(panel, artists(self.page, "placement:"))
            self.assertEqual(len(marks), 20)
            for mark in marks:
                self.assertTrue(GRID.is_trained(GRID.cell_of(Point(*on_table(panel, mark)))))

    def test_one_line_of_counts_is_enough_and_the_pictures_are_as_wide_as_the_page_allows(self):
        self.assertEqual(texts(self.page, "counts-name:"), ["all", "all"])
        self.assertAlmostEqual(self.left.scale, (88.0 - 2 * gridmap.SIDE_MM - 1.5) / 500.0)
        lookup = results.lookup(self.made, "act_clustered")
        predicted = f"{lookup.expected:.1f} ({lookup.lo} to {lookup.hi})"
        self.assertEqual(texts(self.page, "counts-predicted:")[0], predicted)


class RoomTests(unittest.TestCase):
    def test_the_discs_are_cut_off_at_the_grids_edge(self):
        # The smallest radius the trials allow is larger than a cell here: without the cut, the tint would run over
        # the other picture and the counts.
        page = ws.figure(fixtures_results.load("open_above"), GRID)
        left, right = panels(page)
        to_page = page.axes.transData.inverted()
        for panel, dataset in ((left, "clustered"), (right, "spread")):
            for disc in artists(page, f"disc:{dataset}"):
                self.assertGreater(panel.length(disc.get_radius()), panel.length(GRID.cell_mm))
                (x0, y0), (x1, y1) = to_page.transform(disc.get_clip_box().get_points())
                for got, wanted in ((x0, panel.x0_mm), (y0, panel.y0_mm), (x1, panel.right_mm), (y1, panel.top_mm)):
                    self.assertAlmostEqual(got, wanted, places=6)

    def test_with_two_regions_the_pictures_keep_a_useful_size(self):
        page = ws.figure(fixtures_results.load("held_out_placements"), GRID)
        left, _ = panels(page)
        self.assertGreaterEqual(left.right_mm - left.x0_mm, ws.LEAST_PANEL_MM)
        self.assertEqual(page.notes, [])

    def test_counts_by_region_that_would_leave_the_pictures_too_small_give_way_to_the_totals_and_the_page_says_so(self):
        made = fixtures_results.load("held_out_placements")
        del made["workspace"]["by_region"]
        for index, pair in enumerate(made["workspace"]["pairs"]):  # five regions: six lines of counts
            pair["region"] = ("trained_cells", "held_out_cells", "near the edge", "far corner", "middle")[index % 5]
        self.assertEqual(len(ws.count_rows(made, "act_clustered")[0]), 6)
        page = ws.figure(made, GRID)
        self.assertEqual(texts(page, "counts-name:"), ["all", "all"])
        self.assertEqual(page.notes, ["the counts by region do not fit below the pictures at this size: only the "
                                      "totals are shown"])  # fmt: skip
        left, _ = panels(page)
        self.assertAlmostEqual(left.scale, panels(ws.figure(fixtures_results.load("base"), GRID))[0].scale)
        self.assertEqual(len(artists(page, "placement:")), 52)  # every placement is still drawn

    def test_a_third_region_is_already_more_than_the_figure_has_room_for(self):
        made = fixtures_results.load("held_out_placements")
        del made["workspace"]["by_region"]
        for pair in made["workspace"]["pairs"][:4]:
            pair["region"] = "margin"
        self.assertEqual([name for name, _, _ in ws.count_rows(made, "act_spread")[0]],
                         ["all", "trained_cells", "held_out_cells", "margin"])  # fmt: skip
        page = ws.figure(made, GRID)
        self.assertEqual(texts(page, "counts-name:"), ["all", "all"])
        self.assertEqual(len(page.notes), 1)


class ScaleTests(unittest.TestCase):
    def test_with_distances_as_read_a_disc_has_the_radius_itself(self):
        made = fixtures_results.load("no_spread_threshold")
        fit = results.fit(made, "act_clustered")
        page = ws.figure(made, GRID)
        self.assertEqual({disc.get_radius() for disc in artists(page, "disc:")}, {fit.r50_median_mm})
        self.assertEqual(page.notes, [])

    def test_with_corrected_distances_a_disc_is_larger_on_the_table_by_the_factor(self):
        made = fixtures_results.load("base")
        fit = results.fit(made, "act_clustered")
        self.assertEqual({round(disc.get_radius(), 9) for disc in artists(ws.figure(made, GRID), "disc:")},
                         {round(fit.r50_median_mm * 1.031, 9)})  # fmt: skip

    def test_the_factor_is_what_keeps_a_placement_just_inside_the_radius_inside_its_disc(self):
        # Take a placement some way out and put the radius 0.2 mm beyond its corrected distance. On the table, where
        # positions are as read, that placement is farther from its demonstration than the radius is long.
        made = fixtures_results.load("base")
        demos = results.demonstrations(made, "clustered")
        chosen = next(p for p in results.placements(made, "act_clustered") if 12.0 < p.d_mm < 30.0)
        radius = chosen.d_mm + 0.2
        made["conditions"]["act_clustered"]["block_fit"] |= {"r50_lo_mm": radius - 3.0, "r50_median_mm": radius,
                                                             "r50_hi_mm": radius + 3.0}  # fmt: skip
        read_mm = nearest_mm((chosen.x_mm, chosen.y_mm), demos)
        self.assertGreater(read_mm, radius)
        page = ws.figure(made, GRID)
        (disc, *_) = artists(page, "disc:clustered")
        self.assertLess(read_mm, disc.get_radius())  # inside the disc as drawn
        self.assertEqual(page.notes, [])  # and so is every other placement on the side its distance says
        # Had the disc the radius itself, the picture and the file would disagree, and the figure would say so.
        made["distance_scale"] |= {"applied": False, "divisor": 1.0}
        notes = ws.figure(made, GRID).notes
        self.assertTrue(any("placement" in note for note in notes))


class OpenIntervalTests(unittest.TestCase):
    def test_an_interval_open_above_draws_the_smallest_radius_dashed_and_says_so(self):
        made = fixtures_results.load("open_above")
        fit = results.fit(made, "act_clustered")
        page = ws.figure(made, GRID)
        self.assertEqual({round(disc.get_radius(), 9) for disc in artists(page, "disc:")},
                         {round(fit.r50_lo_mm * results.distance_divisor(made), 9)})  # fmt: skip
        outlines = artists(page, "disc-outline:")
        self.assertTrue(outlines)
        for outline in outlines:
            self.assertTrue(outline.get_gid().endswith(":smallest"))
            self.assertIsNotNone(outline.get_linestyles()[0][1])  # dashed
        self.assertEqual(texts(page, "key-text")[-1], "within the smallest radius the trials allow")
        swatches = artists(page, "key-disc:smallest")
        self.assertEqual(len(swatches), 2)
        self.assertNotIn("solid", {swatch.get_linestyle() for swatch in swatches})
        # The counts and the prediction are the file's, whatever the discs show.
        lookup = results.lookup(made, "act_spread")
        self.assertIn(f"{lookup.expected:.1f} ({lookup.lo} to {lookup.hi})", texts(page, "counts-predicted:"))
        self.assertEqual(page.notes, [])

    def test_an_open_interval_is_drawn_the_same_when_the_file_gives_null_for_what_it_does_not_establish(self):
        made = fixtures_results.load("open_above")
        with_numbers = style.pdf_bytes(ws.figure(made, GRID))
        made["conditions"]["act_clustered"]["block_fit"] |= {"r50_hi_mm": None, "r50_median_mm": None}
        self.assertEqual(style.pdf_bytes(ws.figure(made, GRID)), with_numbers)

    def test_an_interval_open_below_draws_the_largest_radius_dashed_and_says_so(self):
        made = with_fit(open_below=True)
        fit = results.fit(made, "act_clustered")
        page = ws.figure(made, GRID)
        self.assertEqual({round(disc.get_radius(), 9) for disc in artists(page, "disc:")},
                         {round(fit.r50_hi_mm * 1.031, 9)})  # fmt: skip
        self.assertTrue(all(outline.get_gid().endswith(":largest") for outline in artists(page, "disc-outline:")))
        self.assertEqual(texts(page, "key-text")[-1], "within the largest radius the trials allow")
        self.assertEqual(len(artists(page, "key-disc:largest")), 2)

    def test_an_interval_open_on_both_sides_draws_no_disc_and_says_why(self):
        page = ws.figure(with_fit(open_below=True, open_above=True), GRID)
        self.assertEqual(artists(page, "disc"), [])
        self.assertEqual(artists(page, "key-disc"), [])
        self.assertEqual(texts(page, "key-text"), ["success", "failure", "no disc: the trials do not bound the radius"])
        self.assertEqual(len(artists(page, "placement:")), 40)
        self.assertEqual(len(artists(page, "demonstration:")), 110)

    def test_without_the_clustered_policys_block_fit_there_is_no_disc_and_no_prediction(self):
        nothing_fitted = {"conditions.act_clustered.block_fit": None, "workspace.lookup": None}
        made = fixtures_results.changed("base", **nothing_fitted)
        for pair in made["workspace"]["pairs"]:
            for policy in results.POLICIES:
                pair[policy]["p_lookup"] = None
        page = ws.figure(made, GRID)
        self.assertEqual(artists(page, "disc"), [])
        self.assertEqual(texts(page, "key-text"), ["success", "failure"])
        self.assertEqual(texts(page, "counts-heading"), ["cells", "observed"] * 2)
        self.assertEqual(artists(page, "counts-predicted:"), [])
        self.assertEqual(len(artists(page, "counts-success:")), 2)
        self.assertEqual(page.notes, [])


class RegionsTests(unittest.TestCase):
    def test_a_region_shows_the_interval_of_its_predicted_count_when_the_file_gives_one(self):
        made = fixtures_results.load("held_out_placements")
        plain = texts(ws.figure(made, GRID), "counts-predicted:trained_cells")
        self.assertNotIn("(", "".join(plain))
        for region, ends in (("trained_cells", (5, 11)), ("held_out_cells", (0, 4))):
            for policy in results.POLICIES:
                made["workspace"]["by_region"][region][policy] |= {"lookup_lo": ends[0], "lookup_hi": ends[1]}
        page = ws.figure(made, GRID)
        expected = made["workspace"]["by_region"]["trained_cells"]["act_clustered"]["lookup_expected"]
        self.assertEqual(texts(page, "counts-predicted:trained_cells")[0], f"{expected:.1f} (5 to 11)")
        self.assertTrue(all(text.endswith("(0 to 4)") for text in texts(page, "counts-predicted:held_out_cells")))
        self.assertEqual(page.notes, [])
        self.assertEqual(texts(page, "counts-name:"), ["all", "trained", "held-out"] * 2)  # and they still fit

    def test_counts_by_region_are_made_from_the_pairs_when_the_file_has_none(self):
        made = fixtures_results.load("held_out_placements")
        stated = texts(ws.figure(made, GRID), "counts-")
        del made["workspace"]["by_region"]
        page = ws.figure(made, GRID)
        self.assertEqual(texts(page, "counts-"), stated)
        self.assertEqual(page.notes, [])

    def test_counts_by_region_in_an_unknown_shape_are_passed_over_and_the_page_says_so(self):
        made = fixtures_results.load("held_out_placements")
        stated = texts(ws.figure(made, GRID), "counts-")
        made["workspace"]["by_region"] = {"act_clustered": {}, "act_spread": {}}
        page = ws.figure(made, GRID)
        self.assertEqual(texts(page, "counts-"), stated)
        self.assertEqual(len(page.notes), 2)  # one for each policy, each with what could not be read
        for note, policy in zip(page.notes, results.POLICIES, strict=True):
            self.assertIn("workspace.by_region is not in the shape this code reads", note)
            self.assertIn(policy, note)
            self.assertIn("made from the pairs", note)

    def test_cut_down_to_the_contract_the_totals_are_kept_and_both_pictures_show_the_pairs_positions(self):
        made = fixtures_results.contract_only("held_out_placements")
        page = ws.figure(made, GRID)
        left, right = panels(page)
        self.assertEqual(texts(page, "counts-name:"), ["all", "all"])
        self.assertGreater(left.scale, panels(ws.figure(fixtures_results.load("held_out_placements"), GRID))[0].scale)
        places = [sorted((round(x, 6), round(y, 6)) for x, y in (on_table(panel, mark) for mark in
                                                                  in_panel(panel, artists(page, "placement:"))))
                  for panel in (left, right)]  # fmt: skip
        self.assertEqual(places[0], places[1])
        self.assertEqual(places[0], sorted((round(pair["x_mm"], 6), round(pair["y_mm"], 6))
                                           for pair in made["workspace"]["pairs"]))  # fmt: skip
        self.assertEqual(len(places[0]), 26)

    def test_a_placement_drawn_at_its_pairs_position_on_the_wrong_side_of_a_discs_edge_is_reported(self):
        page = ws.figure(fixtures_results.contract_only("held_out_placements"), GRID)
        self.assertEqual(len(page.notes), 1)
        self.assertRegex(page.notes[0], r"^figure 4, act_spread, placement 4: drawn inside a disc, while its distance "
                                        r"in the file \(21\.2 mm\) is on the other side of the discs' radius "
                                        r"\(21\.0 mm\)$")  # fmt: skip


class NotesAndRefusalsTests(unittest.TestCase):
    def test_a_distance_that_is_not_the_distance_between_the_positions_drawn_is_reported(self):
        made = fixtures_results.load("base")
        made["workspace"]["pairs"][2]["act_clustered"]["d_mm"] += 30.0
        page = ws.figure(made, GRID)
        self.assertEqual(len(page.notes), 1)
        self.assertRegex(page.notes[0], r"^figure 4, act_clustered, placement 3: the file gives d = \d+\.\d mm, the "
                                        r"positions drawn are \d+\.\d mm apart$")  # fmt: skip

    def test_counts_that_are_not_the_pairs_counts_stop_the_figure(self):
        made = fixtures_results.load("base")
        made["workspace"]["success"]["act_clustered"]["k"] -= 1
        with self.assertRaisesRegex(ResultsError, r"workspace\.success\.act_clustered"):
            ws.figure(made, GRID)

    def test_results_without_a_workspace_test_or_of_another_version_are_refused(self):
        with self.assertRaisesRegex(FigureError, "no workspace test"):
            ws.figure(fixtures_results.load("threshold_blocks_only"), GRID)
        with self.assertRaisesRegex(ResultsError, "schema_version"):
            ws.figure(fixtures_results.changed("base", schema_version=3), GRID)

    def test_a_set_without_demonstrations_or_a_position_off_the_grid_is_refused(self):
        with self.assertRaisesRegex(FigureError, "no demonstration the spread policy was trained on"):
            ws.figure(fixtures_results.changed("base", **{"demonstrations.spread": []}), GRID)
        made = fixtures_results.load("base")
        made["workspace"]["pairs"][0]["act_spread"]["x_mm"] = 262.0
        with self.assertRaisesRegex(FigureError, "off the grid"):
            ws.figure(made, GRID)

    def test_demonstrations_that_were_not_trained_on_carry_no_disc_and_no_dot(self):
        made = fixtures_results.load("base")
        for demo in made["demonstrations"]["spread"][:5]:
            demo["trained"] = False
        page = ws.figure(made, GRID)
        self.assertEqual(len(artists(page, "demonstration:spread")), 50)
        self.assertEqual(len(artists(page, "disc:spread")), 50)
        self.assertEqual(len(artists(page, "disc:clustered")), 55)


if __name__ == "__main__":
    unittest.main()
