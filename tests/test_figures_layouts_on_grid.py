"""Figure 2: the two layouts on the grid, from start-state audits (synthetic here) and from the plan."""

import dataclasses
import unittest

from matplotlib.colors import to_hex

import pdf_facts
from fixtures_demos import GRID, PROTOCOL, audit
from tolerance_radius import layouts
from tolerance_radius.figures import HEIGHTS_MM, gridmap, style
from tolerance_radius.figures import layouts_on_grid as lg
from tolerance_radius.figures.style import FigureError
from tolerance_radius.schema import DemoStart, Point

#: Audits as a camera might measure them: every start up to 1.5 mm off its planned position in x and in y.
AUDITS = {name: list(audit(name)) for name in ("clustered", "spread")}
PLANNED = {name: layouts.planned_demo_starts(layouts.layout(name, PROTOCOL)) for name in ("clustered", "spread")}


def by_gid(page, wanted: str) -> list:
    """The artists with an identifier; one that ends in a colon stands for every identifier that starts with it."""
    artists = [*page.axes.lines, *page.axes.patches, *page.axes.texts, *page.axes.collections]
    if wanted.endswith(":"):
        return [artist for artist in artists if (artist.get_gid() or "").startswith(wanted)]
    return [artist for artist in artists if artist.get_gid() == wanted]


def panels(page) -> list[gridmap.Panel]:
    """The two pictures of the grid, found again from the outlines drawn."""
    found = []
    for outline in by_gid(page, "grid-outline"):
        found.append(gridmap.Panel(outline.get_x(), outline.get_y(), outline.get_width() / GRID.width_mm, GRID))
    return found


class PointsTests(unittest.TestCase):
    def test_the_clustered_set_starts_at_eleven_points_of_five(self):
        points = lg.points_of(AUDITS["clustered"])
        self.assertEqual(sorted(len(point) for point in points), [5] * 11)
        for point in points:
            self.assertEqual(len({demo.cell for demo in point}), 1)
        self.assertEqual(lg.summary(AUDITS["clustered"]), "55 starts at 11 points")

    def test_every_start_of_the_spread_set_is_a_point_of_its_own(self):
        self.assertEqual([len(point) for point in lg.points_of(AUDITS["spread"])], [1] * 55)
        self.assertEqual(lg.summary(AUDITS["spread"]), "55 starts at 55 points")

    def test_the_plan_gives_the_same_points_as_the_audit(self):
        for name in ("clustered", "spread"):
            self.assertEqual(lg.summary(PLANNED[name]), lg.summary(AUDITS[name]))

    def test_starts_are_one_point_when_a_chain_of_near_neighbours_joins_them(self):
        def at(index: int, x: float) -> DemoStart:
            return DemoStart("clustered", index, "C2", measured=Point(x, 75.0))

        chain = [at(0, 100.0), at(1, 104.0), at(2, 108.0), at(3, 120.0)]
        self.assertEqual([[demo.episode_index for demo in point] for point in lg.points_of(chain)], [[0, 1, 2], [3]])
        self.assertEqual(lg.points_of([]), [])
        self.assertEqual(lg.summary(chain[:1]), "1 start at 1 point")

    def test_an_episode_left_out_of_training_is_not_counted(self):
        starts = list(AUDITS["clustered"])
        starts[7] = dataclasses.replace(starts[7], trained=False)
        self.assertEqual(lg.summary(starts), "54 starts at 11 points")


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = lg.figure(AUDITS, GRID)
        cls.left, cls.right = panels(cls.page)

    def test_it_is_a_column_wide_and_32_mm_high(self):
        self.assertEqual((self.page.width_mm, self.page.height_mm, HEIGHTS_MM[2]), (88.0, 32.0, 32.0))

    def test_the_grid_is_drawn_twice_at_one_scale_clustered_on_the_left(self):
        self.assertAlmostEqual(self.left.scale, self.right.scale)
        self.assertLess(self.left.right_mm, self.right.x0_mm)
        titles = by_gid(self.page, "panel-title")
        self.assertEqual([title.get_text() for title in titles], ["Clustered", "Spread"])
        self.assertAlmostEqual(titles[0].get_position()[0], self.left.x0_mm)
        self.assertAlmostEqual(titles[1].get_position()[0], self.right.x0_mm)
        self.assertEqual([note.get_text() for note in by_gid(self.page, "panel-note")],
                         ["55 starts at 11 points", "55 starts at 55 points"])  # fmt: skip

    def test_every_demonstration_is_a_dot_at_its_measured_start(self):
        for panel, name, series in ((self.left, "clustered", style.CLUSTERED), (self.right, "spread", style.SPREAD)):
            dots = by_gid(self.page, f"start:{name}:trained")
            self.assertEqual(len(dots), 55)
            drawn = sorted(
                (round((dot.get_xdata()[0] - panel.x0_mm) / panel.scale, 6),
                 round((dot.get_ydata()[0] - panel.y0_mm) / panel.scale, 6))
                for dot in dots
            )  # fmt: skip
            measured = sorted((round(demo.measured.x_mm, 6), round(demo.measured.y_mm, 6)) for demo in AUDITS[name])
            self.assertEqual(drawn, measured)
            for dot in dots:
                self.assertEqual(dot.get_marker(), series.marker)
                self.assertEqual(to_hex(dot.get_markerfacecolor()), series.color)

    def test_the_measured_positions_are_drawn_not_the_planned_ones(self):
        dots = by_gid(self.page, "start:clustered:trained")
        centre = GRID.cell_centre("C2")
        on_table = [(dot.get_xdata()[0] - self.left.x0_mm) / self.left.scale for dot in dots]
        at_the_centre = [x for x in on_table if abs(x - centre.x_mm) < 1e-6]
        self.assertEqual(at_the_centre, [])  # no measured start is exactly on its planned point

    def test_a_point_where_several_start_says_how_many(self):
        counts = by_gid(self.page, "count:clustered")
        self.assertEqual([count.get_text() for count in counts], ["\N{MULTIPLICATION SIGN}5"] * 11)
        self.assertEqual(by_gid(self.page, "count:spread"), [])
        cells = set()
        for count in counts:
            x = (count.get_position()[0] - self.left.x0_mm) / self.left.scale
            y = (count.get_position()[1] - self.left.y0_mm) / self.left.scale
            cell = GRID.cell_of(Point(x, y))
            self.assertTrue(GRID.is_trained(cell))  # the count stays inside its own cell
            cells.add(cell)
        self.assertEqual(cells, set(GRID.trained_cells))

    def test_the_held_out_cells_are_hatched_in_both_pictures_and_hold_no_dot(self):
        self.assertEqual(len(by_gid(self.page, "held-out-hatching")), 2)
        for panel, name in ((self.left, "clustered"), (self.right, "spread")):
            for dot in by_gid(self.page, f"start:{name}:"):
                x = (dot.get_xdata()[0] - panel.x0_mm) / panel.scale
                y = (dot.get_ydata()[0] - panel.y0_mm) / panel.scale
                self.assertTrue(GRID.is_trained(GRID.cell_of(Point(x, y))))

    def test_cell_names_run_along_the_edges(self):
        self.assertEqual([text.get_text() for text in by_gid(self.page, "column-name")], list("ABCDE") * 2)
        self.assertEqual([text.get_text() for text in by_gid(self.page, "row-name")], list("123") * 2)

    def test_a_bar_gives_the_scale_in_millimetres(self):
        (bar,) = by_gid(self.page, "scale-bar")
        self.assertAlmostEqual(bar.get_xdata()[1] - bar.get_xdata()[0], 50.0 * self.left.scale)
        self.assertEqual([text.get_text() for text in by_gid(self.page, "scale-bar-label")], ["50 mm"])
        self.assertLess(bar.get_ydata()[0], self.left.y0_mm)  # below the grid, not across it

    def test_the_cubes_footprint_is_drawn_once_to_scale_and_not_around_a_start(self):
        (cube,) = by_gid(self.page, "cube-footprint")
        self.assertAlmostEqual(cube.get_width(), 30.0 * self.left.scale)
        self.assertAlmostEqual(cube.get_height(), 30.0 * self.left.scale)
        self.assertGreater(cube.get_x(), self.left.right_mm)  # between the two grids
        self.assertLess(cube.get_x() + cube.get_width(), self.right.x0_mm - gridmap.NUMBERS_MM + 1e-9)
        self.assertEqual([text.get_text() for text in by_gid(self.page, "cube-label")], ["cube"])
        # 17.2 mm between two spread starts is less than the cube is wide: the picture shows both to one scale.
        self.assertGreater(cube.get_width(), 17.2 * self.right.scale)

    def test_nothing_drawn_runs_off_the_page(self):
        for line in self.page.axes.lines:
            for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
                self.assertTrue(0.0 <= x <= 88.0 and 0.0 <= y <= 32.0, line.get_gid())
        for text in self.page.texts():
            width = style.text_width_mm(text.get_text(), text.get_fontsize(), bold=text.get_gid() == "panel-title")
            x = text.get_position()[0]
            left = {"left": x, "center": x - width / 2.0, "right": x - width}[text.get_ha()]
            self.assertTrue(left >= 0.0 and left + width <= 88.0, text.get_text())
            self.assertTrue(0.3 <= text.get_position()[1] <= 32.0 - 2.0, text.get_text())

    def test_the_audit_figure_has_no_stamp_and_no_small_text(self):
        self.assertEqual(self.page.stamps, [])
        self.assertGreaterEqual(min(text.get_fontsize() for text in self.page.texts()), 7.0)

    def test_an_episode_left_out_of_training_has_an_open_dot_and_the_figure_says_what_that_is(self):
        starts = {name: list(demos) for name, demos in AUDITS.items()}
        starts["spread"][3] = dataclasses.replace(starts["spread"][3], trained=False)
        page = lg.figure(starts, GRID)
        (open_dot,) = by_gid(page, "start:spread:not-trained")
        self.assertEqual(to_hex(open_dot.get_markerfacecolor()), "#ffffff")
        self.assertEqual(to_hex(open_dot.get_markeredgecolor()), style.ORANGE)
        self.assertEqual(len(by_gid(page, "start:spread:trained")), 54)
        self.assertEqual([text.get_text() for text in by_gid(page, "not-trained-label")], ["recorded, not trained on"])
        self.assertEqual(by_gid(self.page, "not-trained-label"), [])
        self.assertIn("54 starts at 54 points", [note.get_text() for note in by_gid(page, "panel-note")])


class PlannedPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = lg.figure(PLANNED, GRID, stamp="PLANNED")

    def test_a_picture_of_the_plan_says_planned_across_it(self):
        self.assertEqual(self.page.stamps, ["PLANNED"])
        (stamp,) = by_gid(self.page, "stamp:PLANNED")
        self.assertEqual(stamp.get_text(), "PLANNED")
        self.assertGreater(stamp.get_fontsize(), 20.0)

    def test_the_planned_positions_are_the_committed_layouts(self):
        for name in ("clustered", "spread"):
            committed = layouts.read_layout(layouts.layout_path(name))
            self.assertEqual(layouts.planned_demo_starts(committed), PLANNED[name])
        left, _ = panels(self.page)
        dots = by_gid(self.page, "start:clustered:trained")
        places = {(round((dot.get_xdata()[0] - left.x0_mm) / left.scale, 6),
                   round((dot.get_ydata()[0] - left.y0_mm) / left.scale, 6)) for dot in dots}  # fmt: skip
        self.assertEqual(places, {(GRID.cell_centre(cell).x_mm, GRID.cell_centre(cell).y_mm)
                                  for cell in GRID.trained_cells})  # fmt: skip
        self.assertEqual(len(by_gid(self.page, "count:clustered")), 11)


class RefusalTests(unittest.TestCase):
    def test_both_sets_are_needed(self):
        with self.assertRaisesRegex(FigureError, "both sets"):
            lg.figure({"clustered": AUDITS["clustered"]}, GRID)

    def test_a_set_under_the_wrong_name_is_refused(self):
        with self.assertRaisesRegex(FigureError, "spread"):
            lg.figure({"clustered": AUDITS["spread"], "spread": AUDITS["spread"]}, GRID)

    def test_an_empty_set_and_a_start_off_the_grid_are_refused(self):
        with self.assertRaisesRegex(FigureError, "no demonstration"):
            lg.figure({"clustered": [], "spread": AUDITS["spread"]}, GRID)
        off = [*AUDITS["clustered"][:-1], DemoStart("clustered", 54, "E2", measured=Point(260.0, 75.0))]
        with self.assertRaisesRegex(FigureError, "off the grid"):
            lg.figure({"clustered": off, "spread": AUDITS["spread"]}, GRID)


class FileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = {
            "audit": style.pdf_bytes(lg.figure(AUDITS, GRID)),
            "planned": style.pdf_bytes(lg.figure(PLANNED, GRID, stamp="PLANNED")),
        }

    def test_the_page_is_88_by_32_mm(self):
        for name, data in self.files.items():
            width, height = pdf_facts.page_size_mm(data)
            self.assertAlmostEqual(width, 88.0, delta=0.5, msg=name)
            self.assertAlmostEqual(height, 32.0, delta=0.5, msg=name)

    def test_no_text_is_under_seven_points(self):
        for name, data in self.files.items():
            sizes = pdf_facts.text_sizes_pt(data)
            self.assertGreater(len(sizes), 20, name)
            self.assertGreaterEqual(min(sizes), 7.0, name)

    def test_the_bytes_say_nothing_of_the_maker(self):
        for name, data in self.files.items():
            self.assertEqual(pdf_facts.forbidden_hits(data), [], name)
            self.assertEqual(pdf_facts.names_this_machine(data), [], name)
            self.assertNotIn(b"CreationDate", data, name)

    def test_the_same_starts_give_the_same_bytes(self):
        self.assertEqual(style.pdf_bytes(lg.figure(AUDITS, GRID)), self.files["audit"])
        self.assertNotEqual(self.files["audit"], self.files["planned"])

    def test_the_fonts_are_embedded_truetype(self):
        for data in self.files.values():
            self.assertEqual(pdf_facts.font_kinds(data), {b"Type0", b"CIDFontType2"})


if __name__ == "__main__":
    unittest.main()
