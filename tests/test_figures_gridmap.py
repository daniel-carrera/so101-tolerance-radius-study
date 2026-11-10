"""The grid as a picture on the page: where a picture sits, its scale, and what is drawn of the grid."""

import unittest

from fixtures_demos import GRID
from tolerance_radius.figures import gridmap, style
from tolerance_radius.figures.gridmap import Panel
from tolerance_radius.figures.style import FigureError, Page
from tolerance_radius.schema import Point


def texts(page: Page, gid: str) -> list:
    return [text for text in page.texts() if text.get_gid() == gid]


class PanelTests(unittest.TestCase):
    PANEL = Panel(x0_mm=3.0, y0_mm=4.0, scale=0.15, grid=GRID)

    def test_table_millimetres_become_page_millimetres(self):
        self.assertEqual((self.PANEL.x(0.0), self.PANEL.y(0.0)), (3.0, 4.0))
        self.assertAlmostEqual(self.PANEL.x(250.0), 40.5)
        self.assertAlmostEqual(self.PANEL.y(150.0), 26.5)
        self.assertAlmostEqual(self.PANEL.length(30.0), 4.5)
        self.assertAlmostEqual(self.PANEL.right_mm, 40.5)
        self.assertAlmostEqual(self.PANEL.top_mm, 26.5)

    def test_y_runs_up_the_page_so_row_1_is_at_the_bottom(self):
        row_1, row_3 = GRID.cell_centre("C1"), GRID.cell_centre("C3")
        self.assertLess(self.PANEL.y(row_1.y_mm), self.PANEL.y(row_3.y_mm))
        self.assertLess(self.PANEL.x(GRID.cell_centre("A2").x_mm), self.PANEL.x(GRID.cell_centre("E2").x_mm))

    def test_an_artist_given_in_table_millimetres_lands_at_its_place(self):
        page = Page(32.0)
        to_display = self.PANEL.transform(page)
        point = Point(125.0, 75.0)
        on_page = page.axes.transData.transform((self.PANEL.x(point.x_mm), self.PANEL.y(point.y_mm)))
        through_panel = to_display.transform((point.x_mm, point.y_mm))
        self.assertAlmostEqual(float(on_page[0]), float(through_panel[0]))
        self.assertAlmostEqual(float(on_page[1]), float(through_panel[1]))


class SideBySideTests(unittest.TestCase):
    def test_two_pictures_have_one_scale_and_fit_the_page(self):
        page = Page(32.0)
        left, right = gridmap.side_by_side(page, GRID, top_mm=31.7, bottom_mm=3.2)
        self.assertEqual(left.scale, right.scale)
        self.assertEqual(left.y0_mm, right.y0_mm)
        self.assertGreaterEqual(left.x0_mm - gridmap.NUMBERS_MM, gridmap.SIDE_MM - 1e-9)
        self.assertLessEqual(right.right_mm, 88.0 - gridmap.SIDE_MM + 1e-9)
        self.assertGreaterEqual(left.y0_mm, 3.2 - 1e-9)
        self.assertLessEqual(left.top_mm + gridmap.LETTERS_MM + gridmap.TITLE_MM, 31.7 + 1e-9)
        self.assertGreater(right.x0_mm - gridmap.NUMBERS_MM, left.right_mm)  # they do not touch

    def test_the_pictures_are_as_large_as_the_tighter_of_width_and_height_allows(self):
        tall, short = Page(60.0), Page(20.0)
        (by_width, _), (by_height, _) = (
            gridmap.side_by_side(tall, GRID, top_mm=59.0, bottom_mm=1.0),
            gridmap.side_by_side(short, GRID, top_mm=19.0, bottom_mm=1.0),
        )
        self.assertAlmostEqual(by_width.scale, (88.0 - 2 * gridmap.SIDE_MM - 2 * gridmap.NUMBERS_MM - 1.5) / 500.0)
        self.assertAlmostEqual(by_height.scale, (18.0 - gridmap.TITLE_MM - gridmap.LETTERS_MM) / 150.0)
        self.assertLess(by_height.scale, by_width.scale)

    def test_room_between_the_pictures_can_be_kept_at_their_own_scale(self):
        page = Page(60.0)
        left, right = gridmap.side_by_side(page, GRID, top_mm=59.0, bottom_mm=1.0, gap_mm=2.0, gap_table_mm=30.0)
        between = right.x0_mm - gridmap.NUMBERS_MM - left.right_mm
        self.assertAlmostEqual(between, 2.0 + left.length(30.0))  # room for the cube's footprint and 1 mm either side

    def test_width_that_is_left_over_is_shared_out(self):
        page = Page(20.0)  # the height decides the scale, so there is width to spare
        left, right = gridmap.side_by_side(page, GRID, top_mm=19.0, bottom_mm=1.0)
        margin_left = left.x0_mm - gridmap.NUMBERS_MM
        margin_right = 88.0 - right.right_mm
        self.assertAlmostEqual(margin_left, margin_right)
        self.assertGreater(margin_left, gridmap.SIDE_MM + 1.0)

    def test_without_names_the_pictures_take_the_room_of_the_letters_and_numbers(self):
        named = gridmap.side_by_side(Page(60.0), GRID, top_mm=59.0, bottom_mm=1.0)
        plain = gridmap.side_by_side(Page(60.0), GRID, top_mm=59.0, bottom_mm=1.0, names=False)
        self.assertAlmostEqual(plain[0].scale, (88.0 - 2 * gridmap.SIDE_MM - 1.5) / 500.0)
        self.assertGreater(plain[0].scale, named[0].scale)
        self.assertAlmostEqual(plain[0].x0_mm, gridmap.SIDE_MM)  # nothing is kept free at the left
        self.assertAlmostEqual(plain[1].right_mm, 88.0 - gridmap.SIDE_MM)
        self.assertAlmostEqual(plain[0].top_mm, 59.0 - gridmap.TITLE_MM)  # the title sits right above the grid
        self.assertAlmostEqual(named[0].top_mm, 59.0 - gridmap.TITLE_MM - gridmap.LETTERS_MM)
        short = gridmap.side_by_side(Page(20.0), GRID, top_mm=19.0, bottom_mm=1.0, names=False)
        self.assertAlmostEqual(short[0].scale, (18.0 - gridmap.TITLE_MM) / 150.0)

    def test_no_room_is_an_error(self):
        with self.assertRaises(FigureError):
            gridmap.side_by_side(Page(20.0), GRID, top_mm=5.0, bottom_mm=4.0)
        with self.assertRaises(FigureError):
            gridmap.side_by_side(Page(20.0), GRID, count=0, top_mm=19.0, bottom_mm=1.0)


class DrawGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = Page(32.0)
        cls.panel, cls.other = gridmap.side_by_side(cls.page, GRID, top_mm=31.7, bottom_mm=3.2)
        gridmap.draw_grid(cls.page, cls.panel, title="Clustered", note="55 starts at 11 points")
        gridmap.draw_grid(cls.page, cls.other)

    def test_the_columns_are_lettered_along_the_top_and_the_rows_numbered_along_the_left(self):
        letters = texts(self.page, "column-name")[:5]
        self.assertEqual([text.get_text() for text in letters], ["A", "B", "C", "D", "E"])
        for index, letter in enumerate(letters):
            self.assertAlmostEqual(letter.get_position()[0], self.panel.x(25.0 + 50.0 * index))
            self.assertGreater(letter.get_position()[1], self.panel.top_mm)
        numbers = texts(self.page, "row-name")[:3]
        self.assertEqual([text.get_text() for text in numbers], ["1", "2", "3"])
        heights = [number.get_position()[1] for number in numbers]
        self.assertEqual(heights, sorted(heights))  # row 1 at the bottom
        for number in numbers:
            self.assertLess(number.get_position()[0], self.panel.x0_mm)
        self.assertEqual(len(texts(self.page, "column-name")), 10)  # both pictures name their columns

    def test_the_outline_is_the_grids_250_by_150_mm(self):
        outline = [patch for patch in self.page.axes.patches if patch.get_gid() == "grid-outline"][0]
        self.assertAlmostEqual(outline.get_x(), self.panel.x(0.0))
        self.assertAlmostEqual(outline.get_y(), self.panel.y(0.0))
        self.assertAlmostEqual(outline.get_width(), self.panel.length(250.0))
        self.assertAlmostEqual(outline.get_height(), self.panel.length(150.0))
        borders = [line for line in self.page.axes.lines if line.get_gid() == "cell-border"]
        self.assertEqual(len(borders), 2 * (4 + 2))

    def test_the_held_out_cells_and_only_they_are_hatched(self):
        hatching = [c for c in self.page.axes.collections if c.get_gid() == "held-out-hatching"][0]
        held_out = gridmap.cell_rects(self.panel, GRID.held_out_cells)
        hatched_cells = set()
        for segment in hatching.get_segments():
            (xa, ya), (xb, yb) = segment
            middle = Point((xa + xb) / 2.0, (ya + yb) / 2.0)
            inside = [index for index, rect in enumerate(held_out) if rect.contains(middle)]
            self.assertEqual(len(inside), 1, "a hatch line lies outside the held-out cells")
            rect = held_out[inside[0]]
            for x, y in ((xa, ya), (xb, yb)):
                self.assertTrue(rect.x_min_mm - 1e-6 <= x <= rect.x_max_mm + 1e-6)
                self.assertTrue(rect.y_min_mm - 1e-6 <= y <= rect.y_max_mm + 1e-6)
            hatched_cells.add(inside[0])
        self.assertEqual(hatched_cells, {0, 1, 2, 3})
        self.assertEqual(sorted(GRID.held_out_cells), ["A1", "B2", "D2", "E3"])
        self.assertGreater(len(hatching.get_segments()), 4 * 8)  # dense enough to read as a hatching

    def test_the_hatching_is_quieter_than_anything_drawn_on_it(self):
        hatching = [c for c in self.page.axes.collections if c.get_gid() == "held-out-hatching"][0]
        self.assertLess(hatching.get_zorder(), 2.0)
        self.assertLessEqual(float(hatching.get_linewidths()[0]), 0.5)

    def test_a_title_goes_above_the_letters_at_the_left_and_a_note_at_the_right(self):
        (title,) = texts(self.page, "panel-title")
        (note,) = texts(self.page, "panel-note")
        (letter, *_) = texts(self.page, "column-name")
        self.assertEqual((title.get_text(), title.get_ha()), ("Clustered", "left"))
        self.assertEqual(title.get_position()[0], self.panel.x0_mm)
        self.assertEqual((note.get_text(), note.get_ha()), ("55 starts at 11 points", "right"))
        self.assertAlmostEqual(note.get_position()[0], self.panel.right_mm)
        self.assertGreater(title.get_position()[1], letter.get_position()[1] + 2.0)
        self.assertEqual(title.get_fontsize(), style.LABEL_PT)
        # The title and the note do not run into each other.
        used = style.text_width_mm("Clustered", style.LABEL_PT, bold=True) + style.text_width_mm(note.get_text())
        self.assertLess(used + 1.5, self.panel.right_mm - self.panel.x0_mm)

    def test_a_grid_can_be_drawn_without_the_names_of_its_columns_and_rows(self):
        page = Page(32.0)
        panel, _ = gridmap.side_by_side(page, GRID, top_mm=31.7, bottom_mm=3.2, names=False)
        gridmap.draw_grid(page, panel, title="Clustered policy", names=False)
        self.assertEqual(texts(page, "column-name") + texts(page, "row-name"), [])
        (title,) = texts(page, "panel-title")
        self.assertAlmostEqual(title.get_position()[1], panel.top_mm + 0.75)
        self.assertLess(title.get_position()[1] + 2.0, 31.7)
        self.assertEqual(len([c for c in page.axes.collections if c.get_gid() == "held-out-hatching"]), 1)
        self.assertEqual(len([line for line in page.axes.lines if line.get_gid() == "cell-border"]), 6)

    def test_a_scale_bar_is_as_long_on_the_page_as_its_length_on_the_table(self):
        page = Page(32.0)
        panel, _ = gridmap.side_by_side(page, GRID, top_mm=31.7, bottom_mm=3.2)
        gridmap.draw_scale_bar(page, panel, 50.0, x_table_mm=0.0, y_mm=1.6)
        (bar,) = [line for line in page.axes.lines if line.get_gid() == "scale-bar"]
        self.assertAlmostEqual(bar.get_xdata()[1] - bar.get_xdata()[0], panel.length(50.0))
        self.assertAlmostEqual(bar.get_xdata()[0], panel.x(0.0))
        (label,) = texts(page, "scale-bar-label")
        self.assertEqual(label.get_text(), "50 mm")
        self.assertGreater(label.get_position()[0], bar.get_xdata()[1])
        self.assertEqual(len([line for line in page.axes.lines if line.get_gid() == "scale-bar-end"]), 2)


if __name__ == "__main__":
    unittest.main()
