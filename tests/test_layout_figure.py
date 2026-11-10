"""The preview figure of the two layouts: what is drawn, and that it is saved as a picture."""

import tempfile
import unittest
from pathlib import Path

from fixtures_demos import GRID, PROTOCOL
from tolerance_radius import layout_figure, layouts

LAYOUTS = {name: layouts.layout(name, PROTOCOL) for name in ("clustered", "spread")}


class LayoutPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.figure = layout_figure.layout_preview(LAYOUTS, GRID)
        cls.panels = dict(zip(LAYOUTS, cls.figure.axes, strict=True))

    def patches(self, name: str, gid: str) -> list:
        return [patch for patch in self.panels[name].patches if (patch.get_gid() or "").startswith(gid)]

    def texts(self, name: str) -> list[str]:
        return [text.get_text() for text in self.panels[name].texts]

    def test_there_is_one_panel_for_each_layout_named_after_it(self):
        self.assertEqual(len(self.figure.axes), 2)
        for name, panel in self.panels.items():
            self.assertIn(name, panel.get_title(loc="left"))

    def test_every_start_position_is_a_dot_at_its_target(self):
        for name, episodes in LAYOUTS.items():
            (dots,) = self.panels[name].collections
            drawn = {(float(x), float(y)) for x, y in dots.get_offsets()}
            self.assertEqual(drawn, {(e.target.x_mm, e.target.y_mm) for e in episodes})

    def test_every_start_position_carries_the_outline_of_the_cube(self):
        for name, episodes in LAYOUTS.items():
            footprints = self.patches(name, "footprint")
            self.assertEqual(len(footprints), len({e.target for e in episodes}))
            centres = set()
            for footprint in footprints:
                self.assertEqual((footprint.get_width(), footprint.get_height()), (30.0, 30.0))
                self.assertEqual(footprint.get_facecolor()[3], 0.0)  # an outline, not a filled square
                self.assertLess(footprint.get_edgecolor()[3], 0.5)  # and a faint one
                centres.add((footprint.get_x() + 15.0, footprint.get_y() + 15.0))
            self.assertEqual(centres, {(e.target.x_mm, e.target.y_mm) for e in episodes})

    def test_held_out_cells_are_shaded_and_say_so_and_trained_cells_are_not(self):
        for name in LAYOUTS:
            shaded = {patch.get_gid(): patch for patch in self.patches(name, "held-out")}
            self.assertEqual(set(shaded), {f"held-out:{cell}" for cell in GRID.held_out_cells})
            for cell in GRID.held_out_cells:
                rect, patch = GRID.cell_rect(cell), shaded[f"held-out:{cell}"]
                self.assertEqual((patch.get_x(), patch.get_y()), (rect.x_min_mm, rect.y_min_mm))
                self.assertEqual((patch.get_width(), patch.get_height()), (50.0, 50.0))
            self.assertEqual(self.texts(name).count("held out"), len(GRID.held_out_cells))

    def test_columns_are_lettered_along_the_top_and_rows_numbered_along_the_side(self):
        # Labels inside the cells would sit under the dots: the letters and numbers go on the edges instead.
        for name, panel in self.panels.items():
            top, side = panel.child_axes
            self.assertEqual([label.get_text() for label in top.get_xticklabels()], ["A", "B", "C", "D", "E"])
            self.assertEqual(top.get_xticks().tolist(), [25.0, 75.0, 125.0, 175.0, 225.0])
            self.assertEqual([label.get_text() for label in side.get_yticklabels()], ["1", "2", "3"])
            self.assertEqual(side.get_yticks().tolist(), [25.0, 75.0, 125.0])
            self.assertEqual([text for text in self.texts(name) if text != "held out"], [])

    def test_the_test_region_is_outlined(self):
        for name in LAYOUTS:
            (outline,) = self.patches(name, "test-region")
            self.assertEqual((outline.get_x(), outline.get_y()), (5.0, 5.0))
            self.assertEqual((outline.get_width(), outline.get_height()), (240.0, 140.0))

    def test_the_largest_gap_is_drawn_as_the_circle_it_leaves_empty(self):
        for name, episodes in LAYOUTS.items():
            quality = layouts.layout_quality(episodes, GRID)
            (circle,) = self.patches(name, "largest-gap")
            self.assertAlmostEqual(circle.get_radius(), quality.largest_gap_mm)
            self.assertEqual(tuple(circle.get_center()), (quality.largest_gap_at.x_mm, quality.largest_gap_at.y_mm))
            self.assertIn(f"{quality.largest_gap_mm:.1f} mm", self.panels[name].get_title(loc="left"))

    def test_a_millimetre_is_as_long_across_as_up_and_the_whole_grid_and_every_cube_fit(self):
        for panel in self.panels.values():
            self.assertEqual(panel.get_aspect(), 1.0)
            x_low, x_high = panel.get_xlim()
            y_low, y_high = panel.get_ylim()
            self.assertLessEqual(x_low, -10.0)
            self.assertGreaterEqual(x_high, 260.0)
            self.assertLessEqual(y_low, -10.0)
            self.assertGreaterEqual(y_high, 160.0)
        limits = {(panel.get_xlim(), panel.get_ylim()) for panel in self.panels.values()}
        self.assertEqual(len(limits), 1)  # the panels can be compared by eye: the same scale and the same window

    def test_the_circle_of_the_largest_gap_is_shown_whole(self):
        for name, panel in self.panels.items():
            (circle,) = self.patches(name, "largest-gap")
            (x, y), radius = circle.get_center(), circle.get_radius()
            self.assertLess(panel.get_xlim()[0], x - radius)
            self.assertGreater(panel.get_xlim()[1], x + radius)
            self.assertLess(panel.get_ylim()[0], y - radius)
            self.assertGreater(panel.get_ylim()[1], y + radius)

    def test_the_figure_explains_its_marks_in_words(self):
        words = " ".join(text.get_text() for text in self.figure.texts).lower()
        for word in ("footprint", "held out", "test region", "largest gap"):
            self.assertIn(word, words)

    def test_it_is_saved_as_a_png_in_a_folder_made_for_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis" / "layouts" / "layouts.png"
            layout_figure.save_layout_preview(path, LAYOUTS, GRID)
            data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(data), 20_000)
        self.assertNotIn(str(Path.home()).encode(), data)
        # No note of the drawing library's version in the file: the same picture drawn elsewhere is the same file.
        self.assertNotIn(b"Software", data)
        self.assertNotIn(b"atplotlib", data)


if __name__ == "__main__":
    unittest.main()
