"""What the figures share: the page in millimetres, the type, the colours, and a file that says nothing of its maker."""

import itertools
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
from matplotlib.lines import Line2D

import pdf_facts
from tolerance_radius.figures import style
from tolerance_radius.figures.style import FigureError, Forbidden, Page

# Machado, Oliveira and Fernandes (2009): what a colour looks like without one kind of cone, at full severity, as a
# matrix on linear RGB. The colour differences below are distances in OKLab times 100, the measure the project's
# charting method sets its floors in: 8 between two colours for a colour-blind reader, 15 with full colour vision.
MACHADO = {
    "protanopia": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deuteranopia": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    "tritanopia": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}


def linear_rgb(colour: str) -> list[float]:
    channels = [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]


def oklab(rgb: list[float]) -> tuple[float, float, float]:
    r, g, b = rgb
    long = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    medium = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    short = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def colour_difference(first: str, second: str, vision: str | None = None) -> float:
    """How far apart two colours look, to full colour vision or with one kind of cone missing."""

    def seen(colour: str) -> tuple[float, float, float]:
        rgb = linear_rgb(colour)
        if vision is not None:
            rgb = [min(max(sum(m * c for m, c in zip(row, rgb, strict=True)), 0.0), 1.0) for row in MACHADO[vision]]
        return oklab(rgb)

    return 100.0 * math.dist(seen(first), seen(second))


def lightness(colour: str) -> float:
    """CIE L*: how light a colour prints in grey, from 0 (black) to 100 (white)."""
    r, g, b = linear_rgb(colour)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 116.0 * luminance ** (1.0 / 3.0) - 16.0


def contrast(first: str, second: str) -> float:
    def luminance(colour: str) -> float:
        r, g, b = linear_rgb(colour)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def a_page() -> Page:
    """A page with a line of each kind of text and a few marks, as a figure would have."""
    page = Page(40.0)
    page.text(44.0, 3.0, "Distance to the nearest demonstration, $d$ (mm)", size_pt=style.LABEL_PT, ha="center")
    page.text(5.0, 20.0, "Success rate", size_pt=style.LABEL_PT, rotation=90.0)
    page.text(20.0, 30.0, "19 of 19 \N{MULTIPLICATION SIGN}5", bold=True)
    page.axes.add_line(Line2D([20.0, 30.0, 40.0], [15.0, 18.0, 15.0], color=style.BLUE, marker="o", markersize=4.0))
    page.axes.add_line(Line2D([20.0, 40.0], [10.0, 10.0], color=style.ORANGE, dashes=style.SPREAD.dashes))
    return page


class ColourTests(unittest.TestCase):
    def test_the_two_colours_stay_apart_for_colour_blind_readers(self):
        for vision in MACHADO:
            self.assertGreaterEqual(colour_difference(style.BLUE, style.ORANGE, vision), 8.0, vision)
        self.assertGreaterEqual(colour_difference(style.BLUE, style.ORANGE), 15.0)

    def test_the_figures_documented_differences_are_the_measured_ones(self):
        # The numbers quoted beside the colours in the style module.
        worst_red_green = min(colour_difference(style.BLUE, style.ORANGE, v) for v in ("protanopia", "deuteranopia"))
        self.assertAlmostEqual(worst_red_green, 24.4, places=1)
        self.assertAlmostEqual(colour_difference(style.BLUE, style.ORANGE, "tritanopia"), 34.9, places=1)
        self.assertAlmostEqual(colour_difference(style.BLUE, style.ORANGE), 35.7, places=1)

    def test_printed_in_grey_the_two_colours_and_the_ink_are_three_different_greys(self):
        blue, orange, ink = (lightness(colour) for colour in (style.BLUE, style.ORANGE, style.INK))
        self.assertAlmostEqual(blue, 39.3, places=0)
        self.assertAlmostEqual(orange, 59.7, places=0)
        self.assertGreater(orange - blue, 15.0)
        self.assertGreater(blue - ink, 15.0)

    def test_every_colour_of_a_mark_stands_out_from_the_page(self):
        for colour in (style.BLUE, style.ORANGE, style.INK):
            self.assertGreaterEqual(contrast(colour, "#ffffff"), 3.0)

    def test_a_tint_is_light_enough_to_carry_marks_of_its_own_colour_and_of_ink(self):
        for colour, tint in ((style.BLUE, style.BLUE_TINT), (style.ORANGE, style.ORANGE_TINT)):
            self.assertGreater(lightness(tint), 85.0)
            self.assertGreaterEqual(contrast(colour, tint), 2.5)
            self.assertGreaterEqual(contrast(style.INK, tint), 10.0)

    def test_what_colour_says_a_marker_and_a_line_style_say_again(self):
        series = (style.CLUSTERED, style.SPREAD, style.REPLAY)
        for first, second in itertools.combinations(series, 2):
            self.assertNotEqual(first.color, second.color)
            self.assertNotEqual(first.marker, second.marker)
            self.assertNotEqual(first.dashes, second.dashes)
        self.assertEqual(style.CLUSTERED.color, style.BLUE)
        self.assertEqual(style.SPREAD.color, style.ORANGE)
        self.assertEqual(style.REPLAY.color, style.INK)  # no policy, no colour


class PageTests(unittest.TestCase):
    def test_a_page_is_one_column_wide_and_measured_in_millimetres(self):
        page = Page(42.0)
        width_in, height_in = page.figure.get_size_inches()
        self.assertAlmostEqual(width_in * 25.4, 88.0)
        self.assertAlmostEqual(height_in * 25.4, 42.0)
        self.assertEqual(page.axes.get_xlim(), (0.0, 88.0))
        self.assertEqual(page.axes.get_ylim(), (0.0, 42.0))
        self.assertEqual(tuple(page.axes.get_position().bounds), (0.0, 0.0, 1.0, 1.0))
        self.assertFalse(page.axes.axison)
        with self.assertRaises(FigureError):
            Page(0.0)

    def test_text_is_set_in_the_fonts_that_ship_with_matplotlib(self):
        page = Page(30.0)
        fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for options, name in (({}, "STIXGeneral.ttf"), ({"bold": True}, "STIXGeneralBol.ttf"),
                              ({"italic": True}, "STIXGeneralItalic.ttf")):  # fmt: skip
            text = page.text(10.0, 10.0, "radius", **options)
            self.assertEqual(Path(text.get_fontproperties().get_file()), fonts / name)
            self.assertTrue((fonts / name).is_file())
            self.assertEqual(text.get_fontproperties().get_math_fontfamily(), "stix")

    def test_text_below_seven_points_is_refused(self):
        page = Page(30.0)
        self.assertEqual(page.text(1.0, 1.0, "small enough", size_pt=7.0).get_fontsize(), 7.0)
        with self.assertRaises(FigureError):
            page.text(1.0, 1.0, "too small", size_pt=6.9)
        self.assertEqual(style.MIN_TEXT_PT, 7.0)
        self.assertGreaterEqual(style.TEXT_PT, style.MIN_TEXT_PT)

    def test_text_goes_where_it_is_put(self):
        page = Page(30.0)
        text = page.text(12.5, 7.25, "here", ha="right")
        self.assertEqual(text.get_position(), (12.5, 7.25))
        self.assertEqual(text.get_ha(), "right")
        self.assertEqual(page.texts(), [text])

    def test_a_stamp_is_one_large_word_across_the_page(self):
        page = Page(40.0)
        stamp = page.stamp("DRAFT")
        self.assertEqual(page.stamps, ["DRAFT"])
        self.assertEqual(stamp.get_text(), "DRAFT")
        self.assertEqual(stamp.get_position(), (44.0, 20.0))
        self.assertGreaterEqual(stamp.get_fontsize(), 24.0)
        width_mm = style.text_width_mm("DRAFT", stamp.get_fontsize(), bold=True)
        self.assertGreater(width_mm, 40.0)  # it cannot be overlooked
        self.assertLess(width_mm, 80.0)  # and it is whole
        self.assertGreater(stamp.get_zorder(), 20.0)  # over everything
        self.assertGreater(stamp.get_alpha(), 0.1)

    def test_the_width_of_a_text_is_known_before_it_is_drawn(self):
        self.assertEqual(style.text_width_mm(""), 0.0)
        word = style.text_width_mm("demonstration", 8.0)
        self.assertGreater(word, 14.0)
        self.assertLess(word, 18.0)
        self.assertAlmostEqual(style.text_width_mm("demonstration", 16.0), 2.0 * word, delta=0.1)
        self.assertGreater(style.text_width_mm("demonstration", 8.0, bold=True), word)

    def test_matplotlibs_settings_are_put_back_and_do_not_leak_in(self):
        before = matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["font.size"]
        with matplotlib.rc_context({"pdf.fonttype": 3, "font.size": 23.0, "lines.linewidth": 9.0}):
            with style.settings():
                self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
                self.assertEqual(matplotlib.rcParams["lines.linewidth"], matplotlib.rcParamsDefault["lines.linewidth"])
                self.assertEqual(matplotlib.rcParams["font.family"], ["STIXGeneral"])
                self.assertFalse(matplotlib.rcParams["text.usetex"])
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 3)
            self.assertEqual(matplotlib.rcParams["font.size"], 23.0)
        self.assertEqual((matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["font.size"]), before)


    def test_a_block_of_settings_inside_another_leaves_them_as_they_are_and_the_outer_one_puts_them_back(self):
        before = matplotlib.rcParams["font.size"], matplotlib.rcParams["pdf.fonttype"]
        with style.settings():
            matplotlib.rcParams["font.size"] = 11.0  # as a drawing in progress might
            with style.settings():
                self.assertEqual(matplotlib.rcParams["font.size"], 11.0)  # not set again from scratch
                self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
            self.assertEqual(matplotlib.rcParams["font.size"], 11.0)
            with self.assertRaises(ZeroDivisionError), style.settings():
                1 / 0
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)  # an error inside does not end the outer block
        self.assertEqual((matplotlib.rcParams["font.size"], matplotlib.rcParams["pdf.fonttype"]), before)
        with style.settings():  # and the next block starts from scratch again
            self.assertEqual(matplotlib.rcParams["font.size"], style.LABEL_PT)
        with self.assertRaises(ZeroDivisionError), style.settings():
            1 / 0
        self.assertEqual((matplotlib.rcParams["font.size"], matplotlib.rcParams["pdf.fonttype"]), before)
        with style.settings():
            self.assertEqual(matplotlib.rcParams["font.size"], style.LABEL_PT)


class SavedFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = style.pdf_bytes(a_page())

    def test_the_page_of_the_file_is_the_page_drawn(self):
        width, height = pdf_facts.page_size_mm(self.data)
        self.assertAlmostEqual(width, 88.0, places=3)
        self.assertAlmostEqual(height, 40.0, places=3)

    def test_the_same_page_gives_the_same_bytes(self):
        self.assertEqual(style.pdf_bytes(a_page()), self.data)
        self.assertEqual(style.pdf_bytes(a_page()), style.pdf_bytes(a_page()))

    def test_the_file_carries_no_note_of_who_made_it_with_what_or_when(self):
        for word in (b"CreationDate", b"ModDate", b"Producer", b"Creator", b"Author", b"Title", b"atplotlib"):
            self.assertNotIn(word, self.data)
        self.assertEqual(pdf_facts.forbidden_hits(self.data), [])
        self.assertEqual(pdf_facts.names_this_machine(self.data), [])
        information = re.search(rb"/Info (\d+) 0 R", self.data)
        self.assertIsNotNone(information)
        number = information.group(1)
        self.assertRegex(self.data, rb"\n" + number + rb" 0 obj\n<< >>\nendobj")  # the dictionary is there, empty

    def test_the_fonts_are_embedded_as_truetype(self):
        self.assertEqual(pdf_facts.font_kinds(self.data), {b"Type0", b"CIDFontType2"})
        self.assertNotIn(b"/Type3", self.data)
        self.assertGreaterEqual(self.data.count(b"/FontFile2"), 3)  # the regular, the bold and the italic of $d$
        self.assertIn(b"+STIXGeneral-Regular", self.data)
        self.assertIn(b"+STIXGeneral-Italic", self.data)

    def test_the_sizes_of_the_text_can_be_read_from_the_file(self):
        sizes = pdf_facts.text_sizes_pt(self.data)
        self.assertEqual({round(size, 6) for size in sizes}, {7.0, 8.0})
        self.assertGreater(len(sizes), 3)

    def test_a_text_drawn_small_is_seen_small_in_the_file(self):
        page = a_page()
        page.axes.text(10.0, 10.0, "footnote", fontsize=5.5)  # past the page's own guard
        self.assertEqual(min(pdf_facts.text_sizes_pt(style.pdf_bytes(page))), 5.5)

    def test_the_drawing_is_vector_not_a_picture(self):
        self.assertNotIn(b"/Subtype /Image", self.data)
        self.assertLess(len(self.data), 40_000)


class SaveTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.folder = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_it_writes_the_file_in_a_folder_made_for_it(self):
        path = self.folder / "paper" / "figures" / "fig0-test.pdf"
        style.save_pdf(a_page(), path)
        self.assertEqual(path.read_bytes(), style.pdf_bytes(a_page()))
        self.assertEqual(path.read_bytes()[:5], b"%PDF-")

    def test_a_picture_for_looking_at_is_300_dots_per_inch(self):
        path = self.folder / "look" / "fig0-test.png"
        style.save_png(a_page(), path)
        data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        self.assertEqual((width, height), (1039, 472))  # 88 mm and 40 mm at 300 dots per inch
        self.assertNotIn(b"atplotlib", data)
        self.assertEqual(pdf_facts.names_this_machine(data), [])

    def test_a_text_that_is_too_small_is_not_saved(self):
        page = a_page()
        page.axes.text(10.0, 10.0, "footnote", fontsize=6.0)
        path = self.folder / "small.pdf"
        with self.assertRaisesRegex(FigureError, "6 pt"):
            style.save_pdf(page, path)
        self.assertFalse(path.exists())

    def test_a_text_that_names_this_machines_home_is_not_saved_and_the_error_does_not_repeat_it(self):
        page = a_page()
        page.text(1.0, 1.0, f"drawn from {Path.home()}/results.json")
        path = self.folder / "home.pdf"
        with self.assertRaises(FigureError) as caught:
            style.save_pdf(page, path)
        self.assertIn("home directory", str(caught.exception))
        self.assertNotIn(str(Path.home()), str(caught.exception))
        self.assertFalse(path.exists())

    def test_a_text_that_matches_a_forbidden_pattern_is_not_saved(self):
        forbidden = [Forbidden(re.compile(r"[a-z]+@[a-z]+\.[a-z]{2,}", re.I), "an address")]
        page = a_page()
        style.save_pdf(page, self.folder / "clean.pdf", forbidden=forbidden)
        page.text(1.0, 1.0, "write to someone@example.org")
        with self.assertRaisesRegex(FigureError, "an address"):
            style.save_pdf(page, self.folder / "address.pdf", forbidden=forbidden)
        self.assertFalse((self.folder / "address.pdf").exists())

    def test_bytes_that_match_by_accident_are_written_again_with_another_compression(self):
        forbidden = [Forbidden(re.compile(r"""(^|[\s"'(=:])~/"""), "a home path")]
        clean = style.pdf_bytes(a_page(), compression=9)
        unlucky = clean.replace(b"endstream", b" ~/endstream", 1)
        with mock.patch.object(style, "pdf_bytes", side_effect=[unlucky, clean]) as written:
            style.save_pdf(a_page(), self.folder / "retried.pdf", forbidden=forbidden)
        self.assertEqual((self.folder / "retried.pdf").read_bytes(), clean)
        self.assertEqual([call.kwargs["compression"] for call in written.call_args_list], [6, 9])

    def test_bytes_that_match_whatever_the_compression_are_not_saved(self):
        forbidden = [Forbidden(re.compile(r"/Type /Catalog"), "something in the file's plain text")]
        with self.assertRaisesRegex(FigureError, "plain text"):
            style.save_pdf(a_page(), self.folder / "never.pdf", forbidden=forbidden)
        self.assertFalse((self.folder / "never.pdf").exists())

    def test_a_date_of_creation_in_the_bytes_is_refused(self):
        dated = style.pdf_bytes(a_page()).replace(b"<< >>", b"<< /CreationDate (D:20261005) >>", 1)
        with mock.patch.object(style, "pdf_bytes", return_value=dated):
            with self.assertRaisesRegex(FigureError, "creation date"):
                style.save_pdf(a_page(), self.folder / "dated.pdf")
        self.assertFalse((self.folder / "dated.pdf").exists())

    def test_different_compressions_change_the_compressed_bytes_only(self):
        first, second = (style.pdf_bytes(a_page(), compression=level) for level in (6, 1))
        self.assertNotEqual(first, second)
        self.assertEqual(pdf_facts.streams(first), pdf_facts.streams(second))


if __name__ == "__main__":
    unittest.main()
