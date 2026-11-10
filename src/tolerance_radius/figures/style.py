"""What the paper's figures share: the page, the type, the colours, and how a figure is saved.

The page
--------
A figure is one column wide (88 mm) and as high as the plan says. It is drawn on in **page millimetres**, measured
from its bottom-left corner: `Page` holds a Matplotlib figure with one pair of invisible axes that covers it, whose
data units are millimetres. A mark's place on the printed page is therefore what the code says, not what an automatic
layout decides, and a test can check it.

The type
--------
The STIX fonts that ship with Matplotlib, read from Matplotlib's own files and not looked up by name, so that a font
of the same name installed on a machine cannot take their place. They are close to the paper's Times. No text is
smaller than `MIN_TEXT_PT`: `Page.text` refuses a smaller size, and `save_pdf` checks again.

The colours
-----------
Blue stands for the clustered layout and its policy, orange for the spread layout and its policy, wherever they
appear. Everything that is neither (the replay, which has no policy; marks of outcomes; text and axes) is drawn in
ink. The pair was checked for colour-blind readers and for print in grey (see `BLUE`); each `Series` also has a marker
and a line style of its own, so that nothing is told by colour alone.

Saving
------
`save_pdf` writes a vector PDF with its fonts embedded as TrueType and with no metadata: no creator, no producer, no
date. Before writing it checks that nothing drawn, and nothing in the file's bytes, names a path or a user. The same
figure gives the same bytes every time.
"""

from __future__ import annotations

import contextlib
import getpass
import io
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.textpath import TextPath

MM_PER_INCH = 25.4
MM_PER_PT = MM_PER_INCH / 72.0

#: The width of every figure: one column of the paper.
COLUMN_MM = 88.0

#: No text in a figure is smaller than this at printed size.
MIN_TEXT_PT = 7.0
#: Axis labels and panel titles.
LABEL_PT = 8.0
#: Tick labels, notes inside a figure, and everything else.
TEXT_PT = 7.0

# --- colours --------------------------------------------------------------------------------------------------------
#: Text, axes, and marks that belong to neither demonstration layout.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
#: Axis lines and tick marks: present, and quieter than the data.
AXIS = "#6f6e69"
#: Hairlines that only guide the eye (the levels of the rate axis, the borders of the cells).
GRIDLINE = "#d9d8d0"
#: The background of a stretch in which nothing was tested.
SHADE = "#e7e6e0"
#: The lines of the hatching on held-out cells.
HATCH = "#aaa9a1"

#: The pair that tells the two layouts apart. Checked with the palette validator of the project's charting method on
#: a white page (2026-10-05): difference under simulated protanopia and deuteranopia 24.4 and under tritanopia 34.9
#: (the floor is 8), with full colour vision 35.7 (the floor is 15), both above 3:1 against white. Printed in grey
#: the blue is the darker one: lightness (CIE L*) 39 against 60.
BLUE = "#1c5cab"
ORANGE = "#eb6834"
#: Light versions for areas (the discs of Figure 4), which must stay behind the marks drawn on them.
BLUE_TINT = "#d3e4f9"
ORANGE_TINT = "#fbdfd3"

# --- line widths, in points -----------------------------------------------------------------------------------------
HAIRLINE_PT = 0.4
AXIS_PT = 0.5
CURVE_PT = 1.1
INTERVAL_PT = 0.8

_FONT_FOLDER = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
_FONT_FILES = {
    (False, False): "STIXGeneral.ttf",
    (True, False): "STIXGeneralBol.ttf",
    (False, True): "STIXGeneralItalic.ttf",
    (True, True): "STIXGeneralBolIta.ttf",
}

#: Matplotlib's settings while a figure is built and saved. Everything else is Matplotlib's own default, whatever the
#: machine's configuration file says.
_SETTINGS: dict[str, Any] = {
    "pdf.fonttype": 42,  # fonts embedded as TrueType, not as drawings of each letter
    "pdf.compression": 6,
    "text.usetex": False,  # no TeX is installed where the figures are drawn
    "font.family": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": LABEL_PT,
    "axes.unicode_minus": True,
}
#: The entries Matplotlib would otherwise write into the file's information dictionary.
_NO_METADATA = {"Creator": None, "Producer": None, "CreationDate": None}
#: The strengths of compression tried in turn when the bytes of a file match a forbidden pattern by accident (see
#: `save_pdf`).
_COMPRESSION_LEVELS = (6, 9, 3, 1)
#: A login name shorter than this is not searched for: two or three letters turn up by chance in any file.
_MIN_IDENTITY_LENGTH = 4


class FigureError(ValueError):
    """A figure cannot be drawn or saved as asked."""


@dataclass(frozen=True, slots=True)
class Series:
    """How one condition, or one group of trials, is drawn wherever it appears.

    The colour is repeated by a marker shape and a line style, so that a figure printed in grey, or read by someone
    who does not see the colours apart, says the same thing.
    """

    #: The name printed in a figure.
    label: str
    color: str
    #: A light version of the colour, for areas.
    tint: str
    #: A Matplotlib marker.
    marker: str
    #: The line's dashes as lengths of ink and gap in points; None for a solid line.
    dashes: tuple[float, float] | None = None


#: A square, a diamond or a triangle of the same nominal size as a disc does not look the same size. A marker's size
#: is multiplied by its factor here, so that marks of different shapes weigh the same on the page.
_MARKER_FACTOR = {"o": 1.0, "s": 0.88, "D": 0.8, "^": 1.12, "v": 1.12}


def marker_size(marker: str, size_pt: float) -> float:
    """The size to give Matplotlib for a marker that should look as large as a disc of `size_pt` across."""
    return size_pt * _MARKER_FACTOR.get(marker, 1.0)


#: The clustered layout and the policy trained on it.
CLUSTERED = Series("clustered", BLUE, BLUE_TINT, "o")
#: The spread layout and the policy trained on it.
SPREAD = Series("spread", ORANGE, ORANGE_TINT, "D", (4.0, 1.6))
#: The replay of a demonstration: no policy, so no colour.
REPLAY = Series("replay", INK, SHADE, "s", (3.0, 1.5))


@dataclass(frozen=True, slots=True)
class Forbidden:
    """A pattern that must not appear in a figure, and what an error may call it.

    `shown` never repeats an entry of a private list: those are the authors' own strings.
    """

    pattern: re.Pattern[str]
    shown: str


#: How many `settings()` blocks are open, one inside the other.
_open_blocks = 0


@contextlib.contextmanager
def settings() -> Iterator[None]:
    """Matplotlib's settings as these figures need them, for the length of a `with` block.

    The settings in force before are put back afterwards, so nothing outside the block is changed. A block inside
    another one finds the settings in force already and leaves them alone: a figure opens one block for the whole
    drawing, and every line of text it writes would otherwise set them again.
    """
    global _open_blocks
    if _open_blocks:
        _open_blocks += 1
        try:
            yield
        finally:
            _open_blocks -= 1
        return
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(_SETTINGS)
        _open_blocks = 1
        try:
            yield
        finally:
            _open_blocks = 0


def font(size_pt: float, *, bold: bool = False, italic: bool = False) -> FontProperties:
    """The figures' typeface at a size, from the font files that ship with Matplotlib."""
    if size_pt < MIN_TEXT_PT:
        raise FigureError(f"text of {size_pt:g} pt is below the {MIN_TEXT_PT:g} pt a figure may use")
    path = _FONT_FOLDER / _FONT_FILES[(bold, italic)]
    return FontProperties(fname=str(path), size=size_pt, math_fontfamily="stix")


def text_width_mm(text: str, size_pt: float = TEXT_PT, *, bold: bool = False) -> float:
    """How wide a piece of text is when drawn at a size, in millimetres."""
    if not text:
        return 0.0
    with settings():
        extents = TextPath((0.0, 0.0), text, prop=font(size_pt, bold=bold)).get_extents()
    return float(extents.x1) * MM_PER_PT


def centred_baseline_mm(y_mm: float, size_pt: float = TEXT_PT) -> float:
    """The baseline that puts a line of text's capitals and figures at mid-height on `y_mm`.

    Matplotlib's own centring looks at the letters of each text, so "policy" and "success" would sit at different
    heights; this does not.
    """
    return y_mm - 0.33 * size_pt * MM_PER_PT


class Page:
    """One figure of the paper: a page of the stated size, drawn on in millimetres from its bottom-left corner.

    `axes` is the single pair of axes: invisible, covering the page, with data units of one millimetre. Anything
    Matplotlib can draw can be added to it at page coordinates. Sizes of markers, lines and text stay in points.
    """

    def __init__(self, height_mm: float, *, width_mm: float = COLUMN_MM) -> None:
        if not (width_mm > 0.0 and height_mm > 0.0):
            raise FigureError(f"a page has a positive width and height, got {width_mm!r} x {height_mm!r} mm")
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        #: The words stamped across the page ("DRAFT", "PLANNED"), in the order stamped.
        self.stamps: list[str] = []
        #: Sentences for whoever runs the figures: what the drawing found that does not add up, or had to leave out.
        self.notes: list[str] = []
        with settings():
            self.figure = Figure(figsize=(self.width_mm / MM_PER_INCH, self.height_mm / MM_PER_INCH), facecolor="white")
            FigureCanvasAgg(self.figure)
            self.axes: Axes = self.figure.add_axes((0.0, 0.0, 1.0, 1.0))
            self.axes.set_xlim(0.0, self.width_mm)
            self.axes.set_ylim(0.0, self.height_mm)
            self.axes.set_axis_off()

    def text(
        self,
        x_mm: float,
        y_mm: float,
        text: str,
        *,
        size_pt: float = TEXT_PT,
        ha: str = "left",
        va: str = "baseline",
        color: str = INK,
        bold: bool = False,
        italic: bool = False,
        rotation: float = 0.0,
        zorder: float = 6.0,
        alpha: float | None = None,
        gid: str | None = None,
    ) -> Text:
        """Write a line of text at a place on the page. A size below `MIN_TEXT_PT` is refused."""
        with settings():
            return self.axes.text(
                x_mm,
                y_mm,
                text,
                fontproperties=font(size_pt, bold=bold, italic=italic),
                ha=ha,
                va=va,
                color=color,
                rotation=rotation,
                rotation_mode="anchor",
                zorder=zorder,
                alpha=alpha,
                gid=gid,
                clip_on=False,
            )

    def stamp(self, word: str) -> Text:
        """Write a word across the whole page, large and grey, over everything: "DRAFT" on a figure that is not a
        result, "PLANNED" on one that shows what is meant to be recorded and not what was.
        """
        size_pt = 0.62 * self.width_mm / max(text_width_mm(word, 10.0, bold=True), 1e-9) * 10.0
        size_pt = max(MIN_TEXT_PT, min(size_pt, 0.5 * self.height_mm / MM_PER_PT))
        self.stamps.append(word)
        return self.text(
            self.width_mm / 2.0,
            self.height_mm / 2.0,
            word,
            size_pt=size_pt,
            ha="center",
            va="center",
            color=INK,
            bold=True,
            rotation=18.0,
            zorder=50.0,
            alpha=0.13,
            gid=f"stamp:{word}",
        )

    def texts(self) -> list[Text]:
        """Every piece of text drawn on the page."""
        return [text for text in (*self.axes.texts, *self.figure.texts) if text.get_visible() and text.get_text()]


def pdf_bytes(page: Page, *, compression: int = 6) -> bytes:
    """The page as a PDF: vector drawing, fonts embedded as TrueType, an empty information dictionary."""
    buffer = io.BytesIO()
    with settings():
        matplotlib.rcParams["pdf.compression"] = compression
        page.figure.savefig(buffer, format="pdf", facecolor="white", metadata=_NO_METADATA)
    return buffer.getvalue()


def save_pdf(page: Page, path: str | Path, *, forbidden: Sequence[Forbidden] = ()) -> None:
    """Save the page as a PDF, making its folder if need be, after checking that it tells nothing about its makers.

    Refused, with nothing written: a text smaller than `MIN_TEXT_PT`; a text that matches a forbidden pattern, or
    names this machine's user or home directory; a file whose bytes do.

    The bytes of a compressed file can match a pattern by accident (two bytes that happen to read as the start of a
    home path, say, about once in a million bytes). The check of the paper's folder would then fail on a file that
    holds no such text. So when the bytes match, the file is written again with another strength of compression,
    which changes every compressed byte and no plain one. A match that survives every strength is in the file's plain
    text, and is refused.
    """
    path = Path(path)
    problems = _text_problems(page, forbidden)
    if problems:
        raise FigureError(f"{path.name} was not written: " + "; ".join(problems))
    found: list[str] = []
    for level in _COMPRESSION_LEVELS:
        data = pdf_bytes(page, compression=level)
        found = _matches(data.decode("latin-1"), forbidden) + (["a creation date"] if b"CreationDate" in data else [])
        if not found:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return
    raise FigureError(f"{path.name} was not written: its bytes hold " + ", ".join(sorted(set(found))))


def save_png(page: Page, path: str | Path, *, dpi: int = 300) -> None:
    """Save the page as a picture, for looking at: 300 dots per inch is the printed page under a magnifying glass.

    The paper takes the PDF. The picture carries no note of the drawing library either.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with settings():
        page.figure.savefig(path, format="png", dpi=dpi, facecolor="white", metadata={"Software": None})


def _text_problems(page: Page, forbidden: Sequence[Forbidden]) -> list[str]:
    """What is wrong with the page's texts: too small, or saying what a figure must not say."""
    problems = []
    for text in page.texts():
        words = text.get_text()
        if text.get_fontsize() < MIN_TEXT_PT - 1e-9:
            problems.append(f"the text {words!r} is {text.get_fontsize():g} pt, below {MIN_TEXT_PT:g} pt")
        problems += [f"a text holds {what}" for what in _matches(words, forbidden)]
    return problems


def _matches(text: str, forbidden: Sequence[Forbidden]) -> list[str]:
    """What a text holds that it must not: the forbidden patterns that match a line of it, and this machine's user
    name and home directory. Each is named without repeating what was found.
    """
    found = []
    lines = text.split("\n")
    for entry in forbidden:
        if any(entry.pattern.search(line) for line in lines):
            found.append(entry.shown)
    found += [name for name, value in _identity() if len(value) >= _MIN_IDENTITY_LENGTH and value in text]
    return found


def _identity() -> list[tuple[str, str]]:
    """This machine's user name and home directory, read when asked and never stored."""
    found = []
    try:
        found.append(("the name of this machine's user", getpass.getuser()))
    except (OSError, KeyError):  # no user database entry: nothing to search for
        pass
    try:
        found.append(("the home directory of this machine's user", str(Path.home())))
    except RuntimeError:
        pass
    return found
