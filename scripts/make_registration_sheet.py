#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell (it needs OpenCV and ReportLab),
# without an absolute path in the first line: to `sh` they are an `exec`, to Python a string.
__doc__ = """Make the table registration sheet: one Letter page to print at 100 %.

The sheet is seated once where the grid should be. It carries the grid's outline and six small tags at known places
inside it; `scripts/rig/measure.py register-table` measures the two floor tags against them
(`docs/runbook/00-table-layout.md`). The tags are drawn as vector squares, so no viewer or printer can blur them, and
the page carries two 100 mm rulers, across and up, because a printer that scales the page spoils every measurement
made through it.

    scripts/make_registration_sheet.py                     # writes docs/print/registration_sheet_letter.pdf
    scripts/make_registration_sheet.py --check PAGE.png    # a 300 dpi picture of the page: are the tags where designed?
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib.colors import HexColor, black  # noqa: E402
from reportlab.lib.pagesizes import landscape, letter  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfgen import canvas as pdfcanvas  # noqa: E402

from tolerance_radius.config import load_protocol  # noqa: E402
from tolerance_radius.rig import lerobot101, registration  # noqa: E402

OUT = REPO / "docs" / "print" / "registration_sheet_letter.pdf"
PAGE_W_MM, PAGE_H_MM = 279.4, 215.9
#: Where the grid's origin (the outer corner of cell A1) is on the page, from the page's lower-left corner.
ORIGIN_MM = ((PAGE_W_MM - 250.0) / 2.0, 22.0)
GREY = HexColor("#8a8a8a")


def modules(tag_id: int) -> np.ndarray:
    """A tag as its 8 x 8 squares, True where black, row 0 at the tag's top."""
    dictionary = cv2.aruco.getPredefinedDictionary(lerobot101.load("tag_check").DICT)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, 8) < 128


def draw_tag(c: pdfcanvas.Canvas, tag_id: int, centre_mm: tuple[float, float], side_mm: float) -> None:
    """One tag, upright with the page, as filled squares."""
    cells = modules(tag_id)
    size = side_mm / cells.shape[0]
    left, top = centre_mm[0] - side_mm / 2.0, centre_mm[1] + side_mm / 2.0
    c.setFillColor(black)
    for row in range(cells.shape[0]):
        for col in range(cells.shape[1]):
            if cells[row, col]:
                # a hair of overlap, so that no viewer shows a seam between two black squares
                c.rect((left + col * size) * mm - 0.05, (top - (row + 1) * size) * mm - 0.05, size * mm + 0.1,
                       size * mm + 0.1, stroke=0, fill=1)


def draw_ruler(c: pdfcanvas.Canvas, start_mm: tuple[float, float], *, upright: bool) -> None:
    """A 100 mm ruler with a tick every 10 mm."""
    x0, y0 = start_mm
    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    for i in range(0, 101, 10):
        tick = 3.5 if i % 50 == 0 else 2.0
        if upright:
            c.line(x0 * mm, (y0 + i) * mm, (x0 - tick) * mm, (y0 + i) * mm)
        else:
            c.line((x0 + i) * mm, y0 * mm, (x0 + i) * mm, (y0 + tick) * mm)
    if upright:
        c.line(x0 * mm, y0 * mm, x0 * mm, (y0 + 100) * mm)
    else:
        c.line(x0 * mm, y0 * mm, (x0 + 100) * mm, y0 * mm)


def build(out: Path) -> Path:
    protocol = load_protocol()
    cols, rows, cell = protocol["table"]["cols"], protocol["table"]["rows"], protocol["table"]["cell_mm"]
    if (cols, rows, cell) != (5, 3, 50.0):
        raise SystemExit("the sheet is laid out for the 5 x 3 grid of 50 mm cells; the protocol says otherwise")
    out.parent.mkdir(parents=True, exist_ok=True)
    c = pdfcanvas.Canvas(str(out), pagesize=landscape(letter), invariant=1)
    c.setTitle("Table registration sheet")
    c.setAuthor("")
    ox, oy = ORIGIN_MM

    # the grid: outline, and lighter lines between the cells
    c.setStrokeColor(GREY)
    c.setLineWidth(0.35)
    for col in range(1, cols):
        c.line((ox + col * cell) * mm, oy * mm, (ox + col * cell) * mm, (oy + rows * cell) * mm)
    for row in range(1, rows):
        c.line(ox * mm, (oy + row * cell) * mm, (ox + cols * cell) * mm, (oy + row * cell) * mm)
    c.setStrokeColor(black)
    c.setLineWidth(0.6)
    c.rect(ox * mm, oy * mm, cols * cell * mm, rows * cell * mm, stroke=1, fill=0)

    # cell names, in the cells that carry no tag
    tagged = {(int(x // cell), int(y // cell)) for x, y in registration.SHEET_TAGS.values()}
    c.setFillColor(GREY)
    c.setFont("Helvetica", 11)
    for col in range(cols):
        for row in range(rows):
            if (col, row) not in tagged:
                c.drawCentredString((ox + (col + 0.5) * cell) * mm, (oy + (row + 0.5) * cell) * mm - 4,
                                    f"{'ABCDE'[col]}{row + 1}")

    for tag_id, (x, y) in registration.SHEET_TAGS.items():
        draw_tag(c, tag_id, (ox + x, oy + y), registration.SHEET_TAG_MM)
        c.setFillColor(GREY)
        c.setFont("Helvetica", 5)
        c.drawString((ox + x - 24) * mm, (oy + y - 24) * mm, f"id {tag_id}")

    # top band: what this is and how to print it; the ruler across
    top = oy + rows * cell
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(ox * mm, (top + 30) * mm, "TABLE REGISTRATION SHEET")
    c.setFont("Helvetica", 8)
    c.drawString(ox * mm, (top + 25) * mm,
                 "Print at 100 % / actual size: no 'fit to page'. Check BOTH rulers with calipers: each must measure "
                 "100.0 mm.")
    c.drawString(ox * mm, (top + 20.5) * mm,
                 "Seat it flat where the grid should be, tape its corners, run the registration, then take it away. "
                 "Nothing is marked on the floor.")
    draw_ruler(c, (ox, top + 9), upright=False)
    c.setFont("Helvetica", 6)
    c.drawString(ox * mm, (top + 5.5) * mm, "0")
    c.drawRightString((ox + 100) * mm, (top + 5.5) * mm, "100 mm")
    draw_ruler(c, (PAGE_W_MM - 7.5, oy + 25), upright=True)
    c.saveState()
    c.translate((PAGE_W_MM - 9.5) * mm, (oy + 128) * mm)
    c.rotate(90)
    c.drawString(0, 0, "100 mm")
    c.restoreState()

    # bottom band: which way round
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (oy - 9) * mm,
                        "v   THIS EDGE TOWARD THE ARM (row 1)   v      column A on your LEFT as you look into the box")
    c.setFont("Helvetica", 6)
    c.setFillColor(GREY)
    c.drawString(ox * mm, (oy - 3.2) * mm, "origin (0, 0): the outer corner of A1")
    c.drawRightString((ox + cols * cell) * mm, (oy - 3.2) * mm,
                      f"tags: 36h11, {registration.SHEET_TAG_MM:g} mm, ids 10 to 15, "
                      "at the centres of A1 C1 E1 A3 C3 E3")
    c.showPage()
    c.save()
    return out


def check(picture_path: Path, dpi: float = 300.0) -> int:
    """Are the tags on a rendered page where they were designed to be? Prints the largest error in millimetres."""
    page = cv2.imread(str(picture_path), cv2.IMREAD_GRAYSCALE)
    if page is None:
        raise SystemExit(f"cannot read {picture_path}")
    found = lerobot101.load("tag_check").detect(page)
    per_px = 25.4 / dpi
    worst = 0.0
    for tag_id in sorted(registration.SHEET_TAGS):
        if tag_id not in found:
            print(f"  tag {tag_id}: NOT FOUND")
            return 1
        # page millimetres from the lower-left corner, then into the grid's frame
        mm_xy = np.column_stack([found[tag_id][:, 0] * per_px, (page.shape[0] - found[tag_id][:, 1]) * per_px])
        error = float(np.abs(mm_xy - ORIGIN_MM - registration.sheet_corners_mm(tag_id)).max())
        worst = max(worst, error)
        print(f"  tag {tag_id}: corners within {error:.3f} mm of the design")
    print(f"  largest error {worst:.3f} mm  ({'PASS' if worst <= 0.2 else 'FAIL'}: limit 0.2 mm)")
    return 0 if worst <= 0.2 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", type=Path, default=None, help="a 300 dpi picture of the printed or rendered page")
    args = parser.parse_args()
    if args.check:
        return check(args.check)
    out = build(args.out)
    print(out.relative_to(REPO) if out.is_relative_to(REPO) else out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
