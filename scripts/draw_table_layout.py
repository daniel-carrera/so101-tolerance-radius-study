#!/usr/bin/env python3
"""Draw the planned layout of the light-box floor, seen from above: arm, grid, basket and the two tags.

A drawing to place things by, not a measurement: the positions that count are measured afterwards, by registering the
table (`docs/runbook/00-table-layout.md`). Distances are in millimetres from two things a ruler can reach: the inner
edge of the front pipe (depth into the box) and the arm's centre line (left and right, as seen from the open side).

    python scripts/draw_table_layout.py            # writes docs/images/table_layout.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

REPO = Path(__file__).resolve().parents[1]

# The light box of the VLA-REPLICA paper: a 32 inch cube. The arm's centre is 16.5 inches from the left face.
BOX_MM = 813.0
ARM_FROM_LEFT_MM = 419.0
# (lateral from the arm's centre line, depth from the front pipe): lower-left corner, width, depth
GRID = (-125.0, 200.0, 250.0, 150.0)
BASKET_RIM = (-400.0, 15.0, 192.0, 252.0)     # short side across, long side into the box, in the front-left corner
TAG_MM = 60.0
#: The tag's paper as cut on its dashed line: the black square and 15 mm of white on every side.
PAPER_MM = 90.0
TAGS = {"tag 0": (215.0, 130.0), "tag 1": (-215.0, 430.0)}   # centres
HELD_OUT = ("A1", "B2", "D2", "E3")
RED = "#c0392b"


def draw(out: Path) -> Path:
    figure = Figure(figsize=(8.6, 8.0), dpi=130)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.08, 0.07, 0.88, 0.86))
    left, right = -ARM_FROM_LEFT_MM, BOX_MM - ARM_FROM_LEFT_MM
    ax.add_patch(Rectangle((left, 0), BOX_MM, BOX_MM, fill=False, lw=2.0, ec="0.35"))
    ax.text(left + 8, BOX_MM - 28, "rear wall (the camera is above this edge, looking toward the arm)", fontsize=8,
            color="0.35")
    ax.text(left + 8, -30, "open side: you stand here", fontsize=8, color="0.35")

    # the arm's base and its centre line
    ax.add_patch(Rectangle((-55, -20), 110, 130, fc="#f28c28", ec="0.2", alpha=0.85))
    ax.text(0, 45, "arm\nbase", ha="center", va="center", fontsize=8)
    ax.plot([0, 0], [-40, BOX_MM], ls=":", lw=0.8, color="0.5")

    # the grid, row 1 nearest the arm, column A on the left
    gx, gy, gw, gd = GRID
    cell = gw / 5
    for col in range(5):
        for row in range(3):
            label = f"{'ABCDE'[col]}{row + 1}"
            held = label in HELD_OUT
            ax.add_patch(Rectangle((gx + col * cell, gy + row * cell), cell, cell, fc="0.88" if held else "none",
                                   ec="0.25", lw=0.8, hatch="///" if held else None))
            ax.text(gx + (col + 0.5) * cell, gy + (row + 0.5) * cell, label, ha="center", va="center", fontsize=7)
    ax.text(gx + gw / 2 + 90, gy + gd + 10, "grid (where the sheet is seated): row 1 nearest the arm,\n"
            "column A on your left", ha="center", fontsize=8)

    # the basket
    bx, by, bw, bd = BASKET_RIM
    ax.add_patch(Rectangle((bx, by), bw, bd, fc="0.45", ec="0.1", alpha=0.75))
    ax.text(bx + bw / 2, by + bd / 2, "basket\n(rim 192 x 252,\n105 tall)", ha="center", va="center", fontsize=8,
            color="white")

    # the tags: each drawn as its paper, with the two measurements that place it
    half = PAPER_MM / 2
    for name, (cx, cy) in TAGS.items():
        ax.add_patch(Rectangle((cx - half, cy - half), PAPER_MM, PAPER_MM, fc="white", ec="0.45", lw=0.9))
        ax.add_patch(Rectangle((cx - TAG_MM / 2, cy - TAG_MM / 2), TAG_MM, TAG_MM, fc="0.1", ec="0.1"))
        ax.text(cx, cy, name, ha="center", va="center", fontsize=8, color="white")
        near = cy - half                                    # the paper's edge nearest you
        inner = cx - half if cx > 0 else cx + half          # the paper's edge nearest the arm's line
        # from the front pipe to the paper's near edge, drawn just to the right of the paper (clear of the basket)
        x_arrow = cx + half + 16
        ax.annotate("", (x_arrow, 0), (x_arrow, near), arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": RED})
        ax.plot([cx + half, x_arrow], [near, near], lw=0.6, color=RED, ls=":")
        ax.text(x_arrow + 6, near * 0.55, f"{near:.0f}", fontsize=9, color=RED, va="center", ha="left",
                fontweight="bold")
        # from the arm's line to the paper's inner edge, drawn just beyond the paper's far edge
        y_arrow = cy + half + 16
        ax.annotate("", (0, y_arrow), (inner, y_arrow), arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": RED})
        ax.plot([inner, inner], [cy + half, y_arrow], lw=0.6, color=RED, ls=":")
        # the number above the arrow for the far tag, below it for the near one (the grid is just above that one)
        ax.text(inner / 2, y_arrow + 7 if cy > 300 else y_arrow - 20, f"{abs(inner):.0f}", fontsize=9, color=RED,
                ha="center", fontweight="bold")

    # where the basket hides the floor from the camera (roughly): the strip in front of it
    shadow = Polygon([(bx, by), (bx + bw, by), (bx + bw + 10, by - 15), (bx - 10, by - 15)], closed=True, fc="0.8",
                     ec="none", alpha=0.6)
    ax.add_patch(shadow)

    ax.set_xlim(left - 30, right + 30)
    ax.set_ylim(-60, BOX_MM + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("left / right of the arm's centre line (mm)")
    ax.set_ylabel("depth into the box from the inner edge of the front pipe (mm)")
    ax.set_title("Planned floor layout, seen from above. Red: each tag's paper (90 mm, cut on the dashed line), "
                 "by its edge\nnearest you (from the front pipe) and its edge nearest the arm's centre line. "
                 "Within 2 cm is good enough.", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, metadata={"Software": None})
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "images" / "table_layout.png"
    print(draw(target).relative_to(REPO) if target.is_relative_to(REPO) else target)
