#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
__doc__ = """How bright and how warm the floor is in kept frames, place by place: what a window, a flap or the room's light does.

    analysis/instrument/picture_levels.py data/instrument/view_raw_A.png data/instrument/view_raw_B.png ...

For every frame: the grey level (0 to 255, the median) and red over blue of the bare floor in five places, each laid
out in table millimetres through the floor tags as that frame shows them, so that a camera that moved between the
frames reads the same floor: the grid's three thirds from column A's side to column E's, a strip in front of the
grid on the camera's side, and a strip beyond column E. Then every frame against the first, place by place.

The cube, the arm or a ruler lying in a place moves its median only when it covers most of it; a frame taken with
something on the floor is not for this.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tolerance_radius.rig.vision import TableView, load_geometry  # noqa: E402
from tolerance_radius.schema import Point  # noqa: E402

#: The places, as rectangles in table millimetres (x from, x to, y from, y to). The grid is 250 x 150 mm; the camera
#: stands beyond y = 150, column E is at x = 200 to 250.
PLACES = {
    "grid, A's third": (5.0, 83.0, 5.0, 145.0),
    "grid, middle third": (83.0, 167.0, 5.0, 145.0),
    "grid, E's third": (167.0, 245.0, 5.0, 145.0),
    "in front of the grid": (20.0, 230.0, 190.0, 260.0),
    "beyond column E": (265.0, 285.0, 70.0, 150.0),
}


def levels(frame: np.ndarray, view: TableView, place: tuple[float, float, float, float]) -> tuple[float, float]:
    """The median grey level and the median of red over blue of one place in one frame."""
    x0, x1, y0, y1 = place
    corners = [view.to_px(Point(x, y)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.asarray(corners)).astype(np.int32), 255)
    pixels = frame[mask > 0].astype(float)
    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return float(np.median(0.114 * blue + 0.587 * green + 0.299 * red)), float(np.median(red / np.maximum(blue, 1.0)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("frames", nargs="+", type=Path, help="frames as they came (view_raw, session, calibration)")
    args = parser.parse_args(argv)
    geometry = load_geometry()
    table = {}
    for path in args.frames:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"cannot read {path}")
        view = TableView.from_frames([frame], geometry)
        table[path.name] = {name: levels(frame, view, place) for name, place in PLACES.items()}
    names = list(table)
    width = max(len(name) for name in PLACES) + 2
    print("grey level and red over blue of the bare floor, place by place\n")
    for name in names:
        print(f"  {name}")
        for place, (grey, warmth) in table[name].items():
            print(f"    {place:{width}s} grey {grey:6.1f}   red over blue {warmth:.3f}")
    if len(names) > 1:
        print(f"\nagainst the first frame ({names[0]}): grey levels, and red over blue in thousandths\n")
        print(f"    {'':{width}s}" + "".join(f"{name[-10:-4]:>18s}" for name in names[1:]))
        for place in PLACES:
            row = f"    {place:{width}s}"
            for name in names[1:]:
                grey, warmth = table[name][place]
                first_grey, first_warmth = table[names[0]][place]
                row += f"{grey - first_grey:+9.1f} {1000 * (warmth - first_warmth):+5.0f}   "
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
