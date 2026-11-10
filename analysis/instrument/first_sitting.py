#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
__doc__ = """The table registration of the first sitting (2026-10-05), worked through from its two saved frames.

Writes `analysis/instrument/first_sitting.md`. Three things the registration's own figures do not show, each measured
here so that the numbers quoted in the code, the runbook and the protocol can be made again:

1. **The sheet's printed grid lines against the frame.** Every grid line is a thin dark line at a known place on the
   sheet, and no fit uses it. Its place in the picture is the centre of its dip in brightness on short scans across
   it; that is compared with where a frame puts it. (The method is the independent review's of that evening.)
2. **The frame from four of the sheet's six tags** against the frame from all six.
3. **What one pixel on one corner of a floor tag does** to the frame solved from the two floor tags.

    analysis/instrument/first_sitting.py

The same can be asked of any later registration, from the three frames it keeps:

    analysis/instrument/first_sitting.py --pictures data/instrument/table_registration_<time>_raw*.png \\
        --sheet-outline-mm 250.25x149.67 --floor-tags-mm 59.7x60.0,59.78x60.0 \\
        --out analysis/instrument/registration_<date>.md
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tolerance_radius.rig import lerobot101, outline, registration  # noqa: E402

FIRST_SITTING = sorted((REPO / "data" / "instrument").glob("view_raw_20261005_2209*.png"))
GRID = {"cols": 5, "rows": 3, "cell_mm": 50.0}
SCALE = registration.sheet_scale_from_outline(250.25, 149.67)       # the sheet's outline by calipers (first sitting)
HALF_PX, STEP_PX = 3.0, 0.25


def sizes(text: str, count: int) -> list[tuple[float, float]]:
    parts = [tuple(float(value) for value in part.lower().split("x")) for part in text.split(",")]
    if len(parts) != count or any(len(part) != 2 for part in parts):
        raise SystemExit(f"{count} size(s) are needed, each WIDTHxHEIGHT in millimetres: {text!r}")
    return parts


def apply(matrix, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=float).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def fit(tag_corners_px, ids):
    pixels = np.concatenate([tag_corners_px[tag_id] for tag_id in ids])
    table = np.concatenate([registration.sheet_corners_mm(tag_id, SCALE) for tag_id in ids])
    matrix, _ = cv2.findHomography(pixels, table, 0)
    return matrix


def line_segments():
    """The stretches of every grid line that are clear of crossings, labels, small print and the ruler's ticks:
    (axis, value, from, to) in design millimetres."""
    segments = []
    for y in (0.0, 50.0, 100.0, 150.0):
        spans = [(60, 90), (110, 140), (160, 170)] if y == 0.0 else [(a + 10, a + 40) for a in range(0, 250, 50)]
        segments += [("y", y, lo, hi) for lo, hi in spans]
    for x in (0.0, 50.0, 100.0, 150.0, 200.0, 250.0):
        spans = ([(10, 21), (29, 40), (60, 71), (79, 90), (110, 121), (129, 140)] if x == 250.0
                 else [(a + 10, a + 40) for a in range(0, 150, 50)])
        segments += [("x", x, lo, hi) for lo, hi in spans]
    return segments


def line_offsets(gray, px_to_mm):
    """For every millimetre along every clear stretch: how far the line is seen from where the frame puts it, in
    millimetres along the line's normal. Rows of (axis, value, place along the line, offset)."""
    mm_to_px = np.linalg.inv(px_to_mm)
    scan = np.arange(-HALF_PX, HALF_PX + 1e-9, STEP_PX)
    ends = max(2, int(round(0.75 / STEP_PX)))
    rows = []
    for axis, value, lo, hi in line_segments():
        along = np.linspace(lo, hi, int(hi - lo) + 1)
        design = (np.column_stack([along, np.full(len(along), value)]) if axis == "y"
                  else np.column_stack([np.full(len(along), value), along]))
        table = design * np.asarray(SCALE)
        pixels = apply(mm_to_px, table)
        normal = apply(mm_to_px, table + ((0.0, 1.0) if axis == "y" else (1.0, 0.0))) - pixels
        px_per_mm = np.linalg.norm(normal, axis=1)
        normal /= px_per_mm[:, None]
        points = pixels[:, None, :] + scan[None, :, None] * normal[:, None, :]
        profile = cv2.remap(gray, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32),
                            cv2.INTER_CUBIC)
        left, right = profile[:, :ends].mean(axis=1), profile[:, -ends:].mean(axis=1)
        base = left[:, None] + (right - left)[:, None] * (scan[None, :] - scan[:ends].mean()) / (
            scan[-ends:].mean() - scan[:ends].mean())
        dark = np.clip(base - profile, 0, None)
        area = dark.sum(axis=1) * STEP_PX
        centre = (dark * scan[None, :]).sum(axis=1) / np.maximum(dark.sum(axis=1), 1e-9)
        clear = area > 0.5 * np.median(area)
        rows += [(axis, value, a, c / p) for a, c, p, ok in zip(along, centre, px_per_mm, clear, strict=True) if ok]
    return rows


def lines_summary(rows):
    offsets = np.array([row[3] for row in rows])
    by_axis = {axis: np.array([row[3] for row in rows if row[0] == axis]) for axis in ("x", "y")}
    bins = {}
    for axis, value, along, offset in rows:
        bins.setdefault((axis, value, int(along // 10)), []).append(offset)
    worst = max(bins, key=lambda key: abs(np.mean(bins[key])))
    return {
        "scans": len(rows),
        "rms": float(np.sqrt(np.mean(offsets**2))),
        "rms_x": float(np.sqrt(np.mean(by_axis["x"] ** 2))),
        "rms_y": float(np.sqrt(np.mean(by_axis["y"] ** 2))),
        "worst": float(abs(np.mean(bins[worst]))),
        "worst_where": f"the line {worst[0]} = {worst[1]:g} mm, {10 * worst[2]} to {10 * worst[2] + 10} mm along it",
    }


def main() -> int:
    global SCALE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pictures", nargs="+", type=Path, default=None, help="unmarked frames of one registration")
    parser.add_argument("--sheet-outline-mm", default="250.25x149.67")
    parser.add_argument("--floor-tags-mm", default="59.7x60.0,59.78x60.0")
    parser.add_argument("--out", type=Path, default=None, help="the report to write (default: beside this script)")
    args = parser.parse_args()
    frames = args.pictures or FIRST_SITTING
    if not frames or (args.pictures is None and len(frames) != 2):
        raise SystemExit("the two saved frames of 2026-10-05 are needed: data/instrument/view_raw_20261005_2209*.png")
    SCALE = registration.sheet_scale_from_outline(*sizes(args.sheet_outline_mm, 1)[0])
    stated = dict(zip(registration.FLOOR_TAG_IDS, sizes(args.floor_tags_mm, 2), strict=True))
    detect = lerobot101.load("tag_check").detect
    pictures = [cv2.imread(str(path)) for path in frames]
    if any(picture is None for picture in pictures):
        raise SystemExit(f"cannot read one of {[str(path) for path in frames]}")
    grays = [cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) for picture in pictures]
    found = [detect(gray) for gray in grays]
    detector = registration.mean_corners(found)
    outlines = registration.mean_corners([outline.tag_outlines(gray, one) for gray, one in zip(grays, found)])
    ids = sorted(registration.SHEET_TAGS)
    mean_gray = np.mean([gray.astype(np.float32) for gray in grays], axis=0)

    frame = fit(outlines, ids)
    by_outline = lines_summary(line_offsets(mean_gray, frame))
    by_detector = lines_summary(line_offsets(mean_gray, fit(detector, ids)))

    four = {}
    for left_out in combinations(ids, 2):
        kept = [tag_id for tag_id in ids if tag_id not in left_out]
        four[left_out] = registration.frames_apart_mm(frame, fit(outlines, kept), GRID)
    worst_four = max(four, key=four.get)

    result = registration.register(found, [outline.tag_outlines(g, f) for g, f in zip(grays, found)], grid=GRID,
                                   sheet_scale=SCALE, floor_tags_mm=stated, created="-")
    floor = {int(tag_id): tag for tag_id, tag in result.geometry["tags"].items()}
    solve = lerobot101.load("placement").homography_from_tag_geometry
    pixels = {tag_id: np.asarray(outlines[tag_id], dtype=float) for tag_id in floor}
    one_pixel = {}
    for tag_id in pixels:
        for corner in range(4):
            for axis in (0, 1):
                moved = {key: value.copy() for key, value in pixels.items()}
                moved[tag_id][corner, axis] += 1.0
                one_pixel[(tag_id, corner, "xy"[axis])] = registration.frames_apart_mm(
                    result.px_to_mm, solve(moved, floor, 60.0), GRID)
    worst_pixel = max(one_pixel, key=one_pixel.get)

    names = ", ".join(f"`{path.name}`" for path in frames)
    title = ("The first sitting's registration, worked through from its two saved frames" if args.pictures is None
             else "A registration worked through from its saved frames")
    lines = [
        f"# {title}",
        "",
        f"Written by `analysis/instrument/first_sitting.py` from {names} (in `data/instrument/`). Every figure here is",
        "*measured* by that script.",
        "",
        "## The registration's own figures",
        "",
        f"- The sheet's tags fit one flat plane to {result.sheet_fit_mean_mm:.2f} mm on average, "
        f"{result.sheet_fit_max_mm:.2f} mm at worst.",
        f"- A sheet tag left out is predicted by the others to {result.leave_one_out_mean_mm:.2f} mm on average, "
        f"{result.leave_one_out_max_mm:.2f} mm at worst.",
        f"- The detector's corners sit {result.detector_inset_px:.2f} px inside the tags' outlines.",
        *[f"- NOTE: {note}." for note in result.notes],
        *[f"- **REFUSED:** {problem}." for problem in result.problems],
        "",
        "## 1. The sheet's printed grid lines, which no fit uses",
        "",
        "| Frame fitted through | Scans | Lines from the frame, rms (mm) | Lines of constant x | Lines of constant y | "
        "Largest mean over 10 mm of a line (mm) |",
        "|---|---|---|---|---|---|",
        f"| the tags' outlines (what the registration uses) | {by_outline['scans']} | {by_outline['rms']:.2f} | "
        f"{by_outline['rms_x']:.2f} | {by_outline['rms_y']:.2f} | {by_outline['worst']:.2f} |",
        f"| the detector's corners | {by_detector['scans']} | {by_detector['rms']:.2f} | {by_detector['rms_x']:.2f} | "
        f"{by_detector['rms_y']:.2f} | {by_detector['worst']:.2f} |",
        "",
        f"The largest is on {by_outline['worst_where']} (outline frame). Lines of constant y test the frame along the "
        "camera's line of sight, where a sheet that is not flat shows: a millimetre of height moves what the camera "
        "sees by most of a millimetre that way and hardly at all across.",
        "",
        "## 2. The frame from four of the six sheet tags",
        "",
        f"Leaving out two tags, the frame moves by up to {four[worst_four]:.2f} mm somewhere on the grid (without "
        f"tags {list(worst_four)}); over all 15 choices of two the median is {np.median(list(four.values())):.2f} mm.",
        "",
        "## 3. One pixel on one corner of a floor tag",
        "",
        f"Solved from the two floor tags' corners, the frame moves by up to "
        f"{one_pixel[worst_pixel]:.2f} mm somewhere on the grid when one corner is moved by one pixel (tag "
        f"{worst_pixel[0]}, corner {worst_pixel[1]}, along {worst_pixel[2]}); the median over the 16 coordinates is "
        f"{np.median(list(one_pixel.values())):.2f} mm.",
        "",
    ]
    out = args.out or Path(__file__).with_suffix(".md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.resolve().relative_to(REPO) if out.resolve().is_relative_to(REPO) else out)
    print("\n".join(lines[5:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
