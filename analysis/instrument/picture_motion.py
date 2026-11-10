#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
__doc__ = """Has the camera moved, or a floor tag? Two pictures of the table held against each other.

The two floor tags are all a calibration sees of the table. When they have not moved alike in the picture since the
table was registered, a camera that turned and a tag that was moved on the floor look the same to them. The rest of
the picture can tell: the pipes, the arm and the basket are not on the floor sheet, and a camera that moved moves
all of them with the tags.

This cuts the picture into patches, keeps those that have texture both ways and lie clear of the tags and of where
the registration sheet lay, and measures how far each has moved between the pictures (a translation fitted to the
patch, `cv2.findTransformECC`). One motion of the whole picture is fitted to the patches: a shift, a turn about the
picture's centre, and a change of size (a camera that comes closer makes the picture larger; `--no-scale` leaves the
size out, as the first version of this script did). Each floor tag's own shift is then held against that motion.

    analysis/instrument/picture_motion.py                      # the registration's frames against the newest calibration's
    analysis/instrument/picture_motion.py --before A.png B.png --after C.png --tag-shift 0:-1.12,0.13 --tag-shift 1:-0.51,-0.66

`--before` defaults to the three frames the registration in place kept; `--after` to the newest frame a calibration
kept (`data/instrument/calibration_*.png`). The tags' shifts are measured from the pictures by their outlines (from
the registration's stored corners when `--before` is left out). Pass `--tag-shift` instead when a picture is a
marked one (a check's picture has drawings on the tags), and `--marked` to keep the patches clear of those drawings.

What it cannot say: the things it reads stand nearer the camera than the floor, so a camera that moved sideways
(and did not only turn) shifts them a little more than the floor. On 2026-10-06 the patches themselves lay 0.14 px
(rms) and up to 0.33 px from the one motion fitted to them; a tag that moved by less than that against the picture
is not seen here.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tolerance_radius.rig import registration  # noqa: E402
from tolerance_radius.rig.vision import tag_outlines_over  # noqa: E402

INSTRUMENT = REPO / "data" / "instrument"
GEOMETRY = REPO / "config" / "table_geometry.json"
#: Patches: their size and how far apart they start, in pixels.
PATCH, STEP = 48, 24
#: A patch is used when the smaller eigenvalue of its structure tensor (mean squared gradient along the direction it
#: has least of, in grey levels squared) is at least this: a bare pipe or a straight edge has texture one way only
#: and cannot say how far it moved along itself, and the white floor has none. Of about 300 patches clear of the tags
#: and the sheet in the rig's picture of 2026-10-06, 45 reach it and were read; half of all have none (0.0).
MIN_TEXTURE = 8.0
#: And when the fitted translation matches the patch at least this well (correlation), and is not larger than this.
MIN_CORRELATION, MAX_SHIFT_PX = 0.9, 6.0
#: Where the registration sheet lay, the floor has changed: the grid's outline and this much around it are left
#: out, in table millimetres: below x = 0, beyond the grid's width, below y = 0 (the arm's side), and beyond the
#: grid's depth (the camera's side, where the sheet's text is). The page is 279 x 216 mm around a grid of 250 x 150.
SHEET_MARGIN_MM = (25.0, 25.0, 30.0, 65.0)
#: Around each floor tag, as a share of its own size (its paper lies on the floor with it), and, in a marked
#: picture, the band its label is written in.
TAG_MARGIN, LABEL_BAND_PX = 1.8, (120.0, 30.0)


def grey(path: Path) -> np.ndarray:
    picture = cv2.imread(str(path))
    if picture is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY).astype(np.float32)


def to_px(px_to_mm, points_mm) -> np.ndarray:
    points = np.column_stack([np.asarray(points_mm, dtype=float), np.ones(len(points_mm))])
    homogeneous = points @ np.linalg.inv(np.asarray(px_to_mm, dtype=float)).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def left_out(shape, geometry: dict, marked: bool) -> np.ndarray:
    """A mask of the pixels no patch may touch: the sheet's place, the floor tags, and a marked picture's drawings."""
    mask = np.zeros(shape, np.uint8)
    stored = geometry["registration"]
    grid = geometry["grid"]
    width, depth = grid["cols"] * grid["cell_mm"], grid["rows"] * grid["cell_mm"]
    x_low, x_high, y_low, y_high = SHEET_MARGIN_MM
    sheet = [(-x_low, -y_low), (width + x_high, -y_low), (width + x_high, depth + y_high), (-x_low, depth + y_high)]
    cv2.fillConvexPoly(mask, np.round(to_px(stored["px_to_mm"], sheet)).astype(np.int32), 255)
    for corners in stored["floor_tag_corners_px"].values():
        quad = np.asarray(corners, dtype=float)
        centre = quad.mean(axis=0)
        cv2.fillConvexPoly(mask, np.round(centre + TAG_MARGIN * (quad - centre)).astype(np.int32), 255)
        if marked:
            (half_x, half_y), (cx, cy) = LABEL_BAND_PX, centre
            cv2.rectangle(mask, (int(cx - half_x), int(cy - half_y)), (int(cx + half_x), int(cy + half_y)), 255, -1)
    return mask


def texture(patch: np.ndarray) -> float:
    """The smaller eigenvalue of the patch's structure tensor."""
    gy, gx = np.gradient(cv2.GaussianBlur(patch, (0, 0), 1.0))
    xx, yy, xy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    return 0.5 * (xx + yy) - float(np.hypot(0.5 * (xx - yy), xy))


def patch_shifts(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """For every usable patch: its centre (x, y) and how far its content has moved (dx, dy), in pixels."""
    found = []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-9)
    height, width = before.shape
    for y0 in range(0, height - PATCH + 1, STEP):
        for x0 in range(0, width - PATCH + 1, STEP):
            box = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
            if mask[box].any() or texture(before[box]) < MIN_TEXTURE:
                continue
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                correlation, warp = cv2.findTransformECC(before[box], after[box], warp, cv2.MOTION_TRANSLATION,
                                                         criteria, None, 3)
            except cv2.error:
                continue
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            if correlation >= MIN_CORRELATION and np.hypot(dx, dy) <= MAX_SHIFT_PX:
                found.append((x0 + PATCH / 2, y0 + PATCH / 2, dx, dy))
    return found


def fit_motion(points, shifts, centre, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A shift, a turn about `centre` and a change of size fitted to patch shifts: (tx, ty, turn in radians, growth as
    a share) and what is left over.

    A point at (x, y) moves by (tx - turn * (y - cy) + growth * (x - cx), ty + turn * (x - cx) + growth * (y - cy)):
    a positive turn carries the picture's x towards its y, a positive growth makes the picture larger. With
    `scale=False` the growth is held at nought.
    """
    points, shifts = np.asarray(points, dtype=float), np.asarray(shifts, dtype=float)
    rows, values = [], []
    for (x, y), (dx, dy) in zip(points, shifts, strict=True):
        rows += [[1.0, 0.0, -(y - centre[1]), x - centre[0]], [0.0, 1.0, x - centre[0], y - centre[1]]]
        values += [dx, dy]
    design = np.asarray(rows)
    if not scale:
        design = design[:, :3]
    solution, *_ = np.linalg.lstsq(design, np.asarray(values), rcond=None)
    solution = np.append(solution, 0.0) if not scale else solution
    return solution, shifts - motion_at(solution, points, centre)


def motion_at(solution, points, centre) -> np.ndarray:
    tx, ty, turn, growth = solution
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y = points[:, 0] - centre[0], points[:, 1] - centre[1]
    return np.column_stack([tx - turn * y + growth * x, ty + turn * x + growth * y])


def robust_motion(patches, centre, scale: bool = True):
    """The fit with the patches that do not go with the rest left out (something that was moved by hand: the arm, the
    basket).

    It starts from the patches whose shift is near the middle one of all (within 0.6 px, or three times the middle
    distance from it): a turn of a tenth of a degree or a picture a tenth of a percent larger moves no patch further
    than that from the rest, and a basket pushed by two pixels is. Started from all the patches, a fit with a turn
    and a growth would bend itself onto the basket. Then twice: fit, and keep the patches within three times the
    middle distance from the fit, 0.3 px at least.
    """
    data = np.asarray(patches, dtype=float)
    middle = np.median(data[:, 2:], axis=0)
    distance = np.linalg.norm(data[:, 2:] - middle, axis=1)
    keep = distance <= max(0.6, 3.0 * float(np.median(distance)))
    for _ in range(2):
        solution, left = fit_motion(data[keep, :2], data[keep, 2:], centre, scale)
        distance = np.linalg.norm(data[:, 2:] - motion_at(solution, data[:, :2], centre), axis=1)
        keep = distance <= max(0.3, 3.0 * float(np.median(distance[keep])))
    solution, left = fit_motion(data[keep, :2], data[keep, 2:], centre, scale)
    return solution, left, keep


def stored_view_against(geometry: dict, view: dict, solution, centre) -> tuple[float, float, float, float]:
    """A stored calibration against the registered frame carried along with the picture's own motion, over the grid
    every 25 mm: the mean and the largest distance between where the two put the same point (mm), and the largest
    difference of a distance of up to 100 mm between two of those points, in millimetres and as a share.

    The calibration's frame is solved from the two floor tags' outlines; this one never looks at them after the
    registration, so it is a second opinion on where the registered frame now is in the picture. It is only as good
    as the one motion fitted to the patches.
    """
    grid = geometry["grid"]
    xs, ys = np.meshgrid(np.arange(0.0, grid["cols"] * grid["cell_mm"] + 1.0, 25.0),
                         np.arange(0.0, grid["rows"] * grid["cell_mm"] + 1.0, 25.0))
    table = np.column_stack([xs.ravel(), ys.ravel()])
    then_px = to_px(geometry["registration"]["px_to_mm"], table)
    now_px = then_px + motion_at(solution, then_px, centre)
    homogeneous = np.column_stack([now_px, np.ones(len(now_px))]) @ np.asarray(view["px_to_mm"], dtype=float).T
    read = homogeneous[:, :2] / homogeneous[:, 2:3]
    apart = np.linalg.norm(read - table, axis=1)
    worst_mm = worst_share = 0.0
    for i in range(len(table)):
        true = np.linalg.norm(table[i + 1:] - table[i], axis=1)
        seen = np.linalg.norm(read[i + 1:] - read[i], axis=1)
        near = true <= 100.0
        if near.any():
            off = np.abs(seen[near] - true[near])
            worst_mm, worst_share = max(worst_mm, float(off.max())), max(worst_share, float((off / true[near]).max()))
    return float(apart.mean()), float(apart.max()), worst_mm, worst_share


def tag_shift_argument(text: str) -> tuple[int, tuple[float, float]]:
    try:
        tag, pair = text.split(":")
        dx, dy = (float(value) for value in pair.split(","))
        return int(tag), (dx, dy)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a tag's shift is given as TAG:DX,DY in pixels, e.g. 0:-1.12,0.13: {text!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--before", nargs="+", type=Path, help="pictures from the registration (default: the frames "
                        "kept by the registration in place)")
    parser.add_argument("--after", nargs="+", type=Path, help="pictures taken now (default: the newest frame a "
                        "calibration kept)")
    parser.add_argument("--marked", action="store_true", help="a picture carries a check's drawings")
    parser.add_argument("--no-scale", action="store_true", help="fit a shift and a turn only, as the first version did")
    parser.add_argument("--tag-shift", action="append", type=tag_shift_argument, default=[],
                        help="a floor tag's shift as a calibration printed it, TAG:DX,DY in pixels (twice)")
    parser.add_argument("--geometry", type=Path, default=GEOMETRY)
    parser.add_argument("--stored-view", type=Path, default=INSTRUMENT / "table_view.json",
                        help="the stored calibration; it is held against the picture's motion when the picture after "
                        "is the frame that calibration kept")
    args = parser.parse_args(argv)

    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    stamp = geometry["created"][:19].replace("-", "").replace(":", "").replace("T", "_")
    before_paths = args.before or sorted(INSTRUMENT.glob(f"table_registration_{stamp}_raw*.png"))
    after_paths = args.after or sorted(INSTRUMENT.glob("calibration_*.png"))[-1:]
    if not before_paths or not after_paths:
        raise SystemExit("no pictures to compare: give --before and --after")
    before = np.mean([grey(path) for path in before_paths], axis=0).astype(np.float32)
    height, width = before.shape
    centre = (width / 2.0, height / 2.0)
    mask = left_out(before.shape, geometry, args.marked)
    stored = {int(tag): np.asarray(corners, dtype=float)
              for tag, corners in geometry["registration"]["floor_tag_corners_px"].items()}
    first, second = registration.FLOOR_TAG_IDS

    reference = stored                                    # where the tags were: at the registration, by default
    if args.before and not args.marked and not args.tag_shift:
        seen, _found, _measured, _count = tag_outlines_over([cv2.imread(str(path)) for path in before_paths], list(stored))
        reference = seen if sorted(seen) == sorted(stored) else {}
    print(f"the table as registered {geometry['created']}")
    print(f"before: {', '.join(path.name for path in before_paths)}")
    for path in after_paths:
        after = grey(path)
        patches = patch_shifts(before, after, mask)
        print(f"\nafter: {path.name}")
        if len(patches) < 6:
            print(f"  only {len(patches)} patches could be read: nothing can be said from this picture")
            continue
        solution, left, keep = robust_motion(patches, centre, not args.no_scale)
        used = np.asarray(patches)[keep]
        without = [fit_motion(np.delete(used[:, :2], i, 0), np.delete(used[:, 2:], i, 0), centre, not args.no_scale)[0]
                   for i in range(len(used))]
        turns, growths = [np.degrees(one[2]) for one in without], [100 * one[3] for one in without]
        print(f"  {len(patches)} patches with texture both ways, {int(keep.sum())} of them moving as one "
              f"({len(patches) - int(keep.sum())} left out); they reach from x = {used[:, 0].min():.0f} to "
              f"{used[:, 0].max():.0f} and y = {used[:, 1].min():.0f} to {used[:, 1].max():.0f}")
        if keep.sum() < 0.75 * len(patches):
            print("  MORE THAN A QUARTER OF THE PATCHES DO NOT MOVE WITH THE REST: something in the picture was moved "
                  "by hand (the arm, the basket). What follows rests on the larger group; look at the pictures.")
        grown = "" if args.no_scale else (f", and grown by {100 * solution[3]:+.3f} % ({min(growths):+.3f} to "
                                          f"{max(growths):+.3f})")
        print(f"  the picture has moved by ({solution[0]:+.2f}, {solution[1]:+.2f}) px at its centre, turned by "
              f"{np.degrees(solution[2]):+.3f} degrees (with any one patch left out: {min(turns):+.3f} to "
              f"{max(turns):+.3f}){grown}")
        print(f"  the patches lie {np.sqrt((left ** 2).sum(axis=1).mean()):.2f} px (rms) and up to "
              f"{np.linalg.norm(left, axis=1).max():.2f} px from that one motion")
        view = json.loads(args.stored_view.read_text(encoding="utf-8")) if args.stored_view.is_file() else {}
        from_the_registration = args.before is None        # only then is the motion the one since the registration
        if (from_the_registration and (view.get("record") or {}).get("picture") == path.name
                and view.get("geometry_created") == geometry["created"]):
            mean_mm, worst_mm, distance_mm, share = stored_view_against(geometry, view, solution, centre)
            print(f"  the calibration stored from this frame ({view['created']}) lies {mean_mm:.2f} mm on average and "
                  f"{worst_mm:.2f} mm at worst from the registered frame carried along with that motion, over the "
                  f"grid; a distance of up to 100 mm differs by {distance_mm:.2f} mm at most ({100 * share:.2f} %)")

        shifts = dict(args.tag_shift)
        if not shifts and not args.marked and reference:
            now, _found, _measured, _count = tag_outlines_over([cv2.imread(str(path))], list(stored))
            shifts = {tag: tuple((now[tag] - reference[tag]).mean(axis=0)) for tag in now}
        if sorted(shifts) != sorted(stored):
            print("  (the floor tags' own shifts are not known for this picture: pass --tag-shift twice)")
            continue
        over = {}
        for tag in (first, second):
            tag_centre = stored[tag].mean(axis=0)
            expected = motion_at(solution, [tag_centre], centre)[0]
            over[tag] = np.asarray(shifts[tag]) - expected
            print(f"  floor tag {tag} at ({tag_centre[0]:.0f}, {tag_centre[1]:.0f}): moved by ({shifts[tag][0]:+.2f}, "
                  f"{shifts[tag][1]:+.2f}) px; the rest of the picture moved by ({expected[0]:+.2f}, "
                  f"{expected[1]:+.2f}) px there; left over ({over[tag][0]:+.2f}, {over[tag][1]:+.2f}) px")
        line = stored[second].mean(axis=0) - stored[first].mean(axis=0)
        differ = np.asarray(shifts[second]) - np.asarray(shifts[first])
        pair_turn = np.degrees((line[0] * differ[1] - line[1] * differ[0]) / float(line @ line))
        pair_growth = 100 * float(line @ differ) / float(line @ line)
        unexplained = float(np.linalg.norm(over[second] - over[first]))
        px_per_mm = np.mean([np.linalg.norm(stored[tag][0] - stored[tag][1]) for tag in stored]) / geometry["tag_mm"]
        print(f"  the two tags as a pair turned by {pair_turn:+.3f} degrees and drew {pair_growth:+.3f} % apart; the "
              f"rest of the picture turned by {np.degrees(solution[2]):+.3f} degrees"
              + ("" if args.no_scale else f" and grew by {100 * solution[3]:+.3f} %"))
        print(f"  of the {np.linalg.norm(differ):.2f} px by which the two tags' shifts differ, the picture's own "
              f"motion leaves {unexplained:.2f} px unexplained (about {unexplained / px_per_mm:.1f} mm on the floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
