#!/bin/sh
"true" '''\'
exec "${TOLRAD_RIG_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}" "$0" "$@"
'''
# The four lines above start this file with the rig's interpreter from any shell, without an absolute path (which
# would carry a user name) in the first line: to `sh` they are an `exec`, to Python a string. Another interpreter can
# be named in TOLRAD_RIG_PYTHON.
__doc__ = """Read a CSV of logged cube positions and report how the instrument compares with a physical scale.

    scripts/check_instrument.py scale data/instrument/ruler_x_<time>.csv --step 10
    scripts/check_instrument.py repeat data/instrument/repeat_<time>.csv

`scale`: the readings were taken at equal known steps along a ruler; prints each step's error and the pass line of the
protocol (mean step error at most 1.5 mm, no step off by more than 3 mm).
`repeat`: the readings were all meant to be the same place; prints their scatter.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tolerance_radius.instrument import check_against_scale, scatter_mm  # noqa: E402
from tolerance_radius.schema import Point  # noqa: E402

MEAN_LIMIT_MM, MAX_LIMIT_MM = 1.5, 3.0
#: A stretch or shrink of distances larger than this is reported as needing a correction.
SCALE_NOTE_PERCENT = 1.0


def read_points(path: str) -> list[Point]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [Point(float(row["x_mm"]), float(row["y_mm"])) for row in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    scale = modes.add_parser("scale")
    scale.add_argument("csv")
    scale.add_argument("--step", type=float, required=True, help="the true distance between readings, in mm")
    repeat = modes.add_parser("repeat")
    repeat.add_argument("csv")
    args = parser.parse_args()

    points = read_points(args.csv)
    if args.mode == "repeat":
        rms, largest = scatter_mm(points)
        print(f"{len(points)} readings of one place: scatter {rms:.2f} mm (rms), {largest:.2f} mm at worst")
        return 0

    check = check_against_scale(points, args.step)
    print(f"{len(points)} readings, true step {args.step:g} mm")
    for number, (measured, error) in enumerate(zip(check.measured_steps_mm, check.step_errors_mm), start=1):
        print(f"  step {number}: measured {measured:6.2f} mm   error {error:+.2f} mm")
    print(f"  span: measured {check.measured_span_mm:.2f} mm against {check.true_span_mm:.2f} mm "
          f"({check.span_error_mm:+.2f} mm, {check.scale_error_percent:+.1f} %)")
    print(f"  largest distance of a reading from the straight line: {check.max_off_line_mm:.2f} mm")
    print(f"  mean step error {check.mean_abs_step_error_mm:.2f} mm (limit {MEAN_LIMIT_MM}), "
          f"worst step {check.max_abs_step_error_mm:.2f} mm (limit {MAX_LIMIT_MM})")
    if abs(check.scale_error_percent) > SCALE_NOTE_PERCENT:
        print(f"  NOTE: distances along this run read {check.scale_error_percent:+.1f} % against the ruler. Expected in "
              f"part (the camera sees the cube's top, not its footprint). If the other runs agree, distances are "
              f"corrected by this factor (protocol/protocol.md, section 7).")
    passed = check.passes(MEAN_LIMIT_MM, MAX_LIMIT_MM)
    print("  PASS" if passed else "  FAIL: find the cause before any trial (protocol/protocol.md, section 7)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
