#!/usr/bin/env python3
"""Where the 55 August demonstrations started, within their cells: a common offset, and the scatter at one "place".

The August dataset was recorded with a grid drawn over the live picture and no measurement: five demonstrations in
each of 11 cells, each "at the cell's centre". This script measures the cube in the first frame of every episode and
asks two things the earlier project never asked:

1. How far is the measured position from the cell's centre, on average and in which direction? A common offset is
   expected: the detector reports the centre of the cube's visible outline, not of its footprint
   (`protocol/protocol.md`, section 7). It cancels in every distance between two measured positions.
2. With that offset removed, how far apart are the five demonstrations of one cell?

Two conversions from pixels to table millimetres are reported, as in the rest of this folder: August's own (each
episode's picture through its own tag solve, read from the earlier project's `eval/placements.json`) and one conversion
solved from all 55 first frames together (this project's method, which removes the frame-to-frame wander of the solve).

Run with the rig's interpreter from the repository's top folder (it reads the dataset on this machine through the
earlier project's loader, and needs OpenCV):

    "$HOME/miniconda3/envs/lerobot/bin/python" analysis/pilot_august/demo_scatter.py

It rewrites `demo_starts.csv` in this folder and prints the summary the README quotes. It reads files only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tolerance_radius import config, geometry  # noqa: E402
from tolerance_radius.rig import audit, lerobot101  # noqa: E402
from tolerance_radius.rig.vision import TableView, load_geometry  # noqa: E402

OUT = HERE / "demo_starts.csv"


def scatter(points_by_cell: dict[str, list[np.ndarray]]) -> tuple[dict[str, tuple[float, float, float]], float, float]:
    """Per cell: (mean, largest) distance of a start from the cell's mean start, and the two farthest apart.

    Also the mean and the largest distance from the cell's mean start over all cells.
    """
    per_cell, every = {}, []
    for cell, points in sorted(points_by_cell.items()):
        stacked = np.array(points)
        from_mean = np.linalg.norm(stacked - stacked.mean(axis=0), axis=1)
        farthest = max(float(np.linalg.norm(a - b)) for a in stacked for b in stacked)
        per_cell[cell] = (float(from_mean.mean()), float(from_mean.max()), farthest)
        every.extend(from_mean.tolist())
    return per_cell, float(np.mean(every)), float(np.max(every))


def main() -> int:
    lerobot101.use_pinned_lerobot()
    record = json.loads((lerobot101.root() / "eval" / "placements.json").read_text(encoding="utf-8"))
    grid = geometry.Grid.from_protocol(config.load_protocol())
    frames = audit.first_frames(record["repo_id"])
    view = TableView.from_frames([frame for _, frame in frames], load_geometry())
    measuring = view.with_background(lerobot101.load("placement").empty_scene([frame for _, frame in frames]))

    august = {p["episode_index"]: p for p in record["placements"]}
    rows, one_map, as_august, offsets = [], defaultdict(list), defaultdict(list), []
    for index, frame in sorted(frames, key=lambda item: item[0]):
        measurement = measuring.measure(frame)
        if measurement is None:
            raise SystemExit(f"episode {index}: no single clean cube in its first frame")
        cell = august[index]["cell"]
        centre = grid.cell_centre(cell)
        here = np.array([measurement.position.x_mm, measurement.position.y_mm])
        there = np.array(august[index]["mm"], dtype=float)
        one_map[cell].append(here)
        as_august[cell].append(there)
        offsets.append(here - np.array([centre.x_mm, centre.y_mm]))
        rows.append([index, cell, here, there, offsets[-1]])

    offsets_array = np.array(offsets)
    common = offsets_array.mean(axis=0)
    from_centre = np.linalg.norm(offsets_array, axis=1)
    leftover = np.linalg.norm(offsets_array - common, axis=1)
    per_cell, mean_scatter, max_scatter = scatter(one_map)
    per_cell_august, mean_scatter_august, max_scatter_august = scatter(as_august)
    cell_means = {cell: np.array(points).mean(axis=0) for cell, points in one_map.items()}

    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["episode_index", "cell", "x_mm_onemap", "y_mm_onemap", "x_mm_august", "y_mm_august",
                         "dx_from_cell_centre_mm", "dy_from_cell_centre_mm", "from_cell_mean_start_mm"])
        for index, cell, here, there, offset in rows:
            writer.writerow([index, cell, f"{here[0]:.2f}", f"{here[1]:.2f}", f"{there[0]:.2f}", f"{there[1]:.2f}",
                             f"{offset[0]:.2f}", f"{offset[1]:.2f}",
                             f"{float(np.linalg.norm(here - cell_means[cell])):.2f}"])

    print(f"episodes measured: {len(rows)} of {len(frames)}; one conversion from {view.frames_used} frames, "
          f"tag fit {view.residual_mm:.2f} mm")
    print(f"measured position minus cell centre, mean over the episodes: ({common[0]:+.1f}, {common[1]:+.1f}) mm, "
          f"length {float(np.linalg.norm(common)):.1f} mm")
    print(f"distance from the cell centre: mean {from_centre.mean():.1f} mm, largest {from_centre.max():.1f} mm")
    print(f"the same with the common offset removed: mean {leftover.mean():.1f} mm, largest {leftover.max():.1f} mm")
    print("within a cell, one conversion (mean / largest distance from the cell's mean start; the two farthest apart):")
    for cell, (mean, largest, farthest) in per_cell.items():
        print(f"  {cell}: {mean:.1f} / {largest:.1f} mm; {farthest:.1f} mm")
    farthest_all = [farthest for _, _, farthest in per_cell.values()]
    print(f"all cells, one conversion: mean {mean_scatter:.1f} mm, largest {max_scatter:.1f} mm; "
          f"the two farthest apart in a cell: {min(farthest_all):.1f} to {max(farthest_all):.1f} mm")
    farthest_august = [farthest for _, _, farthest in per_cell_august.values()]
    print(f"all cells, August's own conversion: mean {mean_scatter_august:.1f} mm, largest {max_scatter_august:.1f} mm; "
          f"the two farthest apart in a cell: {min(farthest_august):.1f} to {max(farthest_august):.1f} mm")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # the dataset reader's thread pool can hang at exit in this environment (docs/journal.md)
