#!/usr/bin/env python3
"""Write the two demonstration layouts and their preview figure from the protocol.

    python scripts/make_layouts.py

writes `config/layouts/clustered.json`, `config/layouts/spread.json` and `analysis/layouts/layouts.png`, and prints how
evenly each layout covers the trained cells. A layout file that already holds other episodes is left alone unless
`--replace` is given: recording may have started from it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tolerance_radius import config, layout_figure, layouts, planner, targets  # noqa: E402
from tolerance_radius.geometry import Grid  # noqa: E402
from tolerance_radius.schema import DATASETS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the two demonstration layouts and their preview figure.")
    parser.add_argument("--protocol", type=Path, default=config.DEFAULT_PROTOCOL_PATH, help="the protocol file")
    parser.add_argument("--root", type=Path, default=config.REPO_ROOT, help="the folder holding config/ and analysis/")
    parser.add_argument("--replace", action="store_true", help="replace a layout file that holds other episodes")
    args = parser.parse_args(argv)

    protocol = config.load_protocol(args.protocol)
    grid = Grid.from_protocol(protocol)
    seed, version = protocol["datasets"]["layout_seed"], protocol["protocol"]["version"]
    plans = {dataset: layouts.layout(dataset, protocol) for dataset in DATASETS}
    paths = {dataset: layouts.layout_path(dataset, args.root) for dataset in DATASETS}
    figure = args.root / "analysis" / "layouts" / "layouts.png"

    def shown(path: Path) -> str:
        return path.relative_to(args.root).as_posix()

    print(f"Protocol {version}, layout seed {seed}")
    for dataset, episodes in plans.items():
        quality = layouts.layout_quality(episodes, grid)
        at = quality.largest_gap_at
        print(
            f"  {dataset:<10}{quality.episodes} episodes at {quality.distinct_positions} positions; "
            f"closest two {quality.min_spacing_mm:.1f} mm apart; "
            f"mean distance to the nearest other {quality.mean_nearest_neighbour_mm:.1f} mm; "
            f"largest gap {quality.largest_gap_mm:.1f} mm at ({at.x_mm:.1f}, {at.y_mm:.1f})"
        )

    # Nothing is written unless everything can be: a refused layout leaves the other file and the figure as they were.
    changed = [name for name in DATASETS if layouts.layout_file_state(paths[name], plans[name]) == "different"]
    if changed and not args.replace:
        names = ", ".join(shown(paths[dataset]) for dataset in changed)
        print(f"error: {names} holds other episodes than this protocol gives.", file=sys.stderr)
        print("Recording may have started from it. Nothing was written; pass --replace to overwrite.", file=sys.stderr)
        return 1
    for dataset, episodes in plans.items():
        path = paths[dataset]
        wrote = layouts.write_layout(path, episodes, seed=seed, protocol_version=version, replace=args.replace)
        print(f"{'wrote' if wrote else 'unchanged'}: {shown(path)}")
    layout_figure.save_layout_preview(figure, plans, grid)
    print(f"saved: {shown(figure)}")

    print("If every cube started exactly on its planned position (the audit will differ by the placement error):")
    starts = {dataset: layouts.planned_demo_starts(episodes) for dataset, episodes in plans.items()}
    for dataset in DATASETS:
        farthest = targets.max_feasible_level(starts[dataset], grid)
        print(f"  {dataset:<10}a test placement can be at most {farthest:.1f} mm from its nearest demonstration")
    everything = [start for dataset in DATASETS for start in starts[dataset]]
    for block in protocol["blocks"]:
        problems = planner.check_block(block, protocol, everything)
        print(f"  {block:<10}{'; '.join(problems) if problems else 'every distance it may ask for can be placed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
