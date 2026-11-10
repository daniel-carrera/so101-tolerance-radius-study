"""What the paired workspace test can tell apart, for different ways of drawing its placements.

A calculation made before any trial, to choose the design of block T2 (`protocol/protocol.md`, section 8). It answers:
if each policy did no more than work within a tolerance radius of its nearest demonstration (the lookup model), how
often would the exact paired test tell the clustered policy from the spread one, for placements drawn over the trained
cells, over the held-out cells, or over both?

For one set of placements the answer is exact, not simulated: each placement has a success probability for each policy
(from its distance to that policy's nearest planned demonstration), the two outcomes at a placement are taken as
independent, and the distribution of the two discordant counts follows by adding one pair at a time. The answer is then
averaged over many draws of the placements, so that it describes the design and not one lucky draw.

**Nothing here is a measurement.** The lookup model is the hypothesis the workspace test exists to check; if the
policies interpolate between demonstrations, or thin out when spread, the real test will differ, and that is its point.

    python -m tolerance_radius.workspace_power --out analysis/simulation

Standard library and NumPy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius import config, layouts, psychometric
from tolerance_radius.geometry import Grid
from tolerance_radius.intervals import mcnemar_exact
from tolerance_radius.schema import DATASETS, Point
from tolerance_radius.seeds import derived_rng
from tolerance_radius.targets import uniform_placements

#: A design: how many placements are drawn from which region.
Design = Sequence[tuple[str, int]]

#: The designs compared. The first was the protocol's until draft 0.2; the last is the protocol's since draft 0.3
#: (`[blocks.T2]` of the parameter file, decided on 2026-10-05 from this comparison).
DESIGNS: dict[str, Design] = {
    "trained cells, 20": (("trained_cells", 20),),
    "whole grid, 20": (("test_region", 20),),
    "whole grid, 26": (("test_region", 26),),
    "whole grid, 30": (("test_region", 30),),
    "10 trained + 10 held-out": (("trained_cells", 10), ("held_out_cells", 10)),
    "13 trained + 13 held-out": (("trained_cells", 13), ("held_out_cells", 13)),
}
#: The success curves assumed true: every radius with every width.
RADII_MM = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
WIDTHS_MM = (1.0, 3.0, 6.0)
#: Assumed for every curve: a plateau of 0.97 and a floor of 0.03.
LAPSE = 0.03
FLOOR = 0.03
ALPHA = 0.05
DRAWS = 200
SEED = 20261005


def discordant_distribution(p_first: Any, p_second: Any) -> np.ndarray:
    """The joint distribution of the two discordant counts of a paired test.

    `p_first[i]` and `p_second[i]` are the two policies' success probabilities at placement `i`; the two outcomes at a
    placement are independent. Entry `[a, b]` of the result is the probability that only the first policy succeeded at
    exactly `a` placements and only the second at exactly `b`.
    """
    first = np.asarray(p_first, dtype=float).ravel()
    second = np.asarray(p_second, dtype=float).ravel()
    if first.shape != second.shape:
        raise ValueError(f"one probability per placement for each policy, got {first.size} and {second.size}")
    if ((first < 0) | (first > 1) | (second < 0) | (second > 1)).any():
        raise ValueError("a success probability lies between 0 and 1")
    n = first.size
    pmf = np.zeros((n + 1, n + 1))
    pmf[0, 0] = 1.0
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        only_first, only_second = a * (1.0 - b), b * (1.0 - a)
        same = 1.0 - only_first - only_second
        step = same * pmf
        step[1:, :] += only_first * pmf[:-1, :]
        step[:, 1:] += only_second * pmf[:, :-1]
        pmf = step
    return pmf


def _rejection_table(n: int, alpha: float) -> np.ndarray:
    """`[a, b]` is True when the exact McNemar test on `a` and `b` discordant pairs gives p below `alpha`."""
    return np.array([[mcnemar_exact(a, b) < alpha if a + b <= n else False for b in range(n + 1)]
                     for a in range(n + 1)])


def paired_power(p_first: Any, p_second: Any, alpha: float = ALPHA, *, rejects: np.ndarray | None = None) -> float:
    """The probability that the exact McNemar test gives p below `alpha`, for these placements.

    `rejects` may carry `_rejection_table(n, alpha)` when many sets of the same size are evaluated.
    """
    pmf = discordant_distribution(p_first, p_second)
    table = _rejection_table(pmf.shape[0] - 1, alpha) if rejects is None else rejects
    return float(pmf[table].sum())


def design_placements(grid: Grid, design: Design, seed: int) -> list[Point]:
    """One draw of a design's placements: so many drawn evenly over each of its regions."""
    points: list[Point] = []
    for region, count in design:
        points += uniform_placements(grid, count, derived_rng(seed, "workspace-design", region, str(count)), region)
    return points


def lookup_probabilities(
    points: Sequence[Point], demos_xy: np.ndarray, *, r50_mm: float, width_mm: float, lapse: float, floor: float
) -> np.ndarray:
    """Each point's success probability under the lookup model: the curve at its distance to the nearest start."""
    xy = np.array([(point.x_mm, point.y_mm) for point in points], dtype=float)
    nearest = np.sqrt(((xy[:, None, :] - demos_xy[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    return psychometric.success_probability(nearest, r50_mm, width_mm, lapse, floor)


def planned_starts(layout_dir: Path) -> dict[str, np.ndarray]:
    """The planned start positions of both demonstration sets, as arrays of (x, y)."""
    return {
        dataset: np.array([(e.target.x_mm, e.target.y_mm) for e in layouts.read_layout(layout_dir / f"{dataset}.json")])
        for dataset in DATASETS
    }


def compare_designs(
    protocol: Mapping[str, Any],
    designs: Mapping[str, Design] = DESIGNS,
    truths: Sequence[tuple[float, float]] | None = None,
    *,
    draws: int = DRAWS,
    seed: int = SEED,
    layout_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """One row per design and assumed curve: the expected successes of each policy and the test's power.

    Each figure is the mean over `draws` draws of the design's placements; the same draws serve every curve.
    """
    grid = Grid.from_protocol(protocol)
    starts = planned_starts(layout_dir or config.REPO_ROOT / "config" / "layouts")
    truths = list(truths) if truths is not None else [(r, w) for r in RADII_MM for w in WIDTHS_MM]
    rows: list[dict[str, Any]] = []
    for name, design in designs.items():
        pairs = sum(count for _, count in design)
        rejects = _rejection_table(pairs, ALPHA)
        placements = [design_placements(grid, design, seed + draw) for draw in range(draws)]
        for r50, width in truths:
            clustered, spread, power = [], [], []
            curve = {"r50_mm": r50, "width_mm": width, "lapse": LAPSE, "floor": FLOOR}
            for points in placements:
                p_c = lookup_probabilities(points, starts["clustered"], **curve)
                p_s = lookup_probabilities(points, starts["spread"], **curve)
                clustered.append(float(p_c.sum()))
                spread.append(float(p_s.sum()))
                power.append(paired_power(p_c, p_s, rejects=rejects))
            rows.append({"design": name, "pairs": pairs, "r50_mm": r50, "width_mm": width,
                         "expected_clustered": float(np.mean(clustered)), "expected_spread": float(np.mean(spread)),
                         "power": float(np.mean(power))})
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["design", "pairs", "r50_mm", "width_mm", "expected_clustered", "expected_spread", "power"])
        for row in rows:
            writer.writerow([row["design"], row["pairs"], f"{row['r50_mm']:g}", f"{row['width_mm']:g}",
                             f"{row['expected_clustered']:.2f}", f"{row['expected_spread']:.2f}",
                             f"{row['power']:.4f}"])


def render_report(rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int, layouts_sha: str) -> str:
    """The comparison as a short page for a person."""
    designs = list(dict.fromkeys(row["design"] for row in rows))
    radii = sorted({row["r50_mm"] for row in rows})
    widths = sorted({row["width_mm"] for row in rows})
    cell = {(row["design"], row["r50_mm"], row["width_mm"]): row for row in rows}
    out = [
        "# The workspace test: what each way of drawing its placements could tell apart",
        "",
        "**Every number here is calculated from stated assumptions. Nothing here is a measurement.**",
        "",
        f"Written by `python -m tolerance_radius.workspace_power`. Seed {seed}; each figure is the mean over {draws}",
        f"draws of a design's placements. Planned layouts: `config/layouts/`, SHA-256 `{layouts_sha}`. The rows are in",
        "`workspace_designs.csv`.",
        "",
        "## The question",
        "",
        "Block T2 tests the two policies at the same placements and compares them with an exact paired test. Where",
        "should the placements be drawn so that the test can show a difference, whatever the tolerance radius turns",
        "out to be?",
        "",
        "## The assumptions",
        "",
        "- **The lookup model.** Each policy succeeds with the probability a success curve gives at the placement's",
        "  distance to that policy's nearest planned demonstration: plateau "
        f"{1 - LAPSE:g}, floor {FLOOR:g}, and a radius and a width",
        "  as in the tables. This is the hypothesis the workspace test exists to check, used here only to choose",
        "  its design.",
        "- The two policies' outcomes at one placement are independent, and both policies have the same curve.",
        "- For one set of placements the test's power is computed exactly; the mean over the draws describes the",
        "  design.",
        "",
        f"## How often the test would reach p < {ALPHA:g} (%)",
        "",
    ]
    for width in widths:
        out += [f"**Width {width:g} mm.**", "",
                "| Design | " + " | ".join(f"radius {r:g} mm" for r in radii) + " |",
                "|---|" + "---|" * len(radii)]
        for design in designs:
            powers = " | ".join(f"{100 * cell[(design, r, width)]['power']:.0f}" for r in radii)
            out.append(f"| {design} | {powers} |")
        out.append("")
    out += ["## Expected successes of each policy (clustered / spread), width 3 mm", "",
            "| Design | Pairs | " + " | ".join(f"radius {r:g} mm" for r in radii) + " |",
            "|---|---|" + "---|" * len(radii)]
    width = 3.0 if 3.0 in widths else widths[0]
    for design in designs:
        first = cell[(design, radii[0], width)]
        out.append(f"| {design} | {first['pairs']} | " + " | ".join(
            f"{cell[(design, r, width)]['expected_clustered']:.1f} / {cell[(design, r, width)]['expected_spread']:.1f}"
            for r in radii) + " |")
    out += [
        "",
        "## Reading",
        "",
        "- **Placements in the trained cells alone are blind to a large radius.** Eleven clusters on a 50 mm grid",
        "  already cover their own cells once the radius approaches 25 mm, so both policies succeed there and the",
        "  test has nothing to find.",
        "- **Placements in the held-out cells alone are blind to a small radius**: neither policy reaches them.",
        "- **Half in each keeps the test alive over the whole plausible range**, at the cost of being a little weaker",
        "  than the best single region at either end.",
        "- More pairs help every design; the tables show by how much.",
        "",
        "## What this does not show",
        "",
        "- Whether the lookup model is true. If the spread policy is weaker at each point because no start is",
        "  repeated, or the clustered one reaches between its clusters, the real powers differ.",
        "- Outcomes that depend on anything but distance: direction, place on the table, the day.",
        "- Curves that differ between the two policies.",
        "",
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for workspace_designs.md and .csv")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    protocol = config.load_protocol()
    layout_dir = config.REPO_ROOT / "config" / "layouts"
    digest = hashlib.sha256(b"".join((layout_dir / f"{dataset}.json").read_bytes() for dataset in DATASETS))
    rows = compare_designs(protocol, draws=args.draws, seed=args.seed, layout_dir=layout_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "workspace_designs.csv")
    report = render_report(rows, draws=args.draws, seed=args.seed, layouts_sha=digest.hexdigest()[:12])
    (out / "workspace_designs.md").write_text(report, encoding="utf-8")
    print(f"wrote {out / 'workspace_designs.md'} and its table: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
