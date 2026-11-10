"""Do the model's lapse values reach far enough? A simulation made before the protocol is frozen.

The success model has a plateau, the success rate at a demonstration's own start, written `1 - lapse`. The protocol's
list of lapse values stopped at 0.10, which fits a policy that nearly always succeeds where it was taught. A replay, or
a policy trained on starts that are never repeated, may have a lower plateau. This asks what the estimate of the
tolerance radius does then, with the protocol's list and with a list that reaches 0.50, on the same simulated blocks.

    python -m tolerance_radius.lapse_check --out analysis/simulation

**Nothing here is a measurement.** Every block is played against a curve assumed true, as in `simulate`.

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

from tolerance_radius import config, psychometric, simulate
from tolerance_radius.seeds import derived_rng

#: The longer list that is compared with the protocol's, when the protocol file names none of its own
#: (`analysis.lapse_values_sensitivity`).
LAPSE_WIDER = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
#: The curves assumed true: every radius and width with every plateau.
TRUE_LAPSE = (0.0, 0.05, 0.20, 0.35, 0.50)
TRUE_R50_MM = (10.0, 20.0, 30.0)
TRUE_WIDTH_MM = (1.0, 3.0, 6.0)
REPLICATIONS = 200
SEED = 20261005
FITS = ("as the protocol", "wider list")


def wider_list(protocol: Mapping[str, Any]) -> tuple[float, ...]:
    """The longer list of lapse values: the protocol's own if it names one."""
    return tuple(float(v) for v in protocol["analysis"].get("lapse_values_sensitivity", LAPSE_WIDER))


def run(protocol: Mapping[str, Any], *, replications: int = REPLICATIONS, seed: int = SEED,
        lapses: Sequence[float] = TRUE_LAPSE, radii: Sequence[float] = TRUE_R50_MM,
        widths: Sequence[float] = TRUE_WIDTH_MM) -> list[dict[str, Any]]:
    """One row per block, assumed curve and list of lapse values: how the radius was estimated."""
    grids = psychometric.grids_from_protocol(protocol["analysis"])
    fits = {FITS[0]: grids, FITS[1]: grids._replace(lapse_values=np.array(wider_list(protocol)))}
    steps = simulate.steps_from_protocol(dict(protocol))
    rows: list[dict[str, Any]] = []
    for block in simulate.blocks_from_protocol(dict(protocol)):
        for lapse in lapses:
            for r50 in radii:
                for width in widths:
                    truth = simulate.Truth(r50_mm=r50, width_mm=width, lapse=lapse, floor=0.0)
                    rng = derived_rng(seed, "lapse-check", block.name, f"{r50:g}", f"{width:g}", f"{lapse:g}")
                    results = simulate.evaluate_fits(truth, block, steps, rng, fits=fits, replications=replications)
                    for name in FITS:
                        performance, _floor = results[name]
                        rows.append({
                            "block": block.name, "fit": name, "true_lapse": lapse, "true_r50_mm": r50,
                            "true_width_mm": width, "bias_mm": performance.bias_mm, "rmse_mm": performance.rmse_mm,
                            "interval_width_mm": performance.mean_interval_width_mm,
                            "contains_truth": performance.interval_coverage,
                            "open_on_a_side": performance.any_edge_rate,
                        })
    return rows


def pooled(rows: Sequence[Mapping[str, Any]], **where: Any) -> dict[str, float]:
    """The rows that match `where`, averaged (the error as the root of the mean square)."""
    chosen = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if not chosen:
        raise ValueError(f"no rows where {where}")
    return {
        "bias_mm": float(np.mean([row["bias_mm"] for row in chosen])),
        "rmse_mm": float(np.sqrt(np.mean([row["rmse_mm"] ** 2 for row in chosen]))),
        "interval_width_mm": float(np.mean([row["interval_width_mm"] for row in chosen])),
        "contains_truth": float(np.mean([row["contains_truth"] for row in chosen])),
        "worst_contains_truth": float(np.min([row["contains_truth"] for row in chosen])),
        "open_on_a_side": float(np.mean([row["open_on_a_side"] for row in chosen])),
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["block", "fit", "true_lapse", "true_r50_mm", "true_width_mm", "bias_mm", "rmse_mm",
                         "interval_width_mm", "contains_truth", "open_on_a_side"])
        for row in rows:
            writer.writerow([row["block"], row["fit"], f"{row['true_lapse']:g}", f"{row['true_r50_mm']:g}",
                             f"{row['true_width_mm']:g}", f"{row['bias_mm']:.3f}", f"{row['rmse_mm']:.3f}",
                             f"{row['interval_width_mm']:.3f}", f"{row['contains_truth']:.4f}",
                             f"{row['open_on_a_side']:.4f}"])


def render_report(rows: Sequence[Mapping[str, Any]], *, replications: int, seed: int, protocol_sha: str,
                  protocol_lapses: Sequence[float], wider: Sequence[float] = LAPSE_WIDER) -> str:
    lapses = sorted({row["true_lapse"] for row in rows})
    blocks = list(dict.fromkeys(row["block"] for row in rows))
    out = [
        "# The model's lapse values: do they reach far enough?",
        "",
        "**Every number in this report is simulated. Nothing here is a measurement.**",
        "",
        f"Written by `python -m tolerance_radius.lapse_check`. Seed {seed}; {replications} simulated blocks for every",
        f"row of `lapse_values.csv`. Protocol file SHA-256 `{protocol_sha}`.",
        "",
        "## The question",
        "",
        "The plateau of the success curve, the success rate at a demonstration's own start, is `1 - lapse`. The",
        f"protocol's model weighs the lapse values {', '.join(f'{v:g}' for v in protocol_lapses)}. If a condition's",
        "true plateau is lower than that list allows (a replay that does not always repeat, a policy taught from",
        "starts that are never repeated), what happens to the estimate of its tolerance radius? And what does a longer",
        "list",
        f"({', '.join(f'{v:g}' for v in wider)}) cost when the plateau is high after all?",
        "",
        "Every block of the protocol was played against curves with a radius of "
        f"{', '.join(f'{v:g}' for v in sorted({r['true_r50_mm'] for r in rows}))} mm, a width of",
        f"{', '.join(f'{v:g}' for v in sorted({r['true_width_mm'] for r in rows}))} mm and no floor, at each plateau"
        " below, and fitted both ways on the same",
        "simulated trials.",
        "",
        "## All blocks together, by the true plateau",
        "",
        "| True plateau | List | Bias (mm) | RMSE (mm) | Interval width (mm) | Contains the truth (%) | "
        "Lowest row (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for lapse in lapses:
        for fit in FITS:
            p = pooled(rows, true_lapse=lapse, fit=fit)
            out.append(f"| {1 - lapse:.2f} | {fit} | {p['bias_mm']:+.1f} | {p['rmse_mm']:.1f} | "
                       f"{p['interval_width_mm']:.1f} | {100 * p['contains_truth']:.0f} | "
                       f"{100 * p['worst_contains_truth']:.0f} |")
    out += ["", "## By block", "",
            "| Block | True plateau | Bias: protocol / wider (mm) | RMSE: protocol / wider (mm) | "
            "Contains the truth: protocol / wider (%) |",
            "|---|---|---|---|---|"]
    for block in blocks:
        for lapse in lapses:
            a, b = (pooled(rows, block=block, true_lapse=lapse, fit=fit) for fit in FITS)
            out.append(f"| {block} | {1 - lapse:.2f} | {a['bias_mm']:+.1f} / {b['bias_mm']:+.1f} | "
                       f"{a['rmse_mm']:.1f} / {b['rmse_mm']:.1f} | "
                       f"{100 * a['contains_truth']:.0f} / {100 * b['contains_truth']:.0f} |")
    high = [pooled(rows, true_lapse=lapse, fit=fit) for lapse in lapses[:2] for fit in FITS]
    cost = np.mean([high[1]["interval_width_mm"] / high[0]["interval_width_mm"],
                    high[3]["interval_width_mm"] / high[2]["interval_width_mm"]]) - 1.0
    out += [
        "",
        "## Reading",
        "",
        f"- **With a high plateau** (0.95 or more) the longer list makes the interval {100 * cost:+.0f} % wider on",
        "  average: that is its whole cost.",
        "- **With a low plateau** the protocol's list has no value that fits, so the model explains the failures at",
        "  small distances by pulling the radius inward: the bias column shows by how much, and the interval then",
        "  misses the truth more often than its level promises.",
        "- The longer list removes most of that bias. It cannot remove all of it at a plateau of 0.5, where a",
        "  one-up-one-down staircase itself drifts toward zero and half the trials near a demonstration fail.",
        "",
        "## What this does not show",
        "",
        "- Curves with a floor as well as a low plateau, or of another shape than the model's.",
        "- How to tell a low plateau from a small radius with 4 to 6 trials at a demonstration's own start: the fit",
        "  cannot, and the counts at 0 mm are reported beside every radius for that reason.",
        "",
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for lapse_values.md and .csv")
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    protocol = config.load_protocol()
    rows = run(protocol, replications=args.replications, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "lapse_values.csv")
    sha = hashlib.sha256(config.DEFAULT_PROTOCOL_PATH.read_bytes()).hexdigest()[:12]
    report = render_report(rows, replications=args.replications, seed=args.seed, protocol_sha=sha,
                           protocol_lapses=protocol["analysis"]["lapse_values"], wider=wider_list(protocol))
    (out / "lapse_values.md").write_text(report, encoding="utf-8")
    print(f"wrote {out / 'lapse_values.md'} and its table: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
