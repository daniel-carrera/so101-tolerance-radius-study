#!/usr/bin/env python3
"""Write the synthetic `results.json` files the figure tests draw from. Nothing here is a result.

    .venv/bin/python tests/fixtures/make_results_fixtures.py

writes `tests/fixtures/results/*.json`, each in the shape `protocol/data-contracts.md` gives the analysis results. The
figures cannot wait for the experiment, and the program that will write the real file is being built apart from them,
so these files are made here, by hand: trials are **simulated** against success curves that were **invented**
(`TRUTHS`), on demonstration positions that are the committed layouts plus an invented placement error, and fitted
with the project's own model. Every file says so: `draft` is true and `provenance.note` reads "Synthetic ...".

The files, each a whole imagined experiment (`CASES`):

* `base`: every block ran; the workspace test has 20 placements, all in trained cells, as the protocol had it until
  draft 0.2 (`ONE_REGION`); distances are scale-corrected;
* `no_spread_threshold`: the spread policy's threshold block did not run, and the scale check did not agree, so
  distances are as read;
* `open_above`: the clustered policy never fails within reach of its staircases, so its interval is open above;
* `threshold_blocks_only`: only the two threshold blocks ran (no workspace test), and the replay already fails at
  its demonstration's own start, so its interval is open below;
* `held_out_placements`: the workspace test has 13 placements in trained cells and 13 in held-out cells, as the
  protocol has it since draft 0.3, and the file carries each pair's `region` and the counts `by_region`.

The threshold blocks have the sizes the protocol file states. The workspace test does not follow that file: each
case states its own, so that a file without regions stays among the cases whatever the protocol plans. The replay
plays back four fixed demonstrations in turn, which is this script's own short cut and not the planner's rule.

The files are committed and the tests read them; this script is how they were made and how to make them again. It
checks, before writing, that each file shows what it is meant to show.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tolerance_radius import config, layouts, targets  # noqa: E402
from tolerance_radius.geometry import Grid, nearest, trained_demos  # noqa: E402
from tolerance_radius.intervals import mcnemar_exact, wilson_interval  # noqa: E402
from tolerance_radius.psychometric import (  # noqa: E402
    ThresholdPosterior,
    fit_threshold,
    grids_from_protocol,
    success_probability,
)
from tolerance_radius.schema import DATASET_OF_CONDITION, DemoStart, Point  # noqa: E402
from tolerance_radius.seeds import derived_rng  # noqa: E402
from tolerance_radius.staircase import StaircaseParams, next_level  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261005
LEVEL = 0.95
BIN_MM = 10.0
#: The step of the distances a fitted curve is given on.
CURVE_STEP_MM = 0.5
#: How long the instrument reads a true length, in the files whose distances are corrected.
SCALE_FACTOR = 1.031
BLOCK_OF = {"act_clustered": "T1a", "replay": "T1b", "act_spread": "T3"}
#: The workspace test of a case that names no regions: so many placements, all drawn over the trained cells.
ONE_REGION = (("trained_cells", 20),)

#: Invented success curves: `(r50, width, lapse, floor)` in millimetres of true distance.
TRUTHS = {
    "act_clustered": (22.0, 3.0, 0.03, 0.04),
    "replay": (11.0, 2.5, 0.02, 0.0),
    "act_spread": (18.0, 3.0, 0.03, 0.03),
}
#: What each file changes in that: other curves, blocks that did not run, another workspace test.
CASES: dict[str, dict[str, Any]] = {
    "base": {},
    "no_spread_threshold": {"blocks": ("T1a", "T1b", "T2"), "scale_applied": False},
    "open_above": {"truths": {"act_clustered": (75.0, 3.0, 0.0, 0.0)}, "blocks": ("T1a", "T1b", "T2")},
    "threshold_blocks_only": {"truths": {"replay": (2.0, 6.0, 0.05, 0.0)}, "blocks": ("T1a", "T1b")},
    "held_out_placements": {
        "blocks": ("T1a", "T1b", "T2"),
        "regions": (("trained_cells", 13), ("held_out_cells", 13)),
    },
}


# --------------------------------------------------------------------------------------------------------------------
# The imagined experiment
# --------------------------------------------------------------------------------------------------------------------


def audited(dataset: str, rng: np.random.Generator) -> list[DemoStart]:
    """A start-state audit that never happened: every demonstration within 1.2 mm of its planned start."""
    starts = []
    for episode in layouts.read_layout(layouts.layout_path(dataset)):
        radius, angle = 1.2 * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
        measured = Point(episode.target.x_mm + radius * math.cos(angle), episode.target.y_mm + radius * math.sin(angle))
        starts.append(DemoStart(dataset, episode.episode_index, episode.cell, measured=measured, planned=episode.target))
    return starts


def placed_near(target: Point, grid: Grid, rng: np.random.Generator) -> Point:
    """Where the cube was accepted: within 1.6 mm of its target, inside the test region."""
    region = grid.test_region
    radius, angle = 1.6 * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
    x = min(max(target.x_mm + radius * math.cos(angle), region.x_min_mm), region.x_max_mm)
    y = min(max(target.y_mm + radius * math.sin(angle), region.y_min_mm), region.y_max_mm)
    return Point(x, y)


class Experiment:
    """One imagined experiment: its demonstrations, its invented curves, and the trials played against them."""

    def __init__(self, name: str, changes: dict[str, Any]) -> None:
        self.name = name
        self.protocol = config.load_protocol()
        self.grid = Grid.from_protocol(self.protocol)
        self.truths = TRUTHS | changes.get("truths", {})
        self.blocks = changes.get("blocks", ("T1a", "T1b", "T2", "T3"))
        self.regions = changes.get("regions")
        self.scale_applied = changes.get("scale_applied", True)
        self.divisor = SCALE_FACTOR if self.scale_applied else 1.0
        self.demos = {dataset: audited(dataset, derived_rng(SEED, "audit", dataset)) for dataset in ("clustered", "spread")}
        self.trials: dict[str, list[dict[str, Any]]] = {condition: [] for condition in TRUTHS}
        self.pairs: list[dict[str, Any]] = []

    def rng(self, *labels: str | int) -> np.random.Generator:
        return derived_rng(SEED, self.name, *labels)

    def outcome(self, condition: str, d_mm: float, rng: np.random.Generator) -> bool:
        r50, width, lapse, floor = self.truths[condition]
        return bool(rng.random() < float(success_probability(d_mm, r50, width, lapse, floor)))

    def distance(self, condition: str, placed: Point, replayed: DemoStart | None) -> float:
        """The distance the analysis would use: to the nearest trained demonstration of the condition's set (for a
        replay, to the demonstration played back), divided by the scale factor where the file corrects distances.
        """
        if replayed is not None:
            read = placed.distance_to(replayed.measured)
        else:
            read = nearest(placed, trained_demos(self.demos[DATASET_OF_CONDITION[condition]], DATASET_OF_CONDITION[condition]))[1]
        return read / self.divisor

    def record(self, condition: str, block: str, part: str, placed: Point, level: float | None, success: bool,
               d_mm: float, pair_id: str | None, rng: np.random.Generator) -> dict[str, Any]:  # fmt: skip
        number = sum(1 for trials in self.trials.values() for trial in trials if trial["block"] == block) + 1
        trial = {
            "trial_id": f"{block}-{number:03d}", "block": block, "part": part, "d_mm": round(d_mm, 3),
            "success": success, "subgoal": 5 if success else (1 if rng.random() < 0.8 else 3),
            "failure": None if success else ("perception" if rng.random() < 0.7 else "grasp"),
            "x_mm": round(placed.x_mm, 2), "y_mm": round(placed.y_mm, 2), "level_mm": level, "pair_id": pair_id,
        }  # fmt: skip
        self.trials[condition].append(trial)
        return trial

    def threshold_block(self, condition: str) -> None:
        """Zero anchors, far anchors and two staircases, as `[blocks.<name>]` of the protocol plans them."""
        block = BLOCK_OF[condition]
        entry = self.protocol["blocks"][block]
        dataset = DATASET_OF_CONDITION[condition]
        demos = trained_demos(self.demos[dataset], dataset)
        rng = self.rng(block)
        inside = [demo for demo in demos if self.grid.in_test_region(demo.measured)]
        # A replay plays back a few fixed demonstrations, from different cells, in turn.
        replayed = [inside[index] for index in (3, 17, 30, 44)] if condition == "replay" else []
        farthest = targets.max_feasible_level(demos, self.grid) - 1.0

        def run(part: str, level: float, turn: int) -> bool:
            demo = replayed[turn % len(replayed)] if replayed else None
            if level == 0.0:
                target = (demo or inside[int(rng.random() * len(inside))]).measured
            elif demo is not None:
                target = targets.placement_around(demo, level, self.grid, rng)
            else:
                target = targets.placement_at_distance(min(level, farthest), demos, self.grid, rng, 0.5)
            placed = placed_near(target, self.grid, rng)
            d_mm = self.distance(condition, placed, demo)
            success = self.outcome(condition, d_mm, rng)
            self.record(condition, block, part, placed, level, success, d_mm, None, rng)
            return success

        turn = 0
        for _ in range(entry.get("anchors_zero", 0)):
            run("zero", 0.0, turn)
            turn += 1
        for _ in range(entry.get("anchors_far", 0)):
            run("far", float(entry["far_level_mm"]), turn)
            turn += 1
        steps = self.protocol["staircase"]
        for plan in entry["staircases"]:
            rule = StaircaseParams(float(plan["start_level_mm"]), int(plan["trials"]), float(steps["step_initial_mm"]),
                                   float(steps["step_final_mm"]), float(steps["min_level_mm"]),
                                   float(entry["max_level_mm"]))  # fmt: skip
            history: list[tuple[float, bool]] = []
            while (level := next_level(rule, history)) is not None:
                history.append((level, run(plan["id"], level, turn)))
                turn += 1

    def workspace_block(self) -> None:
        """Placements drawn evenly over a region, each tested with both policies."""
        rng = self.rng("T2")
        regions = self.regions or ONE_REGION
        placements = [(region, point) for region, count in regions
                      for point in targets.uniform_placements(self.grid, count, rng, region)]  # fmt: skip
        order = np.argsort([rng.random() for _ in placements], kind="stable").tolist()
        for number, index in enumerate(order, start=1):
            region, target = placements[index]
            pair_id = f"T2-p{number:02d}"
            pair: dict[str, Any] = {"pair_id": pair_id}
            for condition in ("act_clustered", "act_spread"):
                placed = placed_near(target, self.grid, rng)
                d_mm = self.distance(condition, placed, None)
                success = self.outcome(condition, d_mm, rng)
                trial = self.record(condition, "T2", "pair", placed, None, success, d_mm, pair_id, rng)
                pair[condition] = {"trial_id": trial["trial_id"], "success": success, "d_mm": trial["d_mm"],
                                   "x_mm": trial["x_mm"], "y_mm": trial["y_mm"]}  # fmt: skip
            pair["x_mm"] = round((pair["act_clustered"]["x_mm"] + pair["act_spread"]["x_mm"]) / 2.0, 2)
            pair["y_mm"] = round((pair["act_clustered"]["y_mm"] + pair["act_spread"]["y_mm"]) / 2.0, 2)
            if self.regions:
                pair["region"] = region
            self.pairs.append(pair)

    def run(self) -> None:
        for condition, block in BLOCK_OF.items():
            if block in self.blocks:
                self.threshold_block(condition)
        if "T2" in self.blocks:
            self.workspace_block()


# --------------------------------------------------------------------------------------------------------------------
# The analysis of it, in the shape of the contract
# --------------------------------------------------------------------------------------------------------------------


def fit(trials: list[dict[str, Any]], grids: Any) -> ThresholdPosterior:
    return fit_threshold([t["d_mm"] for t in trials], [t["success"] for t in trials], **grids._asdict())


def fit_fields(posterior: ThresholdPosterior, blocks: list[str], curve_d: np.ndarray) -> dict[str, Any]:
    summary = posterior.summary(LEVEL)
    low, high = posterior.band(curve_d, LEVEL)
    return {
        "blocks": blocks, "n": summary.n_trials, "k": summary.n_success,
        "r50_median_mm": summary.r50_median_mm, "r50_lo_mm": summary.r50_low_mm, "r50_hi_mm": summary.r50_high_mm,
        "open_below": summary.reaches_lower_edge, "open_above": summary.reaches_upper_edge,
        "width_median_mm": round(summary.width_median_mm, 4), "lapse_mean": round(summary.lapse_mean, 5),
        "floor_mean": round(summary.floor_mean, 5),
        "curve": {"d_mm": [round(float(d), 2) for d in curve_d],
                  "mean": [round(float(p), 5) for p in posterior.predict(curve_d)],
                  "lo": [round(float(p), 5) for p in low], "hi": [round(float(p), 5) for p in high]},
    }  # fmt: skip


def bins(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Counts in 10 mm bins, closed below and open above; only bins that hold a trial."""
    found = []
    for index in sorted({int(t["d_mm"] // BIN_MM) for t in trials}):
        inside = [t for t in trials if int(t["d_mm"] // BIN_MM) == index]
        k = sum(t["success"] for t in inside)
        low, high = wilson_interval(k, len(inside), LEVEL)
        found.append({"lo_mm": index * BIN_MM, "hi_mm": (index + 1) * BIN_MM, "n": len(inside), "k": k,
                      "wilson_lo": round(low, 6), "wilson_hi": round(high, 6)})  # fmt: skip
    return found


def comparison(first: ThresholdPosterior, second: ThresholdPosterior) -> dict[str, Any]:
    """The radius of `first` minus that of `second`, from the two posteriors taken as independent."""
    grid, p_first = first.r50_marginal()
    _, p_second = second.r50_marginal()
    below = np.concatenate(([0.0], np.cumsum(p_second)[:-1]))
    prob_greater = float(np.sum(p_first * (below + 0.5 * p_second)))
    difference = np.convolve(p_first, p_second[::-1])
    step = float(grid[1] - grid[0])
    values = (np.arange(difference.size) - (grid.size - 1)) * step
    running = np.cumsum(difference)
    median, low, high = (float(values[int(np.searchsorted(running, mark))]) for mark in (0.5, 0.025, 0.975))
    verdict = "larger" if prob_greater >= 0.95 else "smaller" if prob_greater <= 0.05 else "not told apart"
    return {"fits": "block", "median_mm": median, "lo_mm": low, "hi_mm": high, "prob_greater": round(prob_greater, 4),
            "verdict": verdict}  # fmt: skip


def lookup(posterior: ThresholdPosterior, distances: list[float], observed: int, rng: np.random.Generator) -> dict[str, Any]:
    """What the clustered policy's block fit predicts for some placements: the expected count of successes and the
    interval of the predicted count, which carries the fit's uncertainty and the chance in the trials.
    """
    d = np.array(distances)
    expected = float(posterior.predict(d).sum())
    weights = posterior.probabilities.reshape(-1)
    draws = np.searchsorted(np.cumsum(weights), rng.random(4000) * weights.sum())
    r, w, la, fl = np.unravel_index(np.minimum(draws, weights.size - 1), posterior.probabilities.shape)
    chance = success_probability(d[None, :], posterior.r50_grid_mm[r][:, None], posterior.width_grid_mm[w][:, None],
                                 posterior.lapse_values[la][:, None], posterior.floor_values[fl][:, None])  # fmt: skip
    counts = np.sort((rng.random(chance.shape) < chance).sum(axis=1))
    low, high = int(counts[int(0.025 * counts.size)]), int(counts[int(0.975 * counts.size) - 1])
    verdict = "matches" if low <= observed <= high else "falls short" if observed < low else "exceeds"
    return {"expected": round(expected, 3), "lo": low, "hi": high, "observed": observed, "verdict": verdict}


def counted(successes: list[bool]) -> dict[str, Any]:
    k, n = sum(successes), len(successes)
    low, high = wilson_interval(k, n, LEVEL) if n else (None, None)
    return {"k": k, "n": n, "wilson_lo": None if low is None else round(low, 6),
            "wilson_hi": None if high is None else round(high, 6)}  # fmt: skip


def results(experiment: Experiment) -> dict[str, Any]:
    grids = grids_from_protocol(experiment.protocol["analysis"])
    everything = [t for trials in experiment.trials.values() for t in trials]
    reach = CURVE_STEP_MM * math.ceil(max(t["d_mm"] for t in everything) / CURVE_STEP_MM + 1e-9)
    curve_d = np.arange(0.0, max(reach, 50.0) + 1e-9, CURVE_STEP_MM)

    conditions: dict[str, Any] = {}
    block_fits: dict[str, ThresholdPosterior] = {}
    for condition, trials in experiment.trials.items():
        if not trials:
            conditions[condition] = None
            continue
        block = BLOCK_OF[condition]
        own = [t for t in trials if t["block"] == block]
        entry: dict[str, Any] = {"trials": trials, "n_void": {"act_clustered": 2, "replay": 1, "act_spread": 0}[condition],
                                 "bins": bins(trials)}  # fmt: skip
        successes = [t["d_mm"] for t in trials if t["success"]]
        failures = [t["d_mm"] for t in trials if not t["success"]]
        entry["max_success_d_mm"] = max(successes) if successes else None
        entry["min_failure_d_mm"] = min(failures) if failures else None
        if own:
            block_fits[condition] = fit(own, grids)
            entry["block_fit"] = fit_fields(block_fits[condition], [block], curve_d)
        else:
            entry["block_fit"] = None
        if condition == "replay":
            entry["pooled_fit"] = None
        else:
            run_blocks = sorted({t["block"] for t in trials}, key=("T1a", "T3", "T2").index)
            entry["pooled_fit"] = fit_fields(fit(trials, grids), run_blocks, curve_d)
        conditions[condition] = entry

    have = block_fits.keys()
    comparisons = {
        "clustered_minus_replay": comparison(block_fits["act_clustered"], block_fits["replay"])
        if {"act_clustered", "replay"} <= have else None,
        "spread_minus_clustered": comparison(block_fits["act_spread"], block_fits["act_clustered"])
        if {"act_spread", "act_clustered"} <= have else None,
    }  # fmt: skip

    workspace = None
    if experiment.pairs:
        clustered_fit = block_fits.get("act_clustered")
        for pair in experiment.pairs:
            for condition in ("act_clustered", "act_spread"):
                pair[condition]["p_lookup"] = (None if clustered_fit is None else
                                               round(float(clustered_fit.predict(pair[condition]["d_mm"])), 5))  # fmt: skip
        both = [(p["act_clustered"]["success"], p["act_spread"]["success"]) for p in experiment.pairs]
        table = {"both": sum(c and s for c, s in both), "neither": sum(not c and not s for c, s in both),
                 "only_spread": sum(s and not c for c, s in both), "only_clustered": sum(c and not s for c, s in both)}  # fmt: skip
        workspace = {
            "n_pairs": len(experiment.pairs), "pairs": experiment.pairs, "table": table,
            "mcnemar_p": round(mcnemar_exact(table["only_clustered"], table["only_spread"]), 6),
            "success": {c: counted([p[c]["success"] for p in experiment.pairs]) for c in ("act_clustered", "act_spread")},
            "lookup": {"fit": "act_clustered block_fit"} | {
                c: lookup(clustered_fit, [p[c]["d_mm"] for p in experiment.pairs],
                          sum(p[c]["success"] for p in experiment.pairs), experiment.rng("lookup", c))
                for c in ("act_clustered", "act_spread")
            },
        }  # fmt: skip
        if experiment.regions:
            workspace["by_region"] = {}
            for region, _ in experiment.regions:
                inside = [p for p in experiment.pairs if p["region"] == region]
                workspace["by_region"][region] = {"n_pairs": len(inside)} | {
                    c: counted([p[c]["success"] for p in inside])
                    | {"lookup_expected": round(sum(p[c]["p_lookup"] for p in inside), 3)}
                    for c in ("act_clustered", "act_spread")
                }

    n_scored = len(everything)
    n_void = sum(entry["n_void"] for entry in conditions.values() if entry)
    reset = None
    if "act_clustered" in block_fits:
        placed, by_eye = (float(block_fits["act_clustered"].predict(np.array(errors)).mean())
                          for errors in ([0.4, 0.8, 1.1, 1.5], [2.0, 4.5, 7.0, 11.0, 15.0]))  # fmt: skip
        reset = {"fit": "act_clustered block_fit", "placed": round(placed, 4), "by_eye": round(by_eye, 4),
                 "n_placed": n_scored, "n_by_eye": 20}  # fmt: skip
    return {
        "schema_version": 1,
        "draft": True,
        "provenance": {
            "protocol_version": experiment.protocol["protocol"]["version"], "protocol_status": "draft",
            "protocol_sha256": None, "project_commit": None, "logs": [],
            "n_records": n_scored + n_void, "n_scored": n_scored, "n_void": n_void,
            "note": "Synthetic: trials simulated against invented success curves by "
                    "tests/fixtures/make_results_fixtures.py. Not a result of the experiment.",
        },
        "distance_scale": {"applied": experiment.scale_applied, "factor": SCALE_FACTOR,
                           "factor_min": 1.027 if experiment.scale_applied else 1.012,
                           "factor_max": 1.034 if experiment.scale_applied else 1.049, "n_runs": 4,
                           "divisor": experiment.divisor},
        "bin_width_mm": BIN_MM,
        "interval_level": LEVEL,
        "conditions": conditions,
        "comparisons": comparisons,
        "workspace": workspace,
        "reset_sensitivity": reset,
        "demonstrations": {
            dataset: [{"episode_index": demo.episode_index, "cell": demo.cell, "x_mm": round(demo.measured.x_mm, 2),
                       "y_mm": round(demo.measured.y_mm, 2), "trained": demo.trained} for demo in demos]
            for dataset, demos in experiment.demos.items()
        },
    }  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------------------------------


def dumps(value: Any, indent: int = 0) -> str:
    """JSON with one trial, one bin, one demonstration or one list of numbers to a line, so that a file can be read."""
    pad = "  " * (indent + 1)
    flat = (dict, list)
    if isinstance(value, dict) and value and any(isinstance(item, flat) and item for item in value.values()):
        inner = ",\n".join(f"{pad}{json.dumps(key)}: {dumps(item, indent + 1)}" for key, item in value.items())
        return "{\n" + inner + "\n" + "  " * indent + "}"
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        inner = ",\n".join(pad + dumps(item, indent + 1) for item in value)
        return "[\n" + inner + "\n" + "  " * indent + "]"
    return json.dumps(value)


def check(name: str, made: dict[str, Any]) -> None:
    """Stop unless the file shows what its case is there to show."""
    conditions, workspace = made["conditions"], made["workspace"]
    clustered, replay = conditions["act_clustered"]["block_fit"], conditions["replay"]["block_fit"]
    json.loads(dumps(made))
    assert made["draft"] is True and "Synthetic" in made["provenance"]["note"]
    if name == "open_above":
        assert clustered["open_above"] and not clustered["open_below"], clustered
    else:
        assert not clustered["open_above"] and not clustered["open_below"], clustered
    if name == "threshold_blocks_only":
        assert replay["open_below"] and not replay["open_above"], replay
        assert workspace is None and conditions["act_spread"] is None
        assert conditions["act_clustered"]["pooled_fit"]["blocks"] == ["T1a"]
    else:
        assert not replay["open_below"] and not replay["open_above"], replay
        assert workspace["n_pairs"] == (26 if name == "held_out_placements" else 20)
    if name == "base":
        assert conditions["act_spread"]["block_fit"]["blocks"] == ["T3"]
    elif conditions["act_spread"] is not None:
        assert conditions["act_spread"]["block_fit"] is None
        assert conditions["act_spread"]["pooled_fit"]["blocks"] == ["T2"]
        assert made["comparisons"]["spread_minus_clustered"] is None
    assert made["distance_scale"]["applied"] is (name != "no_spread_threshold")
    assert ("by_region" in (workspace or {})) is (name == "held_out_placements")
    if name == "held_out_placements":
        regions = [pair["region"] for pair in workspace["pairs"]]
        assert regions.count("trained_cells") == 13 and regions.count("held_out_cells") == 13
    assert len(conditions["act_clustered"]["trials"]) == 34 + (workspace["n_pairs"] if workspace else 0)
    assert len(conditions["replay"]["trials"]) == 22


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, changes in CASES.items():
        experiment = Experiment(name, changes)
        experiment.run()
        made = results(experiment)
        check(name, made)
        path = OUT / f"{name}.json"
        path.write_text(dumps(made) + "\n", encoding="utf-8")
        fits = {c: (e["block_fit"] or {}).get("r50_median_mm") for c, e in made["conditions"].items() if e}
        print(f"{path.relative_to(REPO)}: {path.stat().st_size} bytes; block-fit radii {fits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
