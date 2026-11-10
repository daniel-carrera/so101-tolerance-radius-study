"""The analysis on a whole synthetic experiment: it runs, it finds the truths it was given, its results have the shape
the contract fixes, and what the data do not support comes out empty.

The experiment is written by `fixtures_experiment`: audits from the committed layouts, a frozen copy of the protocol,
trials planned by the real planner and answered from chosen success curves. NOTHING IN IT WAS MEASURED.
"""

import copy
import csv
import dataclasses
import json
import re
import statistics
import tempfile
import tomllib
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

import fixtures_experiment as fixtures
from tolerance_radius import report, schema
from tolerance_radius.geometry import Grid
from tolerance_radius.intervals import mcnemar_exact, wilson_interval
from tolerance_radius.psychometric import fit_threshold, grids_from_protocol
from tolerance_radius.report import bookkeeping, build, estimates, inputs, macros, markdown
from tolerance_radius.report.inputs import ReportError
from tolerance_radius.schema import Point

CONTRACT = fixtures.REPO / "protocol" / "data-contracts.md"
EXAMPLE = fixtures.REPO / "analysis" / "report-example"
CONDITIONS = ("act_clustered", "act_spread", "replay")
POLICIES = ("act_clustered", "act_spread")
ALL_BLOCKS = fixtures.ALL_BLOCKS
THRESHOLD_BLOCK = {"act_clustered": "T1a", "replay": "T1b", "act_spread": "T3"}

#: Parts of the results whose keys are names from the data (blocks, conditions, settings): every value has the shape
#: of the contract's one example.
MAPS = {
    "blocks",
    "blocks.*.by_condition",
    "conduct.execution.by_condition",
    "conduct.execution.per_trial",
    "conduct.start_pose_deg.joints",
    "conduct.checkpoints.by_condition",
    "conduct.checkpoints.runs",
    "simulation.blocks",
}
#: Parts whose content differs from one entry to the next and is not compared.
OPAQUE = {
    "conduct.execution.by_condition.*",
    "conduct.execution.common",
    "conduct.code.versions",
    "predictions.numbers",
}
#: Where the contract writes "the same as" another part, and which part. In this order: the inner parts first.
SAME_AS = {
    "conditions.act_clustered.descriptive.by_region.held_out_cells": (
        "conditions.act_clustered.descriptive.by_region.trained_cells"
    ),
    "conduct.execution.achieved_hz.replay": "conduct.execution.achieved_hz.policy",
    "conduct.execution.achieved_hz.all": "conduct.execution.achieved_hz.policy",
    "simulation.blocks.T1a.by_width.middling": "simulation.blocks.T1a.by_width.cliff",
    "simulation.blocks.T1a.by_width.slope": "simulation.blocks.T1a.by_width.cliff",
    "conditions.act_clustered.pooled_fit": "conditions.act_clustered.block_fit",
    "conditions.replay": "conditions.act_clustered",
    "conditions.act_spread": "conditions.act_clustered",
    "comparisons.spread_minus_clustered": "comparisons.clustered_minus_replay",
    "datasets.spread": "datasets.clustered",
    "second_scoring.by_kind.policy": "second_scoring.all",
    "second_scoring.by_kind.replay": "second_scoring.all",
}


def contract_blocks() -> tuple[dict, dict]:
    """The two blocks of the contract's section on `results.json`: the fields it fixes and the fields added."""
    text = CONTRACT.read_text(encoding="utf-8")
    section = text[text.index("## The analysis results") :]
    fixed, added = re.findall(r"```\n(\{.*?\n\})\n```", section, flags=re.S)
    return json.loads(fixed), json.loads(added)


def merged(first, second):
    """Two nested examples laid over each other."""
    if isinstance(first, dict) and isinstance(second, dict):
        return {
            key: merged(first[key], second[key])
            if key in first and key in second
            else copy.deepcopy(first.get(key, second.get(key)))
            for key in [*first, *(key for key in second if key not in first)]
        }
    if isinstance(first, list) and isinstance(second, list) and first and second and isinstance(first[0], dict):
        return [merged(first[0], second[0])]
    return copy.deepcopy(first)


def at(tree, path: str):
    for key in path.split("."):
        tree = tree[key]
    return tree


def template(*, with_additions: bool = True) -> dict:
    """The shape `results.json` must have, written out in full: the contract's fields and, unless left out, the
    additions the contract documents."""
    fixed, added = contract_blocks()
    shape = merged(fixed, added) if with_additions else fixed
    for path, source in SAME_AS.items():
        *parents, last = path.split(".")
        try:
            at(shape, ".".join(parents))[last] = copy.deepcopy(at(shape, source))
        except KeyError:  # a part the additions bring, in a shape made without them
            assert not with_additions, f"the contract no longer has {path} or {source}"
    return shape


def close(first: float, second: float, relative: float = 6e-9) -> bool:
    """True for two numbers that agree to the nine significant digits the results keep: rounding to nine digits moves
    a number by at most half a unit of the ninth, which is 5 parts in a thousand million of a number that starts
    with 1."""
    return abs(first - second) <= relative * max(abs(first), abs(second)) + 1e-12


def same_kind(example, value) -> bool:
    """True when a value is of the kind the contract's example shows. Null is allowed wherever a value is."""
    if example is None or value is None:
        return True
    if isinstance(example, bool) or isinstance(value, bool):
        return isinstance(example, bool) and isinstance(value, bool)
    if isinstance(example, (int, float)):
        return isinstance(value, (int, float))
    return type(example) is type(value)


def differences(example, value, path: str = "") -> list[str]:
    """Where a value does not have the shape of the contract's example: fields missing, unknown, or of another kind."""
    if path in OPAQUE or value is None or example is None:
        return []
    if not same_kind(example, value):
        return [f"{path}: {type(value).__name__} where the contract shows {type(example).__name__}"]
    found: list[str] = []
    if isinstance(example, dict):
        if path in MAPS:
            one = next(iter(example.values()))
            for item in value.values():
                found += differences(one, item, f"{path}.*")
            return found
        found += [f"{path}.{key}: in the contract and not in the results" for key in example if key not in value]
        found += [f"{path}.{key}: in the results and not in the contract" for key in value if key not in example]
        for key in example:
            if key in value:
                found += differences(example[key], value[key], f"{path}.{key}".lstrip("."))
    elif isinstance(example, list) and example:
        for item in value:
            found += differences(example[0], item, path)
    return found


def null_paths(value, path: str = "") -> set[str]:
    """Every place of the results that holds null, without the positions in lists."""
    if value is None:
        return {path}
    if isinstance(value, dict):
        return set().union(*(null_paths(item, f"{path}.{key}".lstrip(".")) for key, item in value.items()), set())
    if isinstance(value, list):
        return set().union(*(null_paths(item, path) for item in value), set())
    return set()


class WholeExperimentTests(unittest.TestCase):
    """One full analysis on the protocol's own grid, looked at from every side."""

    @classmethod
    def setUpClass(cls):
        cls.experiment, cls.made = fixtures.main_run()
        cls.results = cls.made.results
        cls.grid = Grid.from_protocol(cls.experiment.protocol)

    def rows(self, condition: str, block: str | None = None) -> list[dict]:
        return [t for t in self.results["conditions"][condition]["trials"] if block is None or t["block"] == block]

    # -------------------------------------------------------------------------------------------------- shape

    def test_the_results_have_every_field_of_the_contract_and_no_field_it_does_not_document(self):
        self.assertEqual(differences(template(), json.loads(self.made.results_json)), [])

    def test_the_contract_s_own_fields_are_all_there_by_name(self):
        results = json.loads(self.made.results_json)
        found = differences(template(with_additions=False), results)
        self.assertEqual([d for d in found if "in the contract and not in the results" in d], [])
        self.assertEqual([d for d in found if " where the contract shows " in d], [])
        for name in (
            "schema_version",
            "draft",
            "provenance",
            "distance_scale",
            "bin_width_mm",
            "interval_level",
            "conditions",
            "comparisons",
            "workspace",
            "reset_sensitivity",
            "demonstrations",
        ):
            self.assertIn(name, results)

    def test_only_what_cannot_be_is_null_when_every_block_was_run(self):
        expected = {
            "conditions.act_clustered.replay_episodes",
            "conditions.act_spread.replay_episodes",
            "conditions.replay.pooled_fit",
            "conditions.replay.descriptive.hull",
            "conditions.replay.trials.in_hull",
            "conditions.replay.trials.pair_id",
            "conditions.replay.trials.checkpoint",
            "conditions.act_clustered.trials.achieved_hz",  # the one trial that sent a single step
            "conditions.act_clustered.trials.failure",
            "conditions.act_clustered.trials.level_mm",
            "conditions.act_clustered.trials.pair_id",
            "conditions.act_clustered.trials.replay_episode",
            "conditions.act_spread.trials.failure",
            "conditions.act_spread.trials.level_mm",
            "conditions.act_spread.trials.pair_id",
            "conditions.act_spread.trials.replay_episode",
            "conditions.replay.trials.failure",
            "datasets.clustered.spacing_min_mm",
            "datasets.spread.scatter_max_mm",
            "predictions.held",  # prediction 5 is not a prediction
            "predictions.why",  # a prediction judged by a stated rule needs no remark
            "scene.basket_start_bins.up_to_mm",  # the last bin has no upper end
            "second_scoring.why_not",
            "simulation.why_no_case",
        }
        self.assertEqual(null_paths(self.results), expected)

    def test_it_is_json_with_the_banner_first_and_sorted_keys(self):
        text = self.made.results_json
        first, second = text.splitlines()[:2]
        self.assertTrue(first.startswith('{"banner": "SYNTHETIC'), first)
        self.assertEqual(second, '  "bin_width_mm": 10.0,')
        self.assertEqual(json.loads(text), self.results)
        self.assertTrue(text.endswith("}\n"))
        rest = {key: value for key, value in self.results.items() if key != "banner"}
        self.assertIn(json.dumps(rest, indent=2, sort_keys=True, ensure_ascii=False)[1:], text)

    def test_a_number_is_a_number_and_holds_nine_significant_digits(self):
        def numbers(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from numbers(item)
            elif isinstance(value, list):
                for item in value:
                    yield from numbers(item)
            elif isinstance(value, float):
                yield value

        found = list(numbers(self.results))
        self.assertGreater(len(found), 3000)
        self.assertTrue(all(np.isfinite(found)))
        self.assertTrue(all(value == estimates.stable(value) for value in found))

    # ------------------------------------------------------------------------------------------- provenance

    def test_the_provenance_says_what_was_read(self):
        source = self.results["provenance"]
        records = self.experiment.records
        self.assertEqual((source["protocol_version"], source["protocol_status"]), ("1.0", "frozen"))
        self.assertEqual(len(source["protocol_sha256"]), 64)
        self.assertEqual(source["logs"], ["data/trials/main.jsonl"])
        self.assertEqual(source["pilot_logs_not_read"], ["data/trials/pilot_replay.jsonl"])
        self.assertEqual(source["n_records"], len(records))
        self.assertEqual(source["n_scored"], sum(r.scored for r in records))
        self.assertEqual(source["n_void"], sum(not r.scored for r in records))
        self.assertEqual((source["project_commit"], source["project_commits"]), ("0123abc456", ["0123abc456"]))
        self.assertTrue(self.results["draft"])
        self.assertEqual(self.results["schema_version"], 1)
        self.assertEqual((self.results["bin_width_mm"], self.results["interval_level"]), (10.0, 0.95))

    def test_every_output_says_on_its_first_line_that_it_is_synthetic(self):
        self.assertIn("SYNTHETIC", self.made.results_json.splitlines()[0])
        self.assertIn("SYNTHETIC", self.made.report_md.splitlines()[0])
        self.assertIn("DRAFT", self.made.report_md.splitlines()[0])
        self.assertEqual(self.results["provenance"]["note"], fixtures.SYNTHETIC_NOTE)
        self.assertTrue(self.results["banner"].startswith(fixtures.SYNTHETIC_NOTE))

    # ------------------------------------------------------------------------------------------------ trials

    def test_every_scored_trial_is_listed_once_under_its_condition(self):
        for condition in CONDITIONS:
            expected = [r.planned.trial_id for r in self.experiment.scored(condition=condition)]
            section = self.results["conditions"][condition]
            with self.subTest(condition=condition):
                self.assertEqual([t["trial_id"] for t in section["trials"]], expected)
                self.assertEqual(section["n_scored"], len(expected))
                self.assertEqual(section["n_void"], len(self.experiment.void(condition=condition)))
                self.assertEqual(section["threshold_block"], THRESHOLD_BLOCK[condition])

    def test_a_trial_s_distance_is_the_recorded_one_divided_by_the_scale_factor(self):
        factor = self.results["distance_scale"]["factor"]
        self.assertAlmostEqual(factor, self.experiment.divisor, places=9)
        by_id = {t["trial_id"]: t for condition in CONDITIONS for t in self.rows(condition)}
        for record in self.experiment.scored():
            row = by_id[record.planned.trial_id]
            self.assertEqual(row["d_mm"], estimates.stable(record.nearest_demo_distance_mm / factor))
            self.assertEqual(
                (row["x_mm"], row["y_mm"]), (record.placed.x_mm, record.placed.y_mm), "positions are as read"
            )
            self.assertEqual(row["level_mm"], record.planned.level_mm, "the planner's request is as asked")
            self.assertEqual(row["placement_error_mm"], estimates.stable(record.placement_error_mm / factor))
            self.assertEqual(
                (row["success"], row["subgoal"], row["failure"]), (record.success, record.subgoal, record.failure)
            )
            self.assertEqual(row["nearest_demo_episode"], record.nearest_demo_episode)
            self.assertEqual(row["cell"], self.grid.cell_of(record.placed))

    def test_a_trial_s_part_is_its_place_in_the_block(self):
        parts = Counter((t["block"], t["part"]) for condition in CONDITIONS for t in self.rows(condition))
        blocks = self.experiment.protocol["blocks"]
        self.assertEqual(parts[("T1a", "zero")], blocks["T1a"]["anchors_zero"])
        self.assertEqual(parts[("T1a", "far")], blocks["T1a"]["anchors_far"])
        self.assertEqual(parts[("T1b", "zero")], blocks["T1b"]["anchors_zero"])
        for block in ("T1a", "T1b", "T3"):
            for staircase in blocks[block]["staircases"]:
                self.assertEqual(parts[(block, f"{block}-{staircase['id']}")], staircase["trials"])
        self.assertEqual(parts[("T2", "pair")], len(self.experiment.scored(block="T2")))
        for condition in CONDITIONS:
            for trial in self.rows(condition):
                self.assertEqual(trial["pair_id"] is not None, trial["part"] == "pair")
                self.assertEqual(trial["replay_episode"] is not None, condition == "replay")

    def test_the_counts_by_block_are_those_of_the_log(self):
        blocks = self.results["blocks"]
        self.assertEqual(list(blocks), list(self.experiment.protocol["blocks"]))
        for name, entry in blocks.items():
            with self.subTest(block=name):
                self.assertEqual(entry["n_scored"], len(self.experiment.scored(block=name)))
                self.assertEqual(entry["n_void"], len(self.experiment.void(block=name)))
                self.assertEqual(entry["kind"], "paired" if name == "T2" else "threshold")
                for condition, counts in entry["by_condition"].items():
                    self.assertEqual(counts["n_scored"], len(self.experiment.scored(name, condition)))
                    self.assertEqual(counts["n_void"], len(self.experiment.void(name, condition)))
        stated = self.experiment.protocol["blocks"]
        self.assertEqual(
            {name: blocks[name]["planned"] for name in blocks},
            {name: fixtures.planned_total(stated[name]) for name in stated},
        )
        self.assertEqual(blocks["T2"]["planned"], 2 * stated["T2"]["pairs"])

    def test_the_anchors_are_counted_in_the_threshold_block(self):
        for condition in CONDITIONS:
            block = THRESHOLD_BLOCK[condition]
            anchors = self.results["conditions"][condition]["anchors"]
            for part in ("zero", "far"):
                found = [t for t in self.rows(condition, block) if t["part"] == part]
                self.assertEqual(anchors[part], {"n": len(found), "k": sum(t["success"] for t in found)})

    def test_the_bins_hold_every_trial_once_with_its_wilson_interval(self):
        for condition in CONDITIONS:
            section = self.results["conditions"][condition]
            trials = section["trials"]
            self.assertEqual(sum(entry["n"] for entry in section["bins"]), len(trials))
            for entry in section["bins"]:
                inside = [t for t in trials if entry["lo_mm"] <= t["d_mm"] < entry["hi_mm"]]
                self.assertEqual((entry["n"], entry["k"]), (len(inside), sum(t["success"] for t in inside)))
                self.assertEqual(entry["hi_mm"] - entry["lo_mm"], 10.0)
                low, high = wilson_interval(entry["k"], entry["n"], 0.95)
                self.assertAlmostEqual(entry["wilson_lo"], low, places=8)
                self.assertAlmostEqual(entry["wilson_hi"], high, places=8)
            succeeded = [t["d_mm"] for t in trials if t["success"]]
            failed = [t["d_mm"] for t in trials if not t["success"]]
            self.assertEqual((section["max_success_d_mm"], section["min_failure_d_mm"]), (max(succeeded), min(failed)))

    def test_a_fit_carries_the_counts_of_its_own_trials(self):
        for condition in CONDITIONS:
            section = self.results["conditions"][condition]
            for kind, block in (("block_fit", THRESHOLD_BLOCK[condition]), ("pooled_fit", None)):
                fit = section[kind]
                if fit is None:
                    continue
                trials = self.rows(condition, block)
                zero = [t for t in trials if t["part"] == "zero"]
                with self.subTest(condition=condition, fit=kind):
                    self.assertEqual(fit["n"], len(trials))
                    self.assertEqual(fit["max_success_d_mm"], max(t["d_mm"] for t in trials if t["success"]))
                    self.assertEqual(fit["min_failure_d_mm"], min(t["d_mm"] for t in trials if not t["success"]))
                    self.assertEqual((fit["zero_k"], fit["zero_n"]), (sum(t["success"] for t in zero), len(zero)))
            self.assertEqual(
                (section["block_fit"]["zero_k"], section["block_fit"]["zero_n"]),
                (section["anchors"]["zero"]["k"], section["anchors"]["zero"]["n"]),
            )
        for policy in POLICIES:  # a pooled fit holds every scored trial of its policy, the block's among them
            section = self.results["conditions"][policy]
            block, pooled = section["block_fit"], section["pooled_fit"]
            self.assertLessEqual(pooled["min_failure_d_mm"], block["min_failure_d_mm"])
            self.assertGreaterEqual(pooled["max_success_d_mm"], block["max_success_d_mm"])
            self.assertEqual(
                (section["max_success_d_mm"], section["min_failure_d_mm"]),
                (pooled["max_success_d_mm"], pooled["min_failure_d_mm"]),
            )

    def test_the_trials_are_split_by_the_region_of_their_placement_in_the_same_bins(self):
        for condition in CONDITIONS:
            section = self.results["conditions"][condition]
            split = section["descriptive"]["by_region"]
            self.assertEqual(list(split), ["trained_cells", "held_out_cells"])
            self.assertEqual(section["descriptive"]["n_off_grid"], 0)
            for trial in section["trials"]:
                cell = self.grid.cell_of(Point(trial["x_mm"], trial["y_mm"]))
                self.assertEqual(
                    trial["region"], "trained_cells" if self.grid.is_trained(cell) else "held_out_cells"
                )
            for region, entry in split.items():
                inside = [t for t in section["trials"] if t["region"] == region]
                with self.subTest(condition=condition, region=region):
                    self.assertEqual((entry["n"], entry["k"]), (len(inside), sum(t["success"] for t in inside)))
                    self.assertEqual(sum(found["n"] for found in entry["bins"]), len(inside))
                    for found in entry["bins"]:
                        here = [t for t in inside if found["lo_mm"] <= t["d_mm"] < found["hi_mm"]]
                        self.assertEqual((found["n"], found["k"]), (len(here), sum(t["success"] for t in here)))
                        self.assertEqual(found["hi_mm"] - found["lo_mm"], self.results["bin_width_mm"])
                        low, high = wilson_interval(found["k"], found["n"], 0.95)
                        self.assertAlmostEqual(found["wilson_lo"], low, places=8)
                        self.assertAlmostEqual(found["wilson_hi"], high, places=8)
            for whole in section["bins"]:
                parts = [f for entry in split.values() for f in entry["bins"] if f["lo_mm"] == whole["lo_mm"]]
                self.assertEqual(
                    (sum(f["n"] for f in parts), sum(f["k"] for f in parts)), (whole["n"], whole["k"])
                )
        clustered = self.results["conditions"]["act_clustered"]["descriptive"]["by_region"]
        self.assertGreater(clustered["held_out_cells"]["n"], 0)
        self.assertGreater(clustered["trained_cells"]["n"], 0)

    # -------------------------------------------------------------------------------------------------- fits

    def posterior(self, condition: str, block: str | None = None, lapse_values=None):
        """The fit of a condition's trials as the results list them, made here with the model's own function."""
        rows = self.rows(condition, block)
        grids = grids_from_protocol(self.experiment.protocol["analysis"])
        if lapse_values is not None:
            grids = grids._replace(lapse_values=np.array(lapse_values))
        return fit_threshold([t["d_mm"] for t in rows], [t["success"] for t in rows], **grids._asdict())

    def test_every_fit_is_made_a_second_time_with_the_longer_list_of_lapse_values(self):
        analysis = self.experiment.protocol["analysis"]
        longer, limit = analysis["lapse_values_sensitivity"], analysis["sensitivity_limit_mm"]
        self.assertGreater(len(longer), len(analysis["lapse_values"]))
        checked = 0
        for condition in CONDITIONS:
            for kind, block in (("block_fit", THRESHOLD_BLOCK[condition]), ("pooled_fit", None)):
                fit = self.results["conditions"][condition][kind]
                if fit is None:
                    continue
                check = fit["sensitivity"]
                told = self.posterior(condition, block, lapse_values=longer).summary(0.95)
                with self.subTest(condition=condition, fit=kind):
                    self.assertEqual(check["lapse_values"], longer)
                    self.assertEqual(
                        (check["r50_median_mm"], check["r50_lo_mm"], check["r50_hi_mm"]),
                        (told.r50_median_mm, told.r50_low_mm, told.r50_high_mm),
                    )
                    self.assertEqual(
                        (check["open_below"], check["open_above"]), (told.reaches_lower_edge, told.reaches_upper_edge)
                    )
                    self.assertAlmostEqual(check["lapse_mean"], told.lapse_mean, places=8)
                    self.assertAlmostEqual(
                        check["median_shift_mm"], told.r50_median_mm - fit["r50_median_mm"], places=9
                    )
                    self.assertEqual(check["limit_mm"], limit)
                    self.assertEqual(check["plateau_in_doubt"], abs(check["median_shift_mm"]) > limit)
                    main = self.posterior(condition, block).summary(0.95)
                    self.assertEqual(fit["r50_median_mm"], main.r50_median_mm, "the reported radius is the first fit")
                checked += 1
        self.assertEqual(checked, 5)

    def test_the_fits_land_near_the_truths_they_were_answered_from(self):
        for condition, truth in self.experiment.truths.items():
            fit = self.results["conditions"][condition]["block_fit"]
            with self.subTest(condition=condition):
                self.assertLess(abs(fit["r50_median_mm"] - truth.r50_mm), 6.0)
                self.assertLessEqual(fit["r50_lo_mm"], truth.r50_mm)
                self.assertGreaterEqual(fit["r50_hi_mm"], truth.r50_mm)
                self.assertFalse(fit["open_below"] or fit["open_above"])

    def test_without_the_scale_correction_the_radii_would_read_long(self):
        # The synthetic instrument reads 3 % long; the truths are in true millimetres. The correction is what brings
        # the trials back: the same trials at their recorded distances are all farther out.
        factor = self.results["distance_scale"]["factor"]
        self.assertGreater(factor, 1.02)
        by_id = {t["trial_id"]: t["d_mm"] for condition in CONDITIONS for t in self.rows(condition)}
        for record in self.experiment.scored():
            if record.nearest_demo_distance_mm > 1.0:
                self.assertLess(by_id[record.planned.trial_id], record.nearest_demo_distance_mm)

    def test_a_block_fit_is_the_fit_of_the_threshold_block_s_trials(self):
        for condition in CONDITIONS:
            block = THRESHOLD_BLOCK[condition]
            told = self.posterior(condition, block).summary(0.95)
            fit = self.results["conditions"][condition]["block_fit"]
            with self.subTest(condition=condition):
                self.assertEqual(fit["blocks"], [block])
                self.assertEqual((fit["n"], fit["k"]), (told.n_trials, told.n_success))
                self.assertEqual(
                    (fit["r50_median_mm"], fit["r50_lo_mm"], fit["r50_hi_mm"]),
                    (told.r50_median_mm, told.r50_low_mm, told.r50_high_mm),
                )
                self.assertEqual(
                    (fit["open_below"], fit["open_above"]), (told.reaches_lower_edge, told.reaches_upper_edge)
                )
                self.assertAlmostEqual(fit["width_median_mm"], told.width_median_mm, places=8)
                self.assertAlmostEqual(fit["lapse_mean"], told.lapse_mean, places=8)
                self.assertAlmostEqual(fit["floor_mean"], told.floor_mean, places=8)

    def test_a_pooled_fit_is_the_fit_of_all_of_a_policy_s_trials_and_the_replay_has_none(self):
        for condition, blocks in (("act_clustered", ["T1a", "T2"]), ("act_spread", ["T2", "T3"])):
            told = self.posterior(condition).summary(0.95)
            fit = self.results["conditions"][condition]["pooled_fit"]
            self.assertEqual(fit["blocks"], blocks)
            self.assertEqual((fit["n"], fit["k"]), (told.n_trials, told.n_success))
            self.assertEqual(
                (fit["r50_median_mm"], fit["r50_lo_mm"], fit["r50_hi_mm"]),
                (told.r50_median_mm, told.r50_low_mm, told.r50_high_mm),
            )
            self.assertEqual(fit["n"], len(self.rows(condition)))
        self.assertIsNone(self.results["conditions"]["replay"]["pooled_fit"])

    def test_a_curve_is_the_posterior_mean_with_its_band_from_zero_past_every_trial(self):
        posterior = self.posterior("replay", "T1b")
        curve = self.results["conditions"]["replay"]["block_fit"]["curve"]
        d = np.array(curve["d_mm"])
        self.assertEqual((d[0], d[1]), (0.0, 0.5))
        self.assertGreaterEqual(d[-1], max(t["d_mm"] for condition in CONDITIONS for t in self.rows(condition)))
        at_some = [0, 20, 24, 40, 160]
        self.assertTrue(np.allclose(np.array(curve["mean"])[at_some], posterior.predict(d[at_some]), rtol=0, atol=1e-8))
        low, high = posterior.band(d[at_some], 0.95)
        self.assertTrue(np.allclose(np.array(curve["lo"])[at_some], low, rtol=0, atol=1e-8))
        self.assertTrue(np.allclose(np.array(curve["hi"])[at_some], high, rtol=0, atol=1e-8))
        for condition in CONDITIONS:
            for kind in ("block_fit", "pooled_fit"):
                fit = self.results["conditions"][condition][kind]
                if fit is not None:
                    self.assertEqual({len(fit["curve"][key]) for key in ("d_mm", "mean", "lo", "hi")}, {d.size})

    # ------------------------------------------------------------------------------------------- comparisons

    def test_the_comparisons_use_the_block_fits(self):
        for name, first, second in (
            ("clustered_minus_replay", "act_clustered", "replay"),
            ("spread_minus_clustered", "act_spread", "act_clustered"),
        ):
            expected = estimates.radius_difference(
                self.posterior(first, THRESHOLD_BLOCK[first]), self.posterior(second, THRESHOLD_BLOCK[second]), 0.95
            )
            found = self.results["comparisons"][name]
            with self.subTest(comparison=name):
                self.assertEqual(found["fits"], "block")
                self.assertEqual(
                    (found["median_mm"], found["lo_mm"], found["hi_mm"]),
                    (expected["median_mm"], expected["lo_mm"], expected["hi_mm"]),
                )
                self.assertAlmostEqual(found["prob_greater"], expected["prob_greater"], places=8)
                self.assertEqual(found["verdict"], estimates.verdict(found["prob_greater"]))
        self.assertEqual(self.results["comparisons"]["clustered_minus_replay"]["verdict"], "larger")

    # --------------------------------------------------------------------------------------------- workspace

    def test_the_workspace_test_pairs_the_two_policies_at_every_placement(self):
        workspace = self.results["workspace"]
        scored = self.experiment.scored(block="T2")
        by_pair: dict[str, dict[str, object]] = {}
        for record in scored:
            by_pair.setdefault(record.planned.pair_id, {})[record.planned.condition] = record
        planned = self.experiment.protocol["blocks"]["T2"]["pairs"]
        self.assertEqual(workspace["n_pairs"], planned + self.experiment.held_out_pairs)
        self.assertEqual(workspace["n_pairs"], len(by_pair))
        self.assertEqual(workspace["n_unpaired"], 0)
        self.assertEqual([pair["pair_id"] for pair in workspace["pairs"]], sorted(by_pair))
        table = Counter()
        for pair in workspace["pairs"]:
            records = by_pair[pair["pair_id"]]
            for policy in POLICIES:
                self.assertEqual(pair[policy]["trial_id"], records[policy].planned.trial_id)
                self.assertEqual(pair[policy]["success"], records[policy].success)
                self.assertEqual(
                    (pair[policy]["x_mm"], pair[policy]["y_mm"]),
                    (records[policy].placed.x_mm, records[policy].placed.y_mm),
                )
                self.assertEqual(pair[policy]["checkpoint"], records[policy].checkpoint)
            self.assertAlmostEqual(
                pair["x_mm"], (pair["act_clustered"]["x_mm"] + pair["act_spread"]["x_mm"]) / 2, places=9
            )
            self.assertAlmostEqual(
                pair["y_mm"], (pair["act_clustered"]["y_mm"] + pair["act_spread"]["y_mm"]) / 2, places=9
            )
            table[(pair["act_clustered"]["success"], pair["act_spread"]["success"])] += 1
        self.assertEqual(
            workspace["table"],
            {
                "both": table[(True, True)],
                "neither": table[(False, False)],
                "only_spread": table[(False, True)],
                "only_clustered": table[(True, False)],
            },
        )
        self.assertEqual(sum(workspace["table"].values()), workspace["n_pairs"])

    def test_the_paired_test_is_the_exact_mcnemar_test_over_all_pairs_together(self):
        workspace = self.results["workspace"]
        table = workspace["table"]
        self.assertAlmostEqual(
            workspace["mcnemar_p"], mcnemar_exact(table["only_spread"], table["only_clustered"]), places=9
        )
        for policy in POLICIES:
            k = sum(pair[policy]["success"] for pair in workspace["pairs"])
            low, high = wilson_interval(k, workspace["n_pairs"], 0.95)
            success = workspace["success"][policy]
            self.assertEqual((success["k"], success["n"]), (k, workspace["n_pairs"]))
            self.assertAlmostEqual(success["wilson_lo"], low, places=8)
            self.assertAlmostEqual(success["wilson_hi"], high, places=8)
            distances = [pair[policy]["d_mm"] for pair in workspace["pairs"]]
            self.assertTrue(close(workspace["distances"][policy]["median_mm"], float(np.median(distances))))
            self.assertEqual(workspace["distances"][policy]["max_mm"], max(distances))

    def test_a_pair_s_region_is_the_cell_of_its_measured_position_and_both_regions_have_pairs(self):
        workspace = self.results["workspace"]
        for pair in workspace["pairs"]:
            cell = self.grid.cell_of(Point(pair["x_mm"], pair["y_mm"]))
            self.assertEqual(pair["region"], "trained_cells" if self.grid.is_trained(cell) else "held_out_cells")
        regions = Counter(pair["region"] for pair in workspace["pairs"])
        self.assertEqual(set(regions), {"trained_cells", "held_out_cells"})
        self.assertEqual(set(workspace["by_region"]), {"trained_cells", "held_out_cells"})
        for region, entry in workspace["by_region"].items():
            inside = [pair for pair in workspace["pairs"] if pair["region"] == region]
            self.assertEqual(entry["n_pairs"], regions[region])
            for policy in POLICIES:
                k = sum(pair[policy]["success"] for pair in inside)
                low, high = wilson_interval(k, len(inside), 0.95)
                self.assertEqual((entry[policy]["k"], entry[policy]["n"]), (k, len(inside)))
                self.assertAlmostEqual(entry[policy]["wilson_lo"], low, places=8)
                self.assertAlmostEqual(entry[policy]["wilson_hi"], high, places=8)
                expected = sum(pair[policy]["p_lookup"] for pair in inside)
                self.assertTrue(close(entry[policy]["lookup_expected"], expected, 1e-7))
                counts = self.posterior("act_clustered", "T1a").count_distribution(
                    [pair[policy]["d_mm"] for pair in inside]
                )
                self.assertEqual(
                    (entry[policy]["lookup_lo"], entry[policy]["lookup_hi"]), estimates.count_interval(counts, 0.95)
                )
                self.assertLessEqual(entry[policy]["lookup_lo"], entry[policy]["lookup_expected"])
                self.assertGreaterEqual(entry[policy]["lookup_hi"], entry[policy]["lookup_expected"])
        for policy in POLICIES:
            total = sum(workspace["by_region"][region][policy]["lookup_expected"] for region in workspace["by_region"])
            self.assertTrue(close(total, workspace["lookup"][policy]["expected"], 1e-7))

    def test_the_block_is_the_stratified_one_and_its_regions_are_those_of_the_placed_cube(self):
        stated = self.experiment.protocol["blocks"]["T2"]
        self.assertEqual([(entry["region"], entry["pairs"]) for entry in stated["regions"]], list(fixtures.STRATIFIED))
        self.assertNotIn("region", stated)
        self.assertEqual(stated["pairs"], 26)
        workspace = self.results["workspace"]
        self.assertEqual(workspace["n_pairs"], 26)
        self.assertEqual(self.results["blocks"]["T2"]["planned"], 52)
        self.assertEqual(len(workspace["lookup"]["act_clustered"]["pmf"]), 27)
        self.assertEqual(sum(workspace["by_region"][region]["n_pairs"] for region in workspace["by_region"]), 26)
        # Where each pair was planned, by its target: the results never look at that. They agree here unless a
        # placement error carried a cube across a border, and then the results follow the cube.
        by_pair = {r.planned.pair_id: r for r in self.experiment.scored(block="T2")}
        for pair in workspace["pairs"]:
            planned = self.grid.is_trained(self.grid.cell_of(by_pair[pair["pair_id"]].planned.target))
            placed = self.grid.is_trained(self.grid.cell_of(Point(pair["x_mm"], pair["y_mm"])))
            self.assertEqual(pair["region"], "trained_cells" if placed else "held_out_cells")
            if planned != placed:
                target, cell = by_pair[pair["pair_id"]].planned.target, self.grid.cell_mm
                edge = min(target.x_mm % cell, target.y_mm % cell, -target.x_mm % cell, -target.y_mm % cell)
                tolerance = self.experiment.protocol["placement"]["tolerance_mm"]
                self.assertLess(edge, tolerance, "only a target that close to a border can change sides")

    def test_the_pairs_are_also_counted_by_the_training_runs_they_were_run_with(self):
        workspace = self.results["workspace"]
        self.assertEqual(len(workspace["by_run"]), 1, "one run of each layout: one pair of runs")
        entry = workspace["by_run"][0]
        self.assertEqual(
            entry["runs"], {"act_clustered": "act-tolrad-clustered-v1", "act_spread": "act-tolrad-spread-v1"}
        )
        self.assertEqual(entry["n_pairs"], workspace["n_pairs"])
        for policy in POLICIES:
            self.assertEqual(
                {key: entry[policy][key] for key in ("k", "n", "wilson_lo", "wilson_hi")}, workspace["success"][policy]
            )
            self.assertTrue(close(entry[policy]["lookup_expected"], workspace["lookup"][policy]["expected"], 1e-7))
            self.assertEqual(
                (entry[policy]["lookup_lo"], entry[policy]["lookup_hi"]),
                (workspace["lookup"][policy]["lo"], workspace["lookup"][policy]["hi"]),
                "one pair of runs: its interval is that of all the pairs",
            )

    def test_the_lookup_prediction_is_the_clustered_block_fit_at_each_trial_s_own_distance(self):
        workspace = self.results["workspace"]
        fit = self.posterior("act_clustered", "T1a")  # it has not seen the workspace trials
        self.assertEqual(workspace["lookup"]["fit"], "act_clustered block_fit")
        by_id = {t["trial_id"]: t for policy in POLICIES for t in self.rows(policy, "T2")}
        for policy in POLICIES:
            distances = []
            for pair in workspace["pairs"]:
                trial = by_id[pair[policy]["trial_id"]]
                self.assertEqual(pair[policy]["d_mm"], trial["d_mm"])
                self.assertAlmostEqual(pair[policy]["p_lookup"], float(fit.predict(trial["d_mm"])), places=8)
                distances.append(trial["d_mm"])
            found = workspace["lookup"][policy]
            counts = fit.count_distribution(distances)
            self.assertTrue(close(found["expected"], float(fit.predict(distances).sum())))
            self.assertEqual((found["lo"], found["hi"]), estimates.count_interval(counts, 0.95))
            self.assertTrue(np.allclose(found["pmf"], counts, rtol=1e-8, atol=1e-12))
            self.assertEqual(found["observed"], workspace["success"][policy]["k"])
            self.assertEqual(found["verdict"], estimates.lookup_verdict(found["observed"], found["lo"], found["hi"]))
            self.assertLess(found["lo"], found["expected"])
            self.assertLess(found["expected"], found["hi"])
        # The spread policy's trials are nearer its demonstrations, so the lookup expects more of it.
        self.assertGreater(
            workspace["lookup"]["act_spread"]["expected"], workspace["lookup"]["act_clustered"]["expected"]
        )

    # ------------------------------------------------------------------------------------- sensitivity, V0, V1

    def test_the_reset_sensitivity_averages_the_block_fit_over_the_two_kinds_of_error(self):
        sensitivity = self.results["reset_sensitivity"]
        fit = self.posterior("act_clustered", "T1a")
        factor = self.results["distance_scale"]["factor"]
        placed = [estimates.stable(r.placement_error_mm / factor) for r in self.experiment.scored()]
        by_eye = [estimates.stable(error / factor) for error in self.experiment.by_eye_errors_mm]
        self.assertEqual((sensitivity["n_placed"], sensitivity["n_by_eye"]), (len(placed), 20))
        self.assertEqual(sensitivity["fit"], "act_clustered block_fit")
        self.assertAlmostEqual(sensitivity["placed"], float(fit.predict(placed).mean()), places=8)
        self.assertAlmostEqual(sensitivity["by_eye"], float(fit.predict(by_eye).mean()), places=8)
        self.assertGreater(sensitivity["placed"], sensitivity["by_eye"])

    def test_the_scale_check_is_given_as_read_and_decides_the_correction(self):
        scale, check = self.results["distance_scale"], self.results["instrument"]["scale_check"]
        factors = self.experiment.scale_factors
        self.assertEqual((scale["applied"], scale["n_runs"]), (True, len(factors)))
        self.assertAlmostEqual(scale["factor"], sum(factors) / len(factors), places=9)
        self.assertAlmostEqual(scale["factor_min"], min(factors), places=9)
        self.assertAlmostEqual(scale["factor_max"], max(factors), places=9)
        self.assertEqual(scale["divisor"], scale["factor"])
        self.assertEqual((check["n_runs"], check["n_steps"], check["true_step_mm"]), (4, 20, 10.0))
        self.assertEqual(sorted(run["factor"] for run in check["runs"]), sorted(factors))
        errors = [10.0 * (factor - 1.0) for factor in factors for _ in range(5)]
        self.assertAlmostEqual(check["mean_abs_step_error_mm"], sum(errors) / len(errors), places=6)
        self.assertAlmostEqual(check["max_abs_step_error_mm"], max(errors), places=6)
        self.assertTrue(check["passes"])

    def test_the_by_eye_errors_and_the_repeatability_are_summarised(self):
        by_eye = self.results["instrument"]["by_eye"]
        factor = self.results["distance_scale"]["factor"]
        errors = np.array(self.experiment.by_eye_errors_mm) / factor
        self.assertEqual((by_eye["n"], by_eye["n_operators"], by_eye["operators"]), (20, 1, ["op1"]))
        self.assertTrue(close(by_eye["error_median_mm"], float(np.median(errors))))
        self.assertTrue(close(by_eye["error_p95_mm"], float(np.percentile(errors, 95))))
        self.assertTrue(close(by_eye["error_max_mm"], float(errors.max())))
        repeat = self.results["instrument"]["repeatability"]
        self.assertEqual((repeat["n"], repeat["n_sets"]), (10, 1))
        self.assertGreater(repeat["scatter_max_mm"], 0.0)
        self.assertLess(repeat["scatter_max_mm"], 0.5)

    # ----------------------------------------------------------------------------------- datasets and conduct

    def test_the_datasets_are_described_from_their_audits_and_placement_logs(self):
        factor = self.results["distance_scale"]["factor"]
        for name in ("clustered", "spread"):
            starts = fixtures.audited_starts(name)
            summary = self.results["datasets"][name]
            errors = [start.measured.distance_to(start.planned) / factor for start in starts]
            self.assertEqual((summary["n_episodes"], summary["n_trained"]), (len(starts), len(starts)))
            self.assertTrue(close(summary["placement_error_median_mm"], float(np.median(errors))))
            self.assertTrue(close(summary["placement_error_max_mm"], max(errors)))
            recording = summary["recording"]
            self.assertTrue(recording["matches_audit"])
            takes = [fixtures.takes_before_saving(start.episode_index) for start in starts]
            self.assertEqual(recording["n_kept"], len(starts))
            self.assertEqual(recording["n_rerecorded_episodes"], sum(count > 0 for count in takes))
            self.assertEqual(recording["n_rerecords"], sum(takes))
            self.assertGreater(recording["n_rerecords"], recording["n_rerecorded_episodes"])
            self.assertEqual(recording["days"], ["2026-10-06"])
            self.assertEqual(
                self.results["demonstrations"][name][3],
                {
                    "episode_index": 3,
                    "cell": starts[3].cell,
                    "x_mm": starts[3].measured.x_mm,
                    "y_mm": starts[3].measured.y_mm,
                    "trained": True,
                },
            )
            self.assertEqual(len(self.results["demonstrations"][name]), len(starts))
        clustered, spread = self.results["datasets"]["clustered"], self.results["datasets"]["spread"]
        self.assertEqual((clustered["recording"]["n_superseded"], spread["recording"]["n_superseded"]), (3, 0))
        self.assertEqual(clustered["recording"]["superseded_sessions"], [fixtures.SMOKE_SESSION])
        self.assertLess(clustered["scatter_max_mm"], 2.5)
        positions = np.array([[s.measured.x_mm, s.measured.y_mm] for s in fixtures.audited_starts("spread")])
        apart = np.hypot(*(positions[:, None, :] - positions[None, :, :]).transpose(2, 0, 1))
        self.assertTrue(close(spread["spacing_min_mm"], float(apart[apart > 0].min()) / factor))
        self.assertEqual(self.results["datasets"]["both"]["recording_days"], ["2026-10-06"])

    def test_the_void_trials_are_listed_with_their_reasons_and_counted_by_reason(self):
        conduct = self.results["conduct"]
        void = [record for record in self.experiment.records if not record.scored]
        self.assertEqual(
            [(t["trial_id"], t["reason"]) for t in conduct["void_trials"]],
            [(record.planned.trial_id, record.void_reason) for record in void],
        )
        self.assertEqual(sum(reason["n"] for reason in conduct["void_reasons"]), len(void))
        counts = [reason["n"] for reason in conduct["void_reasons"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        bumped = sum(record.void_reason.rstrip(".").casefold() == "bumped the table" for record in void)
        told = {reason["reason"]: reason["n"] for reason in conduct["void_reasons"]}
        self.assertEqual(told["bumped the table"], bumped, "capitals and a full stop do not make another reason")
        slips = [record.void_reason for record in void if record.void_reason.startswith("ended by a key")]
        self.assertGreater(len(set(slips)), 1, "the runner writes the time into the reason, so the reasons differ")
        self.assertEqual(told["ended by a key less than 1 s after it began"], len(slips), "and they are one reason")
        self.assertEqual(len(told), 4)
        self.assertEqual(self.results["provenance"]["n_void"], len(void), "a slip of the key is a void trial")

    def test_how_the_trials_were_run_comes_from_the_records(self):
        conduct = self.results["conduct"]
        self.assertEqual(conduct["trial_days"], ["2026-10-07", "2026-10-08"])
        self.assertEqual(conduct["people"], ["op1"])
        self.assertEqual(conduct["n_trial_intervals"], len(self.experiment.records) - 1)
        self.assertGreater(conduct["trial_minutes_median"], 1.5)
        self.assertLess(conduct["trial_minutes_median"], 3.5)
        stated = self.experiment.protocol["execution"]
        self.assertGreaterEqual(len(stated), 10)
        self.assertEqual(conduct["execution"]["common"], stated, "every setting of the protocol, in every condition")
        for condition in CONDITIONS:
            self.assertEqual(conduct["execution"]["by_condition"][condition], stated)
        for key in fixtures.PER_TRIAL_KEYS:
            self.assertNotIn(key, conduct["execution"]["common"], "what belongs to one trial is not a setting")
        self.assertEqual(
            conduct["execution"]["per_trial"],
            {
                "act_clustered": {"device": ["mps"], "handoff_tolerance_used_deg": [stated["handoff_tolerance_deg"]]},
                "act_spread": {"device": ["mps"], "handoff_tolerance_used_deg": [stated["handoff_tolerance_deg"]]},
                "replay": {"device": ["mps"], "handoff_tolerance_used_deg": [stated["replay_handoff_tolerance_deg"]]},
            },
        )
        poses = conduct["start_pose_deg"]
        self.assertEqual(poses["n"], len(self.experiment.scored()))
        self.assertEqual(sorted(poses["joints"]), sorted(fixtures.START_POSE_DEG))
        recorded = [r.execution["start_pose_deg"]["elbow_flex.pos"] for r in self.experiment.scored()]
        elbow = poses["joints"]["elbow_flex.pos"]
        self.assertEqual((elbow["min"], elbow["max"]), (min(recorded), max(recorded)))
        self.assertTrue(close(elbow["median"], float(np.median(recorded))))
        self.assertEqual(
            conduct["checkpoints"],
            {
                "by_condition": {
                    "act_clustered": ["act-tolrad-clustered-v1@080000"],
                    "act_spread": ["act-tolrad-spread-v1@080000"],
                },
                "runs": {"act_clustered": ["act-tolrad-clustered-v1"], "act_spread": ["act-tolrad-spread-v1"]},
                "step": "080000",
                "evaluation_step": 80000,
            },
        )
        self.assertEqual(conduct["code"]["lerobot_commit"], "6adf515116")
        self.assertEqual(self.results["deviations"], {"n": 0, "rows": []})
        errors = [t["placement_error_mm"] for condition in CONDITIONS for t in self.rows(condition)]
        self.assertTrue(close(conduct["reset_error_median_mm"], float(np.median(errors))))
        self.assertEqual(conduct["reset_error_max_mm"], max(errors))

    def test_the_camera_s_creep_before_an_attempt_is_summarised_as_the_sessions_read_it(self):
        crept = self.results["conduct"]["view_crept"]
        readings = [record.execution["view_crept_mm"] for record in self.experiment.scored()]
        self.assertEqual((crept["n"], crept["n_without"]), (len(readings), 0))
        self.assertTrue(close(crept["median_mm"], float(np.median(readings))))
        self.assertEqual(crept["max_mm"], max(readings), "as read: it is not divided by the scale factor")
        low, high = fixtures.VIEW_CREPT_MM
        self.assertTrue(low <= min(readings) <= crept["median_mm"] <= crept["max_mm"] <= high)
        limit = self.experiment.protocol["execution"]["view_crept_limit_mm"]
        self.assertEqual((crept["limit_mm"], crept["n_over_limit"]), (limit, 0))
        self.assertEqual(self.made.values["viewCreptMax"]["value"], f"{max(readings):.2f}")
        self.assertRegex(self.made.values["viewCreptMedian"]["value"], r"^0\.\d\d$")
        text = self.made.report_md
        self.assertIn(
            f"- **The camera's creep before an attempt:** {len(readings)} of the {len(readings)} scored trials "
            "record how far the front camera's picture stood",
            text,
        )
        self.assertIn(f"at most {max(readings):.2f} mm. These are as the sessions read them", text)
        self.assertIn(f"None lies beyond the protocol's limit of {limit:g} mm", text)

    def test_the_achieved_control_rate_is_the_steps_after_the_first_over_the_loop_s_own_duration(self):
        requested = self.experiment.protocol["execution"]["control_hz"]
        rates = self.results["conduct"]["execution"]["achieved_hz"]
        self.assertEqual(rates["shortfall_limit"], 0.02)
        by_id = {t["trial_id"]: t for condition in CONDITIONS for t in self.rows(condition)}
        by_kind: dict[str, list[float]] = {"policy": [], "replay": [], "all": []}
        without = Counter()
        for record in self.experiment.scored():
            row = by_id[record.planned.trial_id]
            steps, loop_s = record.execution["steps_sent"], record.execution["loop_s"]
            kinds = ("replay" if record.planned.condition == "replay" else "policy", "all")
            self.assertEqual((row["steps_sent"], row["loop_s"]), (steps, loop_s))
            if steps < 2:
                self.assertEqual((loop_s, row["achieved_hz"]), (0.0, None))
                without.update(kinds)
                continue
            self.assertTrue(close(row["achieved_hz"], (steps - 1) / loop_s))
            # The synthetic loop runs a little under what it is asked for: the analysis finds that, trial by trial.
            asked_of_it = fixtures.achieved_hz(record.planned.trial_id, requested)
            self.assertAlmostEqual(row["achieved_hz"], asked_of_it, delta=0.003)
            self.assertLess(row["achieved_hz"], requested)
            for kind in kinds:
                by_kind[kind].append((steps - 1) / loop_s)
        self.assertEqual(dict(without), {"policy": 1, "all": 1}, "one scored trial sent a single step")
        self.assertEqual(by_id[fixtures.TOO_SHORT_TRIAL]["steps_sent"], 1)
        for kind, found in by_kind.items():
            with self.subTest(kind=kind):
                told = rates[kind]
                self.assertEqual((told["n"], told["n_without"]), (len(found), without[kind]))
                self.assertEqual(told["requested"], requested)
                self.assertTrue(close(told["median"], float(np.median(found))))
                self.assertTrue(close(told["min"], min(found)))
                self.assertTrue(close(told["max"], max(found)))
                self.assertIs(told["short_of_requested"], False, "within 2 % of what was asked")
                low, high = fixtures.ACHIEVED_SHARE
                self.assertGreater(told["min"], requested * low - 0.01)
                self.assertLess(told["max"], requested * high + 0.01)
        self.assertEqual(rates["all"]["n"], rates["policy"]["n"] + rates["replay"]["n"])
        self.assertEqual(
            self.made.values["execControlRateAchieved"]["value"], f"{float(np.median(by_kind['policy'])):.1f}"
        )
        text = self.made.report_md
        self.assertIn(f"  - policy trials: asked {requested:g} Hz; achieved a median of ", text)
        self.assertIn(f"over {len(by_kind['replay'])} trials).", text)
        self.assertIn("1 scored trial(s) have no rate", text)
        self.assertNotIn("below the rate they were asked for", text)

    def test_the_scene_at_each_trial_s_start_is_counted_from_the_audit_and_the_records(self):
        scene = self.results["scene"]
        lines = {line["trial_id"]: line for line in self.experiment.scene["trials"]}
        scored = [(record, lines[record.planned.trial_id]) for record in self.experiment.scored()]
        self.assertEqual(self.results["provenance"]["scene_audits"], ["data/audit/scene_main.json"])
        self.assertEqual(scene["files"], ["data/audit/scene_main.json"])
        self.assertEqual((scene["basket_named_mm"], scene["wrist_named_px"]), (2.0, 2.0))
        self.assertEqual((scene["n_scored"], scene["n_audited"], scene["not_audited"]), (len(scored), len(scored), []))

        def off(reading, keys):
            return None if reading is None else max(abs(reading[key]) for key in keys)

        basket = [off(line["basket_start"], ("across_mm", "down_mm")) for _, line in scored]
        wrist = [off(line["wrist_start"], ("across_px", "down_px")) for _, line in scored]
        self.assertEqual(scene["n_basket_read"], len(scored) - 1, "one front video could not be read")
        self.assertEqual(scene["n_wrist_read"], len(scored) - 1, "and one wrist video")
        at = [record.planned.trial_id for record, _ in scored]
        self.assertEqual(scene["basket_unread"], [at[fixtures.SCENE_FRONT_UNREAD_AT]])
        self.assertEqual(scene["wrist_unread"], [at[fixtures.SCENE_WRIST_UNREAD_AT]])
        self.assertEqual(scene["named_wrist"], [at[fixtures.SCENE_WRIST_NAMED_AT]])
        self.assertEqual(scene["n_named_wrist"], 1)
        self.assertEqual([entry["trial_id"] for entry in scene["basket_moved"]], [at[fixtures.SCENE_MOVED_AT]])
        moved = scene["basket_moved"][0]
        record = scored[fixtures.SCENE_MOVED_AT][0]
        self.assertEqual((moved["condition"], moved["outcome"]), (record.planned.condition, record.outcome))
        self.assertGreater(moved["moved_mm"], 30.0)
        self.assertEqual(scene["n_basket_moved"], 1)
        self.assertEqual(
            scene["moved_unknown"], [at[fixtures.SCENE_END_UNSEEN_AT], at[fixtures.SCENE_FRONT_UNREAD_AT]]
        )
        named = [record.planned.trial_id for (record, _), far in zip(scored, basket, strict=True) if far and far > 2.0]
        self.assertEqual(scene["named_basket"], named)
        self.assertEqual(scene["n_named_basket"], len(named))
        self.assertGreater(len(named), 10)
        read = [far for far in basket if far is not None]
        self.assertEqual(scene["basket_start_max_mm"], max(read))
        self.assertEqual(scene["named_basket_min_mm"], min(far for far in read if far > 2.0))
        self.assertEqual(scene["wrist_start_max_px"], max(far for far in wrist if far is not None))
        self.assertEqual(scene["wrist_start_max_px"], 2.4)
        bins = scene["basket_start_bins"]
        self.assertEqual(
            [(entry["over_mm"], entry["up_to_mm"]) for entry in bins], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, None)]
        )
        self.assertEqual(sum(entry["n"] for entry in bins), scene["n_basket_read"], "every trial read is in one bin")
        self.assertEqual(bins[0]["n"], sum(far <= 1.0 for far in read))
        self.assertEqual(bins[2]["n"] + bins[3]["n"], len(named), "beyond 2 mm is named")
        success = scene["success"]
        self.assertEqual(sum(group["n"] for group in success.values()), len(scored))
        self.assertEqual(sum(group["k"] for group in success.values()), sum(record.success for record, _ in scored))
        either = {*named, *scene["named_wrist"]}
        self.assertEqual(success["named"]["n"], len(either))
        self.assertEqual(
            success["named"]["k"], sum(record.success for record, _ in scored if record.planned.trial_id in either)
        )
        self.assertEqual(success["start_not_read"]["n"], 2)
        values = self.made.values
        self.assertEqual(values["nTrialsBasketNamed"]["value"], str(len(named)))
        self.assertEqual(values["nTrialsWristNamed"]["value"], "1")
        self.assertEqual(values["nTrialsBasketMoved"]["value"], "1")
        self.assertEqual(values["basketStartMax"]["value"], f"{max(read):.1f}")
        self.assertEqual(values["wristStartMax"]["value"], "2.4")

    def test_the_report_names_the_trials_that_began_with_the_scene_out_of_place(self):
        scene, text = self.results["scene"], self.made.report_md
        section = text[text.index("## The scene at each trial's start") : text.index("## How the trials were run")]
        self.assertIn("(protocol section 8). From `data/audit/scene_main.json`", section)
        self.assertIn(f"{scene['n_audited']} of the {scene['n_scored']} scored trials are in the audit.", section)
        self.assertIn(f"the front camera's of {scene['basket_unread'][0]}; the wrist camera's of", section)
        moved = scene["basket_moved"][0]
        self.assertIn("**The basket was moved in 1 trial(s)** (it stood more than 5 mm from where", section)
        self.assertIn(f"{moved['trial_id']} (the ", section)
        self.assertIn(f", {moved['outcome']}, {moved['moved_mm']:.1f} mm)", section)
        self.assertIn(
            f"- **Named for the basket:** {scene['n_named_basket']} trial(s) began with it more than 2 mm from its "
            f"place: {', '.join(scene['named_basket'])}.",
            section,
        )
        self.assertIn("- **Named for the wrist camera:** 1 trial(s) began with it more than 2 px", section)
        success = scene["success"]
        self.assertIn(
            f"**Successes, as plain counts and with no test:** {success['named']['k']} of the {success['named']['n']} "
            f"trials named for the basket or for the wrist camera succeeded, and {success['unnamed']['k']} of the "
            f"{success['unnamed']['n']} trials named for neither succeeded.",
            section,
        )
        self.assertNotRegex(section, r"(?i)p = |significan|test of|McNemar|interval", "counts only, no test")
        for entry, name in zip(scene["basket_start_bins"], ("up to 1 mm", "over 1 mm, up to 2 mm"), strict=False):
            self.assertIn(f"| {name} | {entry['n']} |", section)
        self.assertIn(f"| over 3 mm | {scene['basket_start_bins'][3]['n']} |", section)
        self.assertIn("One pixel of the front camera is about 1.08 mm at the basket (`MM_PER_PX` in", section)
        self.assertIn(
            "so the 2 mm line lies 1.9 pixels from the basket's place, and an offset of 2 whole pixels, about 2.2 mm, "
            "is the first pixel step past it.",
            section,
        )
        self.assertIn(f"began {scene['named_basket_min_mm']:.1f} mm to {scene['basket_start_max_mm']:.1f} mm", section)
        self.assertIn("**Scene audits:** `data/audit/scene_main.json`.", text)

    def test_the_second_scoring_is_given_as_its_agreement_file_has_it(self):
        second = self.results["second_scoring"]
        written = self.experiment.second_scoring
        self.assertEqual(self.results["provenance"]["second_scoring"], "data/second_scoring/main/agreement.json")
        self.assertEqual((second["file"], second["why_not"]), ("data/second_scoring/main/agreement.json", None))
        self.assertEqual((second["all"], second["by_kind"]), (written["all"], written["by_kind"]))
        whole = second["all"]
        self.assertEqual(second["n_scored"], self.results["provenance"]["n_scored"], "drawn from these very trials")
        self.assertEqual(whole["n_items"], second["n_drawn"])
        self.assertEqual(whole["n_items"], sum(-(-len(self.rows(c, b)) // 5) for c in CONDITIONS for b in ALL_BLOCKS))
        self.assertEqual((whole["n_disagree"], whole["n_cannot_tell"]), (1, 1))
        self.assertEqual(whole["n_agree"], whole["n_items"] - 2)
        low, high = wilson_interval(whole["n_agree"], whole["n_compared"], 0.95)
        self.assertTrue(close(whole["wilson_lo"], low) and close(whole["wilson_hi"], high))
        self.assertEqual(second["scorers"], {"first": ["op1"], "second": ["sc2"], "n_same_role": 0})
        self.assertIs(second["same_person"], False)
        by_id = {t["trial_id"]: t for condition in CONDITIONS for t in self.rows(condition)}
        (differing,), (untold,) = second["disagreements"], second["cannot_tell"]
        self.assertEqual(differing["first"], "success" if by_id[differing["trial_id"]]["success"] else "failure")
        self.assertNotEqual(differing["first"], differing["second"])
        self.assertEqual(set(untold), {"trial_id", "first"})
        values = self.made.values
        self.assertEqual(values["nSecondScored"]["value"], str(whole["n_items"]))
        self.assertEqual(values["nSecondScoredAgree"]["value"], str(whole["n_agree"]))
        self.assertEqual(values["nSecondScoredCannotTell"]["value"], "1")
        text = self.made.report_md
        self.assertIn("## The second scoring from video", text)
        self.assertIn(f"{whole['n_items']} of the {second['n_scored']} scored trials were scored again", text)
        self.assertIn(
            f"the two scores agree in {whole['n_agree']} of the {whole['n_compared']} trials the second scorer "
            "could tell",
            text,
        )
        self.assertIn("the first score was given at the rig by op1, the second from the video by sc2", text)
        self.assertNotIn("by the same person", text)
        self.assertIn(f"- {differing['trial_id']}: {differing['first']} at the rig, {differing['second']} from", text)
        self.assertIn(f"- {untold['trial_id']}: {untold['first']} at the rig; the second scorer could not tell.", text)
        table = whole["table"]
        self.assertIn(
            f"| success | {table['success_success']} | {table['success_failure']} | "
            f"{whole['cannot_tell_first']['success']} |",
            text,
        )
        self.assertIn("- **Second scoring from video:** `data/second_scoring/main/agreement.json`.", text)

    def test_the_descriptive_figures_split_the_trials_they_describe(self):
        for condition in CONDITIONS:
            described = self.results["conditions"][condition]["descriptive"]
            trials = self.rows(condition)
            failures = Counter(t["failure"] for t in trials if not t["success"])
            self.assertEqual({entry["label"]: entry["n"] for entry in described["failures"]}, dict(failures))
            if condition == "replay":
                self.assertIsNone(described["hull"])
                continue
            hull = described["hull"]
            self.assertEqual(hull["inside"]["n"] + hull["outside"]["n"], len(trials))
            self.assertEqual(hull["inside"]["k"] + hull["outside"]["k"], sum(t["success"] for t in trials))
            self.assertEqual(hull["inside"]["n"], sum(t["in_hull"] for t in trials))

    # ------------------------------------------------------------------------- predictions, simulation, text

    def test_the_five_predictions_are_judged_by_their_rules(self):
        by_id = {entry["id"]: entry for entry in self.results["predictions"]}
        self.assertEqual(sorted(by_id), [1, 2, 3, 4, 5])
        anchors = self.results["conditions"]["act_clustered"]["anchors"]
        radius = self.results["conditions"]["act_clustered"]["block_fit"]["r50_median_mm"]
        rule = build.PREDICTION_CURVE
        self.assertEqual(rule["zero_trials"], 6, "protocol section 9 states prediction 1 for 6 trials at d = 0")
        self.assertEqual((rule["zero_min_successes"], rule["far_trials"], rule["far_max_successes"]), (5, 4, 1))
        self.assertEqual((rule["r50_min_mm"], rule["r50_max_mm"]), (5.0, 40.0))
        if (anchors["zero"]["n"], anchors["far"]["n"]) == (rule["zero_trials"], rule["far_trials"]):
            expected = anchors["zero"]["k"] >= 5 and anchors["far"]["k"] <= 1 and 5.0 <= radius <= 40.0
        else:  # the protocol's block no longer has the trials the prediction was stated for
            expected = None
        self.assertEqual(by_id[1]["held"], expected)
        self.assertEqual(by_id[2]["held"], self.results["comparisons"]["clustered_minus_replay"]["verdict"] == "larger")
        workspace = self.results["workspace"]
        success = workspace["success"]
        ahead = success["act_spread"]["k"] > success["act_clustered"]["k"]
        self.assertEqual(by_id[3]["held"], workspace["mcnemar_p"] < 0.05 and ahead)
        self.assertIn("p below 0.05 with the spread policy ahead", by_id[3]["why"])
        numbers = by_id[3]["numbers"]
        self.assertEqual(numbers["outcome"], "spread ahead" if by_id[3]["held"] else numbers["outcome"])
        self.assertIn(numbers["outcome"], ("spread ahead", "clustered ahead", "not decided"))
        self.assertEqual(
            (numbers["k_spread"], numbers["k_clustered"], numbers["n_pairs"], numbers["mcnemar_p"]),
            (success["act_spread"]["k"], success["act_clustered"]["k"], workspace["n_pairs"], workspace["mcnemar_p"]),
        )
        self.assertEqual(
            sum(entry["k_spread"] for entry in numbers["by_region"].values()), success["act_spread"]["k"]
        )
        lookup = self.results["workspace"]["lookup"]
        self.assertEqual(by_id[4]["held"], all(lookup[policy]["verdict"] == "matches" for policy in POLICIES))
        self.assertIsNone(by_id[5]["held"])
        self.assertEqual(by_id[5]["why"], "no prediction was made")
        fields = {"id", "name", "held", "why", "caveats", "numbers"}
        self.assertTrue(all(set(entry) == fields for entry in by_id.values()))
        self.assertEqual(by_id[3]["caveats"], [], "the paired test rests on no fit")
        self.assertEqual(by_id[2]["caveats"], self.results["comparisons"]["clustered_minus_replay"]["caveats"])
        self.assertEqual(by_id[4]["caveats"], self.results["workspace"]["lookup"]["caveats"])
        self.assertEqual(by_id[5]["caveats"], self.results["comparisons"]["spread_minus_clustered"]["caveats"])

    def test_the_simulated_case_is_the_one_fixed_before_the_data(self):
        simulation = self.results["simulation"]
        self.assertFalse(simulation["stale"])
        case = simulation["case"]
        self.assertEqual(
            (case["block"], case["r50_true_mm"], case["width_true_mm"], case["lapse_true"]), ("T1a", 20.0, 3.0, 0.0)
        )
        self.assertEqual((case["replications"], case["bias_mm"], case["rmse_mm"]), (300, -0.2417, 1.6266))
        self.assertEqual((case["rmse_fixed_mm"], case["interval_width_fixed_mm"]), (2.3177, 9.0083))
        pairs = self.experiment.protocol["blocks"]["T2"]["pairs"]
        self.assertEqual(case["paired"], {"pairs": pairs, "rate_low": 0.4, "rate_high": 0.9, "power": 0.879086})

    def test_the_simulation_is_also_quoted_for_each_threshold_block_by_width_of_the_true_curve(self):
        simulation = self.results["simulation"]
        with open(self.experiment.root / "analysis" / "simulation" / "procedure.csv", encoding="utf-8") as handle:
            table = [row for row in csv.DictReader(handle) if row["design"] == "protocol"]
        self.assertEqual(sorted(simulation["blocks"]), ["T1a", "T1b", "T3"])
        falling_everywhere = []
        for block, entry in simulation["blocks"].items():
            rows = [row for row in table if row["block"] == block]
            self.assertEqual(entry["n_trials"], fixtures.planned_total(self.experiment.protocol["blocks"][block]))
            for name, width in (("cliff", 1.0), ("middling", 3.0), ("slope", 10.0)):
                truths = [row for row in rows if float(row["width_true_mm"]) == width]
                found = entry["by_width"][name]
                with self.subTest(block=block, width=name):
                    self.assertEqual((found["width_true_mm"], found["n_truths"]), (width, len(truths)))
                    self.assertEqual(found["n_truths"], 6, "three radii and two lapse rates in the synthetic table")
                    self.assertTrue(close(found["rmse_mm"], statistics.median(float(r["rmse_mm"]) for r in truths)))
                    self.assertTrue(
                        close(
                            found["interval_width_mm"],
                            statistics.median(float(r["mean_interval_width_mm"]) for r in truths),
                        )
                    )
            falling = [
                float(row["lower_edge_rate"])
                for row in rows
                if float(row["r50_true_mm"]) < 2.0 * float(row["width_true_mm"])
            ]
            self.assertEqual(entry["n_truths_falling_at_zero"], len(falling))
            self.assertEqual(len(falling), 4, "a radius of 5 mm against widths of 3 and 10 mm, two lapse rates")
            self.assertTrue(close(entry["open_below_falling_at_zero"], statistics.fmean(falling)))
            falling_everywhere += falling
        self.assertTrue(close(simulation["open_below_falling_at_zero"], statistics.fmean(falling_everywhere)))
        self.assertGreater(simulation["blocks"]["T1b"]["by_width"]["slope"]["rmse_mm"], 4.0)

    def test_the_comparison_rule_s_simulation_is_quoted_case_by_case(self):
        rule = self.results["simulation"]["comparison_rule"]
        digest = self.results["provenance"]["protocol_sha256"][:12]
        self.assertEqual((rule["stale"], rule["protocol_digest_simulated"]), (False, digest))
        self.assertEqual((rule["pairs"], rule["replay_r50_mm"]), (300, 15.0))
        self.assertEqual(len(rule["cases"]), 18)
        for case in rule["cases"]:
            larger, between, smaller = fixtures.simulated_rule_shares(case["width_mm"], case["difference_mm"])
            self.assertEqual((case["larger"], case["not_told_apart"], case["smaller"]), (larger, between, smaller))
        values = self.made.values
        self.assertEqual(values["simRuleReplayRadius"]["value"], "15.0")
        self.assertEqual(values["simRulePairs"]["value"], "300")
        self.assertEqual(values["simRuleLargerWidthThreeDiffFive"]["value"], "35")  # 0.03 + 0.19 * 5 / 3 = 0.3467
        self.assertEqual(values["simRuleLargerWidthOneDiffTen"]["value"], "100")
        self.assertEqual(values["simRuleLargerWidthSixDiffEqual"]["value"], "3")
        self.assertEqual(values["simRuleSmallerWidthSixDiffEqual"]["value"], "2")
        self.assertIn("| 3 mm | 5 mm | 35 % | 65 % | 0 % |", self.made.report_md)
        self.assertIn("analysis/simulation/comparison_rule.csv", self.results["simulation"]["files"])

    def test_something_rests_on_an_unsettled_fit_exactly_when_a_block_fit_is_unsettled(self):
        unsettled = {
            condition
            for condition in CONDITIONS
            for fit in [self.results["conditions"][condition]["block_fit"]]
            if fit["open_below"] or fit["open_above"] or fit["sensitivity"]["plateau_in_doubt"]
        }
        caveats = self.results["caveats"]
        named = {condition for condition in CONDITIONS for text in caveats if f"of {condition}," in text}
        self.assertEqual(named, unsettled)
        self.assertEqual("does not establish a radius" in self.made.report_md, bool(unsettled))

    def test_what_no_record_holds_is_listed_as_not_computed(self):
        listed = {entry["what"] for entry in self.results["not_computed"]}
        self.assertEqual(
            listed,
            {
                "the largest disagreement of a session's camera solve with the stored one",
                "torn frames and mispaced episodes of the new datasets",
                "the number of practice episodes",
            },
        )

    def test_the_report_gives_the_predictions_the_void_trials_and_what_could_not_be_computed(self):
        text = self.made.report_md
        for heading in (
            "## The radii at a glance",
            "## Distances and the scale correction",
            "## Trials",
            "## The workspace test",
            "## The predictions of protocol section 9",
            "## What the analysis could not compute",
            "## The paper's number macros",
        ):
            self.assertIn(heading, text)
        for number, name in enumerate(("The curve", "Hardware", "Layout", "Lookup", "Layout and radius"), start=1):
            self.assertRegex(text, rf"(?m)^{number}\. \*\*{name}\.\*\* ")
        self.assertEqual(len(re.findall(r"\*\*held\*\*|\*\*did not hold\*\*|\*\*not judged\*\*", text)), 4)
        for record in self.experiment.records:
            if not record.scored:
                self.assertIn(
                    f"| {record.planned.trial_id} | {record.planned.block} | {record.planned.condition} | ", text
                )
        self.assertIn("The number of practice episodes:", text)
        self.assertIn("divided by 1.0303", text)
        self.assertIn(f"`data/trials/main.jsonl`: {len(self.experiment.records)} records", text)
        for name, value in self.made.values.items():
            self.assertIn(f"| `\\{name}` | `{value['value']}` |", text)

    # ----------------------------------------------------------------------------------- hygiene, the example

    def test_no_output_names_a_folder_of_this_machine_or_a_person(self):
        for text in (self.made.results_json, self.made.report_md, json.dumps(self.made.values)):
            self.assertNotIn(str(self.experiment.root), text)
            self.assertNotIn(str(self.experiment.root.resolve()), text)
            self.assertNotIn(str(Path.home()), text)
            self.assertNotRegex(text, r"(?<![\w.])(?:/Users|/home|/private|/var|/tmp)/")
            self.assertNotIn(tempfile.gettempdir(), text)
            name = Path.home().name
            if len(name) >= 5:  # a short login name can be an ordinary word
                self.assertNotIn(name, text)

    def test_the_example_in_the_repository_says_it_is_synthetic_and_has_the_contract_s_shape(self):
        # Its numbers are those of the day it was written (the planner or the protocol may have changed since); its
        # shape must be today's. `PYTHONPATH=src python tests/fixtures_experiment.py` writes it again.
        for name in ("README.md", "results.json", "report.md"):
            self.assertIn("SYNTHETIC", (EXAMPLE / name).read_text(encoding="utf-8").splitlines()[0], name)
        example = json.loads((EXAMPLE / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(differences(template(), example), [])
        self.assertIn("SYNTHETIC", example["provenance"]["note"])
        self.assertTrue(example["draft"])
        self.assertEqual(null_paths(example), null_paths(self.results), "the example shows every block run")
        self.assertEqual(example["workspace"]["n_pairs"], 26, "and the stratified workspace test")

    def test_the_same_synthetic_inputs_give_the_example_s_very_bytes_again(self):
        # Under either interpreter, and on any day: as long as the synthetic inputs are the ones the example was
        # made from, the analysis must write the same files. When the planner or the protocol file has changed the
        # inputs, nothing can be compared, and the test above still holds the example to today's shape.
        made_from = fixtures.example_made_from(EXAMPLE)
        self.assertRegex(made_from or "", r"^[0-9a-f]{64}$", "the example's README names the inputs it was made from")
        if made_from != fixtures.inputs_digest(self.experiment.root):
            self.skipTest("the synthetic inputs have changed since the example was written: only its shape is checked")
        for name, text in (("results.json", self.made.results_json), ("report.md", self.made.report_md)):
            with self.subTest(file=name):
                self.assertEqual(
                    (EXAMPLE / name).read_text(encoding="utf-8"),
                    text,
                    f"the analysis no longer gives analysis/report-example/{name} from the same inputs: if that "
                    "is meant, write the example again with `PYTHONPATH=src python tests/fixtures_experiment.py`",
                )

    def test_the_example_is_written_from_the_shared_run(self):
        with tempfile.TemporaryDirectory() as folder:
            written = fixtures.write_example(folder)
            self.assertEqual(sorted(path.name for path in written), ["README.md", "report.md", "results.json"])
            self.assertEqual((Path(folder) / "results.json").read_text(encoding="utf-8"), self.made.results_json)
            self.assertEqual(fixtures.example_made_from(folder), fixtures.inputs_digest(self.experiment.root))
            self.assertIn("SYNTHETIC", (Path(folder) / "README.md").read_text(encoding="utf-8").splitlines()[0])


class VariantTests(unittest.TestCase):
    """Smaller experiments on a coarse grid: what is missing comes out null, and nothing else changes shape."""

    def analysed(self, *, draft: bool = False, **arguments):
        """A synthetic experiment written to a folder of its own and analysed: `(experiment, report)`."""
        folder = self.enterContext(tempfile.TemporaryDirectory())
        experiment = fixtures.build_experiment(folder, coarse=True, **arguments)
        made = report.make_report(report.load_inputs(experiment.root, draft=draft))
        self.assertEqual(
            differences(template(), json.loads(made.results_json)), [], "the shape does not depend on the data"
        )
        self.assertIn("## What the analysis could not compute", made.report_md)
        return experiment, made

    def not_computed(self, made) -> str:
        return " | ".join(entry["what"] for entry in made.results["not_computed"])

    def test_without_the_spread_policy_s_block_its_block_fit_and_its_comparison_are_null(self):
        _, made = self.analysed(blocks=("T1a", "T1b", "T2"))
        results = made.results
        spread = results["conditions"]["act_spread"]
        self.assertIsNone(spread["block_fit"])
        self.assertEqual(spread["pooled_fit"]["blocks"], ["T2"], "a pooled fit only, from its workspace trials")
        self.assertEqual(spread["pooled_fit"]["n"], spread["n_scored"])
        self.assertEqual(spread["anchors"], {"zero": {"n": 0, "k": 0}, "far": {"n": 0, "k": 0}})
        self.assertIsNone(results["comparisons"]["spread_minus_clustered"])
        self.assertEqual(results["comparisons"]["clustered_minus_replay"]["fits"], "block")
        self.assertEqual((results["blocks"]["T3"]["n_scored"], results["blocks"]["T3"]["n_void"]), (0, 0))
        self.assertIsNotNone(results["workspace"]["lookup"])
        self.assertIsNone(results["predictions"][4]["numbers"]["difference"])
        self.assertIn("the block fit of act_spread", self.not_computed(made))
        self.assertIn("the comparison spread_minus_clustered", self.not_computed(made))
        self.assertIn("**Block fit:** none, block T3 has no scored trial.", made.report_md)

    def test_without_the_workspace_test_it_is_null_and_a_pooled_fit_is_the_block_fit(self):
        _, made = self.analysed(blocks=("T1a", "T1b", "T3"))
        results = made.results
        self.assertIsNone(results["workspace"])
        clustered = results["conditions"]["act_clustered"]
        self.assertEqual(clustered["pooled_fit"], {**clustered["block_fit"], "blocks": ["T1a"]})
        self.assertEqual(results["conditions"]["act_spread"]["pooled_fit"]["blocks"], ["T3"])
        self.assertEqual(results["blocks"]["T2"]["n_scored"], 0)
        by_id = {entry["id"]: entry for entry in results["predictions"]}
        self.assertEqual((by_id[3]["held"], by_id[4]["held"]), (None, None))
        self.assertIn("no placement tested with both policies", by_id[3]["why"])
        self.assertIsNotNone(results["comparisons"]["spread_minus_clustered"])
        self.assertIsNotNone(results["reset_sensitivity"]["placed"])
        self.assertIn("the workspace test", self.not_computed(made))
        self.assertIn("Not run: its block has no scored trial.", made.report_md)

    def test_with_the_first_two_blocks_alone_there_is_still_a_report(self):
        _, made = self.analysed(blocks=("T1a", "T1b"))
        results = made.results
        self.assertEqual(results["conditions"]["act_spread"]["trials"], [])
        self.assertEqual(
            (results["conditions"]["act_spread"]["n_scored"], results["conditions"]["act_spread"]["n_void"]), (0, 0)
        )
        self.assertIsNone(results["conditions"]["act_spread"]["pooled_fit"])
        self.assertIsNone(results["conditions"]["act_spread"]["max_success_d_mm"])
        self.assertIsNone(results["workspace"])
        self.assertEqual(results["comparisons"]["clustered_minus_replay"]["verdict"], "larger")
        self.assertTrue(results["predictions"][0]["held"])
        self.assertIn("No scored trial.", made.report_md)

    def test_without_a_by_eye_file_that_half_of_the_reset_sensitivity_is_null(self):
        _, made = self.analysed(blocks=("T1a", "T1b"), by_eye=False)
        results = made.results
        self.assertIsNone(results["instrument"]["by_eye"])
        sensitivity = results["reset_sensitivity"]
        self.assertEqual((sensitivity["by_eye"], sensitivity["n_by_eye"]), (None, 0))
        self.assertGreater(sensitivity["placed"], 0.5)
        self.assertEqual(results["provenance"]["by_eye"], [])
        self.assertIn("the by-eye reset errors", self.not_computed(made))
        self.assertIn("With by-eye errors: not computed", made.report_md)

    def test_without_scale_runs_no_distance_is_corrected_and_the_results_say_so(self):
        experiment, made = self.analysed(blocks=("T1a", "T1b"), scale_factors=())
        results = made.results
        scale = results["distance_scale"]
        self.assertEqual(
            (scale["applied"], scale["factor"], scale["factor_min"], scale["factor_max"]), (False, None, None, None)
        )
        self.assertEqual((scale["n_runs"], scale["divisor"]), (0, 1.0))
        self.assertIsNone(results["instrument"]["scale_check"])
        by_id = {
            t["trial_id"]: t["d_mm"] for condition in CONDITIONS for t in results["conditions"][condition]["trials"]
        }
        for record in experiment.scored():
            self.assertEqual(by_id[record.planned.trial_id], estimates.stable(record.nearest_demo_distance_mm))
        self.assertIn("No scale run was found, so no correction was made", made.report_md)
        self.assertIn("the scale check and the scale correction", self.not_computed(made))

    def test_scale_runs_that_disagree_leave_distances_as_read_and_report_their_spread(self):
        factors = (1.020, 1.027, 1.033, 1.031)
        experiment, made = self.analysed(blocks=("T1a", "T1b"), scale_factors=factors)
        scale = made.results["distance_scale"]
        self.assertFalse(scale["applied"])
        self.assertEqual(
            (scale["factor_min"], scale["factor_max"], scale["n_runs"], scale["divisor"]), (1.02, 1.033, 4, 1.0)
        )
        self.assertAlmostEqual(scale["factor"], sum(factors) / 4, places=9)
        self.assertAlmostEqual(scale["spread_points"], 1.3, places=6)
        self.assertEqual(experiment.divisor, 1.0)
        trials = made.results["conditions"]["replay"]["trials"]
        recorded = {r.planned.trial_id: r.nearest_demo_distance_mm for r in experiment.scored(condition="replay")}
        self.assertEqual([t["d_mm"] for t in trials], [estimates.stable(recorded[t["trial_id"]]) for t in trials])
        self.assertIn("Distances are left as the camera reads them", made.report_md)

    def test_scale_runs_on_the_agreeing_side_of_the_line_divide_every_distance(self):
        factors = (1.021, 1.030, 1.025, 1.024)  # 0.9 of a point apart
        experiment, made = self.analysed(blocks=("T1a", "T1b"), scale_factors=factors)
        scale = made.results["distance_scale"]
        self.assertTrue(scale["applied"])
        self.assertAlmostEqual(scale["divisor"], 1.025, places=9)
        trial = made.results["conditions"]["act_clustered"]["trials"][5]
        record = next(r for r in experiment.scored() if r.planned.trial_id == trial["trial_id"])
        self.assertEqual(trial["d_mm"], estimates.stable(record.nearest_demo_distance_mm / scale["divisor"]))
        self.assertLess(trial["d_mm"], record.nearest_demo_distance_mm)

    def test_a_single_scale_run_makes_no_correction(self):
        _, made = self.analysed(blocks=("T1b",), scale_factors=(1.03,))
        scale = made.results["distance_scale"]
        self.assertEqual((scale["applied"], scale["n_runs"], scale["factor"], scale["divisor"]), (False, 1, 1.03, 1.0))
        self.assertEqual(made.results["instrument"]["scale_check"]["n_runs"], 1)

    def test_a_stale_simulation_gives_no_case(self):
        _, made = self.analysed(blocks=("T1b",), simulation="stale")
        simulation = made.results["simulation"]
        self.assertTrue(simulation["stale"])
        self.assertIsNone(simulation["case"])
        self.assertEqual((simulation["blocks"], simulation["open_below_falling_at_zero"]), (None, None))
        self.assertTrue(simulation["comparison_rule"]["stale"])
        self.assertEqual(simulation["protocol_digest_simulated"], "0123456789ab")
        self.assertNotEqual(simulation["protocol_digest_now"], "0123456789ab")
        self.assertIn("The simulation is stale: run it again", made.report_md)
        self.assertEqual([name for name in made.values if name.startswith("sim")], [])

    def test_without_the_simulation_s_tables_it_is_null(self):
        _, made = self.analysed(blocks=("T1b",), simulation=None)
        self.assertIsNone(made.results["simulation"])
        self.assertIn("the simulated case", self.not_computed(made))
        self.assertEqual([name for name in made.values if name.startswith("sim")], [])

    def test_a_stale_comparison_rule_empties_its_own_macros_and_no_others(self):
        _, made = self.analysed(blocks=("T1b",), rule_simulation="stale")
        rule = made.results["simulation"]["comparison_rule"]
        self.assertEqual((rule["stale"], rule["cases"], rule["pairs"]), (True, None, None))
        self.assertEqual(rule["protocol_digest_simulated"], "0123456789ab")
        self.assertFalse(made.results["simulation"]["stale"])
        self.assertEqual([name for name in made.values if name.startswith("simRule")], [])
        self.assertIn("simRmse", made.values)
        self.assertIn("simRmseReplayCliff", made.values)
        self.assertIn("The simulation of the comparison rule names `0123456789ab`", made.report_md)
        self.assertIn("the simulation of the comparison rule", self.not_computed(made))

    def test_a_stale_procedure_simulation_leaves_a_fresh_comparison_rule_its_macros(self):
        _, made = self.analysed(blocks=("T1b",), simulation="stale", rule_simulation="fresh")
        self.assertTrue(made.results["simulation"]["stale"])
        self.assertFalse(made.results["simulation"]["comparison_rule"]["stale"])
        sims = [name for name in made.values if name.startswith("sim")]
        self.assertEqual(len(sims), 23)
        self.assertTrue(all(name.startswith("simRule") for name in sims))

    def test_without_the_comparison_rule_s_table_that_part_is_null(self):
        _, made = self.analysed(blocks=("T1b",), rule_simulation=None)
        self.assertIsNone(made.results["simulation"]["comparison_rule"])
        self.assertEqual([name for name in made.values if name.startswith("simRule")], [])
        self.assertIn("simRmse", made.values)
        self.assertIn("(`analysis/simulation/comparison_rule.csv`) was not found", made.report_md)
        self.assertIn("the simulation of the comparison rule", self.not_computed(made))

    def test_without_a_scene_audit_the_scene_is_null_and_the_report_says_so(self):
        _, made = self.analysed(blocks=("T1b",), scene=False)
        self.assertIsNone(made.results["scene"])
        self.assertEqual(made.results["provenance"]["scene_audits"], [])
        missing = {entry["what"]: entry["why"] for entry in made.results["not_computed"]}
        why = missing["the scene at each trial's start (the basket's place and the wrist camera's)"]
        self.assertIn("there is no scene audit of data/trials/main.jsonl (data/audit/scene_<log>.json)", why)
        of_the_scene = r"nTrials(Basket|Wrist)\w+|basketStartMax|wristStartMax"
        self.assertEqual([name for name in made.values if re.fullmatch(of_the_scene, name)], [])
        self.assertIn("No scene audit (`data/audit/scene_<log>.json`) holds a scored trial", made.report_md)
        self.assertIn("**Scene audits:** none.", made.report_md)

    def test_trials_logged_since_the_audit_are_named_as_not_in_it(self):
        experiment, whole = self.analysed(blocks=("T1b",))
        scored = [record.planned.trial_id for record in experiment.scored()]
        audit = json.loads(json.dumps(experiment.scene))
        kept = [line for line in audit["trials"] if line["trial_id"] not in scored[-2:]]
        fixtures.write_scene_audit(experiment.root / "data" / "audit", {**audit, "trials": kept})
        made = report.make_report(report.load_inputs(experiment.root))
        scene = made.results["scene"]
        self.assertEqual((scene["n_scored"], scene["n_audited"]), (len(scored), len(scored) - 2))
        self.assertEqual(scene["not_audited"], scored[-2:])
        self.assertEqual(sum(group["n"] for group in scene["success"].values()), len(scored) - 2)
        self.assertIn(f"**Not in the audit:** {', '.join(scored[-2:])}.", made.report_md)
        self.assertLessEqual(scene["n_named_basket"], whole.results["scene"]["n_named_basket"])
        # An audit that holds void trials alone holds none of the scored ones: there is nothing to report.
        void = [line for line in audit["trials"] if line["trial_id"] not in scored]
        self.assertTrue(void)
        fixtures.write_scene_audit(experiment.root / "data" / "audit", {**audit, "trials": void})
        made = report.make_report(report.load_inputs(experiment.root))
        self.assertIsNone(made.results["scene"])
        self.assertIn("the scene audit holds none of the scored trials", json.dumps(made.results["not_computed"]))
        self.assertNotIn("nTrialsBasketNamed", made.values, "an audit that read nothing does not print as zero")

    def test_without_an_agreement_file_the_second_scoring_is_null_and_the_report_says_so(self):
        _, made = self.analysed(blocks=("T1b",), second_scorer=None)
        self.assertIsNone(made.results["second_scoring"])
        self.assertIsNone(made.results["provenance"]["second_scoring"])
        self.assertIn("the second scoring from video", self.not_computed(made))
        self.assertEqual([name for name in made.values if name.startswith("nSecondScored")], [])
        self.assertIn("No agreement file (`data/second_scoring/main/agreement.json`) was found", made.report_md)
        self.assertIn("- **Second scoring from video:** none.", made.report_md)

    def test_when_the_operator_scored_again_the_report_says_that_both_scores_are_by_one_person(self):
        experiment, made = self.analysed(blocks=("T1b",), second_scorer="op1")
        second = made.results["second_scoring"]
        items = second["all"]["n_items"]
        self.assertEqual(second["scorers"], {"first": ["op1"], "second": ["op1"], "n_same_role": items})
        self.assertIs(second["same_person"], True)
        self.assertIn("the first score was given at the rig by op1, the second from the video by op1", made.report_md)
        self.assertIn(
            f"- **Both scores of all {items} trials are by the same person**, the second given from the video with "
            "the trials' identities hidden",
            made.report_md,
        )
        # Some items by the operator and some by a second person: the report says in how many.
        partly = json.loads(json.dumps(experiment.second_scoring))
        partly["scorers"].update(second=["op1", "sc2"], n_same_role=2)
        fixtures.write_second_scoring(experiment.root / "data" / "second_scoring" / "main", partly)
        made = report.make_report(report.load_inputs(experiment.root))
        self.assertIs(made.results["second_scoring"]["same_person"], False)
        self.assertIn(f"- **In 2 of the {items} trials both scores are by the same person**", made.report_md)
        self.assertIn("the second from the video by op1 and sc2", made.report_md)

    def test_an_agreement_that_is_not_of_these_trials_gives_no_count_and_the_reason(self):
        experiment, whole = self.analysed(blocks=("T1b",))
        folder = experiment.root / "data" / "second_scoring" / "main"
        self.assertIsNone(whole.results["second_scoring"]["why_not"])

        def reported(change=None, **arguments):
            agreement = json.loads(json.dumps(experiment.second_scoring))
            if change is not None:
                change(agreement)
            fixtures.write_second_scoring(folder, agreement)
            made = report.make_report(report.load_inputs(experiment.root, **arguments))
            self.assertEqual(differences(template(), json.loads(made.results_json)), [])
            return made

        cases = {
            "a rehearsal": (lambda a: a.update(rehearsal=True), "is a rehearsal, drawn from trials that are not"),
            "other logs": (
                lambda a: a["provenance"].update(logs=["data/trials/other.jsonl"]),
                "is the second scoring of data/trials/other.jsonl, and this report reads data/trials/main.jsonl",
            ),
            "a log that changed": (
                lambda a: a["provenance"].update(logs_digest="0" * 64),
                "the file has changed since (its digest is no longer the one the draw recorded)",
            ),
        }
        for name, (change, said) in cases.items():
            with self.subTest(case=name):
                made = reported(change)
                second = made.results["second_scoring"]
                self.assertIn(said, second["why_not"])
                self.assertEqual(
                    [second[key] for key in ("all", "by_kind", "scorers", "same_person", "disagreements")], [None] * 5
                )
                self.assertEqual([name for name in made.values if name.startswith("nSecondScored")], [])
                missing = {entry["what"]: entry["why"] for entry in made.results["not_computed"]}
                self.assertIn(said, missing["the second scoring from video"])
                self.assertTrue(second["why_not"].startswith("data/second_scoring/main/agreement.json "))
                self.assertIn(f"Not reported: {second['why_not']}.", made.report_md)
        # A trial logged after the draw changes the log's digest: the agreement is no longer of these trials.
        last = experiment.records[-1]
        later = dataclasses.replace(last, planned=dataclasses.replace(last.planned, trial_id="T1b-999"))
        schema.append_trial(experiment.log, later)
        made = reported()
        self.assertEqual(made.results["provenance"]["n_records"], len(experiment.records) + 1)
        self.assertIn("has changed since", made.results["second_scoring"]["why_not"])

    def test_without_placement_logs_the_recording_figures_are_null(self):
        _, made = self.analysed(blocks=("T1b",), placement_logs=False)
        datasets = made.results["datasets"]
        self.assertIsNone(datasets["clustered"]["recording"])
        self.assertIsNone(datasets["both"]["recording_days"])
        self.assertEqual(datasets["clustered"]["n_trained"], len(fixtures.audited_starts("clustered")))
        self.assertNotIn("nRerecordedClustered", made.values)
        self.assertNotIn("nRecordingDays", made.values)
        self.assertIn("demoPlacementErrorMedian", made.values)

    def test_the_number_of_pairs_comes_from_the_protocol_and_the_data(self):
        # Seven placements in trained cells as the planner draws them, and three built by hand in held-out cells:
        # pairs that no plan foresaw. Their region is where the cube was.
        experiment, made = self.analysed(blocks=("T1a", "T2"), workspace=(("trained_cells", 7),), held_out_pairs=3)
        stated = experiment.protocol["blocks"]["T2"]
        self.assertEqual((stated["pairs"], stated["region"]), (7, "trained_cells"))
        workspace = made.results["workspace"]
        self.assertEqual(workspace["n_pairs"], 10)
        self.assertEqual(len(workspace["lookup"]["act_spread"]["pmf"]), 11, "a count of 0 to 10 successes")
        self.assertEqual(made.results["blocks"]["T2"]["planned"], 14)
        self.assertEqual(sum(workspace["by_region"][region]["n_pairs"] for region in workspace["by_region"]), 10)
        self.assertEqual(workspace["by_region"]["held_out_cells"]["n_pairs"], 3)
        self.assertEqual([pair["region"] for pair in workspace["pairs"][7:]], ["held_out_cells"] * 3)
        self.assertEqual(made.results["simulation"]["case"]["paired"]["pairs"], 7)
        self.assertEqual(made.values["nPairs"]["value"], "10")
        self.assertEqual((made.values["nPairsTrained"]["value"], made.values["nPairsHeldOut"]["value"]), ("7", "3"))
        self.assertLessEqual(workspace["lookup"]["act_clustered"]["hi"], 10)

    def test_a_block_drawn_from_two_regions_is_analysed_as_one_test_and_counted_by_region(self):
        drawn = (("trained_cells", 5), ("held_out_cells", 4))
        experiment, made = self.analysed(blocks=("T1a", "T2"), workspace=drawn)
        stated = experiment.protocol["blocks"]["T2"]
        self.assertEqual(stated["pairs"], 9)
        self.assertEqual([(entry["region"], entry["pairs"]) for entry in stated["regions"]], list(drawn))
        workspace = made.results["workspace"]
        self.assertEqual((workspace["n_pairs"], made.results["blocks"]["T2"]["planned"]), (9, 18))
        grid = Grid.from_protocol(experiment.protocol)
        placed: dict[str, list] = {}
        for record in experiment.scored(block="T2"):
            placed.setdefault(record.planned.pair_id, []).append(record.placed)
        regions = Counter()
        for pair in workspace["pairs"]:
            first, second = placed[pair["pair_id"]]
            middle = Point((first.x_mm + second.x_mm) / 2, (first.y_mm + second.y_mm) / 2)
            expected = "trained_cells" if grid.is_trained(grid.cell_of(middle)) else "held_out_cells"
            self.assertEqual(pair["region"], expected)
            regions[expected] += 1
        self.assertEqual(
            {region: entry["n_pairs"] for region, entry in workspace["by_region"].items()}, dict(regions)
        )
        self.assertEqual(set(regions), {"trained_cells", "held_out_cells"})
        table = workspace["table"]
        self.assertEqual(sum(table.values()), 9, "one table over all the pairs")
        self.assertAlmostEqual(
            workspace["mcnemar_p"], mcnemar_exact(table["only_spread"], table["only_clustered"]), places=9
        )
        for policy in POLICIES:
            by_region = [workspace["by_region"][region][policy] for region in workspace["by_region"]]
            self.assertEqual(sum(entry["k"] for entry in by_region), workspace["success"][policy]["k"])
            self.assertEqual(sum(entry["n"] for entry in by_region), 9)
            self.assertTrue(
                close(
                    sum(entry["lookup_expected"] for entry in by_region), workspace["lookup"][policy]["expected"], 1e-7
                )
            )
            self.assertEqual(len(workspace["lookup"][policy]["pmf"]), 10)
        self.assertEqual(made.results["simulation"]["case"]["paired"]["pairs"], 9)
        self.assertEqual(made.values["nPairs"]["value"], "9")
        self.assertEqual(made.values["nTrialsWorkspace"]["value"], "18")
        for region, name in (("trained_cells", "Trained"), ("held_out_cells", "HeldOut")):
            self.assertEqual(made.values[f"nPairs{name}"]["value"], str(regions[region]))
            for policy, suffix in (("act_clustered", "Clustered"), ("act_spread", "Spread")):
                found = workspace["by_region"][region][policy]
                self.assertEqual(made.values[f"workspaceSuccess{suffix}{name}"]["value"], str(found["k"]))
                self.assertRegex(made.values[f"lookupPredicted{suffix}{name}"]["value"], r"^\d+\.\d$")
        self.assertIn("| held-out cells | ", made.report_md)

    def test_pairs_run_with_a_second_pair_of_training_runs_are_counted_apart_and_tested_together(self):
        experiment, whole = self.analysed(blocks=("T1a", "T2"), workspace=(("trained_cells", 4), ("held_out_cells", 4)))
        second = {f"T2-p{number:02d}" for number in (2, 4, 6, 8)}

        def other_runs(index, entry):
            if entry["planned"]["pair_id"] in second and entry["outcome"] != "void":
                entry["checkpoint"] = fixtures.checkpoint_name(entry["planned"]["condition"], run="v1-s2")

        fixtures.rewrite_log(experiment.log, other_runs)
        made = report.make_report(report.load_inputs(experiment.root))
        self.assertEqual(differences(template(), json.loads(made.results_json)), [])
        workspace = made.results["workspace"]
        self.assertEqual(
            [entry["runs"] for entry in workspace["by_run"]],
            [
                {"act_clustered": "act-tolrad-clustered-v1", "act_spread": "act-tolrad-spread-v1"},
                {"act_clustered": "act-tolrad-clustered-v1-s2", "act_spread": "act-tolrad-spread-v1-s2"},
            ],
        )
        self.assertEqual([entry["n_pairs"] for entry in workspace["by_run"]], [4, 4])
        first = {pair["pair_id"] for pair in workspace["pairs"]} - second
        for policy in POLICIES:
            for entry, pairs in zip(workspace["by_run"], (first, second), strict=True):
                inside = [pair for pair in workspace["pairs"] if pair["pair_id"] in pairs]
                self.assertEqual(entry[policy]["k"], sum(pair[policy]["success"] for pair in inside))
                self.assertEqual(entry[policy]["n"], 4)
                self.assertTrue(
                    close(entry[policy]["lookup_expected"], sum(pair[policy]["p_lookup"] for pair in inside), 1e-7)
                )
            self.assertEqual(
                sum(entry[policy]["k"] for entry in workspace["by_run"]), workspace["success"][policy]["k"]
            )
        for key in ("table", "mcnemar_p", "success", "lookup", "by_region"):
            self.assertEqual(workspace[key], whole.results["workspace"][key], "the test itself does not look at runs")
        self.assertEqual(
            made.results["conduct"]["checkpoints"]["runs"],
            {
                "act_clustered": ["act-tolrad-clustered-v1", "act-tolrad-clustered-v1-s2"],
                "act_spread": ["act-tolrad-spread-v1", "act-tolrad-spread-v1-s2"],
            },
        )
        self.assertEqual(made.results["conduct"]["checkpoints"]["evaluation_step"], 80000)
        self.assertEqual(made.values["evalCheckpoint"]["value"], "80,000")
        self.assertEqual(made.results["notes"], [], "two runs in the workspace test are the plan, not a remark")
        self.assertIn("The pairs were run with 2 pairs of runs", made.report_md)
        self.assertIn("| `act-tolrad-spread-v1-s2` | 4 | the spread policy | ", made.report_md)

    def test_a_threshold_block_run_with_two_training_runs_is_fitted_and_remarked_on(self):
        experiment, _ = self.analysed(blocks=("T1a",))
        scored = [i for i, record in enumerate(experiment.records) if record.scored]

        def other_run(index, entry):
            if index in scored[::2]:
                entry["checkpoint"] = fixtures.checkpoint_name("act_clustered", run="v1-s2")

        fixtures.rewrite_log(experiment.log, other_run)
        made = report.make_report(report.load_inputs(experiment.root))
        self.assertEqual(len(made.results["notes"]), 1)
        self.assertIn("block T1a were run with 2 training runs", made.results["notes"][0])
        self.assertIn("act-tolrad-clustered-v1-s2", made.results["notes"][0])
        self.assertEqual(made.results["conditions"]["act_clustered"]["block_fit"]["n"], len(scored))

    def test_records_that_name_a_step_alone_have_no_run(self):
        experiment, _ = self.analysed(blocks=("T1a", "T2"), workspace=(("trained_cells", 3),))

        def bare(index, entry):
            if entry["checkpoint"] is not None:
                entry["checkpoint"] = entry["checkpoint"].split("@")[1]

        fixtures.rewrite_log(experiment.log, bare)
        made = report.make_report(report.load_inputs(experiment.root))
        checkpoints = made.results["conduct"]["checkpoints"]
        self.assertEqual(checkpoints["by_condition"], {"act_clustered": ["080000"], "act_spread": ["080000"]})
        self.assertEqual(checkpoints["runs"], {"act_clustered": [], "act_spread": []})
        self.assertEqual((checkpoints["step"], checkpoints["evaluation_step"]), ("080000", 80000))
        self.assertEqual(
            [entry["runs"] for entry in made.results["workspace"]["by_run"]],
            [{"act_clustered": None, "act_spread": None}],
        )
        self.assertEqual(made.results["workspace"]["by_run"][0]["n_pairs"], 3)
        self.assertIn("a run the records do not name", made.report_md)
        self.assertEqual(made.values["evalCheckpoint"]["value"], "80,000")

    def test_without_the_longer_list_of_lapse_values_no_fit_is_made_twice(self):
        _, made = self.analysed(blocks=("T1a", "T1b"), sensitivity=False)
        for condition in ("act_clustered", "replay"):
            self.assertIsNone(made.results["conditions"][condition]["block_fit"]["sensitivity"])
        self.assertIsNone(made.results["conditions"]["act_clustered"]["pooled_fit"]["sensitivity"])
        self.assertIn("the plateau check", self.not_computed(made))
        self.assertIn("**The plateau check was not made:**", made.report_md)
        self.assertEqual([name for name in made.values if "Wide" in name], [])
        self.assertEqual([name for name in made.values if name.startswith("plateauShift")], [])
        self.assertIn("radiusReplay", made.values)
        self.assertTrue(made.results["predictions"][1]["held"], "the comparison never used the second fit")

    def test_with_the_longer_list_the_comparisons_still_use_the_first_fit(self):
        _, without = self.analysed(blocks=("T1a", "T1b"), sensitivity=False)
        _, with_it = self.analysed(blocks=("T1a", "T1b"))
        self.assertEqual(with_it.results["comparisons"], without.results["comparisons"])
        self.assertEqual(with_it.results["reset_sensitivity"], without.results["reset_sensitivity"])
        for condition in ("act_clustered", "replay"):
            first, second = (m.results["conditions"][condition]["block_fit"] for m in (with_it, without))
            self.assertIsNotNone(first["sensitivity"])
            self.assertEqual({**first, "sensitivity": None}, second)
        self.assertEqual(with_it.values["radiusReplayWide"]["source"].count("longer list of lapse values"), 1)

    def test_a_plateau_in_doubt_is_said_in_words(self):
        experiment, made = self.analysed(blocks=("T1a", "T1b"))
        limit = experiment.protocol["analysis"]["sensitivity_limit_mm"]
        settled = copy.deepcopy(made.results)
        for condition in ("act_clustered", "replay"):
            for kind in ("block_fit", "pooled_fit"):
                if settled["conditions"][condition][kind] is not None:
                    settled["conditions"][condition][kind]["sensitivity"]["plateau_in_doubt"] = False
        text = report.render_report(settled, made.values)
        self.assertIn(f"No plateau is in doubt: no fit's two medians differ by more than {limit:g} mm.", text)
        doubted = copy.deepcopy(settled)
        check = doubted["conditions"]["replay"]["block_fit"]["sensitivity"]
        check.update(r50_median_mm=check["r50_median_mm"] + 8.0, median_shift_mm=8.0, plateau_in_doubt=True)
        text = report.render_report(doubted, made.values)
        self.assertIn("**The plateau of the replay is in doubt** (block (T1b) fit)", text)
        self.assertIn(f"a shift of +8.0 mm, more than the protocol's {limit:g} mm", text)
        self.assertIn("this condition's radius is not established", text)
        self.assertNotIn("No plateau is in doubt", text)
        self.assertIn("**The plateau is in doubt.**", text)
        self.assertNotIn("The plateau of the clustered policy is in doubt", text)

    def test_a_limit_the_protocol_does_not_state_leaves_the_doubt_unsaid(self):
        experiment, _ = self.analysed(blocks=("T1b",))
        file = experiment.root / "config" / "protocol.toml"
        text = file.read_text(encoding="utf-8")
        file.write_text(re.sub(r"(?m)^sensitivity_limit_mm = .*\n", "", text), encoding="utf-8")
        fixtures.write_simulation(experiment.root / "analysis" / "simulation", file, 20)
        made = report.make_report(report.load_inputs(experiment.root))
        check = made.results["conditions"]["replay"]["block_fit"]["sensitivity"]
        self.assertEqual((check["limit_mm"], check["plateau_in_doubt"]), (None, None))
        self.assertIsNotNone(check["r50_median_mm"])
        self.assertIn("whether a plateau is in doubt", self.not_computed(made))
        self.assertIn("states no limit for the shift", made.report_md)

    def test_a_list_of_lapse_values_the_model_cannot_hold_is_refused(self):
        experiment, _ = self.analysed(blocks=("T1b",))
        file = experiment.root / "config" / "protocol.toml"
        text = file.read_text(encoding="utf-8")
        changed, count = re.subn(r"(?m)^lapse_values_sensitivity = .*$", "lapse_values_sensitivity = [0.0, 0.95]", text)
        self.assertEqual(count, 1)
        file.write_text(changed, encoding="utf-8")
        with self.assertRaises(ReportError) as caught:
            report.make_report(report.load_inputs(experiment.root))
        self.assertIn("the [analysis] section of the protocol file cannot be used", str(caught.exception))

    def test_deviations_are_counted_from_the_decision_log(self):
        _, made = self.analysed(blocks=("T1b",), deviations=2)
        self.assertEqual(made.results["deviations"]["n"], 2)
        self.assertEqual(made.values["nDeviations"]["value"], "2")
        self.assertIn("the decision log marks 2:", made.report_md)

    def test_no_trial_log_at_all_leaves_every_count_null_and_keeps_what_does_not_need_trials(self):
        experiment, _ = self.analysed(blocks=("T1b",))
        experiment.log.unlink()
        made = report.make_report(report.load_inputs(experiment.root))
        results = made.results
        self.assertEqual(differences(template(), json.loads(made.results_json)), [])
        self.assertEqual(
            (results["blocks"], results["conduct"], results["workspace"], results["reset_sensitivity"]),
            (None, None, None, None),
        )
        self.assertEqual(results["provenance"]["n_records"], 0)
        for condition in CONDITIONS:
            section = results["conditions"][condition]
            self.assertEqual((section["trials"], section["bins"], section["n_void"]), ([], [], None))
            self.assertEqual((section["block_fit"], section["pooled_fit"]), (None, None))
        self.assertEqual(results["instrument"]["scale_check"]["n_runs"], 4)
        self.assertEqual(results["datasets"]["clustered"]["n_trained"], len(fixtures.audited_starts("clustered")))
        self.assertEqual(len(results["demonstrations"]["spread"]), len(fixtures.audited_starts("spread")))
        self.assertIn("everything that is counted from trials", self.not_computed(made))
        self.assertIn("No counted trial log was found.", made.report_md)
        of_trials = r"n(Trials|Void|Zero|Far|Pairs)\w*|voidReasons|nOperators|nTrialDays|success\w+|exec\w+"
        counted = [name for name in made.values if re.fullmatch(of_trials, name)]
        self.assertEqual(counted, [], "an experiment that has not run does not print as zeros")
        self.assertIn("nScaleRuns", made.values)
        self.assertIn("simRmse", made.values)

    def test_a_trial_whose_partner_is_missing_is_left_out_of_the_paired_table_only(self):
        # A line is taken out of the log below, so the experiment is written without a scene audit of it.
        experiment, whole = self.analysed(blocks=("T1a", "T2"), held_out_pairs=2, scene=False)
        last = max(i for i, record in enumerate(experiment.records) if record.scored and record.planned.block == "T2")
        gone = experiment.records[last]
        lines = experiment.log.read_text(encoding="utf-8").splitlines()
        experiment.log.write_text(
            "\n".join(line for index, line in enumerate(lines) if index != last) + "\n", encoding="utf-8"
        )
        made = report.make_report(report.load_inputs(experiment.root))
        workspace = made.results["workspace"]
        self.assertEqual(workspace["n_pairs"], whole.results["workspace"]["n_pairs"] - 1)
        self.assertEqual(workspace["n_unpaired"], 1)
        self.assertNotIn(gone.planned.pair_id, [pair["pair_id"] for pair in workspace["pairs"]])
        other = "act_spread" if gone.planned.condition == "act_clustered" else "act_clustered"
        left = [t for t in made.results["conditions"][other]["trials"] if t["pair_id"] == gone.planned.pair_id]
        self.assertEqual(len(left), 1, "the partner that was run is still a trial of its policy")
        self.assertEqual(
            made.results["conditions"][other]["pooled_fit"]["n"], made.results["conditions"][other]["n_scored"]
        )
        self.assertTrue(any(left[0]["trial_id"] in note for note in made.results["notes"]))
        self.assertEqual(made.values["nPairs"]["value"], str(workspace["n_pairs"]))

    def test_a_block_cut_short_is_not_judged_against_a_prediction_stated_for_the_whole_block(self):
        experiment, _ = self.analysed(blocks=("T1a", "T1b"), void_every=0, scene=False)  # a line is taken out
        zero = [
            i
            for i, r in enumerate(experiment.records)
            if r.planned.block == "T1a" and r.planned.level_mm == 0 and r.planned.staircase_id is None
        ]
        lines = experiment.log.read_text(encoding="utf-8").splitlines()
        experiment.log.write_text(
            "\n".join(line for index, line in enumerate(lines) if index != zero[-1]) + "\n", encoding="utf-8"
        )
        made = report.make_report(report.load_inputs(experiment.root))
        curve = made.results["predictions"][0]
        self.assertIsNone(curve["held"])
        stated = experiment.protocol["blocks"]["T1a"]
        self.assertIn(f"the block has {stated['anchors_zero'] - 1} and {stated['anchors_far']}", curve["why"])
        self.assertIn("**not judged**", made.report_md)

    def test_a_void_trial_run_with_another_setting_is_reported_and_not_analysed(self):
        experiment, _ = self.analysed(blocks=("T1b",))
        void = next(i for i, record in enumerate(experiment.records) if not record.scored)

        def other_clamp(index, entry):
            if index == void:
                entry["execution"]["clamp_deg"] = 10.0

        fixtures.rewrite_log(experiment.log, other_clamp)
        made = report.make_report(report.load_inputs(experiment.root))
        self.assertEqual(len(made.results["notes"]), 1)
        self.assertIn("ran with clamp_deg = 10", made.report_md)
        self.assertEqual(made.results["conduct"]["execution"]["common"]["clamp_deg"], 15.0)

    def test_a_step_that_is_not_named_as_a_number_leaves_the_training_step_null(self):
        experiment, _ = self.analysed(blocks=("T1a", "T3"))

        def named(name):
            def change(index, entry):
                if entry["checkpoint"] is not None:
                    entry["checkpoint"] = entry["checkpoint"].split("@")[0] + "@" + name

            return change

        fixtures.rewrite_log(experiment.log, named("last"))
        made = report.make_report(report.load_inputs(experiment.root))
        checkpoints = made.results["conduct"]["checkpoints"]
        self.assertEqual((checkpoints["step"], checkpoints["evaluation_step"]), ("last", None))
        self.assertEqual(checkpoints["runs"]["act_spread"], ["act-tolrad-spread-v1"])
        self.assertNotIn("evalCheckpoint", made.values)
        self.assertIn("the training step of the evaluated checkpoints", self.not_computed(made))
        self.assertIn("Training step: not named as a number", made.report_md)

    def test_another_step_than_the_protocol_plans_is_said_and_not_refused(self):
        experiment, _ = self.analysed(blocks=("T1a", "T3"))

        def earlier(index, entry):
            if entry["checkpoint"] is not None:
                entry["checkpoint"] = entry["checkpoint"].replace("@080000", "@060000")

        fixtures.rewrite_log(experiment.log, earlier)
        made = report.make_report(report.load_inputs(experiment.root))
        self.assertEqual(made.results["conduct"]["checkpoints"]["evaluation_step"], 60000)
        self.assertEqual(made.values["evalCheckpoint"]["value"], "60,000")
        self.assertEqual(len(made.results["notes"]), 1)
        self.assertIn("evaluated at training step 60,000, and the protocol plans step 80,000", made.results["notes"][0])
        self.assertIn("not below 40,000", made.results["notes"][0])


class AchievedRateTests(unittest.TestCase):
    """The control rate a trial achieved, from the two numbers its record gives: by hand, and where there is none."""

    def test_the_rate_is_the_intervals_between_the_steps_over_the_time_from_the_first_step_to_the_last(self):
        # 301 steps are 300 intervals: over 10 s that is 30 a second, and over 10.5 s it is 28.57.
        self.assertEqual(bookkeeping.achieved_rate_hz(301, 10.0), 30.0)
        self.assertEqual(bookkeeping.achieved_rate_hz(301, 10.5), 28.5714286)
        self.assertEqual(bookkeeping.achieved_rate_hz(2, 0.04), 25.0)
        self.assertEqual(bookkeeping.achieved_rate_hz(601, 20), 30.0, "a whole number of seconds is a duration too")

    def test_a_trial_too_short_to_have_a_rate_has_none(self):
        for steps, loop_s in ((0, 0.0), (1, 0.0), (1, 0.5), (2, 0.0), (300, 0.0), (300, -1.0)):
            with self.subTest(steps=steps, loop_s=loop_s):
                self.assertIsNone(bookkeeping.achieved_rate_hz(steps, loop_s))

    def test_a_record_that_does_not_carry_the_two_numbers_has_none(self):
        missing = ((300, None), (None, 10.0), (None, None), ("300", 10.0), (300, "10.0"), (True, 10.0), (300, True))
        for steps, loop_s in (*missing, (300.0, 10.0), (300, float("nan")), (300, float("inf"))):
            with self.subTest(steps=steps, loop_s=loop_s):
                self.assertIsNone(bookkeeping.achieved_rate_hz(steps, loop_s))
        self.assertIsNone(bookkeeping.loop_seconds({}))
        self.assertIsNone(bookkeeping.loop_seconds({"loop_s": "12.5"}))
        self.assertIsNone(bookkeeping.loop_seconds({"loop_s": True}))
        self.assertEqual(bookkeeping.loop_seconds({"loop_s": 12}), 12.0)
        self.assertEqual(bookkeeping.loop_seconds({"loop_s": 0.0}), 0.0)

    def test_what_is_measured_is_said_with_what_is_left_out(self):
        said = bookkeeping.ACHIEVED_RATE_MEANING
        for part in ("between the first step sent and the last", "handoff gate", "first inference", "outside it"):
            self.assertIn(part, said)
        self.assertEqual(bookkeeping.RATE_SHORTFALL_LIMIT, 0.02)
        self.assertEqual(sorted(bookkeeping.RATE_GROUPS["all"]), sorted(CONDITIONS))


class AchievedRateVariantTests(unittest.TestCase):
    """The achieved rate on whole experiments whose records were changed: slower loops, missing durations."""

    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        cls.experiment = fixtures.build_experiment(
            Path(cls._folder.name) / "experiment", coarse=True, blocks=("T1a", "T1b"), void_every=0
        )
        cls.requested = cls.experiment.protocol["execution"]["control_hz"]

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def analysed(self, change):
        """The analysis of the experiment with `change(index, entry)` applied to every line of a copy of its log."""
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        log = folder / "changed.jsonl"
        log.write_bytes(self.experiment.log.read_bytes())
        fixtures.rewrite_log(log, change)
        made = report.make_report(report.load_inputs(self.experiment.root, trials=[log]))
        self.assertEqual(differences(template(), json.loads(made.results_json)), [])
        return made

    def at_rate(self, hz: float, conditions=CONDITIONS):
        """A change that makes the loops of some conditions run at exactly `hz`."""

        def change(index, entry):
            steps = entry["execution"]["steps_sent"]
            if entry["planned"]["condition"] in conditions and steps >= 2:
                entry["execution"]["loop_s"] = (steps - 1) / hz

        return change

    def not_computed(self, made) -> dict[str, str]:
        return {entry["what"]: entry["why"] for entry in made.results["not_computed"]}

    def test_policy_trials_that_ran_slow_are_named_in_plain_words_and_the_replay_is_not(self):
        made = self.analysed(self.at_rate(28.0, conditions=("act_clustered",)))
        rates = made.results["conduct"]["execution"]["achieved_hz"]
        self.assertTrue(close(rates["policy"]["median"], 28.0))
        self.assertIs(rates["policy"]["short_of_requested"], True)
        self.assertIs(rates["replay"]["short_of_requested"], False)
        self.assertEqual(made.values["execControlRateAchieved"]["value"], "28.0")
        self.assertIn(
            f"**The policy trials ran more than 2 % below the rate they were asked for: a median of 28.0 Hz against "
            f"{self.requested:g} Hz, {100 * (self.requested - 28.0) / self.requested:.1f} % below.**",
            made.report_md,
        )
        self.assertNotIn("The replay trials ran more than", made.report_md)

    def test_a_replay_that_ran_slow_is_named_too_and_leaves_the_paper_s_macro_alone(self):
        whole = self.analysed(lambda index, entry: None)
        made = self.analysed(self.at_rate(0.9 * self.requested, conditions=("replay",)))
        rates = made.results["conduct"]["execution"]["achieved_hz"]
        self.assertIs(rates["replay"]["short_of_requested"], True)
        self.assertIs(rates["policy"]["short_of_requested"], False)
        self.assertIn("**The replay trials ran more than 2 % below the rate they were asked for", made.report_md)
        self.assertEqual(made.values["execControlRateAchieved"], whole.values["execControlRateAchieved"])

    def test_two_percent_below_is_not_more_than_two_percent_below(self):
        on_the_line = self.analysed(self.at_rate(0.98 * self.requested))
        just_under = self.analysed(self.at_rate(0.979 * self.requested))
        for kind in ("policy", "replay", "all"):
            for made, short in ((on_the_line, False), (just_under, True)):
                self.assertIs(made.results["conduct"]["execution"]["achieved_hz"][kind]["short_of_requested"], short)
        self.assertNotIn("below the rate they were asked for", on_the_line.report_md)
        faster = self.analysed(self.at_rate(1.05 * self.requested))
        self.assertIs(faster.results["conduct"]["execution"]["achieved_hz"]["all"]["short_of_requested"], False)

    def test_records_without_the_loop_s_duration_leave_the_rate_and_its_macro_empty_and_say_why(self):
        def forget(index, entry):
            del entry["execution"]["loop_s"]

        made = self.analysed(forget)
        rates = made.results["conduct"]["execution"]["achieved_hz"]
        scored = {kind: 0 for kind in ("policy", "replay")}
        for record in self.experiment.scored():
            scored["replay" if record.planned.condition == "replay" else "policy"] += 1
        for kind in ("policy", "replay"):
            self.assertEqual(
                rates[kind],
                {"requested": self.requested, "median": None, "min": None, "max": None, "n": 0}
                | {"n_without": scored[kind], "short_of_requested": None},
            )
            why = self.not_computed(made)[f"the achieved control rate of the {kind} trials"]
            self.assertIn(f"none of the {scored[kind]} scored {kind} trials records loop_s", why)
        self.assertNotIn("execControlRateAchieved", made.values)
        self.assertIn("execControlRate", made.values, "what was asked is still known")
        self.assertIn("achieved: no rate", made.report_md)
        for condition in ("act_clustered", "replay"):
            for trial in made.results["conditions"][condition]["trials"]:
                self.assertEqual((trial["loop_s"], trial["achieved_hz"]), (None, None))
        self.assertIsNotNone(made.results["conditions"]["replay"]["block_fit"], "nothing else needs the duration")

    def test_some_records_without_it_are_counted_and_the_rate_rests_on_the_others(self):
        clustered = [
            index for index, record in enumerate(self.experiment.records) if record.planned.condition == "act_clustered"
        ]

        def forget(index, entry):
            if index in clustered[:5]:
                del entry["execution"]["loop_s"]

        whole = self.analysed(lambda index, entry: None)
        made = self.analysed(forget)
        before = whole.results["conduct"]["execution"]["achieved_hz"]["policy"]
        after = made.results["conduct"]["execution"]["achieved_hz"]["policy"]
        records = self.experiment.records
        lost = sum(records[index].planned.trial_id != fixtures.TOO_SHORT_TRIAL for index in clustered[:5])
        self.assertEqual((after["n"], after["n_without"]), (before["n"] - lost, before["n_without"] + lost))
        self.assertIn("execControlRateAchieved", made.values)
        self.assertNotIn("the achieved control rate of the policy trials", self.not_computed(made))
        self.assertEqual(
            made.results["conduct"]["execution"]["achieved_hz"]["replay"],
            whole.results["conduct"]["execution"]["achieved_hz"]["replay"],
        )

    def test_policy_trials_that_were_all_too_short_have_no_rate_and_the_reason_says_which_is_which(self):
        clustered = [
            index for index, record in enumerate(self.experiment.records) if record.planned.condition == "act_clustered"
        ]

        def cut_short(index, entry):
            if index in clustered[:3]:
                del entry["execution"]["loop_s"]
            elif index in clustered:
                entry["execution"].update(steps_sent=1, loop_s=0.0)

        made = self.analysed(cut_short)
        policy = made.results["conduct"]["execution"]["achieved_hz"]["policy"]
        self.assertEqual((policy["n"], policy["n_without"], policy["median"]), (0, len(clustered), None))
        why = self.not_computed(made)["the achieved control rate of the policy trials"]
        self.assertIn(f"none of the {len(clustered)} scored policy trials has a rate: 3 record no loop_s", why)
        self.assertIn(f"in the other {len(clustered) - 3} fewer than two steps were sent", why)
        self.assertNotIn("execControlRateAchieved", made.values)
        self.assertGreater(made.results["conduct"]["execution"]["achieved_hz"]["replay"]["n"], 0)

    def test_without_a_policy_trial_the_macro_is_empty_and_the_report_says_there_is_none(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = fixtures.build_experiment(folder, coarse=True, blocks=("T1b",), void_every=0)
            made = report.make_report(report.load_inputs(experiment.root))
        rates = made.results["conduct"]["execution"]["achieved_hz"]
        policy = rates["policy"]
        self.assertEqual((policy["n"], policy["n_without"], policy["requested"]), (0, 0, None))
        self.assertGreater(rates["replay"]["n"], 0)
        self.assertEqual(rates["all"], rates["replay"])
        self.assertEqual(
            self.not_computed(made)["the achieved control rate of the policy trials"], "there is no scored policy trial"
        )
        self.assertNotIn("execControlRateAchieved", made.values)
        self.assertNotIn("  - policy trials:", made.report_md)
        self.assertIn("  - replay trials: asked ", made.report_md)
        self.assertNotIn("  - all scored trials:", made.report_md, "one kind only: its line is not said twice")

    def test_a_void_trial_has_no_part_in_the_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = fixtures.build_experiment(folder, coarse=True, blocks=("T1a", "T1b"))
            void = [index for index, record in enumerate(experiment.records) if not record.scored]
            self.assertGreater(len(void), 2)
            made = report.make_report(report.load_inputs(experiment.root))

            def crawl(index, entry):
                if index in void and entry["execution"]["steps_sent"] >= 2:
                    entry["execution"]["loop_s"] = entry["execution"]["steps_sent"] / 3.0

            fixtures.rewrite_log(experiment.log, crawl)
            again = report.make_report(report.load_inputs(experiment.root))
        self.assertEqual(
            again.results["conduct"]["execution"]["achieved_hz"], made.results["conduct"]["execution"]["achieved_hz"]
        )


class ViewCreptVariantTests(unittest.TestCase):
    """The camera's creep before an attempt on experiments whose records were changed."""

    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        cls.experiment = fixtures.build_experiment(Path(cls._folder.name) / "experiment", coarse=True, blocks=("T1b",))
        cls.limit = cls.experiment.protocol["execution"]["view_crept_limit_mm"]
        cls.scored = [index for index, record in enumerate(cls.experiment.records) if record.scored]
        cls.void = [index for index, record in enumerate(cls.experiment.records) if not record.scored]

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def analysed(self, change):
        """The analysis of the experiment with `change(index, entry)` applied to every line of a copy of its log."""
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        log = folder / "changed.jsonl"
        log.write_bytes(self.experiment.log.read_bytes())
        fixtures.rewrite_log(log, change)
        made = report.make_report(report.load_inputs(self.experiment.root, trials=[log]))
        self.assertEqual(differences(template(), json.loads(made.results_json)), [])
        return made

    def test_a_handful_of_readings_by_hand(self):
        readings = {self.scored[0]: 0.10, self.scored[1]: 0.99, self.scored[2]: 0.40, self.scored[3]: 0.30}

        def change(index, entry):
            if index in readings:
                entry["execution"]["view_crept_mm"] = readings[index]
            else:
                del entry["execution"]["view_crept_mm"]

        made = self.analysed(change)
        self.assertEqual(
            made.results["conduct"]["view_crept"],
            {
                "n": 4,
                "n_without": len(self.scored) - 4,
                "median_mm": 0.35,
                "max_mm": 0.99,
                "limit_mm": self.limit,
                "n_over_limit": 0,
            },
        )
        self.assertEqual(made.values["viewCreptMedian"]["value"], "0.35")
        self.assertEqual(made.values["viewCreptMax"]["value"], "0.99")
        self.assertIn(f"4 of the {len(self.scored)} scored trials record", made.report_md)
        self.assertIn("a median of 0.35 mm, at most 0.99 mm.", made.report_md)

    def test_records_written_before_the_runner_kept_the_reading_leave_it_null_and_say_so(self):
        def forget(index, entry):
            del entry["execution"]["view_crept_mm"]

        made = self.analysed(forget)
        self.assertIsNone(made.results["conduct"]["view_crept"])
        missing = {entry["what"]: entry["why"] for entry in made.results["not_computed"]}
        why = missing["how far the front camera's picture had crept before an attempt"]
        self.assertIn(f"none of the {len(self.scored)} scored trials records execution.view_crept_mm", why)
        self.assertEqual([name for name in made.values if name.startswith("viewCrept")], [])
        self.assertIn(f"none of the {len(self.scored)} scored trials records how far", made.report_md)

    def test_a_reading_that_is_not_a_number_is_no_reading(self):
        odd = dict(zip(self.scored, (None, "0.4", True, float("nan")), strict=False))

        def change(index, entry):
            if index in odd:
                entry["execution"]["view_crept_mm"] = odd[index]

        crept = self.analysed(change).results["conduct"]["view_crept"]
        self.assertEqual((crept["n"], crept["n_without"]), (len(self.scored) - 4, 4))

    def test_a_scored_trial_beyond_the_limit_is_said_in_bold_and_a_void_one_has_no_part(self):
        self.assertTrue(self.void, "the experiment has a void trial")

        def change(index, entry):
            if index == self.scored[5] or index in self.void:
                entry["execution"]["view_crept_mm"] = self.limit + 0.21

        made = self.analysed(change)
        crept = made.results["conduct"]["view_crept"]
        self.assertEqual((crept["n"], crept["n_over_limit"]), (len(self.scored), 1))
        self.assertEqual(crept["max_mm"], self.limit + 0.21)
        self.assertIn(f"**1 of them lie beyond the protocol's limit of {self.limit:g} mm**", made.report_md)

        def on_the_line(index, entry):
            if index == self.scored[5]:
                entry["execution"]["view_crept_mm"] = self.limit

        crept = self.analysed(on_the_line).results["conduct"]["view_crept"]
        self.assertEqual((crept["max_mm"], crept["n_over_limit"]), (self.limit, 0), "the limit itself is inside")

    def test_a_protocol_without_a_limit_gives_the_readings_and_no_verdict(self):
        crept = bookkeeping.view_crept(report.load_inputs(self.experiment.root).scored, None)
        self.assertEqual((crept["limit_mm"], crept["n_over_limit"]), (None, None))
        self.assertEqual(crept["n"], len(self.scored))
        self.assertIsNone(bookkeeping.view_crept([], self.limit))


class SceneTests(unittest.TestCase):
    """The scene at each trial's start on audits written by hand, and the figures the report repeats from the rig."""

    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        experiment = fixtures.build_experiment(
            Path(cls._folder.name) / "experiment", coarse=True, blocks=("T1b",), void_every=0, scene=False
        )
        cls.scored = report.load_inputs(experiment.root).scored
        cls.log = cls.scored[0].log

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def scene(self, *offsets_mm: float, wrist_px: float = 0.5) -> dict:
        """The scene of an audit whose first trials began with the basket at these offsets, to one side."""
        readings = tuple(
            inputs.SceneReading(
                trial.trial_id, trial.condition, trial.record.outcome, (-off, 0.0), (0.0, wrist_px), 0.0, False,
                off > 2.0, wrist_px > 2.0,
            )
            for trial, off in zip(self.scored, offsets_mm, strict=False)
        )
        audit = inputs.SceneAudit("data/audit/scene_main.json", "main.jsonl", 2.0, 2.0, readings)
        return bookkeeping.scene_at_start(self.scored, {self.log: audit})

    def test_an_offset_on_the_end_of_a_bin_belongs_to_the_bin_it_ends(self):
        scene = self.scene(0.0, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 7.5)
        self.assertEqual([entry["n"] for entry in scene["basket_start_bins"]], [2, 2, 2, 2])
        self.assertEqual(bookkeeping.SCENE_BIN_ENDS_MM, (1.0, 2.0, 3.0))
        self.assertEqual((scene["n_named_basket"], scene["named_basket_min_mm"]), (4, 2.1))
        self.assertEqual(scene["basket_start_max_mm"], 7.5)
        self.assertEqual((scene["n_audited"], len(scene["not_audited"])), (8, len(self.scored) - 8))

    def test_with_no_trial_named_the_counts_are_zero_and_the_text_says_so(self):
        scene = self.scene(0.2, 0.4, 1.9)
        self.assertEqual((scene["n_named_basket"], scene["n_named_wrist"], scene["n_basket_moved"]), (0, 0, 0))
        self.assertIsNone(scene["named_basket_min_mm"])
        self.assertEqual(scene["success"]["named"], {"n": 0, "k": 0})
        values = macros.macro_values({"scene": scene})
        self.assertEqual(
            [values[name]["value"] for name in ("nTrialsBasketNamed", "nTrialsWristNamed", "nTrialsBasketMoved")],
            ["0", "0", "0"],
        )
        self.assertEqual((values["basketStartMax"]["value"], values["wristStartMax"]["value"]), ("1.9", "0.5"))
        text = "\n".join(markdown._scene({"scene": scene, "blocks": {}}))
        self.assertIn("**The basket was moved in no trial**", text)
        self.assertIn("- **Named for the basket:** no trial began with it more than 2 mm from its place.", text)
        succeeded = scene["success"]["unnamed"]["k"]
        self.assertIn(f"no trial was named, and {succeeded} of the 3 trials named for neither succeeded.", text)
        self.assertNotIn("pixels past the line", text)

    def test_the_first_pixel_step_past_the_line_is_the_fewest_whole_pixels_beyond_it(self):
        scene = self.scene(0.2, 0.4, 2.2)
        cases = ((2.0, "1.9", 2, "2.2"), (1.0, "0.9", 1, "1.1"), (2.16, "2.0", 3, "3.2"))
        for limit, line_px, whole, whole_mm in cases:
            with self.subTest(limit=limit):
                text = "\n".join(markdown._scene({"scene": {**scene, "basket_named_mm": limit}, "blocks": {}}))
                self.assertIn(
                    f"the {limit:g} mm line lies {line_px} pixels from the basket's place, and an offset of {whole} "
                    f"whole pixels, about {whole_mm} mm, is the first pixel step past it.",
                    text,
                )
        text = "\n".join(markdown._scene({"scene": scene, "blocks": {}}))
        self.assertIn("The trials named for the basket began 2.2 mm to 2.2 mm from it: at most 0.2 pixels past", text)

    def test_a_trial_named_for_the_wrist_camera_alone_is_a_named_trial(self):
        scene = self.scene(0.2, 0.4, wrist_px=2.5)
        self.assertEqual((scene["n_named_basket"], scene["n_named_wrist"]), (0, 2))
        self.assertEqual((scene["success"]["named"]["n"], scene["success"]["unnamed"]["n"]), (2, 0))
        self.assertEqual(scene["wrist_start_max_px"], 2.5)

    def test_without_an_audit_or_without_a_scored_trial_in_one_there_is_nothing(self):
        self.assertIsNone(bookkeeping.scene_at_start(self.scored, {self.log: None}))
        self.assertIsNone(bookkeeping.scene_at_start(self.scored, {}))
        self.assertIsNone(self.scene())
        self.assertEqual(macros.macro_values({"scene": None}), {})

    def test_the_figures_the_report_repeats_are_those_of_the_rig_s_code(self):
        # The rig's modules are read here, never imported: the analysis may not depend on them.
        rig = fixtures.REPO / "src" / "tolerance_radius" / "rig"
        basket_place = (rig / "basket_place.py").read_text(encoding="utf-8")
        audit = (rig / "scene_audit.py").read_text(encoding="utf-8")
        per_px = re.search(r"(?m)^#: ([^\n]*)\nMM_PER_PX = ([\d.]+)$", basket_place)
        self.assertEqual(float(per_px.group(2)), bookkeeping.SCENE_MM_PER_PX)
        self.assertIn("the two tags are 526.9 mm and 486 px apart", per_px.group(1))
        self.assertIn("526.9 mm and 486 px apart", bookkeeping.SCENE_MM_PER_PX_SOURCE)
        self.assertAlmostEqual(526.9 / 486, bookkeeping.SCENE_MM_PER_PX, places=2)
        limits = re.search(r"(?m)^BASKET_NAMED_MM, WRIST_NAMED_PX = ([\d.]+), ([\d.]+)$", audit)
        self.assertEqual(
            (float(limits.group(1)), float(limits.group(2))),
            (fixtures.SCENE_BASKET_NAMED_MM, fixtures.SCENE_WRIST_NAMED_PX),
        )
        moved = re.search(r"(?m)^BASKET_MOVED_MM = ([\d.]+)$", audit)
        self.assertEqual(float(moved.group(1)), bookkeeping.SCENE_BASKET_MOVED_MM)
        self.assertEqual(fixtures.SCENE_BASKET_MOVED_MM, bookkeeping.SCENE_BASKET_MOVED_MM)
        command = (fixtures.REPO / "scripts" / "rig" / "scene_audit.py").read_text(encoding="utf-8")
        self.assertIn('"basket_named_mm": scene_audit.BASKET_NAMED_MM', command, "the file states the audit's limit")

    def test_nothing_of_the_rig_is_imported_by_the_analysis(self):
        for path in sorted((fixtures.REPO / "src" / "tolerance_radius" / "report").glob("*.py")):
            self.assertNotRegex(
                path.read_text(encoding="utf-8"), r"(?m)^\s*(from|import) tolerance_radius\.rig", path.name
            )


class WorkspaceSectionTests(unittest.TestCase):
    """The workspace test on a handful of trials written out by hand, where every count can be checked by eye."""

    @classmethod
    def setUpClass(cls):
        protocol = tomllib.loads(fixtures.protocol_text(coarse=True))
        cls.grid = Grid.from_protocol(protocol)
        cls.fit = estimates.fit(
            [2.0, 8.0, 15.0, 30.0, 40.0], [True, True, True, False, False], grids_from_protocol(protocol["analysis"])
        )

    def rows(self, *pairs, runs=None) -> list[dict]:
        """Trials of a paired block: for each pair its cell and the two outcomes, clustered policy first.

        `runs` names, pair by pair, the training run of both trials (`v1` where it is left out); a run of None gives
        a checkpoint that names its step alone.
        """
        rows = []
        for number, (cell, clustered, spread) in enumerate(pairs, start=1):
            centre = self.grid.cell_centre(cell)
            run = runs[number - 1] if runs else "v1"
            for policy, success, distance in (("act_clustered", clustered, 20.0), ("act_spread", spread, 6.0)):
                if success is None:
                    continue
                rows.append(
                    {
                        "block": "T2",
                        "condition": policy,
                        "pair_id": f"T2-p{number:02d}",
                        "trial_id": f"T2-{len(rows) + 1:03d}",
                        "x_mm": centre.x_mm + 1.0,
                        "y_mm": centre.y_mm - 1.0,
                        "d_mm": distance,
                        "success": success,
                        "checkpoint": "080000" if run is None else fixtures.checkpoint_name(policy, run=run),
                    }
                )
        return rows

    def test_pairs_are_counted_by_the_region_of_their_cell(self):
        held_out, trained = self.grid.held_out_cells[0], self.grid.trained_cells[0]
        rows = self.rows(
            (trained, True, True), (trained, False, True), (held_out, False, True), (held_out, False, False)
        )
        notes: list[str] = []
        workspace = build.workspace_section(rows, "T2", self.fit, self.grid, 0.95, notes)
        self.assertEqual(notes, [])
        self.assertEqual(workspace["n_pairs"], 4)
        self.assertEqual(
            [pair["region"] for pair in workspace["pairs"]], ["trained_cells"] * 2 + ["held_out_cells"] * 2
        )
        self.assertEqual(workspace["table"], {"both": 1, "neither": 1, "only_spread": 2, "only_clustered": 0})
        self.assertEqual(workspace["mcnemar_p"], mcnemar_exact(2, 0))
        by_region = workspace["by_region"]
        self.assertEqual((by_region["trained_cells"]["n_pairs"], by_region["held_out_cells"]["n_pairs"]), (2, 2))
        self.assertEqual(
            (by_region["trained_cells"]["act_clustered"]["k"], by_region["trained_cells"]["act_spread"]["k"]), (1, 2)
        )
        self.assertEqual(
            (by_region["held_out_cells"]["act_clustered"]["k"], by_region["held_out_cells"]["act_spread"]["k"]), (0, 1)
        )
        expected = 2 * float(self.fit.predict(6.0))
        self.assertAlmostEqual(by_region["held_out_cells"]["act_spread"]["lookup_expected"], expected, places=12)
        self.assertAlmostEqual(workspace["lookup"]["act_spread"]["expected"], 2 * expected, places=12)

    def test_a_region_without_a_pair_has_counts_of_zero_and_nothing_else(self):
        rows = self.rows((self.grid.trained_cells[0], True, True), (self.grid.trained_cells[3], False, True))
        workspace = build.workspace_section(rows, "T2", self.fit, self.grid, 0.95, [])
        empty = workspace["by_region"]["held_out_cells"]
        self.assertEqual(empty["n_pairs"], 0)
        self.assertEqual(
            empty["act_spread"],
            {"k": 0, "n": 0, "wilson_lo": None, "wilson_hi": None}
            | {"lookup_expected": None, "lookup_lo": None, "lookup_hi": None},
        )
        self.assertEqual(workspace["by_region"]["trained_cells"]["n_pairs"], 2)

    def test_a_trial_without_its_partner_is_noted_and_left_out(self):
        trained = self.grid.trained_cells[0]
        notes: list[str] = []
        workspace = build.workspace_section(
            self.rows((trained, True, True), (trained, True, None)), "T2", self.fit, self.grid, 0.95, notes
        )
        self.assertEqual((workspace["n_pairs"], workspace["n_unpaired"]), (1, 1))
        self.assertEqual(len(notes), 1)
        self.assertIn("T2-003", notes[0])

    def test_without_the_clustered_block_fit_there_is_a_table_and_no_lookup(self):
        workspace = build.workspace_section(
            self.rows((self.grid.trained_cells[0], True, False)), "T2", None, self.grid, 0.95, []
        )
        self.assertIsNone(workspace["lookup"])
        self.assertIsNone(workspace["pairs"][0]["act_clustered"]["p_lookup"])
        without = workspace["by_region"]["trained_cells"]["act_clustered"]
        self.assertEqual((without["lookup_expected"], without["lookup_lo"], without["lookup_hi"]), (None, None, None))
        self.assertEqual(workspace["table"]["only_clustered"], 1)

    def test_a_block_without_a_scored_trial_gives_nothing(self):
        self.assertIsNone(build.workspace_section([], "T2", self.fit, self.grid, 0.95, []))
        self.assertIsNone(
            build.workspace_section(
                self.rows((self.grid.trained_cells[0], True, True)), None, self.fit, self.grid, 0.95, []
            )
        )

    def test_pairs_are_counted_by_the_runs_of_their_two_trials(self):
        trained, held_out = self.grid.trained_cells[0], self.grid.held_out_cells[0]
        rows = self.rows(
            (trained, True, True),
            (trained, False, True),
            (held_out, False, True),
            (held_out, False, False),
            (trained, True, False),
            runs=["v1", "v2", "v1", "v2", "v2"],
        )
        workspace = build.workspace_section(rows, "T2", self.fit, self.grid, 0.95, [])
        self.assertEqual([pair["act_spread"]["checkpoint"] for pair in workspace["pairs"]][:2], [
            "act-tolrad-spread-v1@080000",
            "act-tolrad-spread-v2@080000",
        ])
        first, second = workspace["by_run"]
        self.assertEqual(
            [entry["runs"] for entry in (first, second)],
            [
                {"act_clustered": "act-tolrad-clustered-v1", "act_spread": "act-tolrad-spread-v1"},
                {"act_clustered": "act-tolrad-clustered-v2", "act_spread": "act-tolrad-spread-v2"},
            ],
        )
        self.assertEqual((first["n_pairs"], second["n_pairs"]), (2, 3))
        self.assertEqual((first["act_clustered"]["k"], first["act_spread"]["k"]), (1, 2))
        self.assertEqual((second["act_clustered"]["k"], second["act_spread"]["k"]), (1, 1))
        self.assertEqual(first["act_spread"]["n"], 2)
        low, high = wilson_interval(1, 3, 0.95)
        self.assertAlmostEqual(second["act_spread"]["wilson_lo"], low, places=12)
        self.assertAlmostEqual(second["act_spread"]["wilson_hi"], high, places=12)
        self.assertAlmostEqual(second["act_spread"]["lookup_expected"], 3 * float(self.fit.predict(6.0)), places=12)
        self.assertEqual(workspace["table"], {"both": 1, "neither": 1, "only_spread": 2, "only_clustered": 1})

    def test_a_pair_whose_two_trials_name_other_runs_than_the_rest_is_a_group_of_its_own(self):
        trained = self.grid.trained_cells[0]
        rows = self.rows((trained, True, True), (trained, True, True), (trained, False, True), runs=["v1", "v1", None])
        rows[2]["checkpoint"] = fixtures.checkpoint_name("act_clustered", run="v2")  # the clustered trial of pair 2
        workspace = build.workspace_section(rows, "T2", self.fit, self.grid, 0.95, [])
        found = [(e["runs"]["act_clustered"], e["runs"]["act_spread"], e["n_pairs"]) for e in workspace["by_run"]]
        self.assertEqual(
            found,
            [
                (None, None, 1),
                ("act-tolrad-clustered-v1", "act-tolrad-spread-v1", 1),
                ("act-tolrad-clustered-v2", "act-tolrad-spread-v1", 1),
            ],
        )

    def test_a_region_is_the_cell_of_a_position_and_nothing_off_the_grid(self):
        held_out = self.grid.cell_centre(self.grid.held_out_cells[0])
        self.assertEqual(build.region_of(held_out, self.grid), "held_out_cells")
        self.assertEqual(build.region_of(self.grid.cell_centre(self.grid.trained_cells[0]), self.grid), "trained_cells")
        self.assertIsNone(build.region_of(Point(-3.0, 10.0), self.grid))


class ConditionSectionTests(unittest.TestCase):
    """One condition's part of the results, on a few trials written out by hand."""

    @classmethod
    def setUpClass(cls):
        protocol = tomllib.loads(fixtures.protocol_text(coarse=True))
        cls.fitting = build.fitting_from_protocol(protocol)
        cls.order = list(protocol["blocks"])

    def row(self, block: str, part: str, distance: float, success: bool, region: str = "trained_cells") -> dict:
        return {
            "block": block,
            "part": part,
            "d_mm": distance,
            "success": success,
            "region": region,
            "in_hull": region == "trained_cells",
            "failure": None if success else "grasp",
            "duration_s": 20.0,
            "steps_sent": 500,
            "replay_episode": None,
        }

    def rows(self) -> list[dict]:
        threshold = [
            self.row("T3", "zero", 0.4, True),
            self.row("T3", "zero", 1.1, False),
            self.row("T3", "T3-up", 6.0, True),
            self.row("T3", "T3-up", 14.0, True),
            self.row("T3", "T3-down", 24.0, False, "held_out_cells"),
            self.row("T3", "T3-down", 19.0, False, "held_out_cells"),
        ]
        workspace = [
            self.row("T2", "pair", 0.2, False),
            self.row("T2", "pair", 31.0, True, "held_out_cells"),
            self.row("T2", "pair", 12.5, True, None),
        ]
        return threshold + workspace

    def test_each_fit_counts_its_own_trials_and_the_condition_counts_them_all(self):
        section, posterior = build.condition_section("act_spread", self.rows(), 2, "T3", self.fitting, self.order)
        block, pooled = section["block_fit"], section["pooled_fit"]
        self.assertEqual((block["blocks"], pooled["blocks"]), (["T3"], ["T2", "T3"]))
        self.assertEqual((block["n"], pooled["n"], posterior.n_trials), (6, 9, 6))
        self.assertEqual(
            {key: block[key] for key in ("max_success_d_mm", "min_failure_d_mm", "zero_k", "zero_n")},
            {"max_success_d_mm": 14.0, "min_failure_d_mm": 1.1, "zero_k": 1, "zero_n": 2},
        )
        self.assertEqual(
            {key: pooled[key] for key in ("max_success_d_mm", "min_failure_d_mm", "zero_k", "zero_n")},
            {"max_success_d_mm": 31.0, "min_failure_d_mm": 0.2, "zero_k": 1, "zero_n": 2},
            "a workspace trial near a demonstration is not a trial asked at its start",
        )
        self.assertEqual((section["max_success_d_mm"], section["min_failure_d_mm"]), (31.0, 0.2))
        self.assertEqual(section["anchors"]["zero"], {"n": 2, "k": 1})

    def test_the_split_by_region_leaves_a_trial_off_the_grid_out(self):
        section, _ = build.condition_section("act_spread", self.rows(), 0, "T3", self.fitting, self.order)
        split = section["descriptive"]["by_region"]
        self.assertEqual((split["trained_cells"]["n"], split["trained_cells"]["k"]), (5, 3))
        self.assertEqual((split["held_out_cells"]["n"], split["held_out_cells"]["k"]), (3, 1))
        self.assertEqual(section["descriptive"]["n_off_grid"], 1)
        self.assertEqual(
            [(found["lo_mm"], found["n"], found["k"]) for found in split["held_out_cells"]["bins"]],
            [(10.0, 1, 0), (20.0, 1, 0), (30.0, 1, 1)],
        )
        self.assertEqual(
            [(found["lo_mm"], found["n"], found["k"]) for found in split["trained_cells"]["bins"]],
            [(0.0, 4, 2), (10.0, 1, 1)],
        )

    def test_without_the_threshold_block_the_fit_level_counts_are_those_of_the_pooled_fit_alone(self):
        rows = [row for row in self.rows() if row["block"] == "T2"]
        section, posterior = build.condition_section("act_spread", rows, 0, "T3", self.fitting, self.order)
        self.assertIsNone(posterior)
        self.assertIsNone(section["block_fit"])
        self.assertEqual((section["pooled_fit"]["zero_k"], section["pooled_fit"]["zero_n"]), (0, 0))
        self.assertEqual(section["pooled_fit"]["max_success_d_mm"], 31.0)

    def test_a_replay_has_a_block_fit_with_its_plateau_check_and_no_pooled_fit(self):
        rows = [dict(row, block="T1b", replay_episode=7, in_hull=None) for row in self.rows()[:6]]
        section, _ = build.condition_section("replay", rows, 0, "T1b", self.fitting, self.order)
        self.assertIsNone(section["pooled_fit"])
        check = section["block_fit"]["sensitivity"]
        analysis = tomllib.loads(fixtures.protocol_text(coarse=True))["analysis"]
        self.assertEqual(check["lapse_values"], analysis["lapse_values_sensitivity"])
        self.assertEqual(check["limit_mm"], analysis["sensitivity_limit_mm"])
        self.assertIn(check["plateau_in_doubt"], (True, False))


class CaveatTests(unittest.TestCase):
    """Which results rest on a block fit that does not establish a radius, and that each says so beside itself."""

    def fit(self, **changed) -> dict:
        settled = {"open_below": False, "open_above": False, "sensitivity": {"plateau_in_doubt": False}}
        return settled | changed

    #: Both comparisons, the lookup prediction and the reset sensitivity, all computed.
    COMPARED = {"clustered_minus_replay": {"verdict": "larger"}, "spread_minus_clustered": {"verdict": "smaller"}}
    LOOKUP, RESETS = {"lookup": {"fit": "act_clustered block_fit"}}, {"placed": 0.9}
    EVERYTHING = (
        "the comparison clustered_minus_replay",
        "the comparison spread_minus_clustered",
        "the lookup prediction",
        "the reset sensitivity",
        "prediction 1",
    )

    def conditions(self, **fits) -> dict:
        settled = {condition: self.fit() for condition in CONDITIONS}
        return {condition: {"block_fit": fit} for condition, fit in (settled | fits).items()}

    def test_the_results_that_can_rest_on_a_fit_are_those_the_protocol_names(self):
        self.assertEqual(tuple(build.RESTS_ON), self.EVERYTHING)
        self.assertEqual(build.RESTS_ON["the comparison clustered_minus_replay"], ("act_clustered", "replay"))
        self.assertEqual(build.RESTS_ON["prediction 1"], ("act_clustered",))
        protocol = " ".join((fixtures.REPO / "protocol" / "protocol.md").read_text(encoding="utf-8").split())
        self.assertIn(
            "(a comparison, the lookup prediction, the reset sensitivity, prediction 1) is still computed by its rule "
            "and is reported with that caveat beside it, never without",
            protocol,
            "protocol section 10 no longer says what rests on a radius that is not established",
        )

    def test_a_fit_does_not_establish_a_radius_when_it_is_open_on_a_side_or_its_plateau_is_in_doubt(self):
        self.assertEqual(build.unsettled(None), [])
        self.assertEqual(build.unsettled(self.fit()), [])
        self.assertEqual(build.unsettled(self.fit(sensitivity=None)), [], "no second fit: no doubt to speak of")
        self.assertEqual(build.unsettled(self.fit(open_below=True)), ["its interval is open below"])
        self.assertEqual(len(build.unsettled(self.fit(open_above=True, sensitivity={"plateau_in_doubt": True}))), 2)
        self.assertEqual(build.unsettled(self.fit(sensitivity={"plateau_in_doubt": None})), [], "no limit, no doubt")

    def test_settled_fits_leave_nothing_to_say(self):
        self.assertEqual(build.caveats(self.conditions(), self.COMPARED, self.LOOKUP, self.RESETS), [])
        by_result = build.caveats_by_result(self.conditions(), self.COMPARED, self.LOOKUP, self.RESETS)
        self.assertEqual(by_result, {what: [] for what in self.EVERYTHING})

    def test_a_replay_whose_interval_is_open_above_is_named_under_its_comparison_only(self):
        conditions = self.conditions(replay=self.fit(open_above=True))
        by_result = build.caveats_by_result(conditions, self.COMPARED, self.LOOKUP, self.RESETS)
        (sentence,) = by_result["the comparison clustered_minus_replay"]
        self.assertIn("the comparison clustered_minus_replay uses the block fit of replay", sentence)
        self.assertIn("open above", sentence)
        self.assertEqual([what for what in by_result if by_result[what]], ["the comparison clustered_minus_replay"])
        self.assertEqual(build.caveats(conditions, self.COMPARED, self.LOOKUP, self.RESETS), [sentence])

    def test_a_clustered_plateau_in_doubt_touches_everything_made_from_that_fit_prediction_1_too(self):
        doubted = self.fit(open_below=True, sensitivity={"plateau_in_doubt": True})
        conditions = self.conditions(act_clustered=doubted)
        found = build.caveats(conditions, self.COMPARED, self.LOOKUP, self.RESETS)
        self.assertEqual([text.split(" uses ")[0] for text in found], list(self.EVERYTHING))
        for text in found:
            self.assertIn("of act_clustered, and its interval is open below and its plateau is in doubt", text)
        by_result = build.caveats_by_result(conditions, self.COMPARED, self.LOOKUP, self.RESETS)
        self.assertEqual([len(by_result[what]) for what in self.EVERYTHING], [1, 1, 1, 1, 1])

    def test_a_comparison_of_two_unsettled_fits_says_both(self):
        conditions = self.conditions(
            act_clustered=self.fit(open_above=True), replay=self.fit(sensitivity={"plateau_in_doubt": True})
        )
        by_result = build.caveats_by_result(conditions, self.COMPARED, self.LOOKUP, self.RESETS)
        first, second = by_result["the comparison clustered_minus_replay"]
        self.assertIn("of act_clustered, and its interval is open above", first)
        self.assertIn("of replay, and its plateau is in doubt", second)

    def test_what_was_not_computed_has_no_caveat(self):
        comparisons = {"clustered_minus_replay": None, "spread_minus_clustered": None}
        unchecked = self.fit(open_above=True, sensitivity=None)
        conditions = self.conditions(act_clustered=unchecked) | {"act_spread": {"block_fit": None}}
        for workspace in (None, {"lookup": None}):
            by_result = build.caveats_by_result(conditions, comparisons, workspace, None)
            self.assertEqual(
                [what for what, found in by_result.items() if found], ["prediction 1"], "the block fit is there"
            )
        by_result = build.caveats_by_result(conditions, comparisons, self.LOOKUP, None)
        self.assertEqual([what for what in by_result if by_result[what]], ["the lookup prediction", "prediction 1"])
        without_fit = self.conditions() | {"act_clustered": {"block_fit": None}}
        self.assertEqual(build.caveats(without_fit, comparisons, None, None), [])

    def test_a_block_whose_every_trial_succeeded_marks_whatever_is_made_from_its_fit(self):
        # Every clustered-policy trial of the threshold block succeeds: nothing says where success ends, the
        # interval is open above, and the fit establishes no radius. Everything made from it must say so.
        with tempfile.TemporaryDirectory() as folder:
            drawn = (("trained_cells", 4), ("held_out_cells", 4))
            experiment = fixtures.build_experiment(
                folder, coarse=True, blocks=("T1a", "T1b", "T2"), workspace=drawn, scene=False  # outcomes change
            )

            def all_succeed(index, entry):
                if entry["planned"]["block"] == "T1a" and entry["outcome"] == "failure":
                    entry.update(outcome="success", subgoal=5, failure=None)

            fixtures.rewrite_log(experiment.log, all_succeed)
            made = report.make_report(report.load_inputs(experiment.root))
        results = made.results
        self.assertEqual(differences(template(), json.loads(made.results_json)), [])
        self.assertTrue(results["conditions"]["act_clustered"]["block_fit"]["open_above"])
        by_id = {entry["id"]: entry for entry in results["predictions"]}
        carried = {
            "the comparison clustered_minus_replay": results["comparisons"]["clustered_minus_replay"]["caveats"],
            "the lookup prediction": results["workspace"]["lookup"]["caveats"],
            "the reset sensitivity": results["reset_sensitivity"]["caveats"],
            "prediction 1": by_id[1]["caveats"],
        }
        for what, found in carried.items():
            with self.subTest(result=what):
                # The clustered policy's fit comes first; a comparison may name the other fit after it.
                self.assertEqual(len(found), 1 + (what.startswith("the comparison") and len(found) == 2))
                self.assertTrue(found[0].startswith(f"{what} uses the block fit of act_clustered, and its interval is"))
        self.assertEqual(results["caveats"], [sentence for found in carried.values() for sentence in found])
        self.assertIsNone(results["comparisons"]["spread_minus_clustered"], "no spread block: nothing to mark")
        self.assertEqual(by_id[2]["caveats"], carried["the comparison clustered_minus_replay"])
        self.assertEqual(by_id[4]["caveats"], carried["the lookup prediction"])
        self.assertEqual((by_id[3]["caveats"], by_id[5]["caveats"]), ([], []))
        # The report says it under the radii, and again beside each result it concerns.
        text = made.report_md
        self.assertIn("**Computed from a fit that does not establish a radius**", text)
        for what, found in carried.items():
            sentence = f"{found[0][0].upper()}{found[0][1:]}."
            self.assertIn(f"- {sentence}", text)
            self.assertIn(f" **Caveat:** {sentence}", text)
        lines = {line.split(" ")[0]: line for line in text.splitlines() if re.match(r"^[1-5]\. \*\*", line)}
        self.assertIn("**Caveat:** Prediction 1 uses the block fit of act_clustered", lines["1."])
        self.assertIn("**Caveat:** The comparison clustered_minus_replay uses", lines["2."])
        self.assertNotIn("Caveat", lines["3."])
        self.assertIn("**Caveat:** The lookup prediction uses", lines["4."])
        self.assertNotIn("Caveat", lines["5."])
        compared = [line for line in text.splitlines() if line.startswith("**The clustered policy's radius minus")]
        (comparison,) = compared
        self.assertIn("**Caveat:** The comparison clustered_minus_replay uses", comparison)
        sensitivity = text[text.index("## Reset sensitivity") : text.index("## The predictions")]
        self.assertIn("**Caveat:** The reset sensitivity uses the block fit of act_clustered", sensitivity)
        workspace = text[text.index("## The workspace test") : text.index("## Reset sensitivity")]
        self.assertIn("matches. **Caveat:** The lookup prediction uses the block fit of act_clustered", workspace)

    def test_results_that_rest_on_settled_fits_carry_empty_lists_and_the_report_no_caveat(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = fixtures.build_experiment(folder, coarse=True, blocks=("T1a", "T1b"))
            made = report.make_report(report.load_inputs(experiment.root))
        settled = copy.deepcopy(made.results)
        for condition in ("act_clustered", "replay"):
            settled["conditions"][condition]["block_fit"].update(open_below=False, open_above=False)
            settled["conditions"][condition]["block_fit"]["sensitivity"]["plateau_in_doubt"] = False
        parts = (settled[key] for key in ("conditions", "comparisons", "workspace", "reset_sensitivity"))
        self.assertEqual(build.caveats(*parts), [])
        settled["caveats"] = []
        settled["comparisons"]["clustered_minus_replay"]["caveats"] = []
        settled["reset_sensitivity"]["caveats"] = []
        for entry in settled["predictions"]:
            entry["caveats"] = []
        text = report.render_report(settled, made.values)
        self.assertNotIn("Caveat", text)
        self.assertNotIn("does not establish a radius", text)
        doubted = copy.deepcopy(settled)
        doubted["caveats"] = ["the lookup prediction uses the block fit of act_clustered, and its plateau is in doubt"]
        text = report.render_report(doubted, made.values)
        self.assertIn("- The lookup prediction uses the block fit of act_clustered, and its plateau is in doubt.", text)


class DeterminismTests(unittest.TestCase):
    def outputs(self, folder: str) -> tuple[str, str, dict]:
        experiment = fixtures.build_experiment(folder, coarse=True)
        made = report.make_report(report.load_inputs(experiment.root))
        return made.results_json, made.report_md, made.values

    def test_two_runs_on_the_same_inputs_give_identical_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.outputs(folder)
            experiment = fixtures.build_experiment(Path(folder) / "again", coarse=True)
            again = report.make_report(report.load_inputs(experiment.root))
            once_more = report.make_report(report.load_inputs(experiment.root))
        self.assertEqual((again.results_json, again.report_md), (once_more.results_json, once_more.report_md))
        self.assertEqual(
            first, (again.results_json, again.report_md, again.values), "where the folder is does not matter"
        )

    def test_a_counted_report_has_no_banner_and_a_draft_has_one(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = fixtures.build_experiment(folder, coarse=True, blocks=("T1b",))
            counted = report.make_report(report.load_inputs(experiment.root))
            draft = report.make_report(report.load_inputs(experiment.root, draft=True))
        self.assertTrue(counted.results_json.startswith('{"banner": null,\n'))
        self.assertEqual((counted.results["draft"], counted.results["banner"]), (False, None))
        self.assertTrue(counted.report_md.startswith("# Analysis report\n"))
        self.assertTrue(draft.results["draft"])
        self.assertTrue(draft.results_json.startswith('{"banner": "DRAFT: written with --draft.'))
        self.assertTrue(draft.report_md.startswith("**DRAFT: written with --draft."))
        self.assertIn("This is a draft", draft.report_md)
        self.assertIn("This is the counted analysis", counted.report_md)
        without = {key: value for key, value in draft.results.items() if key not in ("banner", "draft")}
        self.assertEqual(
            without, {key: value for key, value in counted.results.items() if key not in ("banner", "draft")}
        )


if __name__ == "__main__":
    unittest.main()
