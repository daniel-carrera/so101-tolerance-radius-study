"""Reading the analysis results: what the figures take from the file, what they refuse, and the synthetic files."""

import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

import fixtures_results
from tolerance_radius.figures import results
from tolerance_radius.figures.results import Bin, Count, ResultsError, ScoredTrial
from tolerance_radius.intervals import wilson_interval

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "protocol" / "data-contracts.md"
#: The fields a figure can use that the contract's example does not have (see the notes of `results`), and the note
#: that says a file is synthetic.
BEYOND_THE_EXAMPLE = {
    "distance_scale.divisor",
    "provenance.note",
    "workspace.pairs.[].region",
    "workspace.pairs.[].act_clustered.x_mm",
    "workspace.pairs.[].act_clustered.y_mm",
    "workspace.pairs.[].act_spread.x_mm",
    "workspace.pairs.[].act_spread.y_mm",
}
BEYOND_THE_EXAMPLE_UNDER = "workspace.by_region"


def contract_example() -> dict:
    """The example of the results file in `protocol/data-contracts.md`, with what it leaves to "the same" written out."""
    text = CONTRACT.read_text(encoding="utf-8")
    section = text[text.index("## The analysis results") :]
    example = json.loads(re.search(r"```\n(.*?)\n```", section, re.S).group(1))
    conditions = example["conditions"]
    conditions["act_clustered"]["pooled_fit"] = copy.deepcopy(conditions["act_clustered"]["block_fit"])
    for condition in ("replay", "act_spread"):
        conditions[condition] = copy.deepcopy(conditions["act_clustered"])
    conditions["replay"]["pooled_fit"] = None  # "the same; pooled_fit is null"
    example["comparisons"]["spread_minus_clustered"] = copy.deepcopy(example["comparisons"]["clustered_minus_replay"])
    example["demonstrations"]["spread"] = copy.deepcopy(example["demonstrations"]["clustered"])
    return example


def key_paths(value, prefix: str = "") -> set[str]:
    """Where the fields of a JSON value are: `a.b` for a field of an object, `[]` for the entries of a list."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(prefix + key)
            found |= key_paths(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for item in value:
            found |= key_paths(item, f"{prefix}[].")
    return found


class SyntheticFilesTests(unittest.TestCase):
    def test_every_file_says_that_it_is_a_draft_and_synthetic(self):
        for name in fixtures_results.CASES:
            made = fixtures_results.load(name)
            self.assertIs(made["draft"], True, name)
            self.assertIs(results.is_draft(made), True, name)
            self.assertTrue(made["provenance"]["note"].startswith("Synthetic"), name)
            self.assertIn("Not a result", made["provenance"]["note"], name)
            self.assertEqual(made["provenance"]["protocol_status"], "draft", name)
            self.assertEqual(results.read_results(fixtures_results.path(name)), made)

    def test_the_files_have_the_fields_of_the_contracts_example_and_a_few_known_others(self):
        example = key_paths(contract_example())
        self.assertGreater(len(example), 150)
        for name in fixtures_results.CASES:
            ours = key_paths(fixtures_results.load(name))
            beyond = {path for path in ours - example if not path.startswith(BEYOND_THE_EXAMPLE_UNDER)}
            self.assertLessEqual(beyond, BEYOND_THE_EXAMPLE, name)
            missing = example - ours
            # What a file lacks of the example is what its case is there to lack: a block that was not run.
            not_run = ("workspace.", "conditions.act_spread.", "comparisons.")
            for path in missing:
                self.assertTrue(path.startswith(not_run), f"{name}: {path}")
        self.assertEqual(example - key_paths(fixtures_results.load("base")), set())

    def test_cut_down_to_the_contract_a_file_has_no_field_beyond_the_example_but_its_note(self):
        example = key_paths(contract_example())
        for name in fixtures_results.CASES:
            self.assertLessEqual(key_paths(fixtures_results.contract_only(name)) - example, {"provenance.note"}, name)

    def test_each_file_shows_what_it_is_there_to_show(self):
        def fit(name: str, condition: str, which: str = "block"):
            return results.fit(fixtures_results.load(name), condition, which)

        # An interval open above, and one open below.
        self.assertEqual((fit("open_above", "act_clustered").open_below, fit("open_above", "act_clustered").open_above),
                         (False, True))  # fmt: skip
        replay = fit("threshold_blocks_only", "replay")
        self.assertEqual((replay.open_below, replay.open_above), (True, False))
        open_ones = (("open_above", "act_clustered"), ("threshold_blocks_only", "replay"))
        for name in fixtures_results.CASES:
            for condition in ("act_clustered", "replay"):
                self.assertEqual(not fit(name, condition).bounded, (name, condition) in open_ones, (name, condition))
        # The spread policy with no block fit, and with one.
        self.assertIsNone(fit("no_spread_threshold", "act_spread"))
        self.assertEqual(fit("no_spread_threshold", "act_spread", "pooled").blocks, ("T2",))
        self.assertEqual(fit("base", "act_spread").blocks, ("T3",))
        # No workspace block at all.
        self.assertFalse(results.has_workspace(fixtures_results.load("threshold_blocks_only")))
        self.assertIsNone(fixtures_results.load("threshold_blocks_only")["conditions"]["act_spread"])
        # The scale correction applied, and not.
        self.assertTrue(results.scale_applied(fixtures_results.load("base")))
        self.assertFalse(results.scale_applied(fixtures_results.load("no_spread_threshold")))
        # Placements in held-out cells, counted by region.
        placed = results.placements(fixtures_results.load("held_out_placements"), "act_clustered")
        self.assertEqual([sum(p.region == region for p in placed) for region in results.REGIONS], [13, 13])
        self.assertEqual(len(results.placements(fixtures_results.load("base"), "act_spread")), 20)

    def test_the_blocks_have_the_sizes_the_protocol_plans(self):
        for name in fixtures_results.CASES:
            made = fixtures_results.load(name)
            blocks = [trial.block for trial in results.trials(made, "act_clustered")]
            self.assertEqual(blocks.count("T1a"), 34, name)
            self.assertEqual([trial.block for trial in results.trials(made, "replay")], ["T1b"] * 22, name)

    def test_the_files_are_what_the_script_beside_them_makes(self):
        path = REPO / "tests" / "fixtures" / "make_results_fixtures.py"
        spec = importlib.util.spec_from_file_location("make_results_fixtures", path)
        maker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maker)
        self.assertEqual(set(maker.CASES), set(fixtures_results.CASES))
        for name in ("held_out_placements", "threshold_blocks_only"):  # the fullest and the barest: a second or two
            experiment = maker.Experiment(name, maker.CASES[name])
            experiment.run()
            made = maker.results(experiment)
            maker.check(name, made)
            self.assertEqual(maker.dumps(made) + "\n", fixtures_results.path(name).read_text(encoding="utf-8"), name)


class HeaderTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        self.addCleanup(lambda: Path(handle.name).unlink())
        with handle:
            handle.write(text)
        return Path(handle.name)

    def test_what_is_not_a_results_file_of_this_version_is_refused(self):
        for text, said in (("not json", "not a JSON file"), ("[1, 2]", "one JSON object"),
                           ('{"schema_version": 2, "draft": false}', "schema_version is 2"),
                           ('{"draft": false}', "schema_version is None"),
                           ('{"schema_version": 1}', "draft must be true or false"),
                           ('{"schema_version": 1, "draft": "no"}', "draft must be true or false")):  # fmt: skip
            with self.assertRaisesRegex(ResultsError, said):
                results.read_results(self.write(text))

    def test_the_error_names_the_file_and_not_its_folder(self):
        path = self.write("[]")
        with self.assertRaises(ResultsError) as caught:
            results.read_results(path)
        self.assertIn(path.name, str(caught.exception))
        self.assertNotIn(str(path.parent), str(caught.exception))

    def test_draft_level_and_bin_width_are_read(self):
        made = fixtures_results.load()
        self.assertEqual((results.is_draft(made), results.interval_level(made), results.bin_width_mm(made)),
                         (True, 0.95, 10.0))  # fmt: skip
        self.assertFalse(results.is_draft(fixtures_results.changed(draft=False)))
        for dotted, value in (("interval_level", 1.0), ("interval_level", "95 %"), ("bin_width_mm", 0.0),
                              ("bin_width_mm", None), ("interval_level", True)):  # fmt: skip
            with self.assertRaises(ResultsError):
                changed = fixtures_results.changed(**{dotted: value})
                results.interval_level(changed), results.bin_width_mm(changed)

    def test_a_missing_field_is_named_by_its_place_in_the_file(self):
        made = fixtures_results.load()
        del made["conditions"]["replay"]["trials"][3]["d_mm"]
        with self.assertRaisesRegex(ResultsError, r"conditions\.replay\.trials\[3\]\.d_mm is missing"):
            results.trials(made, "replay")
        del made["distance_scale"]["applied"]
        with self.assertRaisesRegex(ResultsError, r"distance_scale\.applied is missing"):
            results.scale_applied(made)


class ScaleTests(unittest.TestCase):
    def test_distances_were_divided_by_the_factor_when_the_correction_applies_and_by_1_when_not(self):
        applied = fixtures_results.contract_only("base")
        self.assertNotIn("divisor", applied["distance_scale"])
        self.assertEqual((results.scale_applied(applied), results.distance_divisor(applied)), (True, 1.031))
        as_read = fixtures_results.contract_only("no_spread_threshold")
        self.assertEqual((results.scale_applied(as_read), results.distance_divisor(as_read)), (False, 1.0))

    def test_the_divisor_is_taken_from_the_file_when_it_gives_one(self):
        self.assertEqual(results.distance_divisor(fixtures_results.changed(**{"distance_scale.divisor": 1.04})), 1.04)
        self.assertEqual(results.distance_divisor(fixtures_results.load("no_spread_threshold")), 1.0)

    def test_without_a_scale_run_the_factor_may_be_null(self):
        made = fixtures_results.contract_only("no_spread_threshold")
        made["distance_scale"] |= {"factor": None, "factor_min": None, "factor_max": None, "n_runs": 0}
        self.assertEqual(results.distance_divisor(made), 1.0)

    def test_a_scale_that_contradicts_itself_or_is_no_scale_is_refused(self):
        with self.assertRaisesRegex(ResultsError, "not applied"):
            contradicting = fixtures_results.changed("no_spread_threshold", **{"distance_scale.divisor": 1.03})
            results.distance_divisor(contradicting)
        for factor in (0.0, 3.0, None, "1.03"):
            with self.assertRaises(ResultsError):
                made = fixtures_results.contract_only("base")
                made["distance_scale"]["factor"] = factor
                results.distance_divisor(made)


class ConditionTests(unittest.TestCase):
    def test_the_trials_of_a_condition_come_in_the_files_order(self):
        made = fixtures_results.load()
        trials = results.trials(made, "act_clustered")
        listed = made["conditions"]["act_clustered"]["trials"]
        self.assertEqual(len(trials), 54)
        self.assertEqual([(t.block, t.d_mm, t.success) for t in trials],
                         [(t["block"], t["d_mm"], t["success"]) for t in listed])  # fmt: skip
        self.assertIsInstance(trials[0], ScoredTrial)

    def test_a_condition_the_file_does_not_have_has_no_trial_no_fit_and_no_bin(self):
        barest = fixtures_results.load("threshold_blocks_only")
        self.assertIsNone(barest["conditions"]["act_spread"])
        without_key = fixtures_results.load("threshold_blocks_only")
        del without_key["conditions"]["act_spread"]
        emptied = fixtures_results.load("threshold_blocks_only")
        emptied["conditions"]["act_spread"] = {"trials": [], "bins": [], "block_fit": None, "pooled_fit": None}
        for made in (barest, without_key, emptied):
            self.assertEqual(results.trials(made, "act_spread"), [])
            self.assertIsNone(results.fit(made, "act_spread"))
            self.assertIsNone(results.fit(made, "act_spread", "pooled"))
            self.assertEqual(results.file_bins(made, "act_spread"), [])

    def test_a_trial_that_is_not_one_is_refused(self):
        for change in ({"d_mm": -1.0}, {"d_mm": "12"}, {"d_mm": float("nan")}, {"success": 1}, {"block": 3}):
            made = fixtures_results.load()
            made["conditions"]["replay"]["trials"][0] |= change
            with self.assertRaises(ResultsError, msg=change):
                results.trials(made, "replay")
        with self.assertRaises(ValueError):
            results.trials(fixtures_results.load(), "act_other")

    def test_a_fit_has_its_radius_its_interval_and_its_curve(self):
        made = fixtures_results.load()
        stated = made["conditions"]["act_clustered"]["block_fit"]
        fit = results.fit(made, "act_clustered")
        self.assertEqual(fit.blocks, ("T1a",))
        self.assertEqual((fit.r50_lo_mm, fit.r50_median_mm, fit.r50_hi_mm),
                         (stated["r50_lo_mm"], stated["r50_median_mm"], stated["r50_hi_mm"]))  # fmt: skip
        self.assertTrue(fit.bounded)
        self.assertEqual(len(fit.curve_d_mm), len(fit.curve_mean))
        self.assertEqual((fit.curve_d_mm[0], fit.curve_d_mm[1]), (0.0, 0.5))
        self.assertGreaterEqual(fit.curve_d_mm[-1], max(t.d_mm for t in results.trials(made, "act_clustered")))
        pooled = results.fit(made, "act_clustered", "pooled")
        self.assertEqual(pooled.blocks, ("T1a", "T2"))
        self.assertIsNone(results.fit(made, "replay", "pooled"))
        with self.assertRaises(ValueError):
            results.fit(made, "replay", "other")

    def test_what_an_open_interval_does_not_establish_may_be_null_and_nothing_else_may(self):
        def with_fit(**changes) -> dict:
            made = fixtures_results.load("open_above")
            made["conditions"]["act_clustered"]["block_fit"] |= changes
            return made

        stated = fixtures_results.load("open_above")["conditions"]["act_clustered"]["block_fit"]
        self.assertTrue(stated["open_above"] and not stated["open_below"])
        # The file gives numbers today; a file that gives null for the open end and for the radius is read as well.
        fit = results.fit(with_fit(r50_hi_mm=None, r50_median_mm=None), "act_clustered")
        self.assertEqual((fit.r50_lo_mm, fit.r50_median_mm, fit.r50_hi_mm), (stated["r50_lo_mm"], None, None))
        self.assertFalse(fit.bounded)
        # The end the trials do close must be a number, and so must every end of a bounded interval.
        with self.assertRaisesRegex(ResultsError, r"block_fit\.r50_lo_mm should be a number"):
            results.fit(with_fit(r50_lo_mm=None), "act_clustered")
        for key in ("r50_lo_mm", "r50_median_mm", "r50_hi_mm"):
            bounded = fixtures_results.load("base")
            bounded["conditions"]["replay"]["block_fit"][key] = None
            with self.assertRaisesRegex(ResultsError, key):
                results.fit(bounded, "replay")
        missing = with_fit()
        del missing["conditions"]["act_clustered"]["block_fit"]["r50_hi_mm"]
        with self.assertRaisesRegex(ResultsError, "r50_hi_mm is missing"):
            results.fit(missing, "act_clustered")
        with self.assertRaisesRegex(ResultsError, "not inside its interval"):
            results.fit(with_fit(r50_lo_mm=-1.0), "act_clustered")

    def test_the_curve_is_read_between_its_points(self):
        fit = results.fit(fixtures_results.load(), "act_clustered")
        self.assertEqual(fit.mean_at(0.0), fit.curve_mean[0])
        self.assertEqual(fit.mean_at(0.5), fit.curve_mean[1])
        self.assertAlmostEqual(fit.mean_at(0.25), (fit.curve_mean[0] + fit.curve_mean[1]) / 2.0)
        self.assertEqual(fit.mean_at(-3.0), fit.curve_mean[0])
        self.assertEqual(fit.mean_at(1e6), fit.curve_mean[-1])
        self.assertEqual(results.interpolate([0.0, 10.0, 20.0], [1.0, 0.5, 0.0], 15.0), 0.25)

    def test_a_fit_that_is_missing_or_makes_no_sense_is_refused(self):
        def with_fit(**changes) -> dict:
            made = fixtures_results.load()
            made["conditions"]["replay"]["block_fit"] |= changes
            return made

        missing = fixtures_results.load()
        del missing["conditions"]["replay"]["block_fit"]
        with self.assertRaisesRegex(ResultsError, r"conditions\.replay\.block_fit is missing"):
            results.fit(missing, "replay")
        curve = fixtures_results.load()["conditions"]["replay"]["block_fit"]["curve"]
        for changes, said in (
            ({"r50_median_mm": 99.0}, "not inside its interval"),
            ({"open_above": "no"}, "open_above"),
            ({"curve": curve | {"mean": curve["mean"][:-1]}}, "one length"),
            ({"curve": curve | {"d_mm": [*curve["d_mm"][:-1], 0.0]}}, "must increase"),
            ({"curve": curve | {"hi": [1.2, *curve["hi"][1:]]}}, "not a probability"),
            ({"curve": {"d_mm": [], "mean": [], "lo": [], "hi": []}}, "not empty"),
            ({"blocks": "T1b"}, "blocks"),
        ):
            with self.assertRaisesRegex(ResultsError, said):
                results.fit(with_fit(**changes), "replay")
        with self.assertRaisesRegex(ResultsError, "neither an object nor null"):
            results.fit(fixtures_results.changed(**{"conditions.replay.block_fit": [1]}), "replay")


class BinTests(unittest.TestCase):
    def test_trials_are_counted_in_stretches_closed_below_and_open_above(self):
        trials = [ScoredTrial("T1a", d, success) for d, success in
                  ((0.0, True), (9.99, True), (10.0, False), (19.5, True), (30.0, False), (30.0, False))]  # fmt: skip
        bins = results.binned(trials, 10.0, 0.95)
        self.assertEqual([(b.lo_mm, b.hi_mm, b.n, b.k) for b in bins],
                         [(0.0, 10.0, 2, 2), (10.0, 20.0, 2, 1), (30.0, 40.0, 2, 0)])  # fmt: skip
        self.assertEqual((bins[1].wilson_lo, bins[1].wilson_hi), wilson_interval(1, 2, 0.95))
        self.assertEqual(bins[1].rate, 0.5)
        self.assertEqual(results.binned([], 10.0, 0.95), [])

    def test_the_bins_of_a_file_are_those_of_all_the_conditions_trials(self):
        for name in fixtures_results.CASES:
            made = fixtures_results.load(name)
            for condition in results.CONDITIONS:
                trials = results.trials(made, condition)
                written = results.file_bins(made, condition)
                self.assertTrue(results.same_bins(written, results.binned(trials, 10.0, 0.95)), (name, condition))
                self.assertEqual(sum(b.n for b in written), len(trials))

    def test_a_bin_without_a_trial_is_passed_over_even_with_no_interval(self):
        made = fixtures_results.load()
        bins = made["conditions"]["replay"]["bins"]
        before = results.file_bins(made, "replay")
        bins.append({"lo_mm": 70.0, "hi_mm": 80.0, "n": 0, "k": 0, "wilson_lo": None, "wilson_hi": None})
        self.assertEqual(results.file_bins(made, "replay"), before)

    def test_two_lists_of_bins_are_the_same_only_when_counts_edges_and_intervals_are(self):
        first = [Bin(0.0, 10.0, 4, 3, 0.3, 0.95)]
        self.assertTrue(results.same_bins(first, [Bin(0.0, 10.0, 4, 3, 0.3 + 1e-9, 0.95)]))
        for other in ([Bin(0.0, 10.0, 4, 2, 0.3, 0.95)], [Bin(0.0, 10.0, 5, 3, 0.3, 0.95)], [], first * 2,
                      [Bin(10.0, 20.0, 4, 3, 0.3, 0.95)], [Bin(0.0, 10.0, 4, 3, 0.31, 0.95)],
                      [Bin(0.0, 10.0, 4, 3, 0.3, 0.9)]):  # fmt: skip
            self.assertFalse(results.same_bins(first, other))


class WorkspaceTests(unittest.TestCase):
    def test_a_file_has_a_workspace_test_when_it_lists_pairs(self):
        self.assertTrue(results.has_workspace(fixtures_results.load()))
        self.assertFalse(results.has_workspace(fixtures_results.load("threshold_blocks_only")))
        self.assertFalse(results.has_workspace(fixtures_results.changed(**{"workspace.pairs": []})))
        without = fixtures_results.load()
        del without["workspace"]
        with self.assertRaisesRegex(ResultsError, "workspace is missing"):
            results.has_workspace(without)
        with self.assertRaisesRegex(ResultsError, "no workspace test"):
            results.placements(fixtures_results.load("threshold_blocks_only"), "act_clustered")
        with self.assertRaises(ValueError):
            results.placements(fixtures_results.load(), "replay")

    def test_a_placement_is_where_the_policys_own_trial_was_when_the_file_says(self):
        made = fixtures_results.load()
        pair = made["workspace"]["pairs"][4]
        for policy in results.POLICIES:
            placement = results.placements(made, policy)[4]
            self.assertEqual((placement.x_mm, placement.y_mm), (pair[policy]["x_mm"], pair[policy]["y_mm"]))
            self.assertEqual((placement.success, placement.d_mm, placement.p_lookup),
                             (pair[policy]["success"], pair[policy]["d_mm"], pair[policy]["p_lookup"]))  # fmt: skip
            self.assertTrue(placement.own_position)
            self.assertIsNone(placement.region)
        self.assertNotEqual(pair["act_clustered"]["x_mm"], pair["act_spread"]["x_mm"])

    def test_without_its_own_position_a_placement_is_at_the_pairs(self):
        made = fixtures_results.contract_only("held_out_placements")
        pair = made["workspace"]["pairs"][4]
        for policy in results.POLICIES:
            placement = results.placements(made, policy)[4]
            self.assertEqual((placement.x_mm, placement.y_mm), (pair["x_mm"], pair["y_mm"]))
            self.assertFalse(placement.own_position)
            self.assertIsNone(placement.region)
        full = results.placements(fixtures_results.load("held_out_placements"), "act_spread")
        self.assertEqual({placement.region for placement in full}, set(results.REGIONS))

    def test_the_lookup_prediction_and_the_count_of_successes(self):
        made = fixtures_results.load()
        stated = made["workspace"]["lookup"]["act_spread"]
        lookup = results.lookup(made, "act_spread")
        self.assertEqual((lookup.expected, lookup.lo, lookup.hi), (stated["expected"], stated["lo"], stated["hi"]))
        self.assertEqual((lookup.observed, lookup.verdict), (stated["observed"], stated["verdict"]))
        count = results.success_count(made, "act_spread")
        self.assertEqual((count.k, count.n, count.lookup_expected), (lookup.observed, 20, lookup.expected))
        without = fixtures_results.changed(**{"workspace.lookup": None})
        self.assertIsNone(results.lookup(without, "act_spread"))
        self.assertEqual(results.success_count(without, "act_spread"), Count(count.k, 20, None))
        missing = fixtures_results.load()
        del missing["workspace"]["lookup"]
        with self.assertRaisesRegex(ResultsError, r"workspace\.lookup is missing"):
            results.lookup(missing, "act_spread")
        with self.assertRaisesRegex(ResultsError, "whole number"):
            results.lookup(fixtures_results.changed(**{"workspace.lookup.act_spread.lo": 11.5}), "act_spread")

    def test_the_demonstrations_of_a_set(self):
        made = fixtures_results.load()
        clustered = results.demonstrations(made, "clustered")
        self.assertEqual(len(clustered), 55)
        self.assertTrue(all(demo.trained for demo in clustered))
        first = made["demonstrations"]["clustered"][0]
        self.assertEqual((clustered[0].x_mm, clustered[0].y_mm), (first["x_mm"], first["y_mm"]))
        without = fixtures_results.changed(**{"demonstrations.spread": []})
        self.assertEqual(results.demonstrations(without, "spread"), [])
        with self.assertRaises(ResultsError):
            results.demonstrations(fixtures_results.changed(**{"demonstrations.spread": None}), "spread")


class CountsByRegionTests(unittest.TestCase):
    def test_the_files_own_counts_by_region_are_used_trained_cells_first(self):
        made = fixtures_results.load("held_out_placements")
        for policy in results.POLICIES:
            counts, notes = results.region_counts(made, policy)
            self.assertEqual(notes, [])
            self.assertEqual(list(counts), ["trained_cells", "held_out_cells"])
            for region, count in counts.items():
                stated = made["workspace"]["by_region"][region][policy]
                self.assertEqual((count.k, count.n, count.lookup_expected),
                                 (stated["k"], stated["n"], stated["lookup_expected"]))  # fmt: skip
            total = results.success_count(made, policy)
            self.assertEqual((sum(c.k for c in counts.values()), sum(c.n for c in counts.values())), (total.k, 26))
            self.assertAlmostEqual(sum(c.lookup_expected for c in counts.values()), total.lookup_expected, places=2)

    def test_an_interval_of_the_predicted_count_by_region_is_read_when_the_file_gives_one(self):
        made = fixtures_results.load("held_out_placements")
        self.assertEqual([(c.lookup_lo, c.lookup_hi) for c in results.region_counts(made, "act_spread")[0].values()],
                         [(None, None), (None, None)])  # fmt: skip
        made["workspace"]["by_region"]["trained_cells"]["act_spread"] |= {"lookup_lo": 9, "lookup_hi": 13}
        made["workspace"]["by_region"]["held_out_cells"]["act_spread"] |= {"lookup_lo": 2, "lookup_hi": None}
        counts, notes = results.region_counts(made, "act_spread")
        self.assertEqual(notes, [])
        self.assertEqual((counts["trained_cells"].lookup_lo, counts["trained_cells"].lookup_hi), (9, 13))
        self.assertEqual((counts["held_out_cells"].lookup_lo, counts["held_out_cells"].lookup_hi), (None, None))
        # An interval that cannot be one makes the whole of by_region unreadable: the counts come from the pairs.
        made["workspace"]["by_region"]["trained_cells"]["act_spread"] |= {"lookup_lo": 12, "lookup_hi": 40}
        counts, notes = results.region_counts(made, "act_spread")
        self.assertEqual((counts["trained_cells"].lookup_lo, counts["trained_cells"].n), (None, 13))
        self.assertIn("the interval of the predicted count runs from 12 to 40 of 13", notes[0])

    def test_without_them_the_counts_are_made_from_the_pairs_that_name_their_region(self):
        made = fixtures_results.load("held_out_placements")
        stated, _ = results.region_counts(made, "act_spread")
        del made["workspace"]["by_region"]
        counted, notes = results.region_counts(made, "act_spread")
        self.assertEqual(notes, [])
        self.assertEqual(list(counted), list(stated))
        for region in stated:
            self.assertEqual((counted[region].k, counted[region].n), (stated[region].k, stated[region].n))
            self.assertAlmostEqual(counted[region].lookup_expected, stated[region].lookup_expected, places=2)

    def test_a_file_that_says_nothing_of_regions_has_no_count_by_region(self):
        self.assertEqual(results.region_counts(fixtures_results.load("base"), "act_clustered"), (None, []))
        self.assertEqual(results.region_counts(fixtures_results.contract_only("held_out_placements"), "act_spread"),
                         (None, []))  # fmt: skip

    def test_without_a_prediction_a_region_has_its_count_and_no_expected_number(self):
        made = fixtures_results.changed("held_out_placements", **{"workspace.lookup": None})
        del made["workspace"]["by_region"]
        counted, _ = results.region_counts(made, "act_clustered")
        self.assertEqual([count.lookup_expected for count in counted.values()], [None, None])
        self.assertEqual([count.n for count in counted.values()], [13, 13])

    def test_counts_by_region_in_another_shape_are_passed_over_with_a_note(self):
        made = fixtures_results.load("held_out_placements")
        good, _ = results.region_counts(made, "act_clustered")
        # Policy first and region second: not the shape this code reads.
        made["workspace"]["by_region"] = {policy: {"trained_cells": {"k": 1, "n": 13}} for policy in results.POLICIES}
        counts, notes = results.region_counts(made, "act_clustered")
        self.assertEqual({region: (c.k, c.n) for region, c in counts.items()}, {r: (c.k, c.n) for r, c in good.items()})
        self.assertEqual(len(notes), 1)
        self.assertIn("workspace.by_region is not in the shape this code reads", notes[0])
        self.assertIn("made from the pairs", notes[0])
        # And with nothing to count them from, no count by region at all.
        for pair in made["workspace"]["pairs"]:
            del pair["region"]
        counts, notes = results.region_counts(made, "act_clustered")
        self.assertIsNone(counts)
        self.assertIn("no count by region is shown", notes[0])

    def test_counts_by_region_that_are_not_the_pairs_counts_are_reported(self):
        made = fixtures_results.load("held_out_placements")
        made["workspace"]["by_region"]["held_out_cells"]["act_spread"]["k"] += 1
        made["workspace"]["by_region"]["trained_cells"]["act_spread"]["lookup_expected"] += 1.0
        counts, notes = results.region_counts(made, "act_spread")
        self.assertEqual(len(notes), 2)
        self.assertRegex(notes[0], r"workspace\.by_region\.held_out_cells\.act_spread gives \d+ of 13, the pairs of "
                                   r"that region hold \d+ of 13")  # fmt: skip
        self.assertIn("lookup_expected", notes[1])
        self.assertEqual(results.region_counts(made, "act_clustered")[1], [])

    def test_a_pair_without_a_region_among_pairs_with_one_is_said(self):
        made = fixtures_results.load("held_out_placements")
        del made["workspace"]["by_region"]
        made["workspace"]["pairs"][0]["region"] = None
        counts, notes = results.region_counts(made, "act_spread")
        self.assertEqual(sum(count.n for count in counts.values()), 25)
        self.assertEqual(notes, ["1 of the 26 placements of act_spread name no region: they are counted under all "
                                 "placements only"])  # fmt: skip

    def test_a_region_of_another_name_comes_after_the_protocols_two(self):
        made = fixtures_results.load("held_out_placements")
        del made["workspace"]["by_region"]
        made["workspace"]["pairs"][0]["region"] = "margin"
        counts, _ = results.region_counts(made, "act_spread")
        self.assertEqual(list(counts), ["trained_cells", "held_out_cells", "margin"])


class PairChecksTests(unittest.TestCase):
    def test_the_synthetic_files_agree_with_themselves(self):
        for name in fixtures_results.CASES:
            made = fixtures_results.load(name)
            if results.has_workspace(made):
                for policy in results.POLICIES:
                    self.assertEqual(results.pair_checks(made, policy), [], (name, policy))

    def test_a_count_of_successes_that_is_not_the_pairs_is_an_error(self):
        made = fixtures_results.load()
        made["workspace"]["success"]["act_spread"]["k"] -= 1
        with self.assertRaisesRegex(ResultsError, r"workspace\.success\.act_spread counts \d+ of 20, and the pairs "
                                                  r"listed hold \d+ successes in 20 placements"):  # fmt: skip
            results.pair_checks(made, "act_spread")
        self.assertEqual(results.pair_checks(made, "act_clustered"), [])

    def test_an_observed_count_or_an_interval_that_cannot_be_is_an_error(self):
        made = fixtures_results.load()
        made["workspace"]["lookup"]["act_clustered"]["observed"] += 1
        with self.assertRaisesRegex(ResultsError, "observed"):
            results.pair_checks(made, "act_clustered")
        with self.assertRaisesRegex(ResultsError, "interval"):
            results.pair_checks(fixtures_results.changed(**{"workspace.lookup.act_spread.lo": 21}), "act_spread")

    def test_an_expected_count_that_the_pairs_do_not_add_up_to_is_a_note(self):
        made = fixtures_results.load()
        made["workspace"]["lookup"]["act_spread"]["expected"] += 2.0
        (note,) = results.pair_checks(made, "act_spread")
        self.assertRegex(note, r"workspace\.lookup\.act_spread\.expected is \d+\.\d\d, the pairs' p_lookup add up to")

    def test_without_a_prediction_only_the_counts_are_checked(self):
        self.assertEqual(results.pair_checks(fixtures_results.changed(**{"workspace.lookup": None}), "act_spread"), [])


if __name__ == "__main__":
    unittest.main()
