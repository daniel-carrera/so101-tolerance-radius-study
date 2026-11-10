"""The paper's number macros: every name of the list is accounted for, and every value is written as the list says."""

import copy
import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import fixtures_experiment as fixtures
from tolerance_radius.report import macros

_SPEC = importlib.util.spec_from_file_location("make_numbers_stub", fixtures.REPO / "paper" / "make_numbers_stub.py")
stub = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stub
_SPEC.loader.exec_module(stub)

MINUS = r"\\ensuremath\{-\}"
#: How a value of each kind looks, as the head of `paper/generated/MACROS.md` states it. A macro with two decimals
#: says so in its own description.
FORMS = {
    "distance": rf"({MINUS})?\d+\.\d",
    "one_decimal": r"\d+\.\d",
    "two_decimals": r"\d+\.\d\d",
    "count": r"\d{1,3}(,\d{3})*",
    "percent": r"\d{1,3}",
    "probability": r"0\.\d\d|<0\.01|>0\.99",
    "p_value": r"0\.\d\d|1\.00|<0\.01|>0\.99",
    "factor": r"\d\.\d{3}",
    "plain_number": r"\d{1,3}(,\d{3})*|\d+\.\d+",
}
#: Macros that are not numbers of one of those kinds, and what they hold instead.
OTHER_FORMS = {"voidReasons": r"[^\n]+", "lerobotCommit": r"[0-9a-f]{7,40}"}
SETTINGS = {"execClamp", "execEnsembleCoeff", "execControlRate", "execTimeLimit"}


def kind_stated(name: str, holds: str) -> str:
    """The kind of value a macro holds, read from its description in the list of `paper/make_numbers_stub.py`."""
    if name in OTHER_FORMS:
        return name
    if name in SETTINGS:
        return "plain_number"
    if holds.endswith("percent"):
        return "percent"
    if "three decimals" in holds:
        return "factor"
    if "p-value" in holds:
        return "p_value"
    if "probability" in holds:
        return "probability"
    if holds.endswith(", mm, two decimals"):
        return "two_decimals"
    if holds.endswith(", mm"):
        return "distance"
    if holds.endswith("one decimal"):
        return "one_decimal"
    return "count"


DESCRIPTIONS = {result.name: result.holds for group in stub.RESULTS for result in group.results}


class ListTests(unittest.TestCase):
    def test_every_macro_of_the_list_is_produced_or_named_as_one_no_record_can_fill(self):
        produced = [macro.name for macro in macros.MACROS]
        self.assertEqual(len(set(produced)), len(produced), "a macro has one source")
        self.assertEqual(set(produced) & set(macros.NOT_PRODUCED), set())
        self.assertEqual(set(produced) | set(macros.NOT_PRODUCED), set(stub.result_names()))

    def test_each_macro_is_written_in_the_form_its_description_states(self):
        writers = {"str": "lerobotCommit", "void_reasons_text": "voidReasons"}
        for macro in macros.MACROS:
            stated = kind_stated(macro.name, DESCRIPTIONS[macro.name])
            with self.subTest(macro=macro.name):
                self.assertEqual(writers.get(macro.write.__name__, macro.write.__name__), stated)

    def test_every_source_is_one_line_that_names_no_folder_of_this_machine(self):
        for macro in macros.MACROS:
            self.assertRegex(macro.source, r"^[^\n\r]+$")
            self.assertNotIn(str(Path.home()), macro.source)
            self.assertNotRegex(macro.source, r"(^|[\s(])/")
        for why in macros.NOT_PRODUCED.values():
            self.assertRegex(why, r"^[^\n\r]+$")


class ValueTests(unittest.TestCase):
    """The values of the full synthetic experiment, in which every block was run."""

    @classmethod
    def setUpClass(cls):
        cls.experiment, cls.made = fixtures.main_run()
        cls.values = cls.made.values

    def test_every_macro_it_can_produce_has_a_value_when_every_block_was_run(self):
        self.assertEqual(set(self.values), {macro.name for macro in macros.MACROS})
        self.assertEqual(len(self.values), len(stub.result_names()) - len(macros.NOT_PRODUCED))

    def test_every_value_has_the_form_the_macro_list_states(self):
        for name, value in self.values.items():
            stated = kind_stated(name, DESCRIPTIONS[name])
            form = OTHER_FORMS.get(stated) or FORMS[stated]
            with self.subTest(macro=name, kind=stated):
                self.assertRegex(value["value"], rf"^(?:{form})$")
                self.assertTrue(value["source"])

    def test_the_stub_takes_every_value_and_writes_it_under_its_name(self):
        checked = stub.checked_values(self.values)
        self.assertEqual(
            {name: text for name, (text, _) in checked.items()}, {n: v["value"] for n, v in self.values.items()}
        )
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "generated"
            written = stub.write_generated(
                self.values,
                out_dir=out,
                protocol_toml=self.experiment.root / "config" / "protocol.toml",
                by_hand_tex=Path(folder) / "none.tex",
            )
            self.assertEqual({path.name for path in written}, {"numbers.tex", "protocol.tex", "MACROS.md"})
            numbers = (out / "numbers.tex").read_text(encoding="utf-8")
            listing = (out / "MACROS.md").read_text(encoding="utf-8")
        defined = stub.read_definitions(numbers)
        self.assertEqual(set(defined), set(stub.result_names()))
        for name in stub.result_names():
            body, comment = defined[name]
            if name in self.values:
                self.assertEqual(body, self.values[name]["value"])
                self.assertIn(f"src: {self.values[name]['source']}", comment)
            else:
                self.assertEqual(body, stub.PLACEHOLDER)
                self.assertIn(name, macros.NOT_PRODUCED)
        self.assertIn(f"{len(self.values)} of {len(stub.result_names())} hold a value", listing)
        self.assertEqual(sorted(stub.holds_values(numbers)), sorted(self.values))

    def test_the_values_are_the_results_rounded_for_the_paper(self):
        results = self.made.results
        clustered, replay = results["conditions"]["act_clustered"], results["conditions"]["replay"]
        clustered_starts = fixtures.audited_starts("clustered")
        expected = {
            "radiusClustered": f"{clustered['pooled_fit']['r50_median_mm']:.1f}",
            "radiusClusteredThreshold": f"{clustered['block_fit']['r50_median_mm']:.1f}",
            "radiusClusteredThresholdLow": f"{clustered['block_fit']['r50_lo_mm']:.1f}",
            "radiusReplayHigh": f"{replay['block_fit']['r50_hi_mm']:.1f}",
            "nTrialsClustered": str(clustered["n_scored"]),
            "nTrialsClusteredThreshold": str(results["blocks"]["T1a"]["n_scored"]),
            "nVoidClusteredThreshold": str(results["blocks"]["T1a"]["n_void"]),
            "nVoidClustered": str(clustered["n_void"]),
            "successZeroClustered": str(clustered["anchors"]["zero"]["k"]),
            "nFarClustered": str(clustered["anchors"]["far"]["n"]),
            "nReplayEpisodes": str(self.experiment.protocol["blocks"]["T1b"].get("replay_demos", 4)),
            "nPairs": str(results["workspace"]["n_pairs"]),
            "nTrialsWorkspace": str(results["blocks"]["T2"]["n_scored"]),
            "pairsDiscordant": str(
                results["workspace"]["table"]["only_spread"] + results["workspace"]["table"]["only_clustered"]
            ),
            "workspaceSuccessSpread": str(results["workspace"]["success"]["act_spread"]["k"]),
            "lookupClusteredLow": str(results["workspace"]["lookup"]["act_clustered"]["lo"]),
            "nTrialsBasketNamed": str(results["scene"]["n_named_basket"]),
            "nTrialsWristNamed": "1",
            "nTrialsBasketMoved": "1",
            "basketStartMax": f"{results['scene']['basket_start_max_mm']:.1f}",
            "wristStartMax": "2.4",
            "nSecondScored": str(self.experiment.second_scoring["all"]["n_items"]),
            "nSecondScoredAgree": str(self.experiment.second_scoring["all"]["n_agree"]),
            "nSecondScoredCannotTell": "1",
            "nTrialsTotal": str(results["provenance"]["n_scored"]),
            "nVoidTotal": str(results["provenance"]["n_void"]),
            "nScaleRuns": "4",
            "nScaleSteps": "20",
            "scaleFactor": "1.030",
            "scaleFactorMin": "1.027",
            "scaleFactorMax": "1.033",
            "nByEye": "20",
            "nOperatorsByEye": "1",
            "nDemosTrainedClustered": str(len(clustered_starts)),
            "nRerecordedClustered": str(
                sum(fixtures.takes_before_saving(s.episode_index) > 0 for s in clustered_starts)
            ),
            "nRecordingDays": "1",
            "nTrialDays": "2",
            "nOperators": "1",
            "nDeviations": "0",
            "evalCheckpoint": "80,000",
            "execClamp": "15",
            "execEnsembleCoeff": "0.01",
            "execControlRate": "30",
            "execControlRateAchieved": macros.one_decimal(
                results["conduct"]["execution"]["achieved_hz"]["policy"]["median"]
            ),
            "execTimeLimit": "45",
            "viewCreptMedian": macros.rounded(results["conduct"]["view_crept"]["median_mm"], 2),
            "viewCreptMax": f"{results['conduct']['view_crept']['max_mm']:.2f}",
            "lerobotCommit": "6adf515116",
            "simTruthRadius": "20.0",
            "simTruthWidth": "3.0",
            "simReplications": "300",
            "simBias": r"\ensuremath{-}0.2",
            "simRmse": "1.6",
            "simCoverage": "97",
            "simPairedRateLow": "40",
            "simPairedRateHigh": "90",
            "simPairedPower": "88",
            "simWidthCliff": "1.0",
            "simWidthMiddling": "3.0",
            "simWidthSlope": "10.0",
        }
        for name, text in expected.items():
            with self.subTest(macro=name):
                self.assertEqual(self.values[name]["value"], text)

    def test_the_counts_on_a_block_s_row_of_the_paper_s_table_are_those_of_the_threshold_block(self):
        conditions = self.made.results["conditions"]
        for suffix, condition in (("Replay", "replay"), ("Clustered", "act_clustered"), ("Spread", "act_spread")):
            block = conditions[condition]["block_fit"]
            with self.subTest(condition=condition):
                self.assertEqual(self.values[f"maxSuccessDist{suffix}"]["value"], f"{block['max_success_d_mm']:.1f}")
                self.assertEqual(self.values[f"minFailureDist{suffix}"]["value"], f"{block['min_failure_d_mm']:.1f}")
                self.assertEqual(self.values[f"successZero{suffix}"]["value"], str(block["zero_k"]))
                self.assertEqual(self.values[f"nZero{suffix}"]["value"], str(block["zero_n"]))
                self.assertIn("alone", self.values[f"maxSuccessDist{suffix}"]["source"])
                for stem in ("maxSuccessDist", "minFailureDist", "successZero", "nZero"):
                    self.assertRegex(
                        DESCRIPTIONS[f"{stem}{suffix}"], r"block T\w+|threshold block", "the list says which trials"
                    )
        # A workspace trial that fails nearer, or succeeds farther, than any trial of the threshold block changes
        # the condition's own two distances and the pooled fit's, and not the macros.
        changed = copy.deepcopy(self.made.results)
        spread = changed["conditions"]["act_spread"]
        spread["block_fit"].update(max_success_d_mm=21.25, min_failure_d_mm=7.5)
        spread["pooled_fit"].update(max_success_d_mm=44.4, min_failure_d_mm=0.3)
        spread.update(max_success_d_mm=44.4, min_failure_d_mm=0.3)
        values = macros.macro_values(changed)
        self.assertEqual(
            (values["maxSuccessDistSpread"]["value"], values["minFailureDistSpread"]["value"]), ("21.3", "7.5")
        )

    def test_the_block_fits_and_their_plateau_checks_have_macros_for_all_three_conditions(self):
        conditions = self.made.results["conditions"]
        for prefix, condition in (
            ("radiusReplay", "replay"),
            ("radiusClusteredThreshold", "act_clustered"),
            ("radiusSpreadThreshold", "act_spread"),
        ):
            block = conditions[condition]["block_fit"]
            check = block["sensitivity"]
            with self.subTest(condition=condition):
                self.assertEqual(self.values[prefix]["value"], f"{block['r50_median_mm']:.1f}")
                self.assertEqual(self.values[f"{prefix}Low"]["value"], f"{block['r50_lo_mm']:.1f}")
                self.assertEqual(self.values[f"{prefix}High"]["value"], f"{block['r50_hi_mm']:.1f}")
                self.assertEqual(self.values[f"{prefix}Wide"]["value"], f"{check['r50_median_mm']:.1f}")
                self.assertEqual(self.values[f"{prefix}WideLow"]["value"], f"{check['r50_lo_mm']:.1f}")
                self.assertEqual(self.values[f"{prefix}WideHigh"]["value"], f"{check['r50_hi_mm']:.1f}")
        pooled = conditions["act_spread"]["pooled_fit"]
        self.assertEqual(self.values["radiusSpread"]["value"], f"{pooled['r50_median_mm']:.1f}")

    def test_the_pooled_fits_of_the_two_policies_have_macros_for_their_plateau_checks_too(self):
        conditions = self.made.results["conditions"]
        for prefix, condition in (("radiusClustered", "act_clustered"), ("radiusSpread", "act_spread")):
            pooled, block = conditions[condition]["pooled_fit"], conditions[condition]["block_fit"]
            check = pooled["sensitivity"]
            with self.subTest(condition=condition):
                self.assertNotEqual(pooled["n"], block["n"], "the pooled fit has the workspace trials as well")
                self.assertEqual(self.values[f"{prefix}Wide"]["value"], macros.distance(check["r50_median_mm"]))
                self.assertEqual(self.values[f"{prefix}WideLow"]["value"], macros.distance(check["r50_lo_mm"]))
                self.assertEqual(self.values[f"{prefix}WideHigh"]["value"], macros.distance(check["r50_hi_mm"]))
                self.assertIn(f"pooled fit of {condition} made again", self.values[f"{prefix}Wide"]["source"])
                self.assertRegex(DESCRIPTIONS[f"{prefix}Wide"], r"all its scored trials.*longer list of lapse values")

    def test_the_shift_of_every_plateau_check_the_paper_quotes_has_a_macro(self):
        conditions = self.made.results["conditions"]
        for name, condition, kind in (
            ("plateauShiftClusteredThreshold", "act_clustered", "block_fit"),
            ("plateauShiftClustered", "act_clustered", "pooled_fit"),
            ("plateauShiftReplay", "replay", "block_fit"),
            ("plateauShiftSpread", "act_spread", "pooled_fit"),
        ):
            fit = conditions[condition][kind]
            check = fit["sensitivity"]
            with self.subTest(macro=name):
                self.assertEqual(self.values[name]["value"], macros.distance(check["median_shift_mm"]))
                self.assertAlmostEqual(
                    check["median_shift_mm"], check["r50_median_mm"] - fit["r50_median_mm"], places=9
                )
                named = f"{'block' if kind == 'block_fit' else 'pooled'} fit of {condition}"
                self.assertIn(named, self.values[name]["source"])

    def test_a_shift_is_the_second_fit_s_median_minus_the_first_s_with_its_sign(self):
        def results(shift) -> dict:
            check = {"r50_median_mm": 20.0 + (shift or 0.0), "median_shift_mm": shift}
            fit = {"r50_median_mm": 20.0, "sensitivity": check}
            return {"conditions": {"act_clustered": {"block_fit": fit, "pooled_fit": fit}}}

        for shift, text in ((3.5, "3.5"), (6.0, "6.0"), (-2.25, r"\ensuremath{-}2.3"), (0.0, "0.0"), (-0.04, "0.0")):
            values = macros.macro_values(results(shift))
            with self.subTest(shift=shift):
                self.assertEqual(values["plateauShiftClusteredThreshold"]["value"], text)
                self.assertEqual(values["plateauShiftClustered"]["value"], text)
        without = {"conditions": {"act_clustered": {"block_fit": {"r50_median_mm": 20.0, "sensitivity": None}}}}
        self.assertEqual([name for name in macros.macro_values(without) if "plateauShift" in name], [])

    def test_the_workspace_test_has_macros_by_region(self):
        by_region = self.made.results["workspace"]["by_region"]
        for region, name in (("trained_cells", "Trained"), ("held_out_cells", "HeldOut")):
            self.assertEqual(self.values[f"nPairs{name}"]["value"], str(by_region[region]["n_pairs"]))
            for policy, suffix in (("act_clustered", "Clustered"), ("act_spread", "Spread")):
                found = by_region[region][policy]
                self.assertEqual(self.values[f"workspaceSuccess{suffix}{name}"]["value"], str(found["k"]))
                self.assertEqual(
                    self.values[f"lookupPredicted{suffix}{name}"]["value"], macros.one_decimal(found["lookup_expected"])
                )
        total = sum(int(self.values[f"nPairs{name}"]["value"]) for name in ("Trained", "HeldOut"))
        self.assertEqual(total, int(self.values["nPairs"]["value"]))

    def test_the_simulation_has_macros_by_block_and_width(self):
        blocks = self.made.results["simulation"]["blocks"]
        for suffix, block in (("Replay", "T1b"), ("Clustered", "T1a"), ("Spread", "T3")):
            for curve in ("cliff", "middling", "slope"):
                found = blocks[block]["by_width"][curve]
                self.assertEqual(
                    self.values[f"simRmse{suffix}{curve.capitalize()}"]["value"], macros.distance(found["rmse_mm"])
                )
                self.assertEqual(
                    self.values[f"simIntervalWidth{suffix}{curve.capitalize()}"]["value"],
                    macros.distance(found["interval_width_mm"]),
                )
            self.assertEqual(
                self.values[f"simOpenBelow{suffix}"]["value"],
                macros.percent(blocks[block]["open_below_falling_at_zero"]),
            )
        self.assertEqual(
            self.values["simOpenBelow"]["value"],
            macros.percent(self.made.results["simulation"]["open_below_falling_at_zero"]),
        )

    def test_the_reasons_of_void_trials_are_one_escaped_text_with_counts(self):
        text = self.values["voidReasons"]["value"]
        self.assertRegex(text, r"^bumped the table \(\d+\), ")
        self.assertIn(r"camera dropped 50\% of frames\_in \#2 (", text)
        self.assertNotRegex(text, r"(?<!\\)[%#&_$]")
        total = sum(int(count) for count in re.findall(r"\((\d+)\)", text))
        self.assertEqual(total, self.made.results["provenance"]["n_void"])


class FormatTests(unittest.TestCase):
    def test_a_number_is_rounded_half_up_on_its_decimal_form(self):
        cases = {
            (2.25, 1): "2.3",
            (2.35, 1): "2.4",
            (0.125, 2): "0.13",
            (21.0, 1): "21.0",
            (9.96, 1): "10.0",
            (0.04, 1): "0.0",
            (1234.5, 0): "1235",
            (1.0305, 3): "1.031",
        }
        for (value, decimals), text in cases.items():
            with self.subTest(value=value):
                self.assertEqual(macros.rounded(value, decimals), text)

    def test_a_negative_number_carries_a_sign_that_prints_in_text_and_in_mathematics(self):
        self.assertEqual(macros.distance(-1.5), r"\ensuremath{-}1.5")
        self.assertEqual(macros.distance(-11.04), r"\ensuremath{-}11.0")
        self.assertEqual(macros.distance(-0.04), "0.0", "a number that rounds to zero has no sign")
        self.assertEqual(macros.distance(-0.05), r"\ensuremath{-}0.1")

    def test_a_share_is_a_whole_percentage(self):
        cases = {0.879086: "88", 0.285: "29", 0.005: "1", 0.0049: "0", 1.0: "100", 0.9733: "97", 0.645669565: "65"}
        for share, text in cases.items():
            with self.subTest(share=share):
                self.assertEqual(macros.percent(share), text)

    def test_a_probability_is_two_decimals_or_a_bound(self):
        cases = {
            0.3845: "0.38",
            0.385: "0.39",
            0.0099: "<0.01",
            0.0: "<0.01",
            0.01: "0.01",
            0.99: "0.99",
            0.9981: ">0.99",
            1.0: ">0.99",
            0.95: "0.95",
        }
        for value, text in cases.items():
            with self.subTest(value=value):
                self.assertEqual(macros.probability(value), text)

    def test_a_p_value_is_one_only_when_it_is_exactly_one(self):
        cases = {1.0: "1.00", 0.9999: ">0.99", 0.2668: "0.27", 0.039: "0.04", 0.0049: "<0.01", 0.004: "<0.01"}
        for value, text in cases.items():
            with self.subTest(value=value):
                self.assertEqual(macros.p_value(value), text)

    def test_counts_factors_and_settings(self):
        self.assertEqual([macros.count(n) for n in (0, 20, 1234, 80000)], ["0", "20", "1,234", "80,000"])
        self.assertEqual(macros.count(4.0), "4")
        for bad in (True, 4.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                macros.count(bad)
        self.assertEqual([macros.factor(f) for f in (1.03025, 1.0, 0.9996)], ["1.030", "1.000", "1.000"])
        self.assertEqual(
            [macros.plain_number(v) for v in (15.0, 30, 0.01, 30.0, 2.5)], ["15", "30", "0.01", "30", "2.5"]
        )
        self.assertEqual(macros.one_decimal(2.0833), "2.1")

    def test_typed_words_are_made_safe_for_tex(self):
        cases = {
            "camera dropped 50% of frames_in #2": r"camera dropped 50\% of frames\_in \#2",
            "cost $5 & more": r"cost \$5 \& more",
            "a {brace} and a \\ slash": r"a \textbraceleft{}brace\textbraceright{} and a \textbackslash{} slash",
            "x < y > z | w": r"x \textless{} y \textgreater{} z \textbar{} w",
            "about ~3 mm, 2^3": r"about \textasciitilde{}3 mm, 2\textasciicircum{}3",
            "two\nlines\tand   spaces ": "two lines and spaces",
            "what happened?? no idea???": "what happened? no idea?",
            "caf\u00e9 \u2013 \u201cbumped\u201d \u2026": 'cafe - "bumped" ...',
            "\u6f22\u5b57 only": "only",
            "a `quoted` word": "a 'quoted' word",
        }
        for typed, safe in cases.items():
            with self.subTest(typed=typed):
                self.assertEqual(macros.tex_escaped(typed), safe)
                self.assertTrue(macros.tex_escaped(typed).isascii())

    def test_whatever_is_typed_the_stub_takes_the_text(self):
        awful = "50% {of} $it_ #1 & ~ ^ \\ <|> ?? ??? caf\u00e9\n\ttwo  lines `x` }{"
        text = macros.void_reasons_text([{"reason": awful, "n": 2}, {"reason": "bumped the table", "n": 1}])
        self.assertEqual(stub.checked_values({"voidReasons": text})["voidReasons"][0], text)
        self.assertTrue(text.endswith(", bumped the table (1)"))

    def test_reasons_are_joined_with_their_counts(self):
        reasons = [{"reason": "bumped the table", "n": 2}, {"reason": "camera fault", "n": 1}]
        self.assertEqual(macros.void_reasons_text(reasons), "bumped the table (2), camera fault (1)")
        self.assertEqual(macros.void_reasons_text([]), "none")
        with_comma = [{"reason": "cube knocked, then replaced", "n": 1}, {"reason": "camera fault", "n": 1}]
        self.assertEqual(macros.void_reasons_text(with_comma), "cube knocked, then replaced (1); camera fault (1)")


class RepositorySimulationTests(unittest.TestCase):
    """The repository's own procedure simulation, read as the macros read it, against the simulation's own report."""

    @classmethod
    def setUpClass(cls):
        from tolerance_radius.report import bookkeeping, inputs

        folder = fixtures.REPO / "analysis" / "simulation"
        cls.tables = inputs.read_simulation(folder, fixtures.REPO)
        cls.report = (folder / "procedure.md").read_text(encoding="utf-8")
        largest = {
            match.group(1): float(match.group(2))
            for match in re.finditer(r"(?m)^\| (T\w+) \| \d+ \| \d+ \| [^|]+\| [^|]+\| (\d+) \|$", cls.report)
        }
        cls.largest = largest
        cls.found = bookkeeping.simulated_blocks(cls.tables.sweep, largest)

    def test_the_blocks_table_of_the_report_names_three_threshold_blocks(self):
        self.assertEqual(sorted(self.largest), ["T1a", "T1b", "T3"])

    def test_the_figures_by_width_are_the_first_numbers_of_the_report_s_table_1(self):
        cells = re.findall(
            r"(?m)^\| (T\w+) \| \d+ \| (\d+) \| ([\d.]+) \([\d.]+\) \| ([\d.]+) \([\d.]+\) \|", self.report
        )
        self.assertGreaterEqual(len(cells), 9)
        names = {"1": "cliff", "3": "middling", "10": "slope"}
        compared = 0
        for block, width, rmse, span in cells:
            if width not in names:
                continue
            found = self.found["blocks"][block]["by_width"][names[width]]
            with self.subTest(block=block, width=width):
                # The report rounds the figure itself to one decimal, the table holds it to four: the two agree to
                # within half a unit of the report's last digit, and print alike unless the figure sits on a tie.
                self.assertAlmostEqual(found["rmse_mm"], float(rmse), delta=0.0502)
                self.assertAlmostEqual(found["interval_width_mm"], float(span), delta=0.0502)
            compared += 1
        self.assertEqual(compared, 9)

    def test_the_share_open_below_is_that_of_the_report_s_table_6_and_of_its_reading(self):
        rows = re.findall(
            r"(?m)^\| (T\w+) \| already under way at 0 mm[^|]*\| (\d+) \| [\d.]+ \| [\d.]+ \| (\d+) \|", self.report
        )
        self.assertEqual(len(rows), 3)
        for block, truths, open_below in rows:
            entry = self.found["blocks"][block]
            self.assertEqual(entry["n_truths_falling_at_zero"], int(truths))
            self.assertAlmostEqual(100.0 * entry["open_below_falling_at_zero"], float(open_below), delta=0.502)
        reading = re.search(r"under two widths\),\s+it\s+is\s+open\s+below\s+in\s+(\d+)\s+%", self.report)
        self.assertIsNotNone(reading, "the report's reading no longer says how often the interval is open below")
        self.assertAlmostEqual(100.0 * self.found["open_below_falling_at_zero"], float(reading.group(1)), delta=0.502)

    def test_the_rule_for_a_curve_that_falls_at_zero_is_the_simulation_report_s_own(self):
        from tolerance_radius import procedure_report
        from tolerance_radius.report import bookkeeping

        self.assertEqual(bookkeeping.FALL_HALF_SPAN_WIDTHS, procedure_report.FALL_HALF_SPAN_WIDTHS)
        self.assertEqual(list(bookkeeping.SIMULATED_WIDTHS_MM.values()), [1.0, 3.0, 10.0])


class PlannedByRegionTests(unittest.TestCase):
    """The paper's constants for where the workspace test draws its placements, read from a protocol file."""

    def constants(self, workspace=None) -> dict[str, str]:
        return stub.protocol_texts(tomllib.loads(fixtures.protocol_text(workspace=workspace)))

    def test_the_repository_s_two_regions_add_up_to_the_placements_planned(self):
        texts = self.constants()
        block = tomllib.loads(fixtures.protocol_text())["blocks"]["T2"]
        stated = {entry["region"]: str(entry["pairs"]) for entry in block["regions"]}
        self.assertEqual(texts["nPairsTrainedPlanned"], stated["trained_cells"])
        self.assertEqual(texts["nPairsHeldOutPlanned"], stated["held_out_cells"])
        self.assertEqual(int(texts["nPairsTrainedPlanned"]) + int(texts["nPairsHeldOutPlanned"]),
                         int(texts["nPairsPlanned"]))
        self.assertEqual(int(texts["plannedWorkspace"]), 2 * int(texts["nPairsPlanned"]))

    def test_a_block_drawn_over_one_region_plans_none_in_the_other(self):
        texts = self.constants((("trained_cells", 7),))
        self.assertEqual((texts["nPairsTrainedPlanned"], texts["nPairsHeldOutPlanned"]), ("7", "0"))
        texts = self.constants((("held_out_cells", 5), ("trained_cells", 4)))
        self.assertEqual((texts["nPairsTrainedPlanned"], texts["nPairsHeldOutPlanned"]), ("4", "5"))

    def test_a_block_drawn_over_the_whole_grid_is_refused_because_the_method_s_sentence_would_not_hold(self):
        with self.assertRaisesRegex(ValueError, "method section"):
            self.constants((("test_region", 9),))


class PlateauCheckConstantTests(unittest.TestCase):
    """The paper's constants for the plateau check, read from a protocol file."""

    def test_the_longer_list_and_the_limit_are_the_protocol_s_analysis_section(self):
        protocol = tomllib.loads(fixtures.protocol_text())
        texts = stub.protocol_texts(protocol)
        stated = protocol["analysis"]
        self.assertEqual(stated["lapse_values_sensitivity"], list(fixtures.LAPSE_VALUES_SENSITIVITY))
        self.assertEqual(texts["lapseValuesWide"], "0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50")
        self.assertEqual(texts["plateauShiftLimit"], f"{stated['sensitivity_limit_mm']:g}")
        self.assertEqual(texts["plateauShiftLimit"], "5")
        self.assertTrue(texts["lapseValuesWide"].startswith(texts["lapseValues"]), "the same form, and a longer list")

    def test_they_are_written_into_protocol_tex_with_the_key_they_come_from(self):
        defined = stub.read_definitions(stub.render_protocol(tomllib.loads(fixtures.protocol_text())))
        self.assertEqual(defined["lapseValuesWide"][0], "0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50")
        self.assertIn("src: config/protocol.toml, analysis.lapse_values_sensitivity", defined["lapseValuesWide"][1])
        self.assertEqual(defined["plateauShiftLimit"][0], "5")
        self.assertIn("src: config/protocol.toml, analysis.sensitivity_limit_mm", defined["plateauShiftLimit"][1])

    def test_a_protocol_file_without_them_says_which_key_is_missing(self):
        protocol = tomllib.loads(fixtures.protocol_text(sensitivity=False))
        with self.assertRaisesRegex(KeyError, "lapse_values_sensitivity"):
            stub.protocol_texts(protocol)


class AchievedRateMacroTests(unittest.TestCase):
    def results(self, **policy) -> dict:
        rates = {"requested": 30, "median": None, "min": None, "max": None, "n": 0, "n_without": 0}
        return {"conduct": {"execution": {"achieved_hz": {"policy": rates | policy, "replay": dict(rates)}}}}

    def test_it_is_the_median_of_the_policy_trials_with_one_decimal(self):
        for median, text in ((29.7172794, "29.7"), (28.05, "28.1"), (30.0, "30.0"), (9.96, "10.0")):
            values = macros.macro_values(self.results(median=median, n=40))
            self.assertEqual(values["execControlRateAchieved"]["value"], text)
        self.assertRegex(DESCRIPTIONS["execControlRateAchieved"], r"scored policy trials.*one decimal$")
        self.assertNotIn("execControlRateAchieved", macros.NOT_PRODUCED)

    def test_without_a_policy_trial_that_has_a_rate_it_stays_a_placeholder(self):
        self.assertNotIn("execControlRateAchieved", macros.macro_values(self.results(n_without=12)))
        self.assertNotIn("execControlRateAchieved", macros.macro_values({"conduct": None}))
        replay_only = self.results()
        replay_only["conduct"]["execution"]["achieved_hz"]["replay"].update(median=29.9, n=22)
        self.assertNotIn("execControlRateAchieved", macros.macro_values(replay_only), "the replay's is not the paper's")


class ViewCreptMacroTests(unittest.TestCase):
    def values(self, **crept) -> dict:
        return macros.macro_values({"conduct": {"view_crept": crept or None}})

    def test_the_creep_is_printed_with_two_decimals_so_that_it_cannot_read_as_its_limit(self):
        values = self.values(median_mm=0.315, max_mm=0.99, limit_mm=1.0)
        self.assertEqual((values["viewCreptMedian"]["value"], values["viewCreptMax"]["value"]), ("0.32", "0.99"))
        self.assertEqual(macros.one_decimal(0.99), "1.0", "one decimal would print the largest reading as the limit")
        self.assertEqual(macros.two_decimals(1.0), "1.00")
        self.assertEqual(macros.two_decimals(0.004), "0.00")
        for name in ("viewCreptMedian", "viewCreptMax"):
            self.assertTrue(DESCRIPTIONS[name].endswith(", mm, two decimals"))
            self.assertIn("execution.view_crept_mm", values[name]["source"])

    def test_without_a_reading_the_two_macros_stay_placeholders(self):
        self.assertEqual(self.values(), {})
        self.assertNotIn("viewCreptMax", macros.macro_values({"conduct": None}))


class SecondScoringMacroTests(unittest.TestCase):
    def test_the_three_counts_are_those_of_all_the_items(self):
        whole = {"n_items": 24, "n_agree": 24, "n_disagree": 0, "n_cannot_tell": 0, "n_compared": 24}
        values = macros.macro_values({"second_scoring": {"why_not": None, "all": whole}})
        self.assertEqual(
            [values[name]["value"] for name in ("nSecondScored", "nSecondScoredAgree", "nSecondScoredCannotTell")],
            ["24", "24", "0"],
        )
        for name in ("nSecondScored", "nSecondScoredAgree", "nSecondScoredCannotTell"):
            self.assertIn("data/second_scoring/main/agreement.json", values[name]["source"])

    def test_an_agreement_that_is_not_of_these_trials_fills_none_of_them(self):
        not_reported = {"second_scoring": {"why_not": "a rehearsal", "all": None}}
        for results in ({}, {"second_scoring": None}, not_reported):
            self.assertEqual(macros.macro_values(results), {})


class MissingValueTests(unittest.TestCase):
    def test_a_result_that_is_null_gives_no_value(self):
        self.assertEqual(macros.macro_values({}), {})
        results = {"blocks": {}, "provenance": {"n_scored": 0, "n_void": 0}, "conduct": {"void_reasons": []}}
        values = macros.macro_values(results)
        self.assertEqual(values["nTrialsTotal"]["value"], "0")
        self.assertEqual(values["voidReasons"]["value"], "none")
        self.assertEqual(values["nPairs"]["value"], "0")
        self.assertNotIn("radiusClustered", values)
        self.assertNotIn("mcnemarP", values)

    def test_successes_among_no_trials_are_not_a_number(self):
        anchors = {"zero": {"n": 0, "k": 0}, "far": {"n": 4, "k": 0}}
        results = {"blocks": {}, "conditions": {"act_clustered": {"anchors": anchors, "n_scored": 4, "n_void": 0}}}
        values = macros.macro_values(results)
        self.assertNotIn("successZeroClustered", values)
        self.assertEqual(values["nZeroClustered"]["value"], "0")
        self.assertEqual((values["successFarClustered"]["value"], values["nFarClustered"]["value"]), ("0", "4"))

    def test_a_setting_that_two_conditions_ran_differently_is_not_one_number(self):
        ran = {"act_clustered": {"clamp_deg": 15.0, "temporal_ensemble_coeff": 0.01}, "replay": {"clamp_deg": 10.0}}
        values = macros.macro_values({"conduct": {"execution": {"by_condition": ran}}})
        self.assertNotIn("execClamp", values)
        self.assertEqual(values["execEnsembleCoeff"]["value"], "0.01", "the replay has no ensembling to disagree about")


if __name__ == "__main__":
    unittest.main()
