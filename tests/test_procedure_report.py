"""The report of the procedure simulation: what it says follows from the numbers it is given."""

import unittest
from dataclasses import replace

import numpy as np

from tolerance_radius import procedure_report, simulate
from tolerance_radius.simulate import blocks_from_protocol, design_variants, paired_power, steps_from_protocol

# A small protocol written out here, for the tests of what the report says: one block that stops short and has no far
# anchors, one that goes farther and has some, and a block of pairs.
MINI = {
    "protocol": {"version": "test", "status": "draft"},
    "staircase": {"rule": "1up1down", "step_initial_mm": 10.0, "step_final_mm": 5.0, "min_level_mm": 0.0},
    "blocks": {
        "near": {
            "anchors_zero": 4,
            "max_level_mm": 30.0,
            "staircases": [
                {"id": "up", "start_level_mm": 5.0, "trials": 9},
                {"id": "down", "start_level_mm": 25.0, "trials": 9},
            ],
        },
        "wide": {
            "anchors_zero": 6,
            "anchors_far": 4,
            "far_level_mm": 50.0,
            "max_level_mm": 50.0,
            "staircases": [
                {"id": "up", "start_level_mm": 10.0, "trials": 12},
                {"id": "down", "start_level_mm": 40.0, "trials": 12},
            ],
        },
        "both": {"pairs": 20},
    },
    "analysis": {
        "r50_grid_mm": {"min": 0.0, "max": 80.0, "step": 0.5},
        "width_grid_mm": {"min": 0.5, "max": 20.0, "count": 30},
        "lapse_values": [0.0, 0.02, 0.05, 0.10],
        "interval_level": 0.95,
    },
}


def made_up_rows(factors: dict | None = None, inside: float = 0.95, open_above: tuple = (0.0, 0.0)) -> list:
    """Sweep rows with numbers chosen by hand. As written, a row's RMSE is the true width and its interval four
    times that; every other design has both multiplied by its factor. `open_above` is the share of intervals open
    above, as written and with the larger range, for the truths whose fall runs past the block's largest distance."""
    usual = {"constant": 1.4, "fine_step": 1.0, "longer": 0.8, "shorter": 1.3, simulate.WIDER_RANGE: 1.0}
    factors = usual | (factors or {})
    blocks = blocks_from_protocol(MINI)
    widest = max(block.max_level_mm for block in blocks)
    rows = []
    for block in blocks:
        for design, (variant, _) in design_variants(block, steps_from_protocol(MINI), widest).items():
            factor = 1.0 if design == "protocol" else factors[design]
            for truth in simulate.sweep_truths():
                at_top = truth.r50_mm > block.max_level_mm - 2.0 * truth.width_mm
                above = (open_above[1] if design == simulate.WIDER_RANGE else open_above[0]) if at_top else 0.0
                performance = simulate.DesignPerformance(
                    truth=truth,
                    block=block.name,
                    n_trials=variant.n_trials,
                    replications=300,
                    bias_mm=0.0,
                    rmse_mm=truth.width_mm * factor,
                    mean_interval_width_mm=4.0 * truth.width_mm * factor,
                    interval_coverage=inside,
                    lower_edge_rate=0.0,
                    upper_edge_rate=above,
                    any_edge_rate=above,
                )
                rows.append(simulate.SweepRow(design, performance))
    return rows


def made_up_floor_rows(
    harm: float = 1.4, cost: float = 1.02, far: float = 1.1, unprotected: float = 1.3, estimates: tuple = (0.054, 0.041)
) -> list:
    """Floor-check rows with numbers chosen by hand. `harm`: how many times larger the RMSE and the interval are
    without a floor in the model when the truth has one. `cost`: the same for a floor in the model when the truth has
    none. `far`: far anchors against staircase trials, with floor values. `unprotected`: far anchors against none,
    without floor values and with a floor in the truth. `estimates`: the posterior mean floor with and without a
    floor in the truth, for the block that has far anchors of its own; the other blocks have theirs below."""
    others = {"as_written": (0.051, 0.049), "far_anchors": (0.053, 0.043), "longer_staircases": (0.052, 0.048)}
    rows = []
    for variant, block in simulate.floor_check_designs(MINI):
        for truth in simulate.floor_check_truths():
            for fit in ("no_floor", "floor"):
                factor = 1.0
                if truth.floor > 0.0 and fit == "no_floor":
                    factor *= harm * (unprotected if variant == "far_anchors" else 1.0)
                if truth.floor == 0.0 and fit == "floor":
                    factor *= cost
                if variant == "far_anchors" and fit == "floor":
                    factor *= far
                performance = simulate.DesignPerformance(
                    truth=truth,
                    block=block.name,
                    n_trials=block.n_trials,
                    replications=300,
                    bias_mm=0.5 if truth.floor > 0.0 and fit == "no_floor" else 0.0,
                    rmse_mm=truth.width_mm * factor,
                    mean_interval_width_mm=4.0 * truth.width_mm * factor,
                    interval_coverage=0.95,
                    lower_edge_rate=0.0,
                    upper_edge_rate=0.0,
                    any_edge_rate=0.0,
                )
                with_floor, without = estimates if block.anchors_far and variant == "as_written" else others[variant]
                estimate = 0.0 if fit == "no_floor" else with_floor if truth.floor > 0.0 else without
                rows.append(simulate.FloorCheckRow(variant, fit, performance, estimate))
    return rows


class ReportRuleTests(unittest.TestCase):
    """The reading and the recommendation are written from the numbers: numbers chosen by hand, and what is said."""

    INFO = procedure_report.RunInfo(
        seed=1, replications=300, placement_sd_mm=1.0, protocol_file="protocol.toml", protocol_digest="0123456789ab"
    )
    POWER = simulate.paired_power_table((16, 20, 24))

    def report(self, rows: list | None = None, floor_rows: list | None = None, protocol: dict = MINI) -> str:
        text = procedure_report.render_report(
            protocol,
            made_up_rows() if rows is None else rows,
            made_up_floor_rows() if floor_rows is None else floor_rows,
            self.POWER,
            self.INFO,
        )
        return " ".join(text.split())

    def test_staircases_are_kept_when_fixed_distances_are_worse_or_no_better_and_dropped_when_they_are_better(self):
        self.assertIn('**Keep the staircases** (`staircase.rule = "1up1down"`). With the same trials', self.report())
        self.assertIn("the RMSE is 40 % larger and the interval 40 % wider", self.report())
        same = self.report(made_up_rows({"constant": 1.02}))
        self.assertIn("**Keep the staircases** (`staircase.rule = \"1up1down\"`): fixed distances do no better.", same)
        better = self.report(made_up_rows({"constant": 0.8}))
        self.assertIn("**Replace the staircases by fixed, equally spaced distances**", better)
        self.assertNotIn("**Keep the staircases**", better)

    def test_the_final_step_changes_only_when_the_finer_one_is_better(self):
        self.assertIn("**Keep `staircase.step_final_mm = 5`.**", self.report())
        self.assertIn("**Keep `staircase.step_final_mm = 5`.**", self.report(made_up_rows({"fine_step": 1.2})))
        better = self.report(made_up_rows({"fine_step": 0.8}))
        self.assertIn("**Change `staircase.step_final_mm` from 5 to 2.5.**", better)
        self.assertNotIn("**Keep `staircase.step_final_mm", better)

    def test_a_design_is_better_only_if_it_is_not_bought_with_open_intervals(self):
        # A smaller RMSE, but intervals open on a side in 30 % of blocks where the design as written has none.
        rows = [
            simulate.SweepRow(row.design, replace(row.performance, any_edge_rate=0.3, lower_edge_rate=0.3))
            if row.design == "fine_step"
            else row
            for row in made_up_rows({"fine_step": 0.8})
        ]
        self.assertIn("**Keep `staircase.step_final_mm = 5`.**", self.report(rows))

    def test_the_range_of_a_block_that_stops_short_is_raised_only_when_that_helps(self):
        helps = self.report(made_up_rows(open_above=(0.4, 0.05)))
        self.assertIn("**Raise `blocks.near.max_level_mm` from 30 to 50**", helps)
        self.assertIn("the share of intervals open above goes from 40 % to 5 %", helps)
        self.assertIn("**Keep `blocks.near.max_level_mm = 30`.**", self.report())
        self.assertNotIn("blocks.wide.max_level_mm", self.report())
        # A gain at the top that is paid for everywhere else is not a gain.
        costly = self.report(made_up_rows({simulate.WIDER_RANGE: 1.3}, open_above=(0.4, 0.05)))
        self.assertIn("**Keep `blocks.near.max_level_mm = 30`.**", costly)

    def test_trials_go_first_to_the_block_they_narrow_most(self):
        report = self.report()
        self.assertIn("if trials can be added, add them to near or wide first**", report)
        self.assertIn("**Do not shorten a staircase, and if trials can be added", report)
        self.assertIn("`blocks.near.staircases` from 9 and 9 trials to 13 and 13", report)
        self.assertIn("near: 20 % narrower; wide: 20 % narrower", report)
        self.assertIn("near: 30 % wider; wide: 30 % wider", report)
        def narrower(row):
            # Longer staircases halve the interval of the block `wide`, where they take a fifth off that of `near`.
            if (row.design, row.performance.block) != ("longer", "wide"):
                return row
            interval = 0.5 * 4.0 * row.performance.truth.width_mm
            return simulate.SweepRow(row.design, replace(row.performance, mean_interval_width_mm=interval))

        rows = [narrower(row) for row in made_up_rows()]
        expected = "add them to wide first** (`blocks.wide.staircases` from 12 and 12 trials to 16 and 16)"
        self.assertIn(expected, self.report(rows))

    def test_the_floor_is_recommended_when_ignoring_it_hurts_and_having_it_is_cheap(self):
        adopted = self.report()
        self.assertIn("**Set `analysis.floor_values = [0.00, 0.05, 0.10]`.**", adopted)
        self.assertIn("leaving the floor out of the model makes the RMSE 40 % larger", adopted)
        self.assertIn("having it in the model makes the interval 2 % wider", adopted)
        useless = self.report(floor_rows=made_up_floor_rows(harm=1.0))
        self.assertIn("**Leave `floor_values` out of the protocol.**", useless)
        self.assertNotIn("**Set `analysis.floor_values", useless)
        dear = self.report(floor_rows=made_up_floor_rows(harm=1.4, cost=1.3))
        self.assertIn("**`analysis.floor_values = [0.00, 0.05, 0.10]` is a trade to decide, not a clear gain.**", dear)

    def test_the_floor_advice_knows_what_the_protocol_already_has(self):
        has_them = dict(MINI, analysis=dict(MINI["analysis"], floor_values=[0.0, 0.05, 0.10]))
        kept = self.report(protocol=has_them)
        self.assertIn("**Keep `analysis.floor_values = [0.00, 0.05, 0.10]`.**", kept)
        self.assertNotIn("**Set `analysis.floor_values", kept)
        self.assertIn("no parameter of the protocol file needs to change", kept)
        # The protocol has floor values and the check finds no use for them: taking them out is the change.
        dropped = self.report(floor_rows=made_up_floor_rows(harm=1.0), protocol=has_them)
        self.assertIn("**Take `floor_values` out of the protocol.**", dropped)
        self.assertIn("with 1 change to the protocol file:** `analysis.floor_values` taken out.", dropped)
        # Other floor values than the ones that were checked: setting the checked ones is a change.
        others = dict(MINI, analysis=dict(MINI["analysis"], floor_values=[0.0, 0.2]))
        replaced = "**Set `analysis.floor_values = [0.00, 0.05, 0.10]` in place of [0.00, 0.20].**"
        self.assertIn(replaced, self.report(protocol=others))

    def test_far_anchors_are_judged_against_the_same_trials_spent_on_the_staircases(self):
        self.assertIn("**Do not add far anchors to near for the estimate's sake.**", self.report())
        numbers = "4 far anchors give an RMSE 10 % larger and an interval 10 % wider than 4 more staircase trials"
        self.assertIn(numbers, self.report())
        self.assertIn("**Give near far anchors.**", self.report(floor_rows=made_up_floor_rows(far=0.8)))
        self.assertIn("**Far anchors for near are optional.**", self.report(floor_rows=made_up_floor_rows(far=1.0)))

    def test_far_anchors_without_a_floor_in_the_model_are_warned_against_when_they_hurt(self):
        self.assertIn("**Do not run far anchors with a model that has no floor.**", self.report())
        self.assertIn("far anchors make the RMSE 30 % larger in near", self.report())
        harmless = self.report(floor_rows=made_up_floor_rows(unprotected=1.0))
        self.assertNotIn("**Do not run far anchors with a model that has no floor.**", harmless)

    def test_the_report_says_whether_the_floor_can_be_read_from_the_trials(self):
        self.assertIn("**The size of the floor cannot be read from 4 far anchors.**", self.report())
        numbers = "is 0.054 when the true floor is 0.08 and 0.041 when it is 0; its prior mean is 0.050"
        self.assertIn(numbers, self.report())
        self.assertIn("by 0.81 for a floor of 0.05 and 0.66 for a floor of 0.1", self.report())
        readable = self.report(floor_rows=made_up_floor_rows(estimates=(0.09, 0.01)))
        self.assertIn("**4 far anchors show the size of the floor.**", readable)

    def test_the_floor_estimates_quoted_are_those_of_the_right_blocks(self):
        report = self.report()
        self.assertIn("In wide the posterior mean of the floor is 0.054 when the true floor is 0.08 and 0.041", report)
        added = "With the far anchors the posterior mean of the floor is 0.053 when the true floor is 0.08 and 0.043"
        self.assertIn(added, report)
        self.assertIn("(without them: 0.051 and 0.049): they do not make the floor readable either.", report)

    def test_ratios_are_averaged_so_that_twice_and_half_cancel(self):
        # Fixed distances with twice the RMSE for cliffs and half of it at width 3 mm: over all truths, no change.
        # (An ordinary mean of 2, 1/2, 1 and 1 would say 12 % worse.)
        def uneven(row):
            if row.design != "constant":
                return row
            width = row.performance.truth.width_mm
            rmse = width * {1.0: 2.0, 3.0: 0.5}.get(width, 1.0)
            return simulate.SweepRow(row.design, replace(row.performance, rmse_mm=rmse))

        rows = [uneven(row) for row in made_up_rows()]
        text = procedure_report.render_report(MINI, rows, made_up_floor_rows(), self.POWER, self.INFO)
        line = next(line for line in text.splitlines() if line.startswith("| wide | fixed distances |"))
        self.assertEqual([cell.strip() for cell in line.split("|")][4:9], ["1.00", "2.00", "0.50", "1.00", "1.00"])
        words = " ".join(text.split())
        self.assertIn("With fixed distances the RMSE is the same (from 50 % smaller at width 3 mm", words)

    def test_the_interval_is_said_to_hold_only_when_it_does(self):
        self.assertIn("**The interval holds what it promises.**", self.report())
        low = self.report(made_up_rows(inside=0.80))
        self.assertIn("**The interval falls short of 95 % for some truths.**", low)
        self.assertNotIn("holds what it promises", low)
        # One low row among good ones is named.
        rows = made_up_rows()
        rows[0] = simulate.SweepRow(rows[0].design, replace(rows[0].performance, interval_coverage=0.70))
        self.assertIn("1 rows are below 95 % by more than the simulation's own noise", self.report(rows))
        self.assertIn("near, `r50` 5 mm, width 1 mm, lapse 0: 70 %", self.report(rows))

    def test_the_paired_test_is_described_from_the_exact_table(self):
        report = self.report()
        needed = simulate.rate_needed(20, 0.4)
        self.assertIn(f"It tells 0.40 from {needed:.2f}", report)
        self.assertIn("**Keep the paired test at 20 pairs and write down now what it can show.**", report)
        self.assertIn("The false-alarm rate is at most", report)

    def test_more_pairs_are_suggested_only_when_they_buy_more_than_one_step(self):
        # From 20 to 24 pairs the rate that can be told from 0.40 or 0.50 falls by one step of 0.05 at most: keep 20.
        gains = [simulate.rate_needed(20, low) - simulate.rate_needed(24, low) for low in (0.4, 0.5)]
        self.assertLessEqual(max(gains), 0.05 + 1e-9)
        self.assertIn("at most one step of 0.05 better", self.report())
        # With 8 pairs in the protocol, 12 pairs are a different test: 8 can tell nothing from 0.40 or 0.50.
        few = dict(MINI, blocks=dict(MINI["blocks"], both={"pairs": 8}))
        text = procedure_report.render_report(
            few, made_up_rows(), made_up_floor_rows(), simulate.paired_power_table((4, 8, 12)), self.INFO
        )
        words = " ".join(text.split())
        self.assertIsNone(simulate.rate_needed(8, 0.4))
        self.assertIsNotNone(simulate.rate_needed(12, 0.4))
        self.assertIn("**With 8 pairs the paired test can show very little; 12 pairs would show more.**", words)
        self.assertNotIn("**Keep the paired test at 8 pairs", words)

    def test_more_pairs_matter_when_they_bring_the_rate_down_by_more_than_one_step(self):
        # From a rate of 0.30: 12 pairs tell it from 0.95, 24 pairs from 0.75. From 20 to 24 pairs: 0.80 to 0.75.
        self.assertEqual((simulate.rate_needed(12, 0.3), simulate.rate_needed(24, 0.3)), (0.95, 0.75))
        self.assertTrue(procedure_report._more_pairs_matter(12, 24, [0.3]))
        self.assertEqual(simulate.rate_needed(20, 0.3), 0.80)
        self.assertFalse(procedure_report._more_pairs_matter(20, 24, [0.3]))
        self.assertFalse(procedure_report._more_pairs_matter(24, 24, [0.3]))

    def test_the_mean_error_of_the_estimate_is_read_out_by_width(self):
        rows = [
            simulate.SweepRow(row.design, replace(row.performance, bias_mm=0.2 * row.performance.truth.width_mm))
            for row in made_up_rows()
        ]
        report = self.report(rows)
        self.assertIn("**Which way the estimate leans.**", report)
        self.assertIn("+0.2 mm at width 1 mm and +2.0 mm at width 10 mm", report)
        self.assertIn("The largest in any row is +2.0 mm", report)

    def test_the_floor_check_says_what_it_left_out(self):
        report = self.report()
        self.assertIn("Only a model without a floor and one with the floor values 0, 0.05 and 0.1 were", report)
        self.assertIn("For a condition in which a lucky success far out cannot happen", report)
        # A protocol with two blocks without far anchors: the smaller one is named as not run.
        second = dict(MINI["blocks"]["near"], staircases=[{"id": "up", "start_level_mm": 5.0, "trials": 6}])
        protocol = dict(MINI, blocks=dict(MINI["blocks"], small=second))
        self.assertEqual({block.name for _, block in simulate.floor_check_designs(protocol)}, {"near", "wide"})
        rows = made_up_rows() + [
            simulate.SweepRow(row.design, replace(row.performance, block="small"))
            for row in made_up_rows()
            if row.performance.block == "near"
        ]
        text = procedure_report.render_report(protocol, rows, made_up_floor_rows(), self.POWER, self.INFO)
        self.assertIn("small has no far anchors either and was not run in this check", " ".join(text.split()))
        self.assertNotIn("was not run in this check", report)

    def test_the_report_does_not_say_what_other_documents_say(self):
        # It is written from the protocol file and the rows alone.
        report = self.report()
        self.assertIn("**Trials from other blocks.** Each block is fitted alone here.", report)
        self.assertNotIn("prediction", report)

    def test_the_recommendation_opens_with_the_changes_it_asks_for(self):
        # The answer first: which parameters of the protocol file change, before the reasons.
        one = self.report()
        self.assertIn(
            "**In short: the simulation supports the design as written, with 1 change to the protocol file:** "
            "`analysis.floor_values = [0.00, 0.05, 0.10]`.",
            one,
        )
        two = self.report(made_up_rows(open_above=(0.4, 0.05)))
        self.assertIn(
            "with 2 changes to the protocol file:** `blocks.near.max_level_mm` from 30 to 50; "
            "`analysis.floor_values = [0.00, 0.05, 0.10]`.",
            two,
        )
        none = self.report(floor_rows=made_up_floor_rows(harm=1.0))
        self.assertIn(
            "**In short: the simulation supports the design as written; no parameter of the protocol file needs to "
            "change.**",
            none,
        )
        everything = self.report(
            made_up_rows({"constant": 0.8, "fine_step": 0.8}, open_above=(0.4, 0.05)),
            made_up_floor_rows(far=0.8),
        )
        self.assertIn("**In short: the simulation asks for 5 changes to the protocol file:**", everything)
        for change in (
            "fixed, equally spaced distances in place of the staircases",
            "`staircase.step_final_mm` from 5 to 2.5",
            "`blocks.near.max_level_mm` from 30 to 50",
            "`analysis.floor_values = [0.00, 0.05, 0.10]`",
            "far anchors for near",
        ):
            self.assertIn(change, everything.split("## Recommendation")[1].split("1. **")[0])

    def test_the_rule_for_an_open_interval_is_always_asked_for(self):
        asked = "**Write the rule for an open interval into `protocol/protocol.md` before the freeze.**"
        self.assertIn(asked, self.report())
        self.assertIn("`psychometric.EDGE_RATIO = 0.5`", self.report())

    def test_the_fit_s_floor_values_are_described_as_the_protocol_has_them(self):
        self.assertIn("no floor (the protocol file has no `floor_values`", self.report())
        with_floors = dict(MINI, analysis=dict(MINI["analysis"], floor_values=[0.0, 0.05, 0.10]))
        self.assertIn("floor values 0, 0.05 and 0.1, equally weighted", self.report(protocol=with_floors))

    def test_a_protocol_without_a_block_that_stops_short_or_without_pairs_still_gets_a_report(self):
        blocks = {name: block for name, block in MINI["blocks"].items() if name == "wide"}
        protocol = dict(MINI, blocks=blocks)
        rows = [row for row in made_up_rows() if row.performance.block == "wide"]
        floor_rows = [row for row in made_up_floor_rows() if row.performance.block == "wide"]
        text = procedure_report.render_report(protocol, rows, floor_rows, [], self.INFO)
        self.assertIn("Every block of the protocol goes out to the same largest distance", " ".join(text.split()))
        self.assertIn("The protocol has no block of paired trials.", text)
        self.assertIn("## Recommendation", text)


class ReportTableTests(unittest.TestCase):
    """The tables of the report carry the numbers of the rows they are made from."""

    def setUp(self):
        def marked(row):
            # One row made recognisable: block `near`, r50 = 5 mm, width = 3 mm, both lapse rates.
            performance = row.performance
            which = (row.design, performance.block, performance.truth.r50_mm, performance.truth.width_mm)
            if which != ("protocol", "near", 5.0, 3.0):
                return row
            changed = replace(
                performance,
                bias_mm=2.5,
                rmse_mm=7.0,
                mean_interval_width_mm=33.0,
                interval_coverage=0.60,
                lower_edge_rate=0.20,
                upper_edge_rate=0.10,
                any_edge_rate=0.30,
            )
            return simulate.SweepRow(row.design, changed)

        rows = [marked(row) for row in made_up_rows(open_above=(0.4, 0.05))]
        floor_rows, power, info = made_up_floor_rows(), ReportRuleTests.POWER, ReportRuleTests.INFO
        self.lines = procedure_report.render_report(MINI, rows, floor_rows, power, info).splitlines()

    def assert_row(self, row: str) -> None:
        """The report has this table row. If not, show the rows that start alike, not the whole report."""
        start = "|".join(row.split("|")[:3])
        alike = "\n".join(line for line in self.lines if line.startswith(start))
        self.assertTrue(row in self.lines, f"not found: {row}\nrows that start alike:\n{alike}")

    def test_the_headline_table_gives_the_middle_and_the_worst_of_each_width(self):
        # Width 1: every row has RMSE 1 and interval 4; the intervals are open above (40 %) for the 4 truths of 12
        # at 30 and 40 mm, so the middle row is at 0 and the worst at 40. Width 3: the marked rows are the worst in
        # RMSE, interval and share inside; with them 6 of 12 rows have open intervals (two at 30 %, four at 40 %), and
        # the middle of the twelve lies between 0 and 30.
        self.assert_row("| near | 22 | 1 | 1.0 (1.0) | 4.0 (4.0) | 95 (95) | 0 (40) |")
        self.assert_row("| near | 22 | 3 | 3.0 (7.0) | 12.0 (33.0) | 95 (60) | 15 (40) |")
        self.assert_row("| wide | 34 | 10 | 10.0 (10.0) | 40.0 (40.0) | 95 (95) | 0 (40) |")
        self.assert_row("| wide | 34 | 1 | 1.0 (1.0) | 4.0 (4.0) | 95 (95) | 0 (0) |")

    def test_the_detail_tables_give_each_radius_and_width(self):
        self.assert_row("| near | 5 | 0.0 / 1.0 | +2.5 / 7.0 | 0.0 / 6.0 | 0.0 / 10.0 |")  # bias / RMSE
        self.assert_row("| near | 5 | 4.0 | 33.0 | 24.0 | 40.0 |")  # interval width
        self.assert_row("| near | 5 | 0 / 0 | 20 / 10 | 0 / 0 | 0 / 0 |")  # open below / above
        self.assert_row("| near | 5 | 95 | 60 | 95 | 95 |")  # contains the truth
        # At 30 mm the block `near` (largest distance 30 mm) cannot bound any curve from above.
        self.assert_row("| near | 30 | 0 / 40 | 0 / 40 | 0 / 40 | 0 / 40 |")

    def test_the_truths_are_sorted_by_where_their_fall_lies(self):
        # Largest distance 30 mm. Under way at 0 mm: r50 < 2 widths. Past the largest distance: r50 > 30 - 2 widths.
        # Width 10 and r50 = 15 are both at once.
        counts = {}
        for line in self.lines:
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) > 4 and cells[1] == "near" and cells[2] in procedure_report._FALL_NAMES.values():
                counts[cells[2]] = int(cells[3])
        truths = simulate.sweep_truths()
        early = {t for t in truths if t.r50_mm < 2 * t.width_mm}
        late = {t for t in truths if t.r50_mm > 30.0 - 2 * t.width_mm}
        expected = {
            procedure_report._FALL_NAMES["inside"]: len(set(truths) - early - late),
            procedure_report._FALL_NAMES["early"]: len(early - late),
            procedure_report._FALL_NAMES["late"]: len(late - early),
            procedure_report._FALL_NAMES["both"]: len(early & late),
        }
        self.assertEqual(counts, {name: count for name, count in expected.items() if count})
        self.assertEqual(sum(counts.values()), len(truths))

    def test_the_comparison_tables_give_each_design_relative_to_the_design_as_written(self):
        # Open on a side: 4 of the 48 truths of `wide` (40 mm, widths 6 and 10) at 40 %, in every design: 3 %.
        for design, trials, factor in (
            ("fixed distances", 34, "1.40"),
            ("final step 2.5 mm", 34, "1.00"),
            ("staircases 4 longer", 42, "0.80"),
            ("staircases 4 shorter", 26, "1.30"),
        ):
            ratios = " | ".join([factor] * 6)  # the RMSE over all truths and for each of four widths, and the interval
            self.assert_row(f"| wide | {design} | {trials} | {ratios} | 3 : 3 | 95 : 95 |")

    def test_the_larger_range_is_shown_radius_by_radius(self):
        # Means over the four widths (1, 3, 6, 10 mm: RMSE 5, interval 20). At 40 mm every width is beyond the
        # block's reach; at 10 mm none is.
        self.assert_row("| near | 40 | 5.0 → 5.0 | 20.0 → 20.0 | 40 → 5 |")
        self.assert_row("| near | 10 | 5.0 → 5.0 | 20.0 → 20.0 | 0 → 0 |")
        self.assertFalse(any(line.startswith("| wide | 40 | 5.0 →") for line in self.lines))

    def test_the_floor_table_gives_both_fits_of_each_block_and_floor(self):
        mean_width = np.mean(simulate.FLOOR_CHECK_WIDTH_MM)
        harmed = 1.4 * mean_width  # a floor in the truth, none in the model
        block = "| wide, as written | 34 | 0.08 |"
        self.assert_row(f"{block} without a floor | +0.5 | {harmed:.1f} | {4 * harmed:.1f} | 95 | - |")
        self.assert_row(f"{block} with floor values | 0.0 | {mean_width:.1f} | {4 * mean_width:.1f} | 95 | 0.054 |")
        far = 1.02 * 1.1 * mean_width  # the cost of the floor and of the far anchors, on the mean width
        block = "| near, with 4 far anchors added | 26 | 0 |"
        self.assert_row(f"{block} with floor values | 0.0 | {far:.1f} | {4 * far:.1f} | 95 | 0.043 |")
        by_width = "4.2 → 3.0 | +0.5 → 0.0 | 16.8 → 12.0 | 3.0 → 3.1 | 0.0 → 0.0 | 12.0 → 12.2"
        self.assert_row(f"| wide | 3 | {by_width} |")

    def test_the_power_tables_are_the_exact_numbers(self):
        powers = " | ".join(f"{100 * paired_power(20, 0.4, rate):.0f}" for rate in simulate.POWER_RATES)
        self.assert_row(f"| 0.4 | {powers} |")
        needed = [simulate.rate_needed(pairs, 0.4) for pairs in (16, 20, 24)]
        self.assert_row("| 0.40 | " + " | ".join(f"{rate:.2f}" for rate in needed) + " |")

    def test_the_blocks_are_listed_as_the_protocol_states_them(self):
        self.assert_row("| near | 22 | 4 | up from 5 mm (9 trials); down from 25 mm (9 trials) | none | 30 |")
        self.assert_row("| wide | 34 | 6 | up from 10 mm (12 trials); down from 40 mm (12 trials) | 4 at 50 mm | 50 |")


if __name__ == "__main__":
    unittest.main()
