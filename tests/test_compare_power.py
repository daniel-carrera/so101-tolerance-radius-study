"""The simulation of the comparison rule: its cases, one pair of blocks, the counts, the table, the report, the command.

Everything here runs on a coarse analysis grid and a few pairs of blocks: the tests are about what is counted and
written, not about the figures of the report in `analysis/simulation/`.
"""

import contextlib
import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import fixtures_experiment as fixtures
from tolerance_radius import compare_power, config, simulate
from tolerance_radius.compare_power import RuleCase, RuleRow
from tolerance_radius.psychometric import grids_from_protocol
from tolerance_radius.report import estimates
from tolerance_radius.seeds import derived_rng

SIMULATION = fixtures.REPO / "analysis" / "simulation"


def coarse_protocol() -> dict:
    """The repository's protocol with a coarse analysis grid, which makes a fit about fifteen times quicker."""
    protocol = config.load_protocol()
    protocol["analysis"] = protocol["analysis"] | {
        "r50_grid_mm": {"min": 0.0, "max": 80.0, "step": 2.0},
        "width_grid_mm": {"min": 0.5, "max": 20.0, "count": 8},
    }
    return protocol


def row(width: float, difference: float, larger: int, smaller: int = 0, pairs: int = 20) -> RuleRow:
    """A row written by hand: so many pairs called larger, so many smaller, the rest not told apart."""
    return RuleRow(
        case=RuleCase(15.0, 15.0 + difference, width),
        policy_block="T1a",
        replay_block="T1b",
        pairs=pairs,
        n_larger=larger,
        n_not_told_apart=pairs - larger - smaller,
        n_smaller=smaller,
        mean_difference_mm=difference - 0.04,
        mean_interval_width_mm=4.0 * width + 3.0,
        interval_coverage=0.95,
    )


class CaseTests(unittest.TestCase):
    def test_the_cases_are_every_width_with_every_difference_from_one_replay_radius(self):
        cases = compare_power.cases()
        self.assertEqual(len(cases), 18)
        self.assertEqual({case.replay_r50_mm for case in cases}, {15.0})
        self.assertEqual(sorted({case.difference_mm for case in cases}), [0.0, 3.0, 5.0, 8.0, 10.0, 15.0])
        self.assertEqual(sorted({case.width_mm for case in cases}), [1.0, 3.0, 6.0])
        self.assertEqual({case.lapse for case in cases}, {0.0})
        self.assertEqual(len(set(cases)), 18)
        for case in cases:
            self.assertEqual(case.policy_r50_mm, case.replay_r50_mm + case.difference_mm)
            self.assertEqual(case.policy_truth, simulate.Truth(case.policy_r50_mm, case.width_mm, 0.0))
            self.assertEqual(case.replay_truth, simulate.Truth(15.0, case.width_mm, 0.0))

    def test_the_blocks_are_the_two_threshold_blocks_as_the_protocol_file_states_them(self):
        protocol = config.load_protocol()
        policy, replay = compare_power.compared_blocks(protocol)
        stated = protocol["blocks"]
        self.assertEqual(stated[policy.name]["condition"], "act_clustered")
        self.assertEqual(stated[replay.name]["condition"], "replay")
        for block in (policy, replay):
            self.assertEqual(block.n_trials, fixtures.planned_total(stated[block.name]))
            self.assertEqual(block.max_level_mm, stated[block.name]["max_level_mm"])
            self.assertEqual(
                [(plan.start_level_mm, plan.n_trials) for plan in block.staircases],
                [(entry["start_level_mm"], entry["trials"]) for entry in stated[block.name]["staircases"]],
            )
        self.assertEqual(policy.anchors_far, stated[policy.name].get("anchors_far", 0))

    def test_a_protocol_without_one_of_the_two_blocks_is_refused(self):
        protocol = config.load_protocol()
        protocol["blocks"] = {name: entry for name, entry in protocol["blocks"].items() if name != "T1b"}
        with self.assertRaises(ValueError) as caught:
            compare_power.compared_blocks(protocol)
        self.assertIn("one threshold block of replay", str(caught.exception))


class PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = coarse_protocol()
        cls.grids = grids_from_protocol(cls.protocol["analysis"])
        cls.steps = simulate.steps_from_protocol(cls.protocol)
        cls.policy, cls.replay = compare_power.compared_blocks(cls.protocol)

    def pair(self, case: RuleCase, rng) -> dict:
        return compare_power.simulate_pair(case, self.policy, self.replay, self.steps, rng, grids=self.grids)

    def test_a_pair_is_two_blocks_played_apart_and_compared_by_the_analysis_s_own_rule(self):
        case = RuleCase(15.0, 23.0, 3.0)
        found = self.pair(case, derived_rng(7, "pair"))
        rng = derived_rng(7, "pair")
        policy = simulate.simulate_block(case.policy_truth, self.policy, self.steps, rng, grids=self.grids)
        replay = simulate.simulate_block(case.replay_truth, self.replay, self.steps, rng, grids=self.grids)
        self.assertEqual(policy.distances_mm.size, self.policy.n_trials)
        self.assertEqual(replay.distances_mm.size, self.replay.n_trials)
        expected = estimates.radius_difference(
            estimates.fit(policy.distances_mm, policy.successes, self.grids),
            estimates.fit(replay.distances_mm, replay.successes, self.grids),
            0.95,
        )
        self.assertEqual(found, expected)
        self.assertEqual(found["verdict"], estimates.verdict(found["prob_greater"]))
        self.assertAlmostEqual(
            found["median_mm"], policy.summary.r50_median_mm - replay.summary.r50_median_mm, delta=4.0
        )

    def test_the_counts_of_a_case_are_those_of_its_pairs(self):
        case = RuleCase(15.0, 20.0, 3.0)
        counted = compare_power.evaluate_case(
            case, self.policy, self.replay, self.steps, derived_rng(11, "case"), grids=self.grids, pairs=12
        )
        rng = derived_rng(11, "case")
        pairs = [self.pair(case, rng) for _ in range(12)]
        verdicts = [found["verdict"] for found in pairs]
        self.assertEqual(
            (counted.n_larger, counted.n_not_told_apart, counted.n_smaller),
            (verdicts.count("larger"), verdicts.count("not told apart"), verdicts.count("smaller")),
        )
        self.assertEqual(counted.n_larger + counted.n_not_told_apart + counted.n_smaller, counted.pairs)
        self.assertEqual((counted.pairs, counted.policy_block, counted.replay_block), (12, "T1a", "T1b"))
        self.assertAlmostEqual(counted.mean_difference_mm, sum(f["median_mm"] for f in pairs) / 12, places=12)
        self.assertAlmostEqual(
            counted.mean_interval_width_mm, sum(f["hi_mm"] - f["lo_mm"] for f in pairs) / 12, places=12
        )
        self.assertAlmostEqual(
            counted.interval_coverage, sum(f["lo_mm"] <= 5.0 <= f["hi_mm"] for f in pairs) / 12, places=12
        )
        self.assertAlmostEqual(sum(counted.share(verdict) for verdict in compare_power.VERDICTS), 1.0, places=12)
        with self.assertRaises(ValueError):
            compare_power.evaluate_case(
                case, self.policy, self.replay, self.steps, derived_rng(11, "case"), grids=self.grids, pairs=0
            )


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = coarse_protocol()
        cls.far_apart, cls.equal = RuleCase(15.0, 30.0, 1.0), RuleCase(15.0, 15.0, 3.0)
        cls.rows = compare_power.run(cls.protocol, pairs=40, seed=5, rule_cases=[cls.far_apart, cls.equal])

    def test_cliffs_far_apart_are_called_larger_and_equal_radii_mostly_are_not_told_apart(self):
        far_apart, equal = self.rows
        self.assertEqual((far_apart.case, equal.case), (self.far_apart, self.equal))
        self.assertGreaterEqual(far_apart.share("larger"), 0.9)
        self.assertEqual(far_apart.n_smaller, 0)
        self.assertGreaterEqual(equal.share("not told apart"), 0.7)
        self.assertAlmostEqual(far_apart.mean_difference_mm, 15.0, delta=2.0)
        self.assertAlmostEqual(equal.mean_difference_mm, 0.0, delta=2.5)
        self.assertGreater(equal.mean_interval_width_mm, far_apart.mean_interval_width_mm, "a wider curve, less told")

    def test_a_case_s_numbers_follow_the_seed_and_not_the_other_cases(self):
        alone = compare_power.run(self.protocol, pairs=40, seed=5, rule_cases=[self.equal])
        self.assertEqual(alone, [self.rows[1]])
        other_seed = compare_power.run(self.protocol, pairs=40, seed=6, rule_cases=[self.equal])
        self.assertNotEqual(other_seed, alone)

    def test_left_to_itself_it_runs_the_report_s_cases(self):
        rows = compare_power.run(self.protocol, pairs=1, seed=5)
        self.assertEqual([found.case for found in rows], compare_power.cases())
        self.assertEqual({found.pairs for found in rows}, {1})


class TableAndReportTests(unittest.TestCase):
    ROWS = (
        row(1.0, 0.0, larger=1, smaller=1),
        row(1.0, 3.0, larger=12),
        row(1.0, 5.0, larger=17),
        row(1.0, 8.0, larger=20),
        row(3.0, 0.0, larger=0),
        row(3.0, 3.0, larger=17),
        row(3.0, 5.0, larger=9, smaller=1),
        row(3.0, 8.0, larger=16),
        row(6.0, 0.0, larger=2),
        row(6.0, 3.0, larger=2),
        row(6.0, 5.0, larger=5),
        row(6.0, 8.0, larger=11),
    )
    INFO = compare_power.RunInfo(
        seed=5,
        pairs=20,
        placement_sd_mm=1.0,
        protocol_file="protocol.toml",
        protocol_version="0.2",
        protocol_status="draft",
        protocol_digest="0123456789ab",
    )

    def test_the_table_has_a_line_for_every_case(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison_rule.csv"
            compare_power.write_csv(path, self.ROWS)
            with open(path, newline="", encoding="utf-8") as handle:
                lines = list(csv.DictReader(handle))
            text = path.read_text(encoding="utf-8")
        self.assertEqual(tuple(lines[0]), compare_power.CSV_COLUMNS)
        self.assertEqual(len(lines), len(self.ROWS))
        self.assertEqual(
            text.splitlines()[2], "T1a,T1b,15,18,3,1,0,20,12,8,0,0.6000,0.4000,0.0000,2.9600,7.0000,0.9500"
        )
        for line in lines:
            counts = [int(line[key]) for key in ("n_larger", "n_not_told_apart", "n_smaller")]
            self.assertEqual(sum(counts), int(line["pairs"]))
            shares = [float(line[key]) for key in ("share_larger", "share_not_told_apart", "share_smaller")]
            self.assertAlmostEqual(sum(shares), 1.0, places=3)
            difference = float(line["policy_r50_mm"]) - float(line["replay_r50_mm"])
            self.assertEqual(difference, float(line["difference_true_mm"]))

    def test_the_smallest_difference_that_is_reliably_called_larger(self):
        self.assertEqual(compare_power.RELIABLE_SHARE, 0.8)
        self.assertEqual(compare_power.smallest_reliable_difference(self.ROWS, 1.0), 5.0)
        self.assertEqual(
            compare_power.smallest_reliable_difference(self.ROWS, 3.0),
            8.0,
            "reliable at 3 mm and not at 5 mm: only from 8 mm on does every larger difference give it",
        )
        self.assertIsNone(compare_power.smallest_reliable_difference(self.ROWS, 6.0))
        self.assertIsNone(compare_power.smallest_reliable_difference([row(1.0, 0.0, larger=20)], 1.0))

    def test_the_report_says_it_is_simulated_what_was_assumed_and_what_it_does_not_show(self):
        protocol = config.load_protocol()
        policy, replay = compare_power.compared_blocks(protocol)
        text = compare_power.render_report(protocol, self.ROWS, self.INFO)
        lines = text.splitlines()
        self.assertEqual(lines[0], "# The comparison rule: what two blocks of trials can tell apart")
        self.assertEqual(lines[2], "**Every number in this report is simulated. Nothing here is a measurement.**")
        self.assertIn("Seed 5; 20 simulated pairs of blocks for every row.", text)
        self.assertRegex(text, r"version 0\.2, status\s+draft,\s+SHA-256\s+`0123456789ab`")
        for heading in (
            "## The question",
            "## What was simulated, and what was assumed",
            "## 1. How often each verdict comes out",
            "## 2. The difference that is reported beside the verdict",
            "## Reading",
            "## What this simulation does not show",
        ):
            self.assertIn(heading, lines)
        self.assertIn("| 1 | 3 | 60 | 40 | 0 |", lines)
        self.assertIn("| 3 | 5 | 45 | 50 | 5 |", lines)
        self.assertIn("| 1 | 3 | +3.0 | 7.0 | 95 |", lines)
        self.assertIn("| 1 | 0 | 0.0 | 7.0 | 95 |", lines, "a difference that rounds to nothing carries no sign")
        self.assertEqual(sum(line.startswith("| ") and "Width" not in line for line in lines), 2 * len(self.ROWS))
        flat = " ".join(text.split())
        self.assertIn(f"The policy's: {policy.name}, {policy.n_trials} trials ({policy.anchors_zero} at 0 mm", flat)
        self.assertIn(f"The replay's: {replay.name}, {replay.n_trials} trials ({replay.anchors_zero} at 0 mm", flat)
        self.assertIn('"larger" means a posterior probability of at least 0.95', flat)
        self.assertIn('The rule says "larger" in 60 % of pairs at a true difference of 3 mm, 85 % at 5 mm, ', flat)
        self.assertIn("The smallest simulated difference from which it does so in at least 80 % of pairs is 5 mm", flat)
        self.assertIn("No simulated difference up to 8 mm gives that verdict in 80 % of pairs", flat)
        self.assertIn('the rule said "smaller" in at most 5 % of pairs', flat)
        self.assertIn("At most 10 % of pairs got a verdict for a difference that is not there", flat)
        self.assertIn("a share near 50 % is known to about 11.2 points", flat)
        self.assertTrue(all(len(line) <= 118 or line.startswith("|") for line in lines))
        self.assertTrue(text.endswith("\n") and not text.endswith("\n\n"))
        self.assertTrue(text.isascii())


class CommandTests(unittest.TestCase):
    def run_command(self, *arguments: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = compare_power.main(list(arguments))
        return code, out.getvalue()

    def test_it_writes_the_table_and_the_report_and_the_same_bytes_again(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol = Path(folder) / "protocol.toml"
            protocol.write_text(fixtures.protocol_text(status="draft", version="0.9", coarse=True), encoding="utf-8")
            digest = hashlib.sha256(protocol.read_bytes()).hexdigest()[:12]
            first, second = Path(folder) / "first", Path(folder) / "second"
            code, said = self.run_command("--out", str(first), "--protocol", str(protocol), "--pairs", "2")
            self.assertEqual(code, 0)
            self.assertNotIn(folder, said)
            self.assertIn("18 cases, 2 pairs of blocks each", said)
            self.run_command("--out", str(second), "--protocol", str(protocol), "--pairs", "2")
            written = sorted(path.name for path in first.iterdir())
            self.assertEqual(written, ["comparison_rule.csv", "comparison_rule.md"])
            for name in ("comparison_rule.csv", "comparison_rule.md"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            report = (first / "comparison_rule.md").read_text(encoding="utf-8")
            self.assertNotIn(folder, report)
            self.assertRegex(report, rf"`protocol\.toml`: version 0\.9, status draft, SHA-256\s+`{digest}`")
            self.assertEqual(len((first / "comparison_rule.csv").read_text(encoding="utf-8").splitlines()), 19)
            self.run_command("--out", str(second), "--protocol", str(protocol), "--pairs", "2", "--seed", "9")
            self.assertNotEqual(
                (first / "comparison_rule.csv").read_bytes(), (second / "comparison_rule.csv").read_bytes()
            )

    def test_a_count_of_pairs_below_one_is_refused(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                compare_power.main(["--out", folder, "--pairs", "0"])
            self.assertEqual(list(Path(folder).iterdir()), [])


class RepositoryTests(unittest.TestCase):
    """The report in `analysis/simulation/`: a simulation of the design, committed like the others in that folder."""

    def test_the_committed_table_holds_the_report_s_cases_and_its_report_names_a_protocol_file(self):
        with open(SIMULATION / "comparison_rule.csv", newline="", encoding="utf-8") as handle:
            lines = list(csv.DictReader(handle))
        self.assertEqual(tuple(lines[0]), compare_power.CSV_COLUMNS)
        found = [
            RuleCase(*(float(line[key]) for key in ("replay_r50_mm", "policy_r50_mm", "width_mm", "lapse")))
            for line in lines
        ]
        self.assertEqual(found, compare_power.cases())
        self.assertEqual({int(line["pairs"]) for line in lines}, {compare_power.DEFAULT_PAIRS})
        report = (SIMULATION / "comparison_rule.md").read_text(encoding="utf-8")
        self.assertIn("**Every number in this report is simulated. Nothing here is a measurement.**", report)
        self.assertRegex(report, r"SHA-256\s+`[0-9a-f]{12}`")
        self.assertIn(f"Seed {compare_power.DEFAULT_SEED}; {compare_power.DEFAULT_PAIRS} simulated pairs", report)
        self.assertNotRegex(report, r"(?<![\w.])(?:/Users|/home|/private|/var|/tmp)/")
        for line in lines:  # the report's first table is the table's shares, rounded
            cell = (
                f"| {line['width_mm']} | {line['difference_true_mm']} | {100 * float(line['share_larger']):.0f} | "
                f"{100 * float(line['share_not_told_apart']):.0f} | {100 * float(line['share_smaller']):.0f} |"
            )
            self.assertIn(cell, report)


if __name__ == "__main__":
    unittest.main()
