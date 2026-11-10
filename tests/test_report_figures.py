"""Where the two halves of the pipeline meet: the paper's figures, drawn from what the analysis writes.

The figure code reads `results.json` by the contract of `protocol/data-contracts.md` and was written without sight of the
analysis, and the analysis without sight of the figure code. Here the analysis's output for the whole synthetic
experiment is handed to the figure code as a file. Everything is written to a temporary folder.
"""

import tempfile
import unittest
from pathlib import Path

import fixtures_experiment as fixtures
from tolerance_radius.figures import make
from tolerance_radius.figures import results as reader

CONDITIONS = ("act_clustered", "replay", "act_spread")
POLICIES = ("act_clustered", "act_spread")


class FiguresFromTheAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment, cls.made = fixtures.main_run()
        cls.results = cls.made.results
        cls._folder = tempfile.TemporaryDirectory()
        cls.folder = Path(cls._folder.name)
        cls.file = cls.folder / "results.json"
        cls.file.write_text(cls.made.results_json, encoding="utf-8")
        cls.read = reader.read_results(cls.file)

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def test_the_figure_code_reads_the_fits_the_trials_and_the_bins_as_the_analysis_wrote_them(self):
        self.assertTrue(reader.is_draft(self.read))
        self.assertEqual(reader.distance_divisor(self.read), self.results["distance_scale"]["divisor"])
        for condition in CONDITIONS:
            section = self.results["conditions"][condition]
            trials = reader.trials(self.read, condition)
            with self.subTest(condition=condition):
                self.assertEqual(len(trials), section["n_scored"])
                self.assertEqual([trial.d_mm for trial in trials], [trial["d_mm"] for trial in section["trials"]])
                for which in reader.FITS:
                    found, written = reader.fit(self.read, condition, which), section[f"{which}_fit"]
                    self.assertEqual(found is None, written is None)
                    if found is not None:
                        radius = (found.r50_median_mm, found.r50_lo_mm, found.r50_hi_mm)
                        self.assertEqual(radius, tuple(written[k] for k in ("r50_median_mm", "r50_lo_mm", "r50_hi_mm")))
                        self.assertEqual(
                            (found.open_below, found.open_above), (written["open_below"], written["open_above"])
                        )
                        self.assertEqual(list(found.blocks), written["blocks"])
                # The figure code counts the trials into bins by its own reading of the protocol's rule.
                counted = reader.binned(trials, reader.bin_width_mm(self.read), reader.interval_level(self.read))
                self.assertTrue(reader.same_bins(reader.file_bins(self.read, condition), counted))

    def test_the_workspace_test_adds_up_for_the_figure_code_pair_by_pair_and_region_by_region(self):
        self.assertTrue(reader.has_workspace(self.read))
        workspace = self.results["workspace"]
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.assertEqual(reader.pair_checks(self.read, policy), [])
                counts, notes = reader.region_counts(self.read, policy)
                self.assertEqual(notes, [])
                self.assertEqual(set(counts), {"trained_cells", "held_out_cells"})
                for region, count in counts.items():
                    written = workspace["by_region"][region][policy]
                    self.assertEqual((count.k, count.n), (written["k"], written["n"]))
                    self.assertEqual(count.lookup_expected, written["lookup_expected"])
                    self.assertEqual((count.lookup_lo, count.lookup_hi), (written["lookup_lo"], written["lookup_hi"]))
                placed = reader.placements(self.read, policy)
                self.assertEqual(len(placed), workspace["n_pairs"])
                self.assertEqual([p.region for p in placed], [pair["region"] for pair in workspace["pairs"]])
                found = reader.lookup(self.read, policy)
                self.assertEqual(
                    (found.lo, found.hi, found.observed, found.verdict),
                    tuple(workspace["lookup"][policy][key] for key in ("lo", "hi", "observed", "verdict")),
                )
        for dataset in ("clustered", "spread"):
            self.assertEqual(len(reader.demonstrations(self.read, dataset)), len(fixtures.audited_starts(dataset)))

    def test_figures_3_and_4_are_drawn_from_the_analysis_s_file_with_nothing_to_remark(self):
        paper, draft = self.folder / "paper-figures", self.folder / "draft-figures"
        made = make.make_figures([3, 4], root=fixtures.REPO, results_path=self.file, out_dir=paper, draft_dir=draft)
        self.assertEqual([figure.number for figure in made], [3, 4])
        for figure in made:
            with self.subTest(figure=figure.number):
                self.assertEqual(figure.status, make.WRITTEN, figure.line)
                self.assertEqual(figure.notes, ())
                self.assertTrue(figure.path.is_file())
                self.assertEqual(figure.path.parent, draft, "synthetic results are a draft: never the paper's folder")
                self.assertIn("DRAFT", figure.line)
        self.assertFalse(paper.exists())


if __name__ == "__main__":
    unittest.main()
