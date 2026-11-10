"""The command that draws the figures: what it prints, where it writes, and how it ends."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import fixtures_results
from fixtures_figures import add_audits, add_results, files_under, new_root
from tolerance_radius import config

_PATH = config.REPO_ROOT / "scripts" / "make_figures.py"
_SPEC = importlib.util.spec_from_file_location("make_figures", _PATH)
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)


class MakeFiguresTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = new_root(Path(self._dir.name))
        self.before = files_under(self.root)

    def tearDown(self):
        self._dir.cleanup()

    def run_command(self, *arguments: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = make_figures.main(["--root", str(self.root), *arguments])
        return code, out.getvalue(), err.getvalue()

    def written(self) -> list[str]:
        return sorted(set(files_under(self.root)) - set(self.before))

    def test_the_file_starts_as_a_script_that_any_python_3_runs(self):
        self.assertEqual(_PATH.read_text(encoding="utf-8").splitlines()[0], "#!/usr/bin/env python3")

    def test_with_what_is_there_today_it_writes_figure_1_says_what_the_others_wait_for_and_ends_well(self):
        code, out, err = self.run_command()
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.splitlines(), [
            "figure 1: wrote paper/figures/fig1-earlier-evaluation.pdf",
            "figure 2: skipped, waits for data/audit/clustered.json",
            "figure 3: skipped, waits for analysis/report/results.json",
            "figure 4: skipped, waits for analysis/report/results.json",
        ])  # fmt: skip
        self.assertEqual(self.written(), ["paper/figures/fig1-earlier-evaluation.pdf"])

    def test_only_names_the_figures_to_draw(self):
        code, out, _ = self.run_command("--only", "2", "4")
        self.assertEqual(code, 0)
        self.assertEqual([line.split(":")[0] for line in out.splitlines()], ["figure 2", "figure 4"])
        self.assertEqual(self.written(), [])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            make_figures.main(["--root", str(self.root), "--only", "5"])

    def test_with_everything_there_all_four_are_written_and_a_draft_stays_out_of_the_paper(self):
        add_audits(self.root)
        add_results(self.root, fixtures_results.load("held_out_placements"))
        self.before = files_under(self.root)
        code, out, err = self.run_command("--planned")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(self.written(), [
            "analysis/layouts/fig2-layouts-planned.pdf",
            "analysis/report-draft/figures/fig3-response-curves.pdf",
            "analysis/report-draft/figures/fig4-workspace.pdf",
            "paper/figures/fig1-earlier-evaluation.pdf",
            "paper/figures/fig2-layouts.pdf",
        ])  # fmt: skip
        self.assertEqual(out.count("DRAFT across it"), 2)
        self.assertEqual(out.count("PLANNED across it"), 1)

    def test_results_out_and_the_drafts_folder_can_be_named(self):
        results = add_results(self.root, fixtures_results.load("base"), Path("some/where/results.json"))
        self.before = files_under(self.root)
        named = {"--results": results, "--out": self.root / "a", "--draft-out": self.root / "b",
                 "--png": self.root / "look"}  # fmt: skip
        options = [text for option, path in named.items() for text in (option, str(path))]
        code, out, _ = self.run_command("--only", "1", "3", *options)
        self.assertEqual(code, 0)
        self.assertEqual(self.written(), ["a/fig1-earlier-evaluation.pdf", "b/fig3-response-curves.pdf",
                                          "look/fig1-earlier-evaluation.png",
                                          "look/fig3-response-curves.png"])  # fmt: skip
        self.assertIn("figure 3: wrote b/fig3-response-curves.pdf", out)

    def test_a_refused_figure_is_an_error_on_the_error_stream_and_the_others_are_still_made(self):
        add_results(self.root, fixtures_results.load("base"))
        self.before = files_under(self.root)
        code, out, err = self.run_command("--draft-out", str(self.root / "paper" / "figures"))
        self.assertEqual(code, 1)
        self.assertEqual([line.split(":")[:2] for line in err.splitlines()],
                         [["error", " figure 3"], ["error", " figure 4"]])  # fmt: skip
        self.assertIn("never written under paper/", err)
        self.assertIn("figure 1: wrote paper/figures/fig1-earlier-evaluation.pdf", out)
        self.assertEqual(self.written(), ["paper/figures/fig1-earlier-evaluation.pdf"])

    def test_one_more_list_of_forbidden_strings_can_be_named(self):
        private = self.root / "own.txt"
        private.write_text("held-out cell\n", encoding="utf-8")
        self.before = files_under(self.root)
        code, _, err = self.run_command("--only", "1", "--forbidden", str(private))
        self.assertEqual(code, 1)
        self.assertIn("line 1 of own.txt", err)
        self.assertEqual(self.written(), [])

    def test_what_the_figure_found_that_does_not_add_up_is_printed_under_its_line(self):
        results = fixtures_results.load("base")
        results["workspace"]["pairs"][0]["act_clustered"]["d_mm"] += 30.0
        add_results(self.root, results)
        code, out, _ = self.run_command("--only", "4")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("figure 4: wrote "))
        self.assertTrue(lines[1].startswith("  note: "))

    def test_what_it_prints_names_no_folder_of_this_machine(self):
        add_audits(self.root)
        add_results(self.root, fixtures_results.load("base"))
        _, out, err = self.run_command("--planned")
        for text in (out, err):
            self.assertNotIn(str(self.root), text)
            self.assertNotIn(str(Path.home()), text)


if __name__ == "__main__":
    unittest.main()
