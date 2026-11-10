"""The command that writes the analysis: what it writes and where, what it refuses, and that a draft stays out of the
paper. Every run here is given a temporary folder as its root: the command is never let loose on the repository."""

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import fixtures_experiment as fixtures

_SPEC = importlib.util.spec_from_file_location("make_report", fixtures.REPO / "scripts" / "make_report.py")
make_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_report)


def files_below(folder: Path) -> dict[str, bytes]:
    """Every file below a folder, by its name relative to the folder, with its bytes."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()
    }


class MakeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        cls.pristine = Path(cls._folder.name) / "pristine"
        cls.experiment = fixtures.build_experiment(cls.pristine, coarse=True)

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def setUp(self):
        # A copy of the experiment for each test, so that what one run writes is not there for the next.
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "root"
        shutil.copytree(self.pristine, self.root)

    def run_command(self, *arguments: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = make_report.main(["--root", str(self.root), *arguments])
        return code, out.getvalue(), err.getvalue()

    def refused(self, *arguments: str) -> str:
        """The one sentence a refused run prints, after checking that it wrote nothing."""
        before = files_below(self.root)
        code, out, err = self.run_command(*arguments)
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(files_below(self.root), before, "a refused run writes nothing")
        self.assertRegex(err, r"^make_report: refused: [^\n]+\.\n$")
        self.assertNotIn(str(self.root), err)
        return err

    def test_it_writes_the_results_the_report_and_the_paper_s_macros(self):
        code, out, err = self.run_command()
        self.assertEqual((code, err), (0, ""))
        written = files_below(self.root)
        for name in (
            "analysis/report/results.json",
            "analysis/report/report.md",
            "paper/generated/numbers.tex",
            "paper/generated/protocol.tex",
            "paper/generated/MACROS.md",
        ):
            self.assertIn(name, written)
            self.assertIn(f"wrote {name}", out)
        self.assertNotIn("analysis/report-draft/results.json", written)
        results = json.loads(written["analysis/report/results.json"])
        self.assertEqual((results["draft"], results["banner"]), (False, None))
        self.assertEqual(results["provenance"]["n_scored"], len(self.experiment.scored()))
        self.assertIn(f"trials: {len(self.experiment.scored())} scored", out)
        self.assertIn(f"of {len(make_report.load_stub().result_names())} have a value", out)

    def test_the_macros_it_writes_are_the_values_of_the_same_results(self):
        self.run_command()
        stub = make_report.load_stub()
        numbers = (self.root / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8")
        defined = stub.read_definitions(numbers)
        results = json.loads((self.root / "analysis" / "report" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(set(defined), set(stub.result_names()))
        self.assertEqual(defined["nTrialsTotal"][0], str(results["provenance"]["n_scored"]))
        self.assertEqual(
            defined["radiusReplay"][0], f"{results['conditions']['replay']['block_fit']['r50_median_mm']:.1f}"
        )
        self.assertEqual(defined["gridDriftMax"][0], stub.PLACEHOLDER, "what no record holds stays a visible ??")
        policy = results["conduct"]["execution"]["achieved_hz"]["policy"]
        self.assertEqual(defined["execControlRateAchieved"][0], f"{policy['median']:.1f}")
        self.assertIn("src: the counted trial logs, scored policy trials", defined["execControlRateAchieved"][1])
        self.assertIn(
            "status: frozen", (self.root / "paper" / "generated" / "protocol.tex").read_text(encoding="utf-8")
        )
        report_md = (self.root / "analysis" / "report" / "report.md").read_text(encoding="utf-8")
        self.assertIn(f"| `\\radiusReplay` | `{defined['radiusReplay'][0]}` |", report_md)

    def test_two_runs_write_identical_bytes(self):
        self.run_command()
        first = files_below(self.root)
        code, _, _ = self.run_command()
        self.assertEqual(code, 0)
        self.assertEqual(files_below(self.root), first)

    def test_what_it_writes_and_prints_names_no_folder_of_this_machine(self):
        _, out, _ = self.run_command()
        written = files_below(self.root)
        outputs = [written[name].decode("utf-8") for name in written if name.startswith(("analysis/report/", "paper/"))]
        for text in (out, *outputs):
            self.assertNotIn(str(self.root), text)
            self.assertNotIn(str(self.root.resolve()), text)
            self.assertNotIn(str(Path.home()), text)
            self.assertNotRegex(text, r"(?<![\w.])(?:/Users|/home|/private|/var|/tmp)/")

    def test_a_draft_goes_to_its_own_folder_and_writes_nothing_under_paper(self):
        code, out, err = self.run_command("--draft")
        self.assertEqual((code, err), (0, ""))
        written = files_below(self.root)
        self.assertIn("analysis/report-draft/results.json", written)
        self.assertIn("analysis/report-draft/report.md", written)
        self.assertEqual([name for name in written if name.startswith(("paper/", "analysis/report/"))], [])
        results = json.loads(written["analysis/report-draft/results.json"])
        self.assertTrue(results["draft"])
        self.assertTrue(written["analysis/report-draft/results.json"].startswith(b'{"banner": "DRAFT'))
        self.assertTrue(written["analysis/report-draft/report.md"].startswith(b"**DRAFT"))
        self.assertIn("nothing was written under paper/", out)

    def test_a_draft_accepts_trials_that_were_not_run_under_a_frozen_protocol(self):
        def unfreeze(index, entry):
            entry["versions"]["protocol_status"] = "draft"

        fixtures.rewrite_log(self.root / "data" / "trials" / "main.jsonl", unfreeze)
        self.assertIn("not run under a frozen protocol", self.refused())
        code, _, _ = self.run_command("--draft")
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "paper").exists())

    def test_a_draft_can_carry_a_note_and_go_to_another_folder(self):
        elsewhere = self.root / "analysis" / "report-example"
        code, _, _ = self.run_command("--draft", "--out", str(elsewhere), "--note", fixtures.SYNTHETIC_NOTE)
        self.assertEqual(code, 0)
        self.assertTrue((elsewhere / "results.json").read_text(encoding="utf-8").startswith('{"banner": "SYNTHETIC'))
        self.assertTrue((elsewhere / "report.md").read_text(encoding="utf-8").startswith("**SYNTHETIC"))
        self.assertFalse((self.root / "analysis" / "report-draft").exists())
        results = json.loads((elsewhere / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["provenance"]["note"], fixtures.SYNTHETIC_NOTE)

    def test_a_draft_is_never_written_where_the_counted_report_goes(self):
        for place in (self.root / "analysis" / "report", self.root / "paper" / "generated", self.root / "paper"):
            with self.subTest(place=place.name):
                self.assertIn("kept for the counted report", self.refused("--draft", "--out", str(place)))

    def test_the_counted_report_takes_no_note_and_no_other_place(self):
        self.assertIn("belong to a draft", self.refused("--note", "anything"))
        self.assertIn("belong to a draft", self.refused("--out", str(self.root / "elsewhere")))

    def test_a_pilot_log_is_refused(self):
        err = self.refused("--trials", str(self.root / "data" / "trials" / "pilot_replay.jsonl"))
        self.assertIn("data/trials/pilot_replay.jsonl is a pilot log", err)

    def test_an_unfrozen_protocol_file_is_refused_for_a_counted_report(self):
        (self.root / "config" / "protocol.toml").write_text(
            fixtures.protocol_text(status="draft", version="1.0", coarse=True), encoding="utf-8"
        )
        self.assertIn("config/protocol.toml has status 'draft'", self.refused())

    def test_a_distance_that_does_not_match_the_audit_is_refused(self):
        at = next(index for index, record in enumerate(self.experiment.records) if record.scored)

        def farther(index, entry):
            if index == at:
                entry["nearest_demo_distance_mm"] += 1.0

        fixtures.rewrite_log(self.root / "data" / "trials" / "main.jsonl", farther)
        self.assertIn(self.experiment.records[at].planned.trial_id, self.refused())

    def test_trials_run_with_another_setting_are_refused_by_name(self):
        scored = [i for i, r in enumerate(self.experiment.records) if r.scored and r.planned.condition == "act_spread"]

        def no_smoothing(index, entry):
            if index in scored[3:6]:
                entry["execution"]["temporal_ensemble_coeff"] = 0.0

        fixtures.rewrite_log(self.root / "data" / "trials" / "main.jsonl", no_smoothing)
        err = self.refused()
        self.assertIn("temporal_ensemble_coeff is not the same in every scored act_spread trial", err)
        for index in scored[3:6]:
            self.assertIn(self.experiment.records[index].planned.trial_id, err)
        self.assertIn("refused", self.refused("--draft"), "a draft does not excuse it")

    def test_two_protocol_versions_are_refused(self):
        def later(index, entry):
            if index > 60:
                entry["versions"]["protocol_version"] = "1.1"

        fixtures.rewrite_log(self.root / "data" / "trials" / "main.jsonl", later)
        self.assertIn("more than one protocol version", self.refused())

    def test_other_folders_can_be_named(self):
        instrument = self.root / "elsewhere"
        instrument.mkdir()
        code, _, _ = self.run_command("--draft", "--instrument-dir", str(instrument))
        self.assertEqual(code, 0)
        results = json.loads((self.root / "analysis" / "report-draft" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["distance_scale"]["n_runs"], 0)
        self.assertIsNone(results["instrument"]["by_eye"])
        self.assertEqual(results["second_scoring"]["all"], self.experiment.second_scoring["all"])

    def test_the_folder_of_the_second_scoring_can_be_named(self):
        elsewhere = self.root / "scored_again"
        shutil.move(self.root / "data" / "second_scoring" / "main", elsewhere)
        code, _, _ = self.run_command()
        self.assertEqual(code, 0)
        results = json.loads((self.root / "analysis" / "report" / "results.json").read_text(encoding="utf-8"))
        self.assertIsNone(results["second_scoring"], "nothing is where the comparison writes by default")
        numbers = (self.root / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8")
        stub = make_report.load_stub()
        self.assertEqual(stub.read_definitions(numbers)["nSecondScored"][0], stub.PLACEHOLDER)
        code, _, _ = self.run_command("--second-scoring-dir", str(elsewhere))
        self.assertEqual(code, 0)
        results = json.loads((self.root / "analysis" / "report" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["second_scoring"]["file"], "scored_again/agreement.json")
        self.assertEqual(results["second_scoring"]["all"], self.experiment.second_scoring["all"])
        numbers = (self.root / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8")
        self.assertEqual(
            stub.read_definitions(numbers)["nSecondScored"][0], str(self.experiment.second_scoring["all"]["n_items"])
        )


class RepositoryTests(unittest.TestCase):
    """What is committed where the real numbers will go."""

    def test_the_command_starts_as_a_script_and_puts_the_package_on_its_path(self):
        text = (fixtures.REPO / "scripts" / "make_report.py").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], "#!/usr/bin/env python3")
        self.assertIn('sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))', text)

    def test_nothing_synthetic_is_where_a_real_number_will_go(self):
        places = [fixtures.REPO / "paper" / "generated", fixtures.REPO / "analysis" / "report", fixtures.REPO / "data"]
        marks = ("SYNTHETIC", "0123abc456", "fixtures_experiment")
        for place in places:
            for path in sorted(place.rglob("*")) if place.is_dir() else []:
                if path.is_file() and path.suffix in {".tex", ".md", ".json", ".jsonl", ".csv"}:
                    text = path.read_text(encoding="utf-8")
                    for mark in marks:
                        self.assertNotIn(mark, text, f"{path.relative_to(fixtures.REPO)} holds something synthetic")

    def test_the_committed_macros_hold_a_value_only_if_a_counted_report_is_committed_with_them(self):
        stub = make_report.load_stub()
        numbers = fixtures.REPO / "paper" / "generated" / "numbers.tex"
        filled = stub.holds_values(numbers.read_text(encoding="utf-8"))
        results = fixtures.REPO / "analysis" / "report" / "results.json"
        if not results.is_file():
            self.assertEqual(filled, [], "numbers.tex holds values and there is no analysis/report/results.json")
        else:
            self.assertFalse(json.loads(results.read_text(encoding="utf-8"))["draft"])


if __name__ == "__main__":
    unittest.main()
