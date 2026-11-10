"""Making the figures: which are written, which wait for a file, where a draft goes, and the files themselves."""

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fixtures_results
import pdf_facts
from fixtures_figures import add_audits, add_results, files_under, new_root
from tolerance_radius import config
from tolerance_radius.figures import FILE_NAMES, HEIGHTS_MM, make, style
from tolerance_radius.figures.make import REFUSED, SKIPPED, WRITTEN

REPO = pdf_facts.REPO
CONTRACT_ONLY = ", cut down to the contract"


class RootTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = new_root(Path(self._dir.name))
        self.before = files_under(self.root)

    def tearDown(self):
        self._dir.cleanup()

    def make(self, *numbers: int, **options) -> dict[str, make.Made]:
        """Make some figures (all four when none is named); what became of each, by the name its line starts with."""
        made = make.make_figures(numbers or make.NUMBERS, root=self.root, **options)
        return {figure.line.split(":")[0]: figure for figure in made}

    def written(self) -> list[str]:
        """The files that are there now and were not there at the start."""
        return sorted(set(files_under(self.root)) - set(self.before))


class ForbiddenListTests(RootTestCase):
    def test_the_papers_list_is_read_as_the_papers_checker_reads_it(self):
        ours = make.read_forbidden(REPO / "paper" / "forbidden.txt")
        theirs = pdf_facts._check_paper().read_forbidden(REPO / "paper" / "forbidden.txt")
        self.assertGreater(len(ours), 10)
        self.assertEqual([(entry.pattern.pattern, entry.pattern.flags, entry.shown) for entry in ours],
                         [(entry.pattern.pattern, entry.pattern.flags, entry.shown) for entry in theirs])  # fmt: skip

    def test_an_entry_of_a_private_list_is_named_by_its_line_and_never_repeated(self):
        private = self.root / "own.txt"
        lines = ["# the author's strings", "", "Someone Surname", "re:lab\\s+name", "pdf:their-hub-name"]
        private.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries = make.read_forbidden(private)
        self.assertEqual([entry.shown for entry in entries], [f"line {number} of own.txt" for number in (3, 4, 5)])
        self.assertTrue(entries[0].pattern.search("by someone surname, 2026"))  # case is ignored
        self.assertTrue(entries[1].pattern.search("the Lab  Name"))
        self.assertFalse(entries[0].pattern.search("someone else"))
        self.assertTrue(entries[2].pattern.search("their-hub-name/dataset"))

    def test_the_lists_are_the_papers_the_authors_own_when_it_is_there_and_those_named(self):
        shipped = len(make.forbidden_entries(self.root))
        (self.root / "paper" / make.LOCAL_LIST).write_text("Someone Surname\n", encoding="utf-8")
        extra = self.root / "more.txt"
        extra.write_text("one\ntwo\n", encoding="utf-8")
        self.assertEqual(len(make.forbidden_entries(self.root)), shipped + 1)
        self.assertEqual(len(make.forbidden_entries(self.root, [extra])), shipped + 3)
        with self.assertRaisesRegex(style.FigureError, "missing.txt"):
            make.forbidden_entries(self.root, [self.root / "missing.txt"])

    def test_without_the_papers_list_no_figure_is_written(self):
        (self.root / "paper" / make.SHIPPED_LIST).unlink()
        made = self.make(1)["figure 1"]
        self.assertEqual(made.status, REFUSED)
        self.assertIn("paper/forbidden.txt", made.line)
        self.assertEqual(self.written(), [])

    def test_a_figure_that_holds_a_string_of_the_authors_list_is_refused_without_repeating_it(self):
        # "held-out cell" is written in Figure 1's legend: here it stands in for a name.
        (self.root / "paper" / make.LOCAL_LIST).write_text("# private\nheld-out cell\n", encoding="utf-8")
        self.before = files_under(self.root)
        made = self.make(1)["figure 1"]
        self.assertEqual(made.status, REFUSED)
        self.assertIn("line 2 of forbidden.local.txt", made.line)
        self.assertNotIn("held-out", made.line)
        self.assertEqual(self.written(), [])


class WhatIsThereTodayTests(RootTestCase):
    def test_figure_1_is_written_and_the_others_wait_for_their_files(self):
        made = self.make()
        self.assertEqual(list(made), ["figure 1", "figure 2", "figure 3", "figure 4"])
        self.assertEqual([figure.status for figure in made.values()], [WRITTEN, SKIPPED, SKIPPED, SKIPPED])
        self.assertEqual(made["figure 1"].line, "figure 1: wrote paper/figures/fig1-earlier-evaluation.pdf")
        self.assertEqual(made["figure 2"].line, "figure 2: skipped, waits for data/audit/clustered.json")
        self.assertEqual(made["figure 3"].line, "figure 3: skipped, waits for analysis/report/results.json")
        self.assertEqual(made["figure 4"].line, "figure 4: skipped, waits for analysis/report/results.json")
        self.assertEqual(self.written(), ["paper/figures/fig1-earlier-evaluation.pdf"])
        self.assertEqual(made["figure 1"].path, self.root / "paper" / "figures" / FILE_NAMES[1])

    def test_only_the_figures_asked_for_are_made_each_once(self):
        made = self.make(3, 1, 3)
        self.assertEqual(list(made), ["figure 1", "figure 3"])
        with self.assertRaisesRegex(ValueError, "5"):
            make.make_figures([1, 5], root=self.root)
        self.assertEqual(self.written(), ["paper/figures/fig1-earlier-evaluation.pdf"])

    def test_without_its_csv_figure_1_waits_too(self):
        (self.root / make.EARLIER_TRIALS).unlink()
        made = self.make(1)["figure 1"]
        self.assertEqual(made.status, SKIPPED)
        self.assertEqual(made.line, "figure 1: skipped, waits for analysis/pilot_august/trials_with_distance.csv")

    def test_another_folder_can_be_named_for_the_papers_figures(self):
        made = self.make(1, out_dir=self.root / "elsewhere")["figure 1"]
        self.assertEqual(made.line, "figure 1: wrote elsewhere/fig1-earlier-evaluation.pdf")
        self.assertEqual(self.written(), ["elsewhere/fig1-earlier-evaluation.pdf"])

    def test_a_picture_for_looking_at_can_be_saved_beside(self):
        self.make(1, png_dir=self.root / "look")
        data = (self.root / "look" / "fig1-earlier-evaluation.png").read_bytes()
        self.assertEqual(data[:4], b"\x89PNG")
        self.assertEqual(int.from_bytes(data[16:20], "big"), 1039)  # 88 mm at 300 dots per inch

    def test_a_csv_that_cannot_be_read_is_refused_and_named(self):
        path = self.root / make.EARLIER_TRIALS
        path.write_text(path.read_text(encoding="utf-8").replace(",ood,", ",elsewhere,", 1), encoding="utf-8")
        made = self.make(1)["figure 1"]
        self.assertEqual(made.status, REFUSED)
        self.assertRegex(made.line, r"^figure 1: not written: trials_with_distance.csv, line \d+")
        self.assertEqual(self.written(), [])


class AuditTests(RootTestCase):
    def test_one_audit_is_not_enough_and_the_missing_one_is_named(self):
        add_audits(self.root, ["clustered"])
        made = self.make(2)["figure 2"]
        self.assertEqual((made.status, made.line), (SKIPPED, "figure 2: skipped, waits for data/audit/spread.json"))

    def test_with_both_audits_figure_2_is_written_into_the_papers_folder_with_no_stamp(self):
        add_audits(self.root)
        self.before = files_under(self.root)
        made = self.make(2)["figure 2"]
        self.assertEqual((made.status, made.line), (WRITTEN, "figure 2: wrote paper/figures/fig2-layouts.pdf"))
        self.assertEqual(self.written(), ["paper/figures/fig2-layouts.pdf"])

    def test_an_audit_that_is_not_one_is_refused(self):
        add_audits(self.root)
        make.audit_path("spread", self.root).write_text('{"schema_version": 1}', encoding="utf-8")
        made = self.make(2)["figure 2"]
        self.assertEqual(made.status, REFUSED)
        self.assertIn("data/audit/spread.json is not a start-state audit", made.line)
        make.audit_path("spread", self.root).write_text("not json", encoding="utf-8")
        self.assertEqual(self.make(2)["figure 2"].status, REFUSED)

    def test_the_picture_of_the_plan_goes_beside_the_layouts_and_says_planned(self):
        made = self.make(2, planned=True)
        self.assertEqual(list(made), ["figure 2", "figure 2 as planned"])
        self.assertEqual(made["figure 2"].status, SKIPPED)
        planned = made["figure 2 as planned"]
        self.assertEqual(planned.status, WRITTEN)
        self.assertEqual(planned.line, "figure 2 as planned: wrote analysis/layouts/fig2-layouts-planned.pdf "
                                       "(PLANNED across it: not a figure of the paper)")  # fmt: skip
        self.assertEqual(self.written(), ["analysis/layouts/fig2-layouts-planned.pdf"])  # nothing under paper/

    def test_the_plan_is_drawn_only_with_figure_2_and_only_when_asked(self):
        self.assertEqual(list(self.make(1, 3, planned=True)), ["figure 1", "figure 3"])
        self.assertEqual(list(self.make(2)), ["figure 2"])

    def test_with_the_audits_there_the_plan_can_still_be_drawn_and_the_two_are_different_files(self):
        add_audits(self.root)
        made = self.make(2, planned=True)
        self.assertEqual([figure.status for figure in made.values()], [WRITTEN, WRITTEN])
        self.assertNotEqual(made["figure 2"].path.read_bytes(), made["figure 2 as planned"].path.read_bytes())

    def test_without_the_layouts_the_plan_waits_for_them(self):
        (self.root / "config" / "layouts" / "clustered.json").unlink()
        planned = self.make(2, planned=True)["figure 2 as planned"]
        self.assertEqual(planned.status, SKIPPED)
        self.assertEqual(planned.line, "figure 2 as planned: skipped, waits for config/layouts/clustered.json")


class DraftTests(RootTestCase):
    def test_a_draft_goes_to_the_drafts_folder_with_draft_across_it_wherever_the_results_file_is(self):
        # Even a draft that sits where the counted results belong is not drawn into the paper's folder.
        add_results(self.root, fixtures_results.load("base"))
        self.before = files_under(self.root)
        made = self.make(3, 4)
        for number in (3, 4):
            figure = made[f"figure {number}"]
            self.assertEqual(figure.status, WRITTEN)
            self.assertEqual(figure.line, f"figure {number}: wrote analysis/report-draft/figures/{FILE_NAMES[number]} "
                                          "(DRAFT across it: not a figure of the paper)")  # fmt: skip
        self.assertEqual(self.written(), [f"analysis/report-draft/figures/{FILE_NAMES[number]}" for number in (3, 4)])
        self.assertFalse((self.root / "paper" / "figures").exists())

    def test_naming_the_papers_folder_for_the_figures_does_not_send_a_draft_there(self):
        path = add_results(self.root, fixtures_results.load("base"), Path("analysis/report-draft/results.json"))
        self.before = files_under(self.root)
        made = self.make(1, 3, 4, results_path=path, out_dir=self.root / "paper" / "figures")
        self.assertEqual([figure.status for figure in made.values()], [WRITTEN] * 3)
        self.assertEqual(self.written(), ["analysis/report-draft/figures/fig3-response-curves.pdf",
                                          "analysis/report-draft/figures/fig4-workspace.pdf",
                                          "paper/figures/fig1-earlier-evaluation.pdf"])  # fmt: skip

    def test_a_folder_under_paper_is_refused_for_a_draft(self):
        add_results(self.root, fixtures_results.load("base"))
        self.before = files_under(self.root)
        for folder in (self.root / "paper" / "figures", self.root / "paper", self.root / "paper" / "figures" / "new"):
            made = self.make(3, 4, draft_dir=folder)
            for figure in made.values():
                self.assertEqual(figure.status, REFUSED)
                self.assertIn("is a draft, and a figure of a draft is never written under paper/", figure.line)
        self.assertEqual(self.written(), [])

    def test_the_repositorys_own_paper_folder_is_refused_for_a_draft_from_any_root(self):
        add_results(self.root, fixtures_results.load("base"))
        with mock.patch.object(style, "save_pdf") as save:  # whatever the code does, this test writes nothing there
            made = self.make(3, 4, draft_dir=config.REPO_ROOT / "paper" / "figures")
        save.assert_not_called()
        self.assertEqual([figure.status for figure in made.values()], [REFUSED, REFUSED])

    def test_a_figure_of_a_draft_that_does_not_say_so_is_not_saved(self):
        add_results(self.root, fixtures_results.load("base"))
        self.before = files_under(self.root)
        with mock.patch.object(style.Page, "stamp", lambda self, word: None):
            made = self.make(3)["figure 3"]
        self.assertEqual(made.status, REFUSED)
        self.assertIn("must carry the word DRAFT", made.line)
        self.assertEqual(self.written(), [])

    def test_results_that_count_go_to_the_papers_folder_without_a_stamp(self):
        # Synthetic results marked as counted exist in this temporary folder only.
        add_results(self.root, fixtures_results.changed("held_out_placements", draft=False))
        self.before = files_under(self.root)
        made = self.make(3, 4)
        self.assertEqual([figure.line for figure in made.values()],
                         ["figure 3: wrote paper/figures/fig3-response-curves.pdf",
                          "figure 4: wrote paper/figures/fig4-workspace.pdf"])  # fmt: skip
        self.assertEqual(self.written(), ["paper/figures/fig3-response-curves.pdf", "paper/figures/fig4-workspace.pdf"])


class ResultsInputTests(RootTestCase):
    def test_without_a_workspace_test_figure_4_waits_and_figure_3_is_written(self):
        add_results(self.root, fixtures_results.load("threshold_blocks_only"))
        made = self.make(3, 4)
        self.assertEqual(made["figure 3"].status, WRITTEN)
        self.assertEqual(made["figure 4"].status, SKIPPED)
        self.assertEqual(made["figure 4"].line,
                         "figure 4: skipped, waits for the workspace test in analysis/report/results.json")  # fmt: skip

    def test_without_a_threshold_trial_figure_3_waits(self):
        results = fixtures_results.load("base")
        for condition in ("act_clustered", "replay"):
            results["conditions"][condition] = None
        add_results(self.root, results)
        made = self.make(3)["figure 3"]
        self.assertEqual((made.status, made.line),
                         (SKIPPED, "figure 3: skipped, waits for trials of a threshold block in "
                                   "analysis/report/results.json"))  # fmt: skip

    def test_a_results_file_named_elsewhere_is_the_one_read(self):
        path = add_results(self.root, fixtures_results.load("open_above"), Path("elsewhere/results.json"))
        made = self.make(3, results_path=path)["figure 3"]
        self.assertEqual(made.status, WRITTEN)
        missing = self.make(3, results_path=self.root / "nowhere.json")["figure 3"]
        self.assertEqual(missing.line, "figure 3: skipped, waits for nowhere.json")

    def test_a_results_file_that_cannot_be_used_is_refused_not_skipped(self):
        for text, said in (('{"schema_version": 2, "draft": true}', "schema_version"), ("not json", "not a JSON file"),
                           ('{"schema_version": 1}', "draft")):  # fmt: skip
            path = self.root / make.RESULTS
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            made = self.make(3, 4)
            for figure in made.values():
                self.assertEqual(figure.status, REFUSED, text)
                self.assertIn(said, figure.line)

    def test_a_refused_figure_leaves_the_file_of_an_earlier_run_alone_and_says_so(self):
        self.make(1)
        path = self.root / "paper" / "figures" / FILE_NAMES[1]
        earlier = path.read_bytes()
        (self.root / "paper" / make.LOCAL_LIST).write_text("held-out cell\n", encoding="utf-8")
        made = self.make(1)["figure 1"]
        self.assertEqual(made.status, REFUSED)
        self.assertIn("the file of an earlier run is left as it was", made.line)
        self.assertEqual(path.read_bytes(), earlier)


class EveryFileTests(unittest.TestCase):
    """Each figure, written for each case, read back from the disk."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        folder = Path(cls._dir.name)
        #: `(case, figure number) -> the bytes of the file`, and whether its line says a word is stamped across it.
        cls.files: dict[tuple[str, int], bytes] = {}
        cls.stamped: dict[tuple[str, int], bool] = {}

        def keep(name: str, figure: make.Made) -> None:
            cls.files[(name, figure.number)] = figure.path.read_bytes()
            cls.stamped[(name, figure.number)] = "across it" in figure.line

        # Everything at once, twice over, in one folder: Figures 1 and 2 are kept from this run.
        root = new_root(folder / "today")
        add_audits(root)
        add_results(root, fixtures_results.load("base"))
        first = make.make_figures(root=root, planned=True)
        cls.first = [figure.path.read_bytes() for figure in first]
        cls.again = [figure.path.read_bytes() for figure in make.make_figures(root=root, planned=True)]
        for figure in first:
            if figure.number <= 2:
                keep("planned" if figure.path.name == make.PLANNED_PREVIEW.name else "today", figure)
        # Figures 3 and 4 for every synthetic results file, as it is and cut down to the fields of the contract.
        # Figure 3 uses no field beyond the contract's, so only Figure 4 is drawn from the files cut down to it.
        cases = [(name, fixtures_results.load(name), (3, 4)) for name in fixtures_results.CASES]
        cases += [(name + CONTRACT_ONLY, fixtures_results.contract_only(name), (4,)) for name in fixtures_results.CASES]
        cases.append(("counted", fixtures_results.changed("held_out_placements", draft=False), (3, 4)))
        for index, (name, results, numbers) in enumerate(cases):
            root = new_root(folder / str(index))
            add_results(root, results)
            for figure in make.make_figures(numbers, root=root):
                if figure.status == WRITTEN:
                    keep(name, figure)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_every_figure_was_written_for_every_case_that_has_it(self):
        self.assertEqual(sorted(key for key in self.files if key[1] <= 2), [("planned", 2), ("today", 1), ("today", 2)])
        with_workspace = [name for name in fixtures_results.CASES if name != "threshold_blocks_only"]
        for name in fixtures_results.CASES:
            self.assertIn((name, 3), self.files)
        for suffix in ("", CONTRACT_ONLY):
            for name in with_workspace:
                self.assertIn((name + suffix, 4), self.files)
            self.assertNotIn(("threshold_blocks_only" + suffix, 4), self.files)
        self.assertIn(("counted", 3), self.files)
        self.assertIn(("counted", 4), self.files)
        self.assertEqual(len(self.files), 3 + 5 + 2 * 4 + 2)

    def test_the_page_is_a_column_wide_and_as_high_as_the_plan_says(self):
        for (name, number), data in self.files.items():
            width, height = pdf_facts.page_size_mm(data)
            self.assertAlmostEqual(width, 88.0, delta=0.5, msg=f"figure {number}, {name}")
            self.assertAlmostEqual(height, HEIGHTS_MM[number], delta=0.5, msg=f"figure {number}, {name}")

    def test_no_text_is_under_seven_points(self):
        for (name, number), data in self.files.items():
            sizes = pdf_facts.text_sizes_pt(data)
            self.assertGreater(len(sizes), 10, f"figure {number}, {name}")
            self.assertGreaterEqual(min(sizes), 7.0, f"figure {number}, {name}")

    def test_the_bytes_say_nothing_of_the_maker(self):
        for (name, number), data in self.files.items():
            said = f"figure {number}, {name}"
            self.assertEqual(pdf_facts.forbidden_hits(data), [], said)
            self.assertEqual(pdf_facts.names_this_machine(data), [], said)
            for word in (b"CreationDate", b"ModDate", b"Producer", b"Creator", b"atplotlib"):
                self.assertNotIn(word, data, said)

    def test_nor_do_the_compressed_parts_once_opened(self):
        # The paper's checker reads a file's bytes as they are, so it cannot see into a compressed stream. Here every
        # stream is opened: the drawing instructions, the font programs and their tables of characters.
        for (name, number), data in self.files.items():
            said = f"figure {number}, {name}"
            opened = b"\n".join(pdf_facts.streams(data))
            self.assertGreater(len(opened), len(data), said)
            self.assertEqual(pdf_facts.forbidden_hits(opened), [], said)
            self.assertEqual(pdf_facts.names_this_machine(opened), [], said)
            for word in (b"CreationDate", b"atplotlib", b"/Users/", b"/home/"):
                self.assertNotIn(word, opened, said)

    def test_the_file_is_a_vector_drawing_with_its_fonts_embedded_as_truetype(self):
        for (name, number), data in self.files.items():
            said = f"figure {number}, {name}"
            self.assertEqual(data[:5], b"%PDF-", said)
            self.assertEqual(pdf_facts.font_kinds(data), {b"Type0", b"CIDFontType2"}, said)
            self.assertNotIn(b"/Subtype /Image", data, said)
            self.assertLess(len(data), 150_000, said)

    def test_the_same_inputs_give_the_same_bytes(self):
        self.assertEqual(len(self.first), 5)  # the four figures and the picture of the plan
        self.assertEqual(self.again, self.first)

    def test_only_what_is_not_the_papers_carries_a_word_across_it(self):
        for key, stamped in self.stamped.items():
            self.assertEqual(stamped, key[0] not in ("today", "counted"), key)

    def test_the_cases_give_different_pictures(self):
        for number in (3, 4):
            files = [data for (name, n), data in self.files.items() if n == number and CONTRACT_ONLY not in name]
            self.assertEqual(len(set(files)), len(files))


class CommittedFilesTests(unittest.TestCase):
    """The figures kept in the repository are the ones the code draws from the data today."""

    def again(self, *numbers: int, **options) -> dict[str, bytes]:
        with tempfile.TemporaryDirectory() as folder:
            root = new_root(Path(folder))
            made = make.make_figures(numbers, root=root, **options)
            written = [figure.path for figure in made if figure.path]
            return {path.relative_to(root).as_posix(): path.read_bytes() for path in written}

    def test_the_papers_figure_1_is_drawn_from_the_csv_as_it_is(self):
        kept = REPO / make.PAPER_FIGURES / FILE_NAMES[1]
        self.assertTrue(kept.is_file(), "paper/figures/fig1-earlier-evaluation.pdf is missing: scripts/make_figures.py")
        self.assertEqual(kept.read_bytes(), self.again(1)["paper/figures/fig1-earlier-evaluation.pdf"],
                         "the paper's figure is not what the code draws now: make_figures.py --only 1")  # fmt: skip

    def test_the_picture_of_the_plan_is_drawn_from_the_layouts_as_they_are(self):
        kept = REPO / make.PLANNED_PREVIEW
        self.assertTrue(kept.is_file(), "the picture of the plan is missing: make_figures.py --only 2 --planned")
        self.assertEqual(kept.read_bytes(), self.again(2, planned=True)[make.PLANNED_PREVIEW.as_posix()],
                         "run scripts/make_figures.py --only 2 --planned")  # fmt: skip

    def test_the_papers_folder_holds_only_figures_of_the_plan_under_their_names(self):
        # One picture is not of the plan and not drawn by make_figures.py: the appendix's example of the placement
        # overlay, rendered from a recorded trial by scripts/make_overlay_example.py (2026-10-09). It is the only
        # file here that is not a PDF.
        pictures = {"fig5-overlay-example.png"}
        allowed = set(FILE_NAMES.values()) | {"README.md"} | pictures
        present = {path.name for path in (REPO / make.PAPER_FIGURES).iterdir() if path.name != ".DS_Store"}
        self.assertLessEqual(present, allowed)
        for name in present - {"README.md"} - pictures:
            self.assertRegex(name, re.compile(r"^[A-Za-z0-9-]+\.pdf$"))
        for name in present & pictures:
            data = (REPO / make.PAPER_FIGURES / name).read_bytes()
            self.assertRegex(name, re.compile(r"^[A-Za-z0-9-]+\.png$"))
            self.assertNotIn(b"tEXt", data, f"{name} carries a text chunk: a picture for the paper carries no metadata")
            self.assertNotIn(b"eXIf", data, f"{name} carries camera metadata")

    def test_what_is_kept_says_nothing_of_the_maker(self):
        for path in (REPO / make.PAPER_FIGURES / FILE_NAMES[1], REPO / make.PLANNED_PREVIEW):
            data = path.read_bytes()
            self.assertEqual(pdf_facts.forbidden_hits(data), [], path.name)
            self.assertEqual(pdf_facts.names_this_machine(data), [], path.name)
            self.assertNotIn(b"CreationDate", data, path.name)


if __name__ == "__main__":
    unittest.main()
