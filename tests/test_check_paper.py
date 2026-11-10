"""The paper's checker (`paper/check_paper.py`): which placeholders count as open.

A number macro without a value prints a bold ?? where the paper uses it. A macro that the paper does not use prints
nothing, whatever its value: the numbers of a block of trials that was planned and not run stay placeholders for
ever, and must not keep the paper from being final. Every paper here is a small one written to a temporary folder;
the repository's own paper is not read.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("check_paper_for_its_tests", REPO / "paper" / "check_paper.py")
check_paper = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_paper  # its dataclasses look their own module up by name
_SPEC.loader.exec_module(check_paper)

PLACEHOLDER = check_paper.PLACEHOLDER
#: The sections the checker wants `main.tex` to read.
SECTIONS = tuple(check_paper.TARGET_WORDS)


def definition(name: str, body: str) -> str:
    """One macro as the generated files define it."""
    return f"\\newcommand{{\\{name}}}{{{body}\\xspace}} % what it holds"


def write_paper(root: Path, macros: dict[str, str], results: str) -> None:
    """A paper that passes every check: the macros in `generated/numbers.tex`, and `results` as its results section.

    The other sections hold one plain sentence each. The protocol's constants say that the protocol is frozen, so
    that nothing is open but what a test puts into the results section.
    """
    (root / "sections").mkdir(parents=True)
    (root / "generated").mkdir()
    inputs = "\n".join(f"\\input{{sections/{name}}}" for name in SECTIONS)
    (root / "main.tex").write_text(
        "\\documentclass{article}\n\\input{generated/numbers}\n\\begin{document}\n" + inputs + "\n\\end{document}\n",
        encoding="utf-8",
    )
    for name in SECTIONS:
        (root / "sections" / f"{name}.tex").write_text("A plain sentence stands here.\n", encoding="utf-8")
    (root / "sections" / "results.tex").write_text(results, encoding="utf-8")
    numbers = "\n".join(definition(name, body) for name, body in macros.items())
    (root / "generated" / "numbers.tex").write_text(numbers + "\n", encoding="utf-8")
    (root / "generated" / "protocol.tex").write_text("% protocol status: frozen\n", encoding="utf-8")
    (root / "forbidden.txt").write_text("# nothing is forbidden in this small paper\n", encoding="utf-8")


#: A results section that uses one macro in its text, one in a comment and one in a commented-out branch.
RESULTS = """\
The radius is \\radiusUsed~mm over \\nFilled trials.
% An earlier draft said: the spread policy's radius is \\radiusInComment~mm.
% IF the spread policy's block was run
The kept branch counts \\nFilled trials again.
% ELSE it was not run
%| The other branch would print \\radiusInBranch~mm.
% ENDIF
"""
MACROS = {
    "radiusUsed": PLACEHOLDER,
    "radiusInComment": PLACEHOLDER,
    "radiusInBranch": PLACEHOLDER,
    "radiusNowhere": PLACEHOLDER,
    "nFilled": "24",
}


class PlaceholderTests(unittest.TestCase):
    def paper(self, macros: dict[str, str] = MACROS, results: str = RESULTS) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "paper"
        write_paper(root, macros, results)
        return root

    def run_command(self, root: Path, *arguments: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = check_paper.main(["--root", str(root), *arguments])
        return code, out.getvalue()

    def test_a_placeholder_is_open_only_where_the_paper_prints_it(self):
        report = check_paper.check(self.paper(), [], final=False)
        self.assertEqual(report.failures, [])
        summary = "\n".join(report.summary)
        self.assertIn("(1 of the 5 defined are placeholders that the paper uses)", summary)
        self.assertIn("unused      3 placeholders that the paper does not use: they are not open", summary)
        self.assertRegex(summary, r"(?m)^open +1 placeholders that the paper uses, 1 IF blocks; protocol frozen$")
        notes = "\n".join(report.notes)
        self.assertIn("placeholders that the paper uses: radiusUsed\n", notes + "\n")
        self.assertIn("placeholders that the paper does not use: radiusInBranch, radiusInComment, radiusNowhere", notes)

    def test_with_final_a_placeholder_that_the_paper_uses_fails(self):
        root = self.paper()
        code, out = self.run_command(root)
        self.assertEqual(code, 0, "what is open is reported and fails only with --final")
        self.assertTrue(out.rstrip().endswith("PASS"))
        code, out = self.run_command(root, "--final")
        self.assertEqual(code, 1)
        self.assertIn("FAIL  (--final) 1 placeholders that the paper uses left", out)
        self.assertNotIn("3 placeholders", out.replace("unused      3 placeholders", ""))

    def test_with_final_placeholders_that_nothing_uses_do_not_fail(self):
        # The same macros, in a paper whose text uses none of the placeholders and has no IF block left.
        root = self.paper(results="The trials number \\nFilled.\n% The radius was \\radiusInComment~mm in a draft.\n")
        report = check_paper.check(root, [], final=True)
        self.assertEqual((report.failures, report.open_items), ([], []))
        summary = "\n".join(report.summary)
        self.assertIn("unused      4 placeholders that the paper does not use", summary)
        self.assertRegex(summary, r"(?m)^open +nothing; protocol frozen$")
        code, out = self.run_command(root, "--final")
        self.assertEqual(code, 0)
        self.assertTrue(out.rstrip().endswith("PASS"))
        self.assertIn("unused      4 placeholders that the paper does not use", out, "they are still reported")

    def test_a_placeholder_in_the_kept_branch_of_an_if_block_is_used(self):
        said = "The kept branch counts \\nFilled trials again."
        self.assertIn(said, RESULTS)
        kept = RESULTS.replace(said, "The kept branch says \\radiusNowhere~mm.")
        report = check_paper.check(self.paper(results=kept), [], final=True)
        self.assertIn("2 placeholders that the paper uses left", report.open_items)
        self.assertIn("unused      2 placeholders that the paper does not use", "\n".join(report.summary))

    def test_a_paper_without_a_placeholder_has_no_line_about_unused_ones(self):
        filled = dict.fromkeys(MACROS, "24")
        report = check_paper.check(self.paper(macros=filled), [], final=False)
        summary = "\n".join(report.summary)
        self.assertIn("(0 of the 5 defined are placeholders that the paper uses)", summary)
        self.assertNotIn("unused", summary)
        self.assertRegex(summary, r"(?m)^open +1 IF blocks; protocol frozen$")

    def test_a_macro_that_is_used_and_not_defined_still_fails(self):
        without = {name: body for name, body in MACROS.items() if name != "radiusUsed"}
        report = check_paper.check(self.paper(macros=without), [], final=False)
        self.assertIn("number macro used but not defined: \\radiusUsed", report.failures)

    def test_the_split_is_by_the_places_a_macro_stands_in_the_text_as_kept(self):
        defined = {**dict.fromkeys(("radiusUsed", "radiusInBranch", "radiusNowhere"), PLACEHOLDER), "nFilled": "3"}
        used = Counter({"radiusUsed": 2, "radiusInBranch": 0, "nFilled": 1})
        self.assertEqual(
            check_paper.placeholders_by_use(defined, used), (["radiusUsed"], ["radiusInBranch", "radiusNowhere"])
        )
        self.assertEqual(check_paper.placeholders_by_use({}, Counter()), ([], []))

    def test_the_checker_s_own_cases_pass(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check_paper.self_test(), 0, out.getvalue())


if __name__ == "__main__":
    unittest.main()
