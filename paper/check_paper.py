#!/usr/bin/env python3
"""Checks on the paper's sources that need no TeX engine.

    python3 paper/check_paper.py              exit 0 = pass, 1 = fail
    python3 paper/check_paper.py -v           the same, with every finding listed
    python3 paper/check_paper.py --final      before submitting: what is merely reported becomes a failure
    python3 paper/check_paper.py --branches   list the IF blocks that still hold two versions of a sentence
    python3 paper/check_paper.py --numbers    list typed numerals that carry no "% src:" on their line
    python3 paper/check_paper.py --abstract   print the abstract as plain text, for the submission form
    python3 paper/check_paper.py --self-test  check this script's own helpers

It FAILS when
  1. a \\cite key is not in the bibliography;
  2. a number macro used in main.tex or in a section is not defined in generated/ or in numbers_by_hand.tex;
  3. a forbidden identifying string appears (the lists: forbidden.txt, forbidden.local.txt if present, --forbidden);
  4. a section is more than 20 % over its word target, counting the longest branch of every IF block;
and, because the skeleton was written where no TeX engine was installed, when the sources would not build or would
print something wrong: an environment or a brace left open, a \\ref without its \\label, a missing \\input file, a
character outside ASCII, a < > or | in running text, a number macro followed by \\, or ], a macro defined twice, or
generated/protocol.tex out of step with config/protocol.toml.

It REPORTS what is still open, and fails on it only with --final: placeholders that the paper uses, IF blocks,
CONFIRM and TODO marks, \\pending tags, a protocol that is not frozen.

A placeholder is open only if the paper uses its macro: in main.tex or in a section as the text is kept now, outside
comments and outside the commented-out branches of IF blocks. Only there can it print a bold ??. A placeholder that
nothing uses (a number of a block of trials that was planned and not run) is counted in one summary line and never
fails.

Conventions it relies on (README.md explains them):
  - a number macro is letters only and has a capital letter in it (\\radiusClustered); LaTeX's own commands with a
    capital are listed in LATEX_WITH_CAPITALS below;
  - an IF block is
        % IF <condition>
        the sentence kept for now
        % ELSE <condition>            (or % ELIF <condition>, any number of times)
        %| the other version, each line behind "%| "
        % ENDIF
  - a typed numeral is allowed where the quantity is already known, with "% src: where from" on the same line.

Standard library only. Python 3.11 or newer.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PAPER = Path(__file__).resolve().parent

# The page budget: words of running text per section (README.md, "Page budget"). Floats are counted apart.
# Rebalanced on 2026-10-08, after the blind reads of the built paper: the counts by distance and what the policy's
# fit does and does not establish moved from the appendix into the results, and the related work gave up words for it.
# The limit that binds is the page limit, which the build checks (MAIN TEXT ENDS ON PAGE); these keep the sections in
# proportion.
TARGET_WORDS = {
    "abstract": 185,
    "intro": 360,
    "related": 280,
    "method": 650,
    "results": 740,
    "checklist": 280,
    "limits": 150,
}
ALLOWED_OVER = 0.20

PLACEHOLDER = r"\textbf{??}"
ALT_PREFIX = "%|"
SHIPPED_LIST = "forbidden.txt"
LOCAL_LIST = "forbidden.local.txt"

#: LaTeX's and IEEEtran's own commands that have a capital letter (everything else with a capital is a number macro).
LATEX_WITH_CAPITALS = {
    "IfFileExists", "LaTeX", "TeX", "Large", "LARGE", "Huge", "Big", "Bigl", "Bigr", "Bigg", "Delta", "Gamma",
    "Lambda", "Omega", "Phi", "Pi", "Psi", "Sigma", "Theta", "Pr", "MakeUppercase", "AtBeginDocument",
    "ifPDFTeX",  # iftex's test of the engine: main.tex loads its fonts by it (2026-10-09)
}  # fmt: skip

#: Lower-case commands the sources may use. One that is not here is reported (not failed): a typing slip, or a new
#: command to add to this list.
KNOWN_COMMANDS = set(
    """
    documentclass usepackage hypersetup newif fi else newcommand renewcommand textbf textit emph texttt textsc text
    setmainfont setsansfont setmonofont write gamma geq
    includegraphics columnwidth textwidth linewidth fbox parbox centering footnotesize small scriptsize normalsize
    input title author begin end maketitle typeout thepage space bibliographystyle bibliography appendices appendix
    section subsection subsubsection paragraph label cite ref eqref caption toprule midrule bottomrule cmidrule
    multicolumn item frac exp log bigl bigr mid lambda in times xspace noindent par clearpage newpage vspace hspace
    le ge approx sqrt pm cdot ldots dots footnote url raggedright arraybackslash tabcolsep arraystretch setlength
    quad qquad left right mathrm pending figureorbox ifdraftmarks draftmarkstrue draftmarksfalse hfill medskip
    smallskip bigskip mbox addlinespace protect relax
    """.split()
)

MATH_ENVIRONMENTS = r"equation\*?|align\*?|gather\*?|multline\*?|displaymath"
DISPLAY_MATH = re.compile(rf"\\begin\{{({MATH_ENVIRONMENTS})\}}.*?\\end\{{\1\}}|\\\[.*?\\\]", re.S)
INLINE_MATH = re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.S)
FLOAT = re.compile(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", re.S)
CONTROL_WORD = re.compile(r"\\([A-Za-z]+)")
CITE = re.compile(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^{}]*)\}")
DEFINITION = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\\xspace\}\s*(?:%.*)?$")
NAME = re.compile(r"^[a-z][A-Za-z]*[A-Z][A-Za-z]*$")

# What a line may contain besides words without counting as "a number typed by hand".
NUMERAL_MASKS = [
    re.compile(p)
    for p in (
        r"\\(?:ref|eqref|label|cite|input|includegraphics|pending)\*?(?:\[[^\]]*\])*\{[^{}]*\}",
        r"\\figureorbox\{[^{}]*\}\{[^{}]*\}\{[^{}]*\}",
        r"\\begin\{tabular\*?\}\{(?:[^{}]|\{[^{}]*\})*\}",
        r"\\(?:multicolumn|cmidrule)(?:\([^)]*\))?\{[^{}]*\}",
        r"\d+(?:\.\d+)?\s*(?:cm|mm|pt|em|ex|in)(?=\s*[}\]])",
        r"\d+(?:\.\d+)?\\(?:columnwidth|textwidth|linewidth)",
        r"r_\{50\}",
        r"d\s*=\s*0(?![\d.])",
        r"SO-10[01]|STS3215|v0\.1",
        r"\\[A-Za-z]+",
    )
]


# ----------------------------------------------------------------------------------------------------------------------
# Reading TeX
# ----------------------------------------------------------------------------------------------------------------------
def split_comment(line: str) -> tuple[str, str]:
    """One line as (code, comment). The comment keeps its % sign; an escaped \\% is not a comment."""
    at = line.find("%")
    while at >= 0:
        backslashes = len(line[:at]) - len(line[:at].rstrip("\\"))
        if backslashes % 2 == 0:
            return line[:at], line[at:]
        at = line.find("%", at + 1)
    return line, ""


def strip_comments(text: str) -> str:
    """The text with every comment removed and its lines kept."""
    return "\n".join(split_comment(line)[0] for line in text.split("\n"))


def is_number_macro(name: str) -> bool:
    """Whether a control word is one of the paper's number macros: it has a capital and is not LaTeX's own."""
    return name != name.lower() and name not in LATEX_WITH_CAPITALS and not name.startswith("IEEE")


def without_math(code: str) -> str:
    """Comment-free TeX with displayed and inline math blanked out."""
    return INLINE_MATH.sub(" ", DISPLAY_MATH.sub(" ", code))


def plain(code: str) -> str:
    """Comment-free TeX as the words a reader sees. A citation, a reference, a formula in the line and a number macro
    each stand for one word; displayed formulas, labels and \\pending tags for none."""
    t = DISPLAY_MATH.sub(" ", code)
    t = INLINE_MATH.sub(" MATH ", t)
    t = CITE.sub(" CITE ", t)
    t = re.sub(r"\\(?:ref|eqref|pageref)\{[^{}]*\}", " REF ", t)
    t = re.sub(
        r"\\(?:label|input|includegraphics|bibliography|bibliographystyle|pending|vspace|hspace|typeout)"
        r"\*?(?:\[[^\]]*\])*\{[^{}]*\}",
        " ",
        t,
    )
    t = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\[[^\]]*\])?", " ", t)
    for escaped, shown in (("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"), ("\\$", "$")):
        t = t.replace(escaped, shown)
    t = t.replace("~", " ").replace("\\,", "").replace("\\\\", " ").replace("\\ ", " ")
    t = CONTROL_WORD.sub(lambda m: " NUM " if is_number_macro(m.group(1)) else " ", t)
    t = re.sub(r"\\.", " ", t)
    return t.replace("{", "").replace("}", "")


def count_words(code: str) -> int:
    """Words in comment-free TeX: every token of `plain` that has a letter or a digit in it."""
    return sum(1 for token in plain(code).split() if re.search(r"[A-Za-z0-9]", token))


def captions(code: str) -> list[str]:
    """The text of every \\caption{...} in comment-free TeX."""
    found = []
    for match in re.finditer(r"\\caption\{", code):
        depth, at = 1, match.end()
        while at < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[at], 0) if code[at - 1] != "\\" else 0
            at += 1
        found.append(code[match.end() : at - 1])
    return found


# ----------------------------------------------------------------------------------------------------------------------
# IF blocks
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Block:
    """One IF block: where it starts, each branch's condition and each branch's lines (the first is the kept one)."""

    line: int
    conditions: list[str] = field(default_factory=list)
    branches: list[list[str]] = field(default_factory=list)


def parse_blocks(text: str) -> tuple[list[str | Block], list[str]]:
    """A file as plain lines and IF blocks, in order, and what is wrong with its IF marks."""
    parts: list[str | Block] = []
    problems: list[str] = []
    block: Block | None = None
    for number, line in enumerate(text.split("\n"), 1):
        mark = line.strip()
        if re.match(r"% IF\b", mark):
            if block is not None:
                problems.append(f"line {number}: an IF inside the IF of line {block.line}")
                continue
            block = Block(number, [mark[4:].strip()], [[]])
        elif re.match(r"% (ELIF|ELSE)\b", mark):
            if block is None:
                problems.append(f"line {number}: {mark.split()[1]} without an IF")
                continue
            block.conditions.append(mark[2:].strip())
            block.branches.append([])
        elif re.match(r"% ENDIF\b", mark):
            if block is None:
                problems.append(f"line {number}: ENDIF without an IF")
                continue
            if len(block.branches) < 2:
                problems.append(f"line {block.line}: an IF with no ELSE")
            parts.append(block)
            block = None
        elif block is None:
            if line.startswith(ALT_PREFIX):
                problems.append(f"line {number}: a '{ALT_PREFIX}' line outside an IF block")
            parts.append(line)
        elif len(block.branches) == 1:
            if line.startswith(ALT_PREFIX):
                problems.append(f"line {number}: the first branch of an IF is the kept one, without '{ALT_PREFIX}'")
            block.branches[0].append(line)
        elif line.startswith(ALT_PREFIX):
            block.branches[-1].append(line[len(ALT_PREFIX) :].removeprefix(" "))
        elif mark and not mark.startswith("%"):
            problems.append(f"line {number}: inside an ELSE every line of text starts with '{ALT_PREFIX} '")
    if block is not None:
        problems.append(f"line {block.line}: an IF with no ENDIF")
    return parts, problems


def render(parts: list[str | Block], choose) -> str:
    """The file's text with one branch of every IF block, chosen by `choose(block) -> index`."""
    lines: list[str] = []
    for part in parts:
        lines += [part] if isinstance(part, str) else part.branches[choose(part)]
    return "\n".join(lines)


def variants(parts: list[str | Block]) -> list[str]:
    """The file as kept now, then with each IF block's second branch, third branch, and so on."""
    most = max((len(p.branches) for p in parts if isinstance(p, Block)), default=1)
    return [render(parts, lambda b, k=k: min(k, len(b.branches) - 1)) for k in range(most)]


def running_text(text: str) -> str:
    """Comment-free TeX without its figures and tables."""
    return FLOAT.sub(" ", strip_comments(text))


def section_words(parts: list[str | Block]) -> tuple[int, int, int]:
    """Words of running text as kept now, with the longest branch of every IF block, and words in captions."""
    now = render(parts, lambda block: 0)

    def longest_branch(block: Block) -> int:
        return max(range(len(block.branches)), key=lambda i: count_words(running_text("\n".join(block.branches[i]))))

    longest = render(parts, longest_branch)
    in_captions = sum(count_words(c) for c in captions(strip_comments(now)))
    return count_words(running_text(now)), count_words(running_text(longest)), in_captions


# ----------------------------------------------------------------------------------------------------------------------
# Structure: what a TeX engine would have told us
# ----------------------------------------------------------------------------------------------------------------------
def structure_problems(code: str) -> list[str]:
    """What would stop comment-free TeX from building, or make it print the wrong character."""
    problems: list[str] = []
    open_environments: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^{}]*)\}", code):
        kind, name = match.groups()
        if kind == "begin":
            open_environments.append(name)
        elif not open_environments or open_environments.pop() != name:
            problems.append(f"\\end{{{name}}} does not close the environment that is open")
    problems += [f"\\begin{{{name}}} is never closed" for name in open_environments]

    depth = backslashes = 0
    for char in code:
        if char in "{}" and backslashes % 2 == 0:
            depth += 1 if char == "{" else -1
            if depth < 0:
                problems.append("a } with no { before it")
                break
        backslashes = backslashes + 1 if char == "\\" else 0
    if depth > 0:
        problems.append(f"{depth} brace(s) left open")
    if len(re.findall(r"(?<!\\)\$", code)) % 2:
        problems.append("an odd number of $ signs")
    outside_ascii = sorted({char for char in code if ord(char) > 126})
    if outside_ascii:
        problems.append("characters outside ASCII: " + " ".join(f"U+{ord(c):04X}" for c in outside_ascii))
    wrong_glyph = sorted(set(re.findall(r"[<>|]", without_math(code))))
    if wrong_glyph:
        problems.append("in running text these print as other characters, use math or words: " + " ".join(wrong_glyph))
    for match in re.finditer(r"\\([A-Za-z]+)(\\,|\])", code):
        if is_number_macro(match.group(1)):
            problems.append(f"\\{match.group(1)} is followed by {match.group(2)}: write {{}} after the macro first")
    for match in re.finditer(r"\\figureorbox\{([^{}]*)\}", code):
        if re.search(r"[_ #%&]", match.group(1)):
            problems.append(f"figure file name {match.group(1)!r}: letters, digits, hyphens, dots and slashes only")
    return problems


def placeholders_by_use(defined: dict[str, str], used: Counter[str]) -> tuple[list[str], list[str]]:
    """The macros whose body is still the placeholder, as (those the paper uses, those nothing uses), in name order.

    `used` counts, for each number macro, the places where it stands in the text as it is kept now. A macro that
    stands only in a comment, or only in a commented-out branch of an IF block, counts no place: the paper does not
    print it. Only a placeholder that is used is open, and only it fails with --final.
    """
    waiting = sorted(name for name, body in defined.items() if body.strip() == PLACEHOLDER)
    return [name for name in waiting if used[name] > 0], [name for name in waiting if used[name] == 0]


def unsourced_numerals(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every line that has a typed numeral and no '% src:' comment."""
    found = []
    text = DISPLAY_MATH.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)  # a formula is not a quantity
    for number, raw in enumerate(text.split("\n"), 1):
        line = raw[len(ALT_PREFIX) :] if raw.startswith(ALT_PREFIX) else raw
        code, comment = split_comment(line)
        if "src:" in comment:
            continue
        for mask in NUMERAL_MASKS:
            code = mask.sub(" ", code)
        if re.search(r"\d", code):
            found.append((number, raw.strip()))
    return found


# ----------------------------------------------------------------------------------------------------------------------
# The forbidden lists
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Entry:
    """One line of a forbidden list."""

    shown: str
    pattern: re.Pattern[str]
    pdf_only: bool


def read_forbidden(path: Path) -> list[Entry]:
    """The entries of one list. `pdf:` limits an entry to what goes into the PDF; `re:` makes it a regular expression.
    Entries of any list but the shipped one are never printed, only numbered: they are the identifying strings."""
    entries = []
    private = path.name != SHIPPED_LIST
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pdf_only = line.startswith("pdf:")
        line = line.removeprefix("pdf:")
        expression = line.removeprefix("re:") if line.startswith("re:") else re.escape(line)
        shown = f"line {number} of {path.name}" if private else f"{line!r} ({path.name})"
        entries.append(Entry(shown, re.compile(expression, re.I), pdf_only))
    return entries


def read_text(path: Path) -> str:
    """A file's text; a file that is not UTF-8 (a picture, a PDF) is read byte for byte."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def forbidden_hits(root: Path, entries: list[Entry]) -> tuple[list[str], int]:
    """Every place a forbidden entry appears under `root`, and how many files were read."""
    hits: list[str] = []
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not p.name.startswith("forbidden")
        and p.name != ".DS_Store"
        and not {"__pycache__", "build"} & set(p.relative_to(root).parts)
    ]
    for path in files:
        relative = path.relative_to(root)
        in_pdf = path.suffix in {".tex", ".bib"} or (relative.parts[0] == "figures" and path.name != "README.md")
        text = read_text(path)
        code = strip_comments(text) if path.suffix == ".tex" else text
        for entry in entries:
            if entry.pdf_only and not in_pdf:
                continue
            searched = code if entry.pdf_only else text
            for number, line in enumerate(searched.split("\n"), 1):
                if entry.pattern.search(line):
                    hits.append(f"{relative}:{number}: {entry.shown}")
    return hits, len(files)


# ----------------------------------------------------------------------------------------------------------------------
# The whole check
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Report:
    """What the check found."""

    failures: list[str] = field(default_factory=list)
    open_items: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    numerals: list[str] = field(default_factory=list)


def definitions(path: Path) -> tuple[dict[str, str], list[str]]:
    """`{name: body}` for a macro file, and its lines that start a definition this script cannot read."""
    found: dict[str, str] = {}
    unreadable = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("\\newcommand"):
            continue
        match = DEFINITION.match(line)
        if match is None:
            form = "\\newcommand{\\name}{value\\xspace} % comment"
            unreadable.append(f"{path.name}:{number}: a definition must be one line of the form {form}")
        elif match.group(1) in found:
            unreadable.append(f"{path.name}:{number}: \\{match.group(1)} is defined twice in this file")
        else:
            found[match.group(1)] = match.group(2)
    return found, unreadable


def load_stub(root: Path):
    """The module make_numbers_stub.py, or None if it cannot be loaded."""
    path = root / "make_numbers_stub.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("make_numbers_stub", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def check(root: Path, extra_lists: list[Path], final: bool) -> Report:  # noqa: C901 (one pass, read top to bottom)
    report = Report()
    fail, note = report.failures.append, report.notes.append
    structural: list[str] = []  # what a TeX engine would have caught; merged into the failures at the end

    main_path = root / "main.tex"
    if not main_path.exists():
        fail("main.tex is missing")
        return report
    main_code = strip_comments(main_path.read_text(encoding="utf-8"))

    # --- the files that make the paper ------------------------------------------------------------------------------
    sources: dict[str, str] = {"main.tex": main_path.read_text(encoding="utf-8")}
    macro_files: list[Path] = []
    for target in re.findall(r"\\input\{([^{}]*)\}", main_code):
        path = root / (target if target.endswith(".tex") else target + ".tex")
        if not path.exists():
            fail(f"main.tex: \\input{{{target}}} names a file that does not exist")
        elif target.startswith("sections/"):
            sources[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        else:
            macro_files.append(path)
    for name in TARGET_WORDS:
        if f"sections/{name}.tex" not in sources:
            fail(f"sections/{name}.tex is not read by main.tex")

    # --- macro definitions ------------------------------------------------------------------------------------------
    defined: dict[str, str] = {}
    for path in macro_files:
        found, unreadable = definitions(path)
        report.failures += unreadable
        for name, body in found.items():
            if name in defined:
                fail(f"\\{name} is defined in more than one macro file")
            if not NAME.match(name):
                fail(f"{path.name}: \\{name} is not a number-macro name (letters only, lower-case start, one capital)")
            defined[name] = body
    own_commands = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", main_code))
    for switch in re.findall(r"\\newif\\if([A-Za-z]+)", main_code):
        own_commands |= {f"if{switch}", f"{switch}true", f"{switch}false"}

    # --- every source, in every version its IF blocks allow -----------------------------------------------------------
    used: Counter[str] = Counter()
    cited: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    referenced: set[str] = set()
    unknown_commands: set[str] = set()
    n_blocks = n_confirm = n_todo = n_pending = n_literal = 0
    words: dict[str, tuple[int, int, int]] = {}
    for name, text in sources.items():
        parts, problems = parse_blocks(text)
        structural += [f"{name}: {p}" for p in problems]
        blocks = [p for p in parts if isinstance(p, Block)]
        n_blocks += len(blocks)
        for block in blocks:
            report.branches.append(f"{name}:{block.line}: IF " + "  |  ".join(block.conditions))
        n_confirm += len(re.findall(r"%\s*CONFIRM:", text))
        n_todo += len(re.findall(r"%\s*TODO:", text))
        seen: set[str] = set()  # a problem is reported once, for the first version of the file that shows it
        for index, version in enumerate(variants(parts)):
            code = strip_comments(version)
            which = name if index == 0 else f"{name} (with branch {index + 1} of its IF blocks)"
            fresh = [p for p in structure_problems(code) if p not in seen]
            seen.update(fresh)
            structural += [f"{which}: {p}" for p in fresh]
            for keys in CITE.findall(code):
                cited.update(key.strip() for key in keys.split(",") if key.strip())
            referenced |= set(re.findall(r"\\(?:ref|eqref|pageref)\{([^{}]*)\}", code))
            for word in CONTROL_WORD.findall(code):
                if is_number_macro(word):
                    used[word] += index == 0
                elif word.islower() and word not in KNOWN_COMMANDS and word not in own_commands:
                    unknown_commands.add(word)
            for macro, body in defined.items():
                outside_math = re.search(rf"\\{macro}(?![A-Za-z])", without_math(code))
                if re.search(r"[<>]", body) and outside_math and macro not in seen:
                    seen.add(macro)
                    structural.append(f"{which}: \\{macro} holds {body!r}: use it inside $...$ (< and > need math)")
            if index == 0:
                labels.update(re.findall(r"\\label\{([^{}]*)\}", code))
                n_pending += len(re.findall(r"\\pending\{", code)) if name != "main.tex" else 0
                n_literal += code.count("??")
        section = Path(name).stem
        if name.startswith("sections/"):
            words[section] = section_words(parts)
            report.numerals += [f"{name}:{n}: {line}" for n, line in unsourced_numerals(text)]

    # --- 1. citations -------------------------------------------------------------------------------------------------
    bib_keys: Counter[str] = Counter()
    for bib in re.findall(r"\\bibliography\{([^{}]*)\}", main_code):
        for stem in bib.split(","):
            path = root / (stem.strip() + ".bib")
            if not path.exists():
                fail(f"main.tex: bibliography file {path.name} does not exist")
                continue
            text = path.read_text(encoding="utf-8")
            if text.count("{") != text.count("}"):
                fail(f"{path.name}: braces do not balance ({text.count('{')} open, {text.count('}')} close)")
            for kind, key in re.findall(r"^\s*@(\w+)\s*\{\s*([^,\s]+)\s*,", text, re.M):
                if kind.lower() not in {"comment", "string", "preamble"}:
                    bib_keys[key] += 1
    missing = sorted(set(cited) - set(bib_keys))
    report.failures += [f"\\cite key not in the bibliography: {key}" for key in missing]
    report.failures += [f"bibliography key defined twice: {key}" for key, n in bib_keys.items() if n > 1]
    in_bib = len(bib_keys)
    report.summary.append(f"cite keys   {len(cited)} used, {len(missing)} missing ({in_bib} in the bibliography)")

    # --- 2. number macros ---------------------------------------------------------------------------------------------
    undefined = sorted(set(used) - set(defined))
    report.failures += [f"number macro used but not defined: \\{name}" for name in undefined]
    unused = sorted(set(defined) - set(used))
    placeholders, idle_placeholders = placeholders_by_use(defined, used)
    report.summary.append(
        f"macros      {len(set(used))} used, {len(undefined)} undefined "
        f"({len(placeholders)} of the {len(defined)} defined are placeholders that the paper uses)"
    )
    if idle_placeholders:
        report.summary.append(
            f"unused      {len(idle_placeholders)} placeholders that the paper does not use: they are not open, "
            "and --final does not fail on them"
        )
        note("placeholders that the paper does not use: " + ", ".join(idle_placeholders))
    if placeholders:
        note("placeholders that the paper uses: " + ", ".join(placeholders))
    if unused:
        note("defined and not used: " + ", ".join(unused))
    if unknown_commands:
        note("commands not in KNOWN_COMMANDS (a slip, or add them): " + ", ".join(sorted(unknown_commands)))

    stub = load_stub(root)
    protocol_status = "unknown"
    protocol_tex = root / "generated" / "protocol.tex"
    if protocol_tex.exists():
        status = re.search(r"^% protocol status: (\S+)", protocol_tex.read_text(encoding="utf-8"), re.M)
        protocol_status = status.group(1) if status else "unknown"
    if stub is None:
        note("make_numbers_stub.py not found: generated/ was not compared with its lists")
    else:
        numbers_tex = root / "generated" / "numbers.tex"
        on_disk = set(definitions(numbers_tex)[0]) if numbers_tex.exists() else set()
        listed = set(stub.result_names())
        if on_disk != listed:
            differ = sorted(on_disk ^ listed)
            fail("generated/numbers.tex differs from the list in make_numbers_stub.py: " + ", ".join(differ[:6]))
        if stub.PROTOCOL_TOML.exists() and protocol_tex.exists():
            if stub.render_protocol(stub.load_protocol()) != protocol_tex.read_text(encoding="utf-8"):
                fail("generated/protocol.tex is out of step with config/protocol.toml: run make_numbers_stub.py again")
        else:
            note("config/protocol.toml not found here: generated/protocol.tex was not compared with it")

    # --- 3. forbidden strings -----------------------------------------------------------------------------------------
    lists = [root / SHIPPED_LIST] + ([root / LOCAL_LIST] if (root / LOCAL_LIST).exists() else []) + extra_lists
    entries: list[Entry] = []
    for path in lists:
        if path.exists():
            entries += read_forbidden(path)
        else:
            fail(f"forbidden list not found: {path.name}")
    hits, n_files = forbidden_hits(root, entries)
    report.failures += [f"forbidden string: {hit}" for hit in hits]
    scanned = f"{len(entries)} entries, {len(lists)} list(s), {n_files} files"
    report.summary.append(f"forbidden   {len(hits)} hits ({scanned})")

    # --- 4. words -----------------------------------------------------------------------------------------------------
    cells = []
    for section, target in TARGET_WORDS.items():
        if section not in words:
            continue
        now, longest, in_captions = words[section]
        limit = round(target * (1 + ALLOWED_OVER))
        counted = max(now, longest)  # the limit applies whichever branch of an IF block is kept
        cells.append(f"{section} {counted}/{target}")
        if counted > limit:
            fail(f"sections/{section}.tex: {counted} words, over the limit of {limit} ({target} + 20 %)")
        if longest != now:
            note(f"sections/{section}.tex: {now} words as kept now, {longest} with each IF block's longest branch")
        if in_captions:
            note(f"sections/{section}.tex: {in_captions} more words in captions")
    total = sum(max(words[s][:2]) for s in TARGET_WORDS if s in words)
    report.summary.append("words       " + ", ".join(cells) + f"; total {total}/{sum(TARGET_WORDS.values())}")
    if "appendix" in words:
        note(f"sections/appendix.tex: {words['appendix'][0]} words (no limit)")

    # --- the submission form's drafts ---------------------------------------------------------------------------------
    over, n_fields = submission_problems(root)
    report.failures += over
    if n_fields:
        state = f"{n_fields} fields of submission.md, {len(over)} over a limit or out of step"
        report.summary.append(f"form        {state}")

    # --- structure across files ---------------------------------------------------------------------------------------
    structural += [f"\\label defined twice: {label}" for label, n in labels.items() if n > 1]
    structural += [f"\\ref to a label that does not exist: {label}" for label in sorted(referenced - set(labels))]
    report.failures += structural
    report.summary.append(f"structure   {len(structural)} problems ({len(sources)} source files, {len(labels)} labels)")

    # --- what is still open -------------------------------------------------------------------------------------------
    open_counts = {
        "placeholders that the paper uses": len(placeholders),
        "IF blocks": n_blocks,
        "CONFIRM marks": n_confirm,
        "TODO marks": n_todo,
        "\\pending tags": n_pending,
        "?? typed in a section": n_literal,
        "typed numerals without src": len(report.numerals),
    }
    still_open = ", ".join(f"{n} {what}" for what, n in open_counts.items() if n) or "nothing"
    report.summary.append(f"open        {still_open}; protocol {protocol_status}")
    if final:
        report.open_items += [f"{n} {what} left" for what, n in open_counts.items() if n]
        if protocol_status != "frozen":
            report.open_items.append(f"the protocol is '{protocol_status}', not frozen")
        if re.search(r"^\\draftmarkstrue", main_code, re.M):
            report.open_items.append("main.tex still has \\draftmarkstrue")
    return report


def submission_problems(root: Path) -> tuple[list[str], int]:
    """Fields of submission.md that are over their limit, and how many fields were checked.

    A field is a line `<!-- field: NAME max=N -->` followed by a fenced block; N counts characters.
    """
    path = root / "submission.md"
    if not path.exists():
        return [], 0
    marked = r"<!-- field: (\S+) max=(\d+) -->\s*```[a-z]*\n(.*?)\n```"
    fields = re.findall(marked, path.read_text(encoding="utf-8"), re.S)
    over = []
    for name, limit, body in fields:
        length = len(" ".join(body.split()))
        if length >= int(limit):
            over.append(f"submission.md: {name} has {length} characters; it must stay under {limit}")
        if name == "abstract" and " ".join(body.split()) != abstract_text(root):
            over.append("submission.md: its abstract differs from sections/abstract.tex (paste --abstract's output)")
    return over, len(fields)


def abstract_text(root: Path) -> str:
    """The abstract as plain text for the submission form: constants filled in, results without a value as [??]."""
    bodies: dict[str, str] = {}
    for path in [root / "generated" / "protocol.tex", root / "generated" / "numbers.tex", root / "numbers_by_hand.tex"]:
        if path.exists():
            bodies.update(definitions(path)[0])
    parts, _ = parse_blocks((root / "sections" / "abstract.tex").read_text(encoding="utf-8"))
    lines: list[str] = []
    for part in parts:  # an IF block is shown whole, so that no version reads as the result
        if isinstance(part, str):
            lines.append(part)
        else:
            lines.append("[ONE OF: " + " | ".join(" ".join(branch) for branch in part.branches) + "]")
    code = strip_comments("\n".join(lines))
    code = re.sub(r"\\pending\{[^{}]*\}", "", code)
    code = CONTROL_WORD.sub(
        lambda m: ("[??]" if bodies.get(m.group(1), PLACEHOLDER).strip() == PLACEHOLDER else bodies[m.group(1)])
        if is_number_macro(m.group(1))
        else "",
        code,
    )
    code = code.replace("\\%", "%").replace("~", " ").replace("\\,", "").replace("--", "-").replace("$", "")
    code = code.replace("{}", "").replace("{", "").replace("}", "").replace("``", '"').replace("''", '"')
    return " ".join(code.split())


def self_test() -> int:
    """Check the helpers against cases worked out by hand."""
    cases = [
        (count_words(r"The radius is \radiusClustered~mm (95~\% interval)."), 7),
        (count_words(r"Known \cite{a,b}; see Fig.~\ref{fig:x} and $d=0$."), 7),
        (count_words(r"\section{Related Work} \emph{tolerance radius}: one-up-one-down"), 5),
        (count_words(r"\begin{equation} P = \frac{1}{2} \end{equation} after \pending{IF x}the text"), 3),
        (count_words(strip_comments("a 50\\% share % src: nowhere\nb")), 4),
        (split_comment(r"50\% of it % a comment"), (r"50\% of it ", "% a comment")),
        (split_comment(r"a line break \\% then a comment"), ("a line break \\\\", "% then a comment")),
        (is_number_macro("nDemos"), True),
        (is_number_macro("IfFileExists") or is_number_macro("IEEEauthorblockN") or is_number_macro("textbf"), False),
        (structure_problems(r"\begin{a} { \end{a}"), ["1 brace(s) left open"]),
        (structure_problems(r"\begin{a} \end{b}"), ["\\end{b} does not close the environment that is open"]),
        (structure_problems(r"$a < b$ and \nDemos{}\,x"), []),
        (len(structure_problems(r"a < b and \nDemos\,x")), 2),
        ([n for n, _ in unsourced_numerals("5.6~mm % src: f\n5.6~mm\n$r_{50}$, $d=0$, SO-101\n\\vspace{2pt}")], [2]),
    ]
    text = "before\n% IF a\nkept one two\n% ELSE b\n%| other one two three\n% ENDIF\nafter"
    parts, problems = parse_blocks(text)
    cases += [
        (problems, []),
        (variants(parts), ["before\nkept one two\nafter", "before\nother one two three\nafter"]),
        (section_words(parts), (5, 6, 0)),
        (parse_blocks("% IF a\nx\n% ENDIF")[1], ["line 1: an IF with no ELSE"]),
        (len(parse_blocks("% IF a\nx\n% ELSE\ny\n% ENDIF")[1]), 1),
        (captions(r"\caption{A {nested} one} x \caption{two}"), ["A {nested} one", "two"]),
    ]
    # A placeholder is open where the paper prints it, and only there. `inBranch` stands in a commented-out branch
    # alone: the checker has met it (a count of 0) and the paper as kept does not print it.
    defined = {"radiusUsed": PLACEHOLDER, "radiusIdle": PLACEHOLDER, "inBranch": PLACEHOLDER, "nFilled": "24"}
    cases += [
        (placeholders_by_use(defined, Counter({"radiusUsed": 2, "inBranch": 0, "nFilled": 1})),
         (["radiusUsed"], ["inBranch", "radiusIdle"])),
        (placeholders_by_use(defined, Counter()), ([], ["inBranch", "radiusIdle", "radiusUsed"])),
        (placeholders_by_use({"nFilled": "24"}, Counter({"nFilled": 3})), ([], [])),
    ]
    failed = [(index, got, want) for index, (got, want) in enumerate(cases) if got != want]
    for index, got, want in failed:
        print(f"self-test case {index}: got {got!r}, expected {want!r}")
    print(f"self-test: {len(cases) - len(failed)} of {len(cases)} cases pass")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checks on the paper's sources that need no TeX engine.")
    parser.add_argument("--root", type=Path, default=PAPER, help="the paper's folder (default: this script's folder)")
    parser.add_argument("--forbidden", type=Path, action="append", default=[], help="one more forbidden list")
    parser.add_argument("--final", action="store_true", help="fail on anything still open")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every note")
    parser.add_argument("--branches", action="store_true", help="list the IF blocks")
    parser.add_argument("--numbers", action="store_true", help="list typed numerals without a '% src:' comment")
    parser.add_argument("--abstract", action="store_true", help="print the abstract as plain text and stop")
    parser.add_argument("--self-test", action="store_true", help="check this script's helpers and stop")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.abstract:
        print(abstract_text(args.root))
        return 0

    report = check(args.root, args.forbidden, args.final)
    failures = report.failures + [f"(--final) {item}" for item in report.open_items]
    print(f"check_paper{' --final' if args.final else ''}")
    for line in report.summary:
        print("  " + line)
    if args.branches:
        print("\n".join(["IF blocks:"] + ["  " + b for b in report.branches]))
    if args.numbers:
        print("\n".join(["typed numerals without '% src:':"] + ["  " + n for n in report.numerals]))
    if args.verbose:
        print("\n".join(["notes:"] + ["  " + n for n in report.notes]))
    for failure in failures:
        print("  FAIL  " + failure)
    print(f"FAIL: {len(failures)} failure(s)" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
