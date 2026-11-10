"""What `scripts/make_figures.py` does: draw each figure whose input is there, and write it where it belongs.

Inputs
------
* Figure 1: `analysis/pilot_august/trials_with_distance.csv`;
* Figure 2: the two start-state audits, `data/audit/clustered.json` and `data/audit/spread.json`, and the grid of
  `config/protocol.toml`;
* Figures 3 and 4: the analysis results (`analysis/report/results.json` unless another file is named), and for
  Figure 4 the grid.

A figure whose input is not there yet is **skipped**, with one line that names the file it waits for. That is not an
error: the figures are made as the experiment goes. A figure whose input is there and cannot be used (a results file
of another version, an audit with a start off the grid) is **refused**, with the reason. The other figures are made
all the same.

Where a figure is written
-------------------------
The paper's folder, `paper/figures/`, takes the paper's figures and nothing else.

* A results file that says it is a **draft** (made from trials that do not count, or from made-up ones) never reaches
  that folder. Figures 3 and 4 drawn from a draft carry the word DRAFT and go to `analysis/report-draft/figures/`.
  Naming a folder under `paper/` for them is refused.
* Figure 2 can also be drawn from the **planned** start positions, for a look before recording. That picture carries
  the word PLANNED and goes to `analysis/layouts/`.

Every file is saved through `style.save_pdf`, which refuses a file that holds one of the strings the paper must not
hold: the entries of `paper/forbidden.txt`, of the author's own list beside it when it is there, and of any list
named besides.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tolerance_radius import config, layouts
from tolerance_radius.figures import (
    FILE_NAMES,
    earlier_evaluation,
    layouts_on_grid,
    response_curves,
    results,
    style,
    workspace,
)
from tolerance_radius.figures.results import ResultsError
from tolerance_radius.figures.style import FigureError, Forbidden, Page
from tolerance_radius.geometry import GeometryError, Grid
from tolerance_radius.layouts import LayoutError
from tolerance_radius.schema import DATASETS, DemoStart, SchemaError, read_demo_starts

#: The figures of the paper, by number.
NUMBERS = (1, 2, 3, 4)

# Where things are, under the repository's top folder.
EARLIER_TRIALS = Path("analysis/pilot_august/trials_with_distance.csv")
AUDITS = Path("data/audit")
PROTOCOL = Path("config/protocol.toml")
RESULTS = Path("analysis/report/results.json")
PAPER = Path("paper")
PAPER_FIGURES = PAPER / "figures"
DRAFT_FIGURES = Path("analysis/report-draft/figures")
#: The picture of Figure 2 drawn from the plan and not from the audits.
PLANNED_PREVIEW = Path("analysis/layouts/fig2-layouts-planned.pdf")
#: The lists of strings the paper must not hold, in the paper's folder: the one that ships, and the author's own.
SHIPPED_LIST = "forbidden.txt"
LOCAL_LIST = "forbidden.local.txt"

WRITTEN, SKIPPED, REFUSED = "written", "skipped", "refused"
DRAFT, PLANNED = "DRAFT", "PLANNED"


@dataclass(frozen=True, slots=True)
class Made:
    """What became of one figure."""

    number: int
    #: `WRITTEN`; `SKIPPED`, when an input is not there yet; or `REFUSED`, when the figure could not be made as asked.
    status: str
    #: One plain line for the person who ran the command.
    line: str
    #: The file written; None when none was.
    path: Path | None = None
    #: What the drawing found that does not add up, a sentence each.
    notes: tuple[str, ...] = ()


class _Waiting(Exception):
    """An input of a figure is not there yet. The message names it."""


def audit_path(dataset: str, root: str | Path) -> Path:
    """Where the start-state audit of a demonstration set is kept."""
    return Path(root) / AUDITS / f"{dataset}.json"


def read_forbidden(path: str | Path) -> list[Forbidden]:
    """The entries of one list of strings the paper must not hold, as `paper/check_paper.py` reads them.

    One entry to a line; `re:` in front makes it a regular expression, `pdf:` in front of either is dropped (a figure
    is part of the PDF); case is ignored; a line that starts with `#` is a comment. An entry of any list but the one
    that ships is named by its line number alone when it is found, never repeated: those are the authors' own strings.
    """
    path = Path(path)
    private = path.name != SHIPPED_LIST
    entries = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("pdf:")
        expression = line.removeprefix("re:") if line.startswith("re:") else re.escape(line)
        shown = f"line {number} of {path.name}" if private else f"{line!r} ({path.name})"
        entries.append(Forbidden(re.compile(expression, re.IGNORECASE), shown))
    return entries


def forbidden_entries(root: str | Path, extra_lists: Sequence[str | Path] = ()) -> list[Forbidden]:
    """Every string the figures are checked against: the paper's list, the author's own beside it when it is there,
    and the lists named besides. Without the paper's list nothing can be checked, and that is an error.
    """
    folder = Path(root) / PAPER
    shipped = folder / SHIPPED_LIST
    if not shipped.is_file():
        raise FigureError(f"{(PAPER / SHIPPED_LIST).as_posix()} is not there: a figure could not be checked against it")
    lists = [shipped] + ([folder / LOCAL_LIST] if (folder / LOCAL_LIST).is_file() else [])
    lists += [Path(path) for path in extra_lists]
    entries: list[Forbidden] = []
    for path in lists:
        if not path.is_file():
            raise FigureError(f"the list of forbidden strings {path.name} is not there")
        entries += read_forbidden(path)
    return entries


def shown(path: str | Path, root: str | Path) -> str:
    """A path as it is printed: from the repository's top folder when it lies inside it."""
    path, root = Path(path), Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def make_figures(
    numbers: Iterable[int] = NUMBERS,
    *,
    root: str | Path = config.REPO_ROOT,
    results_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    draft_dir: str | Path | None = None,
    planned: bool = False,
    png_dir: str | Path | None = None,
    extra_forbidden: Sequence[str | Path] = (),
) -> list[Made]:
    """Make the figures asked for and say what became of each, in the order of their numbers.

    `root` is the folder that holds `analysis/`, `config/`, `data/` and `paper/`. `results_path` is the analysis
    results (default `analysis/report/results.json`), `out_dir` the folder for the paper's figures (default
    `paper/figures`), `draft_dir` the folder for figures of draft results (default `analysis/report-draft/figures`).
    `planned` adds the picture of Figure 2 from the planned positions when Figure 2 is among those asked for.
    `png_dir`, when given, also gets every figure written as a picture at 300 dots per inch, for looking at.
    """
    root = Path(root)
    wanted = sorted(set(numbers))
    unknown = [number for number in wanted if number not in NUMBERS]
    if unknown:
        raise ValueError(f"the paper has figures {NUMBERS}, not {unknown}")
    job = _Job(
        root=root,
        results_path=Path(results_path) if results_path is not None else root / RESULTS,
        out_dir=Path(out_dir) if out_dir is not None else root / PAPER_FIGURES,
        draft_dir=Path(draft_dir) if draft_dir is not None else root / DRAFT_FIGURES,
        png_dir=Path(png_dir) if png_dir is not None else None,
        extra_forbidden=tuple(Path(path) for path in extra_forbidden),
    )
    made = []
    for number in wanted:
        made.append(job.make(number, f"figure {number}", _DRAWERS[number]))
        if number == 2 and planned:
            made.append(job.make(number, "figure 2 as planned", _planned_figure_2))
    return made


# --------------------------------------------------------------------------------------------------------------------
# One figure
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Drawn:
    """A figure drawn and not yet saved: its page, the file it goes to, and the word it must carry, if any."""

    page: Page
    path: Path
    must_say: str | None = None


@dataclass(slots=True)
class _Job:
    """One run of the command: where things are read from and written to, and what was read already."""

    root: Path
    results_path: Path
    out_dir: Path
    draft_dir: Path
    png_dir: Path | None
    extra_forbidden: tuple[Path, ...]
    _results: dict[str, Any] | None = field(default=None, init=False)

    def shown(self, path: Path) -> str:
        return shown(path, self.root)

    def need(self, path: Path) -> Path:
        """An input file, or the word that the figure waits for it."""
        if not path.is_file():
            raise _Waiting(self.shown(path))
        return path

    def grid(self) -> Grid:
        return Grid.from_protocol(config.load_protocol(self.need(self.root / PROTOCOL)))

    def results(self) -> dict[str, Any]:
        if self._results is None:
            self._results = results.read_results(self.need(self.results_path))
        return self._results

    def results_figure(self, number: int, draw: Callable[[dict[str, Any]], Page]) -> _Drawn:
        """A figure drawn from the analysis results, bound for the paper's folder, or for the drafts' when the results
        are a draft. Where it may go is settled before anything is drawn.
        """
        results_ = self.results()
        if not results.is_draft(results_):
            return _Drawn(draw(results_), self.out_dir / FILE_NAMES[number])
        # Neither this folder's paper nor the paper of the repository this code runs from.
        for paper in {(self.root / PAPER).resolve(), (config.REPO_ROOT / PAPER).resolve()}:
            if self.draft_dir.resolve().is_relative_to(paper):
                raise FigureError(
                    f"{self.shown(self.results_path)} is a draft, and a figure of a draft is never written under "
                    f"{PAPER.as_posix()}/: name another folder for it"
                )
        return _Drawn(draw(results_), self.draft_dir / FILE_NAMES[number], must_say=DRAFT)

    def make(self, number: int, name: str, draw: Callable[[_Job], _Drawn]) -> Made:
        """Draw one figure and save it, or say why not."""
        try:
            drawn = draw(self)
        except _Waiting as waiting:
            return Made(number, SKIPPED, f"{name}: skipped, waits for {waiting}")
        except _CANNOT_USE as error:
            return Made(number, REFUSED, f"{name}: not written: {error}")
        page, path = drawn.page, drawn.path
        try:
            if drawn.must_say is not None and drawn.must_say not in page.stamps:
                raise FigureError(f"{path.name} must carry the word {drawn.must_say} and does not")
            style.save_pdf(page, path, forbidden=forbidden_entries(self.root, self.extra_forbidden))
        except (FigureError, OSError) as error:
            left = " (the file of an earlier run is left as it was)" if path.is_file() else ""
            return Made(number, REFUSED, f"{name}: not written{left}: {error}", notes=tuple(page.notes))
        if self.png_dir is not None:
            style.save_png(page, self.png_dir / f"{path.stem}.png")
        said = f" ({', '.join(page.stamps)} across it: not a figure of the paper)" if page.stamps else ""
        return Made(number, WRITTEN, f"{name}: wrote {self.shown(path)}{said}", path, tuple(page.notes))


#: What an input that is there and cannot be used raises.
_CANNOT_USE = (
    FigureError,
    ResultsError,
    SchemaError,
    GeometryError,
    LayoutError,
    config.ConfigError,
    json.JSONDecodeError,
    tomllib.TOMLDecodeError,
    UnicodeDecodeError,
    OSError,
)


def _figure_1(job: _Job) -> _Drawn:
    trials = earlier_evaluation.read_trials(job.need(job.root / EARLIER_TRIALS))
    return _Drawn(earlier_evaluation.figure(trials), job.out_dir / FILE_NAMES[1])


def _audit(job: _Job, dataset: str) -> list[DemoStart]:
    path = job.need(audit_path(dataset, job.root))
    try:
        return read_demo_starts(path)
    except (KeyError, TypeError, AttributeError) as error:
        raise SchemaError(f"{job.shown(path)} is not a start-state audit: {error!r}") from error


def _figure_2(job: _Job) -> _Drawn:
    for dataset in DATASETS:  # name the first audit that is missing before reading any
        job.need(audit_path(dataset, job.root))
    grid = job.grid()
    starts = {dataset: _audit(job, dataset) for dataset in DATASETS}
    return _Drawn(layouts_on_grid.figure(starts, grid), job.out_dir / FILE_NAMES[2])


def _planned_figure_2(job: _Job) -> _Drawn:
    paths = [job.need(layouts.layout_path(dataset, job.root)) for dataset in DATASETS]
    grid = job.grid()
    starts = {
        dataset: layouts.planned_demo_starts(layouts.read_layout(path))
        for dataset, path in zip(DATASETS, paths, strict=True)
    }
    return _Drawn(layouts_on_grid.figure(starts, grid, stamp=PLANNED), job.root / PLANNED_PREVIEW, must_say=PLANNED)


def _figure_3(job: _Job) -> _Drawn:
    if not response_curves.has_something_to_draw(job.results()):
        raise _Waiting(f"trials of a threshold block in {job.shown(job.results_path)}")
    return job.results_figure(3, response_curves.figure)


def _figure_4(job: _Job) -> _Drawn:
    if not results.has_workspace(job.results()):
        raise _Waiting(f"the workspace test in {job.shown(job.results_path)}")
    grid = job.grid()
    return job.results_figure(4, lambda results_: workspace.figure(results_, grid))


_DRAWERS: dict[int, Callable[[_Job], _Drawn]] = {1: _figure_1, 2: _figure_2, 3: _figure_3, 4: _figure_4}
