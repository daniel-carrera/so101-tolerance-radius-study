"""Shared by the tests of the figures command: a folder laid out like the repository, and what is put into it."""

import json
import shutil
from pathlib import Path

from fixtures_demos import audit
from tolerance_radius import schema
from tolerance_radius.figures import make

REPO = Path(__file__).resolve().parents[1]
#: What the figures read that exists in the repository today.
THERE_TODAY = (
    make.EARLIER_TRIALS,
    make.PROTOCOL,
    make.PAPER / make.SHIPPED_LIST,
    Path("config/layouts/clustered.json"),
    Path("config/layouts/spread.json"),
)


def new_root(folder: Path) -> Path:
    """Make `folder` a folder laid out like the repository, holding copies of what the figures read that exists
    today: the CSV of the earlier evaluation, the protocol, the two layouts, the paper's list of forbidden strings.
    """
    for relative in THERE_TODAY:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    return folder


def add_audits(root: Path, datasets=schema.DATASETS) -> None:
    """Put start-state audits where the audit writes them: synthetic ones, as a camera might measure."""
    for dataset in datasets:
        schema.write_demo_starts(make.audit_path(dataset, root), audit(dataset))


def add_results(root: Path, results: dict, relative: Path = make.RESULTS) -> Path:
    """Write analysis results into the folder, where the analysis writes them unless another place is named."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def files_under(folder: Path) -> list[str]:
    """Every file under a folder, by its path from it."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())
