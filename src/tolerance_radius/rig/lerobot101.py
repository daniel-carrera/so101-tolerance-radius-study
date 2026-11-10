"""The door to the earlier SO-101 project's tooling, and to the LeRobot that drove the arm in August.

That project (`../lerobot-101`) holds the proven pieces this one reuses: the safety gates around every control loop, the
tag detection that fixes the table frame, and the cube detector. Its scripts are plain modules, not a package, so they
are imported by putting their folders on the path. This module is the only place that does so, and it records which
commit of that project this one was built and tested against.

It also settles which LeRobot an arm session runs. In August the arm was driven by an editable checkout of LeRobot. The
rig's environment has since acquired a different packaged LeRobot (0.6.0) that lacks the servo-bus read retries that
checkout has, and that copy has never driven the arm. `use_pinned_lerobot` puts the August checkout first on the path
and confirms that it is the one imported, so an arm session here runs the code that was proven, whatever the
environment's own copy is. Nothing is installed or removed.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from tolerance_radius.config import REPO_ROOT

#: The commit of the arm tooling this project was built against (docs/journal.md, 2026-10-05).
PINNED_COMMIT = "02b3ced82f98487c4a5eb0c89b22ae6efb5474e7"

#: The commit of the LeRobot checkout that drove the arm in August (the same journal entry).
LEROBOT_PINNED_COMMIT = "6adf51511b7625090eade8d82d9f61a1846ebe56"

#: Folders of the arm tooling whose modules are imported by bare name.
SCRIPT_FOLDERS = ("scripts", "scripts/session", "scripts/camera", "scripts/data", "scripts/policy", "scripts/ops")


class ToolingError(RuntimeError):
    """The arm tooling is missing, or is not the version this project was tested against."""


def root() -> Path:
    """Where the arm tooling lives: `LEROBOT101_ROOT` if set, else the sibling folder `lerobot-101`."""
    return Path(os.environ.get("LEROBOT101_ROOT", REPO_ROOT.parent / "lerobot-101")).resolve()


def available() -> bool:
    """True when the arm tooling's scripts are present on this machine."""
    return (root() / "scripts" / "data" / "placement.py").is_file()


def current_commit() -> str:
    """The commit the arm tooling's checkout is at."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root()), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ToolingError(f"cannot read the commit of {root()}: {error}") from error
    return result.stdout.strip()


def has_local_changes() -> bool:
    """True when tracked files of the arm tooling differ from its commit."""
    result = subprocess.run(
        ["git", "-C", str(root()), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return bool(result.stdout.strip())


def check_pin(*, allow_other: bool = False) -> str:
    """Confirm the arm tooling is the tested version, and return its commit.

    A different commit or locally changed files mean the gates being relied on are not the ones that were tested.
    That is refused unless `allow_other` is set, in which case the caller is expected to log the commit it got.
    """
    if not available():
        raise ToolingError(f"the arm tooling was not found at {root()} (set LEROBOT101_ROOT to point at it)")
    commit = current_commit()
    problems = []
    if commit != PINNED_COMMIT:
        problems.append(f"it is at {commit[:10]}, and this project was tested against {PINNED_COMMIT[:10]}")
    if has_local_changes():
        problems.append("it has uncommitted changes to tracked files")
    if problems and not allow_other:
        raise ToolingError(f"the arm tooling at {root()} is not the tested version: " + "; ".join(problems))
    return commit


def add_to_path() -> None:
    """Make the arm tooling's modules importable by their bare names (`placement`, `tag_check`, ...)."""
    if not available():
        raise ToolingError(f"the arm tooling was not found at {root()} (set LEROBOT101_ROOT to point at it)")
    for folder in reversed(SCRIPT_FOLDERS):
        path = str(root() / folder)
        if path not in sys.path:
            sys.path.insert(0, path)
    put_interpreter_folder_on_path()


def put_interpreter_folder_on_path() -> None:
    """Put the running interpreter's own folder first on PATH, once.

    The sessions spawn a viewer (`rerun`) that is installed beside the rig's interpreter. The runbooks start that
    interpreter by its full path without activating its environment, and then the viewer is not found: the rig check
    of 2026-10-06 stopped on `Failed to find Rerun Viewer executable in PATH`.
    """
    folder = str(Path(sys.executable).parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if folder not in parts:
        os.environ["PATH"] = os.pathsep.join([folder, *[part for part in parts if part]])


def load(name: str) -> ModuleType:
    """Import one module of the arm tooling by name, for example `load("placement")`."""
    add_to_path()
    return importlib.import_module(name)


# --------------------------------------------------------------------------------------------------------------------
# LeRobot itself
# --------------------------------------------------------------------------------------------------------------------


def lerobot_checkout() -> Path:
    """Where the LeRobot checkout lives: `LEROBOT_CHECKOUT` if set, else `<dev>/so-101/lerobot`."""
    return Path(os.environ.get("LEROBOT_CHECKOUT", REPO_ROOT.parent.parent / "so-101" / "lerobot")).resolve()


def use_pinned_lerobot(*, allow_other: bool = False) -> str:
    """Make `import lerobot` mean the checkout that drove the arm in August, and say what was imported.

    Call it before anything imports `lerobot`. It refuses when the checkout is missing, when it is not at the pinned
    commit or has local changes (unless `allow_other`), and when a different LeRobot has already been imported in this
    process. Returns a line for the session log: version, commit and location.
    """
    checkout = lerobot_checkout()
    source = checkout / "src"
    if not (source / "lerobot" / "__init__.py").is_file():
        raise ToolingError(f"no LeRobot checkout at {checkout} (set LEROBOT_CHECKOUT to point at it)")
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=False, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ToolingError(f"cannot read the commit of {checkout}: {error}") from error
    problems = []
    if commit != LEROBOT_PINNED_COMMIT:
        problems.append(f"it is at {commit[:10]}, and the arm was driven at {LEROBOT_PINNED_COMMIT[:10]}")
    if changed:
        problems.append("it has uncommitted changes to tracked files")
    if problems and not allow_other:
        raise ToolingError(f"the LeRobot checkout at {checkout} is not the tested version: " + "; ".join(problems))

    already = sys.modules.get("lerobot")
    if already is not None and not _is_under(Path(already.__file__), source):
        raise ToolingError(
            f"a different LeRobot is already imported in this process ({already.__file__}); "
            f"use_pinned_lerobot must be called before anything imports lerobot"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    lerobot = importlib.import_module("lerobot")
    if not _is_under(Path(lerobot.__file__), source):
        raise ToolingError(f"LeRobot was imported from {lerobot.__file__}, not from the checkout at {source}")
    return f"lerobot {getattr(lerobot, '__version__', 'unknown')} at {commit[:10]} from {source}"


def _is_under(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True
