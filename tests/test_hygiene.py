"""Nothing tracked by git may say whose machine or whose work this is: the review is double blind.

`paper/check_paper.py` guards what goes into the PDF. These tests guard the rest of the repository, because the code,
the runbooks and the logs are what an anonymous link would show.
"""

import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: A home directory in a path: `/Users/<name>` or `/home/<name>`, but not the same letters inside a web address.
HOME_PATH = re.compile(r"(?<![\w.])(?:/Users|/home)/([A-Za-z0-9._-]+)")
#: Names used in examples and tests.
EXAMPLE_NAMES = {"someone"}
#: The author's own strings, kept out of git (`paper/.gitignore`); the header of `paper/forbidden.txt` explains it.
LOCAL_LIST = REPO / "paper" / "forbidden.local.txt"


def tracked_text_files() -> list[tuple[Path, str]]:
    """Every file git tracks that reads as text, with its text."""
    try:
        listing = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True,
                                 timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise unittest.SkipTest(f"not a git checkout, or git is missing: {error}") from error
    files = []
    for name in listing.decode("utf-8").split("\0"):
        path = REPO / name
        if not name or not path.is_file():
            continue
        try:
            files.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue  # a picture or a PDF: paper/check_paper.py searches the bytes of the figures
    return files


class NoIdentityInTheRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tracked_text_files()

    def test_git_tracks_something(self):
        self.assertGreater(len(self.files), 50)

    def test_no_tracked_file_names_a_home_directory(self):
        hits = []
        for path, text in self.files:
            for number, line in enumerate(text.splitlines(), start=1):
                names = set(HOME_PATH.findall(line)) - EXAMPLE_NAMES
                if names:
                    hits.append(f"{path.relative_to(REPO)}:{number}")
        self.assertEqual(hits, [], "a home directory is written into a tracked file; use $HOME or a relative path")

    def test_a_launcher_does_not_name_its_interpreter_by_an_absolute_path(self):
        allowed = ("#!/bin/sh", "#!/usr/bin/env python3")
        for path, text in self.files:
            if text.startswith("#!"):
                with self.subTest(file=str(path.relative_to(REPO))):
                    self.assertIn(text.splitlines()[0], allowed)

    def test_the_authors_own_strings_appear_nowhere(self):
        if not LOCAL_LIST.is_file():
            self.skipTest("paper/forbidden.local.txt is not there: only the generic checks ran")
        entries = [line.strip().lower() for line in LOCAL_LIST.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#") and ":" not in line.split()[0]]
        self.assertTrue(entries, "paper/forbidden.local.txt holds no plain entry")
        hits = []
        for path, text in self.files:
            lowered = text.lower()
            for index, entry in enumerate(entries):
                if entry in lowered:
                    line = lowered[: lowered.index(entry)].count("\n") + 1
                    hits.append(f"{path.relative_to(REPO)}:{line} (entry {index + 1} of the local list)")
        # The entries themselves are not printed: a test log must not carry them either.
        self.assertEqual(hits, [], "an identifying string is written into a tracked file")


if __name__ == "__main__":
    unittest.main()
