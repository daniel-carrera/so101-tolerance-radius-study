"""The command that writes the layouts and their preview: it runs, says what it did, and guards what is on disk."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from fixtures_demos import GRID, PROTOCOL
from tolerance_radius import config, layouts

_SPEC = importlib.util.spec_from_file_location("make_layouts", config.REPO_ROOT / "scripts" / "make_layouts.py")
make_layouts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_layouts)


def toml_text(protocol: dict) -> str:
    """The repository's protocol file with another layout seed."""
    text = config.DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8")
    old = f"layout_seed = {PROTOCOL['datasets']['layout_seed']}"
    assert text.count(old) == 1
    return text.replace(old, f"layout_seed = {protocol['datasets']['layout_seed']}")


class MakeLayoutsTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def run_command(self, *arguments: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = make_layouts.main(["--root", str(self.root), *arguments])
        return code, out.getvalue(), err.getvalue()

    def test_it_writes_both_layouts_and_the_figure_and_prints_the_quality_figures(self):
        code, out, err = self.run_command()
        self.assertEqual((code, err), (0, ""))
        for dataset in ("clustered", "spread"):
            episodes = layouts.read_layout(self.root / "config" / "layouts" / f"{dataset}.json")
            self.assertEqual(episodes, layouts.layout(dataset, PROTOCOL))
            quality = layouts.layout_quality(episodes, GRID)
            self.assertIn(f"{quality.largest_gap_mm:.1f} mm", out)
            self.assertIn(f"{quality.mean_nearest_neighbour_mm:.1f} mm", out)
        self.assertIn("35.4 mm", out)
        figure = self.root / "analysis" / "layouts" / "layouts.png"
        self.assertEqual(figure.read_bytes()[:4], b"\x89PNG")
        self.assertIn("config/layouts/spread.json", out)
        self.assertIn("analysis/layouts/layouts.png", out)

    def test_it_says_whether_the_blocks_can_run_on_these_layouts(self):
        _, out, _ = self.run_command()
        for block in PROTOCOL["blocks"]:
            self.assertIn(block, out)

    def test_what_it_prints_names_no_folder_of_this_machine(self):
        _, out, _ = self.run_command()
        self.assertNotIn(str(self.root), out)
        self.assertNotIn(str(Path.home()), out)

    def test_running_it_again_changes_nothing(self):
        self.run_command()
        path = self.root / "config" / "layouts" / "spread.json"
        before = path.read_bytes()
        code, out, _ = self.run_command()
        self.assertEqual(code, 0)
        self.assertIn("unchanged", out)
        self.assertEqual(path.read_bytes(), before)

    def test_it_does_not_replace_a_different_layout_unless_told_to(self):
        self.run_command()
        path = self.root / "config" / "layouts" / "spread.json"
        figure = self.root / "analysis" / "layouts" / "layouts.png"
        before = path.read_bytes()
        figure.unlink()
        changed = copy.deepcopy(PROTOCOL)
        changed["datasets"]["layout_seed"] = 1
        other = self.root / "other.toml"
        other.write_text(toml_text(changed), encoding="utf-8")
        code, _, err = self.run_command("--protocol", str(other))
        self.assertEqual(code, 1)
        self.assertIn("spread.json", err)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(figure.exists())  # nothing is half done: the figure is not redrawn for a refused layout
        code, out, _ = self.run_command("--protocol", str(other), "--replace")
        self.assertEqual(code, 0)
        self.assertEqual(layouts.read_layout(path), layouts.layout("spread", changed))
        self.assertTrue(figure.exists())


if __name__ == "__main__":
    unittest.main()
