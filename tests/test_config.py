"""The protocol file loads, and says what the planner and the analysis expect of it."""

import tempfile
import unittest
from pathlib import Path

from tolerance_radius import config, schema


class ProtocolFileTests(unittest.TestCase):
    def setUp(self):
        self.protocol = config.load_protocol()

    def test_the_repository_protocol_loads_with_every_section(self):
        for section in config.REQUIRED_SECTIONS:
            self.assertIn(section, self.protocol)

    def test_blocks_are_known_to_the_schema_and_name_known_conditions(self):
        for name, block in self.protocol["blocks"].items():
            with self.subTest(block=name):
                self.assertIn(name, schema.BLOCKS)
                conditions = block.get("conditions", [block.get("condition")])
                for condition in conditions:
                    self.assertIn(condition, schema.CONDITIONS)

    def test_held_out_cells_lie_on_the_grid(self):
        table = self.protocol["table"]
        labels = {f"{chr(ord('A') + c)}{r + 1}" for c in range(table["cols"]) for r in range(table["rows"])}
        self.assertTrue(set(table["held_out_cells"]) <= labels)

    def test_each_dataset_has_55_episodes_in_whole_rounds(self):
        table, datasets = self.protocol["table"], self.protocol["datasets"]
        trained_cells = table["cols"] * table["rows"] - len(table["held_out_cells"])
        self.assertEqual(trained_cells * datasets["episodes_per_cell"], 55)
        for dataset in schema.DATASETS:
            rounds = sorted(r for s in datasets["sittings"] if s["dataset"] == dataset for r in s["rounds"])
            self.assertEqual(rounds, list(range(1, datasets["episodes_per_cell"] + 1)))

    def test_staircase_starts_lie_within_their_block_limits(self):
        low = self.protocol["staircase"]["min_level_mm"]
        for name, block in self.protocol["blocks"].items():
            for staircase in block.get("staircases", []):
                with self.subTest(block=name, staircase=staircase["id"]):
                    self.assertTrue(low <= staircase["start_level_mm"] <= block["max_level_mm"])
                    self.assertGreater(staircase["trials"], 0)

    def test_the_workspace_test_draws_as_many_placements_over_the_held_out_cells_as_over_the_trained_cells(self):
        block = self.protocol["blocks"]["T2"]
        counts = {stratum["region"]: stratum["pairs"] for stratum in block["regions"]}
        self.assertEqual(set(counts), {"trained_cells", "held_out_cells"})
        self.assertEqual(counts["trained_cells"], counts["held_out_cells"])
        self.assertEqual(sum(counts.values()), block["pairs"])
        self.assertNotIn("region", block)  # one way of saying where the placements are drawn, not two

    def test_the_replay_block_has_two_trials_for_a_demonstration_from_every_trained_cell(self):
        table, block = self.protocol["table"], self.protocol["blocks"]["T1b"]
        trained_cells = table["cols"] * table["rows"] - len(table["held_out_cells"])
        self.assertEqual(block["replay_demos"], trained_cells)
        trials = block["anchors_zero"] + sum(staircase["trials"] for staircase in block["staircases"])
        self.assertEqual(trials, 2 * block["replay_demos"])


class PilotOverlayTests(unittest.TestCase):
    def setUp(self):
        self.base = config.load_protocol()
        self.pilot = config.load_protocol(overlay=config.PILOT_OVERLAY_PATH)

    def test_the_pilot_changes_only_what_its_file_names(self):
        self.assertEqual(self.pilot["protocol"]["status"], "pilot")
        self.assertEqual(self.pilot["blocks"]["T1a"]["anchors_zero"], 9)
        self.assertEqual(self.pilot["blocks"]["T1a"]["staircases"], [])
        self.assertEqual(self.pilot["blocks"]["T1a"]["condition"], "act_clustered")  # kept from the protocol
        for untouched in ("table", "placement", "datasets", "staircase", "analysis"):
            self.assertEqual(self.pilot[untouched], self.base[untouched])
        self.assertEqual(self.pilot["blocks"]["T2"], self.base["blocks"]["T2"])
        # The replay pilot is shorter, and plays back as many demonstrations as the counted block does.
        self.assertEqual(self.pilot["blocks"]["T1b"]["replay_demos"], self.base["blocks"]["T1b"]["replay_demos"])

    def test_loading_a_pilot_leaves_the_protocol_as_it_was(self):
        self.assertEqual(config.load_protocol(), self.base)
        self.assertNotEqual(self.base["protocol"].get("status"), "pilot")
        self.assertEqual(self.base["blocks"]["T1a"]["anchors_zero"], 6)


class LoaderTests(unittest.TestCase):
    def test_a_file_missing_a_section_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.toml"
            path.write_text('[protocol]\nversion = "x"\n', encoding="utf-8")
            with self.assertRaises(config.ConfigError):
                config.load_protocol(path)


if __name__ == "__main__":
    unittest.main()
