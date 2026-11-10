"""The replay source: it hands out the recorded actions in order, says when it is done, and never invents a value."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tolerance_radius.rig.replay import ReplayError, ReplaySource, load_episode_actions

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def recorded(steps: int = 4) -> np.ndarray:
    """A small action table: row i is [i, i + 0.1, ..., i + 0.5]."""
    return np.arange(steps)[:, None] + np.arange(6)[None, :] / 10.0


class ReplaySourceTests(unittest.TestCase):
    def test_actions_come_out_in_recorded_order_with_the_followers_keys(self):
        source = ReplaySource(recorded(), JOINTS)
        first = source.predict()
        self.assertEqual(list(first), [f"{name}.pos" for name in JOINTS])
        self.assertEqual(first["shoulder_pan.pos"], 0.0)
        self.assertAlmostEqual(first["gripper.pos"], 0.5)
        self.assertEqual(source.predict()["shoulder_pan.pos"], 1.0)

    def test_reset_rewinds_so_the_gate_and_the_loop_start_from_the_first_action(self):
        source = ReplaySource(recorded(), JOINTS)
        source.predict()
        source.predict()
        source.reset()
        self.assertEqual(source.steps_handed_out, 0)
        self.assertEqual(source.predict()["shoulder_pan.pos"], 0.0)

    def test_the_end_is_signalled_with_the_last_action_and_the_pose_is_then_held(self):
        events = {"exit_early": False}
        source = ReplaySource(recorded(3), JOINTS, events=events)
        source.predict()
        source.predict()
        self.assertFalse(events["exit_early"])
        last = source.predict()
        self.assertTrue(events["exit_early"])
        self.assertEqual(last["shoulder_pan.pos"], 2.0)
        self.assertEqual(source.predict(), last)  # asked again: the same pose, never a value past the recording

    def test_each_step_reads_one_observation_and_offers_it_to_the_recorder(self):
        seen = []
        source = ReplaySource(recorded(), JOINTS, observe=lambda: {"front": "frame"}, on_observation=seen.append)
        source.predict()
        source.predict()
        self.assertEqual(seen, [{"front": "frame"}, {"front": "frame"}])

    def test_names_that_already_carry_the_suffix_are_kept(self):
        source = ReplaySource(recorded(), [f"{name}.pos" for name in JOINTS])
        self.assertEqual(list(source.predict()), [f"{name}.pos" for name in JOINTS])

    def test_the_largest_step_is_reported_for_the_clamp(self):
        actions = recorded(3)
        actions[2, 4] += 7.5
        self.assertAlmostEqual(ReplaySource(actions, JOINTS).largest_step(), 8.5)
        self.assertEqual(ReplaySource(recorded(1), JOINTS).largest_step(), 0.0)

    def test_unusable_recordings_are_refused(self):
        with self.assertRaises(ReplayError):
            ReplaySource(np.empty((0, 6)), JOINTS)
        with self.assertRaises(ReplayError):
            ReplaySource(recorded(), JOINTS[:5])
        broken = recorded()
        broken[1, 2] = np.nan
        with self.assertRaises(ReplayError):
            ReplaySource(broken, JOINTS)


try:
    import pandas as pd

    pd.DataFrame({"a": [1]}).to_parquet(Path(tempfile.gettempdir()) / "_tolrad_parquet_probe.parquet")
    HAVE_PARQUET = True
except Exception:  # noqa: BLE001 -- no pandas or no parquet engine in this environment
    HAVE_PARQUET = False


@unittest.skipUnless(HAVE_PARQUET, "needs pandas with a parquet engine (run with the rig's interpreter)")
class LoadEpisodeActionsTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        (self.root / "meta").mkdir()
        (self.root / "meta" / "info.json").write_text(
            json.dumps({"features": {"action": {"names": [f"{name}.pos" for name in JOINTS]}}}), encoding="utf-8"
        )
        data = self.root / "data" / "chunk-000"
        data.mkdir(parents=True)
        rows = []
        for episode in (0, 1):
            for frame in range(3):
                rows.append({"action": list(recorded(3)[frame] + 100 * episode), "episode_index": episode,
                             "frame_index": frame})
        pd.DataFrame(rows[:4]).to_parquet(data / "file-000.parquet")  # an episode may span two files
        pd.DataFrame(rows[4:]).to_parquet(data / "file-001.parquet")

    def tearDown(self):
        self._dir.cleanup()

    def test_it_returns_one_episodes_actions_in_frame_order_with_their_names(self):
        actions, names = load_episode_actions(self.root, 1)
        self.assertEqual(actions.shape, (3, 6))
        self.assertTrue(np.allclose(actions, recorded(3) + 100))
        self.assertEqual(names[0], "shoulder_pan.pos")

    def test_a_missing_episode_or_dataset_is_an_error(self):
        with self.assertRaises(ReplayError):
            load_episode_actions(self.root, 7)
        with self.assertRaises(ReplayError):
            load_episode_actions(self.root / "nowhere", 0)

    def test_an_episode_with_missing_frames_is_refused(self):
        frame = pd.read_parquet(self.root / "data" / "chunk-000" / "file-001.parquet")
        frame[frame["frame_index"] != 1].to_parquet(self.root / "data" / "chunk-000" / "file-001.parquet")
        with self.assertRaises(ReplayError):
            load_episode_actions(self.root, 1)


if __name__ == "__main__":
    unittest.main()
