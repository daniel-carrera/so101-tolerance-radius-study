"""The scene at each trial's start and end, read from its videos (`rig/scene_audit.py`): drawn videos, no camera."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAVE_CV2 = importlib.util.find_spec("cv2") is not None


def video(path: Path, frames: list) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for frame in frames:
        writer.write(frame)
    writer.release()


@unittest.skipUnless(HAVE_CV2, "needs OpenCV (the rig's interpreter)")
class SceneAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_rig_basket import TAGS, quad, scene
        from test_rig_wrist_place import picture

        from tolerance_radius.rig import basket_place, scene_audit, wrist_place

        cls.audit, cls.scene, cls.picture = scene_audit, staticmethod(scene), staticmethod(picture)
        cls.basket_reference = basket_place.read_basket(*scene())
        cls.jaw_reference = wrist_place.read_jaw(picture())
        # the drawn scenes carry no tags a detector could find: the tags are told, as where the camera stands
        cls.detect = staticmethod(lambda frame: {tag: quad(centre) for tag, centre in TAGS.items()})

    def trial(self, folder: Path, *, basket_end=(0.0, 0.0), wrist_start=(0.0, 0.0), basket_start=(0.0, 0.0)):
        front, wrist = folder / "t_front.mp4", folder / "t_wrist.mp4"
        video(front, [self.scene(basket=basket_start)[0]] * 20 + [self.scene(basket=basket_end)[0]] * 20)
        video(wrist, [self.picture(moved=wrist_start)] * 40)
        return self.audit.audit_trial(front, wrist, self.basket_reference, self.jaw_reference, self.detect, None)

    def test_a_trial_that_left_the_scene_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            row = self.trial(Path(folder))
        self.assertLess(abs(row["basket_start"]["across_mm"]), 1.5)
        self.assertLess(row["basket_moved_mm"], 2.0)
        self.assertIs(row["basket_was_moved"], False)
        self.assertIs(row["named_basket"], False)
        self.assertIs(row["named_wrist"], False)
        self.assertLess(abs(row["wrist_start"]["across_px"]), 1.0)

    def test_a_trial_in_which_the_basket_was_carried_off(self):
        # as on 2026-10-08: in place at the start, four centimetres away at the end
        with tempfile.TemporaryDirectory() as folder:
            row = self.trial(Path(folder), basket_end=(36.0, -10.0))
        self.assertIs(row["basket_was_moved"], True)
        self.assertAlmostEqual(row["basket_moved_mm"], float(np.hypot(36.0, 10.0)) * 1.08, delta=4.0)
        self.assertIs(row["named_basket"], False, "it began in place")
        self.assertGreater(row["basket_end"]["across_mm"], 30.0)

    def test_a_trial_that_began_with_the_scene_out_of_place_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            row = self.trial(Path(folder), basket_start=(5.0, 0.0), basket_end=(5.0, 0.0), wrist_start=(-3.0, 1.0))
        self.assertIs(row["named_basket"], True)
        self.assertIs(row["named_wrist"], True)
        self.assertIs(row["basket_was_moved"], False)
        self.assertAlmostEqual(row["wrist_start"]["across_px"], -3.0, delta=0.8)
        self.assertEqual((self.audit.BASKET_NAMED_MM, self.audit.WRIST_NAMED_PX), (2.0, 2.0), "the protocol's lines")

    def test_the_arm_standing_before_the_basket_at_a_trial_s_end_is_not_read_as_the_basket(self):
        """2026-10-08, T1a-012: the attempt ran out with the arm beside the cube, before the basket. Its dark parts
        were read for the basket's wall, 53 mm away, and the basket had not moved."""
        import cv2

        def with_arm(frame):
            out = frame.copy()
            cv2.rectangle(out, (385, 60), (425, 215), (20, 110, 240), -1)      # the arm, orange
            cv2.rectangle(out, (392, 90), (412, 190), (30, 30, 30), -1)        # its dark camera and cable
            return out

        plain = self.scene()[0]
        self.assertFalse(self.audit.arm_before_basket(plain, self.basket_reference))
        self.assertTrue(self.audit.arm_before_basket(with_arm(plain), self.basket_reference))
        with tempfile.TemporaryDirectory() as folder:
            front = Path(folder) / "t_front.mp4"
            video(front, [plain] * 20 + [with_arm(plain)] * 20)
            row = self.audit.audit_trial(front, None, self.basket_reference, self.jaw_reference, self.detect, None)
        self.assertIsNotNone(row["basket_start"])
        self.assertIsNone(row["basket_end"], "the arm hides the wall: nothing is said of the basket")
        self.assertIsNone(row["basket_was_moved"])

    def test_a_trial_without_videos_is_said_to_be_unread_and_not_in_place(self):
        row = self.audit.audit_trial(None, Path("no/such_wrist.mp4"), self.basket_reference, self.jaw_reference,
                                     self.detect, None)
        self.assertIsNone(row["basket_start"])
        self.assertIsNone(row["wrist_start"])
        self.assertIsNone(row["named_basket"])
        self.assertIsNone(row["named_wrist"])
        self.assertIsNone(row["basket_was_moved"])

    def test_a_log_is_read_trial_by_trial_and_summed_up(self):
        import cv2

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "v").mkdir()
            video(root / "v" / "a_front.mp4", [self.scene()[0]] * 20 + [self.scene(basket=(36.0, 0.0))[0]] * 20)
            video(root / "v" / "a_wrist.mp4", [self.picture()] * 40)
            cv2.imwrite(str(root / "basket.png"), self.scene()[0])
            cv2.imwrite(str(root / "wrist.png"), self.picture())
            log = root / "log.jsonl"
            log.write_text("".join(json.dumps(entry) + "\n" for entry in (
                {"data": {"planned": {"trial_id": "T1a-001", "condition": "act_clustered"}, "outcome": "success",
                          "media": {"front": "v/a_front.mp4", "wrist": "v/a_wrist.mp4"}}},
                {"planned": {"trial_id": "T1a-002", "condition": "replay"}, "outcome": "void", "media": {}},
            )), encoding="utf-8")
            rows = self.audit.audit_log(log, root, root / "basket.png", root / "wrist.png", self.detect, None)
        self.assertEqual([row["trial_id"] for row in rows], ["T1a-001", "T1a-002"])
        self.assertIs(rows[0]["basket_was_moved"], True)
        self.assertEqual(rows[0]["session"], "v")
        self.assertIsNone(rows[1]["basket_start"])
        lines = self.audit.told(rows)
        self.assertIn("basket moved", lines[1])
        self.assertIn("2 trials with a record; the basket was moved in 1;", lines[-1])
        self.assertIn("videos that could not be read: 1", lines[-1])


if __name__ == "__main__":
    unittest.main()
