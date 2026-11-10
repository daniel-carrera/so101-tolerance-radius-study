"""Which cameras the computer offers, by name (`tolerance_radius.rig.camera_list`).

The listings are made up in the form `system_profiler SPCameraDataType -json` gives them; the identifiers are not
this machine's.
"""

from __future__ import annotations

import json
import unittest

from tolerance_radius.rig import camera_list
from tolerance_radius.rig.camera_list import FRONT_NAME, WRIST_NAME

LAPTOP = "MacBook Pro Camera"


def listing(*cameras: tuple[str, str]) -> str:
    """The system's answer for these cameras, given as (name, unique identifier), in the order the system has them."""
    return json.dumps({"SPCameraDataType": [{"_name": name, "spcamera_model-id": name, "spcamera_unique-id": unique}
                                             for name, unique in cameras]})


#: A USB camera's identifier begins with the port it is plugged into; the laptop's own is of another kind.
FRONT_ON_ITS_HUB = (FRONT_NAME, "0x110000011112222")
WRIST_ON_THE_ARM_S_HUB = (WRIST_NAME, "0x114000033334444")
OWN = (LAPTOP, "6C707041-0000-0000-0000-000000000000")


class CameraListTests(unittest.TestCase):
    def names(self, text: str) -> list[str]:
        return [camera.name for camera in camera_list.parse(text)]

    def test_the_cameras_come_in_the_order_the_sessions_number_them_whatever_order_the_system_lists_them_in(self):
        self.assertEqual(self.names(listing(OWN, FRONT_ON_ITS_HUB, WRIST_ON_THE_ARM_S_HUB)),
                         [FRONT_NAME, WRIST_NAME, LAPTOP])

    def test_the_rig_as_it_recorded_is_in_place(self):
        cameras = camera_list.parse(listing(OWN, FRONT_ON_ITS_HUB, WRIST_ON_THE_ARM_S_HUB))
        self.assertEqual(camera_list.problems(cameras, 0, 1), ([], []))
        self.assertEqual(camera_list.lines(cameras, 0, 1), [
            f"camera 0: {FRONT_NAME}  (the session's front camera)",
            f"camera 1: {WRIST_NAME}  (the session's wrist camera)",
            f"camera 2: {LAPTOP}  (not used)",
        ])

    def test_a_wrist_camera_that_did_not_start_leaves_its_number_to_the_laptop_s_camera(self):
        # 2026-10-06, 19:22: the session took the laptop's camera for the wrist and nothing said so.
        cameras = camera_list.parse(listing(OWN, FRONT_ON_ITS_HUB))
        certain, likely = camera_list.problems(cameras, 0, 1)
        self.assertEqual(likely, [])
        self.assertEqual(len(certain), 1)
        self.assertIn(f"the wrist camera ('{WRIST_NAME}') is not among the cameras", certain[0])
        self.assertIn(f"camera 1 is now '{LAPTOP}'", certain[0])
        self.assertIn("WRONG", camera_list.lines(cameras, 0, 1)[1])

    def test_the_camera_plugged_straight_into_the_computer_comes_first(self):
        # 19:41: the wrist camera in the computer's own port, the front camera on a hub: the two numbers swapped.
        cameras = camera_list.parse(listing(OWN, (FRONT_NAME, "0x214000011112222"), (WRIST_NAME, "0x10000033334444")))
        self.assertEqual([camera.name for camera in cameras], [WRIST_NAME, FRONT_NAME, LAPTOP])
        certain, likely = camera_list.problems(cameras, 0, 1)
        self.assertEqual(certain, [])
        self.assertEqual(len(likely), 2)
        self.assertIn("looks to be camera 1, and the session takes camera 0", likely[0])
        self.assertIn("looks to be camera 0, and the session takes camera 1", likely[1])

    def test_no_study_camera_at_all(self):
        cameras = camera_list.parse(listing(OWN))
        certain, _likely = camera_list.problems(cameras, 0, 1)
        self.assertEqual(len(certain), 2)
        self.assertIn("there is no camera 1", certain[1])
        self.assertEqual(camera_list.lines(camera_list.parse(listing()), 0, 1), ["the system offers no camera"])

    def test_a_number_beyond_the_list_is_said_and_does_not_fail(self):
        # Two cameras offered and a session that takes camera 2 for one of them: the review of 2026-10-06 took the
        # guard on the number away and no test said so.
        cameras = camera_list.parse(listing(OWN, FRONT_ON_ITS_HUB))
        certain, likely = camera_list.problems(cameras, 0, 2)
        self.assertEqual(likely, [])
        self.assertEqual(len(certain), 1)
        self.assertIn("there is no camera 2", certain[0])
        cameras = camera_list.parse(listing(FRONT_ON_ITS_HUB, WRIST_ON_THE_ARM_S_HUB))
        certain, likely = camera_list.problems(cameras, 0, 5)
        self.assertEqual(certain, [])
        self.assertIn("looks to be camera 1, and the session takes camera 5", likely[0])
        self.assertEqual(len(camera_list.lines(cameras, 0, 5)), 2)

    def test_the_system_is_asked_for_its_list_without_opening_a_camera_and_a_failed_answer_is_an_error(self):
        from types import SimpleNamespace
        from unittest import mock

        asked = []

        def run(command, **kwargs):
            asked.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout=listing(FRONT_ON_ITS_HUB, WRIST_ON_THE_ARM_S_HUB), stderr="")

        with mock.patch.object(camera_list.subprocess, "run", run):
            names = [camera.name for camera in camera_list.system_cameras()]
        self.assertEqual(names, [FRONT_NAME, WRIST_NAME])
        self.assertEqual(asked[0][0], ["system_profiler", "SPCameraDataType", "-json"])
        self.assertFalse(asked[0][1].get("shell", False))
        failed = lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="no")  # noqa: E731
        with mock.patch.object(camera_list.subprocess, "run", failed), \
                self.assertRaisesRegex(camera_list.CameraListError, "exit 1"):
            camera_list.system_cameras()

        def absent(command, **kwargs):
            raise FileNotFoundError("system_profiler")

        with mock.patch.object(camera_list.subprocess, "run", absent):
            told, wrong = camera_list.check_for_session(0, 1)
        self.assertIsNone(wrong, "a list that cannot be read must not stop a session by itself")
        self.assertIn("not checked", told[0])

    def test_other_names_can_be_given(self):
        cameras = camera_list.parse(listing(("A", "0x1"), ("B", "0x2")))
        self.assertEqual(camera_list.problems(cameras, 0, 1, front_name="A", wrist_name="B"), ([], []))

    def test_an_answer_that_is_not_a_listing_is_said(self):
        for text in ("", "not json", "[1, 2]"):
            with self.subTest(text=text), self.assertRaises(camera_list.CameraListError):
                camera_list.parse(text)
        self.assertEqual(camera_list.parse("{}"), [])

    def test_the_identifier_is_kept_out_of_what_is_printed(self):
        cameras = camera_list.parse(listing(OWN, FRONT_ON_ITS_HUB, WRIST_ON_THE_ARM_S_HUB))
        told = " ".join(camera_list.lines(cameras, 0, 1) + sum(camera_list.problems(cameras, 1, 0), []))
        for _name, unique in (OWN, FRONT_ON_ITS_HUB, WRIST_ON_THE_ARM_S_HUB):
            self.assertNotIn(unique, told)


if __name__ == "__main__":
    unittest.main()
