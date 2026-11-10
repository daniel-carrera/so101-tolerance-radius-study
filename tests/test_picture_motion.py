"""The analysis that holds two pictures of the table against each other (`analysis/instrument/picture_motion.py`).

It decides whether a calibration that was refused means a moved camera or a moved floor tag, so what it says of
pictures whose motion is known is tested here. The pictures are made up: a texture where the rig's picture has its
pipes, arm and basket, a blank floor below, moved by a known shift and turn.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from fixtures_pictures import RIG_FLOOR, RIG_VIEW, rectangle_mm, to_px

try:
    import cv2

    READY = True
except ImportError:  # the development environment has no OpenCV
    READY = False

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "instrument" / "picture_motion.py"
CENTRE = (320.0, 240.0)


def load_script():
    spec = importlib.util.spec_from_file_location("picture_motion", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry() -> dict:
    """A table geometry as the registration stores it, for the rig's view of 2026-10-05."""
    corners = {str(tag): to_px(RIG_VIEW, rectangle_mm(*RIG_FLOOR[tag])).tolist() for tag in RIG_FLOOR}
    return {"tag_mm": 60.0, "grid": {"cols": 5, "rows": 3, "cell_mm": 50.0}, "created": "2026-10-05T22:09:50-07:00",
            "registration": {"px_to_mm": np.linalg.inv(RIG_VIEW).tolist(), "floor_tag_corners_px": corners}}


_WAVES = None


def scene_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The scene as a function of the place in it: a texture of sixty waves, 9 to 40 px long, over the top third of
    the picture (as the pipes, the arm and the basket give), fading into a blank floor below. A function, so that a
    moved picture is the same scene read at other places and its motion is exact: a picture warped by interpolation
    carried an error of 0.05 px in the tool that warps."""
    global _WAVES
    if _WAVES is None:
        rng = np.random.default_rng(3)
        length = rng.uniform(9.0, 40.0, 60)
        angle = rng.uniform(0.0, np.pi, 60)
        _WAVES = (2 * np.pi / length * np.cos(angle), 2 * np.pi / length * np.sin(angle),
                  rng.uniform(0.0, 2 * np.pi, 60), rng.normal(0.0, 1.0, 60))
    kx, ky, phase, height = _WAVES
    texture = sum(h * np.sin(a * x + b * y + c) for a, b, c, h in zip(kx, ky, phase, height, strict=True))
    fade = 1.0 / (1.0 + np.exp((y - 150.0) / 3.0))
    return 200.0 + fade * (-72.0 + 9.0 * texture)


def scene(dx: float = 0.0, dy: float = 0.0, degrees: float = 0.0, growth: float = 0.0) -> np.ndarray:
    """A grey picture, 480 x 640, of the scene with its content shifted, turned about the picture's centre (positive
    from x towards y) and made larger about it by `growth` (a share)."""
    a = np.radians(degrees)
    cx, cy = CENTRE
    px, py = np.meshgrid(np.arange(640.0), np.arange(480.0))
    u, v = px - cx - dx, py - cy - dy                       # back through the motion: where this pixel's content was
    x = (np.cos(a) * u + np.sin(a) * v) / (1.0 + growth) + cx
    y = (-np.sin(a) * u + np.cos(a) * v) / (1.0 + growth) + cy
    return np.clip(scene_at(x, y), 0.0, 255.0).astype(np.float32)


@unittest.skipUnless(READY, "needs OpenCV (run with the rig's interpreter)")
class PictureMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()
        cls.geometry = geometry()
        cls.before = scene()
        cls.mask = cls.script.left_out(cls.before.shape, cls.geometry, False)

    def motion(self, after):
        patches = self.script.patch_shifts(self.before, after, self.mask)
        solution, left, keep = self.script.robust_motion(patches, CENTRE)
        return patches, solution, left, keep

    def test_a_shift_a_turn_and_a_growth_fitted_to_exact_patch_shifts_are_given_back(self):
        points = np.array([(100.0, 30.0), (300.0, 40.0), (600.0, 60.0), (480.0, 210.0), (200.0, 90.0)])
        truth = (-0.8, -0.5, np.radians(-0.1), 0.0007)
        shifts = self.script.motion_at(truth, points, CENTRE)
        solution, left = self.script.fit_motion(points, shifts, CENTRE)
        np.testing.assert_allclose(solution, truth, atol=1e-9)
        self.assertLess(np.abs(left).max(), 1e-9)
        # The signs: a positive turn carries a point right of the centre downwards in the picture; a positive growth
        # carries it further right.
        self.assertGreater(self.script.motion_at((0.0, 0.0, 0.01, 0.0), [(420.0, 240.0)], CENTRE)[0][1], 0.0)
        self.assertGreater(self.script.motion_at((0.0, 0.0, 0.0, 0.001), [(420.0, 240.0)], CENTRE)[0][0], 0.0)
        # Without the size, a picture that grew is fitted by a shift and a turn alone, and something is left over.
        solution, left = self.script.fit_motion(points, shifts, CENTRE, scale=False)
        self.assertEqual(solution[3], 0.0)
        self.assertGreater(np.abs(left).max(), 0.05)

    def test_a_camera_that_came_closer_is_read_as_a_larger_picture(self):
        # 14:12 on 2026-10-06: the picture had moved by 1.2 px and grown by 0.07 %, and the floor tags with it. Read
        # as a shift and a turn alone, the tags seemed to have moved half a pixel against the rest.
        patches, solution, left, keep = self.motion(scene(-0.04, 1.17, 0.013, 0.0007))
        self.assertAlmostEqual(100 * solution[3], 0.07, delta=0.012)
        self.assertAlmostEqual(solution[1], 1.17, delta=0.05)
        tag = np.asarray(self.geometry["registration"]["floor_tag_corners_px"]["1"]).mean(axis=0)
        truth = self.script.motion_at((-0.04, 1.17, np.radians(0.013), 0.0007), [tag], CENTRE)[0]
        self.assertLess(np.linalg.norm(self.script.motion_at(solution, [tag], CENTRE)[0] - truth), 0.12)

    def test_the_motion_of_the_rig_s_first_calibration_is_read_from_the_pictures(self):
        # 2026-10-06: the camera had turned by a tenth of a degree and its picture moved by about a pixel.
        patches, solution, left, keep = self.motion(scene(-0.81, -0.48, -0.098))
        self.assertGreater(len(patches), 20)
        # A patch with little texture, where the scene fades into the floor, can be read wrongly by a pixel: the
        # fit leaves it out.
        self.assertLessEqual(int((~keep).sum()), 2)
        self.assertAlmostEqual(solution[0], -0.81, delta=0.03)
        self.assertAlmostEqual(solution[1], -0.48, delta=0.03)
        self.assertAlmostEqual(np.degrees(solution[2]), -0.098, delta=0.005)
        self.assertLess(np.sqrt((left ** 2).sum(axis=1).mean()), 0.05)

    def test_two_pictures_with_nothing_done_between_them_read_nothing(self):
        noisy = self.before + np.random.default_rng(1).normal(0.0, 1.2, self.before.shape).astype(np.float32)
        patches, solution, _left, _keep = self.motion(noisy)
        self.assertLess(abs(solution[0]) + abs(solution[1]), 0.03)
        self.assertLess(abs(np.degrees(solution[2])), 0.005)

    def test_a_floor_tag_that_moved_while_the_camera_stayed_is_not_explained_by_the_picture(self):
        # The case the calibration must not pass: the rest of the picture has not moved, so nothing of the tags'
        # shifts is explained, and all of what they differ by is left over.
        patches, solution, _left, _keep = self.motion(self.before.copy())
        tag = np.asarray(self.geometry["registration"]["floor_tag_corners_px"]["1"]).mean(axis=0)
        expected = self.script.motion_at(solution, [tag], CENTRE)[0]
        self.assertLess(np.linalg.norm(expected), 0.02)
        self.assertGreater(np.linalg.norm(np.array([0.61, -0.79]) - expected), 0.95)

    def test_something_moved_by_hand_is_left_out_of_the_one_motion(self):
        # The basket pushed, or the arm parked elsewhere: its patches do not go with the rest.
        after = scene(-0.81, -0.48, -0.098)
        pushed = scene(-0.81 + 2.5, -0.48 + 1.5, -0.098)
        after[0:120, 470:640] = pushed[0:120, 470:640]
        patches, solution, _left, keep = self.motion(after)
        self.assertGreater(int((~keep).sum()), 3)
        self.assertGreater(int(keep.sum()), 10)
        self.assertAlmostEqual(solution[0], -0.81, delta=0.08)
        self.assertAlmostEqual(np.degrees(solution[2]), -0.098, delta=0.03)

    def test_a_blank_floor_and_a_straight_edge_give_no_patch(self):
        blank = np.full((480, 640), 200.0, dtype=np.float32)
        self.assertEqual(self.script.patch_shifts(blank, blank, np.zeros(blank.shape, np.uint8)), [])
        pipe = blank.copy()
        pipe[30:40] = 255.0                      # a bare pipe: texture one way only
        self.assertEqual(self.script.patch_shifts(pipe, pipe, np.zeros(pipe.shape, np.uint8)), [])

    def test_the_sheet_s_place_and_the_floor_tags_are_left_out(self):
        registered = self.geometry["registration"]
        middle = self.script.to_px(registered["px_to_mm"], [(125.0, 75.0)])[0]
        self.assertTrue(self.mask[int(middle[1]), int(middle[0])])
        for corners in registered["floor_tag_corners_px"].values():
            x, y = np.asarray(corners).mean(axis=0)
            self.assertTrue(self.mask[int(y), int(x)])
        self.assertFalse(self.mask[20, 320])
        # A marked picture also has the band a tag's label is written in.
        marked = self.script.left_out(self.before.shape, self.geometry, True)
        self.assertGreater(int((marked > 0).sum()), int((self.mask > 0).sum()))

    def test_a_calibration_solved_right_lies_on_the_registered_frame_carried_along(self):
        # The second opinion on a stored calibration: with the view moved exactly as the picture moved, nothing is
        # left; a view that is the registered one, as if the camera had not moved, is off by the motion itself.
        truth = (-0.81, -0.48, np.radians(-0.098), 0.0)
        a = truth[2]
        cx, cy = CENTRE
        turn = np.array([[np.cos(a), -np.sin(a), cx - cx * np.cos(a) + cy * np.sin(a) + truth[0]],
                         [np.sin(a), np.cos(a), cy - cx * np.sin(a) - cy * np.cos(a) + truth[1]],
                         [0.0, 0.0, 1.0]])
        right = {"px_to_mm": np.linalg.inv(turn @ RIG_VIEW).tolist()}
        mean_mm, worst_mm, distance_mm, share = self.script.stored_view_against(self.geometry, right, truth, CENTRE)
        self.assertLess(worst_mm, 0.01)
        self.assertLess(distance_mm, 0.01)
        stale = {"px_to_mm": np.linalg.inv(RIG_VIEW).tolist()}
        mean_mm, worst_mm, _distance, _share = self.script.stored_view_against(self.geometry, stale, truth, CENTRE)
        self.assertGreater(mean_mm, 0.7)


if __name__ == "__main__":
    unittest.main()
