"""The wrist camera's place in its mount, read from its own picture (`rig/wrist_place.py`): drawn pictures, no camera.

2026-10-08: the wrist camera's mount caught the basket's rim, and the gripper's fixed jaw stood 3 px further left in
that camera's picture afterwards. The jaw is rigid with the camera, so the camera had moved.
"""

import importlib.util
import unittest

import numpy as np

HAVE_CV2 = importlib.util.find_spec("cv2") is not None
FLOOR, ORANGE, DARK = (200, 200, 200), (20, 110, 240), (40, 40, 60)


def picture(moved=(0.0, 0.0), hole=True, jaw=True):
    """A wrist picture: the fixed jaw at the bottom left, its inner edge at 273, 265 and 257 px at rows 420, 440 and
    460, its lower hole at (193, 455); the whole moved by `moved` pixels."""
    import cv2

    frame = np.full((480, 640, 3), FLOOR, dtype=np.uint8)
    dx, dy = moved
    if jaw:
        corners = np.array([[309 + dx, 330 + dy], [249 + dx, 480 + dy + 20], [120 + dx, 480 + dy + 20],
                            [250 + dx, 330 + dy]])
        corners[1, 0] = 273 - 0.4 * (500 - 420) + dx          # the inner edge runs on below the picture
        cv2.fillPoly(frame, [corners.round().astype(np.int32)], ORANGE)
        if hole:
            cv2.circle(frame, (round(193 + dx), round(455 + dy)), 4, DARK, -1)
    return frame


@unittest.skipUnless(HAVE_CV2, "needs OpenCV (the rig's interpreter)")
class WristPlaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tolerance_radius.rig import wrist_place

        cls.wp = wrist_place
        cls.reference = wrist_place.read_jaw(picture())

    def against(self, **changes):
        now = self.wp.read_jaw(picture(**changes), self.reference.hole)
        return now, self.wp.offset(self.reference, now)

    def test_the_jaw_is_read_where_it_was_drawn(self):
        self.assertAlmostEqual(self.reference.edge_px, 265.0, delta=1.0)
        self.assertAlmostEqual(self.reference.lean_px, -16.0, delta=1.5)
        self.assertAlmostEqual(self.reference.hole[0], 193.0, delta=0.7)
        self.assertAlmostEqual(self.reference.hole[1], 455.0, delta=0.7)

    def test_a_camera_where_it_was_is_in_place(self):
        now, off = self.against()
        self.assertTrue(off.in_place)
        self.assertEqual(self.wp.said(now, off)[-1], "IN PLACE (within 1 px of where it stood in training)")

    def test_a_camera_that_was_knocked_shows_the_jaw_elsewhere_and_which_way_it_must_come(self):
        # as on 2026-10-08: three pixels to the left and nearly one down
        now, off = self.against(moved=(-3.0, 1.0))
        self.assertAlmostEqual(off.across_px, -3.0, delta=0.4)
        self.assertAlmostEqual(off.down_px, 1.0, delta=0.4)
        self.assertAlmostEqual(off.lean_px, 0.0, delta=0.5)
        self.assertFalse(off.in_place)
        lines = self.wp.said(now, off)
        self.assertRegex(lines[1], r"^the jaw must come [23]\.\d px to the RIGHT$")
        self.assertEqual(len(lines), 2, "one pixel down is within the limit")
        now, off = self.against(moved=(2.0, -3.0))
        lines = self.wp.said(now, off)
        self.assertRegex(lines[1], r"^the jaw must come [12]\.\d px to the LEFT$")
        self.assertRegex(lines[2], r"^the jaw must come [23]\.\d px DOWN$")

    def test_a_jaw_that_cannot_be_seen_is_said_and_not_guessed(self):
        self.assertIsNone(self.wp.read_jaw(picture(jaw=False)))
        self.assertIn("cannot be seen", self.wp.said(None, None)[0])

    def test_without_its_hole_the_jaw_is_judged_by_its_edge_alone(self):
        now, off = self.against(hole=False)
        self.assertIsNone(now.hole)
        self.assertIsNone(off.down_px)
        self.assertTrue(off.in_place)
        self.assertIn("? px down", self.wp.said(now, off)[0])
        now, off = self.against(hole=False, moved=(-3.0, 0.0))
        self.assertFalse(off.in_place)
        self.assertAlmostEqual(off.across_px, -3.0, delta=0.4)

    def test_a_movement_down_is_not_read_as_one_across(self):
        # the inner edge leans: read at fixed rows, three rows up look like 1.2 px across. The hole is a point.
        now, off = self.against(moved=(0.0, -3.0))
        self.assertAlmostEqual(off.across_px, 0.0, delta=0.4)
        self.assertAlmostEqual(off.down_px, -3.0, delta=0.4)
        self.assertAlmostEqual(now.edge_px - self.reference.edge_px, -1.2, delta=0.5)

    def test_a_hole_far_from_where_it_belongs_is_not_taken_for_it(self):
        # the cube in the jaws, a shadow, a screw: only a patch near the reference's hole is the hole
        frame = picture(hole=False)
        import cv2

        cv2.circle(frame, (225, 415), 4, DARK, -1)
        self.assertIsNone(self.wp.read_jaw(frame, self.reference.hole).hole)

    def test_the_old_picture_of_the_jaw_is_laid_over_the_live_one(self):
        old, frame = picture(), picture(moved=(-6.0, 0.0))
        both = self.wp.laid_over(frame, old)
        (top, bottom), (left, right) = self.wp.OVER_ROWS, self.wp.OVER_COLUMNS
        self.assertTrue(np.array_equal(both[:top], frame[:top]))
        self.assertTrue(np.array_equal(both[:, right:], frame[:, right:]), "the moving jaw's side is left as it is")
        self.assertTrue(np.array_equal(frame, picture(moved=(-6.0, 0.0))), "the frame itself is not drawn on")
        self.assertFalse(np.array_equal(both[top:bottom, left:right], frame[top:bottom, left:right]))
        alone = self.wp.laid_over(frame, old, share=1.0)
        inside = (slice(top + 1, bottom - 1), slice(left + 1, right - 1))
        self.assertTrue(np.array_equal(alone[inside], old[inside]))
        marked = self.wp.draw(frame, self.reference, *self.against(moved=(-6.0, 0.0)), ["a line"])
        self.assertFalse(np.array_equal(marked, frame))

    def test_the_live_reading_is_the_median_of_the_last_looks(self):
        readings = [self.wp.JawReading(edge_px=edge, lean_px=-16.0, hole=(193.0, 455.0 + k))
                    for k, edge in enumerate((264.7, 265.0, 265.0, 265.3, 280.0))]
        steady = self.wp.steady(readings)
        self.assertEqual((steady.edge_px, steady.hole), (265.0, (193.0, 457.0)))
        self.assertIsNone(self.wp.steady([]))

    def test_the_tool_has_a_mode_for_it_and_a_reference_picture(self):
        import cv2

        from tolerance_radius.rig import camera_tool

        args = camera_tool.build_parser().parse_args(["wristplace"])
        self.assertEqual((args.mode, args.reference), ("wristplace", None))
        kept = cv2.imread(str(camera_tool.WRIST_REFERENCE))
        self.assertIsNotNone(kept, "the picture that says where the wrist camera belongs")
        reading = self.wp.read_jaw(kept)
        # where the fixed jaw stands in both training sets (medians of 265.0 and 264.7; the hole within half a pixel)
        self.assertAlmostEqual(reading.edge_px, 265.0, delta=0.5)
        self.assertAlmostEqual(reading.hole[0], 193.3, delta=1.0)
        self.assertAlmostEqual(reading.hole[1], 455.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
