"""The basket's place in the front camera's picture (`rig/basket_place.py`): drawn pictures, no camera.

2026-10-08: the wrist camera's mount caught the basket's rim during an attempt and carried it four centimetres. The
basket stands on tape and has no stop to go back to, so its place is read from the picture.
"""

import importlib.util
import unittest

import numpy as np

HAVE_CV2 = importlib.util.find_spec("cv2") is not None
FLOOR, GREY = (200, 200, 200), (70, 70, 70)
TAGS = {0: (71.0, 168.0), 1: (514.0, 368.0)}


def quad(centre, half=30.0):
    x, y = centre
    return np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])


def scene(basket=(0.0, 0.0), camera=(0.0, 0.0), rim_drop=0.0):
    """A floor with the basket in its upper right corner, moved by `basket` pixels, the whole picture (basket and
    tags) moved by `camera`, and the near rim's right end lowered by `rim_drop` pixels. Returns the frame and the
    tags as a detector would give them."""
    import cv2

    frame = np.full((480, 640, 3), FLOOR, dtype=np.uint8)
    dx, dy = basket[0] + camera[0], basket[1] + camera[1]
    corners = np.array([[444 + dx, -20 + dy], [700 + dx, -20 + dy], [700 + dx, 222 + dy + rim_drop],
                        [462 + dx, 240 + dy]])
    cv2.fillPoly(frame, [corners.round().astype(np.int32)], GREY)
    found = {tag: quad((x + camera[0], y + camera[1])) for tag, (x, y) in TAGS.items()}
    return frame, found


@unittest.skipUnless(HAVE_CV2, "needs OpenCV (the rig's interpreter)")
class BasketPlaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tolerance_radius.rig import basket_place

        cls.bp = basket_place
        cls.reference = basket_place.read_basket(*scene())

    def against(self, **changes):
        now = self.bp.read_basket(*scene(**changes))
        return self.bp.offset(self.reference, now, mm_per_px=1.0)

    def test_a_basket_where_it_was_is_in_place(self):
        off = self.against()
        self.assertTrue(off.in_place)
        self.assertEqual(self.bp.advice(off), [])
        self.assertEqual(self.bp.said(self.reference, self.reference, off)[-1], "IN PLACE")

    def test_how_far_it_stands_from_its_place_and_which_way_to_move_it(self):
        # The wall and the rim lean, so a basket moved one way reads a little the other way as well (about a pixel
        # for ten): the figures are for bringing it home, where both are nought, and not for measuring.
        off = self.against(basket=(10.0, -6.0))
        self.assertAlmostEqual(off.across_mm, 10.0, delta=1.2)
        self.assertAlmostEqual(off.down_mm, -6.0, delta=1.2)
        self.assertFalse(off.in_place)
        # too far to the right and too far up in the picture: back to the left and down
        across, down = self.bp.advice(off)
        self.assertRegex(across, r"^move it (9|10|11) mm to the LEFT$")
        self.assertRegex(down, r"^move it [567] mm DOWN \(towards the camera\)$")
        across, down = self.bp.advice(self.against(basket=(-8.0, 5.0)))
        self.assertRegex(across, r"^move it [789] mm to the RIGHT$")
        self.assertRegex(down, r"^move it [456] mm UP \(away from the camera\)$")

    def test_the_camera_s_own_movement_is_taken_off(self):
        # the whole picture three pixels on, basket and tags alike: the camera moved, the basket did not
        off = self.against(camera=(3.0, 2.0))
        self.assertTrue(off.in_place, off)
        self.assertAlmostEqual(off.camera_px[0], 3.0, delta=0.01)
        self.assertAlmostEqual(off.camera_px[1], 2.0, delta=0.01)
        self.assertAlmostEqual(off.across_mm, 0.0, delta=0.6)
        # ... and a basket that moved while the camera did reads its own movement alone
        both = self.against(basket=(10.0, 0.0), camera=(3.0, 2.0))
        self.assertAlmostEqual(both.across_mm, 10.0, delta=0.6)

    def test_without_a_tag_the_reading_says_that_the_camera_is_not_taken_off(self):
        frame, _found = scene(basket=(10.0, 0.0))
        now = self.bp.read_basket(frame, {})
        off = self.bp.offset(self.reference, now, mm_per_px=1.0)
        self.assertIsNone(off.camera_px)
        self.assertIn("no floor tag in view", self.bp.said(self.reference, now, off)[0])

    def test_a_basket_that_was_turned_is_told_to_be_turned_back(self):
        off = self.against(rim_drop=8.0)        # the rim's right end eight pixels lower than it was
        self.assertGreater(off.lean, self.bp.IN_PLACE_LEAN)
        self.assertFalse(off.in_place)
        self.assertIn("turn it: its right end UP in the picture", self.bp.advice(off))
        self.assertIn("turn it: its right end DOWN in the picture", self.bp.advice(self.against(rim_drop=-8.0)))

    def test_the_limit_is_two_millimetres_each_way(self):
        self.assertTrue(self.against(basket=(1.0, -1.0)).in_place)
        self.assertFalse(self.against(basket=(4.0, 0.0)).in_place)
        self.assertFalse(self.against(basket=(0.0, 4.0)).in_place)
        self.assertEqual(self.bp.IN_PLACE_MM, 2.0)

    def test_a_basket_that_cannot_be_seen_is_said_and_not_guessed(self):
        bare = np.full((480, 640, 3), FLOOR, dtype=np.uint8)
        self.assertIsNone(self.bp.read_basket(bare, {}))
        self.assertIn("cannot be seen", self.bp.said(self.reference, None, None)[0])
        # an orange arm or a hand in front of the floor is not the basket
        import cv2

        cv2.rectangle(bare, (440, 60), (640, 240), (20, 110, 240), -1)
        self.assertIsNone(self.bp.read_basket(bare, {}))

    def test_millimetres_per_pixel_come_from_the_floor_tags(self):
        geometry = {"tags": {"0": {"centre": [318.05, 23.05]}, "1": {"centre": [-150.94, 263.15]}}}
        found = {tag: quad(centre) for tag, centre in TAGS.items()}
        apart_px = float(np.hypot(514.0 - 71.0, 368.0 - 168.0))
        self.assertAlmostEqual(self.bp.mm_per_px(found, geometry), 526.88 / apart_px, places=3)
        self.assertEqual(self.bp.mm_per_px({}, geometry), self.bp.MM_PER_PX)
        self.assertEqual(self.bp.mm_per_px(found, None), self.bp.MM_PER_PX)

    def test_the_picture_shows_where_the_edges_belong_and_what_to_do(self):
        frame, found = scene(basket=(12.0, 0.0))
        now = self.bp.read_basket(frame, found)
        off = self.bp.offset(self.reference, now, mm_per_px=1.0)
        lines = self.bp.said(self.reference, now, off)
        picture = self.bp.draw(frame, self.reference, now, off, lines)
        self.assertEqual(picture.shape, frame.shape)
        self.assertFalse(np.array_equal(picture, frame))
        self.assertTrue(np.array_equal(frame, scene(basket=(12.0, 0.0))[0]), "the frame itself is left as it came")
        green = (picture[..., 1] > 240) & (picture[..., 0] < 120) & (picture[..., 2] < 120)
        self.assertGreater(int(green[:260, 380:].sum()), 100, "the green lines of the basket's place")

    def test_the_old_basket_itself_is_laid_over_the_live_picture(self):
        """The operator, 2026-10-08: "more than just supplying the lines, can you provide an actual image overlay of
        the correct basket so we can compare"."""
        old, _found = scene()
        frame, found = scene(basket=(12.0, 0.0))
        off = self.bp.offset(self.reference, self.bp.read_basket(frame, found), mm_per_px=1.0)
        both = self.bp.ghosted(frame, old, self.reference, off.camera_px)
        rows, columns = self.bp.ghost_region(self.reference, frame.shape)
        self.assertLess(columns.start, 444 - 20)
        self.assertGreater(rows.stop, 240 + 20)
        # outside the part that is laid over, the live picture is as it came, and the frame itself is not drawn on
        self.assertTrue(np.array_equal(both[rows.stop:, :], frame[rows.stop:, :]))
        self.assertTrue(np.array_equal(both[:, :columns.start], frame[:, :columns.start]))
        self.assertTrue(np.array_equal(frame, scene(basket=(12.0, 0.0))[0]))
        # a basket twelve pixels out of place is seen twice: where the old one was and the new one is not, half grey
        strip = both[100, 456:462, 0].astype(int)
        self.assertTrue(((strip > 120) & (strip < 150)).all(), strip)
        self.assertEqual(int(both[100, 500, 0]), GREY[0], "where both are, the basket as it is")
        # the old one alone: the part laid over is the reference frame's
        alone = self.bp.ghosted(frame, old, self.reference, off.camera_px, share=1.0)
        inside = (slice(rows.start + 1, rows.stop - 1), slice(columns.start + 1, columns.stop - 1))
        self.assertTrue(np.array_equal(alone[inside], old[inside]))

    def test_the_old_basket_is_laid_over_where_the_camera_now_sees_the_floor(self):
        # the camera three pixels on since the reference: the old picture is moved with it, and a basket that has
        # not moved lies on itself
        old, _found = scene()
        frame, found = scene(camera=(3.0, 2.0))
        off = self.bp.offset(self.reference, self.bp.read_basket(frame, found), mm_per_px=1.0)
        both = self.bp.ghosted(frame, old, self.reference, off.camera_px)
        rows, columns = self.bp.ghost_region(self.reference, frame.shape)
        inside = (slice(rows.start + 4, rows.stop - 4), slice(columns.start + 4, columns.stop - 4))

        def seen_twice(picture) -> int:
            """How many pixels of the part laid over differ from the live picture: the basket's edges seen twice."""
            return int((np.abs(picture[inside].astype(int) - frame[inside].astype(int)).max(axis=2) > 2).sum())

        # (a drawn edge steps a pixel differently here and there when it is drawn two rows lower: a few dozen pixels)
        self.assertLess(seen_twice(both), 60)
        unmoved = self.bp.ghosted(frame, old, self.reference, None)
        self.assertGreater(seen_twice(unmoved), 600, "the old picture laid over without the camera's movement")
        self.assertEqual(len(self.bp.VIEWS), 3)

    def test_the_tool_has_a_mode_for_it_and_a_reference_frame(self):
        from tolerance_radius.rig import camera_tool

        args = camera_tool.build_parser().parse_args(["basket"])
        self.assertEqual((args.mode, args.reference), ("basket", None))
        self.assertTrue(camera_tool.BASKET_REFERENCE.is_file(), "the frame that says where the basket belongs")
        import cv2

        kept = cv2.imread(str(camera_tool.BASKET_REFERENCE))
        reading = self.bp.read_basket(kept, {})
        self.assertIsNotNone(reading, "the basket is found in the reference frame")
        self.assertAlmostEqual(reading.wall_px, 444.0, delta=1.0)
        self.assertAlmostEqual(reading.rim_px, 240.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
