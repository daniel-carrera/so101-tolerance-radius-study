"""The measuring instrument, on synthetic pictures: an orange square on a plain table, with a known scale.

These tests need OpenCV and the arm tooling's scripts, so they run with the rig's interpreter and skip elsewhere.
"""

import unittest

import numpy as np

try:
    import cv2
except ImportError:  # the development environment has no OpenCV
    cv2 = None

from tolerance_radius.schema import Point

if cv2 is not None:
    from tolerance_radius.rig import lerobot101
    from tolerance_radius.rig.placement_phase import (
        KEY_ABORT,
        KEY_ACCEPT,
        KEY_SKIP,
        PlacementPhase,
        assess,
        consume_go,
        render,
    )
    from tolerance_radius.rig.vision import TableView, VisionError, empty_scene, median_position

    READY = lerobot101.available()
else:
    READY = False

#: Two pixels per millimetre, table origin at pixel (100, 60): easy to check by hand.
PX_TO_MM = np.array([[0.5, 0.0, -50.0], [0.0, 0.5, -30.0], [0.0, 0.0, 1.0]])
ORANGE_BGR = (0, 128, 255)
TABLE_BGR = (150, 150, 150)
CUBE_PX = 60  # a 30 mm cube at this scale


def scene(*cubes_mm: Point) -> np.ndarray:
    """A 640 x 480 picture of the table with a cube centred on each given table point."""
    picture = np.full((480, 640, 3), TABLE_BGR, np.uint8)
    for cube in cubes_mm:
        cx, cy = cube.x_mm * 2 + 100, cube.y_mm * 2 + 60
        top_left = (int(round(cx - CUBE_PX / 2)), int(round(cy - CUBE_PX / 2)))
        bottom_right = (top_left[0] + CUBE_PX - 1, top_left[1] + CUBE_PX - 1)
        cv2.rectangle(picture, top_left, bottom_right, ORANGE_BGR, -1)
    return picture


class InterpreterFolderOnPathTests(unittest.TestCase):
    """The viewer a session spawns is installed beside the rig's interpreter, which the runbooks start by its full
    path: its folder has to be on PATH. The rig check of 2026-10-06 stopped for want of it."""

    def test_the_interpreter_s_folder_is_put_first_on_path_once(self):
        import os
        import sys
        from pathlib import Path
        from unittest import mock

        from tolerance_radius.rig import lerobot101

        folder = str(Path(sys.executable).parent)
        with mock.patch.dict(os.environ, {"PATH": os.pathsep.join(["/usr/bin", "/bin"])}):
            lerobot101.put_interpreter_folder_on_path()
            self.assertEqual(os.environ["PATH"].split(os.pathsep), [folder, "/usr/bin", "/bin"])
            lerobot101.put_interpreter_folder_on_path()
            self.assertEqual(os.environ["PATH"].split(os.pathsep).count(folder), 1)
        with mock.patch.dict(os.environ, {"PATH": ""}):
            lerobot101.put_interpreter_folder_on_path()
            self.assertEqual(os.environ["PATH"], folder)


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class TableViewTests(unittest.TestCase):
    def setUp(self):
        self.view = TableView(PX_TO_MM, scene())

    def test_pixels_and_millimetres_convert_both_ways(self):
        point = Point(75.0, 25.0)
        self.assertEqual(self.view.to_px(point), (250.0, 110.0))
        back = self.view.to_mm(self.view.to_px(point))
        self.assertAlmostEqual(back.x_mm, 75.0)
        self.assertAlmostEqual(back.y_mm, 25.0)
        self.assertAlmostEqual(self.view.px_per_mm_at(point), 2.0)

    def test_it_measures_a_cube_where_it_is(self):
        for cube in (Point(75.0, 25.0), Point(131.5, 88.0), Point(20.0, 140.0)):
            with self.subTest(cube=cube):
                measured = self.view.measure(scene(cube))
                self.assertIsNotNone(measured)
                self.assertLess(measured.position.distance_to(cube), 0.5)

    def test_a_move_of_ten_millimetres_measures_as_ten(self):
        first = self.view.measure(scene(Point(60.0, 40.0))).position
        second = self.view.measure(scene(Point(70.0, 40.0))).position
        self.assertAlmostEqual(first.distance_to(second), 10.0, delta=0.3)

    def test_no_cube_and_two_cubes_are_both_refused(self):
        self.assertIsNone(self.view.measure(scene()))
        self.assertIsNone(self.view.measure(scene(Point(40.0, 40.0), Point(160.0, 100.0))))

    def test_measuring_needs_a_background_of_the_same_size(self):
        with self.assertRaises(VisionError):
            TableView(PX_TO_MM).measure(scene(Point(75.0, 25.0)))
        with self.assertRaises(VisionError):
            self.view.measure(np.zeros((240, 320, 3), np.uint8))

    def test_a_transform_that_cannot_be_inverted_is_refused(self):
        with self.assertRaises(VisionError):
            TableView(np.zeros((3, 3)))

    def test_from_frame_refuses_a_picture_with_no_tags(self):
        geometry = {"tag_mm": 60.0, "tags": {"0": {"corners": [[0, 0], [60, 0], [60, 60], [0, 60]]}}}
        with self.assertRaises(VisionError):
            TableView.from_frame(scene(), geometry)

    def test_the_empty_scene_is_the_median_of_the_frames(self):
        frames = iter([scene(), scene(Point(75.0, 25.0)), scene()])
        background = empty_scene(lambda: next(frames), frames=3)
        self.assertTrue(np.array_equal(background, scene()))

    def test_the_median_position_ignores_one_wild_reading(self):
        readings = [self.view.measure(scene(p)) for p in (Point(75.0, 25.0), Point(75.5, 25.0), Point(120.0, 90.0))]
        middle = median_position(readings)
        self.assertLess(middle.distance_to(Point(75.5, 25.0)), 0.6)


def august_geometry() -> dict:
    """A real table geometry to test with: the one the earlier project measured in August, at its pinned commit.

    This rig's own geometry is measured at the rig and may not exist where the tests run.
    """
    from tolerance_radius.rig.vision import load_geometry

    return load_geometry(lerobot101.root() / "eval" / "table_geometry.json")


def tagged_scene(geometry: dict, origin_px=(320.0, 200.0), shift_tag_mm: dict | None = None) -> np.ndarray:
    """A 640 x 480 picture with both tags drawn where the table geometry says, at one pixel per millimetre, y up.

    `shift_tag_mm` moves a tag on the table without telling the geometry, for example `{"1": (6.0, 0.0)}`.
    """
    tag_check = lerobot101.load("tag_check")
    dictionary = cv2.aruco.getPredefinedDictionary(tag_check.DICT)
    picture = np.full((480, 640, 3), 200, np.uint8)
    side = 200
    square = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
    for tag_id, tag in geometry["tags"].items():
        dx, dy = (shift_tag_mm or {}).get(tag_id, (0.0, 0.0))
        corners_px = np.array(
            [(origin_px[0] + x + dx, origin_px[1] - (y + dy)) for x, y in np.array(tag["corners"], float)], np.float32
        )
        warp = cv2.getPerspectiveTransform(square, corners_px)
        marker = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, int(tag_id), side), cv2.COLOR_GRAY2BGR)
        drawn = cv2.warpPerspective(marker, warp, (640, 480), borderValue=(200, 200, 200))
        where = cv2.warpPerspective(np.full((side, side), 255, np.uint8), warp, (640, 480)) > 0
        picture[where] = drawn[where]
    return picture


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class TableFrameFromTagsTests(unittest.TestCase):
    """The whole chain on the rig's real geometry file: tags drawn into a picture, detected, and the frame placed."""

    def setUp(self):
        from tolerance_radius.rig.vision import grid_probes_px

        self.geometry = august_geometry()
        self.grid_probes_px = grid_probes_px

    def test_the_table_frame_is_recovered_from_pictures_of_both_tags(self):
        view = TableView.from_frames([tagged_scene(self.geometry)] * 3, self.geometry)
        for point in (Point(0, 0), Point(125, 75), Point(250, 150), Point(75, 25)):
            with self.subTest(point=point):
                x_px, y_px = view.to_px(point)
                self.assertAlmostEqual(x_px, 320.0 + point.x_mm, delta=3.0)
                self.assertAlmostEqual(y_px, 200.0 - point.y_mm, delta=3.0)
        self.assertAlmostEqual(view.px_per_mm_at(Point(125, 75)), 1.0, delta=0.03)
        self.assertEqual(view.frames_used, 3)
        self.assertLess(view.residual_mm, 1.0)

    def test_a_tag_moved_by_centimetres_shows_in_the_fit(self):
        from tolerance_radius.rig.camera_tool import RESIDUAL_LIMIT_MM

        still = TableView.from_frames([tagged_scene(self.geometry)], self.geometry)
        moved = TableView.from_frames([tagged_scene(self.geometry, shift_tag_mm={"1": (40.0, 30.0)})], self.geometry)
        self.assertLess(still.residual_mm, RESIDUAL_LIMIT_MM / 2)
        self.assertGreater(moved.residual_mm, RESIDUAL_LIMIT_MM)

    def test_a_tag_moved_by_millimetres_does_not_show_in_the_fit_but_stretches_distances(self):
        """A known limit, pinned so that nobody trusts the fit for more than it can say.

        Two small tags far apart leave a flat-table fit enough freedom to absorb a small move of one of them. The fit
        stays good, and the table frame is quietly stretched: this is why the instrument is checked against a ruler.
        """
        from tolerance_radius.rig.camera_tool import RESIDUAL_LIMIT_MM

        still = TableView.from_frames([tagged_scene(self.geometry)], self.geometry)
        moved = TableView.from_frames([tagged_scene(self.geometry, shift_tag_mm={"1": (8.0, 5.0)})], self.geometry)
        self.assertLess(moved.residual_mm, RESIDUAL_LIMIT_MM)
        start, end = still.to_px(Point(100, 75)), still.to_px(Point(150, 75))
        reads = moved.to_mm(start).distance_to(moved.to_mm(end))
        self.assertGreater(abs(reads - 50.0), 0.3)
        self.assertLess(abs(reads - 50.0), 2.0)

    def test_a_view_is_refused_when_a_tag_is_never_seen(self):
        blank = np.full((480, 640, 3), 200, np.uint8)
        with self.assertRaises(VisionError):
            TableView.from_frames([blank, blank], self.geometry)

    def test_a_stored_view_reads_back_the_same(self):
        import tempfile
        from pathlib import Path

        view = TableView.from_frames([tagged_scene(self.geometry)] * 2, self.geometry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instrument" / "table_view.json"
            view.save(path, note="test")
            loaded = TableView.load(path)
        probes = self.grid_probes_px(view, self.geometry)
        self.assertLess(view.disagreement_mm(loaded, probes), 1e-6)
        self.assertAlmostEqual(loaded.residual_mm, view.residual_mm)
        self.assertEqual(loaded.frames_used, 2)
        with self.assertRaises(VisionError):
            TableView.load(Path(directory) / "missing.json")

    def test_two_views_disagree_by_how_far_the_camera_moved(self):
        here = TableView.from_frames([tagged_scene(self.geometry)], self.geometry)
        shifted = TableView.from_frames([tagged_scene(self.geometry, origin_px=(326.0, 200.0))], self.geometry)
        apart = here.disagreement_mm(shifted, self.grid_probes_px(here, self.geometry))
        self.assertAlmostEqual(apart, 6.0, delta=1.5)


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class StoredCalibrationTests(unittest.TestCase):
    """A session measures through the stored view, and refuses when the live picture no longer matches it."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from unittest import mock

        from tolerance_radius.rig import camera_tool

        self.camera_tool = camera_tool
        self.geometry = august_geometry()
        # A session reads this rig's own geometry; here it is the fixture.
        patcher = mock.patch.object(camera_tool, "load_geometry", lambda path=None: self.geometry)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "table_view.json"
        TableView.from_frames([tagged_scene(self.geometry)] * 4, self.geometry).save(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def test_a_session_measures_through_a_view_solved_now_from_the_tags(self):
        view = self.camera_tool.checked_stored_view(lambda: tagged_scene(self.geometry), path=self.path, frames=3)
        self.assertEqual(view.frames_used, 3)                       # solved from this session's own frames
        # The frame it started on is kept beside the calibration, as it came.
        kept = list(self.path.parent.glob("session_*.png"))
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(cv2.imread(str(kept[0])), tagged_scene(self.geometry)))

    def test_a_camera_that_moved_a_little_is_followed_and_said(self):
        # The rig's camera is carried by the light box's walls and moved by a millimetre or two within an hour of
        # its calibration (2026-10-06). The floor tags did not. Measured through the stored conversion, every
        # position would be off by the camera's movement; through the tags as they are now, none is.
        import contextlib
        import io

        stored = TableView.load(self.path)
        # by whole pixels: this fixture draws its tags by warping, and is only exact on the pixel grid
        moved = lambda: tagged_scene(self.geometry, origin_px=(322.0, 198.0))  # noqa: E731
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            view = self.camera_tool.checked_stored_view(moved, path=self.path, frames=3)
        said = out.getvalue()
        self.assertIn("the table frame is taken from the two floor tags as they are now", said)
        self.assertIn("NOTE: the camera has moved since the calibration", said)
        # A point of the table is read where the calibration read it, though the camera now shows it elsewhere.
        before = stored.to_mm((320.0 + 60.0, 200.0 - 40.0))
        after = view.to_mm((322.0 + 60.0, 198.0 - 40.0))
        self.assertLess(after.distance_to(before), 0.1)
        # What the stored conversion would say of the same pixel: off by the camera's movement.
        self.assertGreater(stored.to_mm((322.0 + 60.0, 198.0 - 40.0)).distance_to(before), 2.5)

    def test_a_camera_moved_far_is_refused(self):
        # Then the picture is no longer the one the demonstrations were recorded in.
        moved = lambda: tagged_scene(self.geometry, origin_px=(328.0, 204.0))  # noqa: E731
        with self.assertRaisesRegex(VisionError, "no longer the calibration's"):
            self.camera_tool.checked_stored_view(moved, path=self.path, frames=3)

    def session_view(self):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            return self.camera_tool.checked_stored_view(lambda: tagged_scene(self.geometry), path=self.path, frames=3)

    def test_after_a_take_a_picture_that_is_still_the_session_s_is_passed(self):
        view = self.session_view()
        self.assertIsNone(self.camera_tool.changed_since(view, lambda: tagged_scene(self.geometry), wait_s=0.0))
        # a camera that crept by a millimetre during the take is this rig's camera, not another picture
        crept = lambda: tagged_scene(self.geometry, origin_px=(321.0, 200.0))  # noqa: E731
        self.assertIsNone(self.camera_tool.changed_since(view, crept, wait_s=0.0))

    def test_how_far_the_picture_has_crept_since_the_session_s_conversion_is_read_in_millimetres(self):
        # 2026-10-07: the camera moves as it warms, and a session measures through the conversion of its start
        view = self.session_view()
        still = self.camera_tool.crept_since(view, lambda: tagged_scene(self.geometry))
        self.assertLess(still, 0.05)
        one_px = self.camera_tool.crept_since(view, lambda: tagged_scene(self.geometry, origin_px=(321.0, 200.0)))
        two_px = self.camera_tool.crept_since(view, lambda: tagged_scene(self.geometry, origin_px=(322.0, 200.0)))
        # What a picture moved by one pixel is on the floor, by the session's own conversion: the largest over the
        # grid's probe points, which is what the reading must be (a reading scaled, or taken at one point, is not).
        probes = self.camera_tool.grid_probes_px(view, self.geometry)
        moved = max(view.to_mm(pixel).distance_to(view.to_mm((pixel[0] - 1.0, pixel[1]))) for pixel in probes)
        self.assertGreater(moved, 0.3)
        self.assertAlmostEqual(one_px, moved, delta=0.05)
        self.assertAlmostEqual(two_px, 2 * moved, delta=0.08)

    def test_a_creep_that_cannot_be_read_is_none_and_not_an_error(self):
        view = self.session_view()
        blank = lambda: np.full_like(tagged_scene(self.geometry), 200)  # noqa: E731 -- no tag in the picture
        self.assertIsNone(self.camera_tool.crept_since(view, blank))

        def no_frame():
            raise TimeoutError("the camera gave no frame")

        self.assertIsNone(self.camera_tool.crept_since(view, no_frame))

    def test_a_creep_reading_that_is_no_number_cannot_say(self):
        # A solve that gives no number is no reading: compared with a limit it would be over nothing, or over all.
        from unittest import mock

        view = self.session_view()
        still = lambda: tagged_scene(self.geometry)  # noqa: E731
        self.assertLess(self.camera_tool.crept_since(view, still), 0.05)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), \
                    mock.patch.object(self.camera_tool.TableView, "disagreement_mm", return_value=value):
                self.assertIsNone(self.camera_tool.crept_since(view, still))

    def test_after_a_take_a_picture_that_jumped_is_named(self):
        # 2026-10-06: the front camera went back to its own zoom 3.4 s into a take and everything lay elsewhere.
        view = self.session_view()
        jumped = lambda: tagged_scene(self.geometry, origin_px=(360.0, 230.0))  # noqa: E731
        told = self.camera_tool.changed_since(view, jumped, wait_s=0.0, patience_s=0.0)
        self.assertRegex(told, r"stands \d+ mm from the one this session started on \(allowed 5\)")

    def test_after_a_take_a_darker_picture_is_named_though_nothing_moved(self):
        # the exposure back on automatic: the floor at 150 for 200
        self.store_with_levels()
        view, _said = self.said_by_a_session(lambda: tagged_scene(self.geometry))
        self.assertTrue(view.floor_levels, "the session's view carries the calibration's levels")
        self.assertIsNone(self.camera_tool.changed_since(view, lambda: tagged_scene(self.geometry), wait_s=0.0))
        darker = lambda: (tagged_scene(self.geometry).astype(float) * 0.72).astype(np.uint8)  # noqa: E731
        told = self.camera_tool.changed_since(view, darker, wait_s=0.0, patience_s=0.0)
        self.assertIn("the front camera's picture is darker than the calibration's", told)
        # The words are this look's own. A session's start says "Start this command again": after a take that
        # sentence was read as the operator's own slip (the stop of 20:53 on 2026-10-06).
        self.assertNotIn("Start this command again", told)
        self.assertNotIn("has opened", told)

    def test_after_a_take_a_stretched_tone_range_is_named_in_this_look_s_own_words(self):
        def normal():                                # black at 16 and white below 235, as a camera hands it over
            return (16.0 + tagged_scene(self.geometry).astype(float) * 219.0 / 255.0).astype(np.uint8)

        def stretched():
            return np.clip((normal().astype(float) - 16.0) * 255.0 / 219.0, 0, 255).astype(np.uint8)

        stored = TableView.from_frames([normal()] * 4, self.geometry)
        stored.floor_levels = self.camera_tool.picture_levels(stored, normal(), self.geometry)
        stored.save(self.path)
        view, _said = self.said_by_a_session(normal)
        self.assertIsNone(self.camera_tool.changed_since(view, normal, wait_s=0.0))
        told = self.camera_tool.changed_since(view, stretched, wait_s=0.0, patience_s=0.0)
        self.assertIn("the front camera's picture is in another tone state: white", told)
        self.assertRegex(told, r"a contrast \d+\.\d\d of what it was \(allowed 0\.7 to 1\.4\)")
        self.assertNotIn("Start this command again", told)

    def test_after_a_take_a_hand_in_the_way_is_waited_out_and_the_operator_is_told_once(self):
        # The review and the dry run of 2026-10-06: a hand at the basket hides a floor tag for two seconds when the
        # reset ends; the take was thrown away and the sitting stopped.
        view = self.session_view()
        blank = np.full((480, 640, 3), 200, np.uint8)
        pictures = [blank] * 40 + [tagged_scene(self.geometry)] * 10     # eight looks see no tag, the ninth does
        said = []
        clock = iter(float(second) for second in range(1000))
        self.assertIsNone(self.camera_tool.changed_since(view, lambda: pictures.pop(0), wait_s=0.0, told=said.append,
                                                         clock=lambda: next(clock)))
        self.assertEqual(len(said), 1)
        self.assertIn("The floor tags cannot be seen. Hands and the arm out of the picture", said[0])
        self.assertIn("for 15 seconds", said[0])
        self.assertEqual(len(pictures), 5, "it went on after the first clean look")

    def test_after_a_take_a_shadow_over_the_floor_for_a_moment_is_waited_out_too(self):
        self.store_with_levels()
        view, _said = self.said_by_a_session(lambda: tagged_scene(self.geometry))
        darker = (tagged_scene(self.geometry).astype(float) * 0.72).astype(np.uint8)
        pictures = [darker] * 30 + [tagged_scene(self.geometry)] * 10
        said = []
        clock = iter(float(second) for second in range(1000))
        self.assertIsNone(self.camera_tool.changed_since(view, lambda: pictures.pop(0), wait_s=0.0, told=said.append,
                                                         clock=lambda: next(clock)))
        self.assertIn("does not read as it did", said[0])

    def test_after_a_take_what_is_still_wrong_when_the_patience_is_over_is_said(self):
        view = self.session_view()
        blank = np.full((480, 640, 3), 200, np.uint8)
        looks, said = [0], []

        def grab():
            looks[0] += 1
            return blank

        told = self.camera_tool.changed_since(view, grab, wait_s=0.0, told=said.append, frames=1,
                                              clock=lambda: float(looks[0]))      # a second to a look
        self.assertIn("the floor tags could not be seen", told)
        self.assertEqual(len(said), 1)
        self.assertEqual(looks[0], 4 + 15, "four looks, the sentence, then fifteen seconds of looks")

    def test_the_view_a_session_looks_through_keeps_the_calibration_s_levels(self):
        # The look after a take holds the picture's levels against the ones the view carries. A view that lost
        # them on the way (the empty-scene picture, the cube's region) makes that half of the look do nothing,
        # and nothing says so (the review of 2026-10-06).
        stored = self.store_with_levels()
        view, _said = self.said_by_a_session(lambda: tagged_scene(self.geometry))
        through = view.with_background(tagged_scene(self.geometry)).with_cube_region(self.geometry["grid"])
        self.assertEqual(through.floor_levels, stored.floor_levels)
        darker = lambda: (tagged_scene(self.geometry).astype(float) * 0.72).astype(np.uint8)  # noqa: E731
        self.assertIn("darker", self.camera_tool.changed_since(through, darker, wait_s=0.0, patience_s=0.0))

    def test_after_a_take_the_looks_are_half_a_second_apart_and_four_come_before_the_operator_is_told(self):
        from unittest import mock

        view = self.session_view()
        blank = np.full((480, 640, 3), 200, np.uint8)
        events = []
        with mock.patch.object(self.camera_tool.time, "sleep", lambda seconds: events.append(("wait", seconds))):
            told = self.camera_tool.changed_since(view, lambda: events.append("frame") or blank, frames=1,
                                                  told=lambda _text: events.append("told"), patience_s=0.0)
        self.assertIn("could not be seen", told)
        self.assertEqual(events, ["frame", ("wait", 0.5), "frame", ("wait", 0.5), "frame", ("wait", 0.5), "frame"])
        self.assertEqual((self.camera_tool.AFTER_TAKE_TRIES, self.camera_tool.AFTER_TAKE_PATIENCE_S), (4, 15.0))

    def test_after_a_take_a_camera_that_gives_no_frame_is_said_and_does_not_end_the_session(self):
        # LeRobot's reader raises when the camera's latest frame is older than half a second; on a link that drops
        # frames that happens, and it ended the session with the finished take lost (the review of 2026-10-06).
        view = self.session_view()
        calls = [0]

        def stalled():
            calls[0] += 1
            if calls[0] <= 7:
                raise TimeoutError("frame is 612 ms old")
            return tagged_scene(self.geometry)

        self.assertIsNone(self.camera_tool.changed_since(view, stalled, wait_s=0.0))

        def dead():
            raise TimeoutError("frame is 9000 ms old")

        told = self.camera_tool.changed_since(view, dead, wait_s=0.0, patience_s=0.0)
        self.assertIn("the front camera gave no frame (frame is 9000 ms old)", told)

    def test_after_a_take_tags_out_of_sight_for_a_moment_are_waited_for_and_tags_gone_are_said(self):
        view = self.session_view()
        blank = np.full((480, 640, 3), 200, np.uint8)
        pictures = [blank] * 5 + [tagged_scene(self.geometry)] * 20      # a hand over a tag for the first look
        self.assertIsNone(self.camera_tool.changed_since(view, lambda: pictures.pop(0), wait_s=0.0))
        told = self.camera_tool.changed_since(view, lambda: blank, wait_s=0.0, patience_s=0.0)
        self.assertIn("the floor tags could not be seen", told)
        self.assertIn("cannot be said", told)

    # ------------------------------------------------------------------------------- the wrist camera's readings

    def test_the_wrist_camera_s_reading_is_the_middle_of_three_looks(self):
        from fixtures_pictures import wrist_rest_picture

        pictures = [wrist_rest_picture(brighter=by) for by in (0.0, 0.0, 0.0)]
        self.assertAlmostEqual(self.camera_tool.wrist_reading(lambda: pictures.pop(0), wait_s=0.0), 2.576, places=2)
        # one look in the other state among three does not make the reading, whichever of the three it is
        for order in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            pictures = [wrist_rest_picture(stretched=bool(other)) for other in order]
            self.assertLess(self.camera_tool.wrist_reading(lambda: pictures.pop(0), wait_s=0.0), 2.7)

    def test_a_wrist_look_that_cannot_be_judged_or_gets_no_frame_says_nothing(self):
        from fixtures_pictures import wrist_rest_picture

        away = np.full((480, 640, 3), 200, np.uint8)
        pictures = [away, wrist_rest_picture(stretched=True), away]
        self.assertGreater(self.camera_tool.wrist_reading(lambda: pictures.pop(0), wait_s=0.0), 3.08)
        self.assertIsNone(self.camera_tool.wrist_reading(lambda: away, wait_s=0.0))

        def dead():
            raise TimeoutError("frame is 700 ms old")

        self.assertIsNone(self.camera_tool.wrist_reading(dead, wait_s=0.0))

    def test_a_wrist_change_is_said_only_when_a_second_reading_says_it_too(self):
        from fixtures_pictures import wrist_rest_picture

        usual, other = wrist_rest_picture(), wrist_rest_picture(stretched=True)
        start = self.camera_tool.wrist_reading(lambda: usual, wait_s=0.0)
        now, told = self.camera_tool.wrist_changed_since(start, lambda: usual, wait_s=0.0)
        self.assertIsNone(told)
        self.assertAlmostEqual(now, start, places=3)
        now, told = self.camera_tool.wrist_changed_since(start, lambda: other, wait_s=0.0)
        self.assertIn("has changed its tone state since this session began", told)
        self.assertGreater(now, 3.08)
        # three looks in the other state (a first reading), then the usual again: not a change
        pictures = [other] * 3 + [usual] * 3
        self.assertIsNone(self.camera_tool.wrist_changed_since(start, lambda: pictures.pop(0), wait_s=0.0)[1])
        # a session whose first reading could not be judged holds nothing against it
        self.assertEqual(self.camera_tool.wrist_changed_since(None, lambda: other, wait_s=0.0)[1], None)
        # ... and a look that is not the rest picture says nothing
        away = np.full((480, 640, 3), 200, np.uint8)
        self.assertEqual(self.camera_tool.wrist_changed_since(start, lambda: away, wait_s=0.0), (None, None))

    def test_no_stored_view_is_an_error_that_says_what_to_run(self):
        with self.assertRaisesRegex(VisionError, "calibrate"):
            self.camera_tool.checked_stored_view(lambda: tagged_scene(self.geometry), path=self.path.with_name("none.json"))

    # ---- the picture a session opens on, against the picture the calibration saw

    def said_by_a_session(self, grab):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            view = self.camera_tool.checked_stored_view(grab, path=self.path, frames=3)
        return view, out.getvalue()

    def refused(self, grab):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(VisionError) as caught:
            self.camera_tool.checked_stored_view(grab, path=self.path, frames=3)
        return str(caught.exception)

    def store_with_levels(self):
        view = TableView.from_frames([tagged_scene(self.geometry)] * 4, self.geometry)
        view.floor_levels = self.camera_tool.picture_levels(view, tagged_scene(self.geometry), self.geometry)
        view.save(self.path)
        return view

    def test_the_picture_s_levels_are_read_and_kept_with_the_stored_view(self):
        stored = self.store_with_levels()
        self.assertEqual(stored.floor_levels["grey"], 200.0)             # the fixture's floor
        self.assertEqual(stored.floor_levels["red_over_blue"], 1.0)
        self.assertLess(stored.floor_levels["tag_black"], 30.0)          # the tags' black
        self.assertEqual(TableView.load(self.path).floor_levels, stored.floor_levels)
        self.assertEqual(TableView.load(self.path).with_background(tagged_scene(self.geometry)).floor_levels,
                         stored.floor_levels)

    def test_a_cube_on_the_grid_does_not_change_what_is_read(self):
        view = self.store_with_levels()
        frame = tagged_scene(self.geometry)
        x, y = (int(round(v)) for v in view.to_px(Point(125.0, 75.0)))
        cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), ORANGE_BGR, -1)
        self.assertEqual(self.camera_tool.picture_levels(view, frame, self.geometry), view.floor_levels)

    def test_a_session_says_the_picture_s_levels_and_goes_on_when_they_are_as_they_were(self):
        self.store_with_levels()
        view, said = self.said_by_a_session(lambda: tagged_scene(self.geometry))
        self.assertIn("the floor inside the grid: grey level 200 (at the calibration 200)", said)
        self.assertIn("the picture's contrast is 1.00 of the calibration's", said)
        self.assertNotIn("WARNING", said)
        self.assertEqual(view.frames_used, 3)

    def test_a_little_darker_is_let_through(self):
        # Lamps warm up and daylight comes and goes: a few grey levels are not another picture.
        self.store_with_levels()
        dimmer = lambda: (tagged_scene(self.geometry).astype(float) * 0.93).astype(np.uint8)  # noqa: E731
        view, said = self.said_by_a_session(dimmer)
        self.assertIn("grey level 186 (at the calibration 200)", said)
        self.assertEqual(view.frames_used, 3)
        # It is said, though: an opening of the camera read 12 darker at the rig with nothing changed.
        self.assertIn("NOTE: the floor reads 14 grey levels darker than at the calibration", said)
        barely = lambda: (tagged_scene(self.geometry).astype(float) * 0.985).astype(np.uint8)  # noqa: E731
        _view, said = self.said_by_a_session(barely)
        self.assertNotIn("NOTE", said)

    def test_a_much_darker_picture_is_refused(self):
        # A hand-set exposure that was lost: 207 became 148 at the rig on 2026-10-06.
        self.store_with_levels()
        darker = lambda: (tagged_scene(self.geometry).astype(float) * 0.72).astype(np.uint8)  # noqa: E731
        why = self.refused(darker)
        self.assertIn("the picture is darker than at the calibration", why)
        self.assertIn("Start this command again", why)

    def test_a_picture_stretched_from_video_range_to_full_range_is_refused(self):
        # The same camera picture handed over in another tone state: whites up by a few percent, blacks halved.
        # The floor alone would pass (it moves by less than the limit); the contrast does not.
        from unittest import mock

        def normal():
            return (16.0 + tagged_scene(self.geometry).astype(float) * 219.0 / 255.0).astype(np.uint8)

        def stretched():
            return np.clip((normal().astype(float) - 16.0) * 255.0 / 219.0, 0, 255).astype(np.uint8)

        view = TableView.from_frames([normal()] * 4, self.geometry)
        view.floor_levels = self.camera_tool.picture_levels(view, normal(), self.geometry)
        view.save(self.path)
        _view, said = self.said_by_a_session(normal)
        self.assertNotIn("WARNING", said)
        self.assertLess(abs(self.camera_tool.picture_levels(view, stretched(), self.geometry)["grey"]
                            - view.floor_levels["grey"]), self.camera_tool.PICTURE_GREY_LIMIT)
        why = self.refused(stretched)
        self.assertIn("another picture state", why)
        self.assertIn("Start this command again", why)
        # And the other way round: calibrated on a stretched picture, opened on a normal one.
        view.floor_levels = self.camera_tool.picture_levels(view, stretched(), self.geometry)
        view.save(self.path)
        self.assertIn("another picture state", self.refused(normal))
        with mock.patch.object(self.camera_tool, "PICTURE_CONTRAST_RANGE", (0.0, 100.0)):
            self.said_by_a_session(normal)            # the contrast alone was what refused it

    def test_a_picture_of_another_colour_is_warned_about_and_goes_on(self):
        self.store_with_levels()

        def warmer():
            frame = tagged_scene(self.geometry).astype(float)
            frame[:, :, 0] *= 0.9    # less blue
            return frame.astype(np.uint8)

        view, said = self.said_by_a_session(warmer)
        self.assertIn("WARNING: the picture's colour balance", said)
        self.assertEqual(view.frames_used, 3)

    def test_a_view_stored_before_this_was_kept_says_nothing_about_the_picture(self):
        _view, said = self.said_by_a_session(lambda: tagged_scene(self.geometry))
        self.assertNotIn("the floor inside the grid", said)

    def test_a_reading_of_the_picture_that_fails_never_stops_a_session(self):
        from unittest import mock

        self.store_with_levels()
        with mock.patch.object(self.camera_tool, "floor_levels", side_effect=RuntimeError("no picture")):
            view, said = self.said_by_a_session(lambda: tagged_scene(self.geometry))
        self.assertIn("could not be read", said)
        self.assertEqual(view.frames_used, 3)


class FakeDesk:
    """Stands in for the camera, the window and the keyboard: a scripted list of (frame, key) pairs."""

    def __init__(self, script):
        self._script = list(script)
        self._index = -1
        self.shown = 0
        self.now = 0.0

    def grab(self):
        self._index = min(self._index + 1, len(self._script) - 1)
        return self._script[self._index][0]

    def show(self, _title, _picture):
        self.shown += 1

    def wait_key(self, _ms):
        self.now += 0.03
        return self._script[self._index][1]

    def clock(self):
        return self.now


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class GoKeyTests(unittest.TestCase):
    """What counts as the operator's go during a placement, from the flags the session's key listener sets."""

    def test_the_go_key_alone_is_a_go_once(self):
        events = {"exit_early": True, "rerecord_episode": False, "stop_recording": False}
        self.assertTrue(consume_go(events))
        self.assertFalse(events["exit_early"])
        self.assertFalse(consume_go(events))

    def test_no_key_is_no_go(self):
        events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        self.assertFalse(consume_go(events))
        self.assertEqual(events, {"exit_early": False, "rerecord_episode": False, "stop_recording": False})

    def test_the_stop_key_is_not_a_go_and_the_stop_is_left_for_the_session(self):
        events = {"exit_early": True, "rerecord_episode": False, "stop_recording": True}
        self.assertFalse(consume_go(events))
        self.assertTrue(events["stop_recording"])

    def test_the_rerecord_key_is_not_a_go_and_is_forgotten(self):
        # LeRobot's listener sets both flags for r and for the left arrow.
        events = {"exit_early": True, "rerecord_episode": True, "stop_recording": False}
        self.assertFalse(consume_go(events))
        self.assertEqual(events, {"exit_early": False, "rerecord_episode": False, "stop_recording": False})
        self.assertFalse(consume_go(events))

    def test_flags_that_are_missing_count_as_not_set(self):
        self.assertFalse(consume_go({}))
        self.assertTrue(consume_go({"exit_early": True}))


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class RerunDisplayTests(unittest.TestCase):
    """The placement picture goes where the session's viewer shows it, on top, and is taken away afterwards."""

    def setUp(self):
        import sys
        import types

        from tolerance_radius.rig import display

        self.display = display
        self.logged = []
        fake = types.ModuleType("rerun")
        fake.set_time = lambda timeline, sequence: self.logged.append(("time", timeline, sequence))
        fake.log = lambda entity, thing: self.logged.append(("log", entity, thing))
        fake.EncodedImage = lambda **fields: ("encoded", fields)
        fake.Clear = lambda recursive: ("clear", recursive)
        self._saved = sys.modules.get("rerun")
        sys.modules["rerun"] = fake
        self.addCleanup(self._restore)

    def _restore(self):
        import sys

        if self._saved is None:
            sys.modules.pop("rerun", None)
        else:
            sys.modules["rerun"] = self._saved

    def test_the_picture_is_logged_under_the_front_cameras_view(self):
        # LeRobot's layout has one view per camera, rooted at the camera's own path, and nothing else is shown.
        self.assertTrue(self.display.ENTITY.startswith("observation.front/"))

    def test_a_picture_is_sent_compressed_and_drawn_over_the_camera_picture(self):
        screen = self.display.RerunDisplay()
        screen.show("title", scene(Point(75.0, 25.0)))
        kind, entity, (what, fields) = self.logged[-1]
        self.assertEqual((kind, entity, what), ("log", self.display.ENTITY, "encoded"))
        self.assertEqual(fields["media_type"], "image/jpeg")
        self.assertGreater(fields["draw_order"], 0)
        decoded = cv2.imdecode(np.frombuffer(fields["contents"], dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, scene().shape)
        # The same picture, channel for channel (a JPEG, so nearly): the colours were not swapped on the way.
        self.assertLess(float(np.abs(decoded.astype(int) - scene(Point(75.0, 25.0)).astype(int)).mean()), 6.0)

    def test_clearing_takes_the_picture_away_so_the_camera_shows_again(self):
        screen = self.display.RerunDisplay()
        screen.show("title", scene())
        screen.clear()
        self.assertEqual(self.logged[-1], ("log", self.display.ENTITY, ("clear", False)))

    def test_the_picture_sets_no_timeline_of_its_own(self):
        # A timeline set by the display stays set for everything the thread logs afterwards, the cameras' frames
        # among them: in the first session shown in a real viewer the live cameras stood still after a placement.
        screen = self.display.RerunDisplay()
        screen.show("title", scene())
        screen.show("title", scene())
        screen.clear()
        self.assertEqual([entry for entry in self.logged if entry[0] == "time"], [])

    def test_a_viewer_that_fails_is_reported_once_and_never_raises(self):
        import sys

        def broken(*_args, **_kwargs):
            raise RuntimeError("viewer gone")

        sys.modules["rerun"].log = broken
        screen = self.display.RerunDisplay()
        with self.assertLogs(level="WARNING") as seen:
            screen.show("title", scene())
            screen.show("title", scene())
            screen.clear()
        self.assertEqual(len(seen.records), 1)

    def test_no_keys_are_read_from_the_viewer(self):
        slept = []
        screen = self.display.RerunDisplay(sleep=slept.append)
        self.assertEqual(screen.wait_key(30), -1)
        self.assertEqual(slept, [0.03])


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class PlacementPhaseTests(unittest.TestCase):
    TARGET = Point(75.0, 25.0)

    def phase(self, script, **options):
        desk = FakeDesk(script)
        view = TableView(PX_TO_MM, scene())
        phase = PlacementPhase(
            view,
            desk.grab,
            tolerance_mm=2.0,
            stable_frames=3,
            show=desk.show,
            wait_key=desk.wait_key,
            clock=desk.clock,
            **options,
        )
        return phase, desk

    def test_accepts_when_steady_on_target_and_reports_what_it_measured(self):
        on_target = scene(Point(75.5, 25.0))
        phase, desk = self.phase([(scene(Point(90.0, 25.0)), -1)] + [(on_target, -1)] * 3 + [(on_target, KEY_ACCEPT)])
        result = phase.run(self.TARGET, "episode 3")
        self.assertTrue(result.accepted)
        self.assertLess(result.measured.distance_to(Point(75.5, 25.0)), 0.5)
        self.assertLess(result.error_mm, 2.0)
        self.assertEqual(desk.shown, result.frames_seen)

    def test_space_does_nothing_until_the_cube_has_been_steady_on_target(self):
        off = scene(Point(90.0, 25.0))
        on = scene(self.TARGET)
        # space while off target, then space after only two good frames, then a third good frame and space
        script = [(off, KEY_ACCEPT), (on, KEY_ACCEPT), (on, KEY_ACCEPT), (on, KEY_ACCEPT)]
        phase, _ = self.phase(script)
        result = phase.run(self.TARGET)
        self.assertTrue(result.accepted)
        self.assertEqual(result.frames_seen, 4)

    def test_a_hand_in_the_picture_breaks_the_steady_run(self):
        on = scene(self.TARGET)
        with_hand = scene(self.TARGET, Point(180.0, 120.0))
        script = [(on, -1), (on, -1), (with_hand, KEY_ACCEPT), (on, KEY_ACCEPT), (on, KEY_ACCEPT), (on, KEY_ACCEPT)]
        phase, _ = self.phase(script)
        result = phase.run(self.TARGET)
        self.assertTrue(result.accepted)
        self.assertEqual(result.frames_seen, 6)

    def test_skip_abort_and_timeout_return_without_a_measurement(self):
        on = scene(self.TARGET)
        for key, status in ((KEY_SKIP, "skipped"), (KEY_ABORT, "aborted")):
            with self.subTest(status=status):
                phase, _ = self.phase([(on, key)])
                result = phase.run(self.TARGET)
                self.assertEqual(result.status, status)
                self.assertIsNone(result.measured)
        phase, _ = self.phase([(scene(), -1)], timeout_s=0.5)
        self.assertEqual(phase.run(self.TARGET).status, "timeout")

    def test_with_auto_accept_a_steady_cube_is_accepted_without_a_key(self):
        on = scene(self.TARGET)
        phase, _ = self.phase([(scene(Point(90.0, 25.0)), -1)] + [(on, -1)] * 10, auto_accept=True)
        result = phase.run(self.TARGET)
        self.assertTrue(result.accepted)
        self.assertEqual(result.frames_seen, 4)  # one frame off target, then three steady ones

    def test_steady_also_means_long_enough_on_the_clock(self):
        on = scene(self.TARGET)
        # the fake desk's clock moves 0.03 s per frame: 0.2 s needs 8 frames after the first good one
        phase, _ = self.phase([(on, -1)] * 40, auto_accept=True, hold_s=0.2)
        result = phase.run(self.TARGET)
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.frames_seen, 7)
        self.assertLessEqual(result.frames_seen, 9)

    def test_a_session_confirms_with_its_own_key_and_an_early_press_is_forgotten(self):
        off, on = scene(Point(90.0, 25.0)), scene(self.TARGET)
        presses = iter([True, False, False, False, False, True, False])  # pressed while off target, then once steady
        asked = []

        def confirm():
            pressed = next(presses)
            asked.append(pressed)
            return pressed

        phase, _ = self.phase([(off, -1)] + [(on, -1)] * 8, confirm=confirm)
        result = phase.run(self.TARGET)
        self.assertTrue(result.accepted)
        self.assertEqual(result.frames_seen, 6)   # not at frame 1 (off target), not at frame 4 (steady, no press)
        self.assertEqual(asked[0], True)           # the early press was asked for and consumed

    def test_a_steady_cube_is_not_accepted_until_someone_confirms(self):
        on = scene(self.TARGET)
        phase, _ = self.phase([(on, -1)] * 12, confirm=lambda: False, timeout_s=0.3)
        self.assertEqual(phase.run(self.TARGET).status, "timeout")

    def test_the_session_can_stop_a_placement(self):
        stop_after = iter([False, False, True])
        phase, _ = self.phase([(scene(), -1)] * 10, should_stop=lambda: next(stop_after))
        result = phase.run(self.TARGET)
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.frames_seen, 2)

    def test_the_end_of_a_placement_is_announced_however_it_ends(self):
        on = scene(self.TARGET)
        ends = []
        phase, _ = self.phase([(on, -1)] * 3 + [(on, KEY_ACCEPT)], on_end=lambda: ends.append("end"))
        self.assertTrue(phase.run(self.TARGET).accepted)
        self.assertEqual(ends, ["end"])
        for script, options, status in (
            ([(scene(), KEY_SKIP)], {}, "skipped"),
            ([(scene(), KEY_ABORT)], {}, "aborted"),
            ([(scene(), -1)] * 12, {"timeout_s": 0.2}, "timeout"),
            ([(scene(), -1)] * 4, {"should_stop": lambda: True}, "stopped"),
        ):
            ends.clear()
            phase, _ = self.phase(script, on_end=lambda: ends.append("end"), **options)
            with self.subTest(status=status):
                self.assertEqual(phase.run(self.TARGET).status, status)
                self.assertEqual(ends, ["end"])

    def test_the_end_is_announced_when_the_camera_fails_and_the_failure_still_comes_out(self):
        ends = []

        def broken_grab():
            raise OSError("camera gone")

        view = TableView(PX_TO_MM, scene())
        phase = PlacementPhase(view, broken_grab, tolerance_mm=2.0, show=lambda *_: None, wait_key=lambda _ms: -1,
                               on_end=lambda: ends.append("end"))
        with self.assertRaises(OSError):
            phase.run(self.TARGET)
        self.assertEqual(ends, ["end"])

    def test_a_failure_while_announcing_the_end_does_not_cost_the_placement(self):
        on = scene(self.TARGET)

        def broken_end():
            raise RuntimeError("viewer gone")

        phase, _ = self.phase([(on, -1)] * 3 + [(on, KEY_ACCEPT)], on_end=broken_end)
        with self.assertLogs(level="WARNING"):
            result = phase.run(self.TARGET)
        self.assertTrue(result.accepted)

    def test_assess_reports_the_offset_in_table_axes(self):
        view = TableView(PX_TO_MM, scene())
        reading = assess(view.measure(scene(Point(80.0, 22.0))), self.TARGET, 2.0)
        self.assertEqual(reading.status, "off")
        self.assertAlmostEqual(reading.dx_mm, 5.0, delta=0.5)
        self.assertAlmostEqual(reading.dy_mm, -3.0, delta=0.5)
        self.assertEqual(assess(None, self.TARGET, 2.0).status, "none")

    def test_render_returns_a_picture_and_leaves_the_frame_alone(self):
        view = TableView(PX_TO_MM, scene())
        frame = scene(Point(80.0, 22.0))
        before = frame.copy()
        picture = render(frame, view, self.TARGET, assess(view.measure(frame), self.TARGET, 2.0), 2.0, False, "T1a-001")
        from tolerance_radius.rig.placement_phase import ZOOM, ZOOM_HALF_PX

        # The camera's picture, and beside it the target's surroundings magnified.
        self.assertEqual(picture.shape, (480, 640 + 2 * ZOOM_HALF_PX * ZOOM, 3))
        self.assertTrue(np.array_equal(frame, before))
        self.assertFalse(np.array_equal(picture[:, :640], frame))

    def test_the_cube_s_own_outline_is_drawn_where_the_cube_belongs(self):
        # What the operator slides the cube into: the outline the detector sees now, moved so that its centre lies on
        # the target. Nothing is modelled, so the outline and the ring cannot disagree.
        from tolerance_radius.rig.placement_phase import ZOOM, ZOOM_HALF_PX

        view = TableView(PX_TO_MM, scene())
        frame = scene(Point(80.0, 22.0))                       # 5 mm right of the target and 3 mm below it
        reading = view.measure(frame)
        self.assertIsNotNone(reading.outline_px)
        current = assess(reading, self.TARGET, 2.0)
        picture = render(frame, view, self.TARGET, current, 2.0, False, "T1a-001")
        close = picture[:, 640:]
        red = (close[:, :, 2] > 200) & (close[:, :, 1] < 110) & (close[:, :, 0] < 110)   # the colour of "off"
        ys, xs = np.nonzero(red)
        middle = ZOOM_HALF_PX * ZOOM
        # The cube is 60 px wide in this scene, 300 px magnified: its outline about the target reaches 150 px either
        # side of the middle of the magnified picture.
        self.assertLess(xs.min(), middle - 130)
        self.assertGreater(xs.max(), middle + 130)
        # With the cube on the target the outline lies on the cube itself: nothing of it stands clear of the cube.
        on = scene(self.TARGET)
        settled = render(on, view, self.TARGET, assess(view.measure(on), self.TARGET, 2.0), 2.0, True, "T1a-001")
        self.assertEqual(settled.shape, picture.shape)

    def test_the_grid_is_drawn_when_it_is_given(self):
        view = TableView(PX_TO_MM, scene())
        frame = scene(Point(80.0, 22.0))
        current = assess(view.measure(frame), self.TARGET, 2.0)
        without = render(frame, view, self.TARGET, current, 2.0, False)
        with_grid = render(frame, view, self.TARGET, current, 2.0, False, grid={"cols": 5, "rows": 3, "cell_mm": 50.0})
        self.assertFalse(np.array_equal(without, with_grid))
        from tolerance_radius.rig.placement_phase import cell_names_mm

        names = dict(cell_names_mm({"cols": 5, "rows": 3, "cell_mm": 50.0}))
        self.assertEqual((names["C2"].x_mm, names["C2"].y_mm), (125.0, 75.0))
        self.assertEqual((names["A1"].x_mm, names["E3"].y_mm), (25.0, 125.0))

    def test_the_cube_is_looked_for_on_the_grid_and_around_it_and_nowhere_else(self):
        # The arm is cube-coloured. Resting a centimetre from where it rested in the empty-scene picture it is a
        # second orange thing, and the detector then reports no cube at all: at the rig that made the arm's rest pose
        # something to hit to two degrees before every placement (2026-10-06).
        grid = {"cols": 5, "rows": 3, "cell_mm": 50.0}
        # In this scene (2 px per mm) the picture ends 60 mm beyond the grid's near side; the "arm" rests there, and
        # the region is the grid and 20 mm around it.
        arm_then, arm_now = Point(120.0, 195.0), Point(160.0, 195.0)
        empty = scene(arm_then)
        whole = TableView(PX_TO_MM, empty)
        there = whole.with_cube_region(grid, margin_mm=20.0)
        frame = scene(arm_now, Point(80.0, 22.0))                        # the arm has moved; the cube is on the table
        self.assertIsNone(whole.measure(frame))                          # two orange things that changed
        reading = there.measure(frame)
        self.assertIsNotNone(reading)
        self.assertLess(reading.position.distance_to(Point(80.0, 22.0)), 0.6)
        # A cube at the grid's very edge is still whole inside the region.
        edge = there.measure(scene(arm_now, Point(5.0, 5.0)))
        self.assertLess(edge.position.distance_to(Point(5.0, 5.0)), 0.6)
        # Something orange inside the region is still a second thing: the arm must stay out of it.
        self.assertIsNone(there.measure(scene(arm_now, Point(80.0, 22.0), Point(200.0, 100.0))))
        # The region is kept when the view is given another empty-scene picture, and needs one to begin with.
        self.assertIsNotNone(there.with_background(empty).measure(frame))
        with self.assertRaises(VisionError):
            TableView(PX_TO_MM).with_cube_region(grid)

    def test_a_view_without_a_background_is_refused_up_front(self):
        with self.assertRaises(ValueError):
            PlacementPhase(TableView(PX_TO_MM), lambda: scene(), tolerance_mm=2.0, show=lambda *_: None, wait_key=lambda _: -1)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class PinnedToolingTests(unittest.TestCase):
    """Which arm tooling and which LeRobot a session runs is checked, not assumed."""

    def test_the_arm_tooling_is_at_the_pinned_commit(self):
        self.assertEqual(lerobot101.check_pin(), lerobot101.PINNED_COMMIT)

    def test_a_session_imports_lerobot_from_the_august_checkout(self):
        import subprocess
        import sys
        from pathlib import Path

        if not (lerobot101.lerobot_checkout() / "src" / "lerobot").is_dir():
            self.skipTest("no LeRobot checkout on this machine")
        source = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from tolerance_radius.rig import lerobot101;"
            "print(lerobot101.use_pinned_lerobot());"
            "import lerobot; print(lerobot.__file__)"
        )
        result = subprocess.run([sys.executable, "-c", code, str(source)], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        described, imported_from = result.stdout.strip().splitlines()[-2:]
        self.assertIn(lerobot101.LEROBOT_PINNED_COMMIT[:10], described)
        self.assertTrue(imported_from.startswith(str(lerobot101.lerobot_checkout() / "src")), imported_from)

    def test_it_refuses_once_another_lerobot_is_already_imported(self):
        import subprocess
        import sys
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import lerobot;"  # the environment's own copy
            "from tolerance_radius.rig import lerobot101\n"
            "try:\n    lerobot101.use_pinned_lerobot(); print('accepted')\n"
            "except lerobot101.ToolingError as e:\n    print('refused')"
        )
        result = subprocess.run([sys.executable, "-c", code, str(source)], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "refused", result.stderr[-800:])


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class ResetByEyeTests(unittest.TestCase):
    """The reset-by-eye measurement, driven by a scripted camera and keyboard."""

    def test_each_reset_is_logged_with_its_distance_from_the_guided_placement(self):
        import argparse
        import csv
        import tempfile
        from pathlib import Path
        from unittest import mock

        from tolerance_radius.config import load_protocol
        from tolerance_radius.geometry import Grid
        from tolerance_radius.rig import camera_tool
        from tolerance_radius.seeds import derived_rng
        from tolerance_radius.targets import uniform_placements

        protocol = load_protocol()
        rng = derived_rng(int(protocol["datasets"]["layout_seed"]), "reset-by-eye", "t")
        targets = uniform_placements(Grid.from_protocol(protocol), 3, rng, "trained_cells")
        offsets = [(4.0, 0.0), (0.0, -7.0), (3.0, 4.0)]
        order = camera_tool.byeye_order(3, rng)  # the same random stream the tool draws its order from
        frames = []
        for target in targets:  # first pass: every reference
            frames += [scene(target)] * 12  # placed with the ring
            frames += [scene(target)]  # the reference picture
        for index in order:  # second pass: each put back by eye, in the shuffled order
            (dx, dy), target = offsets[index], targets[index]
            frames += [scene()] * 10  # taken off the table
            frames += [scene(Point(target.x_mm + dx, target.y_mm + dy))] * 12  # put back by eye, off by (dx, dy)
        feed = iter(frames)

        class ScriptedCamera:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                pass

            def grab(self):
                return next(feed)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "byeye.csv"
            args = argparse.Namespace(camera=None, count=3, label="t", out=str(out))
            with (
                mock.patch.object(camera_tool, "Camera", ScriptedCamera),
                mock.patch.object(camera_tool, "table_view", lambda camera, ask=input: TableView(PX_TO_MM, scene())),
                mock.patch.object(cv2, "imshow", lambda *_: None),
                mock.patch.object(cv2, "waitKey", lambda _ms: ord(" ")),  # space on every frame: only steady counts
                mock.patch("builtins.print"),
            ):
                self.assertEqual(camera_tool.mode_byeye(args), 0)
            with open(out, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        # Rows come in the order the resets were made, each naming its reference and its place in that order.
        self.assertEqual([int(row["index"]) for row in rows], order)
        self.assertEqual([int(row["order"]) for row in rows], [1, 2, 3])
        for row in rows:
            dx, dy = offsets[int(row["index"])]
            self.assertAlmostEqual(float(row["error_mm"]), float(np.hypot(dx, dy)), delta=0.5)

    def test_the_resets_are_made_in_another_order_than_the_references_were_taken(self):
        from tolerance_radius.rig import camera_tool

        for seed in range(40):
            order = camera_tool.byeye_order(20, np.random.default_rng(seed))
            self.assertEqual(sorted(order), list(range(20)))
            self.assertNotEqual(order[0], 19)  # never the reference taken a moment ago
            self.assertNotEqual(order, list(range(20)))
        self.assertEqual(camera_tool.byeye_order(1, np.random.default_rng(0)), [0])
        self.assertEqual(camera_tool.byeye_order(0, np.random.default_rng(0)), [])

    def test_the_side_by_side_picture_puts_no_mark_on_the_table(self):
        from tolerance_radius.rig.camera_tool import side_by_side

        reference, live = scene(Point(75.0, 25.0)), scene(Point(90.0, 25.0))
        picture = side_by_side(reference, live, "put it back by eye", False)
        self.assertEqual(picture.shape, (480, 1280, 3))
        # below the text line both halves are the untouched pictures: no ring, no arrow, no target
        self.assertTrue(np.array_equal(picture[40:, :640], reference[40:]))
        self.assertTrue(np.array_equal(picture[40:, 640:], live[40:]))
