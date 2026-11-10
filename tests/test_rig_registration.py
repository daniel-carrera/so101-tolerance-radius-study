"""Registering the table: the two floor tags measured against the printed sheet, on synthetic pictures.

The pictures are made as a camera makes them (`fixtures_pictures`), at the rig's own view of 2026-10-05, so that the
tag detector behaves here as it does at the rig: it reports a tag's corners about half a pixel inside its outline.
"""

import unittest

import numpy as np

from fixtures_pictures import RIG_FLOOR, RIG_VIEW, rectangle_mm, render, sheet_grid_lines, to_px

try:
    import cv2

    from fixtures_pictures import tag_bits
    from tolerance_radius.rig import lerobot101, outline, registration

    READY = lerobot101.available()
except ImportError:  # the development environment has no OpenCV
    READY = False

GRID = {"cols": 5, "rows": 3, "cell_mm": 50.0}
#: The floor tags' true sizes, as calipers would give them.
STATED = {tag_id: size for tag_id, (_centre, size, _turn) in RIG_FLOOR.items()}


def shifted(view, dx=0.0, dy=0.0):
    """The same view with the picture pushed sideways or up and down, in pixels."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]) @ view


def turned(view, degrees, about=(320.0, 240.0)):
    """The same view with the picture turned about a point, as a camera turned about its own line of sight shows
    it. Positive is from the picture's x towards its y."""
    a = np.radians(degrees)
    cx, cy = about
    turn = np.array([[np.cos(a), -np.sin(a), cx - cx * np.cos(a) + cy * np.sin(a)],
                     [np.sin(a), np.cos(a), cy - cx * np.sin(a) - cy * np.cos(a)],
                     [0.0, 0.0, 1.0]])
    return turn @ view


def floor_corners(tag_id, floor=RIG_FLOOR):
    centre, size, turn = floor[tag_id]
    return rectangle_mm(centre, size, turn)


def picture(view=RIG_VIEW, *, sheet_ids=tuple(range(10, 16)), floor=RIG_FLOOR, sheet_scale=(1.0, 1.0), moved=None,
            resized=None, bump=None, seed=0, **options):
    """A 640 x 480 colour picture of the floor: the sheet with its tags and printed grid, and the floor tags, seen
    through `view`.

    `sheet_scale` is the scale the sheet was really printed at. `moved` shifts single sheet tags on the table, in
    millimetres, as a lifted corner of the sheet shifts what the camera sees. `resized` draws single sheet tags
    larger or smaller about their own centres, across and up, as a tag on a slope is seen. `bump` puts a bulge on the
    grid's far outline: a function of x giving how far the line is seen off its place there, in millimetres.
    """
    tags = {}
    for tag_id in sheet_ids:
        corners = registration.sheet_corners_mm(tag_id, np.broadcast_to(np.asarray(sheet_scale, dtype=float), (2,)))
        centre = corners.mean(axis=0)
        corners = centre + (corners - centre) * np.asarray((resized or {}).get(tag_id, (1.0, 1.0)))
        tags[tag_id] = (corners + np.asarray((moved or {}).get(tag_id, (0.0, 0.0))), tag_bits(tag_id))
    for tag_id in floor:
        tags[tag_id] = (floor_corners(tag_id, floor), tag_bits(tag_id))
    scale = np.broadcast_to(np.asarray(sheet_scale, dtype=float), (2,))
    lines = sheet_grid_lines(scale, bump) if sheet_ids else None
    grey = render(view, tags, seed=seed, lines=lines, **options)
    return np.repeat(grey[:, :, None], 3, axis=2)


def seen(image, frames=3):
    """What a registration is given for `frames` equal frames: the detector's tags, and their outlines."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = lerobot101.load("tag_check").detect(grey)
    return [found] * frames, [outline.tag_outlines(grey, found)] * frames


def seen_in(images):
    """The same for a run of different frames."""
    found, outlines = [], []
    for image in images:
        one_found, one_outline = seen(image, 1)
        found += one_found
        outlines += one_outline
    return found, outlines


def register(image_or_seen, **options):
    """Register from a picture (its printed lines are then read too) or from tags already measured."""
    if isinstance(image_or_seen, tuple):
        found, outlines = image_or_seen
    else:
        found, outlines = seen(image_or_seen)
        options.setdefault("picture", cv2.cvtColor(image_or_seen, cv2.COLOR_BGR2GRAY))
    options.setdefault("floor_tags_mm", STATED)
    return registration.register(found, outlines, grid=GRID, **options)


def grid_error_mm(result, view=RIG_VIEW, sheet_scale=(1.0, 1.0)):
    """How far the registered frame is from the truth over the grid: mean and largest distance."""
    xs, ys = np.meshgrid(np.linspace(0, 250, 11), np.linspace(0, 150, 7))
    truth = np.column_stack([xs.ravel(), ys.ravel()]) * np.asarray(sheet_scale, dtype=float)
    pixels = to_px(view, truth)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))]) @ result.px_to_mm.T
    distance = np.linalg.norm(homogeneous[:, :2] / homogeneous[:, 2:3] - truth, axis=1)
    return float(distance.mean()), float(distance.max())


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class SheetTests(unittest.TestCase):
    def test_the_sheet_s_tags_sit_at_the_centres_of_six_cells_inside_the_grid(self):
        self.assertEqual(sorted(registration.SHEET_TAGS), [10, 11, 12, 13, 14, 15])
        for tag_id, (cx, cy) in registration.SHEET_TAGS.items():
            with self.subTest(tag=tag_id):
                self.assertEqual((cx % 50.0, cy % 50.0), (25.0, 25.0))
                corners = registration.sheet_corners_mm(tag_id)
                self.assertTrue(((corners >= 0) & (corners <= [250, 150])).all())
                # A quiet margin inside the cell on every side.
                self.assertGreaterEqual(50.0 - registration.SHEET_TAG_MM, 2 * 6.0)

    def test_corners_run_clockwise_from_the_top_left_with_y_up(self):
        corners = registration.sheet_corners_mm(10)
        self.assertEqual(corners.tolist(), [[7.0, 43.0], [43.0, 43.0], [43.0, 7.0], [7.0, 7.0]])

    def test_the_floor_tags_and_the_sheet_s_tags_share_no_id(self):
        self.assertFalse(set(registration.FLOOR_TAG_IDS) & set(registration.SHEET_TAGS))


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class RegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = picture()
        cls.seen = seen(cls.image)
        cls.result = register(cls.seen)

    def test_the_synthetic_picture_shows_all_eight_tags_and_every_outline_is_measured(self):
        found, outlines = self.seen
        self.assertEqual(sorted(found[0]), [0, 1, 10, 11, 12, 13, 14, 15])
        self.assertEqual(sorted(outlines[0]), [0, 1, 10, 11, 12, 13, 14, 15])

    def test_at_the_rig_s_view_the_detector_puts_corners_about_half_a_pixel_inside_the_outline(self):
        # The reason the frame is not fitted through the detector's corners. Measured at the rig on 2026-10-05: 0.6.
        self.assertTrue(self.result.ok, self.result.problems)
        self.assertGreater(self.result.detector_inset_px, 0.2)
        self.assertLess(self.result.detector_inset_px, 1.0)

    def test_the_frame_is_within_a_quarter_of_a_millimetre_of_the_truth_everywhere_on_the_grid(self):
        mean, worst = grid_error_mm(self.result)
        self.assertLess(mean, 0.12)
        self.assertLess(worst, 0.25)

    def test_fitted_through_the_detector_s_corners_the_same_frame_would_be_several_times_worse(self):
        found, _outlines = self.seen
        through_corners = registration.register(found, found, grid=GRID, floor_tags_mm=STATED)
        self.assertGreater(grid_error_mm(through_corners)[0], 2.0 * grid_error_mm(self.result)[0])
        self.assertGreater(through_corners.sheet_fit_mean_mm, 2.0 * self.result.sheet_fit_mean_mm)

    def test_what_a_registration_says_about_itself_is_small_when_all_is_well(self):
        result = self.result
        self.assertEqual(result.problems, ())
        self.assertEqual(result.notes, ())
        self.assertLess(result.sheet_fit_mean_mm, 0.3)
        self.assertLess(result.sheet_fit_max_mm, 0.6)
        self.assertLess(result.leave_one_out_max_mm, 1.0)
        for value in result.sheet_tag_scale:
            self.assertAlmostEqual(value, 1.0, delta=0.012)
        for tag_id in registration.FLOOR_TAG_IDS:
            for value in result.floor_tag_scale[tag_id]:
                self.assertAlmostEqual(value, 1.0, delta=0.012)

    def test_the_floor_tags_are_stored_by_their_outlines_where_they_lie(self):
        for tag_id in registration.FLOOR_TAG_IDS:
            truth = floor_corners(tag_id)
            stored = np.array(self.result.geometry["tags"][str(tag_id)]["corners"])
            centre = np.array(self.result.geometry["tags"][str(tag_id)]["centre"])
            with self.subTest(tag=tag_id):
                self.assertLess(float(np.linalg.norm(centre - truth.mean(axis=0))), 0.4)
                self.assertLess(float(np.abs(stored - truth).max()), 0.6)
                for side, true_side in zip(self.result.floor_sides_mm[tag_id], STATED[tag_id] * 2, strict=True):
                    self.assertAlmostEqual(side, true_side, delta=0.6)
                # The detector's own corners make the same tag smaller: that is why they are not what is stored.
                for side, true_side in zip(self.result.floor_detector_sides_mm[tag_id], STATED[tag_id] * 2,
                                           strict=True):
                    self.assertLess(side, true_side + 0.2)
                    self.assertGreater(side, true_side - 2.0)
            pixels = np.array(self.result.geometry["registration"]["floor_tag_corners_px"][str(tag_id)])
            self.assertLess(float(np.abs(pixels - to_px(RIG_VIEW, truth)).max()), 0.4)

    def test_the_geometry_has_the_form_the_rest_of_the_instrument_reads(self):
        geometry = self.result.geometry
        self.assertEqual(sorted(geometry["tags"]), ["0", "1"])
        self.assertEqual(geometry["grid"], {"cols": 5, "rows": 3, "cell_mm": 50.0})
        self.assertEqual(geometry["tag_mm"], 60.0)
        stored = geometry["registration"]
        self.assertEqual(stored["sheet_tags"], [10, 11, 12, 13, 14, 15])
        for key in ("sheet_fit_mean_mm", "sheet_fit_max_mm", "leave_one_out_mean_mm", "leave_one_out_max_mm",
                    "sheet_tag_scale", "detector_inset_px", "floor_tags_stated_mm", "floor_tag_sides_mm",
                    "floor_tag_detector_sides_mm", "floor_tag_scale", "single_frame_wander_mean_mm",
                    "single_frame_wander_max_mm", "notes", "sheet_tag_scale_each", "px_to_mm",
                    "floor_tag_corners_px"):
            self.assertIn(key, stored)
        self.assertNotIn("check_mean_mm", stored)   # the figure that could only repeat the sheet's fit is gone
        # The frame itself is kept, to hold the calibration against, and so are the floor tags' pixels.
        self.assertTrue(np.allclose(np.array(stored["px_to_mm"]), self.result.px_to_mm))
        self.assertEqual(np.array(stored["floor_tag_corners_px"]["0"]).shape, (4, 2))
        self.assertEqual(sorted(stored["sheet_tag_scale_each"]), ["10", "11", "12", "13", "14", "15"])

    def test_solved_again_from_the_floor_tags_alone_the_instrument_gets_the_same_frame_back(self):
        # What every session does. The floor tags' stored corners were made through the frame, so the frame comes
        # back whatever the tags' true size: this is a property of the method, and it is why it is no check of it.
        from tolerance_radius.rig.vision import TableView
        from tolerance_radius.schema import Point

        view = TableView.from_frames([self.image] * 3, self.result.geometry)
        for point in ((0.0, 0.0), (250.0, 150.0), (125.0, 75.0), (25.0, 125.0), (250.0, 0.0)):
            x_px, y_px = view.to_px(Point(*point))
            back = np.array([x_px, y_px, 1.0]) @ self.result.px_to_mm.T
            with self.subTest(point=point):
                self.assertLess(float(np.linalg.norm(back[:2] / back[2] - point)), 0.02)
                self.assertLess(float(np.linalg.norm(np.array([x_px, y_px]) - to_px(RIG_VIEW, [point])[0])), 0.4)

    def test_a_tag_turned_on_the_floor_is_measured_as_turned(self):
        floor = {0: (RIG_FLOOR[0][0], RIG_FLOOR[0][1], 20.0), 1: RIG_FLOOR[1]}
        result = register(picture(floor=floor))
        self.assertTrue(result.ok, result.problems)
        measured = np.array(result.geometry["tags"]["0"]["corners"])
        self.assertLess(float(np.abs(measured - floor_corners(0, floor)).max()), 1.3)
        self.assertGreater(float(np.abs(measured - floor_corners(0)).max()), 5.0)  # and not as it lay before

    def test_all_six_of_the_sheet_s_tags_are_needed_and_a_missing_one_is_named(self):
        with self.assertRaisesRegex(registration.RegistrationError, r"Missing: \[14\]"):
            register(picture(sheet_ids=(10, 11, 12, 13, 15)))
        with self.assertRaisesRegex(registration.RegistrationError, r"Missing: \[11, 14\]"):
            register(picture(sheet_ids=(10, 12, 13, 15)))
        # A tag the detector found but whose outline could not be measured is missing too.
        found, outlines = self.seen
        without = [{tag_id: quad for tag_id, quad in one.items() if tag_id != 13} for one in outlines]
        with self.assertRaisesRegex(registration.RegistrationError, r"Missing: \[13\]"):
            registration.register(found, without, grid=GRID)

    def test_a_missing_floor_tag_is_refused_by_name(self):
        with self.assertRaisesRegex(registration.RegistrationError, r"\[1\]"):
            register(picture(floor={0: RIG_FLOOR[0]}), floor_tags_mm=None)

    def test_no_frames_is_refused_and_so_are_outlines_from_other_frames(self):
        with self.assertRaises(registration.RegistrationError):
            registration.register([], [], grid=GRID)
        found, outlines = self.seen
        with self.assertRaisesRegex(registration.RegistrationError, "same frames"):
            registration.register(found, outlines[:-1], grid=GRID)

    def test_a_sheet_printed_at_the_wrong_scale_shows_in_the_floor_tags_size(self):
        # Printed at 93 % and taken as printed at 100 %: every distance measured through the sheet reads 1 / 0.93
        # too long. The sheet agrees with itself; only the floor tags, measured with calipers, can tell.
        result = register(picture(sheet_scale=0.93))
        self.assertFalse(result.ok)
        self.assertTrue(any("floor tag" in problem and "60 mm tag" in problem for problem in result.problems))
        for value in result.sheet_tag_scale:
            self.assertAlmostEqual(value, 1.0, delta=0.015)

    def test_a_floor_tag_that_reads_four_percent_long_is_said_and_stored(self):
        # The rig on 2026-10-05: floor tag 0 read 4 % long up the tag, as a tag on a slope of two degrees would.
        floor = {0: (RIG_FLOOR[0][0], (59.74, 62.4), RIG_FLOOR[0][2]), 1: RIG_FLOOR[1]}
        result = register(picture(floor=floor))
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(len(result.notes), 1)
        self.assertIn("floor tag 0", result.notes[0])
        # It is the one figure that does not rest on the sheet alone: the note must not wave it through.
        self.assertIn("only figure here that can see a sheet lying tilted", result.notes[0])
        self.assertIn("register again", result.notes[0])
        self.assertNotIn("not touched", result.notes[0])
        self.assertEqual(result.geometry["registration"]["notes"], list(result.notes))
        self.assertAlmostEqual(result.floor_tag_scale[0][1], 1.04, delta=0.012)
        self.assertAlmostEqual(result.floor_tag_scale[0][0], 1.0, delta=0.012)

    def test_a_floor_tag_ten_percent_off_is_refused(self):
        floor = {0: (RIG_FLOOR[0][0], (59.74, 66.0), RIG_FLOOR[0][2]), 1: RIG_FLOOR[1]}
        result = register(picture(floor=floor))
        self.assertFalse(result.ok)
        self.assertTrue(any("floor tag 0" in problem and "bent" in problem for problem in result.problems))

    def test_without_caliper_sizes_the_floor_tags_are_taken_as_sixty_millimetres(self):
        result = register(self.seen, floor_tags_mm=None)
        self.assertEqual(result.geometry["registration"]["floor_tags_stated_mm"], {"0": [60.0, 60.0], "1": [60.0, 60.0]})
        with self.assertRaisesRegex(registration.RegistrationError, "tag 0 or 1"):
            register(self.seen, floor_tags_mm={7: (60.0, 60.0)})

    def test_a_floor_tag_whose_outline_cannot_be_measured_is_refused(self):
        found, outlines = self.seen
        without = [{tag_id: quad for tag_id, quad in one.items() if tag_id != 1} for one in outlines]
        with self.assertRaisesRegex(registration.RegistrationError, r"\[1\] are in the picture but their outline"):
            registration.register(found, without, grid=GRID, floor_tags_mm=STATED)

    def test_a_lifted_corner_of_the_sheet_is_refused(self):
        # A corner of the sheet 4 mm off the floor shifts what the camera sees of it by about 3 mm, away from the
        # camera. Both the fit and the tag left out say so.
        image = picture(moved={12: (0.0, -3.0)})
        result = register(image)
        self.assertFalse(result.ok)
        # With the printed lines read, they say it: the frame is pulled off them by the tag that is out of place.
        self.assertTrue(any("printed lines" in problem and "four edges" in problem for problem in result.problems),
                        result.problems)
        self.assertGreater(result.lines["worst_mm"], registration.LINES_LIMIT_MM[1])
        # Without the picture the tags judge alone, by the fit and by the tag left out.
        alone = register(seen(image))
        self.assertIsNone(alone.lines)
        self.assertTrue(any("flat plane" in problem for problem in alone.problems), alone.problems)
        self.assertTrue(any("left out" in problem for problem in alone.problems), alone.problems)
        self.assertGreater(alone.sheet_fit_max_mm, registration.SHEET_FIT_LIMIT_MM[1])
        self.assertGreater(alone.leave_one_out_max_mm, registration.LEAVE_ONE_OUT_LIMIT_MM)

    def test_a_sheet_that_is_not_quite_flat_is_registered_and_said_to_be_so(self):
        # Between what a flat sheet gives and what cannot be used: stored, with a note that says what to do.
        result = register(seen(picture(moved={12: (0.0, -1.3)})))       # judged by its tags alone
        self.assertTrue(result.ok, result.problems)
        self.assertGreater(result.sheet_fit_max_mm, registration.SHEET_FIT_NOTE_MM[1])
        self.assertLess(result.sheet_fit_max_mm, registration.SHEET_FIT_LIMIT_MM[1])
        self.assertTrue(any("flat plane" in note and "four edges" in note for note in result.notes), result.notes)

    def test_the_tag_left_out_is_really_left_out(self):
        # A tag that sits 1.3 mm from its place pulls a fit it is part of toward itself, so its own misfit is the
        # smaller figure; predicted by the other five alone, it is off by about the whole 1.3 mm.
        result = register(seen(picture(moved={12: (0.0, -1.3)})))       # judged by its tags alone
        self.assertGreater(result.leave_one_out_max_mm, 1.25 * result.sheet_fit_max_mm)
        self.assertGreater(result.leave_one_out_max_mm, registration.LEAVE_ONE_OUT_NOTE_MM)
        self.assertTrue(any("left out" in note for note in result.notes), result.notes)

    # ---- the sheet's printed lines: the frame's error between the tags, measured, and the judge when readable

    @staticmethod
    def bulge(height_mm):
        """A bulge of the sheet's far outline around x = 75 mm, between two tags: seen off its place by this much."""
        return lambda x: height_mm * np.exp(-((x - 75.0) / 12.0) ** 2)

    def test_a_flat_sheet_s_printed_lines_lie_on_the_frame(self):
        result = register(picture())
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.notes, ())
        self.assertGreater(result.lines["scans"], 900)
        self.assertLess(result.lines["rms_mm"], registration.LINES_NOTE_MM[0])
        self.assertLess(result.lines["worst_mm"], registration.LINES_NOTE_MM[1])
        self.assertEqual(result.geometry["registration"]["sheet_lines"]["scans"], result.lines["scans"])

    def test_a_bulge_between_the_tags_shows_in_the_lines_and_nowhere_else(self):
        # The tags do not move, so the tags' own fit and the tag left out read as on a flat sheet.
        flat, bulged = register(picture()), register(picture(bump=self.bulge(1.0)))
        self.assertTrue(bulged.ok, bulged.problems)
        self.assertAlmostEqual(bulged.sheet_fit_max_mm, flat.sheet_fit_max_mm, delta=0.05)
        self.assertAlmostEqual(bulged.leave_one_out_max_mm, flat.leave_one_out_max_mm, delta=0.05)
        self.assertGreater(bulged.lines["worst_mm"], registration.LINES_NOTE_MM[1])
        self.assertIn("y = 150 mm, 70 to 80 mm", bulged.lines["worst_where"])
        noted = [note for note in bulged.notes if "printed lines" in note]
        self.assertEqual(len(noted), 1, bulged.notes)
        self.assertIn("four edges", noted[0])

    def test_a_larger_bulge_is_refused_by_the_lines(self):
        result = register(picture(bump=self.bulge(1.8)))
        self.assertFalse(result.ok)
        self.assertGreater(result.lines["worst_mm"], registration.LINES_LIMIT_MM[1])
        self.assertTrue(any("printed lines" in problem and "not lying flat there" in problem
                            for problem in result.problems), result.problems)

    def test_with_the_lines_read_the_small_tags_noise_is_no_note(self):
        # At the rig on 2026-10-06 a taped sheet read 0.39 and 0.88 mm for the tags' fit and 1.9 mm for a tag left
        # out, with its lines within 0.6 mm: figures that the tags alone would note, and nearly refuse.
        image = picture(moved={10: (0.5, 0.4), 13: (-0.4, 0.5), 15: (0.3, -0.5)})
        alone, with_lines = register(seen(image)), register(image)
        self.assertTrue(alone.notes, "this picture was meant to be in the tags' note band")
        self.assertTrue(with_lines.ok, with_lines.problems)
        self.assertFalse(any("flat plane" in note or "left out" in note for note in with_lines.notes))

    def test_lines_that_cannot_be_read_leave_the_tags_to_judge_and_say_so(self):
        found, outlines = self.seen
        blank = np.full((480, 640), 230, np.uint8)
        result = registration.register(found, outlines, grid=GRID, floor_tags_mm=STATED, picture=blank)
        self.assertIsNone(result.lines)
        self.assertTrue(any("printed lines could not be read" in note for note in result.notes), result.notes)
        self.assertIsNone(result.geometry["registration"]["sheet_lines"])

    def test_one_sheet_tag_that_reads_short_is_named_as_lying_on_a_slope(self):
        result = register(picture(resized={13: (1.0, 0.968)}))
        self.assertTrue(result.ok, result.problems)
        named = [note for note in result.notes if "tag 13" in note and "slope" in note]
        self.assertEqual(len(named), 1, result.notes)
        self.assertIn("up", named[0])
        self.assertNotIn("tag 10", named[0])
        self.assertAlmostEqual(result.geometry["registration"]["sheet_tag_scale_each"]["13"][1], 0.968, delta=0.012)
        self.assertAlmostEqual(result.geometry["registration"]["sheet_tag_scale_each"]["13"][0], 1.0, delta=0.012)

    def test_sheet_tags_that_all_read_small_are_refused_as_outlines_not_measured_rightly(self):
        small = {tag_id: (0.96, 0.96) for tag_id in range(10, 16)}
        result = register(picture(resized=small))
        self.assertFalse(result.ok)
        self.assertTrue(any("outlines are not being measured rightly" in problem for problem in result.problems))

    def test_the_caliper_sizes_are_across_and_up_the_tag_in_that_order(self):
        # A tag that is plainly not square: 54 mm across, 60 mm up.
        floor = {0: RIG_FLOOR[0], 1: (RIG_FLOOR[1][0], (54.0, 60.0), RIG_FLOOR[1][2])}
        image = picture(floor=floor)
        right = register(image, floor_tags_mm={0: STATED[0], 1: (54.0, 60.0)})
        self.assertTrue(right.ok, right.problems)
        for value in right.floor_tag_scale[1]:
            self.assertAlmostEqual(value, 1.0, delta=0.015)
        swapped = register(image, floor_tags_mm={0: STATED[0], 1: (60.0, 54.0)})
        self.assertFalse(swapped.ok)

    def test_a_sheet_size_far_from_what_printers_do_is_asked_about(self):
        # Nothing on the sheet can tell a mistyped size from a true one: the registration can only ask.
        typed = registration.sheet_scale_from_outline(252.0, 149.67)
        result = register(self.seen, sheet_scale=typed)
        self.assertTrue(any("Check the two numbers" in note for note in result.notes), result.notes)
        usual = registration.sheet_scale_from_outline(250.25, 149.67)
        self.assertFalse(any("Check the two numbers" in note for note in register(self.seen, sheet_scale=usual).notes))

    def test_two_conversions_are_compared_all_over_the_grid(self):
        lattice = registration.grid_lattice_mm(GRID)
        self.assertEqual(len(lattice), 11 * 7)
        self.assertEqual(lattice.min(axis=0).tolist(), [0.0, 0.0])
        self.assertEqual(lattice.max(axis=0).tolist(), [250.0, 150.0])
        frame = self.result.px_to_mm
        self.assertAlmostEqual(registration.frames_apart_mm(frame, frame, GRID), 0.0, places=9)
        moved = np.array([[1.0, 0.0, 1.5], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]) @ frame      # 1.5 and 2 mm aside
        self.assertAlmostEqual(registration.frames_apart_mm(frame, moved, GRID), 2.5, places=6)

    def test_a_tag_seen_in_too_few_frames_is_not_used(self):
        found, outlines = self.seen
        full, full_outline = found[0], outlines[0]
        without = {tag_id: quad for tag_id, quad in full.items() if tag_id != 1}
        with self.assertRaisesRegex(registration.RegistrationError, r"\[1\]"):
            registration.register([full] + [without] * 4, [full_outline] * 5, grid=GRID)

    def test_corners_are_averaged_over_the_frames_and_one_frame_s_noise_shows_as_wander(self):
        frames = seen_in([picture(seed=seed) for seed in range(6)])
        result = register(frames)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.frames, 6)
        self.assertLess(grid_error_mm(result)[1], 0.25)
        self.assertGreater(result.wander_max_mm, 0.0)
        self.assertLess(result.wander_max_mm, 0.8)
        self.assertLessEqual(result.wander_mean_mm, result.wander_max_mm)

    def test_a_floor_tag_corner_that_flickers_between_frames_is_refused(self):
        found, outlines = seen_in([picture(seed=seed) for seed in range(6)])
        jumpy = []
        for index, one in enumerate(outlines):
            one = {tag_id: quad.copy() for tag_id, quad in one.items()}
            if index % 2:
                one[1][2] += (8.0, -8.0)
            jumpy.append(one)
        result = registration.register(found, jumpy, grid=GRID, floor_tags_mm=STATED)
        self.assertFalse(result.ok)
        self.assertTrue(any("flickering" in problem for problem in result.problems))
        self.assertGreater(result.wander_mean_mm, registration.WANDER_MEAN_LIMIT_MM)


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class PrintedScaleTests(unittest.TestCase):
    """A printer rarely prints at exactly 100 % both ways. The sheet's measured size puts the table frame in true
    millimetres."""

    def test_the_scale_comes_from_the_outline_measured_edge_to_edge(self):
        # The outline is a line 0.21 mm wide: calipers on its outer edges read the grid plus one line width.
        line = registration.SHEET_OUTLINE_LINE_MM
        self.assertAlmostEqual(line, 0.2117, places=3)
        sx, sy = registration.sheet_scale_from_outline(250.0 + line, 150.0 + line)
        self.assertAlmostEqual(sx, 1.0, places=9)
        self.assertAlmostEqual(sy, 1.0, places=9)
        sx, sy = registration.sheet_scale_from_outline(250.25, 149.67)
        self.assertAlmostEqual(sx, 1.0002, places=4)
        self.assertAlmostEqual(sy, 0.9964, places=4)

    def test_a_size_that_is_not_this_sheet_at_full_scale_is_refused(self):
        for width, height in ((241.0, 145.0), (250.0, 0.0), (279.4, 215.9), (150.0, 250.0)):
            with self.subTest(size=(width, height)), self.assertRaises(registration.RegistrationError):
                registration.sheet_scale_from_outline(width, height)

    def test_a_tag_s_corners_follow_the_printed_scale(self):
        corners = registration.sheet_corners_mm(15, (1.0, 0.99))
        self.assertTrue(np.allclose(corners, registration.sheet_corners_mm(15) * [1.0, 0.99]))

    #: A sheet that came out 1.5 % short up the page.
    SHORT = (1.0, 0.985)

    def test_with_the_measured_scale_the_frame_and_the_floor_tags_come_out_true(self):
        told = register(picture(sheet_scale=self.SHORT), sheet_scale=self.SHORT)
        self.assertTrue(told.ok, told.problems)
        self.assertLess(grid_error_mm(told, sheet_scale=self.SHORT)[1], 0.25)
        for tag_id in registration.FLOOR_TAG_IDS:
            centre = np.array(told.geometry["tags"][str(tag_id)]["centre"])
            with self.subTest(tag=tag_id):
                self.assertLess(float(np.linalg.norm(centre - floor_corners(tag_id).mean(axis=0))), 0.5)
                for value in told.floor_tag_scale[tag_id]:
                    self.assertAlmostEqual(value, 1.0, delta=0.012)
        self.assertEqual(told.geometry["registration"]["sheet_scale"], [1.0, 0.985])

    def test_without_it_the_same_picture_stretches_the_floor_tags_in_that_direction(self):
        untold = register(picture(sheet_scale=self.SHORT))
        across, up = untold.floor_tag_scale[1]
        # Tag 1 lies square to the grid: across it is right, up it reads 1 / 0.985 too long.
        self.assertAlmostEqual(across, 1.0, delta=0.012)
        self.assertAlmostEqual(up, 1.0 / 0.985, delta=0.012)
        self.assertGreater(up - across, 0.008)

    def test_a_scale_is_two_positive_numbers_near_one(self):
        found, outlines = seen(picture())
        for scale in ((1.0,), (1.0, 0.0), (1.0, 1.2), (-1.0, 1.0)):
            with self.subTest(scale=scale), self.assertRaises((registration.RegistrationError, ValueError)):
                registration.register(found, outlines, grid=GRID, sheet_scale=scale)


@unittest.skipUnless(READY, "needs OpenCV and the arm tooling (run with the rig's interpreter)")
class RegisterCommandTests(unittest.TestCase):
    """`measure.py register-table` and `measure.py check`, with a camera that hands out a synthetic picture."""

    def setUp(self):
        import argparse
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest import mock

        from tolerance_radius.rig import camera_tool
        from tolerance_radius.rig import vision as vision_module

        self.camera_tool, self.vision = camera_tool, vision_module
        self.argparse, self.contextlib, self.io = argparse, contextlib, io
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)
        self.geometry_path = self.dir / "table_geometry.json"
        self.image = picture()
        test = self

        class FakeCamera:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return None

            def grab(self):
                return test.image.copy()

        for target, name, value in (
            (camera_tool, "Camera", FakeCamera),
            (camera_tool, "INSTRUMENT_DIR", self.dir / "instrument"),
            (camera_tool, "WRIST_LOG", self.dir / "logs" / "wrist_checks.log"),
            (camera_tool, "STORED_VIEW", self.dir / "instrument" / "table_view.json"),
            (vision_module, "TABLE_GEOMETRY", self.geometry_path),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_mode(self, mode, **options):
        args = self.argparse.Namespace(camera=None, **options)
        out = self.io.StringIO()
        with self.contextlib.redirect_stdout(out):
            code = mode(args)
        return code, out.getvalue()

    def register(self, **options):
        defaults = {"frames": 4, "out": None, "note": "synthetic", "replace": False, "sheet_outline_mm": None,
                    "floor_tags_mm": "59.74x60.0,59.78x60.0", "pictures": None}
        return self.run_mode(self.camera_tool.mode_register, **{**defaults, **options})

    def test_a_good_picture_is_registered_and_the_geometry_is_stored_where_the_instrument_reads_it(self):
        code, said = self.register()
        self.assertEqual(code, 0, said)
        self.assertIn("REGISTERED", said)
        geometry = self.vision.load_geometry()
        self.assertEqual(sorted(geometry["tags"]), ["0", "1"])
        self.assertEqual(geometry["note"], "synthetic")
        folder = self.dir / "instrument"
        pictures = [path for path in folder.glob("table_registration_*.png") if "_raw" not in path.name]
        self.assertEqual(len(pictures), 1)
        self.assertEqual(cv2.imread(str(pictures[0])).shape, (480, 640, 3))
        # Three of the frames are kept as they came: the registration can be worked through again from them.
        raw = sorted(folder.glob("table_registration_*_raw*.png"))
        self.assertEqual(len(raw), 3)
        self.assertTrue(np.array_equal(cv2.imread(str(raw[0])), self.image))
        for phrase in ("flat plane", "left out", "against their printed size", "inside its outline",
                       "one frame alone", "the white around the tags reads grey level"):
            self.assertIn(phrase, said)
        self.assertNotIn("NOTE", said)

    def test_the_same_registration_can_be_made_again_from_its_saved_frames(self):
        self.assertEqual(self.register()[0], 0)
        first = self.vision.load_geometry()
        folder = self.dir / "instrument"
        raw = sorted(str(path) for path in folder.glob("table_registration_*_raw*.png"))
        code, said = self.register(pictures=raw, out=str(self.dir / "again.json"))
        self.assertEqual(code, 0, said)
        self.assertIn("3 saved pictures", said)
        import json

        again = json.loads((self.dir / "again.json").read_text())
        self.assertEqual(again["tags"], first["tags"])
        self.assertEqual(len(list(folder.glob("table_registration_*_raw*.png"))), 3)   # no copies of the copies

    def test_a_saved_picture_that_is_missing_or_of_another_size_is_refused(self):
        small = self.dir / "small.png"
        cv2.imwrite(str(small), self.image[:240, :320])
        for name in (str(self.dir / "nothing.png"), str(small)):
            with self.subTest(name=name), self.assertRaises(self.vision.VisionError):
                self.register(pictures=[name])
        self.assertFalse(self.geometry_path.exists())

    def test_the_floor_tags_caliper_sizes_are_used_and_written_down(self):
        code, said = self.register()
        self.assertEqual(code, 0, said)
        stored = self.vision.load_geometry()["registration"]
        self.assertEqual(stored["floor_tags_stated_mm"], {"0": [59.74, 60.0], "1": [59.78, 60.0]})
        for text in ("59.7x60", "59.7x60,abc", "59.7x60,59.78x60,60x60", "59.7x60,-1x60"):
            code, said = self.register(floor_tags_mm=text, replace=True)
            with self.subTest(text=text):
                self.assertEqual(code, 1)
                self.assertIn("--floor-tags-mm", said)

    def test_a_floor_tag_that_reads_long_is_noted_and_the_table_is_still_registered(self):
        floor = {0: (RIG_FLOOR[0][0], (59.74, 62.4), RIG_FLOOR[0][2]), 1: RIG_FLOOR[1]}
        self.image = picture(floor=floor)
        code, said = self.register()
        self.assertEqual(code, 0, said)
        self.assertIn("NOTE: floor tag 0 reads", said)
        self.assertIn("REGISTERED WITH 1 NOTE(S)", said)
        self.assertIn("register again with --replace", said)
        self.assertEqual(len(self.vision.load_geometry()["registration"]["notes"]), 1)

    def test_an_over_exposed_picture_is_refused_by_that_name_and_not_as_a_sheet_that_does_not_fit(self):
        self.image = picture(white=1.6)
        code, said = self.register()
        self.assertEqual(code, 1)
        self.assertIn("over-exposed", said)
        self.assertNotIn("flat plane", said)
        self.assertFalse(self.geometry_path.exists())

    def calibrate(self, **options):
        defaults = {"frames": 6, "note": "synthetic", "replace": False}
        return self.run_mode(self.camera_tool.mode_calibrate, **{**defaults, **options})

    def test_the_calibration_is_solved_from_the_floor_tags_and_agrees_with_the_registered_frame(self):
        import json

        self.assertEqual(self.register()[0], 0)
        geometry = self.vision.load_geometry()
        self.image = picture(sheet_ids=())  # the sheet has been taken away
        code, said = self.calibrate()
        self.assertEqual(code, 0, said)
        self.assertIn("since the table was registered: floor tag 0 has moved by", said)
        self.assertNotIn("NOTE", said)
        stored = json.loads((self.dir / "instrument" / "table_view.json").read_text())
        apart = registration.frames_apart_mm(geometry["registration"]["px_to_mm"], stored["px_to_mm"], GRID)
        self.assertLess(apart, 0.1)
        self.assertEqual(stored["geometry_created"], geometry["created"])

    def test_a_floor_tag_moved_between_the_registration_and_the_calibration_is_refused(self):
        # Pressed flat, or bumped while the sheet was taken away: the two tags then give another frame, and the fit
        # of the two tags to the stored geometry alone would not say so.
        self.assertEqual(self.register()[0], 0)
        centre, size, turn = RIG_FLOOR[0]
        moved = {0: ((centre[0] + 3.0, centre[1] - 4.0), size, turn), 1: RIG_FLOOR[1]}
        self.image = picture(sheet_ids=(), floor=moved)
        code, said = self.calibrate()
        self.assertEqual(code, 1, said)
        self.assertIn("NOT STORED: the two floor tags have not moved alike", said)
        self.assertIn("register the table again", said)
        self.assertIn("Do not seat the sheet again yet", said)
        self.assertFalse((self.dir / "instrument" / "table_view.json").exists())
        # The frame it saw is kept all the same: it is what the registration's pictures are held against.
        self.assertEqual(len(list((self.dir / "instrument").glob("calibration_*.png"))), 1)

    def test_a_camera_that_turned_a_little_is_said_and_followed(self):
        # The first calibration at the rig, 2026-10-06: while the sheet was peeled off the camera turned by 0.11
        # degrees about its line of sight. One floor tag then moves a pixel against the other, as it would if it had
        # been moved on the floor by a millimetre; the pipes, the arm and the basket had turned with the tags. The
        # rule of that morning refused it.
        import json

        self.assertEqual(self.register()[0], 0)
        now = shifted(turned(RIG_VIEW, -0.11), dx=-0.76, dy=-0.42)
        self.image = picture(now, sheet_ids=())
        code, said = self.calibrate()
        self.assertEqual(code, 0, said)
        self.assertIn("NOTE: the two floor tags have not moved alike", said)
        self.assertIn("turned by -0.1", said)
        self.assertIn("cannot say which", said)
        stored = json.loads((self.dir / "instrument" / "table_view.json").read_text())
        # The tags lie where they lay, so the frame solved from them is right for the picture as it now is.
        self.assertLess(registration.frames_apart_mm(np.linalg.inv(now), stored["px_to_mm"], GRID), 0.5)
        since = stored["record"]["since_registration"]
        self.assertAlmostEqual(since["turn_deg"], -0.11, delta=0.03)
        self.assertGreater(since["differ_px"], 0.7)
        self.assertLess(since["relative_mm"], 1.5)
        # A session an hour later, the camera moved again as it did that day: the frame follows the tags, and the
        # view check is ready where it used to say that the view had moved.
        later = shifted(turned(RIG_VIEW, -0.09), dx=-0.4, dy=-0.1)
        self.image = picture(later, sheet_ids=())
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 0, said)
        self.assertIn("the table frame is taken from the two floor tags as they are now", said)
        self.assertIn("READY", said)
        # With a calibration stored, the check no longer says how to set the exposure: the picture is held against
        # the calibration's own levels. (On 2026-10-06 it told the operator DARK at every check after calibrating.)
        self.assertIn("the white around the tags reads grey level", said)
        self.assertNotIn("for a registration it should be", said)
        view = self.camera_tool.checked_stored_view(lambda: self.image.copy(), frames=3)
        self.assertLess(registration.frames_apart_mm(np.linalg.inv(later), view.px_to_mm, GRID), 0.5)
        self.assertEqual(sorted(stored["record"]["floor_tag_corners_px"]), ["0", "1"])
        self.assertTrue((self.dir / "instrument" / stored["record"]["picture"]).is_file())

    def test_a_larger_turn_of_the_camera_is_refused_until_it_is_known_what_moved(self):
        # The two tags cannot tell a camera that turned from a tag that was moved. Beyond the limit the calibration
        # waits for the pictures to be held against each other, and does not send the operator to tape the sheet.
        self.assertEqual(self.register()[0], 0)
        self.image = picture(turned(RIG_VIEW, -0.45), sheet_ids=())
        code, said = self.calibrate()
        self.assertEqual(code, 1, said)
        self.assertIn("NOT STORED: the two floor tags have not moved alike", said)
        self.assertIn("picture_motion.py", said)

    def test_a_floor_tag_knocked_after_the_calibration_stops_a_session(self):
        # The frame is the tags': a session follows a camera that moved, and must not follow a tag that did.
        self.assertEqual(self.register()[0], 0)
        self.image = picture(sheet_ids=())
        self.assertEqual(self.calibrate()[0], 0)
        centre, size, turn = RIG_FLOOR[1]
        knocked = {0: RIG_FLOOR[0], 1: ((centre[0] - 4.0, centre[1] + 3.0), size, turn)}
        self.image = picture(sheet_ids=(), floor=knocked)
        with self.assertRaisesRegex(self.vision.VisionError, "against each other since the table was registered"):
            self.camera_tool.checked_stored_view(lambda: self.image.copy(), frames=3)
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 1, said)
        self.assertIn("NOT READY: the two floor tags read", said)

    def test_a_camera_that_has_turned_past_the_limit_stops_a_session_until_the_operator_has_looked(self):
        # The figure is taken from the registration, whatever is calibrated afterwards: 1.05 mm of it were there at
        # the rig's calibration and 1.7 mm that evening, all of it the camera's turn (the kept frame against the
        # calibration's: the tags moved with the rest of the picture). At 3 mm every session would stop, with
        # nothing to be done but to register the table again.
        self.assertEqual(self.register()[0], 0)
        self.image = picture(sheet_ids=())
        self.assertEqual(self.calibrate()[0], 0)
        self.image = picture(turned(RIG_VIEW, -0.40), sheet_ids=())
        with self.assertRaisesRegex(self.vision.VisionError, "--tags-relative-limit sets another limit for one run"):
            self.camera_tool.checked_stored_view(lambda: self.image.copy(), frames=3)
        # the operator, having seen in the pictures that it is the camera:
        view = self.camera_tool.checked_stored_view(lambda: self.image.copy(), frames=3, relative_limit_mm=6.0)
        self.assertLess(registration.frames_apart_mm(np.linalg.inv(turned(RIG_VIEW, -0.40)), view.px_to_mm, GRID), 0.5)
        # ... and the limit given is a limit: further still is refused again
        self.image = picture(turned(RIG_VIEW, -0.80), sheet_ids=())
        with self.assertRaisesRegex(self.vision.VisionError, r"limit 6 mm"):
            self.camera_tool.checked_stored_view(lambda: self.image.copy(), frames=3, relative_limit_mm=6.0)

    def test_a_floor_tag_moved_by_a_millimetre_is_noted_and_what_it_would_cost_is_said(self):
        # What the rule gives up, said plainly: a tag moved by about a millimetre is not told from a camera that
        # turned, and is stored with a note. The frame is then off by no more than about that millimetre.
        import json

        self.assertEqual(self.register()[0], 0)
        centre, size, turn = RIG_FLOOR[1]
        moved = {0: RIG_FLOOR[0], 1: ((centre[0] + 0.7, centre[1] + 0.8), size, turn)}
        self.image = picture(sheet_ids=(), floor=moved)
        code, said = self.calibrate()
        self.assertEqual(code, 0, said)
        self.assertIn("NOTE: the two floor tags have not moved alike", said)
        self.assertIn("If it was a tag, positions on the grid are off by up to about", said)
        stored = json.loads((self.dir / "instrument" / "table_view.json").read_text())
        self.assertAlmostEqual(stored["record"]["since_registration"]["relative_mm"], 1.06, delta=0.3)
        self.assertLess(registration.frames_apart_mm(np.linalg.inv(RIG_VIEW), stored["px_to_mm"], GRID), 1.2)

    def test_a_camera_nudged_between_the_registration_and_the_calibration_is_said_and_followed(self):
        # Reaching in to peel the sheet moves a camera on a pipe by a pixel or two. Both tags then move together;
        # they lie where they lay, so the frame solved from them is the right one for the picture as it now is.
        import json

        self.assertEqual(self.register()[0], 0)
        nudged = shifted(RIG_VIEW, dx=1.4, dy=-0.9)
        self.image = picture(nudged, sheet_ids=())
        code, said = self.calibrate()
        self.assertEqual(code, 0, said)
        self.assertIn("NOTE: the camera's picture has moved by about 1.7 px", said)
        stored = json.loads((self.dir / "instrument" / "table_view.json").read_text())
        truth_now = np.linalg.inv(nudged)                       # pixels to table millimetres, as the camera now sees
        self.assertLess(registration.frames_apart_mm(truth_now, stored["px_to_mm"], GRID), 0.5)
        # The frame registered before the nudge would be off by the nudge itself.
        registered = self.vision.load_geometry()["registration"]["px_to_mm"]
        self.assertGreater(registration.frames_apart_mm(truth_now, registered, GRID), 1.2)

    def test_a_shift_of_a_tenth_of_a_pixel_is_nothing(self):
        # Through the detector's own corners this put the solve more than a millimetre from the frame.
        self.assertEqual(self.register()[0], 0)
        for angle in range(0, 360, 45):
            dx, dy = 0.15 * np.cos(np.radians(angle)), 0.15 * np.sin(np.radians(angle))
            self.image = picture(shifted(RIG_VIEW, dx=dx, dy=dy), sheet_ids=())
            code, said = self.calibrate(replace=True)
            with self.subTest(angle=angle):
                self.assertEqual(code, 0, said)
                self.assertNotIn("NOTE", said)

    def test_the_calibration_is_refused_while_the_sheet_is_still_on_the_floor(self):
        self.assertEqual(self.register()[0], 0)
        code, said = self.calibrate()           # the picture still shows the sheet
        self.assertEqual(code, 1)
        self.assertIn("the registration sheet is still on the floor", said)
        self.assertFalse((self.dir / "instrument" / "table_view.json").exists())

    def test_a_calibration_made_for_an_earlier_registration_is_not_measured_through(self):
        self.assertEqual(self.register()[0], 0)
        self.image = picture(sheet_ids=())
        self.assertEqual(self.calibrate()[0], 0)
        # The table is registered again and the calibration is forgotten.
        self.image = picture()
        import time

        time.sleep(1.1)                          # the two registrations' stamps are a second apart at least
        code, said = self.register(replace=True)
        self.assertEqual(code, 0, said)
        self.assertIn("calibrate --replace", said)
        self.image = picture(sheet_ids=())
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 1)
        self.assertIn("belongs to an earlier registration", said)
        with self.assertRaisesRegex(self.vision.VisionError, "earlier registration"):
            self.camera_tool.checked_stored_view(lambda: self.image.copy(),
                                                 path=self.dir / "instrument" / "table_view.json", frames=3)
        self.assertEqual(self.calibrate(replace=True)[0], 0)
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 0, said)

    def test_a_grid_too_near_the_picture_s_edge_is_refused_at_the_registration_not_after_it(self):
        self.image = picture(shifted(RIG_VIEW, dx=-16.0))
        code, said = self.register()
        self.assertEqual(code, 1, said)
        self.assertIn("NOT REGISTERED: the grid's corner E3 is", said)
        self.assertFalse(self.geometry_path.exists())

    def test_the_last_lines_say_one_thing(self):
        code, said = self.register()
        self.assertEqual(code, 0, said)
        self.assertIn("Next: take the sheet away", said)
        self.assertNotIn("LEAVE THE SHEET", said)
        floor = {0: (RIG_FLOOR[0][0], (59.74, 62.4), RIG_FLOOR[0][2]), 1: RIG_FLOOR[1]}
        self.image = picture(floor=floor)
        code, said = self.register(replace=True)
        self.assertEqual(code, 0, said)
        self.assertIn("LEAVE THE SHEET WHERE IT IS", said)
        self.assertNotIn("Next: take the sheet away", said)

    def test_a_floor_tag_that_comes_and_goes_is_noted(self):
        from unittest import mock

        with_tag, without_tag = picture(), picture(floor={0: RIG_FLOOR[0]})
        count = {"grabbed": 0}

        class Flickering:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return None

            def grab(self):
                count["grabbed"] += 1
                return (without_tag if count["grabbed"] % 4 == 0 else with_tag).copy()

        with mock.patch.object(self.camera_tool, "Camera", Flickering):
            code, said = self.register(frames=8)
        self.assertEqual(code, 0, said)
        self.assertIn("NOTE: floor tag 1 was found in only 6 of 8 frames", said)
        self.assertEqual(len(self.vision.load_geometry()["registration"]["notes"]), 1)

    def test_the_sheet_s_measured_outline_is_used_and_written_down(self):
        self.image = picture(sheet_scale=(1.0002, 0.9964))
        code, said = self.register(sheet_outline_mm="250.25x149.67")
        self.assertEqual(code, 0, said)
        stored = self.vision.load_geometry()["registration"]
        self.assertEqual(stored["sheet_outline_measured_mm"], [250.25, 149.67])
        self.assertAlmostEqual(stored["sheet_scale"][0], 1.0002, places=4)
        self.assertAlmostEqual(stored["sheet_scale"][1], 0.9964, places=4)
        self.assertIn("0.36 % short", said)

    def test_without_a_measured_outline_the_sheet_is_taken_at_its_design_and_the_output_says_so(self):
        code, said = self.register()
        self.assertEqual(code, 0, said)
        self.assertEqual(self.vision.load_geometry()["registration"]["sheet_scale"], [1.0, 1.0])
        self.assertIn("as designed", said)

    def test_an_outline_that_cannot_be_read_or_is_not_this_sheet_is_refused_before_the_camera_opens(self):
        for text in ("250.25", "250.25 by 149.67", "25.0x15.0"):
            code, said = self.register(sheet_outline_mm=text)
            with self.subTest(text=text):
                self.assertEqual(code, 1)
                self.assertIn("NOT REGISTERED", said)
                self.assertFalse(self.geometry_path.exists())

    def test_a_stored_geometry_is_not_overwritten_unless_asked(self):
        self.assertEqual(self.register()[0], 0)
        before = self.geometry_path.read_text()
        code, said = self.register(note="second")
        self.assertEqual(code, 1)
        self.assertIn("--replace", said)
        self.assertEqual(self.geometry_path.read_text(), before)
        self.assertEqual(self.register(note="second", replace=True)[0], 0)
        self.assertIn("second", self.geometry_path.read_text())

    def test_without_the_sheet_nothing_is_stored_and_a_picture_is_kept(self):
        self.image = picture(sheet_ids=())
        code, said = self.register()
        self.assertEqual(code, 1)
        self.assertIn("NOT REGISTERED", said)
        self.assertFalse(self.geometry_path.exists())
        kept = [path for path in (self.dir / "instrument").glob("table_registration_*.png") if "_raw" not in path.name]
        self.assertEqual(len(kept), 1)

    def test_a_doubtful_registration_is_not_stored(self):
        self.image = picture(sheet_scale=0.93)
        code, said = self.register()
        self.assertEqual(code, 1)
        self.assertIn("NOT REGISTERED", said)
        self.assertFalse(self.geometry_path.exists())

    def test_a_calibration_stored_earlier_is_said_to_be_out_of_date(self):
        stored = self.dir / "instrument" / "table_view.json"
        stored.parent.mkdir(parents=True)
        stored.write_text("{}")
        code, said = self.register()
        self.assertEqual(code, 0)
        self.assertIn("calibrate --replace", said)

    def test_before_registration_the_view_check_says_which_tags_it_sees_and_keeps_a_picture(self):
        self.image = picture(sheet_ids=())
        out = self.dir / "view.png"
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(out))
        self.assertEqual(code, 1)
        self.assertIn("not registered", said)
        self.assertIn("tag 0 (floor tag)", said)
        self.assertIn("tag 1 (floor tag)", said)
        self.assertIn("in view: both", said)
        self.assertTrue(out.is_file())
        # The line the exposure is set by, before anything is registered.
        self.assertIn("the white around the tags reads grey level", said)
        self.assertNotIn("OVER-EXPOSED", said)
        self.image = picture(sheet_ids=(), white=1.6)
        _code, said = self.run_mode(self.camera_tool.mode_check, out=str(out))
        self.assertIn("OVER-EXPOSED", said)
        # And the other way: an automatic exposure looking at a floor that is nearly all white.
        self.image = picture(sheet_ids=(), white=0.26)
        _code, said = self.run_mode(self.camera_tool.mode_check, out=str(out))
        self.assertIn("DARK: the white floor reads grey", said)
        self.assertNotIn("OVER-EXPOSED", said)

    def test_before_registration_a_missing_floor_tag_is_named(self):
        self.image = picture(sheet_ids=(), floor={0: RIG_FLOOR[0]})
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 1)
        self.assertIn("NOT [1]", said)

    def test_after_registration_the_view_check_places_the_grid_and_is_ready(self):
        self.assertEqual(self.register()[0], 0)
        self.image = picture(sheet_ids=())  # the sheet has been taken away
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 0, said)
        self.assertIn("READY", said)

    def test_after_registration_the_calibration_is_stored_with_the_floor_s_brightness_and_sessions_read_it(self):
        import json

        self.assertEqual(self.register()[0], 0)
        self.image = picture(sheet_ids=())  # the sheet has been taken away
        code, said = self.run_mode(self.camera_tool.mode_calibrate, frames=6, note="synthetic", replace=False)
        self.assertEqual(code, 0, said)
        self.assertIn("the picture's levels: the floor inside the grid at grey level", said)
        stored = json.loads((self.dir / "instrument" / "table_view.json").read_text())
        self.assertGreater(stored["floor_levels"]["grey"], 200.0)       # a white floor under the lamps
        self.assertLess(stored["floor_levels"]["tag_black"], 80.0)      # and the floor tags' black
        self.assertAlmostEqual(stored["floor_levels"]["red_over_blue"], 1.0, delta=0.02)
        # A later look through the camera agrees with it, and says how bright the floor is.
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 0, said)
        self.assertIn("READY", said)
        self.assertIn("the floor inside the grid: grey level", said)
        self.assertIn("the picture's contrast is 1.00 of the calibration's", said)
        self.assertNotIn("WARNING", said)
        # The same look after the camera's exposure has changed: the view is not ready, and it says why.
        self.image = (picture(sheet_ids=()).astype(float) * 0.8).astype(np.uint8)
        code, said = self.run_mode(self.camera_tool.mode_check, out=str(self.dir / "view.png"))
        self.assertEqual(code, 1, said)
        self.assertIn("NOT READY: the picture is darker than at the calibration", said)
        self.image = picture(sheet_ids=())
        # A second calibration is not made over the first unless asked.
        code, said = self.run_mode(self.camera_tool.mode_calibrate, frames=6, note="again", replace=False)
        self.assertEqual(code, 1)
        self.assertIn("--replace", said)

    def test_the_live_view_runs_until_the_operator_quits_and_keeps_the_last_picture(self):
        from unittest import mock

        keys = iter([255, ord("s"), 255, ord("x")])
        shown = []
        with (
            mock.patch.object(cv2, "imshow", lambda _title, picture: shown.append(picture.shape)),
            mock.patch.object(cv2, "waitKey", lambda _ms: next(keys)),
        ):
            code, said = self.run_mode(self.camera_tool.mode_live)
        self.assertEqual(len(shown), 4)
        self.assertEqual(set(shown), {(480, 640, 3)})
        folder = self.dir / "instrument"
        self.assertEqual(len(list(folder.glob("view_live_*.png"))), 1)   # the one asked for with s
        self.assertEqual(len(list(folder.glob("view_check_*.png"))), 1)  # the last one, always
        raw = sorted(folder.glob("view_raw_*.png"))                       # the frames themselves, unmarked
        self.assertIn(len(raw), (1, 2))                                   # two, unless both fell in one second
        self.assertTrue(np.array_equal(cv2.imread(str(raw[-1])), self.image))
        self.assertIn("tag 0 (floor tag)", said)
        self.assertIn(code, (0, 1))

    def test_the_live_view_and_the_logging_window_show_where_the_registered_grid_is(self):
        # The grid is not drawn on the floor. On 2026-10-06 the operator laid the ruler for the scale check half
        # outside it: the logging window showed the cube and nothing of the grid.
        from unittest import mock

        self.assertEqual(self.register()[0], 0)
        geometry = self.vision.load_geometry()
        tag_check = lerobot101.load("tag_check")
        with_sheet = self.image
        found = tag_check.detect(with_sheet)
        live = self.camera_tool.live_picture
        # With the sheet on the floor the live view is as it was: the sheet's own outline, not the registered grid.
        self.assertTrue(np.array_equal(live(with_sheet, found, tag_check, geometry), live(with_sheet, found, tag_check)))
        self.image = picture(sheet_ids=())
        found = tag_check.detect(self.image)
        plain, drawn = live(self.image, found, tag_check), live(self.image, found, tag_check, geometry)
        middle = np.round(to_px(RIG_VIEW, [(125.0, 75.0)])[0]).astype(int)      # inside cell C2, on the bare floor
        around = (slice(middle[1] - 30, middle[1] + 30), slice(middle[0] - 40, middle[0] + 40))
        self.assertFalse(np.array_equal(plain[around], drawn[around]))
        self.assertTrue(np.array_equal(plain[:100], drawn[:100]))               # nothing drawn off the grid
        # The logging window draws it through the stored conversion.
        self.assertEqual(self.calibrate()[0], 0)
        keys = iter([255, 255, ord("x")])
        shown = []
        table_view = self.camera_tool.table_view
        with (
            mock.patch.object(cv2, "imshow", lambda _title, shown_picture: shown.append(shown_picture.copy())),
            mock.patch.object(cv2, "waitKey", lambda _ms: next(keys)),
            # the operator's Enter, once the cube is off the table
            mock.patch.object(self.camera_tool, "table_view", lambda camera: table_view(camera, ask=lambda _text: "")),
        ):
            code, said = self.run_mode(self.camera_tool.mode_log, out=str(self.dir / "ruler.csv"), label="ruler_x")
        self.assertEqual(code, 0, said)
        self.assertEqual(len(shown), 3)
        self.assertFalse(np.array_equal(shown[0][around], self.image[around]))
        self.assertIn("the window shows the grid and its cells", said)
        self.assertEqual((self.dir / "ruler.csv").read_text().splitlines()[0], "index,label,x_mm,y_mm,scatter_mm,frames,time")

    def test_sharpness_is_largest_at_best_focus(self):
        sharp = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        values = [self.camera_tool.sharpness(sharp)]
        values += [self.camera_tool.sharpness(cv2.GaussianBlur(sharp, (0, 0), sigma)) for sigma in (1.0, 2.0, 4.0)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], 3 * values[2])
        self.assertEqual(self.camera_tool.sharpness(np.full((480, 640), 128, np.uint8)), 0.0)

    def test_a_minute_of_the_wrist_camera_says_its_tone_state_and_its_link_s_errors(self):
        # For trying a plug or a port before a sitting: on 2026-10-06 the wrist camera's link dropped about five
        # frames a minute and its picture was in its other tone state in 21 of 27 episodes; nothing showed either.
        from unittest import mock

        from fixtures_pictures import wrist_rest_picture
        from tolerance_radius import usb_errors

        verdict = self.camera_tool.wrist_verdict
        self.assertEqual(verdict([2.52, 2.55, 2.56] * 3 + [None], 0), (      # a frame or two not judged is no matter
            ["the floor over the jaws' green read 2.52 to 2.56 (2.50 to 2.64 in the usual tone state, 3.08 to 3.35 "
             "in the other): the usual tone state", "USB transfer errors on its link: none"], True))
        lines, good = verdict([3.20, 3.22, 3.21], 0)
        self.assertFalse(good)
        self.assertIn("THE OTHER TONE STATE", lines[0])
        lines, good = verdict([2.55] * 6 + [3.20] * 4, 0)
        self.assertFalse(good)
        self.assertIn("IT CHANGED ITS TONE STATE while this ran (4 of 10 frames", lines[0])
        lines, good = verdict([2.55] * 10, 5)
        self.assertFalse(good)
        self.assertIn("USB transfer errors on its link: 5. Frames are being dropped", lines[1])
        lines, good = verdict([2.55] * 10, None)
        self.assertTrue(good, "a log that cannot be read is said and does not make the camera bad")
        self.assertIn("not read", lines[1])
        lines, good = verdict([None] * 9 + [2.55], 0)
        self.assertFalse(good)
        self.assertIn("could not be judged in 9 of 10 frames", lines[0])
        self.assertEqual(verdict([], 0), (["the camera gave no frame"], False))

        def watch(picture, log_lines):
            self.image = picture
            asked = []

            def system_log(minutes, timeout_s=0.0):
                asked.append(minutes)
                return log_lines

            with mock.patch.object(usb_errors, "system_log", system_log):
                code, said = self.run_mode(self.camera_tool.mode_wrist, seconds=0.2)
            self.assertEqual(asked, [2])
            return code, said

        code, said = watch(wrist_rest_picture(), [])
        self.assertEqual(code, 0, said)
        self.assertIn("the usual tone state", said)
        self.assertIn("WRIST CAMERA GOOD", said)
        code, said = watch(wrist_rest_picture(stretched=True), [])
        self.assertEqual(code, 4, said)
        self.assertIn("WRIST CAMERA NOT GOOD", said)
        error = ("2099-01-01 00:00:00.000 E  kernel[0:1d0114c] (IOUSBHostFamily) AppleUSBIORequest: "
                 "AppleUSBIORequest::complete: device 1 (USB2.0_CAM1@02100000) endpoint 0x81: status 0xe00002ed "
                 "(transaction error)")
        code, said = watch(wrist_rest_picture(), [error, error.replace("2099", "2001"), error.replace("USB2.0_CAM1", "HD Pro Webcam C920")])
        self.assertEqual(code, 4, said)
        self.assertIn("USB transfer errors on its link: 1.", said)       # this camera's, since the command began
        # every run leaves one line, so that a morning's tries of plugs can be read afterwards
        kept = (self.dir / "logs" / "wrist_checks.log").read_text().splitlines()
        self.assertEqual(len(kept), 3)
        self.assertRegex(kept[0], r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \| camera \d+ \| \d+ frames \| GOOD \| ")
        self.assertIn("| NOT GOOD |", kept[1])
        self.assertIn("the usual tone state | USB transfer errors on its link: 1.", kept[2])

    def test_the_focus_view_shows_the_numbers_and_says_when_the_lens_was_left_short_of_the_peak(self):
        # The wrist camera's lens is turned by hand. The view gives a number to turn it by, and the picture it was
        # left on is kept.
        from unittest import mock

        sharp = self.image
        soft = cv2.GaussianBlur(sharp, (0, 0), 2.5)

        def run(frames, keys):
            frames, keys, shown = iter(frames), iter(keys), []
            self.image = next(frames)

            def wait(_ms):
                key = next(keys)
                self.image = next(frames, self.image)
                return key

            with (
                mock.patch.object(cv2, "imshow", lambda _title, picture: shown.append(picture.shape)),
                mock.patch.object(cv2, "waitKey", wait),
            ):
                code, said = self.run_mode(self.camera_tool.mode_focus)
            return code, said, shown

        # Turned to the peak and left there.
        code, said, shown = run([soft, sharp, sharp], [255, 255, ord("x")])
        self.assertEqual(code, 0, said)
        self.assertEqual(shown, [(480, 640, 3)] * 3)
        self.assertIn("Fix the lens where it is", said)
        kept = list((self.dir / "instrument").glob("focus_camera*.png"))
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(cv2.imread(str(kept[0])), sharp))       # as the camera gave it, unmarked
        # Turned past the peak and left soft.
        code, said, _shown = run([soft, sharp, soft], [255, 255, ord("x")])
        self.assertEqual(code, 1, said)
        self.assertIn("left short of the best", said)
        # r forgets the best so far: a peak seen while a hand was in the picture does not count.
        code, said, _shown = run([sharp, soft, soft], [255, ord("r"), ord("x")])
        self.assertEqual(code, 0, said)

    def test_a_view_is_judged_by_the_floor_tags_size_and_distance_from_the_edge(self):
        problems = self.camera_tool.view_problems
        big = np.array([[100.0, 100.0], [160.0, 100.0], [160.0, 160.0], [100.0, 160.0]])
        self.assertEqual(problems({0: big, 1: big + 200.0}), [])
        small = np.array([[100.0, 100.0], [137.0, 100.0], [137.0, 137.0], [100.0, 137.0]])
        told = problems({0: small, 1: big + 200.0})
        self.assertEqual(len(told), 1)
        self.assertIn("floor tag 0 is 37 px across", told[0])
        near = big - 90.0
        self.assertIn("px from the edge", problems({0: near, 1: big + 200.0})[0])
        self.assertIn("floor tag 1 is not in the picture", problems({0: big}))
        # A sheet tag does not count: the sheet is on the floor only for the registration.
        self.assertEqual(problems({0: big, 1: big + 200.0, 12: small}), [])

    def test_with_the_sheet_in_view_the_grid_must_be_clear_of_the_edge_too(self):
        tag_check = lerobot101.load("tag_check")
        centred = tag_check.detect(picture())
        corners = self.camera_tool.sheet_grid_px(centred)
        self.assertEqual(sorted(corners), ["A1", "A3", "E1", "E3"])
        expected = to_px(RIG_VIEW, np.array([(0.0, 0.0), (250.0, 150.0)]))
        self.assertAlmostEqual(corners["A1"][0], expected[0][0], delta=1.5)
        self.assertAlmostEqual(corners["E3"][1], expected[1][1], delta=1.5)
        self.assertFalse(any("grid's corner" in line for line in self.camera_tool.view_problems(centred)))
        # The same scene with the picture pushed sideways until the grid's E side is at the edge.
        pushed = tag_check.detect(picture(shifted(RIG_VIEW, dx=-30.0)))
        told = [line for line in self.camera_tool.view_problems(pushed) if "grid's corner" in line]
        self.assertTrue(told)
        self.assertTrue(all("corner E" in line for line in told))
        # Without the sheet there is no grid to judge.
        self.assertIsNone(self.camera_tool.sheet_grid_px({0: centred[0], 1: centred[1]}))

    def test_tags_are_described_by_size_edge_and_kind(self):
        found = lerobot101.load("tag_check").detect(self.image)
        lines = self.camera_tool.describe_tags(found)
        self.assertEqual(len(lines), 8)
        self.assertTrue(lines[0].startswith("tag 0 (floor tag)"))
        self.assertTrue(any(line.startswith("tag 12 (sheet tag)") for line in lines))
        near_edge = {0: np.array([[2.0, 100.0], [60.0, 100.0], [60.0, 160.0], [2.0, 160.0]])}
        self.assertIn("too near the edge", self.camera_tool.describe_tags(near_edge)[0])
        self.assertIn("NOT one of ours", self.camera_tool.describe_tags({7: near_edge[0] + 200.0})[0])

    def test_a_rig_without_a_geometry_says_what_to_do(self):
        with self.assertRaisesRegex(self.vision.VisionError, "00-table-layout"):
            self.vision.load_geometry()


if __name__ == "__main__":
    unittest.main()
