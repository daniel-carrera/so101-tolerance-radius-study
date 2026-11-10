"""A tag's outline measured from the picture, on synthetic pictures whose truth is known."""

import unittest

import numpy as np

from fixtures_pictures import RIG_VIEW, encode_srgb, pattern_bits, rectangle_mm, render, to_px
from tolerance_radius.rig import outline


def scene(view=RIG_VIEW, **options):
    """One tag-like square of 36 mm at the centre of the grid, seen as the rig sees a sheet tag: the picture, and the
    true corners in pixels."""
    corners_mm = rectangle_mm((125.0, 75.0), (36.0, 36.0), options.pop("turn_deg", 0.0))
    picture = render(view, {7: (corners_mm, pattern_bits(options.pop("pattern", 3)))}, **options)
    return picture, to_px(view, corners_mm)


def shrunk(quad, by_px):
    """A quad with every corner moved toward the centre: what the tag detector reports at this size."""
    centre = quad.mean(axis=0)
    away = quad - centre
    return quad - by_px * away / np.linalg.norm(away, axis=1, keepdims=True)


class LightTests(unittest.TestCase):
    def test_black_is_no_light_and_white_is_all_of_it(self):
        light = outline.linear_light(np.array([0, 255], dtype=np.uint8))
        self.assertEqual(light.tolist(), [0.0, 1.0])

    def test_the_middle_grey_level_is_about_a_fifth_of_the_light(self):
        self.assertAlmostEqual(float(outline.linear_light(np.array([128]))[0]), 0.216, places=3)

    def test_it_undoes_the_curve_the_pictures_are_stored_through(self):
        light = np.linspace(0.0, 1.0, 11)
        self.assertTrue(np.allclose(outline.linear_light(encode_srgb(light)), light, atol=1e-9))


class OutlineTests(unittest.TestCase):
    def test_a_sheet_tag_at_the_rig_s_view_is_about_thirty_pixels_across(self):
        _picture, truth = scene()
        sides = [np.linalg.norm(truth[k] - truth[(k + 1) % 4]) for k in range(4)]
        self.assertGreater(min(sides), 26.0)
        self.assertLess(max(sides), 38.0)

    def test_the_outline_is_found_to_a_tenth_of_a_pixel_from_corners_half_a_pixel_inside(self):
        picture, truth = scene(sharpen=0.0, noise=0.0)
        measured = outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6))
        self.assertIsNotNone(measured)
        self.assertLess(float(np.abs(measured.corners - truth).max()), 0.1)
        self.assertLess(max(measured.side_rms_px), 0.1)
        self.assertGreaterEqual(min(measured.side_points), outline.MIN_POINTS)

    def test_noise_and_a_webcam_s_sharpening_cost_a_few_tenths_of_a_pixel_at_most(self):
        for seed in range(4):
            picture, truth = scene(seed=seed, pattern=seed)
            measured = outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6))
            with self.subTest(seed=seed):
                self.assertIsNotNone(measured)
                self.assertLess(float(np.abs(measured.corners - truth).max()), 0.35)
                self.assertLess(float(np.linalg.norm(outline.crossing_of_diagonals(measured.corners)
                                                     - outline.crossing_of_diagonals(truth))), 0.12)

    def test_a_corner_the_detector_misplaced_by_two_pixels_does_not_matter(self):
        # Seen on 2026-10-05: one grey level's difference moved a detector corner of a sheet tag by 1.7 px.
        picture, truth = scene(sharpen=0.0, noise=0.0)
        start = shrunk(truth, 0.6)
        start[2] += (1.2, 1.6)
        measured = outline.tag_outline(outline.linear_light(picture), start)
        self.assertIsNotNone(measured)
        self.assertLess(float(np.abs(measured.corners - truth).max()), 0.12)

    def test_corners_that_are_all_two_pixels_off_the_same_way_do_not_matter_either(self):
        # The first pass must look that far for every side at once, not for one corner.
        picture, truth = scene(sharpen=0.0, noise=0.0)
        measured = outline.tag_outline(outline.linear_light(picture), truth + (1.6, -1.4))
        self.assertIsNotNone(measured)
        self.assertLess(float(np.abs(measured.corners - truth).max()), 0.12)

    def test_a_speck_on_one_side_is_left_out_of_that_side_s_line(self):
        picture, truth = scene(sharpen=0.0, noise=0.0)
        clean = outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6))
        # A white speck three pixels wide across the middle of the tag's upper side in the picture.
        top = truth[np.argsort(truth[:, 1])[:2]]
        x, y = top.mean(axis=0)
        speck = (x - 1.5, y - 2.5, x + 1.5, y + 2.5)
        picture, _ = scene(sharpen=0.0, noise=0.0, covered=[speck])
        measured = outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6))
        self.assertIsNotNone(measured)
        self.assertLess(float(np.abs(measured.corners - truth).max()), 0.15)
        self.assertLess(sum(measured.side_points), sum(clean.side_points))     # the speck's scans were dropped

    def test_a_region_of_one_level_gives_nothing_and_raises_nothing(self):
        quad = np.array([[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]])
        for level in (0.0, 0.5, 1.0):
            with self.subTest(level=level):
                self.assertIsNone(outline.tag_outline(np.full((480, 640), level), quad))
        self.assertIsNone(outline.tag_outline(np.full((480, 640), np.nan), quad))

    def test_the_white_around_a_tag_is_read_as_a_grey_level_and_clipping_shows(self):
        picture, truth = scene(sharpen=0.0, noise=0.0)
        self.assertGreater(outline.white_level(picture, truth), 230.0)
        self.assertLess(outline.white_level(picture, truth), outline.CLIPPED_GREY)
        burnt, _ = scene(sharpen=0.0, noise=0.0, white=1.5)
        self.assertGreaterEqual(outline.white_level(burnt, truth), outline.CLIPPED_GREY)
        dim, _ = scene(sharpen=0.0, noise=0.0, white=0.4)
        self.assertLess(outline.white_level(dim, truth), 180.0)

    def test_a_turned_tag_is_measured_as_well(self):
        picture, truth = scene(sharpen=0.0, noise=0.0, turn_deg=17.0)
        measured = outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.5))
        self.assertIsNotNone(measured)
        self.assertLess(float(np.abs(measured.corners - truth).max()), 0.12)

    def test_stored_grey_levels_put_the_outline_inside_the_tag_and_light_does_not(self):
        # The reason for `linear_light`: the middle grey level lies on the dark side of a blurred edge.
        picture, truth = scene(blur_px=0.7, sharpen=0.0, noise=0.0)
        by_light = outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.5))
        by_grey = outline.tag_outline(picture.astype(float) / 255.0, shrunk(truth, 0.5))
        self.assertLess(abs(outline.inset_px(by_light.corners, truth)), 0.06)
        self.assertGreater(outline.inset_px(by_grey.corners, truth), 0.15)

    def test_a_brighter_or_dimmer_picture_gives_the_same_outline(self):
        results = []
        for white in (0.5, 0.86):
            picture, truth = scene(white=white, sharpen=0.0, noise=0.0)
            results.append(outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6)).corners)
        self.assertLess(float(np.abs(results[0] - results[1]).max()), 0.08)

    def test_a_tag_with_a_side_covered_is_not_measured(self):
        picture, truth = scene(sharpen=0.0, noise=0.0)
        left = truth[:, 0].min()
        covered = (left - 4, truth[:, 1].min() - 4, left + 6, truth[:, 1].max() + 4)
        picture, _ = scene(sharpen=0.0, noise=0.0, covered=[covered])
        self.assertIsNone(outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6)))

    def test_a_tag_with_too_little_contrast_is_not_measured(self):
        picture, truth = scene(white=0.5, black=0.42, sharpen=0.0, noise=0.0)
        self.assertIsNone(outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6)))

    def test_a_tag_too_small_for_its_border_to_be_seen_is_not_measured(self):
        tiny = np.array([[100.0, 100.0], [112.0, 100.0], [112.0, 112.0], [100.0, 112.0]])
        self.assertIsNone(outline.tag_outline(np.full((480, 640), 0.8), tiny))

    def test_corners_that_are_not_on_a_tag_at_all_give_nothing(self):
        picture, truth = scene(sharpen=0.0, noise=0.0)
        self.assertIsNone(outline.tag_outline(outline.linear_light(picture), truth + 150.0))

    def test_a_tag_at_the_very_edge_of_the_picture_is_not_measured(self):
        # Its scans would leave the picture; a made-up edge there would be worse than none.
        # One pixel per millimetre, the tag's left side 3 px from the picture's left edge.
        view = np.array([[1.0, 0.0, 3.0 - 107.0], [0.0, 1.0, 100.0], [0.0, 0.0, 1.0]])
        picture, truth = scene(view=view, sharpen=0.0, noise=0.0)
        self.assertLess(truth[:, 0].min(), 5.0)             # the tag's left side is within 5 px of the picture's edge
        self.assertGreater(truth[:, 0].min(), 0.0)          # and still inside it
        self.assertIsNone(outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6)))
        # The same tag a little further in is measured.
        inner = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ view
        picture, truth = scene(view=inner, sharpen=0.0, noise=0.0)
        self.assertIsNotNone(outline.tag_outline(outline.linear_light(picture), shrunk(truth, 0.6)))

    def test_every_tag_of_a_picture_is_measured_and_a_bad_one_is_left_out(self):
        picture, truth = scene(sharpen=0.0, noise=0.0)
        found = {7: shrunk(truth, 0.6), 8: truth + 150.0}
        outlines = outline.tag_outlines(picture, found)
        self.assertEqual(sorted(outlines), [7])
        self.assertLess(float(np.abs(outlines[7] - truth).max()), 0.1)


class GeometryTests(unittest.TestCase):
    SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

    def test_a_square_s_diagonals_cross_at_its_centre(self):
        self.assertTrue(np.allclose(outline.crossing_of_diagonals(self.SQUARE), [5.0, 5.0]))

    def test_seen_in_perspective_the_crossing_is_the_centre_s_picture_and_the_corners_mean_is_not(self):
        corners_mm = rectangle_mm((125.0, 75.0), (60.0, 60.0))
        seen = to_px(RIG_VIEW, corners_mm)
        centre = to_px(RIG_VIEW, [(125.0, 75.0)])[0]
        self.assertLess(float(np.linalg.norm(outline.crossing_of_diagonals(seen) - centre)), 1e-6)
        self.assertGreater(float(np.linalg.norm(seen.mean(axis=0) - centre)), 0.1)

    def test_inset_is_how_far_inside_the_outline_the_corners_are(self):
        self.assertAlmostEqual(outline.inset_px(shrunk(self.SQUARE, 0.5), self.SQUARE), 0.5, places=9)
        self.assertAlmostEqual(outline.inset_px(shrunk(self.SQUARE, -0.3), self.SQUARE), -0.3, places=9)


if __name__ == "__main__":
    unittest.main()
