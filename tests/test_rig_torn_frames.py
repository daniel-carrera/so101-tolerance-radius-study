"""Torn camera frames (`tolerance_radius.rig.torn_frames`): the rule on synthetic pictures, then a dataset read.

The rule's own tests need NumPy only and run in both environments. The pictures are made here: a white floor seen from
above with a rail, a basket, two tags, a cube and an arm that moves, and a wall of holes passing under a camera on
the wrist. A tear is put into one frame the way the rig's cameras were seen to make them (2026-10-06: the lower part
or a band of the picture shifted sideways, or with its colour or brightness wrong) and the way a stale buffer would.
The tests that read a dataset write a small one into a temporary folder and need PyAV and pandas: they skip in the
development environment. No test reads a recorded dataset.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tolerance_radius.rig import torn_frames

HAVE_VIDEO = all(importlib.util.find_spec(name) is not None for name in ("av", "pandas", "pyarrow"))
REPO = Path(__file__).resolve().parents[1]

HEIGHT, WIDTH = 480, 640
#: Y, U, V of things in the pictures, as the front camera of October 2026 shows them (read from a recorded frame).
FLOOR = (205, 129, 125)
BLACK = (22, 128, 128)
WHITE = (228, 128, 128)
BASKET = (58, 128, 128)
ORANGE = (150, 95, 185)

Planes = tuple[np.ndarray, np.ndarray, np.ndarray]


def plain(colour: tuple[int, int, int] = FLOOR) -> Planes:
    """A picture of one colour: Y at full size, U and V at half size, as a camera's frame is stored."""
    return (np.full((HEIGHT, WIDTH), colour[0], np.uint8), np.full((HEIGHT // 2, WIDTH // 2), colour[1], np.uint8),
            np.full((HEIGHT // 2, WIDTH // 2), colour[2], np.uint8))


def box(planes: Planes, top: int, left: int, height: int, width: int, colour: tuple[int, int, int]) -> None:
    """Paint a rectangle, given in the full picture's pixels, clipped to the picture."""
    y, u, v = planes
    bottom, right = max(0, min(HEIGHT, top + height)), max(0, min(WIDTH, left + width))
    top, left = max(0, min(HEIGHT, top)), max(0, min(WIDTH, left))
    y[top:bottom, left:right] = colour[0]
    u[top // 2:(bottom + 1) // 2, left // 2:(right + 1) // 2] = colour[1]
    v[top // 2:(bottom + 1) // 2, left // 2:(right + 1) // 2] = colour[2]


def tag(planes: Planes, top: int, left: int) -> None:
    """A black square of 60 pixels with white cells: a floor tag."""
    box(planes, top, left, 60, 60, BLACK)
    for row, column in ((1, 1), (1, 3), (2, 2), (3, 1), (3, 4), (4, 3)):
        box(planes, top + 10 * row, left + 10 * column, 10, 10, WHITE)


def front_view(step: float, *, arm: bool = True, zoom: float = 1.0, brighter: int = 0, noise_seed: int | None = None
               ) -> Planes:
    """The table from above at one moment: the arm comes down from the top edge and moves to the right."""
    planes = plain()

    def at(value: float) -> int:
        return 2 * int(round(value * zoom / 2))

    box(planes, 0, 0, at(38), WIDTH, BLACK)
    box(planes, at(10), 0, at(12), WIDTH, WHITE)                         # the rail
    box(planes, at(60), at(440), at(200), at(160), BASKET)
    for row in range(70, 250, 12):                                        # the holes in the basket's walls
        for column in (446, 458, 470, 570, 582):
            box(planes, at(row), at(column), 4, 6, WHITE)
    tag(planes, at(150), at(40))
    tag(planes, at(330), at(470))
    box(planes, at(300), at(180), at(30), at(30), ORANGE)                 # the cube
    if arm:
        box(planes, 0, at(200 + 6 * step), at(150 + 4 * step), at(60), ORANGE)
        box(planes, at(40), at(215 + 6 * step), at(30), at(30), BLACK)   # a motor on it
    y, u, v = planes
    if brighter:
        y[:] = np.clip(y.astype(np.int16) + brighter, 0, 255).astype(np.uint8)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        y[:] = np.clip(y + rng.normal(0.0, 1.5, y.shape), 0, 255).astype(np.uint8)
    return planes


def wrist_view(shift_px: float) -> Planes:
    """A dark wall of bright holes, 22 pixels apart, passing under a camera on the wrist, over a plain floor of the
    basket; the jaws stand still at the bottom of the picture."""
    planes = plain((120, 132, 122))
    box(planes, 0, 0, 200, WIDTH, BASKET)
    offset = int(round(shift_px))
    for row in range(8, 190, 22):
        for column in range(-44 + offset % 22, WIDTH, 22):
            box(planes, row, column, 12, 12, WHITE)
    box(planes, 380, 150, 100, 70, ORANGE)
    return planes


def picture(planes: Planes) -> np.ndarray:
    return torn_frames.half_picture(*planes)


def take(count: int = 14, **kwargs) -> list[Planes]:
    """A take of the front view: the arm moves a little every frame."""
    return [front_view(float(step), **kwargs) for step in range(count)]


def found(frames: list[Planes]) -> dict[int, torn_frames.Tear]:
    return dict(torn_frames.scan_pictures(picture(planes) for planes in frames))


def sheared(planes: Planes, row: int, shift_px: int, end: int = HEIGHT) -> Planes:
    """The rows from `row` on shifted sideways, what went out on one side coming in on the other."""
    y, u, v = (plane.copy() for plane in planes)
    y[row:end] = np.roll(y[row:end], shift_px, axis=1)
    u[row // 2:end // 2] = np.roll(u[row // 2:end // 2], shift_px // 2, axis=1)
    v[row // 2:end // 2] = np.roll(v[row // 2:end // 2], shift_px // 2, axis=1)
    return y, u, v


def tinted(planes: Planes, row: int, end: int = HEIGHT, *, y_step: int = 0, u_step: int = 0, v_step: int = 0) -> Planes:
    """The rows from `row` to `end` with a step on a plane: a colour cast, or a brighter or darker band."""
    out = []
    for plane, step, scale in zip(planes, (y_step, u_step, v_step), (1, 2, 2), strict=True):
        changed = plane.astype(np.int16)
        changed[row // scale:end // scale] += step
        out.append(np.clip(changed, 0, 255).astype(np.uint8))
    return out[0], out[1], out[2]


def chroma_out_of_step(planes: Planes, row: int, shift_px: int) -> Planes:
    """The colour planes of the rows from `row` on shifted sideways against the brightness plane."""
    y, u, v = (plane.copy() for plane in planes)
    u[row // 2:] = np.roll(u[row // 2:], shift_px // 2, axis=1)
    v[row // 2:] = np.roll(v[row // 2:], shift_px // 2, axis=1)
    return y, u, v


def stale(planes: Planes, older: Planes, row: int) -> Planes:
    """The rows from `row` on as an earlier frame had them: a buffer that was not filled to its end."""
    out = []
    for plane, old, scale in zip(planes, older, (1, 2, 2), strict=True):
        mixed = plane.copy()
        mixed[row // scale:] = old[row // scale:]
        out.append(mixed)
    return out[0], out[1], out[2]


class HalfPictureTests(unittest.TestCase):
    def test_the_brightness_is_averaged_two_by_two_and_the_colour_planes_are_kept(self):
        y = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
        u = np.array([[1, 2], [3, 4]], np.uint8)
        v = np.array([[5, 6], [7, 8]], np.uint8)
        half = torn_frames.half_picture(y, u, v)
        self.assertEqual(half.shape, (2, 2, 3))
        self.assertEqual(half.dtype, np.uint8)
        self.assertEqual(half[..., 0].tolist(), [[25, 45], [105, 125]])
        self.assertEqual(half[..., 1].tolist(), u.tolist())
        self.assertEqual(half[..., 2].tolist(), v.tolist())

    def test_planes_of_other_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            torn_frames.half_picture(np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8), np.zeros((2, 2), np.uint8))


class TornFramesAreFoundTests(unittest.TestCase):
    """One damaged frame in a take, or a few in a row: named, and only they."""

    def one_torn(self, change, at: int = 6, **kwargs) -> torn_frames.Tear:
        frames = take(**kwargs)
        frames[at] = change(frames[at])
        tears = found(frames)
        self.assertEqual(sorted(tears), [at])
        return tears[at]

    def test_a_lower_part_shifted_sideways_is_found(self):
        tear = self.one_torn(lambda planes: sheared(planes, 200, 96))
        self.assertEqual(tear.held_against, (-2, -1, 1, 2))
        self.assertLessEqual(abs(tear.first_row_px - 200), 8)
        self.assertGreaterEqual(tear.last_row_px, 380, "down to the lower tag, the last thing that stands in its rows")

    def test_a_small_shift_sideways_is_found_where_something_stands_in_its_rows(self):
        # 16 pixels, one block of the camera's own compression: the smallest shift seen (August 2026)
        tear = self.one_torn(lambda planes: sheared(planes, 96, 16))
        self.assertEqual(tear.kind, "displaced")
        self.assertGreaterEqual(tear.first_row_px, 96)

    def test_a_colour_cast_below_a_row_is_found(self):
        tear = self.one_torn(lambda planes: tinted(planes, 304, u_step=30, v_step=-25))
        self.assertEqual((tear.kind, tear.first_row_px, tear.last_row_px), ("band", 304, 480))

    def test_a_brighter_lower_part_is_found(self):
        tear = self.one_torn(lambda planes: tinted(planes, 88, y_step=14))
        self.assertEqual((tear.kind, tear.first_row_px, tear.last_row_px), ("band", 88, 480))

    def test_colour_planes_out_of_step_with_the_brightness_are_found(self):
        tear = self.one_torn(lambda planes: chroma_out_of_step(planes, 200, 48))
        self.assertGreaterEqual(tear.first_row_px, 200)
        self.assertLessEqual(tear.last_row_px, 340, "the cube is the one coloured thing in those rows")

    def test_a_stale_lower_part_is_found(self):
        for back in (2, 3, 4):
            with self.subTest(frames_before=back):
                frames = take()
                frames[7] = stale(frames[7], frames[7 - back], 60)
                tears = found(frames)
                self.assertEqual(sorted(tears), [7])
                self.assertEqual(tears[7].kind, "stale")
                self.assertGreaterEqual(tears[7].first_row_px, 60)

    def test_a_damaged_band_at_the_top_with_the_rest_right_is_found(self):
        # as the front camera gave them on 2026-10-06: the top 24 rows tinted; the top 88 shifted and washed out
        tear = self.one_torn(lambda planes: tinted(planes, 0, 24, u_step=-28, v_step=40))
        self.assertEqual((tear.kind, tear.first_row_px, tear.last_row_px), ("band", 0, 24))
        tear = self.one_torn(lambda planes: tinted(sheared(planes, 0, -160, 88), 0, 88, y_step=60))
        self.assertEqual((tear.first_row_px, tear.last_row_px), (0, 88))

    def test_a_band_of_eight_rows_is_enough_and_one_of_four_is_not_claimed(self):
        tear = self.one_torn(lambda planes: tinted(planes, 472, u_step=30))
        self.assertEqual((tear.first_row_px, tear.last_row_px), (472, 480))
        frames = take()
        frames[6] = tinted(frames[6], 476, u_step=30)
        self.assertEqual(found(frames), {})

    def test_torn_frames_in_a_row_are_each_found(self):
        for run in (2, 3):
            with self.subTest(frames_in_a_row=run):
                frames = take(16)
                for index in range(6, 6 + run):
                    frames[index] = tinted(frames[index], 0, 24, u_step=-28, v_step=40)     # the same damage in each
                self.assertEqual(sorted(found(frames)), list(range(6, 6 + run)))
                frames = take(16)
                for place, index in enumerate(range(6, 6 + run)):
                    frames[index] = sheared(frames[index], 64 + 32 * place, 96 + 32 * place)   # another in each
                self.assertEqual(sorted(found(frames)), list(range(6, 6 + run)))

    def test_a_torn_picture_that_comes_again_is_found_every_time(self):
        # 2026-10-06, August's set: a damaged frame, then the same picture up to four times more, the arm moving on
        for times in (2, 3, 6):
            for change in (lambda planes: tinted(planes, 0, 24, u_step=-28, v_step=40),
                           lambda planes: sheared(planes, 96, 16)):
                with self.subTest(times=times):
                    frames = take(20)
                    torn = change(frames[6])
                    for index in range(6, 6 + times):
                        frames[index] = torn
                    tears = found(frames)
                    self.assertEqual(sorted(tears), list(range(6, 6 + times)))
                    self.assertEqual(tears[6].held_against, (-2, -1, times, times + 1))
                    self.assertEqual(tears[5 + times].held_against, (-times - 1, -times, 1, 2))

    def test_a_torn_picture_that_comes_again_coded_another_way_is_found_both_times(self):
        # a video codes a frame after a sudden change coarsely and the same picture finely the frame after
        frames = take(16)
        torn = tinted(sheared(frames[6], 200, 96), 200, u_step=25)
        coarse = (np.clip(torn[0].astype(np.int16) + np.random.default_rng(3).integers(-2, 3, torn[0].shape), 0, 255)
                  .astype(np.uint8), torn[1], torn[2])
        frames[6], frames[7], frames[8] = coarse, torn, torn
        self.assertEqual(sorted(found(frames)), [6, 7, 8])

    def test_a_torn_first_frame_and_a_torn_last_frame_are_found(self):
        for at in (0, 1, 12, 13):
            with self.subTest(frame=at):
                frames = take(arm=False)                       # the arm is at rest when a take starts and ends
                frames[at] = tinted(frames[at], 240, u_step=30, v_step=-25)
                self.assertEqual(sorted(found(frames)), [at])

    def test_a_tear_is_found_under_the_camera_s_own_noise(self):
        frames = [front_view(float(step), noise_seed=step) for step in range(14)]
        frames[6] = sheared(frames[6], 200, 96)
        self.assertEqual(sorted(found(frames)), [6])

    def test_a_cast_is_found_in_the_wrist_camera_s_moving_picture(self):
        frames = [wrist_view(7.0 * step) for step in range(14)]
        frames[6] = tinted(frames[6], 240, u_step=30, v_step=-25)
        tears = found(frames)
        self.assertEqual(sorted(tears), [6])
        self.assertEqual((tears[6].first_row_px, tears[6].last_row_px), (240, 480))


class TheCountsBehindTheVerdictTests(unittest.TestCase):
    """The rule's parts on small made-up maps, where every number can be checked by hand."""

    def test_a_band_needs_a_quarter_of_the_width_in_four_rows_running(self):
        still = np.ones((40, 100), bool)
        broken = np.zeros((40, 100), bool)
        broken[10:14, :25] = True
        tear = torn_frames.judge(still, broken)
        self.assertEqual((tear.kind, tear.first_row_px, tear.last_row_px, tear.pixels), ("band", 20, 28, 400))
        broken[12, :] = False
        self.assertIsNone(torn_frames.judge(still, broken), "two rows, a gap, one row")
        broken[10:14, :24] = True
        broken[10:14, 24:] = False
        self.assertIsNone(torn_frames.judge(still, broken), "four rows of 24 in 100")

    def test_displaced_things_need_rows_where_nothing_else_moves(self):
        still = np.ones((40, 100), bool)
        broken = np.zeros((40, 100), bool)
        broken[5:30, 40:50] = True                                      # 25 rows of 10: 250 pixels
        tear = torn_frames.judge(still, broken)
        self.assertEqual((tear.kind, tear.first_row_px, tear.last_row_px, tear.pixels), ("displaced", 10, 60, 1000))
        self.assertIsNone(torn_frames.judge(still, broken, displaced_pixels=251))
        moving = still.copy()
        moving[5:30, 60:71] = False                                     # 11 pixels of every such row are not still
        self.assertIsNone(torn_frames.judge(moving, broken), "something moves through those rows: not damage")
        wide_still = np.ones((40, 200), bool)
        flat = np.zeros((40, 200), bool)
        flat[5:12, 10:46] = True                                        # 7 rows of 36: 252 pixels, under a quarter wide
        self.assertIsNone(torn_frames.judge(wide_still, flat), "an edge that swung out: too few rows")
        flat[5:13, 10:46] = True
        self.assertEqual(torn_frames.judge(wide_still, flat).kind, "displaced")

    def test_a_picture_not_held_against_the_four_frames_next_to_it_needs_quiet_rows(self):
        still = np.ones((40, 100), bool)
        broken = np.zeros((40, 100), bool)
        broken[10:14, :40] = True                                       # 4 rows of 40 in 100
        self.assertEqual(torn_frames.judge(still, broken, guarded=True).kind, "band")
        moving = still.copy()
        moving[10:14, 60:87] = False                            # 27 pixels of each row are not still: 40 < 1.5 x 27
        self.assertIsNone(torn_frames.judge(moving, broken, guarded=True), "a wall of holes that stands where it stood")
        self.assertEqual(torn_frames.judge(moving, broken).kind, "band", "next to its neighbours that is a band")
        broken[:] = False
        broken[5:30, 40:50] = True                                      # 250 pixels in rows where nothing moves
        self.assertEqual(torn_frames.judge(still, broken, guarded=True).kind, "displaced")
        moving = still.copy()
        moving[5:30, 60:67] = False                                   # 7 pixels of each row are not still: 10 < 1.5 x 7
        self.assertEqual(torn_frames.judge(moving, broken).kind, "displaced")
        self.assertIsNone(torn_frames.judge(moving, broken, guarded=True))

    def test_a_picture_is_held_against_two_different_pictures_on_each_side(self):
        frames = list(range(10))
        self.assertEqual(torn_frames.references_for(frames, 5, 10, 1, 1), [3, 4, 6, 7])
        self.assertEqual(torn_frames.references_for(frames, 5, 10, 1, 2), [3, 4, 7, 8])
        self.assertEqual(torn_frames.references_for(frames, 5, 10, 3, 1), [1, 2, 6, 7])
        self.assertEqual(torn_frames.references_for(frames, 0, 10, 1, 1), [1, 2, 3, 4], "the first frame: four after")
        self.assertEqual(torn_frames.references_for(frames, 9, 10, 1, 1), [5, 6, 7, 8])
        self.assertEqual(torn_frames.references_for(frames, 8, 10, 1, 1), [5, 6, 7, 9], "one after, three before")
        # a picture that came three times (frames 4, 5 and 6): one picture, for itself and for those next to it
        came_again = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
        self.assertEqual(torn_frames.references_for(came_again, 4, 12, 1, 1), [2, 3, 7, 8])
        self.assertEqual(torn_frames.references_for(came_again, 5, 12, 1, 1), [3, 6, 8, 9], "not frames 5 and 6")
        self.assertEqual(torn_frames.references_for(came_again, 3, 12, 1, 2), [1, 2, 7, 8], "stepped over as one")
        # a scene at rest: one picture before and one after, each for many frames
        at_rest = [0, 6, 7]
        self.assertEqual(torn_frames.references_for(at_rest, 1, 14, 1, 1), [4, 5, 7, 8])
        # nothing beyond SPAN frames
        far = [0, 40, 41, 42, 43]
        self.assertEqual(torn_frames.references_for(far, 1, 80, 1, 1), [38, 39, 41, 42])
        self.assertEqual(torn_frames.references_for([0, 40], 1, 80, 1, 1), [36, 37, 38, 39])

    def test_a_row_is_stale_when_what_was_gone_is_back_and_the_row_is_the_earlier_one(self):
        gone = np.zeros(60, int)
        back = np.zeros(60, int)
        same = np.full(60, 100)
        gone[10:40] = 6
        back[10:40] = 6
        tear = torn_frames.judge_stale(gone, back, same, 100)
        self.assertEqual((tear.kind, tear.first_row_px, tear.last_row_px, tear.pixels), ("stale", 20, 80, 720))
        self.assertIsNone(torn_frames.judge_stale(gone, back, np.full(60, 94), 100),
                          "the rows differ from the earlier frame's at every edge: something that moved on a little")
        back[10:40] = 4
        self.assertIsNone(torn_frames.judge_stale(gone, back, same, 100), "two thirds back: holes of a wall passing by")
        back[10:40] = 0
        back[10:25] = 6
        self.assertIsNone(torn_frames.judge_stale(gone, back, same, 100), "15 rows")

    def test_the_counts_of_a_stale_part_are_taken_from_the_pictures(self):
        frames = [picture(planes) for planes in take(10)]
        mixed = picture(stale(take(10)[6], take(10)[4], 60))
        gone, back, same = torn_frames.stale_rows(mixed, frames[4], [frames[5], frames[7], frames[8]])
        self.assertEqual(torn_frames.judge_stale(gone, back, same, WIDTH // 2).kind, "stale")
        self.assertEqual(int(back[:30].sum()), 0, "above row 60 the frame is its own")
        self.assertGreater(int(back[30:].sum()), 150)
        self.assertTrue((back[30:] == gone[30:]).all() and (same[30:] == WIDTH // 2).all())
        gone, back, same = torn_frames.stale_rows(frames[6], frames[4], [frames[5], frames[7], frames[8]])
        self.assertEqual(int(back.sum()), 0, "a whole frame shows nothing of the one two before where the arm has left")


class WholeFramesAreLeftAloneTests(unittest.TestCase):
    """What a whole picture does from one frame to the next, and is not damage."""

    def assert_none(self, frames: list[Planes]) -> None:
        self.assertEqual(found(frames), {})

    def test_a_take_with_a_moving_arm(self):
        self.assert_none(take())
        self.assert_none([front_view(float(step), noise_seed=step) for step in range(14)])

    def test_a_dark_edge_moving_over_a_white_floor(self):
        frames = []
        for step in range(14):
            planes = plain()
            box(planes, 0, 0, HEIGHT, 40 + 14 * step, BLACK)             # an edge that crosses the picture sideways
            box(planes, 0, 0, 30 + 9 * step, WIDTH, BASKET)              # and one that comes down across its width
            frames.append(planes)
        self.assert_none(frames)

    def test_a_thin_thing_that_is_in_a_place_for_one_frame_only(self):
        for upright in (True, False):
            with self.subTest(upright=upright):
                frames = []
                for step in range(14):
                    planes = front_view(0.0, arm=False)
                    if upright:
                        box(planes, 260, 20 + 44 * step, 110, 10, BLACK)     # a cable crossing sideways
                    else:
                        box(planes, 40 + 30 * step, 250, 8, 90, BLACK)       # the same lying flat, coming down
                    frames.append(planes)
                self.assert_none(frames)

    def test_a_global_exposure_step_and_a_ramp(self):
        self.assert_none([front_view(float(step), brighter=25 if step >= 6 else 0) for step in range(14)])
        self.assert_none([front_view(float(step), brighter=3 * step) for step in range(14)])

    def test_a_camera_whose_zoom_changes_from_one_frame_to_the_next(self):
        self.assert_none([front_view(float(step), zoom=1.0 if step < 6 else 0.8) for step in range(14)])

    def test_a_zoom_change_with_an_exposure_that_overshoots(self):
        # the front camera on 2026-10-06, 3.4 s into a take: back on its own zoom, the first frame 20 levels brighter
        # than before and 11 brighter than after
        frames = [front_view(float(step), zoom=1.0 if step < 6 else 0.8,
                             brighter=0 if step < 6 else 20 if step == 6 else 9) for step in range(14)]
        self.assert_none(frames)

    def test_plain_noise(self):
        rng = np.random.default_rng(7)
        half = (HEIGHT // 2, WIDTH // 2)
        noise = [(rng.integers(0, 256, (HEIGHT, WIDTH), dtype=np.uint8), rng.integers(0, 256, half, dtype=np.uint8),
                  rng.integers(0, 256, half, dtype=np.uint8)) for _ in range(12)]
        self.assert_none(noise)
        self.assert_none([front_view(0.0, arm=False, noise_seed=step) for step in range(12)])

    def test_a_camera_that_shakes_by_a_pixel_or_two(self):
        for shake_px in (1, 2):
            with self.subTest(shake_px=shake_px):
                frames = take()
                frames[6] = tuple(np.roll(plane, shake_px if place == 0 else (shake_px + 1) // 2, axis=0)
                                  for place, plane in enumerate(frames[6]))
                self.assert_none(frames)

    def test_a_wall_of_holes_passing_under_the_wrist_camera(self):
        # 11: half the holes' spacing, every second frame alike; 7.3: a third; at 9 to 10 a hole stands every other
        # frame where one stood (2026-10-07: that was called stale until a row had to be the earlier frame's row)
        for px_per_frame in (3.0, 7.0, 7.3, 8.9, 9.7, 10.2, 11.0, 15.0, 17.5, 22.0, 25.6):
            with self.subTest(px_per_frame=px_per_frame):
                self.assert_none([wrist_view(px_per_frame * step) for step in range(22)])

    def test_a_camera_that_stalls_over_a_wall_of_holes(self):
        # the same picture for three frames while the wall moves on: no picture of its own is damaged, and the
        # frames before and after the stall must not vouch for each other (17.5 pixels a frame: five frames on, the
        # holes stand where they stood)
        for px_per_frame in (5.0, 9.7, 17.5, 25.6):
            with self.subTest(px_per_frame=px_per_frame):
                frames = [wrist_view(px_per_frame * step) for step in range(22)]
                frames[10] = frames[11] = frames[9]
                self.assert_none(frames)

    def test_motion_that_starts_and_stops(self):
        steps = [0, 0, 0, 0, 3, 7, 11, 15, 19, 19, 19, 19, 19, 19]
        self.assert_none([front_view(float(step)) for step in steps])
        self.assert_none([front_view(float(step)) for step in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)])

    def test_an_arm_that_stops_dead_holds_and_goes_back_the_way_it_came(self):
        # a camera's frames of a scene at rest differ by its noise; only a frame that was not delivered is the frame
        # before again, value for value
        steps = [0, 4, 8, 12, 16, 16, 16, 16, 16, 12, 8, 4, 0, 0]
        self.assert_none([front_view(float(step), noise_seed=place) for place, step in enumerate(steps)])
        # and one that slows, turns in a single frame and goes back
        steps = [0, 6, 11, 15, 18, 20, 21, 20, 18, 15, 11, 6, 0, 0]
        self.assert_none([front_view(float(step), noise_seed=place) for place, step in enumerate(steps)])

    def test_a_picture_that_comes_twice_is_not_a_torn_one(self):
        steps = [0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 11, 12, 13]
        self.assert_none([front_view(float(step)) for step in steps])

    def test_an_edge_that_swings_out_for_a_frame_and_comes_back(self):
        # the cube under the wrist camera, 2026-10-06: its upper edge three pixels higher in one frame, then back
        frames = []
        for step in range(14):
            planes = plain()
            box(planes, 200 - (6 if step == 6 else 0), 220, 180, 200, ORANGE)
            frames.append(planes)
        self.assert_none(frames)


class RepeatedFramesTests(unittest.TestCase):
    """A frame the camera did not deliver shows as the frame before again. A measurement, not a verdict."""

    #: pixels that moved from the frame before, frame by frame: the arm on its way, and frame 5 the same picture again
    MOVED = np.array([0, 900, 850, 880, 870, 3, 900, 890, 880, 870, 900, 880])
    STEPS = np.array([0.0, 1.2, 1.1, 1.3, 1.2, 1.0, 1.1, 1.2, 0.9, 1.1, 1.2, 1.0])       # largest joint step, degrees

    def test_a_picture_that_came_twice_moves_nowhere_and_an_arm_on_its_way_does(self):
        frames = take(8)
        self.assertEqual(torn_frames.moved_pixels(picture(frames[3]), picture(frames[3])), 0)
        self.assertEqual(torn_frames.moved_pixels(picture(front_view(3.0, noise_seed=1)),
                                                  picture(front_view(3.0, noise_seed=2))), 0)
        self.assertGreater(torn_frames.moved_pixels(picture(frames[3]), picture(frames[4])), 100)

    def test_a_frame_that_changed_nowhere_while_the_arm_moved_is_counted(self):
        told = torn_frames.repeats(self.MOVED, self.STEPS, np.ones(12, bool))
        self.assertEqual(told["repeats"], [5])
        self.assertEqual(told["judged"], 5, "frames 4, 5, 6, 8 and 9: the picture changing two frames before and after")

    def test_only_frames_coded_by_themselves_are_judged(self):
        moved = self.MOVED.copy()
        moved[6] = 0
        told = torn_frames.repeats(moved, self.STEPS, np.arange(12) % 2 == 0)
        self.assertEqual(told["repeats"], [6], "frame 5 is coded as a change of frame 4: the encoder may have left it")
        self.assertEqual(told["judged"], 1)

    def test_a_picture_that_stood_for_three_frames_while_the_arm_moved_is_named(self):
        same = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1], bool)       # frame i is frame i - 1 again
        steps = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.2, 0, 0], float)
        self.assertEqual(torn_frames.stood(same, steps), [(3, 3), (7, 4)],
                         "frames 3 to 5 and 7 to 10; frames 1 and 2 are two only, and 11 to 13 an arm at rest")
        self.assertEqual(torn_frames.stood(same, steps, frames=2), [(1, 2), (3, 3), (7, 4)])
        self.assertEqual(torn_frames.stood(np.zeros(5, bool), np.ones(5)), [])

    def test_an_arm_at_rest_gives_nothing_to_judge(self):
        told = torn_frames.repeats(np.zeros(12, int), np.zeros(12), np.ones(12, bool))
        self.assertEqual((told["judged"], told["repeats"]), (0, []))
        told = torn_frames.repeats(self.MOVED, np.zeros(12), np.ones(12, bool))
        self.assertEqual(told["judged"], 0, "the picture changes and no joint does: not the arm")


@unittest.skipUnless(HAVE_VIDEO, "needs PyAV and pandas (run with the rig's interpreter)")
class DatasetTests(unittest.TestCase):
    """A small dataset on disk, laid out as the recorder lays it out: two episodes in one video file per camera."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.clean = Path(cls.folder.name) / "someone" / "tolrad-clustered-v1"
        cls.torn = Path(cls.folder.name) / "someone" / "tolrad-spread-v1"
        lengths = (16, 14)
        whole = [take(length) for length in lengths]
        write_dataset(cls.clean, {"front": whole, "wrist": whole})
        damaged = [list(frames) for frames in whole]
        damaged[1][5] = tinted(sheared(damaged[1][5], 200, 96), 200, u_step=25)
        write_dataset(cls.torn, {"front": damaged, "wrist": whole})

    def test_a_whole_dataset_has_no_finding_and_every_frame_is_looked_at(self):
        report = torn_frames.scan_dataset(self.clean)
        self.assertEqual(report.findings, [])
        self.assertEqual(report.frames, {"front": 30, "wrist": 30})
        self.assertEqual(report.episodes, 2)

    def test_a_torn_frame_is_named_by_episode_camera_frame_and_time(self):
        report = torn_frames.scan_dataset(self.torn)
        self.assertEqual([(f.episode, f.camera, f.frame) for f in report.findings], [(1, "front", 5)])
        self.assertAlmostEqual(report.findings[0].time_s, 5 / 30.0, places=6)
        self.assertLessEqual(abs(report.findings[0].tear.first_row_px - 200), 8)

    def test_the_arm_s_steps_leave_out_the_gripper_s_own_joint(self):
        steps = torn_frames.joint_steps(self.clean)
        self.assertEqual(sorted(steps), [0, 1])
        self.assertEqual(steps[0].tolist(), [0.0] + [1.5] * 15)
        self.assertEqual(steps[1].tolist(), [0.0] * 14, "only the jaws moved")

    def test_the_scan_keeps_what_the_count_of_repeats_wants(self):
        report = torn_frames.scan_dataset(self.clean, ["front"], measure_repeats=True)
        self.assertEqual(sorted(report.moved["front"]), [0, 1])
        self.assertEqual([len(report.moved["front"][episode]) for episode in (0, 1)], [16, 14])
        self.assertEqual(int(report.moved["front"][1][0]), 0, "a take's first frame: not held against the take before")
        self.assertGreater(int(report.moved["front"][0][5]), 50, "the arm on its way")
        self.assertTrue(report.keyframes["front"][0][0] and not report.keyframes["front"][0][1])
        self.assertFalse(report.same["front"][0].any())

    def test_the_pictures_for_eyes_are_written(self):
        report = torn_frames.scan_dataset(self.torn)
        with tempfile.TemporaryDirectory() as out:
            written = torn_frames.save_pictures(self.torn, report.findings, Path(out))
            self.assertEqual([path.name for path in written], ["episode-001_front_frame-00005.png"])
            self.assertGreater(written[0].stat().st_size, 10_000)

    def test_the_script_says_no_torn_frame_or_names_them(self):
        import subprocess
        import sys

        script = REPO / "scripts" / "rig" / "torn_frames.py"
        base = [sys.executable, str(script), "--cache", self.folder.name, "--namespace", "someone"]
        clean = subprocess.run([*base, "clustered"], capture_output=True, text=True, timeout=300,
                               stdin=subprocess.DEVNULL)
        self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
        self.assertIn("NO TORN FRAME", clean.stdout)
        with tempfile.TemporaryDirectory() as out:
            torn = subprocess.run([*base, "spread", "--save-frames", out], capture_output=True, text=True, timeout=300,
                                  stdin=subprocess.DEVNULL)
            self.assertEqual(torn.returncode, 4, torn.stdout + torn.stderr)
            self.assertNotIn("NO TORN FRAME", torn.stdout)
            self.assertRegex(torn.stdout, r"episode 1, front camera, frame 5 \(0\.167 s\)")
            self.assertEqual(len(list(Path(out).glob("*.png"))), 1)
        missing = subprocess.run([sys.executable, str(script), "--cache", self.folder.name, "clustered"],
                                 capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
                                 env={key: value for key, value in __import__("os").environ.items()
                                      if key != "TOLRAD_NAMESPACE"})
        self.assertEqual(missing.returncode, 1)


def write_dataset(root: Path, cameras: dict[str, list[list[Planes]]], fps: int = 30) -> None:
    """A dataset folder with one video file per camera holding the episodes one after the other, the episodes'
    table, and the joint readings (the arm moving in every frame)."""
    import os
    from unittest import mock

    import av
    import pandas as pd

    rows = []
    # the recorder's own settings (a frame coded by itself every second frame); one thread, and the encoder's banner
    # kept off the test's output
    options = {"g": "2", "crf": "30", "preset": "12", "svtav1-params": "fast-decode=0:lp=1"}
    for camera, episodes in cameras.items():
        path = root / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.dict(os.environ, {"SVT_LOG": "1"}), av.open(str(path), "w") as container:
            stream = container.add_stream("libsvtav1", rate=fps, options=options)
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            for frames in episodes:
                for y, u, v in frames:
                    frame = av.VideoFrame.from_ndarray(np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(
                        HEIGHT * 3 // 2, WIDTH), format="yuv420p")
                    for packet in stream.encode(frame):
                        container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    lengths = [len(frames) for frames in next(iter(cameras.values()))]
    start = 0
    for index, length in enumerate(lengths):
        row = {"episode_index": index, "length": length}
        for camera in cameras:
            key = f"videos/observation.images.{camera}"
            row.update({f"{key}/chunk_index": 0, f"{key}/file_index": 0, f"{key}/from_timestamp": start / fps,
                        f"{key}/to_timestamp": (start + length) / fps})
        rows.append(row)
        start += length
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    features: dict[str, dict] = {f"observation.images.{camera}": {"shape": [HEIGHT, WIDTH, 3]} for camera in cameras}
    features["observation.state"] = {"names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                                               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps, "total_episodes": len(lengths),
                                                          "total_frames": sum(lengths), "features": features}),
                                             encoding="utf-8")
    # the arm's first joint moves through the first take; in the second only the jaws do
    states = [{"episode_index": index, "frame_index": frame,
               "observation.state": [1.5 * frame * (index == 0), 0.0, 0.0, 0.0, 0.0, 2.0 * frame * (index == 1)]}
              for index, length in enumerate(lengths) for frame in range(length)]
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(states).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


if __name__ == "__main__":
    unittest.main()
