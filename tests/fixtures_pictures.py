"""Synthetic camera pictures of flat tags, made the way a camera makes them, with the truth known.

A picture drawn by warping a black-and-white marker mixes black and white as stored grey levels. A camera mixes
light: a pixel that the edge of a tag cuts in half receives half the light, and stores that as a grey level well above
the middle. Where exactly an edge is to a fraction of a pixel depends on this, so these pictures are made in light:

1. every pixel is cut into `ss` x `ss` parts, each part is traced back to the table and asked whether it lies on a
   black module of a tag: exact geometry, no interpolation;
2. the parts are averaged to the pixel (the pixel's own area), a lens blur is applied, both in light;
3. the result is stored through the sRGB curve, sharpened a little as a webcam does, and given noise.

NumPy only, so that tests of the outline measurement run in the development environment too. The detector's own tag
patterns need OpenCV: `tag_bits` is for the tests that have it.
"""

from __future__ import annotations

import numpy as np

#: The rig's own view on 2026-10-05, table millimetres to pixels, fitted from a frame saved that evening
#: (`data/instrument/view_raw_20261005_220950.png`): the camera on the top rear pipe looks down at the floor from
#: behind, so +x goes left in the picture and +y comes down it, and the far side is drawn smaller. About 0.8 px per
#: mm at the far floor tag and 1.05 at the near one; a sheet tag is 31 to 35 px across.
RIG_VIEW = np.array([
    [-9.664308e-01, -1.950903e-01, 3.167622e+02],
    [-2.184485e-02, 6.484538e-01, 2.088656e+02],
    [-1.821664e-05, -6.229738e-04, 1.000000e+00],
])
#: The two floor tags as they lay that evening: centre, size across and up (the caliper sizes), and turn in degrees.
RIG_FLOOR = {0: ((242.5, -103.0), (59.74, 60.0), -3.0), 1: ((-193.5, 197.0), (59.78, 60.0), 0.6)}


def rectangle_mm(centre, size, turn_deg=0.0):
    """A tag's corners in the table frame, detector order (clockwise from its own top-left, y up)."""
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    corners = np.array([(-half_x, half_y), (half_x, half_y), (half_x, -half_y), (-half_x, -half_y)])
    a = np.radians(turn_deg)
    return corners @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T + np.asarray(centre, dtype=float)


def to_px(matrix, points_mm):
    points_mm = np.asarray(points_mm, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points_mm, np.ones(len(points_mm))]) @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def pattern_bits(seed: int) -> np.ndarray:
    """An 8 x 8 pattern like a tag's, True where black: a black border one module wide and a random inside."""
    rng = np.random.default_rng(seed)
    bits = np.ones((8, 8), dtype=bool)
    bits[1:7, 1:7] = rng.random((6, 6)) < 0.5
    return bits


def tag_bits(tag_id: int) -> np.ndarray:
    """A 36h11 tag as its 8 x 8 modules, True where black, row 0 at the tag's top. Needs OpenCV."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    return cv2.aruco.generateImageMarker(dictionary, int(tag_id), 8) < 128


def gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """A Gaussian blur with mirrored borders, one axis after the other."""
    if sigma_px <= 0:
        return image
    radius = max(1, int(np.ceil(4 * sigma_px)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(image, radius, mode="reflect")
    rows = sum(weight * padded[:, radius + offset: radius + offset + image.shape[1]]
               for weight, offset in zip(kernel, offsets, strict=True))
    return sum(weight * rows[radius + offset: radius + offset + image.shape[0], :]
               for weight, offset in zip(kernel, offsets, strict=True))


def encode_srgb(light: np.ndarray) -> np.ndarray:
    """Light between 0 and 1 as stored grey levels 0 to 255 (not yet rounded)."""
    v = np.clip(light, 0.0, 1.0)
    return 255.0 * np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def render(
    mm_to_px,
    tags,
    *,
    size=(640, 480),
    ss=4,
    blur_px=0.4,
    sharpen=0.8,
    noise=1.2,
    seed=0,
    white=0.86,
    black=0.035,
    covered=None,
    lines=None,
) -> np.ndarray:
    """A grey picture (uint8, height x width) of flat tags on a white floor.

    `tags` maps an id to `(corners_mm, bits)`: the tag's four corners in the table frame in the detector's order, and
    its 8 x 8 modules (True where black, row 0 along the side from corner 0 to corner 1). `covered` is a list of
    `(x0, y0, x1, y1)` pixel boxes painted white afterwards, as a hand or a sheet of paper would. `lines` is a list of
    thin printed lines, each `(axis, value, lo, hi, width_mm, darkness, bump)`: `("y", 50, 0, 250, ...)` is the line
    y = 50 from x = 0 to x = 250 in table millimetres; `darkness` is the share of the light it takes away; `bump`,
    if not None, is a function of the place along the line giving how far the line is seen off its place there.
    """
    width, height = size
    px_to_mm = np.linalg.inv(np.asarray(mm_to_px, dtype=float))
    light = np.full((height, width), float(white))
    for corners_mm, bits in tags.values():
        corners_mm = np.asarray(corners_mm, dtype=float)
        corners_px = to_px(mm_to_px, corners_mm)
        x0, y0 = np.floor(corners_px.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(corners_px.max(axis=0)).astype(int) + 3
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        # the centre of every sub-pixel of the box, in pixel-centre coordinates
        u = x0 - 0.5 + (np.arange((x1 - x0) * ss) + 0.5) / ss
        v = y0 - 0.5 + (np.arange((y1 - y0) * ss) + 0.5) / ss
        uu, vv = np.meshgrid(u, v)
        scale = px_to_mm[2, 0] * uu + px_to_mm[2, 1] * vv + px_to_mm[2, 2]
        x_mm = (px_to_mm[0, 0] * uu + px_to_mm[0, 1] * vv + px_to_mm[0, 2]) / scale
        y_mm = (px_to_mm[1, 0] * uu + px_to_mm[1, 1] * vv + px_to_mm[1, 2]) / scale
        along, down = corners_mm[1] - corners_mm[0], corners_mm[3] - corners_mm[0]
        a = ((x_mm - corners_mm[0, 0]) * along[0] + (y_mm - corners_mm[0, 1]) * along[1]) / float(along @ along)
        b = ((x_mm - corners_mm[0, 0]) * down[0] + (y_mm - corners_mm[0, 1]) * down[1]) / float(down @ down)
        inside = (a >= 0) & (a < 1) & (b >= 0) & (b < 1)
        column = np.clip((a * 8).astype(int), 0, 7)
        row = np.clip((b * 8).astype(int), 0, 7)
        dark = inside & np.asarray(bits, dtype=bool)[row, column]
        box = np.where(dark, float(black), float(white))
        light[y0:y1, x0:x1] = box.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
    for axis, value, lo, hi, width_mm, darkness, bump in lines or ():
        ends_mm = [(lo, value), (hi, value)] if axis == "y" else [(value, lo), (value, hi)]
        ends_px = to_px(mm_to_px, ends_mm)
        x0, y0 = np.floor(ends_px.min(axis=0)).astype(int) - 4
        x1, y1 = np.ceil(ends_px.max(axis=0)).astype(int) + 5
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        fine = 2 * ss                                          # a line is far thinner than a pixel
        u = x0 - 0.5 + (np.arange((x1 - x0) * fine) + 0.5) / fine
        v = y0 - 0.5 + (np.arange((y1 - y0) * fine) + 0.5) / fine
        uu, vv = np.meshgrid(u, v)
        scale = px_to_mm[2, 0] * uu + px_to_mm[2, 1] * vv + px_to_mm[2, 2]
        x_mm = (px_to_mm[0, 0] * uu + px_to_mm[0, 1] * vv + px_to_mm[0, 2]) / scale
        y_mm = (px_to_mm[1, 0] * uu + px_to_mm[1, 1] * vv + px_to_mm[1, 2]) / scale
        across, along = (y_mm, x_mm) if axis == "y" else (x_mm, y_mm)
        off = across - value - (0.0 if bump is None else bump(along))
        # how much of each sub-pixel's own width across the line the line covers
        foot = np.abs(np.gradient(across, axis=0)) + np.abs(np.gradient(across, axis=1))
        overlap = np.minimum(off + foot / 2, width_mm / 2) - np.maximum(off - foot / 2, -width_mm / 2)
        covered_share = np.clip(overlap, 0.0, None) / foot
        covered_share[(along < lo) | (along > hi)] = 0.0
        keep = 1.0 - darkness * covered_share
        light[y0:y1, x0:x1] *= keep.reshape(y1 - y0, fine, x1 - x0, fine).mean(axis=(1, 3))
    light = gaussian_blur(light, blur_px)
    stored = encode_srgb(light)
    if sharpen > 0:
        stored = stored + sharpen * (stored - gaussian_blur(stored, 1.0))
    if noise > 0:
        stored = stored + np.random.default_rng(seed).normal(0.0, noise, stored.shape)
    picture = np.clip(np.rint(stored), 0, 255).astype(np.uint8)
    for x0, y0, x1, y1 in covered or ():
        picture[int(y0):int(y1), int(x0):int(x1)] = int(round(float(encode_srgb(np.array(white)))))
    return picture


def sheet_grid_lines(scale=(1.0, 1.0), bump=None):
    """The registration sheet's printed grid as `render` takes lines: the outline (0.21 mm, black) and the lines
    between the cells (0.12 mm, grey), in table millimetres, at the sheet's printed scale. `bump`, if given, is put
    on the outline's far side (y = 150): how far that line is seen off its place, by its place along x."""
    sx, sy = float(scale[0]), float(scale[1])
    outline, inner = (0.6 * 25.4 / 72.0, 0.96), (0.35 * 25.4 / 72.0, 0.74)
    lines = [("y", 0.0, 0.0, 250.0 * sx, *outline, None), ("y", 150.0 * sy, 0.0, 250.0 * sx, *outline, bump),
             ("x", 0.0, 0.0, 150.0 * sy, *outline, None), ("x", 250.0 * sx, 0.0, 150.0 * sy, *outline, None)]
    lines += [("y", y * sy, 0.0, 250.0 * sx, *inner, None) for y in (50.0, 100.0)]
    lines += [("x", x * sx, 0.0, 150.0 * sy, *inner, None) for x in (50.0, 100.0, 150.0, 200.0)]
    return lines


def wrist_rest_picture(floor: float = 200.0, jaws_bgr: tuple[float, float, float] = (14.0, 79.0, 192.0), *,
                       brighter: float = 0.0, stretched: bool = False, jaw_share: float = 0.076, cube: bool = False,
                       shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    """The wrist camera's picture with the arm at rest, BGR: the white floor, and the two orange jaws at the bottom.

    The levels are the recorded ones of 2026-10-06 in the usual tone state (the floor about 200, the jaws' green
    about 79, the jaws on 7.4 to 7.8 % of the picture). `brighter` is another exposure: the floor that many levels
    higher, and the jaws' green 0.659 of that, the line the two levels kept on the recorded frames. `stretched`
    hands the same picture over with its tone range stretched, 16 to 0 and 235 to 255, the stream's other state;
    `cube` puts the cube at the top, where it shows when it lies near the gripper.
    """
    height, width = shape
    picture = np.full((height, width, 3), float(floor) + brighter, np.float64)
    rows = int(height * 0.3)
    each = int(round(jaw_share * height * width / rows / 2))
    blue, green, red = jaws_bgr
    for left in (int(width * 0.3) - each // 2, int(width * 0.7) - each // 2):
        picture[height - rows:, left:left + each] = (blue, green + 0.659 * brighter, red)
    if cube:
        picture[: int(height * 0.16), int(width * 0.45):int(width * 0.7)] = (20.0, 90.0, 230.0)
    if stretched:
        picture = (picture - 16.0) * 255.0 / 219.0
    return np.clip(np.rint(picture), 0, 255).astype(np.uint8)

